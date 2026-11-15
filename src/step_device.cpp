// src/step_device.cpp -- the device join. See include/r97/step_device.h and docs/STEPS-DEVICE.md.
//
// The forward runs the batch on the device kernel tier, the residual at every layer boundary is
// kept in one VRAM allocation on the stream the step's ops issue on, and each block is replayed
// from that store and walked in reverse with the device adjoints. The head is walked a chunk at a
// time: a logit row is 248320 wide, so its cotangent is never materialised -- the CE loop writes
// one chunk's gradient, the vocabulary head's transposed product consumes it into d_hout, and
// d_hout seeds the head's remaining records. The card's staging holds the step's gradients, one
// f32 accumulator per leaf, and is flushed once, at the step's end.
//
// The host join (src/step_host.cpp) is the blueprint this file follows: the same pure helpers
// (src/join_common.h), the same replay-and-walk structure, the same report. What differs is where
// the forward values come from: on the card the tape's operand contents are device pointers, so a
// replayed or recorded op's operands are pulled back and widened before the walk reads them.
#include "rad_runtime.h"  // RAD_ISSUE_OBSERVER / RAD_REPLAY_BLOCK / RAD_CTX_STREAM, before the guard

#if defined(RAD_ISSUE_OBSERVER) && defined(RAD_REPLAY_BLOCK) && defined(RAD_CTX_STREAM)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine.h"       // the complete rad::Engine behind ForwardDriver's unique_ptr
#include "rad_device.h"   // rad_dev_props / rad_dev_alloc / rad_memcpy_async / rad_stream_sync
#include "rad_plugin.h"   // the fp8 decoders join_common's read_elem calls
#include "r97/adapter_io.h"
#include "r97/adjoint.h"
#include "r97/backward.h"
#include "r97/live_adapter.h"
#include "r97/replay.h"
#include "r97/snapshot_share.h"
#include "r97/step_device.h"
#include "r97/tape.h"

#include "adjoint/device/grad_stage.h"  // the card's per-leaf f32 accumulators
#include "join_common.h"                // the join helpers both joins share
#include "vocab_head.h"                 // the vocabulary head's transposed product

namespace r97 {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(const Clock::time_point& t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

void check_device(int rc, const char* what) {
    if (rc != 0)
        throw std::runtime_error(std::string("device step: ") + what + ": " + rad_dev_last_error());
}

// One device allocation that goes back to the card's pool when the step leaves, by the end of it
// or by an error on the way.
class DeviceBuffer {
public:
    DeviceBuffer(const char* what, int64_t bytes) : bytes_(bytes) {
        if (bytes <= 0) return;
        p_ = rad_dev_alloc(bytes, RAD_MEM_DEVICE);
        if (!p_)
            throw std::runtime_error(std::string("device step: the card refused ") + what + "'s " +
                                     std::to_string(bytes) + " bytes: " + rad_dev_last_error());
    }
    ~DeviceBuffer() {
        if (p_) rad_dev_free(p_, RAD_MEM_DEVICE);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* get() const { return p_; }
    int64_t bytes() const { return bytes_; }

private:
    void* p_ = nullptr;
    int64_t bytes_ = 0;
};

// The head's plan owns the device copies of its class planes (src/vocab_head.h): the builder
// uploads this rank's stored slices and hands the caller nothing but the pointers, about 1.3 GB
// for the real head. The step frees them as soon as the head's product is done -- before the block
// walk, so they never sit beside the residual store -- and the guard frees them on every path that
// leaves the step early.
class PlanGuard {
public:
    explicit PlanGuard(VocabHeadPlan* plan) : plan_(plan) {}
    ~PlanGuard() { release(); }
    PlanGuard(const PlanGuard&) = delete;
    PlanGuard& operator=(const PlanGuard&) = delete;

    void release() {
        if (!plan_) return;
        for (int32_t c = 0; c < plan_->classes && c < 4; ++c) {
            const VocabHeadClass& k = plan_->cls[c];
            free_one(k.decode.codes);
            free_one(k.decode.scale);
            free_one(k.scale_col);
        }
        plan_ = nullptr;
    }

private:
    static void free_one(const void* p) {
        if (p) rad_dev_free(const_cast<void*>(p), RAD_MEM_DEVICE);
    }
    VocabHeadPlan* plan_ = nullptr;
};

// The step's gradients wait in the card's staging until the step ends. A walk that throws must not
// leave them behind: the next step's rows would add to a dead total. The guard ends the step
// either way, and is disarmed once the flush has copied the total out.
class StageGuard {
public:
    StageGuard() = default;
    ~StageGuard() {
        if (armed_) device::grad_stage_discard();
    }
    StageGuard(const StageGuard&) = delete;
    StageGuard& operator=(const StageGuard&) = delete;

    void disarm() { armed_ = false; }

private:
    bool armed_ = true;
};

// The recorder installs its hooks in the tape, and the tape keeps them across steps. A step that
// fails before the detach must take them out anyway, or the next step's first issue calls into a
// freed recorder.
class RecorderGuard {
public:
    explicit RecorderGuard(std::shared_ptr<BoundaryRecorder>& recorder) : recorder_(recorder) {}
    ~RecorderGuard() {
        if (recorder_) recorder_->detach();
    }
    RecorderGuard(const RecorderGuard&) = delete;
    RecorderGuard& operator=(const RecorderGuard&) = delete;

private:
    std::shared_ptr<BoundaryRecorder>& recorder_;
};

// One operand's forward values, pulled back from the card: the host copy the snapshot reads, and
// the content that says how to read it. The tape's byte count is kept as recorded -- a strided
// view reports 0 there, and the snapshot must walk its strides rather than read it packed.
struct PulledContent {
    std::vector<uint8_t> bytes;
    OperandContent content;
};

// The bytes one operand's values occupy on the card: the tape's own count when the issue reported
// one span, and a strided view's reach, from its shape and strides, when it did not.
int64_t pulled_bytes(const Operand& op, const OperandContent& content) {
    if (content.data == nullptr) return 0;
    if (content.bytes > 0) return static_cast<int64_t>(content.bytes);
    if (op.view_shape.empty()) return 0;
    const int64_t es = elem_size(op.dtype);
    // A dtype with no element size (a plugin-private quantised tensor) has no byte span to pull
    // back, and no host-readable value either (snapshot returns the shape and no values for one).
    // Nothing copies it, and no adjoint row reads its value.
    if (es <= 0) return 0;
    if (op.view_off < 0)
        throw std::runtime_error("device step: operand '" + op.buffer +
                                 "' is a view with a negative offset, which has no span to pull "
                                 "back");
    int64_t span = op.view_off + 1;
    for (size_t d = 0; d < op.view_shape.size(); ++d) {
        const int64_t reach = op.view_shape[d] - 1;
        if (reach <= 0) continue;
        if (op.view_stride[d] < 0)
            throw std::runtime_error("device step: operand '" + op.buffer +
                                     "' is a reversed view, which has no span to pull back");
        span += reach * op.view_stride[d];
    }
    return span * es;
}

// One head operand as the forward left it: copied on the issuing lane at the issue, before the
// engine reuses the buffer. `content` is what the walk reads -- the copy's host address and the
// tape's own byte count, exactly as pull_operands hands them over.
struct CapturedOperand {
    std::vector<uint8_t> bytes;  // the host copy of the operand's span; empty when it has none
    OperandContent content;
    std::string error;           // why the copy was not taken; empty when it was, or had no span
};

// One head record's operands, in the tape's schema order.
struct CapturedRecord {
    std::vector<CapturedOperand> operand;
    std::string error;  // why the record was not captured; empty when it was
};

// The head's records, keyed by the position the tape gave them.
using HeadCapture = std::map<int64_t, CapturedRecord>;

// Copy a head record's operands on the issuing lane, at the issue.
//
// The engine overwrites the head's buffers after their ops issue and before the backward pulls
// them: at its issue the final norm's input held a maximum magnitude of 14720 and the norm's
// output 11.06, and after serve() the same two addresses read -7.87e33 and 42.25. No forward op
// computed a non-finite value, so the pull -- not the forward -- is what is wrong, and the head's
// read set rightly refuses rows it cannot prove. The observer runs after the op and while the
// buffers still hold what the op left, so a copy taken here is the value the op saw, and the walk
// reads the copy instead of the buffer.
//
// Which records are the head's: the ones issued outside any block after the tape has recorded
// `layers` blocks. That is exact, not a positional guess. An op inside a layer carries that
// layer's block index, which is never negative; the tape's block list reaches `layers` entries at
// the last layer's begin (the step's own check requires the store's `layers + 1` boundaries to
// match `blocks().size() + 1`); and from that moment the only ops recorded with no block open are
// the ones issued after the last layer's end -- the trailing records the join walks,
// `ops[blocks.back().end..)`. An op of the embedding before the first block, or one inside the
// last layer, fails one half each.
//
// The logits record is left out. It is the join's own seed: the CE loop reads its output through
// the tape's recorded pointer, which is the step's last write of it, and the head walk never
// snapshots the record. Copying it would hold a row per token of the whole vocabulary -- 4 GB at
// 4096 tokens -- for bytes nothing reads.
//
// A copy the card refuses is recorded as that operand's error, not thrown: the hook runs inside
// the engine's issue path, and a capture that could not be taken must leave the walk's refusal to
// name the operand and position (snap_record) rather than stop the step in the middle of serve.
void capture_head_record(const Tape& tape, const OpRecord& op,
                         const std::vector<OperandContent>& content, int64_t layers,
                         RadStream stream, HeadCapture* out) {
    if (op.block >= 0) return;
    if (static_cast<int64_t>(tape.blocks().size()) < layers) return;
    for (const Operand& o : op.operands)
        if (o.written && o.buffer == "logits") return;
    CapturedRecord& held = (*out)[op.position];
    held.operand.resize(op.operands.size());
    for (size_t j = 0; j < op.operands.size(); ++j) {
        CapturedOperand& c = held.operand[j];
        if (j >= content.size()) {
            c.error = "the tape recorded no content for it";
            continue;
        }
        try {
            const int64_t bytes = pulled_bytes(op.operands[j], content[j]);
            if (bytes > 0) {
                c.bytes.resize(static_cast<size_t>(bytes));
                if (rad_memcpy_async(c.bytes.data(), content[j].data, bytes, stream) != 0 ||
                    rad_stream_sync(stream) != 0) {
                    c.error = "copying " + std::to_string(bytes) +
                              " bytes back on the step's stream: " + rad_dev_last_error();
                    c.bytes.clear();
                    c.content.data = content[j].data;
                } else {
                    c.content.data = c.bytes.data();
                }
            } else {
                // Nothing to copy: an absent operand, or one whose values the tape does not span.
                // The recorded pointer is kept so snapshot() sees the operand was there; neither
                // case reads a value from it.
                c.content.data = content[j].data;
            }
            c.content.bytes = content[j].bytes;
            c.content.named = content[j].named;
        } catch (const std::exception& e) {
            c.error = e.what();
            c.bytes.clear();
            c.content = OperandContent{content[j].data, content[j].bytes, content[j].named};
        }
    }
}

// Every operand of `op` as the walk will ask for it, copied back on `stream` and waited for, so
// the caller can snapshot them on the host. The copies are one queue's work: the ops that wrote
// them were issued on the same lane. `skip` (empty, or one entry per operand) marks an operand the
// caller already holds -- a frozen one the step widened once -- and its bytes are not copied.
void pull_operands(const OpRecord& op, const std::vector<OperandContent>& content, RadStream stream,
                   std::vector<PulledContent>* held, const std::vector<char>& skip = {}) {
    if (content.size() != op.operands.size())
        throw std::runtime_error("device step: op '" + op.op + "' recorded " +
                                 std::to_string(content.size()) + " operand contents for " +
                                 std::to_string(op.operands.size()) + " operands");
    held->clear();
    held->resize(op.operands.size());
    for (size_t j = 0; j < op.operands.size(); ++j) {
        const Operand& operand = op.operands[j];
        const OperandContent& recorded = content[j];
        PulledContent& pull = (*held)[j];
        if (!skip.empty() && skip[j]) continue;  // the caller's store answers this operand
        const int64_t bytes = pulled_bytes(operand, recorded);
        if (bytes > 0) {
            pull.bytes.resize(static_cast<size_t>(bytes));
            // A driver on the host kernel tier keeps its operands in host memory; the copy then
            // fails here, so the message names the lane as well as the operand.
            const std::string what = "pulling operand '" + operand.buffer +
                                     "' back on the step's stream";
            check_device(rad_memcpy_async(pull.bytes.data(), recorded.data, bytes, stream),
                         what.c_str());
            pull.content.data = pull.bytes.data();
        } else {
            // Nothing to copy: an absent operand (the tape hands no pointer either, and the
            // snapshot reads it as zeros), or one whose values the tape does not span -- which
            // snapshot() refuses by name rather than reading as zeros. Keep the pointer the tape
            // recorded, so that refusal sees the operand was there.
            pull.content.data = recorded.data;
        }
        pull.content.bytes = recorded.bytes;
        pull.content.named = recorded.named;
    }
    check_device(rad_stream_sync(stream), "waiting for an op's operands");
}

// bf16 storage widened to the report's f64, as the host join's boundary values are.
void widen_bf16(const std::vector<uint16_t>& in, std::vector<double>* out) {
    out->resize(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        const uint32_t u = static_cast<uint32_t>(in[i]) << 16;
        float f;
        std::memcpy(&f, &u, sizeof f);
        (*out)[i] = f;
    }
}

}  // namespace

StepFn device_step(ForwardDriver& driver, const Model& model, const DeviceStepOptions& options,
                   DeviceStepReport* report) {
    const ModelMeta meta = read_model_meta(driver.config().model);
    if (meta.hidden <= 0)
        throw std::runtime_error("device step: the model's metadata gives no hidden size");
    if (meta.layers <= 0)
        throw std::runtime_error("device step: the model's metadata gives no layer count");
    if (options.head_chunk_rows <= 0)
        throw std::runtime_error("device step: the head's chunk holds at least one row");

    return [&driver, model, options, report, meta](const Batch& batch, const ParamSet& params,
                                                   ParamSet* grads) -> StepResult {
        StageGuard stage;  // the card's staging is the step's: it ends here, flushed or dropped

        int64_t tokens = 0;
        for (const TokenSeq& s : batch) tokens += static_cast<int64_t>(s.ids.size());
        if (tokens <= 0) throw std::runtime_error("device step: the batch has no token rows");

        WireBuffers wire;
        if (model.delta) model.delta->to_wire(params, &wire);

        std::set<std::string> target_set;
        int64_t max_rank = 0;
        for (const WireBuffers::Site& site : wire.sites) {
            target_set.insert(module_leaf(site.module));
            max_rank = std::max(max_rank, site.rank);
        }
        const SiteGroups groups = group_sites(wire);
        std::vector<std::string> targets(target_set.begin(), target_set.end());
        const bool live = model.delta && !wire.sites.empty();
        driver.set_lora(live ? 1 : 0, live ? std::max<int64_t>(max_rank, 1) : 0, targets,
                        live ? "train" : std::string(), live ? meta.input_basis : std::string());

        // The residual store's plan, and the VRAM it is checked against. The hook runs after the
        // model's load and before the step's serve, so the free VRAM it reads is the card's with
        // the model on it.
        ResidualPlan plan;
        int64_t budget = options.residual_budget_bytes;
        std::shared_ptr<BoundaryRecorder> recorder;
        // The head's operands, copied at their issue (capture_head_record): the engine overwrites
        // the head's buffers before the backward pulls them, so the walk reads these bytes rather
        // than the buffer. Only the head's records are held, and only a call that walks (one with
        // `grads`) installs the hook at all.
        HeadCapture head_capture;
        driver.set_engine_hook([&driver, &recorder, &wire, &plan, &budget, &head_capture, grads,
                                live, targets, max_rank, tokens, meta](rad::Engine& engine) {
            if (live) {
                LiveAdapterConfig cfg{"train", targets, std::max<int64_t>(max_rank, 1), 1.0};
                std::string err;
                std::unique_ptr<LiveAdapter> adapter = LiveAdapter::open(engine, cfg, &err);
                if (!adapter)
                    throw std::runtime_error("device step: cannot open the live adapter: " + err);
                if (adapter->update(wire, &err) != 0)
                    throw std::runtime_error("device step: cannot fill the live adapter: " + err);
            }
            // The plan: L+1 boundaries, and in the int8 trunk a second copy at every boundary a
            // block after the first starts at, because those blocks read the previous block's
            // output beside the residual (src/step_host.cpp builds the same plan). The recorder
            // cross-checks it against the tape and throws when the two disagree.
            plan.boundaries = meta.layers + 1;
            plan.tokens     = tokens;
            plan.hidden     = meta.hidden;
            if (meta.input_basis == "fwht128")
                for (int64_t b = 0; b < plan.boundaries; ++b)
                    plan.extra.push_back(b >= 1 && b < meta.layers ? 1 : 0);
            if (budget <= 0) {
                RadDeviceProps props{};
                if (rad_dev_props(0, &props) != 0)
                    throw std::runtime_error("device step: cannot read the card's free VRAM: " +
                                             std::string(rad_dev_last_error()));
                if (props.is_host_backend)
                    throw std::runtime_error(
                        "device step: the step's device is the host backend, which has no card "
                        "residual store");
                budget = props.vram_free;
            }
            RadStream stream = rad_ctx_cur_stream(engine.issue_ctx());
            if (!stream)
                throw std::runtime_error(
                    "device step: the engine gave no stream for the step's lane");
            recorder = std::make_shared<BoundaryRecorder>(
                driver.tape(), ResidualStore(stream, plan, budget));
            if (!grads) return;
            // The recorder owns the tape's one issue hook, so wrap it rather than replace it: the
            // boundary store still sees every issue, and the capture copies the head's operands
            // while their buffers still hold what the op left. `stream` is the lane the step's ops
            // issue on, which is where the copy has to order.
            Tape& capture_tape = driver.tape();
            Tape::IssueHook inner = capture_tape.issue_hook();
            capture_tape.set_issue_hook(
                [inner, &head_capture, stream, layers = meta.layers](
                    const Tape& t, const OpRecord& op, const std::vector<OperandContent>& content) {
                    if (inner) inner(t, op, content);
                    capture_head_record(t, op, content, layers, stream, &head_capture);
                });
        });

        const Clock::time_point forward_begin = Clock::now();
        RecorderGuard recorder_guard(recorder);
        const StepResult result = driver.step(model)(batch, params, nullptr);
        const double forward_seconds = seconds_since(forward_begin);
        if (!recorder)
            throw std::runtime_error("device step: the engine hook did not install the recorder");
        recorder->detach();
        const ResidualStore store = recorder->store();

        const Tape& tape = driver.last().tape();
        const std::vector<OpRecord>& ops = tape.ops();
        const std::vector<Tape::Block>& blocks = tape.blocks();
        if (blocks.empty())
            throw std::runtime_error("device step: the step has no block brackets to replay");
        if (store.boundaries() != static_cast<int64_t>(blocks.size()) + 1)
            throw std::runtime_error("device step: the boundary store does not match the blocks");

        // The buffers the step writes: an operand of one of them is a value the step produced,
        // which each block may see differently. Every other read is a constant of the step -- a
        // weight, a table, a norm's scale -- and the same value in every block that reads it.
        std::set<std::string> step_writes;
        for (const OpRecord& rec : ops)
            for (const Operand& o : rec.operands)
                if (o.written) step_writes.insert(o.buffer);

        rad::Engine* engine = driver.engine();
        if (!engine) throw std::runtime_error("device step: the step left no engine");
        RadStream stream = rad_ctx_cur_stream(engine->issue_ctx());
        if (!stream)
            throw std::runtime_error("device step: the engine gave no stream for the step's lane");

        DeviceStepReport measured;
        measured.loss = result.loss;
        measured.scored = result.scored;
        measured.tokens = tokens;
        measured.hidden = meta.hidden;
        measured.layers = static_cast<int64_t>(blocks.size());
        measured.residual_boundaries = store.boundaries();
        measured.residual_extras = plan.extras();
        measured.residual_store_bytes = store.bytes();
        measured.residual_budget_bytes = budget;
        // What the issue hook held for the head: the records it copied and the bytes they take. A
        // step that walks nothing installs no hook, so these stay 0.
        measured.head_capture_records = static_cast<int64_t>(head_capture.size());
        for (const auto& entry : head_capture)
            for (const CapturedOperand& c : entry.second.operand)
                measured.head_capture_bytes += static_cast<int64_t>(c.bytes.size());
        if (options.keep_boundaries) {
            measured.boundary_name.assign(static_cast<size_t>(measured.layers + 1), {});
            measured.boundary_value.assign(static_cast<size_t>(measured.layers + 1), {});
            measured.boundary_cotangent.assign(static_cast<size_t>(measured.layers + 1), {});
        }

        // The head's records are the ops after the last block: the same range the capture hook
        // filled (capture_head_record), so `head_start` is the position from which a record's
        // operands come from the capture rather than a pull. The logits op among them is the
        // join's own: it has no adjoint row (the transposed product replaces it), so the walk gets
        // the records before it and a seed on the buffer it reads.
        const int64_t head_start = static_cast<int64_t>(blocks.back().end);
        std::vector<OpRecord> head_records(ops.begin() + static_cast<std::ptrdiff_t>(head_start),
                                           ops.end());
        size_t logits_at = head_records.size();
        for (size_t i = 0; i < head_records.size() && logits_at == head_records.size(); ++i)
            for (const Operand& o : head_records[i].operands)
                if (o.written && o.buffer == "logits") {
                    logits_at = i;
                    break;
                }
        if (logits_at == head_records.size())
            throw std::runtime_error("device step: the head wrote no logits rows");
        const OpRecord logits_rec = head_records[logits_at];
        const std::vector<OperandContent> logits_content =
            tape.contents(static_cast<size_t>(logits_rec.position));

        size_t logits_j = logits_rec.operands.size();
        size_t hid_j = logits_rec.operands.size();
        for (size_t j = 0; j < logits_rec.operands.size(); ++j) {
            const Operand& o = logits_rec.operands[j];
            if (o.written && o.buffer == "logits") logits_j = j;
        }
        for (size_t j = 0; j < logits_rec.operands.size(); ++j) {
            const Operand& o = logits_rec.operands[j];
            if (!o.written && o.shape.size() == 2) {
                hid_j = j;
                break;
            }
        }
        if (logits_j == logits_rec.operands.size())
            throw std::runtime_error("device step: the head's logits op writes no logits operand");
        if (hid_j == logits_rec.operands.size())
            throw std::runtime_error("device step: the head's logits op does not read a hidden "
                                     "state to give back");
        if (logits_j >= logits_content.size())
            throw std::runtime_error("device step: the head's logits operand has no content on "
                                     "the tape");
        const Operand& logits_op = logits_rec.operands[logits_j];
        const Operand& hid_op = logits_rec.operands[hid_j];
        if (logits_op.dtype != "f32")
            throw std::runtime_error("device step: the head's logits are '" + logits_op.dtype +
                                     "', not f32");
        if (logits_op.row_lo != 0)
            throw std::runtime_error("device step: the head wrote the logits from row " +
                                     std::to_string(logits_op.row_lo));
        if (logits_op.shape.size() != 2 || logits_op.rows <= 0 || logits_op.shape[1] <= 0)
            throw std::runtime_error("device step: the head's logits rows are not a 2-D buffer");
        const int64_t head_rows = logits_op.rows;
        const int64_t vocab = logits_op.shape[1];
        if (hid_op.rows < head_rows || hid_op.shape.size() != 2 || hid_op.shape[1] != meta.hidden)
            throw std::runtime_error("device step: the head's hidden state is not " +
                                     std::to_string(head_rows) + " rows of " +
                                     std::to_string(meta.hidden) + " values");
        const std::vector<int64_t> lshape = driver.last().shape("logits");
        if (lshape.size() != 2 || lshape[1] != vocab)
            throw std::runtime_error("device step: the forward's logits boundary is not " +
                                     std::to_string(vocab) + " values a row");

        // The scored rows in the head's row space, in the order the objective reads them: row w
        // is the head's logits row `hidden_rows()[w]` and predicts the batch's w-th scored token.
        const int64_t scored = result.scored;
        if (scored < 0) throw std::runtime_error("device step: the step scored no rows");
        std::vector<int64_t> head_row(static_cast<size_t>(scored));
        std::vector<int32_t> head_id(static_cast<size_t>(scored));
        {
            const std::vector<int64_t>& hidden_rows = driver.last().hidden_rows();
            int64_t at = 0;
            for (const TokenSeq& s : batch) {
                const int64_t len = static_cast<int64_t>(s.ids.size());
                for (int64_t p = 1; p < len; ++p) {
                    if (p < static_cast<int64_t>(s.score.size()) &&
                        !s.score[static_cast<size_t>(p)])
                        continue;
                    if (at >= scored)
                        throw std::runtime_error("device step: the batch has more scored rows "
                                                 "than the step reported");
                    if (at >= static_cast<int64_t>(hidden_rows.size()))
                        throw std::runtime_error(
                            "device step: hidden_rows() is shorter than the scored rows");
                    const int64_t row = hidden_rows[static_cast<size_t>(at)];
                    if (row < 0 || row >= head_rows)
                        throw std::runtime_error(
                            "device step: a scored row is outside the head's logits rows");
                    const int32_t id = s.ids[static_cast<size_t>(p)];
                    if (id < 0 || id >= vocab)
                        throw std::runtime_error("device step: a scored row's target is outside "
                                                 "the vocabulary");
                    head_row[static_cast<size_t>(at)] = row;
                    head_id[static_cast<size_t>(at)] = id;
                    ++at;
                }
            }
            if (at != scored)
                throw std::runtime_error("device step: the step scored " +
                                         std::to_string(scored) + " rows and the batch has " +
                                         std::to_string(at));
        }

        // The CE loop's chunk: the engine's f32 logits chunk cast to the chunk's bf16, one
        // contiguous run of the head's rows at a time -- the transposed product writes d_hout as
        // one dense run, so a chunk never crosses a sequence's rows.
        const void* logits_dev = logits_content[logits_j].data;
        if (logits_dev == nullptr)
            throw std::runtime_error("device step: the head's logits buffer is not readable");
        int64_t filled = 0;
        int64_t chunks = 0;
        int64_t chunk_row = 0;
        HeadChunkFill fill = [&](HeadChunk& chunk) -> HeadChunkRows {
            if (filled >= scored) return HeadChunkRows{0, {}};
            int64_t take = std::min<int64_t>(chunk.positions, scored - filled);
            int64_t run = 1;
            while (run < take && head_row[static_cast<size_t>(filled + run)] ==
                                     head_row[static_cast<size_t>(filled + run - 1)] + 1)
                ++run;
            take = run;
            const int64_t start = head_row[static_cast<size_t>(filled)];
            const auto* src = static_cast<const uint8_t*>(logits_dev) +
                              static_cast<size_t>(start) * static_cast<size_t>(vocab) *
                                  sizeof(float);
            const std::string what = "casting a logits chunk";
            if (!vocab_head_cast(chunk.logits, src, take * vocab, chunk.stream))
                throw std::runtime_error("device step: " + what + ": the card refused it");
            std::vector<int32_t> ids(static_cast<size_t>(take));
            for (int64_t i = 0; i < take; ++i)
                ids[static_cast<size_t>(i)] = head_id[static_cast<size_t>(filled + i)];
            filled += take;
            ++chunks;
            chunk_row = start;
            return HeadChunkRows{take, std::move(ids)};
        };

        HeadCeWindow window;
        window.rows = scored;
        window.vocab = vocab;
        window.chunk = options.head_chunk_rows;
        window.grad_scale = scored > 0 ? 1.0 / static_cast<double>(scored) : 0.0;

        // The forward-only call: the head's loss on the card, and nothing walked, staged or
        // copied back. A training loop's evaluation pass is this call.
        if (!grads) {
            const Clock::time_point head_begin = Clock::now();
            const HeadCeSum ce = head_ce_loop(window, fill);
            measured.head_seconds = seconds_since(head_begin);
            measured.head_rows = window.rows;
            measured.head_chunks = chunks;
            if (ce.scored != scored)
                throw std::runtime_error("device step: the head's CE loop scored " +
                                         std::to_string(ce.scored) + " rows, not " +
                                         std::to_string(scored));
            measured.forward_seconds = forward_seconds;
            if (report) *report = std::move(measured);
            return result;
        }

        // The head's transposed product: d_hout is the head's cotangent in its logits row space,
        // zero where a row is not scored, and the seed the head's remaining records walk from.
        const int64_t d_hout_values = head_rows * meta.hidden;
        DeviceBuffer d_hout("d_hout", d_hout_values * static_cast<int64_t>(sizeof(float)));
        measured.d_hout_bytes = d_hout.bytes();
        check_device(rad_memset_async(d_hout.get(), 0, d_hout.bytes(), stream),
                     "zeroing d_hout");
        check_device(rad_stream_sync(stream), "zeroing d_hout");

        VocabHeadPlan vocab_plan;
        {
            std::string err;
            if (!vocab_head_plan_from_record(logits_rec, logits_content, driver.config().model,
                                             stream, &vocab_plan, &err))
                throw std::runtime_error("device step: the head's plan: " + err);
        }
        PlanGuard plan_guard(&vocab_plan);
        if (vocab_plan.n_embd != meta.hidden || vocab_plan.vocab != vocab)
            throw std::runtime_error(
                "device step: the head's plan does not describe the logits buffer it wrote");

        const Clock::time_point head_begin = Clock::now();
        HeadGradSink sink = [&](const HeadChunk& chunk) {
            std::string err;
            float* at = static_cast<float*>(d_hout.get()) + static_cast<size_t>(chunk_row) *
                                                               static_cast<size_t>(meta.hidden);
            if (!vocab_head_transpose(vocab_plan, chunk.grad, chunk.positions, at, chunk.stream,
                                      &err))
                throw std::runtime_error("device step: the head's transposed product: " + err);
            // The loop's next chunk reuses the gradient buffer, and the loop's stream dies with
            // it: wait here, so the product is complete before either happens.
            check_device(rad_stream_sync(chunk.stream),
                         "waiting for the head's transposed product");
        };
        const HeadCeSum ce = head_ce_loop(window, fill, sink);
        measured.head_rows = window.rows;
        measured.head_chunks = chunks;
        if (ce.scored != scored)
            throw std::runtime_error("device step: the head's CE loop scored " +
                                     std::to_string(ce.scored) + " rows, not " +
                                     std::to_string(scored));
        std::vector<float> d_hout_host(static_cast<size_t>(d_hout_values));
        check_device(rad_memcpy_async(d_hout_host.data(), d_hout.get(), d_hout.bytes(), stream),
                     "reading the head's cotangent back");
        check_device(rad_stream_sync(stream), "reading the head's cotangent back");
        const double head_seconds = seconds_since(head_begin);
        // The plan's device copies are done with: free them before the block walk, so they never
        // sit beside the residual store and the next block's snapshots.
        plan_guard.release();

        // The head's remaining records: the final norm and the gather into the head's rows. Their
        // forward values are pulled back and widened, as a replayed block's are.
        std::vector<OpRecord> head_walk;
        head_walk.reserve(head_records.size() - 1);
        for (size_t i = 0; i < head_records.size(); ++i) {
            if (i == logits_at) continue;
            OpRecord rec = own_raws(std::move(head_records[i]));
            tighten_rows(rec);
            if (options.reference_head_split) mark_join_lora(rec);
            head_walk.push_back(std::move(rec));
        }
        // The step's backward time outside the walk's own report: the adjoint row calls, per op,
        // from every walk the step makes (the head's and each block's), and the replay's
        // pull-back -- one copy-and-sync per op, the path the walk does not see.
        std::map<std::string, OpTiming> row_time;
        std::map<std::string, OpTiming> pull_time;
        const auto merge_ops = [](std::map<std::string, OpTiming>& into,
                                  const std::vector<OpTiming>& from) {
            for (const OpTiming& t : from) {
                OpTiming& acc = into[t.op];
                acc.calls += t.calls;
                acc.seconds += t.seconds;
                acc.non_finite += t.non_finite;
                acc.rows_zeroed_unreadable += t.rows_zeroed_unreadable;
                acc.rows_readable += t.rows_readable;
                acc.rows_refused_non_finite += t.rows_refused_non_finite;
            }
        };
        // The first non-finite cotangent any of the step's walks handed on, and whether the
        // forward's operands were finite. One walk's hit is the step's: the walks run from the
        // head down through the layers, so this names the first the step met.
        const auto merge_finiteness = [&measured](const BackwardReport& r) {
            measured.forward_all_finite = measured.forward_all_finite && r.forward_all_finite;
            if (measured.forward_first_non_finite_operand.empty() &&
                !r.forward_first_non_finite_operand.empty()) {
                measured.forward_first_non_finite_operand = r.forward_first_non_finite_operand;
                measured.forward_first_non_finite_operand_index =
                    r.forward_first_non_finite_operand_index;
                measured.forward_first_non_finite_position = r.forward_first_non_finite_position;
                measured.forward_first_non_finite_index = r.forward_first_non_finite_index;
            }
            if (measured.first_non_finite_op.empty() && !r.first_non_finite_op.empty()) {
                measured.first_non_finite_op = r.first_non_finite_op;
                measured.first_non_finite_operand = r.first_non_finite_operand;
                measured.first_non_finite_operand_index = r.first_non_finite_operand_index;
                measured.first_non_finite_position = r.first_non_finite_position;
                measured.first_non_finite_index = r.first_non_finite_index;
                measured.first_non_finite_from_row = r.first_non_finite_from_row;
            }
        };
        // The step's frozen operands, widened once and shared by every walk. A block's weights are
        // the same f64 values in all 64 blocks; without this every block pulls them back from the
        // card and widens them again -- ~3.7 GiB of f64 a block on the real container, rebuilt 64
        // times. `shared` holds each once; a block's own store holds what only that block sees.
        SharedSnapshots shared;
        // The key of a frozen read, or empty when the operand is not one: a read of a buffer the
        // step writes is a value of that block, not a constant of the step.
        const auto frozen_key = [&step_writes](const OpRecord& rec, size_t j) -> std::string {
            // R97_NO_SHARED_SNAPSHOTS=1 falls back to widening every operand per block. It exists
            // because the sharing is the only change between the run that completed and the one
            // whose child was killed, so a window can tell them apart in one run.
            static const bool off = [] {
                const char* e = std::getenv("R97_NO_SHARED_SNAPSHOTS");
                return e != nullptr && *e == '1';
            }();
            if (off) return std::string();
            const Operand& o = rec.operands[j];
            if (o.written || step_writes.count(o.buffer)) return std::string();
            return snapshot_key(o);
        };
        // One record's operands as the walk will ask for them. A head record's operands are the
        // bytes copied at their issue (head_capture): the engine has overwritten their buffers
        // since, so a pull here would read another writer's values -- -7.87e33 where the op saw
        // 14720. A head record without a capture is refused by name, never pulled: the pull is
        // exactly what the refusal exists for. Every other record -- the blocks, all before
        // `head_start` -- is pulled back as before: a frozen read the shared store already holds is
        // bound without a pull, the step's first read of one is pulled and widened here and held
        // for every later block, and everything else is pulled back and widened into the caller's
        // per-block store.
        struct SnappedRecord {
            std::vector<AdValue> now;                            // frozen entries stay empty
            std::vector<std::shared_ptr<const AdValue>> frozen;  // non-null: answered by `shared`
        };
        const auto snap_record = [&](const OpRecord& rec,
                                     const std::vector<OperandContent>& content) {
            SnappedRecord out;
            out.now.resize(rec.operands.size());
            out.frozen.assign(rec.operands.size(), nullptr);
            std::vector<char> skip(rec.operands.size(), 0);
            std::vector<std::string> key(rec.operands.size());
            for (size_t j = 0; j < rec.operands.size(); ++j) {
                key[j] = frozen_key(rec, j);
                if (key[j].empty()) continue;
                out.frozen[j] = shared.find(key[j]);
                skip[j] = out.frozen[j] ? 1 : 0;
            }
            const CapturedRecord* captured = nullptr;
            if (rec.position >= head_start) {
                const auto it = head_capture.find(rec.position);
                if (it == head_capture.end())
                    throw std::runtime_error(
                        "device step: the head record '" + rec.op + "' at position " +
                        std::to_string(rec.position) +
                        " has no captured operands: the step's issue hook did not copy it, so the "
                        "walk would read a buffer the engine has overwritten since");
                if (!it->second.error.empty())
                    throw std::runtime_error("device step: the head record '" + rec.op +
                                             "' at position " + std::to_string(rec.position) +
                                             " was not captured: " + it->second.error);
                captured = &it->second;
                if (captured->operand.size() != rec.operands.size())
                    throw std::runtime_error(
                        "device step: the head record '" + rec.op + "' at position " +
                        std::to_string(rec.position) + " captured " +
                        std::to_string(captured->operand.size()) + " operands, not " +
                        std::to_string(rec.operands.size()));
            }
            std::vector<PulledContent> held;
            if (captured) {
                held.resize(rec.operands.size());
                for (size_t j = 0; j < rec.operands.size(); ++j) {
                    if (out.frozen[j]) continue;  // answered by the shared store, not the capture
                    const CapturedOperand& c = captured->operand[j];
                    if (!c.error.empty())
                        throw std::runtime_error(
                            "device step: the head record '" + rec.op + "' at position " +
                            std::to_string(rec.position) + ": operand '" +
                            rec.operands[j].buffer + "' (" + std::to_string(j) +
                            ") was not captured: " + c.error);
                    held[j].content = c.content;
                }
            } else {
                const Clock::time_point pull_begin = Clock::now();
                pull_operands(rec, content, stream, &held, skip);
                OpTiming& pull_acc = pull_time[rec.op];
                pull_acc.calls += 1;
                pull_acc.seconds += seconds_since(pull_begin);
            }
            for (size_t j = 0; j < rec.operands.size(); ++j) {
                if (out.frozen[j]) {
                    shared.bind(rec.position, j, out.frozen[j]);
                    continue;
                }
                out.now[j] = snapshot(rec.operands[j], held[j].content, rec.op.c_str());
                if (key[j].empty()) continue;
                // The step's first read of a frozen operand: its one widen, which every later
                // block binds instead of pulling and widening again.
                out.frozen[j] = shared.insert(key[j], std::move(out.now[j]));
                shared.bind(rec.position, j, out.frozen[j]);
            }
            // R97_DUMP_PULL=1: for the head's records, what the walk will read beside what the
            // card's buffer holds now. The walk's values are the bytes copied at the issue
            // (head_capture), so a card line that differs from the host line is the engine's reuse
            // of the buffer after the op -- the very reason the capture exists -- and a card that
            // agrees says nothing overwrote it.
            if (const char* dp = std::getenv("R97_DUMP_PULL"); dp != nullptr && dp[0] == '1' &&
                                                                    rec.position >= 1458) {
                for (size_t j = 0; j < rec.operands.size(); ++j) {
                    const Operand& o = rec.operands[j];
                    const OperandContent& c = content[j];
                    const AdValue& v = out.now[j];
                    const int64_t es = elem_size(o.dtype);
                    double card[4] = {0, 0, 0, 0};
                    if (c.data != nullptr && es > 0 && c.bytes >= static_cast<size_t>(es)) {
                        const int64_t n = std::min<int64_t>(4, c.bytes / es);
                        std::vector<uint8_t> tmp(static_cast<size_t>(n * es));
                        if (rad_memcpy_async(tmp.data(), c.data, n * es, stream) == 0 &&
                            rad_stream_sync(stream) == 0)
                            for (int64_t k = 0; k < n; ++k)
                                card[k] = read_elem(o.dtype, tmp.data() + k * es, 0);
                    }
                    std::fprintf(stderr,
                                 "DUMPPULL pos=%lld op=%s buffer=%s operand=%zu written=%d dtype=%s "
                                 "bytes=%zu values=%zu raw=%p card=[%.6g %.6g %.6g %.6g] "
                                 "host=[%.6g %.6g %.6g %.6g]\n",
                                 static_cast<long long>(rec.position), rec.op.c_str(),
                                 o.buffer.c_str(), j, static_cast<int>(o.written), o.dtype.c_str(),
                                 c.bytes, v.value.size(), v.raw, card[0], card[1], card[2], card[3],
                                 v.value.size() > 0 ? v.value[0] : 0.0,
                                 v.value.size() > 1 ? v.value[1] : 0.0,
                                 v.value.size() > 2 ? v.value[2] : 0.0,
                                 v.value.size() > 3 ? v.value[3] : 0.0);
                    std::fflush(stderr);
                }
            }
            return out;
        };
        // The op that produced the first non-finite forward value the snapshot loop met. The walk
        // can only name the op that first read such a value, and the first reader of a bad buffer
        // is downstream of the op that wrote it. Only written operands are scanned: a read's value
        // belongs to the op that wrote it, and the frozen weights are reads.
        struct ProducedValue {
            bool found = false;
            std::string op;
            std::string buffer;
            int64_t operand = -1;
            int64_t position = -1;
            int64_t value = -1;
        };
        ProducedValue produced;
        const auto note_produced = [&produced](const OpRecord& rec,
                                               const std::vector<AdValue>& now) {
            if (produced.found) return;
            for (size_t j = 0; j < rec.operands.size() && j < now.size(); ++j) {
                if (!rec.operands[j].written) continue;
                const std::vector<double>& v = now[j].value;
                for (size_t e = 0; e < v.size(); ++e) {
                    if (std::isfinite(v[e])) continue;
                    produced.found = true;
                    produced.op = rec.op;
                    produced.buffer = rec.operands[j].buffer;
                    produced.operand = static_cast<int64_t>(j);
                    produced.position = rec.position;
                    produced.value = static_cast<int64_t>(e);
                    return;
                }
            }
        };

        StoredSnapshots head_snaps;
        shared.set_base(&head_snaps);
        for (const OpRecord& rec : head_walk) {
            SnappedRecord snapped =
                snap_record(rec, tape.contents(static_cast<size_t>(rec.position)));
            note_produced(rec, snapped.now);
            for (size_t j = 0; j < rec.operands.size(); ++j)
                if (!snapped.frozen[j]) head_snaps.set(rec.position, j, std::move(snapped.now[j]));
        }

        // The head's captured operands have been read into the walk's store; the copies are not
        // needed past this point (the block walk's records are all before `head_start`, so
        // snap_record pulls them as before). Drop them before the residual store and the block
        // snapshots grow beside them.
        head_capture.clear();

        // The rows a consumer reads, from the whole tape (src/read_set.h's derive_read_sets):
        // a record whose every written output is read by a gather_rows and nothing else takes the
        // gather's index list as its read set, and every other record is left unproven and keeps
        // the loud refusal. The buffer read outside this walk is the LOGITS -- the objective's --
        // and marking it is what leaves the head's own writer unproven, as the host path does with
        // its seeds' buffers. The hidden state is not external: its value is read by the walk's own
        // logits gemm, through the gather. Marking it external instead made the head's norm
        // unprovable, so the 512 step zeroed and refused nothing and a forward inf in a row the
        // gather never reads flowed into every gradient (all_finite false, layer 0's
        // in_proj_qkv.lora_A/B non-finite).
        const std::map<int64_t, std::vector<int64_t>> head_reads =
            derive_read_sets(head_walk, ops, shared, head_external_reads(logits_rec));
        // The head walk is where the read set has to earn its keep on the real container (a
        // 512-token step reads one of the final norm's rows), so its decision is printed: which
        // records were proven and how many rows each names. Without it a run that zeroes nothing
        // looks the same whether the rule was never asked or never answered.
        {
            std::string line = "r97: head read set: " + std::to_string(head_reads.size()) + " of " +
                               std::to_string(head_walk.size()) + " records proven";
            for (const OpRecord& rec : head_walk) {
                const auto it = head_reads.find(rec.position);
                line += " [" + rec.op + "@" + std::to_string(rec.position) + " ";
                if (it == head_reads.end()) {
                    line += "unproven]";
                    continue;
                }
                line += std::to_string(it->second.size()) + " rows";
                if (it->second.size() <= 4) {
                    line += " (";
                    for (size_t k = 0; k < it->second.size(); ++k) {
                        if (k != 0) line += ",";
                        line += std::to_string(it->second[k]);
                    }
                    line += ")";
                }
                line += "]";
            }
            std::fprintf(stderr, "%s\n", line.c_str());
        }

        BackwardOptions bo;
        bo.side = options.host_walk ? Side::host : Side::device;
        // A device row is an optimisation; where libr97 has none yet, the host row is the same
        // arithmetic in f64 and the walk has every operand pulled back already. The step's
        // coverage names the ops that took the host path. The host A/B asks for the host rows
        // themselves, so a missing one is an error that names it.
        bo.allow_host_fallback = !options.host_walk;
        bo.grads = grads;
        const std::string grad_prefix = model.delta ? model.delta->grad_prefix() : "";
        bo.prefix = grad_prefix.c_str();
        bo.in_place_passthrough = true;
        // The engine pools a fused weight's modules into one lora record; write one fused factor
        // and let split_adapter_grads put each module's rows under its own name.
        bo.fused_slots = true;
        if (!groups.empty()) {
            bo.record_prefix = [groups, grad_prefix](const OpRecord& rec) -> std::string {
                const std::string weight = record_weight(rec);
                if (weight.empty()) return grad_prefix;
                const auto g = groups.find(weight);
                if (g == groups.end()) return grad_prefix;
                return grad_prefix + (g->second.size() > 1 ? fused_prefix(weight)
                                                           : g->second.front().module);
            };
        }
        bo.slot_prefix = [groups, grad_prefix](const OpRecord& rec, int64_t slot) -> std::string {
            const std::string weight = record_weight(rec);
            const auto g = groups.find(weight);
            if (g == groups.end() || slot < 0 ||
                slot >= static_cast<int64_t>(g->second.size()))
                return grad_prefix;
            return grad_prefix + g->second[static_cast<size_t>(slot)].module;
        };
        bo.prepare_record = prepare_adjoint_record;

        const Clock::time_point backward_begin = Clock::now();
        BoundaryNames head_names;
        head_names.add(hid_op.buffer, 0, head_rows);
        std::vector<Seed> head_seeds;
        head_seeds.emplace_back();
        head_seeds.back().buffer = hid_op.buffer;
        head_seeds.back().row_lo = 0;
        head_seeds.back().row_hi = head_rows;
        head_seeds.back().cotangent = std::move(d_hout_host);
        bo.read_rows = &head_reads;
        const BackwardResult head =
            backward_block(head_walk, shared, head_names, head_seeds, {}, bo);
        merge_ops(row_time, head.report.ops);
        merge_finiteness(head.report);
        measured.peak_snapshot_store_bytes = head_snaps.bytes();
        measured.peak_snapshot_op_bytes = head.report.peak_snapshot_bytes;
        measured.peak_snapshot_bytes =
            measured.peak_snapshot_store_bytes + measured.peak_snapshot_op_bytes;
        measured.peak_cotangent_bytes = head.report.peak_cotangent_bytes;
        shared.set_base(nullptr);
        head_snaps = StoredSnapshots{};

        // The head's input cotangents stay alive with `head` for the whole layer loop, so they are
        // part of what the join holds during every block walk.
        int64_t head_input_bytes = 0;
        for (const InputCotangent& in : head.inputs) head_input_bytes += value_bytes(in.value);

        const InputCotangent* above = pick_boundary(head.inputs, tokens, meta.hidden);
        if (!above) {
            std::string dbg = "device step: the head walk left no cotangent on the final hidden; "
                              "blocks=" +
                              std::to_string(blocks.size()) + " head_records=" +
                              std::to_string(head_walk.size()) + " inputs=" +
                              std::to_string(head.inputs.size());
            for (const InputCotangent& in : head.inputs) {
                dbg += " [";
                dbg += in.buffer;
                dbg += " rows=" + std::to_string(in.row_lo) + ".." + std::to_string(in.row_hi) +
                       " shape=";
                for (int64_t d : in.shape) dbg += std::to_string(d) + ",";
                dbg += "]";
            }
            for (const OpRecord& rec : head_walk) dbg += " op=" + rec.op;
            throw std::runtime_error(dbg);
        }

        auto store_boundary = [&](int64_t boundary, const InputCotangent* in) {
            if (!options.keep_boundaries || !in) return;
            measured.boundary_name[static_cast<size_t>(boundary)] = in->buffer;
            measured.boundary_cotangent[static_cast<size_t>(boundary)] = in->value;
            if (!store.has(boundary)) return;
            std::vector<uint16_t> host(static_cast<size_t>(tokens * meta.hidden));
            store.read(boundary, host.data());
            widen_bf16(host, &measured.boundary_value[static_cast<size_t>(boundary)]);
        };

        store_boundary(measured.layers, above);
        IncomingCotangent incoming{above->buffer, 0, tokens,
                                   first_rows(*above, tokens, meta.hidden)};

        LayerReplay replay(*engine, driver.tape(), store);
        for (int64_t layer = measured.layers - 1; layer >= 0; --layer) {
            StoredSnapshots snaps;
            // This block's own values answer every key the step's frozen store does not hold.
            shared.set_base(&snaps);
            // The replay's pull-back is at the replayed op's issue, and the engine reuses its
            // arena between one op and the next: a read of a buffer an earlier op in the block
            // wrote is pulled after that slot has been handed to another op, so it holds the later
            // op's values, not the writer's (a -7.87e33 where the writer left a finite row). An
            // in-place read is the same defect with the writer being this op's own earlier state.
            // Capture every buffer the block writes, at the write's issue, and serve a read of it
            // from that capture: the adjoint then sees the value the op read, not what the arena
            // holds a moment later (the gated norm's gdn_o, above all). A buffer the block never
            // wrote -- a frozen input, the boundary, or the recurrent state the engine cleared
            // before the replay -- is not in the capture and keeps the live pull, entering zero for
            // a kv: the block reads cold.
            std::map<std::string, AdValue> before;
            std::set<std::string> block_wrote;
            // The writer's capture covers the read when the read's rows are a prefix the writer
            // wrote with the same trailing layout (row-major: the writer wrote every row the read
            // wants). A read wanting a row the writer left unwritten is not covered.
            const auto same_layout = [](const AdValue& w, const AdValue& r) {
                if (w.shape.size() != r.shape.size()) return false;
                for (size_t d = 1; d < r.shape.size(); ++d)
                    if (w.shape[d] != r.shape[d]) return false;
                return true;
            };
            const auto covered = [&same_layout](const AdValue& w, const AdValue& r) {
                return same_layout(w, r) && (r.shape.empty() || w.shape[0] >= r.shape[0]);
            };
            // The writer's value restricted to the read's row prefix: the read's span is the
            // writer's, not whatever the arena now holds. A plugin-private tensor has no f64 value,
            // so its pointer and encoding travel as a block, as an in-place read already does.
            const auto writer_rows = [](const AdValue& w, const AdValue& r) {
                AdValue out;
                out.dtype = r.dtype;
                out.shape = r.shape;
                out.raw = w.raw;
                out.enc = w.enc;
                if (w.value.empty()) return out;
                size_t per_row = 1;
                for (size_t d = 1; d < r.shape.size(); ++d) per_row *= (size_t)r.shape[d];
                const size_t need = r.shape.empty() ? w.value.size() : (size_t)r.shape[0] * per_row;
                out.value.assign(w.value.begin(), w.value.begin() + need);
                return out;
            };
            // A strided read of a buffer the block wrote: the writer's capture is the declared
            // buffer, row-major, and the operand's view_off/view_stride/view_shape index that same
            // flattened space -- the walk `snapshot` already does over a live buffer at
            // join_common.h:246-273, run here over the capture instead. Serving the read from the
            // capture is the only correct source for the same reason the row capture exists: the
            // arena is reused between one op and the next, so the live pull at the *reader's* issue
            // is the later op's values. The q-norm's q|gate view is where this bites. A capture that
            // cannot describe the view -- a plugin-private tensor with no f64 value, or an offset
            // off the end of the capture -- keeps the live pull, and the caller says so.
            const auto writer_view = [](const AdValue& w, const Operand& o, const AdValue& r,
                                        bool* served) {
                *served = false;
                AdValue out;
                out.dtype = r.dtype;
                out.shape = r.shape;
                out.raw = w.raw;
                out.enc = w.enc;
                // The reader's shape is the operand's view shape (snapshot, join_common.h:247), so
                // the strides and the extents are the same rank and index the same space.
                if (w.value.empty() || r.shape.empty() || o.view_shape.empty()) return out;
                if (o.view_stride.size() != r.shape.size() || o.view_shape.size() != r.shape.size())
                    return out;
                int64_t n = 1;
                for (int64_t d : r.shape) n *= std::max<int64_t>(d, 0);
                if (n <= 0) return out;
                out.value.assign(static_cast<size_t>(n), 0.0);
                const size_t rank = r.shape.size();
                int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                for (int64_t e = 0; e < n; ++e) {
                    int64_t off = o.view_off;
                    for (size_t d = 0; d < rank; ++d) off += idx[d] * o.view_stride[d];
                    if (off < 0 || static_cast<size_t>(off) >= w.value.size()) return out;
                    out.value[static_cast<size_t>(e)] = w.value[static_cast<size_t>(off)];
                    for (size_t d = rank; d-- > 0;) {
                        if (++idx[d] < r.shape[d]) break;
                        idx[d] = 0;
                    }
                }
                *served = true;
                return out;
            };
            replay.replay(layer, [&snaps, &snap_record, &note_produced, &before, &block_wrote,
                                  &covered, &same_layout, &writer_rows, &writer_view, &options,
                                  &layer](
                                     const OpRecord& op,
                                     const std::vector<OperandContent>& content) {
                // The adjoint sees the step's rows and, for attention, the q view its family
                // reads; the snapshots must have the same shapes the walk will ask for. The
                // operands are the card's, so they are pulled back before they are read; a frozen
                // one the step already holds is bound from the shared store instead.
                OpRecord shaped = op;
                tighten_rows(shaped);
                if (options.reference_head_split) mark_join_lora(shaped);
                SnappedRecord snapped = snap_record(shaped, content);
                note_produced(shaped, snapped.now);
                for (size_t j = 0; j < shaped.operands.size(); ++j) {
                    const Operand& read = shaped.operands[j];
                    if (read.written || snapped.frozen[j]) continue;
                    if (!block_wrote.count(read.buffer)) {
                        // No op before this one wrote it: a frozen input, the boundary, or a
                        // recurrent state the block reads cold. The kv: state the engine cleared
                        // before the replay enters zero; everything else keeps the live pull.
                        if (read.buffer.rfind("kv:", 0) == 0)
                            snapped.now[j].value.assign(snapped.now[j].value.size(), 0.0);
                        continue;
                    }
                    const auto prev = before.find(read.buffer);
                    if (prev == before.end()) {
                        // A `kv:` buffer is the engine's paged pool, not an activation the capture
                        // holds: the block's kv_store writes it by paging, so it lands in
                        // block_wrote but never in `before` (its write is a block view). Its reader
                        // (attn_paged) must see the live pool the replay just wrote; the pool is its
                        // own allocation, not the reused activation arena, so the live pull is the
                        // reader's value and not a stale one.
                        if (read.buffer.rfind("kv:", 0) == 0) continue;
                        // The block wrote it, but the capture did not record it (a view, or a
                        // partial-row write): the capture is not complete for the read. A pull here
                        // reads a reused arena, and a zero would hide the defect -- name it.
                        throw std::runtime_error("device step: block " + std::to_string(layer) +
                                                 " read of '" + read.buffer + "' by '" + op.op +
                                                 "' at position " + std::to_string(op.position) +
                                                 " is not covered by the block's write of it");
                    }
                    if (!covered(prev->second, snapped.now[j])) {
                        // A row-prefix the writer left short is a gap the live pull would fill from
                        // a reused arena: name it. A different layout (rank or trailing dims) is a
                        // strided or column view -- q_proj's query half, its gate half -- that the
                        // writer's row-major capture cannot express; the buffer is live at this
                        // read's issue (a read is always of a live buffer), so the live pull IS the
                        // reader's view, and it is kept.
                        if (same_layout(prev->second, snapped.now[j]) &&
                            prev->second.shape[0] < snapped.now[j].shape[0]) {
                            // Both sides named: the writer's captured AdValue and the reader's
                            // snapshot, each as [shape] plus the row window and value count, and
                            // the operand's own view declaration. The refusal is right whenever
                            // these disagree about rows; when they agree on rows and differ on
                            // the view, the coverage test has asked the wrong question.
                            std::string ws, rs, vs;
                            for (int64_t d : prev->second.shape) ws += " " + std::to_string(d);
                            for (int64_t d : snapped.now[j].shape) rs += " " + std::to_string(d);
                            for (int64_t d : read.view_shape) vs += " " + std::to_string(d);
                            const int64_t w_rows =
                                prev->second.shape.empty() ? 0 : prev->second.shape[0];
                            const int64_t r_rows =
                                snapped.now[j].shape.empty() ? 0 : snapped.now[j].shape[0];
                            std::fprintf(stderr,
                                         "r97: block %lld coverage '%s' op=%s pos=%lld: "
                                         "writer[%s ] rows=%lld vals=%llu | read[%s ] rows=%lld "
                                         "vals=%llu | operand rows=%lld row_lo=%lld view[%s ] "
                                         "view_off=%lld in_place=%d written=%d\n",
                                         static_cast<long long>(layer), read.buffer.c_str(),
                                         op.op.c_str(), static_cast<long long>(op.position),
                                         ws.c_str(), static_cast<long long>(w_rows),
                                         static_cast<unsigned long long>(prev->second.value.size()),
                                         rs.c_str(), static_cast<long long>(r_rows),
                                         static_cast<unsigned long long>(snapped.now[j].value.size()),
                                         static_cast<long long>(read.rows),
                                         static_cast<long long>(read.row_lo), vs.c_str(),
                                         static_cast<long long>(read.view_off),
                                         read.in_place ? 1 : 0, read.written ? 1 : 0);
                            throw std::runtime_error("device step: block " + std::to_string(layer) +
                                                     " read of '" + read.buffer + "' by '" + op.op +
                                                     "' at position " + std::to_string(op.position) +
                                                     " wants rows the block's write of it does not cover");
                        }
                        // A strided read: the writer's capture already holds every row of the
                        // declared buffer, so the reader's view is a gather out of it, not the
                        // arena's later values. Only a view the capture cannot express keeps the
                        // live pull, and that stays visible.
                        bool served = false;
                        AdValue viewed = writer_view(prev->second, read, snapped.now[j], &served);
                        if (served) {
                            snapped.now[j] = std::move(viewed);
                            continue;
                        }
                        std::string ws, rs;
                        for (int64_t d : prev->second.shape) ws += " " + std::to_string(d);
                        for (int64_t d : snapped.now[j].shape) rs += " " + std::to_string(d);
                        std::fprintf(stderr,
                                     "r97: block %lld read of '%s' by '%s' at %lld: writer [%s ] vs "
                                     "read [%s ] -- a view, live pull kept\n",
                                     static_cast<long long>(layer), read.buffer.c_str(),
                                     op.op.c_str(), static_cast<long long>(op.position),
                                     ws.c_str(), rs.c_str());
                        continue;
                    }
                    snapped.now[j] = writer_rows(prev->second, snapped.now[j]);
                }
                for (size_t j = 0; j < shaped.operands.size(); ++j) {
                    const Operand& operand = shaped.operands[j];
                    if (snapped.frozen[j]) continue;  // bound to the shared store by snap_record
                    if (operand.written) block_wrote.insert(operand.buffer);
                    if (operand.written && operand.buffer.rfind("kv:", 0) == 0 &&
                        before.find(operand.buffer) == before.end()) {
                        // A carried state this block has not written yet enters cold; the op's
                        // adjoint wants the state before the op, not the one it leaves.
                        snapped.now[j].value.assign(snapped.now[j].value.size(), 0.0);
                    }
                    snaps.set(op.position, j, snapped.now[j]);
                    if (operand.written && operand.view_shape.empty() && operand.rows > 0 &&
                        operand.rows == operand.shape[0]) {
                        before[operand.buffer] = std::move(snapped.now[j]);
                    }
                }
            });

            const Tape::Block& range = blocks[static_cast<size_t>(layer)];
            std::vector<OpRecord> records(ops.begin() + static_cast<std::ptrdiff_t>(range.begin),
                                          ops.begin() + static_cast<std::ptrdiff_t>(range.end));
            for (OpRecord& rec : records) {
                rec = own_raws(std::move(rec));
                tighten_rows(rec);
                if (options.reference_head_split) mark_join_lora(rec);
            }
            BoundaryNames block_names;
            block_names.add(incoming.buffer, 0, tokens);
            // This block's own read sets, from the whole tape: a consumer in a later block is
            // found in `ops`, and the block's input boundary is read outside the tape. Every block
            // record's consumer is a GEMM or a norm, so none is proven and every one keeps the
            // loud refusal -- the map is built anyway so the rule lives in one place.
            const std::map<int64_t, std::vector<int64_t>> block_reads =
                derive_read_sets(records, ops, shared, {incoming.buffer});
            bo.read_rows = &block_reads;
            const BackwardResult walked =
                backward_block(records, shared, block_names, {}, {incoming}, bo);
            merge_ops(row_time, walked.report.ops);
            merge_finiteness(walked.report);
            // The join holds the whole block's snapshots while the walk copies one op out of
            // them, so what it holds at once is the store plus that one op. `before` holds the
            // last write of every buffer an op replaced in place, a second copy of what the store
            // already keeps, and is part of what the replay holds. The step's frozen operands are
            // not in this store: they are held once for the whole step (shared_snapshot_bytes).
            int64_t before_bytes = 0;
            for (const auto& entry : before) before_bytes += value_bytes(entry.second.value);
            const int64_t store_bytes = snaps.bytes() + before_bytes;
            const int64_t op_bytes = walked.report.peak_snapshot_bytes;
            if (store_bytes + op_bytes > measured.peak_snapshot_bytes) {
                measured.peak_snapshot_bytes = store_bytes + op_bytes;
                measured.peak_snapshot_store_bytes = store_bytes;
                measured.peak_snapshot_op_bytes = op_bytes;
            }
            measured.peak_cotangent_bytes = std::max(
                measured.peak_cotangent_bytes,
                head_input_bytes + value_bytes(incoming.value) +
                    walked.report.peak_cotangent_bytes);

            const InputCotangent* below = pick_boundary(walked.inputs, tokens, meta.hidden);
            if (!below)
                throw std::runtime_error("device step: layer " + std::to_string(layer) +
                                         " left no cotangent on its input boundary");
            store_boundary(layer, below);
            incoming = IncomingCotangent{below->buffer, 0, tokens,
                                         first_rows(*below, tokens, meta.hidden)};
            // `snaps` dies with this iteration, so the base must not outlive it: the share answers
            // a miss from `base_`, and a stale pointer there is a use-after-free waiting for the
            // first walk that asks for a key this block did not store. The head's walk clears its
            // own base the same way.
            shared.set_base(nullptr);
            snaps = StoredSnapshots{};
        }
        measured.backward_seconds = seconds_since(backward_begin);
        // The op that produced the first non-finite forward value the step met, when a walked op
        // wrote it: the walk's own naming names the first reader of the value, this its writer.
        if (produced.found) {
            measured.forward_all_finite = false;
            measured.forward_first_non_finite_op = produced.op;
            measured.forward_first_non_finite_operand = produced.buffer;
            measured.forward_first_non_finite_operand_index = produced.operand;
            measured.forward_first_non_finite_position = produced.position;
            measured.forward_first_non_finite_index = produced.value;
        }
        // The frozen operands the step widened, each held once: the per-block figures above do
        // not include them, and the join holds them beside a block's store for the whole walk.
        measured.shared_snapshot_bytes = shared.bytes();

        // The card's staging is the step's total per leaf: one flush, then the delta's per-site
        // split. The split runs first: a fused weight's engine gradient is named after the
        // container's weight ("@fused:..."), so filling the missing ones before splitting would
        // add a zero under every module name the split then adds the real gradient for.
        measured.grad_stage_bytes = device::grad_stage_bytes();
        if (options.host_walk) {
            // The host rows add to the ParamSet themselves: nothing was staged, and there is
            // nothing to copy out.
            device::grad_stage_discard();
        } else if (!device::grad_stage_flush(grads)) {
            throw std::runtime_error("device step: the card's gradient staging did not copy out");
        }
        stage.disarm();
        split_adapter_grads(grads, params, groups, grad_prefix);
        fill_missing_grads(grads, params);
        order_grads_like(grads, params);

        // The report's own boundary copies grow with the number of layers; name them rather than
        // fold them into the peaks, which describe the join's working set.
        for (const std::vector<double>& v : measured.boundary_value)
            measured.kept_boundary_bytes += value_bytes(v);
        for (const std::vector<double>& v : measured.boundary_cotangent)
            measured.kept_boundary_bytes += value_bytes(v);

        measured.forward_seconds = forward_seconds;
        measured.head_seconds = head_seconds;
        // The maps iterate in op-name order, so the report's lists come out sorted.
        for (auto& [name, t] : row_time) {
            t.op = name;
            measured.row_seconds.push_back(std::move(t));
        }
        for (auto& [name, t] : pull_time) {
            t.op = name;
            measured.pull_seconds.push_back(std::move(t));
        }

        if (report) *report = std::move(measured);
        return result;
    };
}

}  // namespace r97

#endif  // RAD_ISSUE_OBSERVER && RAD_REPLAY_BLOCK && RAD_CTX_STREAM
