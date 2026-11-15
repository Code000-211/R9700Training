// forward.cpp -- the training forward's driver: one engine run per batch, one recorded step out.
//
// The step is radiance's own prefill mode (patches/training/0103): the batch's sequences are
// submitted together as scoring requests, so the engine's batch builder gives each of them a row
// per position, and the mode copies each position's logits off by prompt. The engine is the one
// the fixture's run_prefill drives -- a fresh engine per step, so the key-value pool, the
// scheduler and every buffer start from nothing -- and the graph is read off the rank's context
// with the issue observer (patches/training/0101), on the host kernel tier.
//
// Rows. The boundary "logits" is in the batch's own row space: sequence i's row p predicts token
// p+1 of that sequence, so a sequence of L tokens has L-1 rows and the batch has the sum of them.
// The engine hands back one vector per prompt, one row of `n_vocab` f32 per position of that
// prompt; the driver reads the batch's rows straight out of those vectors, a window at a time, and
// keeps no copy. The step's tape, its block residuals and the head's hidden rows are in the step's
// token order, which is the batch's order when the whole batch ran in one step -- the driver
// checks that it did, rather than assumes it.
#include "r97/forward.h"

#if R97_FORWARD

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine.h"           // radiance's core/; Engine, Config, rad_strerror
#include "rad_device.h"       // rad_dev_props / rad_memcpy_async / rad_stream_sync: the dump's copy
#include "rad_plugin.h"       // the fp8 decoders join_common's read_elem calls
#include "join_common.h"      // elem_size / numel / read_elem: the R97_DUMP_FWD scan reads values
#include "r97/loss.h"         // bf16_to_f32: the step's buffers widened to f32
#include "util/stallwatch.h"  // radiance's core/; the watch serve() takes

namespace r97 {
namespace {

[[noreturn]] void fail(const std::string& why) { throw std::runtime_error("r97: forward: " + why); }

std::string num(int64_t v) { return std::to_string(v); }

int64_t scored_in(const TokenSeq& s) {
    int64_t n = 0;
    for (size_t i = 1; i < s.score.size(); ++i) n += s.score[i] ? 1 : 0;
    return n;
}

size_t element_bytes(const std::string& dtype) {
    if (dtype == "bf16") return 2;
    if (dtype == "f32") return 4;
    fail("a buffer of dtype '" + dtype + "', which this forward cannot widen to f32");
}

float widen(const void* at, const std::string& dtype) {
    if (dtype == "f32") {
        float v = 0.0f;
        std::memcpy(&v, at, sizeof(v));
        return v;
    }
    if (dtype == "bf16") {
        uint16_t v = 0;
        std::memcpy(&v, at, sizeof(v));
        return bf16_to_f32(v);
    }
    fail("a buffer of dtype '" + dtype + "', which this forward cannot widen to f32");
}

// R97_DUMP_FWD=1: is the forward's finiteness being watched? Read once per step, at the issue.
bool dump_fwd_enabled() {
    const char* d = std::getenv("R97_DUMP_FWD");
    return d != nullptr && d[0] == '1';
}

// One writer is enough: once an op has produced a non-finite value, its descendants' lines only
// bury the name. Reset per step by the driver.
bool g_dump_fwd_done = false;
// How many probe lines have printed this step (see dump_fwd_scan). Reset per step by the driver.
long g_dump_fwd_probes = 0;
// The written operands read and the ones skipped, so a silent run is told from one that looked.
long g_dump_fwd_scanned = 0;
long g_dump_fwd_skipped = 0;

// The dtypes read_elem widens to f64. A written operand of any other -- "f8", a plugin-private
// quantised tensor -- has no f64 form to scan, and is left alone rather than read as a wrong one.
bool readable_dtype(const std::string& d) {
    return d == "bf16" || d == "f16" || d == "f32" || d == "f64" || d == "i8" || d == "u8" ||
           d == "i32" || d == "u32" || d == "i64" || d == "u64" || d == "fp8_e4m3" ||
           d == "fp8_e5m2";
}

// The first non-finite value in one operand's span, or -1, and the largest finite magnitude seen:
// a probe line with maxabs 0 says the span read as zeros (a copy that raced, or a wrong address)
// rather than as the values the op wrote. `base` is host memory: on the device tier the caller has
// already copied the span back, because the tape's pointers are the card's and reading one here
// faults (the device join's pull_operands copies for the same reason).
int64_t first_non_finite_at(const std::string& dtype, const uint8_t* base, int64_t count,
                            int64_t es, double* maxabs) {
    int64_t at = -1;
    double m = 0.0;
    for (int64_t i = 0; i < count; ++i) {
        const double v = read_elem(dtype, base + i * es, 0);
        if (!std::isfinite(v)) {
            if (at < 0) at = i;
            continue;
        }
        const double a = std::fabs(v);
        if (a > m) m = a;
    }
    if (maxabs != nullptr) *maxabs = m;
    return at;
}

// R97_DUMP_FWD=1: for every forward op in order, the first non-finite value of each operand the op
// declared it wrote. The tape's issue hook is called after the op is recorded and while the buffers
// still hold what it wrote, so this names the writer of a value the backward only ever meets later,
// once the buffer has been reused -- which is why the join's own scan (backward.cpp) reports the
// reader's record (the head's rmsnorm) rather than the op the inf was born at. One line per
// finding names the op, its position, the operand and its schema index, the value's flat index,
// its row and column, and the value. The first non-finite per operand is enough, and printing
// stops at the first op that produced one: the writer, not its descendants.
//
// `host_memory` says the tape's pointers are host addresses (the host backend) and can be read in
// place; otherwise the operand's span is copied back on `stream`, which is the lane the op issued
// on, so the copy orders behind it. A copy the card refuses names its operand and is skipped: the
// dump must never be the thing that stops a step.
void dump_fwd_scan(const OpRecord& op, const std::vector<OperandContent>& content, bool host_memory,
                   RadStream stream) {
    if (g_dump_fwd_done) return;
    bool found = false;
    for (size_t j = 0; j < op.operands.size() && j < content.size(); ++j) {
        const Operand& o = op.operands[j];
        if (!o.written) continue;
        const OperandContent& c = content[j];
        const int64_t es = elem_size(o.dtype);
        const int64_t inner =
            o.shape.empty() || o.shape[0] <= 0 ? 0 : numel(o.shape) / o.shape[0];
        const int64_t row_lo = o.rows > 0 ? o.row_lo : 0;
        const int64_t rows = o.rows > 0 ? o.rows : (o.shape.empty() ? 0 : o.shape[0]);
        const int64_t count = rows * inner;
        const int64_t bytes = count * es;
        // A written operand the scan will not read says why: silence must mean "read and finite",
        // never "never looked". One line per skipped operand, and the run's totals at the end of
        // the forward.
        const char* why = nullptr;
        if (c.data == nullptr) {
            why = "no-pointer";
        } else if (es <= 0 || !readable_dtype(o.dtype)) {
            why = "dtype";
        } else if (o.shape.empty() || o.shape[0] <= 0) {
            why = "no-shape";
        } else if (inner <= 0 || count <= 0) {
            why = "empty";
        } else if (c.bytes < static_cast<size_t>(bytes)) {
            why = "short-span";
        }
        if (why != nullptr) {
            ++g_dump_fwd_skipped;
            std::fprintf(stderr,
                         "DUMPFWD skip op=%s pos=%lld buffer=%s operand=%zu dtype=%s bytes=%zu "
                         "need=%lld why=%s\n",
                         op.op.c_str(), static_cast<long long>(op.position), o.buffer.c_str(), j,
                         o.dtype.c_str(), c.bytes, static_cast<long long>(bytes), why);
            std::fflush(stderr);
            continue;
        }
        ++g_dump_fwd_scanned;
        std::vector<uint8_t> copy;
        const uint8_t* base;
        if (host_memory) {
            base = static_cast<const uint8_t*>(c.data);
        } else {
            if (stream == nullptr) {
                std::fprintf(stderr,
                             "DUMPFWD op=%s pos=%lld buffer=%s operand=%zu: the card's lane gave "
                             "no stream to copy %lld bytes back\n",
                             op.op.c_str(), static_cast<long long>(op.position), o.buffer.c_str(),
                             j, static_cast<long long>(bytes));
                std::fflush(stderr);
                continue;
            }
            copy.resize(static_cast<size_t>(bytes));
            if (rad_memcpy_async(copy.data(), c.data, bytes, stream) != 0 ||
                rad_stream_sync(stream) != 0) {
                std::fprintf(stderr,
                             "DUMPFWD op=%s pos=%lld buffer=%s operand=%zu: the card would not "
                             "copy %lld bytes back: %s\n",
                             op.op.c_str(), static_cast<long long>(op.position), o.buffer.c_str(),
                             j, static_cast<long long>(bytes), rad_dev_last_error());
                std::fflush(stderr);
                continue;
            }
            base = copy.data();
        }
        double maxabs = 0.0;
        const int64_t at = first_non_finite_at(o.dtype, base, count, es, &maxabs);
        // The first few written operands, and every op of the head, print what was read even when
        // it is finite: a probe with maxabs 0 says the copy raced or the address was wrong, so a
        // run with no finding is told from a scan that read nothing.
        if (g_dump_fwd_probes < 6 || op.position >= 1450) {
            ++g_dump_fwd_probes;
            std::fprintf(stderr,
                         "DUMPFWD probe op=%s pos=%lld buffer=%s operand=%zu bytes=%zu count=%lld "
                         "data=%p maxabs=%.6g nonfinite_at=%lld\n",
                         op.op.c_str(), static_cast<long long>(op.position), o.buffer.c_str(), j,
                         c.bytes, static_cast<long long>(count), c.data, maxabs,
                         static_cast<long long>(at));
        }
        if (at < 0) continue;
        std::fprintf(stderr,
                     "DUMPFWD op=%s pos=%lld buffer=%s operand=%zu index=%lld row=%lld col=%lld "
                     "value=%.9g data=%p\n",
                     op.op.c_str(), static_cast<long long>(op.position), o.buffer.c_str(), j,
                     static_cast<long long>(at), static_cast<long long>(row_lo + at / inner),
                     static_cast<long long>(at % inner),
                     read_elem(o.dtype, base + at * es, 0), c.data);
        found = true;
    }
    if (found) {
        std::fflush(stderr);
        g_dump_fwd_done = true;
    }
}

}  // namespace

// ------------------------------------------------------------------ ForwardStep, the ForwardTape

std::vector<int64_t> ForwardStep::shape(const std::string& buffer) const {
    if (buffer != "logits")
        fail("no boundary '" + buffer + "': the forward has \"logits\" and nothing else");
    return {rows_, vocab_};
}

void ForwardStep::window(const std::string& buffer, int64_t* row_lo, int64_t* row_hi) const {
    if (buffer != "logits")
        fail("no boundary '" + buffer + "': the forward has \"logits\" and nothing else");
    const int64_t lo = window_set_ ? window_lo_ : 0;
    const int64_t hi = window_set_ ? window_hi_ : rows_;
    if (row_lo) *row_lo = lo;
    if (row_hi) *row_hi = hi;
}

std::vector<float> ForwardStep::read(const std::string& buffer, int64_t row_lo, int64_t row_hi,
                                     bool differentiated) const {
    if (buffer != "logits")
        fail("no boundary '" + buffer + "': the forward has \"logits\" and nothing else");
    if (row_lo < 0 || row_hi > rows_ || row_lo > row_hi)
        fail("a read of rows " + num(row_lo) + ".." + num(row_hi) + " of a boundary of " +
             num(rows_) + " rows");
    // The window is what the driver recorded for this call: an objective that read past it would
    // be handed rows that no seed and no block above it covers.
    if (window_set_ && (row_lo < window_lo_ || row_hi > window_hi_))
        fail("a read of rows " + num(row_lo) + ".." + num(row_hi) + " outside the window " +
             num(window_lo_) + ".." + num(window_hi_));

    reads_.push_back(TapeRead{buffer, row_lo, row_hi, differentiated});

    std::vector<float> out(size_t(row_hi - row_lo) * size_t(vocab_));
    for (const Piece& piece : pieces_) {
        const int64_t lo = std::max(row_lo, piece.off);
        const int64_t hi = std::min(row_hi, piece.off + piece.rows);
        if (lo >= hi) continue;
        const std::vector<float>& rows = (*logits_)[size_t(piece.prompt)];
        std::memcpy(out.data() + size_t(lo - row_lo) * size_t(vocab_),
                    rows.data() + size_t(lo - piece.off) * size_t(vocab_),
                    size_t(hi - lo) * size_t(vocab_) * sizeof(float));
    }
    return out;
}

// ------------------------------------------------------------------ the driver

ForwardDriver::~ForwardDriver() = default;

std::unique_ptr<ForwardDriver> ForwardDriver::open(const ForwardConfig& cfg, std::string* err) {
    auto no = [&](const std::string& why) {
        if (err) *err = "r97: forward: " + why;
        return nullptr;
    };
    if (cfg.model.empty()) return no("no model container to train");
    if (cfg.radiance_home.empty())
        return no("no radiance home: the architecture plugin is loaded from it");
    if (cfg.kernels.empty()) return no("no host kernel tier to run the step on");
    // The key-value pool is differentiated through by the backward, and only bf16 has an adjoint.
    if (cfg.kv_dtype != "bf16")
        return no("the key-value pool is bf16, not '" + cfg.kv_dtype +
                  "': the backward differentiates through the pool, and only bf16 has an adjoint");
    if (cfg.max_tok < 1 || cfg.max_ctx < 1 || cfg.max_seqs < 1)
        return no("a budget of max_tok=" + num(cfg.max_tok) + " max_ctx=" + num(cfg.max_ctx) +
                  " max_seqs=" + num(cfg.max_seqs) + ": a step needs room for one token");
    if (cfg.window_rows < 1)
        return no("window_rows=" + num(cfg.window_rows) + ": a window is a row at least");
    {
        std::ifstream in(cfg.model, std::ios::binary);
        if (!in) return no("the container '" + cfg.model + "' does not open");
    }
    return std::unique_ptr<ForwardDriver>(new ForwardDriver(cfg));
}

void ForwardDriver::set_block_hook(std::function<void(const BlockView&)> hook) {
    block_hook_ = std::move(hook);
}

void ForwardDriver::set_lora(int64_t max_loras, int64_t max_rank,
                             std::vector<std::string> targets, std::string adapter_name,
                             std::string input_basis) {
    lora_max_loras_ = max_loras;
    lora_max_rank_ = max_rank;
    lora_targets_ = std::move(targets);
    lora_adapter_name_ = std::move(adapter_name);
    lora_input_basis_ = std::move(input_basis);
}

void ForwardDriver::set_engine_hook(std::function<void(rad::Engine&)> hook) {
    engine_hook_ = std::move(hook);
}

const ForwardStep& ForwardDriver::last() const {
    if (!last_) throw std::logic_error("r97: forward: no step has run yet");
    return *last_;
}

StepFn ForwardDriver::step(const Model& model) {
    Objective objective = model.objective;  // the loss belongs to the model, not to the driver
    ForwardDriver* self = this;
    return [self, objective](const Batch& batch, const ParamSet& params, ParamSet* /*grads*/) {
        // Forward only: `grads` is left alone. The join that walks the recorded tape fills it.
        return self->run(batch, params, objective);
    };
}

StepResult ForwardDriver::run(const Batch& batch, const ParamSet& params,
                              const Objective& objective) {
    if (batch.empty()) fail("an empty batch");
    if (int64_t(batch.size()) > cfg_.max_seqs)
        fail("a batch of " + num(int64_t(batch.size())) + " sequences, one step holds " +
             num(cfg_.max_seqs));

    int64_t tokens = 0;
    int64_t scored = 0;
    for (const TokenSeq& s : batch) {
        if (s.ids.empty()) fail("a sequence with no token");
        if (s.ids.size() != s.score.size())
            fail("a sequence of " + num(int64_t(s.ids.size())) + " tokens with a score mask of " +
                 num(int64_t(s.score.size())));
        tokens += int64_t(s.ids.size());
        scored += scored_in(s);
    }
    // One step for the whole batch, or nothing: a chunked prompt would be a second step's graph,
    // and the recorded tape would no longer be this batch's forward.
    if (tokens > cfg_.max_tok)
        fail("a batch of " + num(tokens) + " tokens, one step holds " + num(cfg_.max_tok) +
             ": raise max_tok or shorten the batch, since a chunked prompt is a second step");
    step_tokens_ = tokens;

    run_engine(batch);

    std::unique_ptr<ForwardStep> step(new ForwardStep());
    assemble(*step, batch);
    if (scored != step->total_scored_)
        fail("the batch's score mask counts " + num(scored) + " scored tokens, the step's rows " +
             num(step->total_scored_));

    // One objective call per window of positions, the Result's loss and scored summed: the driver
    // holds no loss of its own.
    for (int64_t lo = 0; lo < step->rows_; lo += cfg_.window_rows) {
        const int64_t hi = std::min(lo + cfg_.window_rows, step->rows_);
        step->window_lo_ = lo;
        step->window_hi_ = hi;
        step->window_set_ = true;
        step->reads_.clear();  // the check looks at this call's reads, and at this window's rows
        const ObjectiveResult r = objective(*step, params);
        // Every Result is checked before anything walks it: a differentiated read must be answered
        // by a seed on that buffer.
        check_seeds(*step, r);
        step->result_.loss += r.loss;
        step->result_.scored += r.scored;
        for (const Seed& seed : r.seeds) step->seeds_.push_back(seed);
        ++step->windows_;
    }
    step->window_set_ = false;

    const StepResult result = step->result_;
    last_ = std::move(step);
    return result;
}

void ForwardDriver::run_engine(const Batch& batch) {
    residual_seen_ = false;
    residual_name_.clear();
    bracket_error_.clear();

    // The old step's buffers die with its engine, so the tape and the step that point into them go
    // first, and the new engine is started before the tape attaches to it.
    tape_.detach();
    tape_.clear();
    // R97_DUMP_FWD's wrapper is not the recorder's to clear: a step with no recorder (a plain
    // forward) would leave it on the tape, and the next step would nest it. Drop it here, and let
    // the step find its writer afresh.
    if (dump_fwd_enabled()) {
        tape_.set_issue_hook(nullptr);
        g_dump_fwd_done = false;
        g_dump_fwd_probes = 0;
    }
    last_.reset();
    engine_.reset();

    rad::Config cfg;
    cfg.model = cfg_.model;
    cfg.radiance_home = cfg_.radiance_home;
    cfg.kernel_hierarchy = cfg_.kernels;
    cfg.max_tok = cfg_.max_tok;
    cfg.max_ctx = cfg_.max_ctx;
    cfg.max_seqs = cfg_.max_seqs;
    cfg.kv_cache_dtype = cfg_.kv_dtype;
    // Off for training: nothing is drafted, nothing is predicted ahead, the vision tower is not
    // built, and every sequence runs from position 0 in this one step.
    cfg.n_spec = 0;
    cfg.mm_max_patches = 0;
    // The step is one prefill piece (forward.h:3-8), and radiance's linear-state checkpoint interval
    // also caps a prefill chunk at that many tokens (core/sched/geometry.cpp:209), so there is none
    // here: its own default (2048, core/rad_core.h:942) would split a longer prompt into pieces the
    // driver must then refuse. Training never resumes from a checkpoint.
    cfg.checkpoint_interval = 0;
    cfg.prefill_prompts.clear();
    cfg.prefill_prompts.reserve(batch.size());
    for (const TokenSeq& s : batch) cfg.prefill_prompts.push_back(s.ids);
    cfg.prefill_logits.clear();
    cfg.lora.max_loras = lora_max_loras_;
    cfg.lora.max_rank = lora_max_rank_;
    cfg.lora.targets = lora_targets_;
    cfg.lora.input_basis = lora_input_basis_;
    cfg.prefill_lora = lora_adapter_name_;

    engine_.reset(new rad::Engine(std::move(cfg)));
    int s = engine_->start();
    if (s < 0)
        fail("the container '" + cfg_.model + "' did not start on the host tier: " +
             std::string(rad_strerror(s)));

    // The step's graph is read off the rank's own context, from the issues the architecture makes.
    tape_.set_bracket_hook(
        [this](const Tape& tape, int64_t block, bool begin) { on_bracket(tape, block, begin); });
    tape_.attach(engine_->issue_ctx());
    if (!tape_.attached()) fail("the engine gave no rank context to record the step on");
    if (engine_hook_) engine_hook_(*engine_);

    // R97_DUMP_FWD=1: watch the forward's own values for the first non-finite one, at the issue
    // where the buffers still hold what the op left (dump_fwd_scan). The boundary recorder the
    // engine hook just installed owns the tape's one issue hook, so wrap it rather than replace it:
    // the recorder still sees every issue, and the scan reads the record it is handed. The tape's
    // pointers are host addresses only on the host backend; otherwise the scan copies each written
    // span back on the lane the op issued on. The armed line is the evidence the hook ran at all,
    // so "no DUMPFWD line" can mean "every value was finite" rather than "the scan never started".
    if (dump_fwd_enabled()) {
        RadDeviceProps props{};
        const bool host_memory = rad_dev_props(0, &props) == 0 && props.is_host_backend != 0;
        // The lane's stream is looked up per op, not cached: the op just issued on it, and the
        // copy must order behind the op, not behind whatever the stream held before serve began.
        std::fprintf(stderr, "DUMPFWD armed: host_memory=%d\n", static_cast<int>(host_memory));
        std::fflush(stderr);
        Tape::IssueHook inner = tape_.issue_hook();
        tape_.set_issue_hook([this, inner, host_memory](
                                 const Tape& t, const OpRecord& op,
                                 const std::vector<OperandContent>& content) {
            if (inner) inner(t, op, content);
            RadStream stream = nullptr;
#if defined(RAD_CTX_STREAM)
            if (engine_) stream = rad_ctx_cur_stream(engine_->issue_ctx());
#endif
            dump_fwd_scan(op, content, host_memory, stream);
        });
    }

    auto watch = std::make_unique<rad::StallWatch>("r97 forward", 60000);
    s = engine_->serve(std::move(watch));
    tape_.detach();
    if (s < 0)
        fail("the step of " + num(int64_t(batch.size())) + " sequences did not run: " +
             std::string(rad_strerror(s)));
    if (!bracket_error_.empty()) fail(bracket_error_);

    const std::vector<std::vector<float>>& rows = engine_->config().prefill_logits;
    if (rows.size() != batch.size())
        fail("the engine returned " + num(int64_t(rows.size())) +
             " prompts' logits for a batch of " + num(int64_t(batch.size())) + " sequences");
}

void ForwardDriver::assemble(ForwardStep& step, const Batch& batch) {
    step.batch_ = &batch;
    step.tape_ = &tape_;
    step.logits_ = &engine_->config().prefill_logits;

    int64_t rows = 0;
    int64_t positions = 0;
    int64_t vocab = 0;
    for (size_t i = 0; i < batch.size(); ++i) {
        const int64_t len = int64_t(batch[i].ids.size());
        const std::vector<float>& got = (*step.logits_)[i];
        if (got.size() % size_t(len) != 0)
            fail("prompt " + num(int64_t(i)) + " came back with " + num(int64_t(got.size())) +
                 " logits for " + num(len) + " tokens");
        const int64_t cols = int64_t(got.size()) / len;
        if (vocab == 0) vocab = cols;
        if (cols != vocab)
            fail("prompt " + num(int64_t(i)) + " came back over " + num(cols) +
                 " logits a row, the first over " + num(vocab));
        // L-1 rows per sequence: row p predicts token p+1, so the last position predicts nothing.
        step.pieces_.push_back(ForwardStep::Piece{rows, positions, len - 1, int64_t(i)});
        rows += len - 1;
        positions += len;
        step.total_scored_ += scored_in(batch[i]);
    }
    step.vocab_ = vocab;
    step.rows_ = rows;
    step.result_ = StepResult{0.0, 0};

    // The step must have run the whole batch in one pass: a chunked prompt would leave the tape
    // with one chunk's graph and the boundaries with one chunk's rows.
    const std::vector<Tape::Block>& blocks = tape_.blocks();
    int64_t layers = 0;
    for (const Tape::Block& b : blocks) layers = std::max(layers, b.index + 1);
    if (layers > 0 && int64_t(blocks.size()) != layers)
        fail("the engine ran the batch in " + num(int64_t(blocks.size()) / layers) +
             " steps, not one: raise max_tok, or shorten the batch");

    head_hidden(step, batch);
}

void ForwardDriver::head_hidden(ForwardStep& step, const Batch& batch) {
    // The head is the op that writes the boundary's buffer; the first operand it reads is the
    // hidden state it sees. The engine puts the sampler's rows first and the scored rows after
    // them, so the hidden row of the batch's row r is the head's row `skip + r`, with `skip` taken
    // off the rows the head wrote against the tokens the batch holds.
    int64_t tokens = 0;
    for (const TokenSeq& s : batch) tokens += int64_t(s.ids.size());

    const std::vector<OpRecord>& ops = tape_.ops();
    size_t head = ops.size();
    size_t written = 0;
    for (size_t i = ops.size(); i-- > 0;) {
        for (size_t j = 0; j < ops[i].operands.size(); ++j) {
            const Operand& o = ops[i].operands[j];
            if (!o.written || o.buffer != "logits") continue;
            head = i;
            written = j;
            break;
        }
        if (head != ops.size()) break;
    }
    if (head == ops.size()) fail("the step issued no op that writes the \"logits\" boundary");

    const std::vector<Operand>& head_ops = ops[head].operands;
    const std::vector<OperandContent>& head_content = tape_.contents(head);
    const Operand& logits = head_ops[written];
    if (logits.row_lo != 0)
        fail("the head wrote the \"logits\" boundary from row " + num(logits.row_lo));
    const int64_t skip = logits.rows - tokens;
    if (skip < 0)
        fail("the head wrote " + num(logits.rows) + " rows for a batch of " + num(tokens) +
             " tokens: the step's out rows cannot hold the scored ones");

    size_t in = head_ops.size();
    for (size_t j = 0; j < head_ops.size(); ++j)
        if (!head_ops[j].written && head_ops[j].shape.size() == 2) {
            in = j;
            break;
        }
    if (in == head_ops.size()) fail("the op writing \"logits\" reads no hidden state to give back");
    const Operand& hid = head_ops[in];
    if (hid.rows < skip + step.rows_)
        fail("the head read " + num(hid.rows) + " hidden rows, the step's scored rows reach " +
             num(skip + step.rows_));

    // The row order the engine used is the batch's, or this forward's row arithmetic is a guess
    // and the hidden rows and the block boundaries land on the wrong sequence. Checked on each
    // sequence's first row: the engine's own copy is by prompt, the head's is by step, and the two
    // agree only when the plan kept the batch's order.
    const float* step_logits = static_cast<const float*>(head_content[written].data);
    if (!step_logits) fail("the head's \"logits\" rows are not readable");
    for (const ForwardStep::Piece& piece : step.pieces_) {
        if (piece.rows <= 0) continue;
        const int64_t row = skip + piece.pos;
        if (row < 0 || row >= logits.rows) fail("the head's rows do not cover the batch");
        if (std::memcmp(step_logits + size_t(row) * size_t(step.vocab_),
                        (*step.logits_)[size_t(piece.prompt)].data(),
                        size_t(step.vocab_) * sizeof(float)) != 0)
            fail("the plan did not keep the batch's order: prompt " + num(piece.prompt) +
                 "'s rows are not at the batch row they belong to, so the hidden state and the "
                 "block boundaries cannot be mapped onto the batch");
    }

    // The head's input at the scored rows, the container's element type widened to f32.
    const size_t esz = element_bytes(hid.dtype);
    step.hidden_cols_ = hid.shape[1];
    const uint8_t* base = static_cast<const uint8_t*>(head_content[in].data);
    for (const ForwardStep::Piece& piece : step.pieces_) {
        const TokenSeq& s = batch[size_t(piece.prompt)];
        for (int64_t p = 1; p < int64_t(s.ids.size()); ++p) {
            if (!s.score[size_t(p)]) continue;
            const int64_t row = skip + piece.pos + p - 1;
            step.hidden_rows_.push_back(row);
            const uint8_t* src = base + size_t(row) * size_t(step.hidden_cols_) * esz;
            for (int64_t c = 0; c < step.hidden_cols_; ++c)
                step.hidden_.push_back(widen(src + size_t(c) * esz, hid.dtype));
        }
    }
}

BlockView ForwardDriver::residual_at(const Tape& tape, int64_t block, bool begin) {
    // The residual stream: the last op that wrote the step's whole row count. The architecture
    // writes it in place, so the op that has just run still holds the value this bracket names and
    // the next block overwrites it. Which buffer it is, is the architecture's business, and a
    // block-quantised build may hand a different one to each block.
    const std::vector<OpRecord>& ops = tape.ops();
    for (size_t i = ops.size(); i-- > 0;) {
        const std::vector<OperandContent>& content = tape.contents(i);
        for (size_t j = 0; j < ops[i].operands.size(); ++j) {
            const Operand& o = ops[i].operands[j];
            if (!o.written || o.row_lo != 0 || o.shape.size() != 2) continue;
            if (o.rows != step_tokens_) continue;
            BlockView view;
            view.block = block;
            view.begin = begin;
            view.residual = content[j].data;
            view.rows = o.rows;
            view.cols = o.shape[1];
            view.dtype = o.dtype;
            return view;
        }
    }
    return BlockView{};
}

void ForwardDriver::on_bracket(const Tape& tape, int64_t block, bool begin) {
    if (!bracket_error_.empty()) return;
    const BlockView view = residual_at(tape, block, begin);
    if (!view.residual) {
        bracket_error_ = "block " + num(block) + (begin ? "'s entry" : "'s exit") +
                         " holds no residual of the step's " + num(step_tokens_) +
                         " rows: the step did not run the batch in one piece";
        return;
    }
    const std::string what = view.dtype + " x" + num(view.cols);
    if (!residual_seen_) {
        residual_seen_ = true;
        residual_name_ = what;
    } else if (what != residual_name_) {
        bracket_error_ = "block " + num(block) + (begin ? "'s entry" : "'s exit") +
                         " holds a " + what + " residual, the first boundary held " + residual_name_;
        return;
    }
    if (block_hook_) block_hook_(view);
}

}  // namespace r97

#endif  // R97_FORWARD
