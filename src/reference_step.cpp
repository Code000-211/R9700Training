// r97/reference_step.cpp -- the tiny fixture's forward and backward in f64, from the twins.
//
// The value path is deliberately plain: every buffer is a std::vector<double> in an environment,
// every op is a registered twin, and the backward is the registered host adjoints walked by the
// same core the trainer uses. Only the op list and the weights touch radiance: the list comes from
// a recorded program or from one live host-kernel trace, and the weights come from the container
// through rad_core's reader. No kernel output is ever read into the value path.

#include "r97/reference_step.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "op_inventory.h"
#include "r97/adapter_io.h"
#include "r97/adjoint.h"
#include "r97/backward.h"
#include "r97/op.h"
#include "r97/params.h"

#include "format/encoding.h"
#include "format/radfile.h"

#if defined(RAD_ISSUE_OBSERVER)
#include "r97/tape.h"
#endif

namespace r97 {
namespace {

using json = nlohmann::json;

constexpr int64_t kKvBlock = 16;  // the fixture's paged-KV block, records carry it in the shape

int64_t ref_numel(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 0;
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

std::vector<int64_t> first_dim(const std::vector<int64_t>& shape, int64_t rows) {
    std::vector<int64_t> out = shape;
    if (!out.empty()) out[0] = rows;
    return out;
}

AdValue make_value(std::string dtype, std::vector<int64_t> shape, std::vector<double> value) {
    AdValue v;
    v.dtype = std::move(dtype);
    v.shape = std::move(shape);
    v.value = std::move(value);
    if (int64_t(v.value.size()) != ref_numel(v.shape))
        throw std::runtime_error("reference step: value does not fill its shape");
    return v;
}

AdValue zeros_like(const AdValue& v) {
    return make_value(v.dtype, v.shape, std::vector<double>((size_t)ref_numel(v.shape), 0.0));
}

int32_t as_int(const Operand& o) { return int32_t(o.rows); }

// ------------------------------------------------------------------ the op program

struct Program {
    std::vector<std::vector<int32_t>> batch;
    std::vector<OpRecord> ops;
};

std::vector<int64_t> ints_from_json(const json& a) {
    std::vector<int64_t> out;
    for (const json& v : a) out.push_back(v.get<int64_t>());
    return out;
}

Program parse_program(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("reference step: cannot open program " + path);
    json j;
    in >> j;
    if (!j.is_object() || j.value("format", "") != "r97-program")
        throw std::runtime_error("reference step: not an r97-program: " + path);
    if (j.value("version", 0) != 1)
        throw std::runtime_error("reference step: unsupported program version in " + path);

    Program p;
    if (j.contains("batch")) {
        for (const json& seq : j["batch"]) {
            std::vector<int32_t> ids;
            for (const json& v : seq) ids.push_back(v.get<int32_t>());
            p.batch.push_back(std::move(ids));
        }
    }
    if (!j.contains("ops") || !j["ops"].is_array())
        throw std::runtime_error("reference step: program has no ops: " + path);
    for (const json& o : j["ops"]) {
        OpRecord rec;
        rec.op = o.value("op", "");
        rec.block = o.value("block", -1);
        rec.position = o.value("position", 0);
        rec.n = o.value("n", 0);
        if (o.contains("params") && o["params"].is_object()) {
            for (auto it = o["params"].begin(); it != o["params"].end(); ++it) {
                if (it.value().is_number_integer())
                    rec.params.emplace_back(it.key(), it.value().get<int64_t>());
                else if (it.value().is_string())
                    rec.params.emplace_back(it.key(), it.value().get<std::string>());
                else
                    throw std::runtime_error("reference step: bad parameter " + it.key());
            }
        }
        if (o.contains("operands")) {
            for (const json& x : o["operands"]) {
                Operand op;
                op.buffer = x.value("buffer", "");
                op.dtype = x.value("dtype", "");
                if (x.contains("shape")) op.shape = ints_from_json(x["shape"]);
                op.row_lo = x.value("row_lo", 0);
                op.rows = x.value("rows", 0);
                op.written = x.value("written", false);
                rec.operands.push_back(std::move(op));
            }
        }
        p.ops.push_back(std::move(rec));
    }
    return p;
}

json program_json(const std::vector<std::vector<int32_t>>& batch,
                  const std::vector<OpRecord>& ops) {
    json j;
    j["format"] = "r97-program";
    j["version"] = 1;
    j["batch"] = batch;
    json arr = json::array();
    for (const OpRecord& rec : ops) {
        json o;
        o["op"] = rec.op;
        o["block"] = rec.block;
        o["position"] = rec.position;
        o["n"] = rec.n;
        json params = json::object();
        for (const auto& kv : rec.params) {
            if (std::holds_alternative<int64_t>(kv.second))
                params[kv.first] = std::get<int64_t>(kv.second);
            else
                params[kv.first] = std::get<std::string>(kv.second);
        }
        o["params"] = params;
        json operands = json::array();
        for (const Operand& op : rec.operands) {
            json x;
            x["buffer"] = op.buffer;
            x["dtype"] = op.dtype;
            x["shape"] = op.shape;
            x["row_lo"] = op.row_lo;
            x["rows"] = op.rows;
            x["written"] = op.written;
            operands.push_back(std::move(x));
        }
        o["operands"] = std::move(operands);
        arr.push_back(std::move(o));
    }
    j["ops"] = std::move(arr);
    return j;
}

// ------------------------------------------------------------------ weights, through rad_core

class Weights {
public:
    explicit Weights(const std::string& container) : container_(container) {
        if (file_.open(container.c_str()) != RAD_OK)
            throw std::runtime_error("reference step: cannot read container " + container);
    }

    bool has(const std::string& name) const { return file_.find(name.c_str()) != nullptr; }

    const AdValue& get(const std::string& name, const std::vector<int64_t>& wanted) {
        auto it = cache_.find(name);
        if (it != cache_.end()) return it->second;
        const RadFileEntry* e = file_.find(name.c_str());
        if (!e) throw std::runtime_error("reference step: no weight " + name);
        if (e->rank < 1) throw std::runtime_error("reference step: weight rank 0: " + name);
        // A conv weight is stored [out, in, kernel]; the tape sees [out, in*kernel]. Decode the
        // last axis as the logical columns and every leading axis as rows.
        const int64_t cols = e->shape[e->rank - 1];
        int64_t rows = 1;
        for (int i = 0; i + 1 < e->rank; ++i) rows *= e->shape[i];
        std::vector<int64_t> shape = wanted;
        if (ref_numel(shape) != rows * cols) {
            shape.assign(e->shape, e->shape + e->rank);
        }

        const RadEncoding& enc = file_.encoding(*e);
        std::vector<const void*> planes;
        for (int k = 0; k < enc.n_planes; ++k)
            planes.push_back(file_.data(file_.plane(*e, k)));
        rad::EncodedView view;
        rad::enc_view(enc, rows, cols, planes.data(), &view);
        std::vector<float> f((size_t)(rows * cols));
        if (rad::enc_decode_rows(view, 0, rows, f.data()) != RAD_OK)
            throw std::runtime_error("reference step: cannot decode weight " + name);
        std::vector<double> d(f.begin(), f.end());
        AdValue v = make_value("f64", std::move(shape), std::move(d));
        auto r = cache_.emplace(name, std::move(v));
        return r.first->second;
    }

private:
    std::string container_;
    rad::RadFile file_;
    std::map<std::string, AdValue> cache_;
};

// ------------------------------------------------------------------ the step's fixed layout

struct Layout {
    int64_t n_tokens = 0;
    int64_t n_seq = 0;
    int64_t graph_rows = 0;  // out_ids: last token + the block's tokens, per sequence
    int64_t bt_width = 1;
    std::vector<int64_t> start;  // token base per sequence
    std::vector<int64_t> out_start;  // row of the "last token" per sequence in out_ids
    std::vector<std::vector<int64_t>> bt;  // [n_seq][bt_width] physical blocks
    std::vector<int64_t> slots;            // per token, the flat KV slot
    int64_t vocab = 0;
    int64_t hidden = 0;
};

// ------------------------------------------------------------------ normalized graph

struct NormOp {
    OpRecord rec;
    // For each operand: where a write landed in the original full buffer. Reads are already
    // normalized, so these are only needed to put a write back.
    std::vector<int64_t> orig_row_lo;
    std::vector<int64_t> orig_rows;
    std::vector<std::string> orig_buffer;
    bool synthetic = false;
};

bool is_linear_weight(const std::string& name, const ModelMeta& meta) {
    for (const ModelMeta::Linear& l : meta.targetable)
        if (l.weight_name == name) return true;
    return false;
}

int64_t param_int(const OpRecord& rec, const std::string& key, int64_t fallback) {
    for (const auto& kv : rec.params)
        if (kv.first == key && std::holds_alternative<int64_t>(kv.second))
            return std::get<int64_t>(kv.second);
    return fallback;
}

void set_param(OpRecord* rec, const std::string& key, int64_t value) {
    for (auto& kv : rec->params)
        if (kv.first == key) { kv.second = value; return; }
    rec->params.emplace_back(key, value);
}

// The row count an op actually touched. The tape records the resolved count in `n`; a few ops
// (state-only, full-buffer) leave it zero and the operands carry the range instead.
int64_t active_rows(const OpRecord& rec) {
    if (rec.n > 0) return rec.n;
    int64_t rows = 0;
    for (const Operand& o : rec.operands)
        if (o.rows > 0) rows = std::max(rows, o.rows);
    return rows;
}

// The tape reuses raw buffer ids for unrelated constants (raw:10 is a zero vector at
// gdn_conv_prep and [[0],[1],...] at chunk_scan). Name them by what they hold so the backward
// core sees one shape per buffer.
std::string raw_semantic(const OpRecord& rec, const Operand& o) {
    const std::string key = o.buffer.substr(4);
    if (key == "0") return "ids";
    if (key == "1") return rec.op == "gather_rows" ? "out_ids" : "pos";
    if (key == "2") return rec.op == "kv_store" ? "kv_slots" : "bt";
    if (key == "3") return rec.op == "attn_paged" ? "seqused" : "cu";
    if (key == "4") return "out_ids";
    if (key == "5") return "kv_slots";
    if (key == "6" || key == "7") return "cu";
    if (key == "8") return "kv_slots";
    if (key == "9") return "state_idx";
    if (key == "10") return o.shape.size() == 2 ? "arange" : "zeros";
    return "raw_" + key;
}

std::vector<NormOp> normalize(const std::vector<OpRecord>& input, const ModelMeta& meta) {
    std::vector<NormOp> out;
    out.reserve(input.size());
    for (const OpRecord& rec : input) {
        NormOp n;
        n.rec = rec;
        for (Operand& o : n.rec.operands) {
            /* The recurrent state is one buffer name reused by every block; the backward plan
             * keys versions by name, so name the block here and in the forward's environment. */
            if (o.buffer.rfind("kv:", 0) == 0 && o.buffer.find('@') == std::string::npos)
                o.buffer += "@" + std::to_string(rec.block);
            if (o.buffer.rfind("raw:", 0) == 0) o.buffer = "raw:" + raw_semantic(rec, o);
        }
        n.orig_row_lo.resize(rec.operands.size());
        n.orig_rows.resize(rec.operands.size());
        n.orig_buffer.resize(rec.operands.size());
        const int64_t active = active_rows(rec);
        for (size_t i = 0; i < rec.operands.size(); ++i) {
            const Operand& o = rec.operands[i];
            n.orig_row_lo[i] = o.row_lo;
            n.orig_rows[i] = o.rows;
            n.orig_buffer[i] = o.buffer;
            if (o.rows > 0) {
                n.rec.operands[i].shape = first_dim(o.shape, o.rows);
                n.rec.operands[i].row_lo = 0;
                n.rec.operands[i].rows = 0;
            }
        }
        n.rec.n = active;
        if (active > 0) {
            if (param_int(n.rec, "M", -1) >= 0) set_param(&n.rec, "M", active);
            if (param_int(n.rec, "q_len", -1) >= 0) set_param(&n.rec, "q_len", active);
        }
        // A few ops read or write a reshape of the same storage (attn_qg [T,4,64] against the
        // linear's [T,256]); the equal element count carries the values and the core only needs
        // the shape its adjoint expects.
        out.push_back(std::move(n));
    }
    return out;
}

// ------------------------------------------------------------------ the live trace

#if defined(RAD_ISSUE_OBSERVER)
std::vector<OpRecord> trace_batch(const ReferenceStepOptions& opts, const Batch& batch) {
    std::vector<std::vector<int32_t>> prompts;
    for (const TokenSeq& s : batch) prompts.emplace_back(s.ids.begin(), s.ids.end());
    Tape tape;
    std::string error;
    if (!record_prefill_batch(opts.container, opts.radiance_home, opts.kernels, prompts, tape,
                              &error))
        throw std::runtime_error("reference step: live trace failed: " + error);
    std::vector<OpRecord> ops = tape.ops();
    tape.detach();
    return ops;
}
#else
std::vector<OpRecord> trace_batch(const ReferenceStepOptions&, const Batch&) {
    throw std::runtime_error("reference step: this build has no issue observer");
}
#endif

// The twins find a few operands by the schema's name (linear.cpp's read_at/operand_pos); the tape
// names the real buffer. The copy handed to a twin carries the schema names, while the record the
// backward core sees keeps the buffer identities its versioning needs.
// The tape's buffer names are the fixture's own (blk.0.ssm_in.weight, h_out, ...); the host
// adjoints look their operands up by the schema's names. `operand_pos` maps a schema name to the
// read's position, so the graph keeps the names that connect its producers and consumers.
OpRecord twin_record(const OpRecord& rec) { return rec; }

// ------------------------------------------------------------------ the f64 environment

struct ValueEnv {
    std::map<std::string, AdValue> full;  // named buffers, at the shape the tape declares

    AdValue* find(const std::string& name) {
        auto it = full.find(name);
        return it == full.end() ? nullptr : &it->second;
    }
    const AdValue* find(const std::string& name) const {
        auto it = full.find(name);
        return it == full.end() ? nullptr : &it->second;
    }
    void put(const std::string& name, AdValue v) { full[name] = std::move(v); }
};

AdValue reshape_copy(const AdValue& v, std::vector<int64_t> shape, std::string dtype) {
    if (ref_numel(shape) != int64_t(v.value.size()))
        throw std::runtime_error("reference step: reshape changes the value count");
    return make_value(std::move(dtype), std::move(shape), v.value);
}

AdValue slice_rows(const AdValue& v, int64_t rows) {
    if (v.shape.empty()) return v;
    const int64_t row_elems = ref_numel(v.shape) / v.shape[0];
    std::vector<double> out(v.value.begin(), v.value.begin() + (size_t)(rows * row_elems));
    std::vector<int64_t> shape = v.shape;
    shape[0] = rows;
    return make_value(v.dtype, std::move(shape), std::move(out));
}

AdValue narrow_columns(const AdValue& v, int64_t off, int64_t cols) {
    if (v.shape.size() != 2) throw std::runtime_error("reference step: column view of a non-matrix");
    const int64_t rows = v.shape[0];
    std::vector<double> out((size_t)(rows * cols));
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols; ++c)
            out[(size_t)(r * cols + c)] = v.value[(size_t)(r * v.shape[1] + off + c)];
    return make_value(v.dtype, {rows, cols}, std::move(out));
}

// ------------------------------------------------------------------ raw operands

struct RawBuilder {
    Layout layout;
    const Batch* batch = nullptr;

    AdValue make(const std::string& name, const std::string& op, const Operand& o) const {
        const int64_t n = ref_numel(o.shape);
        std::vector<double> v((size_t)n, 0.0);
        auto fill = [&](const std::vector<int64_t>& values, const char* what) {
            if (int64_t(values.size()) != n)
                throw std::runtime_error(std::string("reference step: raw ") + what +
                                         " size mismatch");
            for (int64_t i = 0; i < n; ++i) v[(size_t)i] = double(values[(size_t)i]);
        };
        if (name == "ids") {
            std::vector<int64_t> ids;
            for (const TokenSeq& s : *batch) ids.insert(ids.end(), s.ids.begin(), s.ids.end());
            fill(ids, "ids");
        } else if (name == "pos") {
            std::vector<int64_t> pos;
            for (const TokenSeq& s : *batch)
                for (size_t i = 0; i < s.ids.size(); ++i) pos.push_back((int64_t)i);
            fill(pos, "positions");
        } else if (name == "out_ids") {
            /* gather_rows takes rows of the hidden buffer, not token ids: the block's last token
             * comes first so the first logits row predicts the first scored token, then the
             * sequence's own rows follow. */
            std::vector<int64_t> rows;
            for (size_t s = 0; s < batch->size(); ++s) {
                const int64_t len = (int64_t)(*batch)[s].ids.size();
                const int64_t start = layout.start[s];
                rows.push_back(start + len - 1);
                for (int64_t j = 0; j < len; ++j) rows.push_back(start + j);
            }
            fill(rows, "out_ids");
        } else if (name == "kv_slots") {
            fill(layout.slots, "kv_slots");
        } else if (name == "bt") {
            std::vector<int64_t> flat;
            for (const auto& row : layout.bt) flat.insert(flat.end(), row.begin(), row.end());
            fill(flat, "block table");
        } else if (name == "seqused") {
            std::vector<int64_t> lens;
            for (const TokenSeq& s : *batch) lens.push_back((int64_t)s.ids.size());
            fill(lens, "seqused");
        } else if (name == "cu") {
            std::vector<int64_t> cu{0};
            for (const TokenSeq& s : *batch) cu.push_back(cu.back() + (int64_t)s.ids.size());
            fill(cu, "cu_seqlens");
        } else if (name == "zeros") {
            // gdn_conv_prep passes a zero [n_seq].
        } else if (name == "state_idx") {
            std::vector<int64_t> idx;
            for (int64_t s = 0; s < int64_t(batch->size()); ++s) idx.push_back(s);
            fill(idx, "state index");
        } else if (name == "arange") {
            // chunk_scan passes [[0],[1],...], one row per sequence.
            if (o.shape.size() == 2 && o.shape[1] == 1)
                for (int64_t s = 0; s < int64_t(batch->size()); ++s) v[(size_t)s] = double(s);
        } else {
            throw std::runtime_error("reference step: unknown raw operand " + o.buffer +
                                     " at " + op);
        }
        return make_value(o.dtype.empty() ? std::string("i32") : o.dtype, o.shape,
                          std::move(v));
    }
};

// ------------------------------------------------------------------ the virtual objective tape

class RefTape final : public ForwardTape {
public:
    RefTape(const Batch& batch, const AdValue& logits, int64_t vocab)
        : batch_(batch), logits_(logits), vocab_(vocab),
          rows_(logits.shape.empty() ? 0 : logits.shape[0]) {}

    const Batch& batch() const override { return batch_; }
    int64_t total_scored() const override {
        int64_t n = 0;
        for (const TokenSeq& s : batch_)
            for (size_t i = 1; i < s.ids.size(); ++i) n += s.score[i] == 1 ? 1 : 0;
        return n;
    }
    std::vector<std::string> boundaries() const override { return {"logits"}; }
    std::vector<int64_t> shape(const std::string& buffer) const override {
        if (buffer != "logits") return {};
        return {rows_, vocab_};
    }
    void window(const std::string&, int64_t* lo, int64_t* hi) const override {
        *lo = 0;
        *hi = rows_;
    }
    std::vector<float> read(const std::string& buffer, int64_t lo, int64_t hi,
                            bool) const override {
        (void)buffer;
        std::vector<float> out((size_t)((hi - lo) * vocab_));
        for (int64_t r = lo; r < hi; ++r)
            for (int64_t c = 0; c < vocab_; ++c)
                out[(size_t)((r - lo) * vocab_ + c)] =
                    float(logits_.value[(size_t)(r * vocab_ + c)]);
        return out;
    }

private:
    const Batch& batch_;
    const AdValue& logits_;
    int64_t vocab_;
    int64_t rows_ = 0;
};

// The loss's f64 cotangent on the graph logits. The objective's own gradient is f32 (the
// contract's Seed); the backward wants the value in f64, so it is written once here from the
// same softmax the objective computes.
AdValue ce_dlogits(const Batch& batch, const AdValue& logits, int64_t vocab) {
    const int64_t rows = logits.shape.empty() ? 0 : logits.shape[0];
    std::vector<double> g((size_t)(rows * vocab), 0.0);
    int64_t scored = 0;
    for (const TokenSeq& s : batch)
        for (size_t i = 1; i < s.ids.size(); ++i) scored += s.score[i] == 1 ? 1 : 0;
    if (scored == 0) return make_value("f64", {rows, vocab}, std::move(g));
    const double scale = 1.0 / double(scored);
    int64_t row = 0;
    for (const TokenSeq& seq : batch) {
        for (size_t i = 1; i < seq.ids.size(); ++i) {
            if (seq.score[i] != 1) continue;
            if (row >= rows) throw std::runtime_error("reference step: seed rows overflow");
            const double* r = &logits.value[(size_t)(row * vocab)];
            double m = -std::numeric_limits<double>::infinity();
            for (int64_t c = 0; c < vocab; ++c) m = std::max(m, r[c]);
            double z = 0.0;
            for (int64_t c = 0; c < vocab; ++c) z += std::exp(r[c] - m);
            const double lz = std::log(z);
            const int32_t target = seq.ids[i];
            for (int64_t c = 0; c < vocab; ++c) {
                double d = std::exp(r[c] - m - lz) * scale;
                if (c == target) d -= scale;
                g[(size_t)(row * vocab + c)] += d;
            }
            ++row;
        }
    }
    return make_value("f64", {rows, vocab}, std::move(g));
}

// ------------------------------------------------------------------ the step

// A private op: add a narrow increment into a column range of a wider buffer. The LoRA expand of a
// fused weight produces its own module's [M, W] block; this is how it lands in the fused output
// without widening the module's B or losing the column offset.
void ref_add_cols_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                       std::vector<AdValue>* out) {
    const AdValue& y = operands[0];
    const AdValue& inc = operands[1];
    const int64_t off = param_int(rec, "off", 0);
    if (y.shape.empty() || inc.shape.size() != 2 || inc.shape[0] != y.shape[0])
        throw std::runtime_error("ref_add_cols: shape mismatch");
    const int64_t row = ref_numel(y.shape) / y.shape[0];
    if (off + inc.shape[1] > row)
        throw std::runtime_error("ref_add_cols: increment past the row");
    AdValue o = y;
    for (int64_t m = 0; m < inc.shape[0]; ++m)
        for (int64_t c = 0; c < inc.shape[1]; ++c)
            o.value[(size_t)(m * row + off + c)] += inc.value[(size_t)(m * inc.shape[1] + c)];
    out->push_back(std::move(o));
}

void ref_add_cols_adjoint(const AdjointCall& call) {
    const AdValue& y = call.fwd[0];
    const AdValue& inc = call.fwd[1];
    const AdValue& dy = call.cot[0];
    const int64_t off = param_int(call.rec, "off", 0);
    AdValue& dy_in = call.dc->at(0);
    AdValue& dinc = call.dc->at(1);
    for (size_t i = 0; i < dy_in.value.size(); ++i) dy_in.value[i] += dy.value[i];
    const int64_t row = ref_numel(y.shape) / y.shape[0];
    for (int64_t m = 0; m < inc.shape[0]; ++m)
        for (int64_t c = 0; c < inc.shape[1]; ++c)
            dinc.value[(size_t)(m * inc.shape[1] + c)] +=
                dy.value[(size_t)(m * row + off + c)];

}

// A private op: select rows of the full logits into the boundary's buffer. The forward writes one
// logits row per token (the BOS row included); the objective's tape has one row per scored token,
// so this is where the two row spaces meet.
void ref_select_rows_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                          std::vector<AdValue>* out) {
    const AdValue& x = operands[0];
    const AdValue& idx = operands[1];
    if (x.shape.size() != 2 || idx.value.empty())
        throw std::runtime_error("ref_select_rows: bad operands");
    const int64_t M = (int64_t)idx.value.size();
    const int64_t V = x.shape[1];
    AdValue y = make_value(x.dtype, {M, V}, std::vector<double>((size_t)(M * V), 0.0));
    for (int64_t r = 0; r < M; ++r) {
        const int64_t src = (int64_t)idx.value[(size_t)r];
        if (src < 0 || src >= x.shape[0]) throw std::runtime_error("ref_select_rows: row out of range");
        for (int64_t c = 0; c < V; ++c)
            y.value[(size_t)(r * V + c)] = x.value[(size_t)(src * V + c)];
    }
    out->push_back(std::move(y));
    (void)rec;
}

void ref_select_rows_adjoint(const AdjointCall& call) {
    const AdValue& idx = call.fwd[1];
    const AdValue& dy = call.cot[0];
    AdValue& dx = call.dc->at(0);
    const int64_t V = dy.shape[1];
    for (int64_t r = 0; r < (int64_t)idx.value.size(); ++r) {
        const int64_t dst = (int64_t)idx.value[(size_t)r];
        for (int64_t c = 0; c < V; ++c)
            dx.value[(size_t)(dst * V + c)] += dy.value[(size_t)(r * V + c)];
    }
}

[[maybe_unused]] const bool kRefSelectRowsRegistered = [] {
    twin_register("ref_select_rows", ref_select_rows_twin, nullptr);
    adjoint_register("ref_select_rows", Side::host, ref_select_rows_adjoint, {}, nullptr);
    return true;
}();

[[maybe_unused]] const bool kRefAddColsRegistered = [] {
    twin_register("ref_add_cols", ref_add_cols_twin, nullptr);
    adjoint_register("ref_add_cols", Side::host, ref_add_cols_adjoint, {}, nullptr);
    return true;
}();

class ReferenceStep {
public:
    ReferenceStep(Model model, ReferenceStepOptions opts)
        : model_(std::move(model)), opts_(std::move(opts)),
          weights_(opts_.container), meta_(read_model_meta(opts_.container)) {
        if (!opts_.program.empty()) {
            Program p = parse_program(opts_.program);
            program_batch_ = std::move(p.batch);
            program_ops_ = std::move(p.ops);
            program_mode_ = true;
        }
        for (size_t i = 0; i < meta_.targetable.size(); ++i) {
            const auto& l = meta_.targetable[i];
            int64_t off = 0;
            for (size_t k = 0; k < i; ++k)
                if (meta_.targetable[k].weight_name == l.weight_name &&
                    meta_.targetable[k].slot < l.slot)
                    off += meta_.targetable[k].out;
            slot_out_[l.slot] = off;
        }
    }

    StepResult run(const Batch& batch, const ParamSet& params, ParamSet* grads) {
        WireBuffers wire;
        model_.delta->to_wire(params, &wire);

        if (program_mode_) check_program_batch(batch);
        std::vector<OpRecord> ops = program_mode_ ? program_ops_ : trace_batch(opts_, batch);
        Layout layout = make_layout(batch, ops);
        Values run = forward(batch, layout, ops, wire);

        RefTape tape(batch, run.logits, run.vocab);
        ObjectiveResult obj = model_.objective(tape, params);
        StepResult result;
        result.loss = obj.loss;
        result.scored = obj.scored;
        if (!grads) return result;

        if (!wire.sites.empty()) backward(batch, layout, run, params, grads);
        return result;
    }

    static json program_of(const Batch& batch, const ReferenceStepOptions& opts) {
        std::vector<OpRecord> ops = trace_batch(opts, batch);
        std::vector<std::vector<int32_t>> prompts;
        for (const TokenSeq& s : batch) prompts.emplace_back(s.ids.begin(), s.ids.end());
        return program_json(prompts, ops);
    }

private:
    struct Values {
        std::vector<NormOp> ops;
        std::vector<std::vector<AdValue>> reads;   // per op, reads in schema order
        std::vector<std::vector<AdValue>> writes;  // per op, writes in writes' order
        std::vector<std::vector<size_t>> read_index;   // per op, schema index of each read
        std::vector<std::vector<size_t>> write_index;  // per op, schema index of each write
        AdValue logits;
        int64_t graph_rows = 0;
        int64_t scored_rows = 0;
        int64_t vocab = 0;
        std::map<std::string, AdValue> constants;
    };

    Layout make_layout(const Batch& batch, const std::vector<OpRecord>& ops) const {
        Layout l;
        l.n_seq = (int64_t)batch.size();
        l.bt_width = 1;
        for (const OpRecord& rec : ops) {
            for (const Operand& o : rec.operands) {
                if (o.buffer == "raw:2" && o.shape.size() == 2) l.bt_width = o.shape[1];
                if (o.written && o.buffer == "logits" && o.shape.size() == 2)
                    l.vocab = o.shape[1];
            }
        }
        if (l.vocab == 0) throw std::runtime_error("reference step: the graph writes no logits");
        for (const TokenSeq& s : batch) {
            l.start.push_back(l.n_tokens);
            l.n_tokens += (int64_t)s.ids.size();
            l.out_start.push_back(l.graph_rows);
            l.graph_rows += (int64_t)s.ids.size() + 1;
        }
        int64_t next_block = 0;
        for (const TokenSeq& s : batch) {
            const int64_t len = (int64_t)s.ids.size();
            const int64_t blocks = (len + kKvBlock - 1) / kKvBlock;
            std::vector<int64_t> row((size_t)l.bt_width, -1);
            for (int64_t b = 0; b < blocks; ++b) row[(size_t)b] = next_block++;
            l.bt.push_back(std::move(row));
        }
        l.slots.clear();
        for (size_t s = 0; s < batch.size(); ++s) {
            const int64_t len = (int64_t)batch[s].ids.size();
            for (int64_t p = 0; p < len; ++p) {
                const int64_t block = l.bt[s][(size_t)(p / kKvBlock)];
                l.slots.push_back(block * kKvBlock + p % kKvBlock);
            }
        }
        return l;
    }

    void check_program_batch(const Batch& batch) const {
        if (program_batch_.size() != batch.size())
            throw std::runtime_error("reference step: program batch size differs");
        for (size_t s = 0; s < batch.size(); ++s) {
            if (program_batch_[s].size() != batch[s].ids.size())
                throw std::runtime_error("reference step: program sequence length differs");
            for (size_t i = 0; i < batch[s].ids.size(); ++i)
                if (program_batch_[s][i] != batch[s].ids[i])
                    throw std::runtime_error("reference step: program batch ids differ");
        }
    }

    // The graph the step runs: the fixture's ops, normalized to the rows they touch, with one LoRA
    // triple per wired site after the linear whose output it corrects.
    std::vector<NormOp> build_graph(const std::vector<OpRecord>& ops, const WireBuffers& wire,
                                    const Layout& layout, const std::vector<int64_t>& select_rows,
                                    Values* out) {
        std::vector<NormOp> graph = normalize(ops, meta_);
        std::map<std::string, std::vector<size_t>> by_weight;
        for (size_t i = 0; i < wire.sites.size(); ++i)
            by_weight[wire.sites[i].weight_name].push_back(i);

        std::vector<NormOp> with_lora;
        with_lora.reserve(graph.size() + wire.sites.size() * 3);
        for (size_t gi = 0; gi < graph.size(); ++gi) {
            const NormOp& lin = graph[gi];
            with_lora.push_back(lin);
            if (lin.rec.op != "gemm_nt" && lin.rec.op != "gemm_nt_q" &&
                lin.rec.op != "logits_gemm")
                continue;
            // The weight this linear reads, if a targetable source lands in it.
            std::string weight;
            for (const Operand& o : lin.rec.operands) {
                if (o.written) continue;
                std::string key = o.buffer;
                const std::string suffix = ".weight";
                if (key.size() > suffix.size() &&
                    key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0)
                    key.resize(key.size() - suffix.size());
                if (by_weight.count(key)) { weight = key; break; }
            }
            if (weight.empty()) continue;
            const auto& sites = by_weight[weight];

            // The linear's input and output, both already normalized to the active rows.
            const Operand& in_op = lin.rec.operands[0];
            std::string in_buf = in_op.buffer;
            std::vector<int64_t> in_shape = in_op.shape;
            size_t out_i = 0;
            while (out_i < lin.rec.operands.size() && !lin.rec.operands[out_i].written) ++out_i;
            if (out_i == lin.rec.operands.size())
                throw std::runtime_error("reference step: linear has no output");
            const Operand& out_op = lin.rec.operands[out_i];
            const std::string out_buf = out_op.buffer;
            const std::vector<int64_t> out_shape = out_op.shape;
            const int64_t M = out_shape[0];
            // The linear's output may be rank 3 (a head split); the adapter's columns are the
            // flattened trailing axes.
            const int64_t N_total = M > 0 ? ref_numel(out_shape) / M : 0;
            const std::vector<int64_t> y_shape{M, N_total};

            for (size_t si : sites) {
                const WireBuffers::Site& site = wire.sites[si];
                const int64_t R = site.rank;
                if (R <= 0) continue;
                // The linear's input may be rank 3 (a GDN head split); the adapter's A is
                // [rank, in_features] with both trailing axes flattened.
                const int64_t K = (int64_t)site.a.size() / R;
                const int64_t N = (int64_t)site.b.size() / R;
                const std::vector<int64_t> lora_in_shape = in_shape;
                const int64_t g = (int64_t)si;
                slot_module_[g] = site.module;  // the global site index the LoRA records carry
                const int64_t S = g + 1;
                const int64_t off = slot_out_[site.slot];

                const std::string slot_buf = "lora_slot:" + std::to_string(g);
                const std::string a_buf = "lora_a:" + std::to_string(g);
                const std::string b_buf = "lora_b:" + std::to_string(g);
                const std::string scale_buf = "lora_scale:" + std::to_string(g);
                const std::string t_buf = "lora_t:" + std::to_string(g);
                const std::string inc_buf = "lora_inc:" + std::to_string(g);

                std::vector<double> slot((size_t)M, double(g));
                out->constants[slot_buf] = make_value("i32", {M}, std::move(slot));
                std::vector<double> a((size_t)(S * R * K), 0.0);
                for (int64_t r = 0; r < R; ++r)
                    for (int64_t k = 0; k < K; ++k)
                        a[(size_t)((g * R + r) * K + k)] = site.a[(size_t)(r * K + k)];
                out->constants[a_buf] = make_value("f64", {S, R, K}, std::move(a));
                std::vector<double> b((size_t)(S * N * R), 0.0);
                for (int64_t n = 0; n < N; ++n)
                    for (int64_t r = 0; r < R; ++r)
                        b[(size_t)((g * N + n) * R + r)] = site.b[(size_t)(n * R + r)];
                out->constants[b_buf] = make_value("f64", {S, N, R}, std::move(b));
                std::vector<double> scale((size_t)(S * N), 0.0);
                for (int64_t n = 0; n < N; ++n) scale[(size_t)(g * N + n)] = site.scale;
                out->constants[scale_buf] = make_value("f32", {S, N}, std::move(scale));

                auto operand = [](std::string buffer, std::string dtype, std::vector<int64_t> shape,
                                  bool written) {
                    Operand o;
                    o.buffer = std::move(buffer);
                    o.dtype = std::move(dtype);
                    o.shape = std::move(shape);
                    o.written = written;
                    return o;
                };
                auto add = [&](OpRecord rec, std::vector<int64_t> orig_rows = {},
                               std::vector<std::string> orig_buf = {},
                               std::vector<int64_t> orig_lo = {}) {
                    NormOp n;
                    n.rec = std::move(rec);
                    n.orig_rows = std::move(orig_rows);
                    n.orig_buffer = std::move(orig_buf);
                    n.orig_row_lo = std::move(orig_lo);
                    n.synthetic = true;
                    with_lora.push_back(std::move(n));
                };

                // lora_shrink: t = a @ A^T.
                OpRecord shr;
                shr.op = "lora_shrink";
                shr.block = lin.rec.block;
                shr.n = M;
                set_param(&shr, "M", M);
                set_param(&shr, "R", R);
                set_param(&shr, "K", K);
                shr.params.emplace_back("dtype", std::string("f64"));
                shr.operands.push_back(operand(in_buf, in_op.dtype, lora_in_shape, false));
                shr.operands.push_back(operand("none:lora", "invalid", {}, false));
                shr.operands.push_back(operand(slot_buf, "i32", {M}, false));
                shr.operands.push_back(operand(a_buf, "f64", {S, R, K}, false));
                shr.operands.push_back(operand(t_buf, "f64", {M, R}, true));
                add(shr);

                // lora_expand: inc = scale * t @ B^T, only this module's output width.
                OpRecord exp;
                exp.op = "lora_expand";
                exp.block = lin.rec.block;
                exp.n = M;
                set_param(&exp, "M", M);
                set_param(&exp, "N", N);
                set_param(&exp, "R", R);
                exp.params.emplace_back("dtype", std::string("f64"));
                exp.operands.push_back(operand(t_buf, "f64", {M, R}, false));
                exp.operands.push_back(operand(slot_buf, "i32", {M}, false));
                exp.operands.push_back(operand(b_buf, "f64", {S, N, R}, false));
                exp.operands.push_back(operand(scale_buf, "f32", {S, N}, false));
                exp.operands.push_back(operand(inc_buf, "f64", {M, N}, true));
                add(exp);

                // add_cols: y[:, off:off+N] += inc.
                OpRecord adc;
                adc.op = "ref_add_cols";
                adc.block = lin.rec.block;
                adc.n = M;
                set_param(&adc, "M", M);
                set_param(&adc, "off", off);
                adc.operands.push_back(operand(out_buf, out_op.dtype, out_shape, false));
                adc.operands.push_back(operand(inc_buf, "f64", {M, N}, false));
                adc.operands.push_back(operand(out_buf, out_op.dtype, out_shape, true));
                std::vector<int64_t> o_rows(adc.operands.size(), 0);
                std::vector<std::string> o_buf(adc.operands.size(), "");
                std::vector<int64_t> o_lo(adc.operands.size(), 0);
                o_rows[2] = lin.orig_rows[out_i];
                o_buf[2] = lin.orig_buffer[out_i];
                o_lo[2] = lin.orig_row_lo[out_i];
                add(std::move(adc), std::move(o_rows), std::move(o_buf), std::move(o_lo));
            }
        }
        // The forward's logits have one row per token, BOS included; the objective's tape has one
        // row per scored token. Keep the full buffer under a private name and put the scored rows
        // on the boundary the objective names "logits".
        for (NormOp& n : with_lora)
            for (size_t i = 0; i < n.rec.operands.size(); ++i) {
                if (n.rec.operands[i].buffer == "logits")
                    n.rec.operands[i].buffer = "logits_full";
                if (i < n.orig_buffer.size() && n.orig_buffer[i] == "logits")
                    n.orig_buffer[i] = "logits_full";
            }
        const int64_t M = (int64_t)select_rows.size();
        std::vector<double> idx((size_t)M, 0.0);
        for (int64_t r = 0; r < M; ++r) idx[(size_t)r] = double(select_rows[(size_t)r]);
        out->constants["ref_select_idx"] = make_value("i32", {M}, std::move(idx));
        OpRecord sel;
        sel.op = "ref_select_rows";
        sel.block = with_lora.empty() ? 0 : with_lora.back().rec.block;
        sel.n = M;
        set_param(&sel, "M", M);
        Operand sx;
        sx.buffer = "logits_full";
        sx.dtype = "f64";
        sx.shape = {layout.graph_rows, layout.vocab};
        sel.operands.push_back(sx);
        Operand si;
        si.buffer = "ref_select_idx";
        si.dtype = "i32";
        si.shape = {M};
        sel.operands.push_back(si);
        Operand sy;
        sy.buffer = "logits";
        sy.dtype = "f64";
        sy.shape = {M, layout.vocab};
        sy.written = true;
        sel.operands.push_back(sy);
        NormOp seln;
        seln.rec = std::move(sel);
        seln.synthetic = true;
        with_lora.push_back(std::move(seln));
        for (size_t i = 0; i < with_lora.size(); ++i) with_lora[i].rec.position = (int64_t)i;
        return with_lora;
    }

    Values forward(const Batch& batch, const Layout& layout, const std::vector<OpRecord>& ops_in,
                   const WireBuffers& wire) {
        Values out;
        out.vocab = layout.vocab;
        out.graph_rows = layout.graph_rows;
        batch_nseq_ = layout.n_seq;
        std::vector<int64_t> select_rows;
        for (size_t s = 0; s < batch.size(); ++s)
            for (size_t i = 1; i < batch[s].ids.size(); ++i)
                if (batch[s].score[i] == 1)
                    select_rows.push_back(layout.out_start[s] + 1 + (int64_t)(i - 1));
        out.scored_rows = (int64_t)select_rows.size();
        out.ops = build_graph(ops_in, wire, layout, select_rows, &out);
        out.reads.resize(out.ops.size());
        out.writes.resize(out.ops.size());
        out.read_index.resize(out.ops.size());
        out.write_index.resize(out.ops.size());

        constants_ = out.constants;
        ValueEnv env;
        RawBuilder raw{layout, &batch};
        for (size_t oi = 0; oi < out.ops.size(); ++oi) {
            const NormOp& n = out.ops[oi];
            const OpRecord& rec = n.rec;
            // Every operand as the op saw it: reads are resolved, a write is the value the buffer
            // held just before (an in-place op's input), or a zero of its shape. The backward core
            // asks for a snapshot of every operand, not just the reads.
            std::vector<AdValue> pre(rec.operands.size());
            std::vector<AdValue> reads;
            for (size_t i = 0; i < rec.operands.size(); ++i) {
                const Operand& o = rec.operands[i];
                if (!o.written) {
                    out.read_index[oi].push_back(i);
                    pre[i] = resolve_read(o, rec, env, raw, layout, batch);
                    reads.push_back(pre[i]);
                } else {
                    pre[i] = pre_write(o, env);
                }
            }
            TwinFn twin = twin_find(rec.op, rec.params, nullptr);
            if (!twin) throw std::runtime_error("reference step: no twin for op " + rec.op);
            std::vector<AdValue> writes;
            OpRecord trec = twin_record(rec);
            std::vector<AdValue> twin_reads = reads;
            // rope's record carries only the in-place write and the positions; its twin takes the
            // buffer's prior value as the first read.
            if (rec.op == "rope" && reads.size() == 1 && !pre.empty() && rec.operands[0].written)
                twin_reads = {pre[0], reads[0]};
            twin(trec, twin_reads, &writes);
            size_t wi = 0;
            for (size_t i = 0; i < rec.operands.size(); ++i) {
                const Operand& o = rec.operands[i];
                if (!o.written) continue;
                out.write_index[oi].push_back(i);
                if (wi >= writes.size())
                    throw std::runtime_error("reference step: twin wrote too little for " + rec.op);
                out.writes[oi].push_back(writes[wi]);
                apply_write(n, i, writes[wi], &env);
                if (o.buffer == "logits" && o.shape.size() == 2 &&
                    int64_t(writes[wi].value.size()) == out.scored_rows * layout.vocab)
                    out.logits = writes[wi];
                ++wi;
            }
            out.reads[oi] = std::move(pre);
        }
        if (out.logits.value.empty())
            throw std::runtime_error("reference step: no logits in the graph");
        return out;
    }

    /* The recurrent state buffers are reused by every block: the plugin resets them at the block
     * boundary, and the record names the same buffer for each layer. Key them by block so a block
     * never reads the previous layer's state. */
    std::string state_key(const std::string& name, int64_t block) {
        if (name.rfind("kv:", 0) == 0 && name.find('@') == std::string::npos)
            return name + "@" + std::to_string(block);
        return name;
    }

    AdValue resolve_read(const Operand& o, const OpRecord& rec, ValueEnv& env,
                         const RawBuilder& raw, const Layout& layout, const Batch& batch) {
        (void)layout;
        (void)batch;
        if (o.dtype == "invalid" || o.shape.empty()) {
            AdValue v;
            v.dtype = "invalid";
            v.value.assign(1, 0.0);  // one scalar: the backward core checks numel(shape)
            return v;
        }
        if (o.buffer.rfind("raw:", 0) == 0) {
            return raw.make(o.buffer.substr(4), rec.op, o);
        }
        const std::string key = state_key(o.buffer, rec.block);
        if (!o.buffer.empty() && o.buffer.rfind("kv:", 0) == 0 && !env.find(key)) {
            return make_value(o.dtype, o.shape,
                              std::vector<double>((size_t)ref_numel(o.shape), 0.0));
        }
        if (const AdValue* v = env.find(key)) {
            if (v->shape == o.shape) return *v;
            if (ref_numel(v->shape) == ref_numel(o.shape))
                return reshape_copy(*v, o.shape, o.dtype);
            if (!v->shape.empty() && !o.shape.empty() && v->shape[0] >= o.shape[0] &&
                ref_numel(v->shape) / v->shape[0] == ref_numel(o.shape) / o.shape[0]) {
                AdValue s = slice_rows(*v, o.shape[0]);
                return reshape_copy(s, o.shape, o.dtype);
            }
            throw std::runtime_error("reference step: read of " + o.buffer + " has no value");
        }
        {
            auto it = constants_.find(o.buffer);
            if (it != constants_.end()) return it->second;
        }
        if (o.buffer.rfind("lora_", 0) == 0 || o.buffer == "none:lora")
            throw std::runtime_error("reference step: unknown constant " + o.buffer);
        AdValue w = weights_.get(o.buffer, o.shape);
        if (w.shape == o.shape) return w;
        if (ref_numel(w.shape) == ref_numel(o.shape))
            return reshape_copy(w, o.shape, o.dtype);
        throw std::runtime_error("reference step: weight " + o.buffer + " has the wrong shape");
    }

    AdValue pre_write(const Operand& o, ValueEnv& env) {
        if (o.dtype == "invalid" || o.shape.empty()) {
            AdValue v;
            v.dtype = "invalid";
            v.value.assign(1, 0.0);  // one scalar: the backward core checks numel(shape)
            return v;
        }
        if (const AdValue* v = env.find(o.buffer)) {
            if (v->shape == o.shape) return *v;
            if (ref_numel(v->shape) == ref_numel(o.shape))
                return reshape_copy(*v, o.shape, o.dtype);
            if (!v->shape.empty() && !o.shape.empty() && v->shape[0] >= o.shape[0] &&
                ref_numel(v->shape) / v->shape[0] == ref_numel(o.shape) / o.shape[0])
                return reshape_copy(slice_rows(*v, o.shape[0]), o.shape, o.dtype);
        }
        return make_value(o.dtype, o.shape,
                          std::vector<double>((size_t)ref_numel(o.shape), 0.0));
    }

    void apply_write(const NormOp& n, size_t i, const AdValue& v, ValueEnv* env) {
        const Operand& o = n.rec.operands[i];
        const int64_t orig_rows = i < n.orig_rows.size() ? n.orig_rows[i] : 0;
        const int64_t orig_lo = i < n.orig_row_lo.size() ? n.orig_row_lo[i] : 0;
        const std::string name = state_key(
            i < n.orig_buffer.size() && !n.orig_buffer[i].empty() ? n.orig_buffer[i] : o.buffer,
            n.rec.block);
        if (orig_rows <= 0) {
            env->put(name, v);
            return;
        }
        AdValue* full = env->find(name);
        if (!full) {
            // A first write to a ranged buffer: keep its declared full shape, then fill the rows.
            std::vector<int64_t> shape = o.shape;
            if (!shape.empty()) shape[0] = std::max<int64_t>(shape[0], orig_rows);
            full = &(*env).full.emplace(name, make_value(o.dtype, shape,
                std::vector<double>((size_t)ref_numel(shape), 0.0))).first->second;
        }
        const int64_t row_elems = ref_numel(v.shape) / v.shape[0];
        for (int64_t r = 0; r < v.shape[0]; ++r)
            for (int64_t k = 0; k < row_elems; ++k) {
                const int64_t dst = (orig_lo + r) * row_elems + k;
                if (dst >= 0 && dst < int64_t(full->value.size()))
                    full->value[(size_t)dst] = v.value[(size_t)(r * row_elems + k)];
            }
    }

    void backward(const Batch& batch, const Layout& layout, const Values& run,
                  const ParamSet& params, ParamSet* grads) {
        StoredSnapshots snaps;
        for (size_t oi = 0; oi < run.ops.size(); ++oi)
            for (size_t j = 0; j < run.reads[oi].size(); ++j) {
                snaps.set(run.ops[oi].rec.position, j, run.reads[oi][j]);
            }

        AdValue dlogits = ce_dlogits(batch, run.logits, run.vocab);
        std::vector<IncomingCotangent> incoming;
        incoming.push_back(
            IncomingCotangent{"logits", 0, run.scored_rows, dlogits.value});

        RefTape tape(batch, run.logits, run.vocab);
        BackwardOptions bo;
        bo.grads = grads;
        bo.prefix = "";
        bo.slot_prefix = [this](int64_t slot) {
            auto it = slot_module_.find(slot);
            if (it == slot_module_.end())
                throw std::runtime_error("reference step: adapter slot " + std::to_string(slot) +
                                         " has no module prefix");
            return it->second;
        };
        std::vector<OpRecord> records;
        records.reserve(run.ops.size());
        for (const NormOp& n : run.ops) records.push_back(twin_record(n.rec));
        backward_block(records, snaps, BoundaryNames::from(tape), {}, incoming, bo);
        (void)params;
    }

    Model model_;
    ReferenceStepOptions opts_;
    Weights weights_;
    ModelMeta meta_;
    bool program_mode_ = false;
    std::vector<std::vector<int32_t>> program_batch_;
    std::vector<OpRecord> program_ops_;
    std::map<int64_t, std::string> slot_module_;
    std::map<int64_t, int64_t> slot_out_;
    std::map<std::string, AdValue> constants_;
    int64_t batch_nseq_ = 0;
};

}  // namespace

StepFn make_reference_step(Model model, ReferenceStepOptions opts) {
    auto step = std::make_shared<ReferenceStep>(std::move(model), std::move(opts));
    return [step](const Batch& batch, const ParamSet& params, ParamSet* grads) {
        return step->run(batch, params, grads);
    };
}

}  // namespace r97
