// src/step_host.cpp -- the host join. See include/r97/step_host.h and docs/STEPS-HOST.md.
#include "rad_runtime.h"  // RAD_ISSUE_OBSERVER / RAD_REPLAY_BLOCK, before the guard

#if defined(RAD_ISSUE_OBSERVER) && defined(RAD_REPLAY_BLOCK)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine.h"  // the complete rad::Engine behind ForwardDriver's unique_ptr
#include "rad_planes.h"  // the joined form a quantised lm_head reaches the tape in
#include "rad_plugin.h"  // rad_fp8e4m3_to_f32, for the int8 trunk's fp8 cache operands
#include "r97/step_host.h"
#include "r97/adapter_io.h"
#include "r97/adjoint.h"
#include "r97/backward.h"
#include "r97/live_adapter.h"
#include "r97/replay.h"
#include "r97/tape.h"

namespace r97 {
namespace {

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

// The payload of a stored value or cotangent: what the join's memory figures count.
int64_t value_bytes(const std::vector<double>& v) {
    return static_cast<int64_t>(v.size()) * static_cast<int64_t>(sizeof(double));
}

int64_t elem_size(const std::string& dtype) {
    if (dtype == "bf16" || dtype == "f16") return 2;
    if (dtype == "f32" || dtype == "i32" || dtype == "u32") return 4;
    if (dtype == "f64" || dtype == "i64" || dtype == "u64") return 8;
    if (dtype == "i8" || dtype == "u8" || dtype == "f8" || dtype == "fp8_e4m3" ||
        dtype == "fp8_e5m2")
        return 1;
    return 0;
}

double read_elem(const std::string& dtype, const void* data, int64_t i) {
    if (dtype == "f32") return static_cast<const float*>(data)[i];
    if (dtype == "f64") return static_cast<const double*>(data)[i];
    if (dtype == "i32") return static_cast<const int32_t*>(data)[i];
    if (dtype == "u32") return static_cast<const uint32_t*>(data)[i];
    if (dtype == "i64") return static_cast<double>(static_cast<const int64_t*>(data)[i]);
    if (dtype == "u64") return static_cast<double>(static_cast<const uint64_t*>(data)[i]);
    if (dtype == "i8") return static_cast<const int8_t*>(data)[i];
    if (dtype == "u8") return static_cast<const uint8_t*>(data)[i];
    if (dtype == "fp8_e4m3") return rad_fp8e4m3_to_f32(static_cast<const uint8_t*>(data)[i]);
    if (dtype == "fp8_e5m2") return rad_fp8e5m2_to_f32(static_cast<const uint8_t*>(data)[i]);
    if (dtype == "bf16") {
        const uint32_t u = static_cast<uint32_t>(static_cast<const uint16_t*>(data)[i]) << 16;
        float f;
        std::memcpy(&f, &u, sizeof f);
        return f;
    }
    if (dtype == "f16") {
        const uint16_t h = static_cast<const uint16_t*>(data)[i];
        const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
        uint32_t exp = (h >> 10) & 0x1fu;
        uint32_t mant = h & 0x3ffu;
        uint32_t bits;
        if (exp == 0) {
            if (mant == 0) {
                bits = sign;
            } else {
                exp = 127 - 15 + 1;
                while ((mant & 0x400u) == 0) {
                    mant <<= 1;
                    --exp;
                }
                bits = sign | (exp << 23) | ((mant & 0x3ffu) << 13);
            }
        } else if (exp == 31) {
            bits = sign | 0x7f800000u | (mant << 13);
        } else {
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
        float f;
        std::memcpy(&f, &bits, sizeof f);
        return f;
    }
    throw std::runtime_error("host step: cannot snapshot operand dtype '" + dtype + "'");
}

// A quantised weight the engine joins from planes (abi/rad_planes.h) reaches the tape as one
// buffer: the header, the entries, the codes, then the scales. The adjoints take the dequantised
// weight, the product the GEMM's epilogue forms, so read the planes out by their roles. Only a
// whole-buffer operand can be joined -- a ranged or viewed one is a slice of something else --
// and the codes' extents have to be the operand's own, or this is not the weight the op read.
bool joined_dense(const Operand& op, const OperandContent& content, std::vector<double>* out) {
    if (content.data == nullptr || content.bytes < sizeof(RadPlanesHeader)) return false;
    if (op.rows > 0 || op.shape.size() < 2) return false;
    const RadPlanesEntry* codes = nullptr;
    const RadPlanesEntry* scales = nullptr;
    const void* code_data = rad_planes_find(content.data, "codes", &codes);
    const void* scale_data = rad_planes_find(content.data, "scale", &scales);
    if (code_data == nullptr || codes == nullptr) return false;
    const int64_t rows = codes->rows;
    const int64_t cols = codes->cols;
    if (rows <= 0 || cols <= 0 || rows * cols != numel(op.shape)) return false;
    if (scale_data != nullptr && (scales->rows <= 0 || scales->cols <= 0)) return false;

    const char* code_name = rad_dtype_name(codes->dtype);
    const int64_t code_es = rad_dtype_bytes(codes->dtype, 1);
    const char* scale_name = scale_data != nullptr ? rad_dtype_name(scales->dtype) : nullptr;
    const int64_t scale_es = scale_data != nullptr ? rad_dtype_bytes(scales->dtype, 1) : 0;
    if (code_es <= 0 || (scale_data != nullptr && scale_es <= 0)) return false;
    // The block is the dequantisation group: a scale per block[1] columns, block[0] rows.
    const int64_t block_r = scales != nullptr ? std::max<int64_t>(scales->block[0], 1) : rows;
    const int64_t block_c = scales != nullptr ? std::max<int64_t>(scales->block[1], 1) : cols;

    const auto* code_bytes = static_cast<const uint8_t*>(code_data);
    const auto* scale_bytes = static_cast<const uint8_t*>(scale_data);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            const double code =
                read_elem(code_name, code_bytes + r * codes->row_bytes + c * code_es, 0);
            double scale = 1.0;
            if (scale_bytes != nullptr) {
                const int64_t sr = r / block_r;
                const int64_t sc = c / block_c;
                if (sr >= scales->rows || sc >= scales->cols)
                    throw std::runtime_error("host step: a joined weight's scale plane is short");
                scale =
                    read_elem(scale_name, scale_bytes + sr * scales->row_bytes + sc * scale_es, 0);
            }
            (*out)[static_cast<size_t>(r * cols + c)] = code * scale;
        }
    }
    return true;
}

// One operand as the op saw it, dense over the operand's full declared shape. The tape keeps the
// slice the kernel saw, which for a ranged operand starts at row_lo; the walk's adjoints expect
// the full buffer with the untouched rows zero.
AdValue snapshot(const Operand& op, const OperandContent& content) {
    AdValue v;
    // The family adjoints read the record's dtype off the value too (an absent scale is spelled
    // "invalid"), so keep it with the numbers.
    v.dtype = op.dtype;
    // A strided or narrowed read -- the q norm and the gate read the two halves of the q|gate
    // projection, the gated-delta layer reads q|k|v out of one fused projection -- is packed over
    // its own shape: the adjoint that follows normalises the last dim the kernel normalised, and
    // the declared buffer's last dim is not that one. backward.cpp scatters the cotangent back.
    if (!op.view_shape.empty()) {
        v.shape = op.view_shape;
        const int64_t es = elem_size(op.dtype);
        if (es == 0)
            throw std::runtime_error("host step: cannot snapshot operand dtype '" + op.dtype + "'");
        int64_t n = 1;
        for (int64_t d : v.shape) n *= std::max<int64_t>(d, 0);
        v.value.assign(static_cast<size_t>(std::max<int64_t>(n, 0)), 0.0);
        if (!content.data || n == 0) return v;
        const auto* base = static_cast<const uint8_t*>(content.data);
        int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        const size_t rank = op.view_shape.size();
        // A kept copy of a strided view is the view's own values, packed in its own order (the
        // tape cannot copy a stride out of one span); a live buffer still has to be walked with
        // the strides. The size tells which one this is.
        const bool packed = content.bytes >= static_cast<size_t>(n) * static_cast<size_t>(es);
        for (int64_t e = 0; e < n; ++e) {
            int64_t off = packed ? e : op.view_off;
            if (!packed)
                for (size_t d = 0; d < rank; ++d) off += idx[d] * op.view_stride[d];
            v.value[static_cast<size_t>(e)] = read_elem(op.dtype, base + static_cast<int64_t>(es) * off, 0);
            for (size_t d = rank; d-- > 0;) {
                if (++idx[d] < op.view_shape[d]) break;
                idx[d] = 0;
            }
        }
        return v;
    }
    v.shape = op.shape;
    const int64_t total = numel(op.shape);
    v.value.assign(static_cast<size_t>(std::max<int64_t>(total, 0)), 0.0);
    if (!content.data || total == 0) return v;
    if (joined_dense(op, content, &v.value)) return v;
    if (content.bytes == 0) return v;

    const int64_t es = elem_size(op.dtype);
    if (es == 0)
        throw std::runtime_error("host step: cannot snapshot operand dtype '" + op.dtype + "'");
    const int64_t count = static_cast<int64_t>(content.bytes) / es;
    if (count <= 0) return v;

    int64_t cols = total;
    int64_t declared_rows = 1;
    if (!op.shape.empty()) {
        declared_rows = op.shape[0];
        cols = declared_rows > 0 ? total / declared_rows : 0;
    }
    for (int64_t i = 0; i < count; ++i) {
        int64_t at;
        if (op.rows == 0 || count == total) {
            at = i;
        } else {
            const int64_t r = i / (cols > 0 ? cols : 1);
            const int64_t c = cols > 0 ? i % cols : 0;
            at = (op.row_lo + r) * cols + c;
        }
        if (at < 0 || at >= total) continue;
        v.value[static_cast<size_t>(at)] = read_elem(op.dtype, content.data, i);
    }
    return v;
}

bool residual_shape(const InputCotangent& in, int64_t tokens, int64_t hidden) {
    // The head's ops see the engine's whole buffer, which is longer than the step's rows; its
    // first `tokens` rows are the step's. A block boundary is already the step's row count.
    return in.value.size() >= static_cast<size_t>(tokens * hidden) && !in.shape.empty() &&
           in.shape.back() == hidden && in.row_hi >= tokens;
}

std::vector<double> first_rows(const InputCotangent& in, int64_t tokens, int64_t hidden) {
    const size_t n = static_cast<size_t>(tokens * hidden);
    if (in.value.size() < n) return {};
    return std::vector<double>(in.value.begin(), in.value.begin() + static_cast<ptrdiff_t>(n));
}

const InputCotangent* pick_boundary(const std::vector<InputCotangent>& inputs, int64_t tokens,
                                    int64_t hidden) {
    const InputCotangent* best = nullptr;
    for (const InputCotangent& in : inputs) {
        if (!residual_shape(in, tokens, hidden)) continue;
        if (best == nullptr || in.value.size() > best->value.size()) best = &in;
    }
    return best;
}

// A record names a whole engine buffer, but this step's op only touched `rows` of them, starting
// at `row_lo`. The block walk is over the step's rows, not the engine's capacity: hand every
// adjoint the slice it actually saw, with the row window at zero, so read and write buffers of
// different capacities line up and the snapshots hold no inactive rows. Parameters and other
// whole-buffer operands have no row window and are left as they are.
void tighten_operand(Operand& o) {
    if (o.rows <= 0 || o.shape.empty() || o.rows > o.shape[0]) return;
    o.shape[0] = o.rows;
    o.row_lo = 0;
    // A view's first dim is the same row axis, so it narrows with the buffer.
    if (!o.view_shape.empty() && o.rows <= o.view_shape[0]) o.view_shape[0] = o.rows;
}

// The attention family's adjoint reads q as [rows, heads, head_dim] and takes head_dim from its
// last axis. The engine hands the kernel the flattened [rows, heads * head_dim] buffer, so the
// walk gets the view the family expects. The kv operand carries the same head_dim as a pair.
void annotate_attention(OpRecord& rec) {
    if (rec.op != "attn_paged" || rec.operands.empty()) return;
    Operand& q = rec.operands[0];
    if (q.shape.size() != 2 || !q.view_shape.empty()) return;
    const Operand* kv = nullptr;
    for (const Operand& o : rec.operands)
        if (o.shape.size() == 4) {
            kv = &o;
            break;
        }
    if (kv == nullptr) return;
    const int64_t head_dim = kv->shape[3] / 2;
    if (head_dim <= 0 || q.shape[1] % head_dim != 0) return;
    q.view_shape = {q.shape[0], q.shape[1] / head_dim, head_dim};
    q.view_stride = {q.shape[1], head_dim, 1};
    q.view_off = 0;
}

void tighten_rows(OpRecord& rec) {
    for (Operand& o : rec.operands) tighten_operand(o);
    annotate_attention(rec);
    // The engine applies an adapter to the activation it already wrote: the expand's last
    // operand is both the value it replaces and the value it adds its branch to, but the tape
    // does not mark that write in place. Say so, so the walk can pass the direct path on.
    if (rec.op == "lora_expand" || rec.op == "lora_expand_mm")
        for (Operand& o : rec.operands)
            if (o.written) o.in_place = true;
}

// The family's shrink reads its activation as [rows][K]. The engine can hand it an activation
// split by head ([rows][heads][width]); that one record takes the row that knows the reference
// step's shape, and everything else stays on the family's own row.
void mark_join_lora(OpRecord& rec) {
    if (rec.op != "lora_shrink" && rec.op != "lora_shrink_mm") return;
    for (const Operand& o : rec.operands)
        if (!o.written && o.shape.size() > 2) {
            rec.params.push_back({"r97join", int64_t{1}});
            return;
        }
}

// The objective reads the logits boundary in the batch's row space, L-1 rows a sequence. The
// head writes the step's row space, all L rows a sequence, after the sampler's rows: its logits
// buffer row for the batch's row r is `hidden_rows()[i] + 1`, because row i of hidden() is the
// head's input row that predicts that scored token. A seed has to move to that space before the
// walk, or it lands on the wrong token (or on a sampler row that predicts nothing).
std::vector<Seed> head_seeds(const ForwardStep& step, const Batch& batch, const std::vector<OpRecord>& head,
                             int64_t tokens) {
    int64_t rows = 0, vocab = 0;
    const std::vector<int64_t> lshape = step.shape("logits");
    if (lshape.size() != 2) throw std::runtime_error("host step: the logits boundary is not 2-D");
    rows = lshape[0];
    vocab = lshape[1];

    int64_t head_rows = 0;
    for (const OpRecord& rec : head)
        for (const Operand& o : rec.operands)
            if (o.written && o.buffer == "logits") head_rows = o.rows;

    // One entry a batch row, -1 where the row predicts nothing. hidden_rows() lists the scored
    // rows in the batch's order, so walking the scores in that order pairs them up.
    std::vector<int64_t> at(static_cast<size_t>(rows), -1);
    const std::vector<int64_t>& hidden_rows = step.hidden_rows();
    size_t scored = 0;
    int64_t off = 0;
    for (const TokenSeq& seq : batch) {
        const int64_t len = static_cast<int64_t>(seq.ids.size());
        for (int64_t p = 1; p < len; ++p) {
            if (p < static_cast<int64_t>(seq.score.size()) && !seq.score[static_cast<size_t>(p)]) continue;
            if (scored >= hidden_rows.size())
                throw std::runtime_error("host step: hidden_rows() is shorter than the scored rows");
            const int64_t batch_row = off + p - 1;
            if (batch_row < 0 || batch_row >= rows)
                throw std::runtime_error("host step: a scored row is outside the logits boundary");
            at[static_cast<size_t>(batch_row)] = hidden_rows[scored];
            ++scored;
        }
        off += len > 0 ? len - 1 : 0;
    }
    if (head_rows <= 0)
        throw std::runtime_error("host step: the head wrote no logits rows to seed");
    if (head_rows < tokens)
        throw std::runtime_error("host step: the head wrote fewer logits rows than tokens");

    std::vector<Seed> out;
    for (const Seed& seed : step.seeds()) {
        if (seed.buffer != "logits") {
            out.push_back(seed);
            continue;
        }
        const int64_t span = seed.row_hi - seed.row_lo;
        if (span <= 0) continue;
        if (static_cast<int64_t>(seed.cotangent.size()) != span * vocab)
            throw std::runtime_error("host step: a logits seed's cotangent does not match its rows");
        Seed moved;
        moved.buffer = seed.buffer;
        moved.row_lo = 0;
        moved.row_hi = head_rows;
        moved.cotangent.assign(static_cast<size_t>(head_rows * vocab), 0.0f);
        for (int64_t r = seed.row_lo; r < seed.row_hi; ++r) {
            const int64_t to = at[static_cast<size_t>(r)];
            if (to < 0) continue;
            if (to >= head_rows)
                throw std::runtime_error("host step: a seed maps past the head's logits rows");
            std::memcpy(moved.cotangent.data() + static_cast<size_t>(to * vocab),
                        seed.cotangent.data() + static_cast<size_t>((r - seed.row_lo) * vocab),
                        static_cast<size_t>(vocab) * sizeof(float));
        }
        out.push_back(std::move(moved));
    }
    return out;
}

// The observer names an unnamed operand after its position in the schema ("raw:2"), so two ops
// can give the same name to buffers of different shapes. A raw or absent operand is per-op by
// definition (a position table, a slot list, a token range), so give each its own name; nothing
// reads an unnamed operand by name.
OpRecord own_raws(OpRecord rec) {
    for (Operand& o : rec.operands)
        if (o.buffer.rfind("raw:", 0) == 0 || o.buffer.rfind("none:", 0) == 0)
            o.buffer += "@" + std::to_string(rec.position);
    return rec;
}

// A weight that several delta sites land in is one fused adapter on the engine's side: the
// shrink concatenates their A factors along the rank and the expand makes their B factors a
// block-diagonal matrix. The walk sees one gradient for the fused pair; split it back into the
// per-module pairs the delta's ParamSet names, using the order and shapes the sites gave.
struct SiteSlice {
    std::string module;
    int64_t rank = 0;
    int64_t in = 0;
    int64_t out = 0;
    int64_t rank_off = 0;
    int64_t out_off = 0;
};

using SiteGroups = std::map<std::string, std::vector<SiteSlice>>;

SiteGroups group_sites(const WireBuffers& wire) {
    SiteGroups groups;
    for (const WireBuffers::Site& site : wire.sites) {
        if (site.rank < 1) throw std::runtime_error("host step: a delta site has no rank");
        if (site.b.size() % static_cast<size_t>(site.rank) != 0)
            throw std::runtime_error("host step: delta site '" + site.module +
                                     "' has a B whose size is not its rank times its rows");
        if (site.a.size() % static_cast<size_t>(site.rank) != 0)
            throw std::runtime_error("host step: delta site '" + site.module +
                                     "' has an A whose size is not its rank times its columns");
        SiteSlice slice;
        slice.module = site.module;
        slice.rank = site.rank;
        slice.in = static_cast<int64_t>(site.a.size()) / site.rank;
        slice.out = static_cast<int64_t>(site.b.size()) / site.rank;
        std::vector<SiteSlice>& group = groups[site.weight_name];
        if (!group.empty()) {
            const SiteSlice& prev = group.back();
            slice.rank_off = prev.rank_off + prev.rank;
            slice.out_off = prev.out_off + prev.out;
        }
        group.push_back(std::move(slice));
    }
    return groups;
}

std::string fused_prefix(const std::string& weight) { return "@fused:" + weight; }

void add_grad(ParamSet* grads, const std::string& name, const std::vector<int64_t>& shape,
              const std::vector<float>& values) {
    for (Param& p : *grads) {
        if (p.name != name) continue;
        if (p.shape != shape || p.value.size() != values.size())
            throw std::runtime_error("host step: gradient '" + name + "' arrived twice with shapes");
        for (size_t i = 0; i < values.size(); ++i) p.value[i] += values[i];
        return;
    }
    Param p;
    p.name = name;
    p.shape = shape;
    p.value = values;
    grads->push_back(std::move(p));
}

const Param* find_grad(const ParamSet& grads, const std::string& name) {
    for (const Param& p : grads)
        if (p.name == name) return &p;
    return nullptr;
}

// The live forward may skip an adapter path (a zero B is the usual init), so the walk sees no
// record for that factor. Every delta parameter still has a gradient: zero where the path did not
// run. Fill those before the split, which needs both factors' shapes.
void fill_missing_grads(ParamSet* grads, const ParamSet& params) {
    if (!grads) return;
    for (const Param& p : params) {
        if (find_grad(*grads, p.name)) continue;
        Param zero;
        zero.name = p.name;
        zero.shape = p.shape;
        zero.value.assign(p.value.size(), 0.0f);
        grads->push_back(std::move(zero));
    }
}

// The caller compares a step's gradients to the parameters position by position, and the split
// replaces entries (erase + append) rather than writing them in place. Put the set back in the
// parameters' order once everything is in it. A name this step did not produce is left for the
// caller's own check to report.
void order_grads_like(ParamSet* grads, const ParamSet& params) {
    if (!grads || grads->size() != params.size()) return;
    ParamSet ordered;
    ordered.reserve(params.size());
    for (const Param& p : params) {
        auto it = std::find_if(grads->begin(), grads->end(),
                               [&](const Param& g) { return g.name == p.name; });
        if (it == grads->end()) return;
        ordered.push_back(std::move(*it));
        grads->erase(it);
    }
    *grads = std::move(ordered);
}

// A delta parameter's leaf under its module, found by the shape the split is about to write. The
// adjoint's own leaf names are the engine's spelling, which the join must not repeat: a second
// adapter kind names its factors differently and this still finds and writes them.
std::string param_leaf(const ParamSet& params, const std::string& module,
                       const std::vector<int64_t>& shape, const std::string& fallback) {
    const std::string head = module + ".";
    for (const Param& p : params)
        if (p.name.size() > head.size() && p.name.compare(0, head.size(), head) == 0 &&
            p.shape == shape)
            return p.name.substr(head.size());
    return fallback;
}

std::string leaf_of(const std::string& name) {
    const size_t dot = name.find_last_of('.');
    return dot == std::string::npos ? name : name.substr(dot + 1);
}

// The delta's prefix is what every gradient name starts with; a site's own module comes after it,
// so the split finds the walk's gradients under the same prefix and writes the delta's per-site
// ones. A delta need not use an empty prefix, and a step that names a gradient the delta never
// asked for leaves that parameter with no gradient at all.
//
// The walk's fused gradient carries whatever leaf its family's adjoint gave it; find it by the
// factor shapes the sites gave, not by an adapter's name, and write each module's factor under the
// leaf the delta's own ParamSet uses.
void split_adapter_grads(ParamSet* grads, const ParamSet& params, const SiteGroups& groups,
                         const std::string& grad_prefix) {
    if (!grads) return;
    ParamSet split;
    std::vector<std::string> replaced;
    for (const auto& [weight, slices] : groups) {
        if (slices.empty()) continue;
        const bool fused = slices.size() > 1;
        const std::string prefix =
            grad_prefix + (fused ? fused_prefix(weight) : slices.front().module);
        int64_t rank_total = 0, out_total = 0;
        const int64_t in = slices.front().in;
        for (const SiteSlice& slice : slices) {
            rank_total += slice.rank;
            out_total += slice.out;
            if (slice.in != in)
                throw std::runtime_error("host step: the delta sites of '" + weight +
                                         "' disagree about their input width");
        }
        if (rank_total <= 0 || in <= 0) continue;
        const Param* a = nullptr;
        const Param* b = nullptr;
        std::string a_name, b_name;
        for (const Param& p : *grads) {
            if (p.name.rfind(prefix + ".", 0) != 0) continue;
            const int64_t elems = static_cast<int64_t>(p.value.size());
            if (elems == rank_total * in) {
                a = &p;
                a_name = p.name;
            } else if (elems % rank_total == 0) {
                b = &p;
                b_name = p.name;
            }
        }
        // A live record can carry only one factor (the other path was skipped); the split then
        // has nothing to do for that weight and leaves the factor it does have in place.
        if (!a || !b) continue;
        const int64_t b_rows = static_cast<int64_t>(b->value.size()) / rank_total;
        // The kernel weight can have unadapted output rows after the adapter's own (a fused qkv
        // with a gate tail, say); the site's B is shorter than the weight and those rows are not
        // part of any module's gradient.
        if (b_rows < out_total)
            throw std::runtime_error("host step: the gradient of '" + weight +
                                     "' is shorter than its delta sites' B factors");
        for (const SiteSlice& slice : slices) {
            const std::vector<int64_t> a_shape{slice.rank, in};
            std::vector<float> av(static_cast<size_t>(slice.rank * in));
            for (int64_t j = 0; j < slice.rank; ++j)
                for (int64_t k = 0; k < in; ++k)
                    av[static_cast<size_t>(j * in + k)] =
                        a->value[static_cast<size_t>((slice.rank_off + j) * in + k)];
            add_grad(&split, grad_prefix + slice.module + "." +
                                  param_leaf(params, slice.module, a_shape, leaf_of(a_name)),
                     a_shape, av);

            const std::vector<int64_t> b_shape{slice.out, slice.rank};
            std::vector<float> bv(static_cast<size_t>(slice.out * slice.rank));
            for (int64_t n = 0; n < slice.out; ++n)
                for (int64_t j = 0; j < slice.rank; ++j)
                    bv[static_cast<size_t>(n * slice.rank + j)] =
                        b->value[static_cast<size_t>((slice.out_off + n) * rank_total +
                                                     slice.rank_off + j)];
            add_grad(&split, grad_prefix + slice.module + "." +
                                  param_leaf(params, slice.module, b_shape, leaf_of(b_name)),
                     b_shape, bv);
        }
        replaced.push_back(a_name);
        replaced.push_back(b_name);
    }
    if (split.empty()) return;
    // A parameter filled with zeros before the split may carry the same name the split writes;
    // replace it rather than keep both.
    for (const Param& p : split)
        grads->erase(std::remove_if(grads->begin(), grads->end(),
                                    [&](const Param& g) { return g.name == p.name; }),
                     grads->end());
    grads->erase(std::remove_if(grads->begin(), grads->end(),
                                [&](const Param& p) {
                                    return std::find(replaced.begin(), replaced.end(), p.name) !=
                                           replaced.end();
                                }),
                 grads->end());
    for (Param& p : split) grads->push_back(std::move(p));
}

std::string module_leaf(const std::string& module) {
    const size_t dot = module.find_last_of('.');
    return dot == std::string::npos ? module : module.substr(dot + 1);
}

// The family adjoints for the linear ops find their operands by the schema's names, while the
// live tape carries the engine's buffer names. They also take `M` from the schema's parameter,
// which the tape records as the engine's row capacity, not the rows this step used. Hand the
// adjoint the schema's names and the step's rows; the walk keeps the tape's record for versions,
// boundaries and snapshots.
std::string linear_schema_name(const OpRecord& rec, size_t j) {
    if (rec.op == "gemm_nt") return j == 0 ? "a" : j == 1 ? "b" : "";
    // gemm_nt_q's schema is a, a_scale, b, b_scale: the activation's quantisation sits between
    // the two matrix operands, unlike gemm_nt's plain a, b.
    if (rec.op == "gemm_nt_q")
        return j == 0 ? "a" : j == 1 ? "a_scale" : j == 2 ? "b" : j == 3 ? "b_scale" : "";
    if (rec.op == "logits_gemm") return j == 0 ? "x" : j == 1 ? "lm_head" : "";
    if (rec.op == "gather_rows") return j == 0 ? "x" : j == 1 ? "idx" : "";
    if (rec.op == "embed_lookup") return j == 0 ? "tokens" : j == 1 ? "wte" : "";
    return "";
}

void prepare_adjoint_record(const OpRecord& rec, OpRecord* out) {
    *out = rec;
    for (size_t j = 0; j < out->operands.size(); ++j) {
        std::string name = linear_schema_name(rec, j);
        if (!name.empty()) out->operands[j].buffer = std::move(name);
    }
    int64_t rows = 0;
    for (const Operand& o : rec.operands)
        if (o.written && !o.shape.empty()) {
            rows = o.shape[0];
            break;
        }
    if (rows <= 0)
        for (const Operand& o : rec.operands)
            if (!o.shape.empty()) {
                rows = o.shape[0];
                break;
            }
    if (rows > 0)
        for (auto& [key, value] : out->params)
            if (key == "M") {
                if (int64_t* m = std::get_if<int64_t>(&value)) *m = rows;
            }
    if (out->op == "gemm_nt_q") out->params.push_back({"r97host", int64_t{1}});
}

// The weight a lora record belongs to, from the adapter factor it names: lora.<weight>.a or .b.
// The record names the kernel weight (`blk.3.ffn_down.weight`), while the delta's site names the
// parameter without the container's `.weight` leaf (`blk.3.ffn_down`); drop that leaf so both
// name the same site.
std::string record_weight(const OpRecord& rec) {
    for (const Operand& o : rec.operands) {
        if (o.buffer.rfind("lora.", 0) != 0) continue;
        const size_t dot = o.buffer.find_last_of('.');
        if (dot == std::string::npos) continue;
        const std::string leaf = o.buffer.substr(dot + 1);
        if (leaf != "a" && leaf != "b") continue;
        std::string weight = o.buffer.substr(5, dot - 5);
        const std::string suffix = ".weight";
        if (weight.size() > suffix.size() &&
            weight.compare(weight.size() - suffix.size(), suffix.size(), suffix) == 0)
            weight.resize(weight.size() - suffix.size());
        return weight;
    }
    return {};
}

void set_error(std::string* err, const char* what) {
    if (err) *err = what;
}

}  // namespace

StepFn host_step(ForwardDriver& driver, const Model& model, const HostStepOptions& options,
                 HostStepReport* report) {
    const ModelMeta meta = read_model_meta(driver.config().model);
    if (meta.hidden <= 0)
        throw std::runtime_error("host step: the model's metadata gives no hidden size");

    return [&driver, model, options, report, meta](const Batch& batch, const ParamSet& params,
                                                   ParamSet* grads) -> StepResult {
        int64_t tokens = 0;
        for (const TokenSeq& s : batch) tokens += static_cast<int64_t>(s.ids.size());
        if (tokens <= 0) throw std::runtime_error("host step: the batch has no token rows");

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

        std::shared_ptr<BoundaryRecorder> recorder;
        driver.set_engine_hook([&driver, &recorder, &wire, live, targets, max_rank, tokens,
                               meta](rad::Engine& engine) {
            if (live) {
                LiveAdapterConfig cfg{"train", targets, std::max<int64_t>(max_rank, 1), 1.0};
                std::string err;
                std::unique_ptr<LiveAdapter> adapter = LiveAdapter::open(engine, cfg, &err);
                if (!adapter)
                    throw std::runtime_error("host step: cannot open the live adapter: " + err);
                if (adapter->update(wire, &err) != 0)
                    throw std::runtime_error("host step: cannot fill the live adapter: " + err);
            }
            recorder = std::make_shared<BoundaryRecorder>(
                driver.tape(), tokens, meta.hidden);
        });

        const StepResult result = driver.step(model)(batch, params, nullptr);
        if (!recorder)
            throw std::runtime_error("host step: the engine hook did not install the recorder");
        recorder->detach();
        const ResidualStore store = recorder->store();

        const Tape& tape = driver.last().tape();
        const std::vector<OpRecord>& ops = tape.ops();
        const std::vector<Tape::Block>& blocks = tape.blocks();
        if (blocks.empty())
            throw std::runtime_error("host step: the step has no block brackets to replay");
        if (store.boundaries() != static_cast<int64_t>(blocks.size()) + 1)
            throw std::runtime_error("host step: the boundary store does not match the blocks");

        // The head and the final norm are the ops after the last block. They are the step's last
        // ops, so their operand contents are still live and can be copied now. Everything inside a
        // block is copied by its replay instead, one layer at a time.
        std::vector<OpRecord> head_records(ops.begin() + static_cast<std::ptrdiff_t>(blocks.back().end),
                                           ops.end());
        for (OpRecord& rec : head_records) {
            rec = own_raws(std::move(rec));
            tighten_rows(rec);
            if (options.reference_head_split) mark_join_lora(rec);
        }
        StoredSnapshots head_snaps;
        for (const OpRecord& rec : head_records) {
            const std::vector<OperandContent>& content = tape.contents(static_cast<size_t>(rec.position));
            for (size_t j = 0; j < rec.operands.size(); ++j)
                head_snaps.set(rec.position, j, snapshot(rec.operands[j], content[j]));
        }

        BackwardOptions bo;
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
                // A fused pool is one record on the engine, so its gradient comes back once; the
                // split below puts each module's factor under its own name.
                return grad_prefix + (g->second.size() > 1 ? fused_prefix(weight)
                                                           : g->second.front().module);
            };
        }
        // A fused weight's slots are its delta sites in order; name each slot's factor after the
        // module it belongs to, so an adapter row that knows how to read a head-split activation
        // writes one gradient per module rather than one fused one.
        bo.slot_prefix = [groups, grad_prefix](const OpRecord& rec, int64_t slot) -> std::string {
            const std::string weight = record_weight(rec);
            const auto g = groups.find(weight);
            if (g == groups.end() || slot < 0 ||
                slot >= static_cast<int64_t>(g->second.size()))
                return grad_prefix;
            return grad_prefix + g->second[static_cast<size_t>(slot)].module;
        };
        // The live tape names the engine's buffers; a family's adjoint finds a few operands by the
        // schema's own names. The walk keeps the tape names for versions, boundaries and snapshots.
        bo.prepare_record = prepare_adjoint_record;

        HostStepReport measured;
        measured.loss = result.loss;
        measured.scored = result.scored;
        measured.tokens = tokens;
        measured.hidden = meta.hidden;
        measured.layers = static_cast<int64_t>(blocks.size());
        measured.residual_store_bytes = store.bytes();
        if (options.keep_boundaries) {
            measured.boundary_name.assign(static_cast<size_t>(measured.layers + 1), {});
            measured.boundary_value.assign(static_cast<size_t>(measured.layers + 1), {});
            measured.boundary_cotangent.assign(static_cast<size_t>(measured.layers + 1), {});
        }

        BoundaryNames names = BoundaryNames::from(driver.last());
        int64_t head_logits_rows = 0;
        for (const OpRecord& rec : head_records)
            for (const Operand& o : rec.operands)
                if (o.written && o.buffer == "logits") head_logits_rows = o.rows;
        if (head_logits_rows <= 0)
            throw std::runtime_error("host step: the head wrote no logits rows");
        // The objective's window is L-1 rows a sequence; the head's buffer is all L after the
        // sampler's rows, and the moved seeds live in that space.
        names.add("logits", 0, head_logits_rows);
        const std::vector<Seed> seeds = head_seeds(driver.last(), batch, head_records, tokens);
        const BackwardResult head = backward_block(head_records, head_snaps, names, seeds, {}, bo);
        // What the join held while the head walked: its whole store, which is held up front, plus
        // the copies of the one op being adjointed. The walk's own figure counts the latter only.
        measured.peak_snapshot_store_bytes = head_snaps.bytes();
        measured.peak_snapshot_op_bytes = head.report.peak_snapshot_bytes;
        measured.peak_snapshot_bytes =
            measured.peak_snapshot_store_bytes + measured.peak_snapshot_op_bytes;
        measured.all_snapshot_store_bytes = measured.peak_snapshot_store_bytes;
        measured.peak_cotangent_bytes = head.report.peak_cotangent_bytes;
        // The head's snapshots are spent once its walk has run and the layer walks build their own,
        // so free them: the peak stays one chunk's worth rather than the head's store added to a
        // block's.
        head_snaps = StoredSnapshots{};
        // The head's input cotangents stay alive with `head` for the whole layer loop, so they are
        // part of what the join holds during every block walk.
        int64_t head_input_bytes = 0;
        for (const InputCotangent& in : head.inputs) head_input_bytes += value_bytes(in.value);

        const InputCotangent* above = pick_boundary(head.inputs, tokens, meta.hidden);
        if (!above) {
            std::string dbg = "host step: the head walk left no cotangent on the final hidden; blocks=" +
                              std::to_string(blocks.size()) + " head_records=" +
                              std::to_string(head_records.size()) + " inputs=" +
                              std::to_string(head.inputs.size());
            for (const auto& in : head.inputs) {
                dbg += " [";
                dbg += in.buffer;
                dbg += " rows=" + std::to_string(in.row_lo) + ".." + std::to_string(in.row_hi) + " shape=";
                for (int64_t d : in.shape) dbg += std::to_string(d) + ",";
                dbg += "]";
            }
            for (const auto& rec : head_records) dbg += " op=" + rec.op;
            throw std::runtime_error(dbg);
        }

        auto store_boundary = [&](int64_t boundary, const InputCotangent* in) {
            if (!options.keep_boundaries || !in) return;
            measured.boundary_name[static_cast<size_t>(boundary)] = in->buffer;
            measured.boundary_cotangent[static_cast<size_t>(boundary)] = in->value;
            const uint16_t* value = store.at(boundary);
            if (!value) return;
            const int64_t row = meta.hidden;
            std::vector<double>& out = measured.boundary_value[static_cast<size_t>(boundary)];
            out.resize(static_cast<size_t>(tokens * row));
            for (int64_t i = 0; i < tokens * row; ++i) {
                const uint32_t u = static_cast<uint32_t>(value[i]) << 16;
                float f;
                std::memcpy(&f, &u, sizeof f);
                out[static_cast<size_t>(i)] = f;
            }
        };

        store_boundary(measured.layers, above);
        IncomingCotangent incoming{above->buffer, 0, tokens,
                                   first_rows(*above, tokens, meta.hidden)};

        LayerReplay replay(*driver.engine(), driver.tape(), store);
        for (int64_t layer = measured.layers - 1; layer >= 0; --layer) {
            StoredSnapshots snaps;
            // The post-issue observer runs after the op, so an operand an op replaces in place
            // holds the value the op wrote. A read of that same buffer is the value before the
            // op: the last write it saw, which is the previous op's post-issue content. Keep the
            // latter per buffer and give the in-place read that, or the adjoint differentiates
            // the op at its output instead of its input (the gated norm's gdn_o, above all).
            std::map<std::string, AdValue> before;
            replay.replay(layer, [&snaps, &before, &options](const OpRecord& op,
                                                   const std::vector<OperandContent>& content) {
                // The adjoint sees the step's rows and, for attention, the q view its family
                // reads; the snapshots must have the same shapes the walk will ask for.
                OpRecord shaped = op;
                tighten_rows(shaped);
                if (options.reference_head_split) mark_join_lora(shaped);
                std::vector<AdValue> now(shaped.operands.size());
                for (size_t j = 0; j < shaped.operands.size(); ++j)
                    now[j] = snapshot(shaped.operands[j], content[j]);
                for (size_t j = 0; j < shaped.operands.size(); ++j) {
                    const Operand& read = shaped.operands[j];
                    if (read.written) continue;
                    bool in_place = false;
                    for (size_t k = 0; k < shaped.operands.size(); ++k) {
                        const Operand& write = shaped.operands[k];
                        if (write.written && write.buffer == read.buffer) in_place = true;
                    }
                    if (!in_place) continue;
                    const auto prev = before.find(read.buffer);
                    if (prev != before.end() && prev->second.shape == now[j].shape) {
                        now[j] = prev->second;
                    } else if (read.buffer.rfind("kv:", 0) == 0) {
                        // The engine cleared the recurrent state before the replay, so a
                        // carried state read that no op in the block wrote yet enters zero.
                        now[j].value.assign(now[j].value.size(), 0.0);
                    }
                }
                for (size_t j = 0; j < shaped.operands.size(); ++j) {
                    const Operand& operand = shaped.operands[j];
                    if (operand.written && operand.buffer.rfind("kv:", 0) == 0 &&
                        before.find(operand.buffer) == before.end()) {
                        // A carried state this block has not written yet enters cold; the op's
                        // adjoint wants the state before the op, not the one it leaves.
                        now[j].value.assign(now[j].value.size(), 0.0);
                    }
                    snaps.set(op.position, j, now[j]);
                    if (operand.written && operand.view_shape.empty() && operand.rows > 0 &&
                        operand.rows == operand.shape[0]) {
                        before[operand.buffer] = std::move(now[j]);
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
            const BackwardResult walked = backward_block(records, snaps, block_names, {},
                                                         {incoming}, bo);
            // The join holds the whole block's snapshots while the walk copies one op out of them,
            // so what it holds at once is the store plus that one op, not the walk's figure alone.
            // `before` holds the last write of every buffer an op replaced in place, a second copy
            // of what the store already keeps, and is part of what the replay holds.
            int64_t before_bytes = 0;
            for (const auto& entry : before) before_bytes += value_bytes(entry.second.value);
            const int64_t store_bytes = snaps.bytes() + before_bytes;
            const int64_t op_bytes = walked.report.peak_snapshot_bytes;
            if (store_bytes + op_bytes > measured.peak_snapshot_bytes) {
                measured.peak_snapshot_bytes = store_bytes + op_bytes;
                measured.peak_snapshot_store_bytes = store_bytes;
                measured.peak_snapshot_op_bytes = op_bytes;
            }
            measured.all_snapshot_store_bytes += store_bytes;
            measured.peak_cotangent_bytes = std::max(
                measured.peak_cotangent_bytes,
                head_input_bytes + value_bytes(incoming.value) +
                    walked.report.peak_cotangent_bytes);

            const InputCotangent* below = pick_boundary(walked.inputs, tokens, meta.hidden);
            if (!below)
                throw std::runtime_error("host step: layer " + std::to_string(layer) +
                                         " left no cotangent on its input boundary");
            store_boundary(layer, below);
            incoming = IncomingCotangent{below->buffer, 0, tokens,
                                         first_rows(*below, tokens, meta.hidden)};
        }

        // The split runs first: a fused weight's engine gradient is named after the container's
        // weight ("@fused:..."), so filling the missing ones before splitting would add a zero
        // under every module name the split then adds the real gradient for.
        split_adapter_grads(grads, params, groups, grad_prefix);
        fill_missing_grads(grads, params);
        order_grads_like(grads, params);
        // The report's own boundary copies grow with the number of layers; name them rather than
        // fold them into the peaks, which describe the join's working set.
        for (const std::vector<double>& v : measured.boundary_value)
            measured.kept_boundary_bytes += value_bytes(v);
        for (const std::vector<double>& v : measured.boundary_cotangent)
            measured.kept_boundary_bytes += value_bytes(v);
        if (report) *report = std::move(measured);
        return result;
    };
}

}  // namespace r97

#endif  // RAD_ISSUE_OBSERVER && RAD_REPLAY_BLOCK
