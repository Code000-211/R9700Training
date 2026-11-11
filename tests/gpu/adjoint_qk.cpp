// tests/gpu/adjoint_qk.cpp -- the device adjoint of the gated attention prologue.
//
// `qk_norm_rope_gate` prepares a block's queries and keys and hands the gate out: rmsnorm on the
// query lanes, rmsnorm on the key lanes, partial NeoX rope over both, and the gate lanes copied
// through. `qk_norm_rope_gate_kv_store` is the same op when the step also writes its turned keys
// and its values into the paged pool. The card's reference is the composition of the host
// adjoints of those parts -- rmsnorm, rope, kv_store, and sigmoid then mul for the gate
// (docs/adjoints/qk.md) -- so this file holds the kernel to that composition on the same random
// inputs, at the tiny fixture's geometry and at the real model's.
//
// The gate's cotangent is the one the chain above the op makes: the gate lanes go through a
// sigmoid, whose output the attention multiplies, so the cotangent the fused op sees at its gate
// output is the sigmoid adjoint's, given the mul adjoint's cotangent of the sigmoid's output. Both
// of those are the host's own adjoints, so the gate is checked as a pass-through of a cotangent
// that a real composition produced and not as a copy of a random tensor.
//
// The reference is not taken on trust: a centred step of the forward along a random direction of
// the reads has to move the writes by as much as the reference's cotangent says it does, so a
// reference that is itself wrong fails before any comparison does.
//
// The op has no host twin or adjoint row of its own (its host form is the unfused chain), so the
// twin that gives a case its forward values is registered here: the rmsnorm twin from the
// registry, rope_forward_f64 and kv_store_forward_f64 from src/adjoint/host/qk.h.
#include "../../src/adjoint/host/qk.h"

#include "../support/device_check.h"
#include "../support/gpu_guard.h"
#include "../support/test.h"

#include "r97/adjoint.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using r97::AdjointCall;
using r97::AdjointFn;
using r97::AdValue;
using r97::OpRecord;
using r97::Operand;
using r97::Side;
using r97::TwinFn;
using r97::device_check::Case;
using r97::device_check::HostRef;
using r97::device_check::Report;

namespace device_check = r97::device_check;

using r97::adjoint::qk::KvStoreShape;
using r97::adjoint::qk::RopeShape;
using r97::adjoint::qk::RopeTable;

// Qwen3's ladder and the eps the card's rmsnorm defaults to; the table is data, so a case only has
// to agree with itself, but the real shapes deserve the real numbers.
constexpr double kTheta = 10000.0;
constexpr double kEps = 1e-6;
constexpr double kWadd = 0.0;

/* ------------------------------------------------------------------ small helpers */

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return shape.empty() ? 0 : n;
}

std::string num_text(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
}

double num_param(const OpRecord& rec, const char* name, double fallback) {
    for (const auto& p : rec.params) {
        if (p.first != name) continue;
        if (const int64_t* i = std::get_if<int64_t>(&p.second)) return (double)*i;
        const std::string& s = std::get<std::string>(p.second);
        char* end = nullptr;
        const double v = std::strtod(s.c_str(), &end);
        return (end && end != s.c_str() && *end == '\0') ? v : fallback;
    }
    return fallback;
}

AdValue tensor(const std::string& dtype, const std::vector<int64_t>& shape,
               std::vector<double> values) {
    AdValue v;
    v.dtype = dtype;
    v.shape = shape;
    if (values.empty()) values.assign((size_t)numel(shape), 0.0);
    v.value = std::move(values);
    return v;
}

AdValue zeros(const std::string& dtype, const std::vector<int64_t>& shape) {
    return tensor(dtype, shape, {});
}

// A value as the card holds it: bf16 is a truncated-and-rounded f32, and a table the record calls
// bf16 reaches the op that way too.
double round_bf16(double x) {
    float f = (float)x;
    unsigned int u;
    std::memcpy(&u, &f, sizeof u);
    if ((u & 0x7fffffffu) <= 0x7f800000u) {
        const unsigned int lsb = (u >> 16) & 1u;
        u += 0x7fffu + lsb;
        u &= 0xffff0000u;
    }
    std::memcpy(&f, &u, sizeof f);
    return (double)f;
}

// A random tensor as the card holds it. A bf16 gradient is a bf16 value, and the norm's adjoint
// subtracts two terms that can nearly cancel, so a value the tape could not hold would cost more
// than the tolerance in the difference.
AdValue bf16_value(std::vector<int64_t> shape, uint64_t* seed) {
    AdValue v = device_check::random_value("bf16", std::move(shape), seed);
    for (double& x : v.value) x = round_bf16(x);
    return v;
}

// A norm gain: near one, so a case looks like the block it models and the shared term of the
// adjoint is not dominated by a lane whose weight happens to be nothing.
void make_gain(std::vector<double>* w) {
    for (double& v : *w) v = 1.0 + 0.25 * v;
}

/* ------------------------------------------------------------------ one case's geometry */

struct Geometry {
    std::vector<int64_t> lengths;  // tokens per sequence; the batch may be ragged
    int64_t n_head = 0;
    int64_t n_head_kv = 0;
    int64_t head_dim = 0;
    int64_t rot = 0;  // the rotated lanes, an even number not above head_dim
    int64_t block_size = 16;
    bool kv_store = false;
    bool cs_f32 = true;  // the rope table's dtype, as the record states it
};

// What a case's sequences mean for positions and for the pool: each sequence counts from zero, and
// the pool gives each sequence its own rows, as the block table of a real step does.
struct Layout {
    int64_t tokens = 0;
    int32_t top = 0;  // the largest position, so the table has a row for each
    std::vector<int32_t> positions;
    std::vector<int32_t> slot;
    int64_t n_blocks = 0;
};

Layout layout_of(const Geometry& g) {
    Layout out;
    int64_t widest = 0;
    for (int64_t len : g.lengths) {
        out.tokens += len;
        widest = std::max(widest, len);
    }
    const int64_t stride = (widest + g.block_size - 1) / g.block_size;
    std::vector<int32_t> block_table((size_t)(stride * (int64_t)g.lengths.size()), 0);
    int64_t base = 0;
    for (size_t s = 0; s < g.lengths.size(); ++s) {
        const int64_t len = g.lengths[s];
        for (int64_t p = 0; p < len; ++p) {
            if (p % g.block_size == 0) {
                block_table[(size_t)((int64_t)s * stride + p / g.block_size)] = (int32_t)(base++);
            }
            out.positions.push_back((int32_t)p);
            out.slot.push_back((int32_t)r97::adjoint::qk::kv_slot(block_table.data(), stride,
                                                                 g.block_size, (int32_t)s,
                                                                 (int32_t)p));
        }
    }
    out.top = 0;
    for (int32_t p : out.positions) out.top = std::max(out.top, p);
    out.n_blocks = base;
    return out;
}

/* ------------------------------------------------------------------ the host's parts */

// The host's own rmsnorm row: the twin that produces the normalised lanes, and the adjoint the
// fused op's q or k half must agree with.
struct NormParams {
    double eps = kEps;
    double wadd = kWadd;
};

OpRecord norm_record(const NormParams& np, int64_t rows, int64_t width) {
    OpRecord rec;
    rec.op = "rmsnorm";
    rec.params = {{"eps", num_text(np.eps)}, {"wadd", num_text(np.wadd)}};
    rec.operands = {{"x", "bf16", {rows, width}, 0, rows, false},
                    {"w", "f32", {width}, 0, 1, false},
                    {"y", "bf16", {rows, width}, 0, rows, true}};
    return rec;
}

std::vector<double> norm_forward(const NormParams& np, const std::vector<double>& x,
                                 const std::vector<double>& w, int64_t rows, int64_t width) {
    const OpRecord rec = norm_record(np, rows, width);
    TwinFn twin = r97::twin_find("rmsnorm", rec.params, nullptr);
    if (!twin) throw std::runtime_error("no host twin for 'rmsnorm'");
    std::vector<AdValue> in{tensor("bf16", {rows, width}, x), tensor("f32", {width}, w)};
    std::vector<AdValue> out;
    twin(rec, in, &out);
    return out.at(0).value;
}

std::vector<double> norm_adjoint(const NormParams& np, const std::vector<double>& x,
                                 const std::vector<double>& w, const std::vector<double>& dy,
                                 int64_t rows, int64_t width) {
    const OpRecord rec = norm_record(np, rows, width);
    AdjointFn fn = r97::adjoint_find("rmsnorm", Side::host, rec.params, nullptr);
    if (!fn) throw std::runtime_error("no host adjoint for 'rmsnorm'");
    std::vector<AdValue> fwd{tensor("bf16", {rows, width}, x), tensor("f32", {width}, w),
                             zeros("bf16", {rows, width})};
    std::vector<AdValue> cot{tensor("bf16", {rows, width}, dy)};
    std::vector<AdValue> dc{zeros("bf16", {rows, width})};
    AdjointCall call{rec, fwd, cot, &dc, nullptr, ""};
    fn(call);
    return dc.at(0).value;
}

// sigmoid x -> y and mul a, b -> y, through the host's registered rows: the gate's own chain.
std::vector<double> sigmoid_forward(const std::vector<double>& x, int64_t rows, int64_t width) {
    OpRecord rec;
    rec.op = "sigmoid";
    rec.operands = {{"x", "bf16", {rows, width}, 0, rows, false},
                    {"y", "bf16", {rows, width}, 0, rows, true}};
    TwinFn twin = r97::twin_find("sigmoid", rec.params, nullptr);
    if (!twin) throw std::runtime_error("no host twin for 'sigmoid'");
    std::vector<AdValue> in{tensor("bf16", {rows, width}, x)};
    std::vector<AdValue> out;
    twin(rec, in, &out);
    return out.at(0).value;
}

std::vector<double> sigmoid_adjoint(const std::vector<double>& y, const std::vector<double>& dy,
                                    int64_t rows, int64_t width) {
    OpRecord rec;
    rec.op = "sigmoid";
    rec.operands = {{"x", "bf16", {rows, width}, 0, rows, false},
                    {"y", "bf16", {rows, width}, 0, rows, true}};
    AdjointFn fn = r97::adjoint_find("sigmoid", Side::host, rec.params, nullptr);
    if (!fn) throw std::runtime_error("no host adjoint for 'sigmoid'");
    std::vector<AdValue> fwd{tensor("bf16", {rows, width}, y), tensor("bf16", {rows, width}, y)};
    std::vector<AdValue> cot{tensor("bf16", {rows, width}, dy)};
    std::vector<AdValue> dc{zeros("bf16", {rows, width}), zeros("bf16", {rows, width})};
    AdjointCall call{rec, fwd, cot, &dc, nullptr, ""};
    fn(call);
    return dc.at(0).value;
}

std::vector<double> mul_adjoint_b(const std::vector<double>& dy, const std::vector<double>& a,
                                  const std::vector<double>& b, int64_t rows, int64_t width) {
    OpRecord rec;
    rec.op = "mul";
    rec.params = {{"n", width}};
    rec.operands = {{"a", "bf16", {rows, width}, 0, rows, false},
                    {"b", "bf16", {rows, width}, 0, rows, false},
                    {"y", "bf16", {rows, width}, 0, rows, true}};
    AdjointFn fn = r97::adjoint_find("mul", Side::host, rec.params, nullptr);
    if (!fn) throw std::runtime_error("no host adjoint for 'mul'");
    std::vector<AdValue> fwd{tensor("bf16", {rows, width}, a), tensor("bf16", {rows, width}, b),
                             zeros("bf16", {rows, width})};
    std::vector<AdValue> cot{tensor("bf16", {rows, width}, dy)};
    std::vector<AdValue> dc{zeros("bf16", {rows, width}), zeros("bf16", {rows, width}),
                            zeros("bf16", {rows, width})};
    AdjointCall call{rec, fwd, cot, &dc, nullptr, ""};
    fn(call);
    return dc.at(1).value;
}

// The gate lanes of a projection row, one row per (token, query head).
std::vector<double> gate_lanes(const std::vector<double>& q_gate, int64_t tokens, int64_t n_head,
                               int64_t head_dim) {
    const int64_t qg_len = n_head * 2 * head_dim;
    std::vector<double> out((size_t)(tokens * n_head * head_dim));
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t h = 0; h < n_head; ++h) {
            const int64_t src = t * qg_len + h * 2 * head_dim + head_dim;
            const int64_t dst = (t * n_head + h) * head_dim;
            for (int64_t i = 0; i < head_dim; ++i)
                out[(size_t)(dst + i)] = q_gate[(size_t)(src + i)];
        }
    }
    return out;
}

/* ------------------------------------------------------------------ a case */

// The operand order this family's record carries: the reads in schema order, then the writes. The
// cotangents arrive in the writes' order, so the store's pool comes first when the op stores.
struct Slots {
    size_t v = 0, slot_map = 0, pool = 0, q = 0, k = 0, gate = 0;
    size_t cot_pool = 0, cot_q = 0, cot_k = 0, cot_gate = 0;
};

Slots slots_of(const bool kv) {
    Slots s;
    if (kv) {
        s.v = 6, s.slot_map = 7, s.pool = 8, s.q = 9, s.k = 10, s.gate = 11;
        s.cot_pool = 0, s.cot_q = 1, s.cot_k = 2, s.cot_gate = 3;
    } else {
        s.q = 6, s.k = 7, s.gate = 8;
        s.cot_q = 0, s.cot_k = 1, s.cot_gate = 2;
    }
    return s;
}

Case qk_case(const Geometry& g, std::string label, uint64_t seed) {
    const Layout layout = layout_of(g);
    const int64_t tokens = layout.tokens;
    const int64_t hd = g.head_dim;
    const int64_t nq = g.n_head;
    const int64_t nkv = g.n_head_kv;
    const int64_t qg_len = nq * 2 * hd;
    const int64_t q_len = nq * hd;
    const int64_t k_len = nkv * hd;
    const Slots s = slots_of(g.kv_store);
    if (tokens <= 0 || hd <= 0 || nq <= 0 || nkv <= 0) throw std::runtime_error("qk case: empty");

    const RopeTable table =
        r97::adjoint::qk::rope_table_f64((int64_t)layout.top + 1, g.rot, kTheta);

    Case c;
    c.label = std::move(label);
    c.rec.op = g.kv_store ? "qk_norm_rope_gate_kv_store" : "qk_norm_rope_gate";
    c.rec.params = {{"M", tokens},
                    {"head_dim", hd},
                    {"n_head", nq},
                    {"n_head_kv", nkv},
                    {"rot", g.rot},
                    {"q_dtype", std::string("bf16")},
                    {"eps", num_text(kEps)},
                    {"wadd", num_text(kWadd)},
                    {"cs_f32", (int64_t)(g.cs_f32 ? 1 : 0)},
                    {"pos_i64", (int64_t)0}};
    if (g.kv_store) {
        c.rec.params.push_back({"block_size", g.block_size});
        c.rec.params.push_back({"kv_dtype", std::string("bf16")});
        c.rec.params.push_back({"k_scale", num_text(1.0)});
        c.rec.params.push_back({"v_scale", num_text(1.0)});
    }

    auto operand = [&](const char* buffer, const char* dtype, std::vector<int64_t> shape,
                       bool written) {
        Operand op{buffer, dtype, std::move(shape), 0, 0, written};
        op.rows = op.shape.empty() ? 0 : op.shape[0];
        c.rec.operands.push_back(std::move(op));
    };
    operand("attn_qg", "bf16", {tokens, qg_len}, false);
    operand("attn_k", "bf16", {tokens, k_len}, false);
    operand("cos_sin", g.cs_f32 ? "f32" : "bf16", {(int64_t)layout.top + 1, g.rot}, false);
    operand("positions", "i32", {tokens}, false);
    operand("q_norm", "f32", {hd}, false);
    operand("k_norm", "f32", {hd}, false);
    if (g.kv_store) {
        operand("attn_v", "bf16", {tokens, k_len}, false);
        operand("slot_mapping", "i32", {tokens}, false);
        operand("kv_cache", "bf16", {layout.n_blocks, nkv, g.block_size, 2 * hd}, true);
    }
    operand("q_out", "bf16", {tokens, q_len}, true);
    operand("k_out", "bf16", {tokens, k_len}, true);
    operand("gate_out", "bf16", {tokens, q_len}, true);

    // The table the step's positions index; a bf16 table reaches the op rounded, as on the card.
    std::vector<double> cs = table.cos_sin;
    if (!g.cs_f32) {
        for (double& v : cs) v = round_bf16(v);
    }
    std::vector<double> pos(layout.positions.begin(), layout.positions.end());

    AdValue q_gate = bf16_value({tokens, qg_len}, &seed);
    AdValue w_q = r97::device_check::random_value("f32", {hd}, &seed);
    AdValue w_k = r97::device_check::random_value("f32", {hd}, &seed);
    make_gain(&w_q.value);
    make_gain(&w_k.value);
    c.in = {q_gate,
            bf16_value({tokens, k_len}, &seed),
            tensor(g.cs_f32 ? "f32" : "bf16", {(int64_t)layout.top + 1, g.rot}, cs),
            tensor("i32", {tokens}, pos),
            w_q,
            w_k};
    if (g.kv_store) {
        c.in.push_back(bf16_value({tokens, k_len}, &seed));
        c.in.push_back(tensor("i32", {tokens},
                              std::vector<double>(layout.slot.begin(), layout.slot.end())));
    }

    // The cotangents of the writes, in the writes' order. The gate's is the one the chain above the
    // op makes: sigmoid, then the mul with the attention output.
    const std::vector<double> gate = gate_lanes(q_gate.value, tokens, nq, hd);
    const std::vector<double> s_out = sigmoid_forward(gate, tokens * nq, hd);
    const std::vector<double> o_out = bf16_value({tokens * nq, hd}, &seed).value;
    const std::vector<double> dy_out = bf16_value({tokens * nq, hd}, &seed).value;
    const std::vector<double> d_s = mul_adjoint_b(dy_out, o_out, s_out, tokens * nq, hd);
    const std::vector<double> d_gate = sigmoid_adjoint(s_out, d_s, tokens * nq, hd);

    if (g.kv_store) {
        c.cot.push_back(bf16_value({layout.n_blocks, nkv, g.block_size, 2 * hd}, &seed));
    }
    c.cot.push_back(bf16_value({tokens, q_len}, &seed));
    c.cot.push_back(bf16_value({tokens, k_len}, &seed));
    // The gate's cotangent is the chain's, rounded to what the tape holds.
    std::vector<double> d_gate_bf16 = d_gate;
    for (double& x : d_gate_bf16) x = round_bf16(x);
    c.cot.push_back(tensor("bf16", {tokens, q_len}, d_gate_bf16));
    return c;
}

/* ------------------------------------------------------------------ the reference */

// The composition, in f64, of the host's adjoints: rope's and kv_store's from the qk header, and
// rmsnorm's through its registered row.
HostRef qk_reference(const Geometry& g) {
    return [g](AdjointCall& call) {
        const int64_t tokens = (int64_t)call.fwd[0].shape[0];
        const int64_t hd = g.head_dim;
        const int64_t nq = g.n_head;
        const int64_t nkv = g.n_head_kv;
        const int64_t q_len = nq * hd;
        const int64_t k_len = nkv * hd;
        const int64_t qg_len = nq * 2 * hd;
        const Slots s = slots_of(g.kv_store);
        if (call.fwd.size() < s.gate + 1 || call.cot.size() < s.cot_gate + 1) return;
        if (!call.dc || call.dc->size() < s.gate + 1) return;

        const NormParams np{num_param(call.rec, "eps", kEps), num_param(call.rec, "wadd", kWadd)};
        RopeTable table;
        table.rows = call.fwd[2].shape[0];
        table.rotary_dim = g.rot;
        table.cos_sin = call.fwd[2].value;
        std::vector<int32_t> positions((size_t)tokens);
        for (int64_t t = 0; t < tokens; ++t)
            positions[(size_t)t] = (int32_t)call.fwd[3].value[(size_t)t];
        const RopeShape q_rope{tokens, nq, 0, hd, g.rot};
        const RopeShape k_rope{tokens, nkv, 0, hd, g.rot};

        // The queries: rope's adjoint turns the cotangent back, then the norm's row takes it to the
        // projection's lanes.
        std::vector<double> q_lanes((size_t)(tokens * q_len));
        std::vector<double> gate((size_t)(tokens * q_len));
        for (int64_t t = 0; t < tokens; ++t) {
            for (int64_t h = 0; h < nq; ++h) {
                const int64_t src = t * qg_len + h * 2 * hd;
                const int64_t dst = t * q_len + h * hd;
                for (int64_t i = 0; i < hd; ++i) {
                    q_lanes[(size_t)(dst + i)] = call.fwd[0].value[(size_t)(src + i)];
                    gate[(size_t)(dst + i)] = call.fwd[0].value[(size_t)(src + hd + i)];
                }
            }
        }
        const std::vector<double>& d_q = call.cot[s.cot_q].value;
        std::vector<double> d_turned((size_t)(tokens * q_len));
        r97::adjoint::qk::rope_adjoint_f64(d_q.data(), positions.data(), q_rope, table,
                                           d_turned.data());
        const std::vector<double> d_q_lanes = norm_adjoint(np, q_lanes, call.fwd[4].value, d_turned,
                                                           tokens * nq, hd);
        for (int64_t t = 0; t < tokens; ++t) {
            for (int64_t h = 0; h < nq; ++h) {
                const int64_t dst = t * qg_len + h * 2 * hd;
                const int64_t src = t * q_len + h * hd;
                for (int64_t i = 0; i < hd; ++i) {
                    call.dc->at(0).value[(size_t)(dst + i)] += d_q_lanes[(size_t)(src + i)];
                    // The gate lanes pass the cotangent down unchanged.
                    call.dc->at(0).value[(size_t)(dst + hd + i)] +=
                        call.cot[s.cot_gate].value[(size_t)(src + i)];
                }
            }
        }

        // The keys, and the store's cotangent of the pool rows this step wrote. Only the pool's
        // cotangent of a row the step wrote exists; a store consumes the pool's, it does not
        // produce one.
        std::vector<double> d_k = call.cot[s.cot_k].value;
        if (g.kv_store) {
            std::vector<int32_t> slot((size_t)tokens);
            for (int64_t t = 0; t < tokens; ++t)
                slot[(size_t)t] = (int32_t)call.fwd[s.slot_map].value[(size_t)t];
            const KvStoreShape kv{tokens, nkv, hd, g.block_size, call.fwd[s.pool].shape[0]};
            std::vector<double> d_k_pool((size_t)(tokens * k_len));
            std::vector<double> d_v_pool((size_t)(tokens * k_len));
            r97::adjoint::qk::kv_store_adjoint_f64(call.cot[s.cot_pool].value.data(),
                                                   slot.data(), kv, d_k_pool.data(),
                                                   d_v_pool.data());
            const double k_scale = num_param(call.rec, "k_scale", 1.0);
            const double v_scale = num_param(call.rec, "v_scale", 1.0);
            for (size_t i = 0; i < d_k.size(); ++i) d_k[i] += k_scale * d_k_pool[i];
            for (size_t i = 0; i < d_v_pool.size(); ++i)
                call.dc->at(s.v).value[i] += v_scale * d_v_pool[i];
        }
        std::vector<double> d_k_turned((size_t)(tokens * k_len));
        r97::adjoint::qk::rope_adjoint_f64(d_k.data(), positions.data(), k_rope, table,
                                           d_k_turned.data());
        const std::vector<double> d_k_lanes = norm_adjoint(np, call.fwd[1].value,
                                                           call.fwd[5].value, d_k_turned,
                                                           tokens * nkv, hd);
        for (size_t i = 0; i < d_k_lanes.size(); ++i) call.dc->at(1).value[i] += d_k_lanes[i];
    };
}

/* ------------------------------------------------------------------ the forward */

// The card's forward in f64: the host's rmsnorm twin, rope_forward_f64 and kv_store_forward_f64.
// The pool the store writes is a written operand, so the twin is what gives it a value; the rows
// the step does not write stay zero, which nothing here reads.
void qk_gate_twin(const OpRecord& rec, const std::vector<AdValue>& op, std::vector<AdValue>* out) {
    if (op.size() < 6) return;
    const bool kv = rec.op == "qk_norm_rope_gate_kv_store";
    const int64_t hd = (int64_t)op[4].value.size();
    const int64_t tokens = (int64_t)op[0].shape[0];
    const int64_t qg_len = op[0].shape[1];
    const int64_t k_len = op[1].shape[1];
    const int64_t rot = op[2].shape[1];
    if (hd <= 0 || tokens <= 0) return;
    const int64_t nq = qg_len / (2 * hd);
    const int64_t nkv = k_len / hd;
    const int64_t q_len = nq * hd;
    const Slots s = slots_of(kv);
    if (op.size() < (kv ? 8u : 6u)) return;

    RopeTable table;
    table.rows = op[2].shape[0];
    table.rotary_dim = rot;
    table.cos_sin = op[2].value;
    std::vector<int32_t> positions((size_t)tokens);
    for (int64_t t = 0; t < tokens; ++t) positions[(size_t)t] = (int32_t)op[3].value[(size_t)t];
    const NormParams np{num_param(rec, "eps", kEps), num_param(rec, "wadd", kWadd)};

    std::vector<double> q_lanes((size_t)(tokens * q_len));
    std::vector<double> gate((size_t)(tokens * q_len));
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t h = 0; h < nq; ++h) {
            const int64_t src = t * qg_len + h * 2 * hd;
            const int64_t dst = t * q_len + h * hd;
            for (int64_t i = 0; i < hd; ++i) {
                q_lanes[(size_t)(dst + i)] = op[0].value[(size_t)(src + i)];
                gate[(size_t)(dst + i)] = op[0].value[(size_t)(src + hd + i)];
            }
        }
    }
    const std::vector<double> q_normed = norm_forward(np, q_lanes, op[4].value, tokens * nq, hd);
    std::vector<double> q_out((size_t)(tokens * q_len));
    r97::adjoint::qk::rope_forward_f64(q_normed.data(), positions.data(),
                                       RopeShape{tokens, nq, 0, hd, rot}, table, q_out.data());

    const std::vector<double> k_normed =
        norm_forward(np, op[1].value, op[5].value, tokens * nkv, hd);
    std::vector<double> k_out((size_t)(tokens * k_len));
    r97::adjoint::qk::rope_forward_f64(k_normed.data(), positions.data(),
                                       RopeShape{tokens, nkv, 0, hd, rot}, table, k_out.data());

    auto shaped = [&](size_t slot) {
        AdValue v;
        v.dtype = rec.operands.at(slot).dtype;
        v.shape = rec.operands.at(slot).shape;
        v.value.assign((size_t)numel(v.shape), 0.0);
        return v;
    };
    auto with = [&](size_t slot, std::vector<double> values) {
        AdValue v = shaped(slot);
        v.value = std::move(values);
        return v;
    };

    if (kv) {
        AdValue pool = shaped(s.pool);
        std::vector<int32_t> slot((size_t)tokens);
        for (int64_t t = 0; t < tokens; ++t)
            slot[(size_t)t] = (int32_t)op[s.slot_map].value[(size_t)t];
        const KvStoreShape store{tokens, nkv, hd, (int64_t)num_param(rec, "block_size", 16),
                                 pool.shape[0]};
        r97::adjoint::qk::kv_store_forward_f64(k_out.data(), op[s.v].value.data(), slot.data(),
                                               store, pool.value.data());
        out->push_back(std::move(pool));
    }
    out->push_back(with(s.q, std::move(q_out)));
    out->push_back(with(s.k, std::move(k_out)));
    out->push_back(with(s.gate, std::move(gate)));
}

}  // namespace

// The op has no host row of its own, so its forward comes from the twin this file registers.
R97_REGISTER_TWIN("qk_norm_rope_gate", qk_gate_twin)
R97_REGISTER_TWIN("qk_norm_rope_gate_kv_store", qk_gate_twin)

namespace {

/* ------------------------------------------------------------------ the shapes */

// The tiny fixture: 4 query heads, 2 key-value heads, 32 lanes a head, half of them turned.
Geometry fixture_geometry(bool kv_store) {
    Geometry g;
    g.lengths = {6};
    g.n_head = 4;
    g.n_head_kv = 2;
    g.head_dim = 32;
    g.rot = 16;
    g.block_size = 4;
    g.kv_store = kv_store;
    return g;
}

// The real block: 24 query heads, 4 key-value heads, 256 lanes a head, a quarter of them turned.
Geometry real_geometry(std::vector<int64_t> lengths, bool kv_store) {
    Geometry g;
    g.lengths = std::move(lengths);
    g.n_head = 24;
    g.n_head_kv = 4;
    g.head_dim = 256;
    g.rot = 64;
    g.block_size = 16;
    g.kv_store = kv_store;
    return g;
}

void check_case(const Case& c, const Geometry& g, int reps) {
    // The op has no host row of its own; its reference is the composition this file builds.
    const Report r = device_check::check(c, nullptr, qk_reference(g));
    device_check::require(r);
    // The kernel takes every device buffer from the scratch allocator, and check() fails it if one
    // is still held when it returns.
    R97_CHECK(r.scratch_peak > 0);
    std::printf("qk %s: device == host, scratch peak %lld KiB\n", c.label.c_str(),
                (long long)(r.scratch_peak / 1024));
    std::fflush(stdout);
    if (r97test::bench()) device_check::print_bench(c, reps);
}

// A wrong reference: the gate lanes are dropped, so the comparison has to notice them.
HostRef dgate_dropped(const Geometry& g) {
    const HostRef right = qk_reference(g);
    return [right, g](AdjointCall& call) {
        right(call);
        const int64_t tokens = (int64_t)call.fwd[0].shape[0];
        const int64_t qg_len = g.n_head * 2 * g.head_dim;
        for (int64_t t = 0; t < tokens; ++t) {
            for (int64_t h = 0; h < g.n_head; ++h) {
                for (int64_t i = 0; i < g.head_dim; ++i) {
                    const size_t at =
                        (size_t)(t * qg_len + h * 2 * g.head_dim + g.head_dim + i);
                    call.dc->at(0).value[at] = 0.0;
                }
            }
        }
    };
}

// The reference is the adjoint of the forward: a small step along a random direction of the
// reads must move the writes by as much as the reference's cotangent says it does. Without this the
// comparison would be two opinions, not a check.
void check_reference_against_the_forward(const Geometry& g) {
    Case c = qk_case(g, "reference against the forward", 0x97c0beefull);
    std::vector<size_t> moved{0, 1};  // the projection, and the key projection
    if (g.kv_store) moved.push_back(6);  // the value projection, which the pool cotangent reaches

    uint64_t seed = 0x97c0d1ceull;
    const double step = 1e-3;
    std::vector<AdValue> dir;
    for (size_t i : moved)
        dir.push_back(device_check::random_value(c.in[i].dtype, c.in[i].shape, &seed));

    // The forward a step either way, so the derivative is centred and only the adjoint's error is
    // left to show.
    auto forward_at = [&](double sign) {
        Case shifted = c;
        for (size_t k = 0; k < moved.size(); ++k) {
            const size_t i = moved[k];
            for (size_t j = 0; j < shifted.in[i].value.size(); ++j)
                shifted.in[i].value[j] += sign * step * dir[k].value[j];
        }
        std::vector<AdValue> fwd, dc;
        device_check::make_call(shifted, &fwd, &dc);
        return fwd;
    };
    const std::vector<AdValue> fwd_plus = forward_at(1.0);
    const std::vector<AdValue> fwd_minus = forward_at(-1.0);

    std::vector<AdValue> fwd_a, dc_a;
    AdjointCall call = device_check::make_call(c, &fwd_a, &dc_a);
    qk_reference(g)(call);

    const size_t first_write = c.in.size();
    double moved_up = 0.0;  // what the forward says the step did
    for (size_t k = 0; k < c.cot.size(); ++k) {
        for (size_t j = 0; j < c.cot[k].value.size(); ++j)
            moved_up += 0.5 * c.cot[k].value[j] *
                        (fwd_plus[first_write + k].value[j] - fwd_minus[first_write + k].value[j]);
    }
    double moved_down = 0.0;  // what the reference's cotangent says it should have done
    for (size_t k = 0; k < moved.size(); ++k) {
        const size_t i = moved[k];
        for (size_t j = 0; j < c.in[i].value.size(); ++j)
            moved_down += dc_a.at(i).value[j] * step * dir[k].value[j];
    }
    const double scale = 1.0 + std::abs(moved_up);
    std::printf("qk %s: forward moved %.12g, the reference's cotangent says %.12g\n",
                c.label.c_str(), moved_up, moved_down);
    std::fflush(stdout);
    // A centred step of 1e-3 leaves the forward's own third-order term behind, so the check is
    // held where a cotangent that is wrong by a lane's worth still fails it.
    R97_CHECK(std::abs(moved_up - moved_down) <= 1e-6 * scale);
}

void run_fixture() {
    const auto options = device_check::guard_options(int64_t(1) << 30);
    const r97test::DeviceGuard guard(options);
    device_check::require_device(options.device);
    (void)device_check::load_plugin();

    const r97::OpRows plain = r97::op_rows("qk_norm_rope_gate");
    R97_CHECK(plain.device_adjoint);
    R97_CHECK(plain.twin);  // this file's own
    R97_CHECK(plain.device_frozen == (std::vector<int>{2, 3, 4, 5}));
    const r97::OpRows store = r97::op_rows("qk_norm_rope_gate_kv_store");
    R97_CHECK(store.device_adjoint);
    R97_CHECK(store.device_frozen == (std::vector<int>{2, 3, 4, 5, 7}));
    R97_CHECK(r97::op_rows("qk_norm_rope_gate_kv_store_streamed").device_adjoint);

    check_reference_against_the_forward(fixture_geometry(false));
    check_reference_against_the_forward(fixture_geometry(true));
    check_reference_against_the_forward(real_geometry({64, 48}, true));

    const Geometry plain_geom = fixture_geometry(false);
    const Geometry store_geom = fixture_geometry(true);
    check_case(qk_case(plain_geom, "fixture 6 tok, q and k", 0x97c00001ull), plain_geom, 20);
    check_case(qk_case(store_geom, "fixture 6 tok, with the pool", 0x97c00002ull), store_geom, 20);

    // A table the record calls bf16 reaches the op rounded, so the kernel's own rounding has to
    // agree with the tape's.
    Geometry bf16_table = fixture_geometry(false);
    bf16_table.cs_f32 = false;
    check_case(qk_case(bf16_table, "fixture 6 tok, bf16 table", 0x97c00003ull), bf16_table, 20);

    // A reference that drops the gate's lanes must be rejected in the projection's slot.
    const Report wrong =
        device_check::check(qk_case(bf16_table, "fixture 6 tok, table and gate dropped",
                                    0x97c00004ull),
                            nullptr, dgate_dropped(bf16_table));
    R97_CHECK(!wrong.ok);
    R97_CHECK(wrong.first_slot == 0);
    std::printf("rejected the reference that dropped the gate: %s\n", wrong.note.c_str());
}

void run_real() {
    const auto options = device_check::guard_options(int64_t(2) << 30);
    const r97test::DeviceGuard guard(options);
    device_check::require_device(options.device);
    (void)device_check::load_plugin();

    // One sequence at the two lengths a step sees, both forms of the op, and the ragged batch a
    // prefill of three conversations gives the block.
    const std::vector<Geometry> geoms = {
        real_geometry({1024}, false),
        real_geometry({1024}, true),
        real_geometry({448, 320, 256}, true),
        real_geometry({4096}, true),
        real_geometry({4096}, false),
    };
    const std::vector<Case> cases = {
        qk_case(geoms[0], "1024 tok, 1 seq, q and k", 0x97c00101ull),
        qk_case(geoms[1], "1024 tok, 1 seq, with the pool", 0x97c00102ull),
        qk_case(geoms[2], "1024 tok, 3 ragged seqs, pool", 0x97c00103ull),
        qk_case(geoms[3], "4096 tok, 1 seq, with the pool", 0x97c00104ull),
        qk_case(geoms[4], "4096 tok, 1 seq, q and k", 0x97c00105ull),
    };
    for (size_t i = 0; i < cases.size(); ++i) check_case(cases[i], geoms[i], 6);
}

// The streamed store's pool is quantised, which training does not keep, so its row refuses with the
// reason instead of taking a cotangent it cannot use.
void run_streamed_refusal() {
    const auto options = device_check::guard_options(int64_t(1) << 30);
    const r97test::DeviceGuard guard(options);
    device_check::require_device(options.device);
    (void)device_check::load_plugin();

    OpRecord rec;
    rec.op = "qk_norm_rope_gate_kv_store_streamed";
    std::string err;
    const AdjointFn fn = r97::adjoint_find(rec.op, Side::device, rec.params, &err);
    R97_CHECK(fn != nullptr);
    if (!fn) return;

    std::vector<AdValue> fwd;
    std::vector<AdValue> cot;
    std::vector<AdValue> dc;
    const AdjointCall call{rec, fwd, cot, &dc, nullptr, ""};
    std::string message;
    bool threw = false;
    try {
        fn(call);
    } catch (const std::runtime_error& e) {
        threw = true;
        message = e.what();
    }
    R97_CHECK(threw);
    R97_CHECK(message.find("refused in training") != std::string::npos);
    std::printf("streamed form: %s\n", message.c_str());
}

}  // namespace

R97_TEST(adjoint_qk_fixture) { run_fixture(); }

R97_TEST(adjoint_qk_real) { run_real(); }

R97_TEST(adjoint_qk_streamed_is_refused) { run_streamed_refusal(); }
