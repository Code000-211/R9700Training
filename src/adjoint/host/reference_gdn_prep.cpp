// The reference step's f64 variant of this family: the same rows under the same op names,
// registered behind `reference_record` so only the fixture's composed graph reaches them.
// The family file and its doc page stay the ones every other walk uses.
// src/adjoint/host/reference_gdn_prep.cpp -- the gated delta net's pre-recurrence family.
//
// Two ops, both outside the recurrence, both with frozen weights:
//
//   gdn_conv_prep      the short causal convolution over the projected input, its
//                      activation, the l2 norm of q and k, and the per-chunk gate
//                      and beta the recurrence reads.
//   gdn_gated_rmsnorm  the gated RMS norm on the recurrence's output.
//
// The formulas are in docs/adjoints/gdn_prep.md; the derivation of the recurrence
// around them is docs/adjoints/gdn-math.md.
//
// The conv op is recorded here at a prefill from position 0: the carried
// convolution state is zero, so the op reads no history and what it writes into
// `conv_state` is the tail of `x`. That is also why the state is a written
// operand in the records this family serves -- there is nothing to read at row 0
// -- and why `has_init` is frozen: at a cold start it does not vary. The schema's
// optional operands (`b`, `a`, `b_gate`, `cache_idx`, `has_init`) are all present
// in these records, so the twin implements the shape those operands describe:
// the fused row of `x` carries the conv channels only and the gates arrive in
// their own operands, the path every current caller takes.
//
// Convolution weights, the conv bias, A_log, dt_bias and the norm weights are
// frozen: they are the model's, not the step's, so the walk never asks for their
// cotangent. The index operands (`cu`, `cache_idx`, `has_init`) are frozen for
// the same reason in the other sense: they are integers, and no cotangent of an
// integer means anything.
#include "r97/adjoint.h"
#include "reference_record.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace r97 {
namespace reference_gdn_prep {
namespace {

/* gdn_conv_prep's schema positions, in schema order. */
enum : int {
    kX = 0,        // the fused projection row, [M, conv_dim] bf16
    kW = 1,        // the conv weight, [conv_dim, conv_width] bf16, frozen
    kBias = 2,     // the conv bias, [conv_dim] bf16, frozen (optional in the schema)
    kALog = 3,     // [H] f32, frozen
    kDtBias = 4,   // [H] f32, frozen
    kState = 5,    // conv_state, [S, conv_dim, conv_width-1] bf16, INOUT
    kCu = 6,       // sequence offsets, [S+1] i32, frozen
    kA = 7,        // the gate projection, [M, H] bf16
    kBGate = 8,    // the beta projection, [M, H] bf16
    kSidx = 9,     // conv-state slot per sequence, [S] i32, frozen
    kHasInit = 10, // cold (0) or warm (1) start per sequence, [S] i32, frozen
    kQ = 11,       // [M, Hg, K] bf16, written
    kK = 12,       // [M, Hg, K] bf16, written
    kV = 13,       // [M, H, V] bf16, written
    kG = 14,       // [M, H] f32, written
    kBeta = 15,    // [M, H] f32, written
    kConvPrepOperands = 16,
};

/* The same operands as a twin receives them -- the reads, in schema order. */
enum : int {
    rX = 0,
    rW = 1,
    rBias = 2,
    rALog = 3,
    rDtBias = 4,
    rCu = 5,
    rA = 6,
    rBGate = 7,
    rSidx = 8,
    rHasInit = 9,
    rConvPrepReads = 10,
};

/* And the writes, in the order the twin produces them and the walk hands their
 * cotangents: schema order, which puts the state before the five outputs. */
enum : int {
    wState = 0,
    wQ = 1,
    wK = 2,
    wV = 3,
    wG = 4,
    wBeta = 5,
    wConvPrepWrites = 6,
};

/* gdn_gated_rmsnorm's schema positions. */
enum : int {
    nX = 0,
    nZ = 1,
    nW = 2,  // the norm weight, [channels] f32, frozen
    nO = 3,
    nRmsnormOperands = 4,
};

/* The epsilons and the softplus threshold are f64 parameters the op's schema
 * allows and the tape's parameter list cannot carry (it holds integers and
 * strings), so the twin uses the op's defaults and the kernel does the same:
 * l2_eps 1e-6, eps 1e-6, softplus_thr 20. */
constexpr double kL2Eps = 1e-6;
constexpr double kRmsEps = 1e-6;
constexpr double kSoftplusThr = 20.0;

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

int64_t int_param(const OpRecord& rec, const char* name, int64_t fallback) {
    for (const OpParam& p : rec.params) {
        if (p.first != name) continue;
        if (auto* i = std::get_if<int64_t>(&p.second)) return *i;
        return fallback;
    }
    return fallback;
}

std::string str_param(const OpRecord& rec, const char* name, const char* fallback) {
    for (const OpParam& p : rec.params) {
        if (p.first != name) continue;
        if (auto* s = std::get_if<std::string>(&p.second)) return *s;
        return fallback;
    }
    return fallback;
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

double silu(double x) { return x * sigmoid(x); }

/* d/dx of x * sigmoid(x). */
double silu_grad(double x) {
    const double s = sigmoid(x);
    return s * (1.0 + x * (1.0 - s));
}

double softplus(double x) {
    if (x > kSoftplusThr) return x;
    return x > 0.0 ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x));
}

/* d/dx of softplus, cut off at the same threshold the forward uses. */
double softplus_grad(double x) { return x > kSoftplusThr ? 1.0 : sigmoid(x); }

/* A written operand sized and typed as its schema row says. */
AdValue write_value(const OpRecord& rec, int pos) {
    AdValue v;
    v.dtype = rec.operands[(size_t)pos].dtype;
    v.shape = rec.operands[(size_t)pos].shape;
    v.value.assign((size_t)numel(v.shape), 0.0);
    return v;
}

/* What the conv op's shapes resolve to. `cu` is where the token count and the
 * sequence boundaries come from, as they do for the kernel: `M` is the shape's
 * row count, `cu[N]` the tokens it actually ranges over. */
struct ConvGeom {
    int64_t K = 0;         // head_k: q's and k's per-head width
    int64_t V = 0;         // head_v: v's per-head width
    int64_t chunk = 0;     // tokens per gate chunk
    int64_t width = 0;     // conv_width
    int64_t N = 0;         // sequences
    int64_t T = 0;         // tokens
    int64_t Hg = 0;        // q and k heads
    int64_t H = 0;         // v heads, the gate's and beta's width
    int64_t conv_dim = 0;  // 2 * Hg * K + H * V: q, then k, then v
    int64_t xw = 0;        // x's row width, dense
    int64_t slots = 0;     // conv-state slots
};

bool conv_geom(const OpRecord& rec, const AdValue& x, const AdValue& cu, ConvGeom* g) {
    if (rec.operands.size() < (size_t)kConvPrepOperands) return false;
    g->K = int_param(rec, "head_k", 0);
    g->V = int_param(rec, "head_v", g->K);
    g->chunk = int_param(rec, "chunk", 64);
    g->width = int_param(rec, "conv_width", 4);
    if (g->K <= 0 || g->V <= 0 || g->chunk <= 0 || g->width <= 0) return false;
    if (x.shape.empty()) return false;
    g->xw = x.shape.back();
    if (g->xw <= 0) return false;
    g->N = (int64_t)cu.value.size() - 1;
    if (g->N <= 0) return false;
    g->T = (int64_t)cu.value.back();
    if (g->T <= 0) return false;
    if (numel(x.shape) < g->T * g->xw) return false;
    const int64_t qn = numel(rec.operands[kQ].shape);
    const int64_t vn = numel(rec.operands[kV].shape);
    g->Hg = qn / (g->T * g->K);
    g->H = vn / (g->T * g->V);
    if (g->Hg <= 0 || g->H <= 0) return false;
    g->conv_dim = 2 * g->Hg * g->K + g->H * g->V;
    if (g->conv_dim > g->xw) return false;
    const int64_t per_slot = g->conv_dim * (g->width - 1);
    if (per_slot > 0) g->slots = numel(rec.operands[kState].shape) / per_slot;
    return true;
}

/* Phase 1: the causal convolution and its activation, per (token, channel). The
 * value the op stores before the norm is silu(acc) and both readers of this
 * function need acc back, so both are returned. Rows are global: token t of
 * sequence n is row cu[n] + t, which is the row q, k, v, g and beta share. */
void conv_phase(const ConvGeom& g, const AdValue& x, const AdValue& w, const AdValue& bias,
                const AdValue& cu, std::vector<double>* acc, std::vector<double>* u) {
    const size_t rows = (size_t)(g.T * g.conv_dim);
    acc->assign(rows, 0.0);
    u->assign(rows, 0.0);
    const bool has_bias = bias.value.size() == (size_t)g.conv_dim;
    for (int64_t n = 0; n < g.N; ++n) {
        const int64_t bos = (int64_t)cu.value[(size_t)n];
        const int64_t slen = (int64_t)cu.value[(size_t)(n + 1)] - bos;
        for (int64_t t = 0; t < slen; ++t) {
            const int64_t row = bos + t;
            for (int64_t c = 0; c < g.conv_dim; ++c) {
                double a = has_bias ? bias.value[(size_t)c] : 0.0;
                const size_t wbase = (size_t)(c * g.width);
                for (int64_t j = 0; j < g.width; ++j) {
                    const int64_t src = t - (g.width - 1) + j;
                    if (src < 0) continue;  // the carried state is zero at prefill
                    a += x.value[(size_t)((bos + src) * g.xw + c)] * w.value[wbase + (size_t)j];
                }
                acc->at((size_t)(row * g.conv_dim + c)) = a;
                u->at((size_t)(row * g.conv_dim + c)) = silu(a);
            }
        }
    }
}

/* Phase 2: q and k are l2-normalised per head, over the K channels of a token.
 * `chan0` is the first channel the head's block occupies in the conv row. */
void l2_normalise(const ConvGeom& g, const std::vector<double>& u, AdValue* o, int64_t chan0) {
    for (int64_t row = 0; row < g.T; ++row) {
        for (int64_t h = 0; h < g.Hg; ++h) {
            const int64_t base = row * g.conv_dim + chan0 + h * g.K;
            double ss = 0.0;
            for (int64_t d = 0; d < g.K; ++d) ss += u[(size_t)(base + d)] * u[(size_t)(base + d)];
            const double inv = 1.0 / std::sqrt(ss + kL2Eps);
            const int64_t obase = (row * g.Hg + h) * g.K;
            for (int64_t d = 0; d < g.K; ++d)
                o->value[(size_t)(obase + d)] = u[(size_t)(base + d)] * inv;
        }
    }
}

}  // namespace

/* Twin: gdn_conv_prep's forward in f64, at a prefill from position 0.
 *
 * Reads, in schema order: x, w, b, A_log, dt_bias, cu, a, b_gate, cache_idx,
 * has_init. Writes, in schema order: conv_state, q, k, v, g, beta. */
void gdn_conv_prep_twin(const OpRecord& rec, const std::vector<AdValue>& op,
                        std::vector<AdValue>* out) {
    if (op.size() < (size_t)rConvPrepReads) return;
    const AdValue& x = op[rX];
    const AdValue& w = op[rW];
    const AdValue& bias = op[rBias];
    const AdValue& a_log = op[rALog];
    const AdValue& dt_bias = op[rDtBias];
    const AdValue& cu = op[rCu];
    const AdValue& a = op[rA];
    const AdValue& b_gate = op[rBGate];
    const AdValue& sidx = op[rSidx];
    ConvGeom g;
    if (!conv_geom(rec, x, cu, &g)) return;
    if (a.value.size() < (size_t)(g.T * g.H) || b_gate.value.size() < (size_t)(g.T * g.H))
        return;
    if (a_log.value.size() < (size_t)g.H || dt_bias.value.size() < (size_t)g.H) return;

    std::vector<AdValue> o;
    o.reserve((size_t)wConvPrepWrites);
    for (int pos : {kState, kQ, kK, kV, kG, kBeta}) o.push_back(write_value(rec, pos));

    std::vector<double> acc;
    std::vector<double> u;
    conv_phase(g, x, w, bias, cu, &acc, &u);

    /* Phase 1's outputs: channel c of the conv row is q's channel c while
     * c < Hg*K, then k's, then v's. The channel of a head is its per-head
     * position, and the flat index follows the schema's dense row-major
     * [M, heads, width] shape. */
    for (int64_t row = 0; row < g.T; ++row) {
        for (int64_t c = 0; c < g.conv_dim; ++c) {
            const double v = u[(size_t)(row * g.conv_dim + c)];
            if (c < g.Hg * g.K) {
                const int64_t h = c / g.K;
                const int64_t d = c % g.K;
                o[wQ].value[(size_t)((row * g.Hg + h) * g.K + d)] = v;
            } else if (c < 2 * g.Hg * g.K) {
                const int64_t ck = c - g.Hg * g.K;
                const int64_t h = ck / g.K;
                const int64_t d = ck % g.K;
                o[wK].value[(size_t)((row * g.Hg + h) * g.K + d)] = v;
            } else {
                const int64_t cv = c - 2 * g.Hg * g.K;
                const int64_t h = cv / g.V;
                const int64_t d = cv % g.V;
                o[wV].value[(size_t)((row * g.H + h) * g.V + d)] = v;
            }
        }
    }
    l2_normalise(g, u, &o[wQ], 0);
    l2_normalise(g, u, &o[wK], g.Hg * g.K);

    /* Phase 3: the gate accumulates its per-token increment within a chunk, and
     * beta is the sigmoid of its projection. The engine may record `a` and `b_gate`
     * as the parent x buffer (the runtime hands the kernel a column view, and a tape
     * keeps the buffer it points into); their row stride is then x's, and their first
     * column is x's last 2H. A test that feeds narrow [T, H] views gets the packed
     * layout, so both are read through their own shape. */
    const bool narrow_gate = a.shape != x.shape;
    const int64_t gate_stride = narrow_gate ? g.H : g.xw;
    const int64_t a_off = narrow_gate ? 0 : g.xw - 2 * g.H;
    const int64_t b_off = narrow_gate ? 0 : g.xw - g.H;
    for (int64_t n = 0; n < g.N; ++n) {
        const int64_t bos = (int64_t)cu.value[(size_t)n];
        const int64_t slen = (int64_t)cu.value[(size_t)(n + 1)] - bos;
        for (int64_t h = 0; h < g.H; ++h) {
            const double decay = -std::exp(a_log.value[(size_t)h]);
            double run = 0.0;
            for (int64_t t = 0; t < slen; ++t) {
                if (t % g.chunk == 0) run = 0.0;
                const int64_t at = (bos + t) * g.H + h;
                const size_t av = (size_t)((bos + t) * gate_stride + a_off + h);
                const size_t bv = (size_t)((bos + t) * gate_stride + b_off + h);
                run += decay * softplus(a.value[av] + dt_bias.value[(size_t)h]);
                o[wG].value[(size_t)at] = run;
                o[wBeta].value[(size_t)at] = sigmoid(b_gate.value[bv]);
            }
        }
    }

    /* Phase 4: the state the next step would carry is the tail of x's rows --
     * zero where the sequence is shorter than the window, because the start is
     * cold. */
    const int64_t keep = g.width - 1;
    for (int64_t n = 0; n < g.N && keep > 0 && g.slots > 0; ++n) {
        const int64_t bos = (int64_t)cu.value[(size_t)n];
        const int64_t slen = (int64_t)cu.value[(size_t)(n + 1)] - bos;
        const int64_t slot = (int64_t)sidx.value[(size_t)n];
        if (slot < 0 || slot >= g.slots) continue;
        for (int64_t c = 0; c < g.conv_dim; ++c) {
            for (int64_t i = 0; i < keep; ++i) {
                const int64_t src = slen - (g.width - 1) + i;
                const double v = src >= 0 ? x.value[(size_t)((bos + src) * g.xw + c)] : 0.0;
                o[wState].value[(size_t)((slot * g.conv_dim + c) * keep + i)] = v;
            }
        }
    }

    out->swap(o);
}

/* Host adjoint for gdn_conv_prep, in f64.
 *
 * Frozen: w (1), b (2), A_log (3), dt_bias (4), cu (6), cache_idx (9) and
 * has_init (10). The cotangents that flow are conv_state (writes[0]), q, k, v,
 * g and beta (writes[1..5]), and they reach x, a and b_gate. */
void gdn_conv_prep_host_adjoint(const AdjointCall& call) {
    const OpRecord& rec = call.rec;
    if (call.fwd.size() < (size_t)kConvPrepOperands ||
        call.cot.size() < (size_t)wConvPrepWrites || call.dc->size() < (size_t)kConvPrepOperands)
        return;
    const std::vector<AdValue>& f = call.fwd;
    const AdValue& x = f[kX];
    const AdValue& w = f[kW];
    const AdValue& bias = f[kBias];
    const AdValue& cu = f[kCu];
    ConvGeom g;
    if (!conv_geom(rec, x, cu, &g)) return;

    const AdValue& d_state = call.cot[wState];
    const AdValue& d_q = call.cot[wQ];
    const AdValue& d_k = call.cot[wK];
    const AdValue& d_v = call.cot[wV];
    const AdValue& d_g = call.cot[wG];
    const AdValue& d_beta = call.cot[wBeta];

    std::vector<double> acc;
    std::vector<double> u;
    conv_phase(g, x, w, bias, cu, &acc, &u);

    /* d silu(acc) per (token, channel), then the convolution's backward. */
    std::vector<double> dacc((size_t)(g.T * g.conv_dim), 0.0);

    /* q and k: through the l2 norm. With u the pre-norm value, ss = sum u^2 and
     * inv = 1/sqrt(ss + eps), the outward value is u*inv, so
     *   du_d = inv*dout_d - inv^3 * u_d * sum_e (u_e * dout_e). */
    for (int two = 0; two < 2; ++two) {
        const AdValue& dout = two == 0 ? d_q : d_k;
        const int64_t chan0 = two == 0 ? 0 : g.Hg * g.K;
        for (int64_t row = 0; row < g.T; ++row) {
            for (int64_t h = 0; h < g.Hg; ++h) {
                const int64_t base = row * g.conv_dim + chan0 + h * g.K;
                const int64_t obase = (row * g.Hg + h) * g.K;
                double ss = 0.0;
                double dot = 0.0;
                for (int64_t d = 0; d < g.K; ++d) {
                    const double ud = u[(size_t)(base + d)];
                    ss += ud * ud;
                    dot += ud * dout.value[(size_t)(obase + d)];
                }
                const double inv = 1.0 / std::sqrt(ss + kL2Eps);
                const double inv3 = inv * inv * inv;
                for (int64_t d = 0; d < g.K; ++d) {
                    const double ud = u[(size_t)(base + d)];
                    const double du = inv * dout.value[(size_t)(obase + d)] - inv3 * ud * dot;
                    dacc[(size_t)(base + d)] += du * silu_grad(acc[(size_t)(base + d)]);
                }
            }
        }
    }
    /* v: straight through the activation. */
    for (int64_t row = 0; row < g.T; ++row) {
        for (int64_t h = 0; h < g.H; ++h) {
            const int64_t base = row * g.conv_dim + 2 * g.Hg * g.K + h * g.V;
            const int64_t obase = (row * g.H + h) * g.V;
            for (int64_t d = 0; d < g.V; ++d) {
                const size_t c = (size_t)(base + d);
                dacc[c] += d_v.value[(size_t)(obase + d)] * silu_grad(acc[c]);
            }
        }
    }

    AdValue& dx = call.dc->at(kX);
    /* The convolution: each output token reads the window `width` of x's rows
     * behind it, with the weight of its position in the window. */
    for (int64_t n = 0; n < g.N; ++n) {
        const int64_t bos = (int64_t)cu.value[(size_t)n];
        const int64_t slen = (int64_t)cu.value[(size_t)(n + 1)] - bos;
        for (int64_t t = 0; t < slen; ++t) {
            const int64_t row = bos + t;
            for (int64_t c = 0; c < g.conv_dim; ++c) {
                const double d = dacc[(size_t)(row * g.conv_dim + c)];
                if (d == 0.0) continue;
                const size_t wbase = (size_t)(c * g.width);
                for (int64_t j = 0; j < g.width; ++j) {
                    const int64_t src = t - (g.width - 1) + j;
                    if (src < 0) continue;
                    dx.value[(size_t)((bos + src) * g.xw + c)] += d * w.value[wbase + (size_t)j];
                }
            }
        }
    }

    /* The carried state, which at a cold start is the tail of x itself: its
     * cotangent goes straight back to the rows it holds. */
    const int64_t keep = g.width - 1;
    for (int64_t n = 0; n < g.N && keep > 0 && g.slots > 0; ++n) {
        const int64_t bos = (int64_t)cu.value[(size_t)n];
        const int64_t slen = (int64_t)cu.value[(size_t)(n + 1)] - bos;
        const int64_t slot = (int64_t)f[kSidx].value[(size_t)n];
        if (slot < 0 || slot >= g.slots) continue;
        for (int64_t c = 0; c < g.conv_dim; ++c) {
            for (int64_t i = 0; i < keep; ++i) {
                const int64_t src = slen - (g.width - 1) + i;
                if (src < 0) continue;
                dx.value[(size_t)((bos + src) * g.xw + c)] +=
                    d_state.value[(size_t)((slot * g.conv_dim + c) * keep + i)];
            }
        }
    }

    /* The gate: g is a within-chunk running sum of increments, so the cotangent
     * of an increment is the sum of its own and every later g in its chunk; the
     * increment's two factors give a and, through beta's sigmoid, b_gate. The
     * cotangents keep the shape of the forward operand, parent-buffer view and
     * all, so the walk adds them back to x's columns. */
    AdValue& da = call.dc->at(kA);
    AdValue& d_bgate = call.dc->at(kBGate);
    const bool narrow_gate = f[kA].shape != f[kX].shape;
    const int64_t gate_stride = narrow_gate ? g.H : g.xw;
    const int64_t a_off = narrow_gate ? 0 : g.xw - 2 * g.H;
    const int64_t b_off = narrow_gate ? 0 : g.xw - g.H;
    for (int64_t n = 0; n < g.N; ++n) {
        const int64_t bos = (int64_t)cu.value[(size_t)n];
        const int64_t slen = (int64_t)cu.value[(size_t)(n + 1)] - bos;
        for (int64_t h = 0; h < g.H; ++h) {
            const double decay = -std::exp(f[kALog].value[(size_t)h]);
            double tail = 0.0;
            for (int64_t t = slen - 1; t >= 0; --t) {
                if (t == slen - 1 || (t + 1) % g.chunk == 0) tail = 0.0;
                const size_t at = (size_t)((bos + t) * g.H + h);
                const size_t av = (size_t)((bos + t) * gate_stride + a_off + h);
                const size_t bv = (size_t)((bos + t) * gate_stride + b_off + h);
                tail += d_g.value[at];
                const double a_val = f[kA].value[av] + f[kDtBias].value[(size_t)h];
                da.value[av] += tail * decay * softplus_grad(a_val);
                const double s = sigmoid(f[kBGate].value[bv]);
                d_bgate.value[bv] += d_beta.value[at] * s * (1.0 - s);
            }
        }
    }
}

/* Twin: gdn_gated_rmsnorm's forward in f64, over dense [M, channels] rows.
 *
 * o = x * rsqrt(mean(x^2) + eps) * w * act(z), with act silu (the default) or
 * sigmoid. The gate reads z's own row width, as the kernel's gate_rowoff
 * resolves z against the channel count. */
void gdn_gated_rmsnorm_twin(const OpRecord& rec, const std::vector<AdValue>& op,
                            std::vector<AdValue>* out) {
    if (op.size() < 3 || rec.operands.size() < (size_t)nRmsnormOperands) return;
    const AdValue& x = op[nX];
    const AdValue& z = op[nZ];
    const AdValue& w = op[nW];
    const int64_t ch = int_param(rec, "channels", x.shape.empty() ? 0 : x.shape.back());
    if (ch <= 0 || w.value.size() < (size_t)ch) return;
    const int64_t rows = (int64_t)x.value.size() / ch;
    if (rows <= 0 || z.value.size() < (size_t)(rows * ch)) return;
    const bool sig = str_param(rec, "act", "silu") == "sigmoid";

    AdValue o = write_value(rec, nO);
    if (o.value.size() < (size_t)(rows * ch)) return;
    for (int64_t r = 0; r < rows; ++r) {
        const size_t base = (size_t)(r * ch);
        double ss = 0.0;
        for (int64_t i = 0; i < ch; ++i) {
            const double xv = x.value[base + (size_t)i];
            ss += xv * xv;
        }
        const double sc = 1.0 / std::sqrt(ss / (double)ch + kRmsEps);
        for (int64_t i = 0; i < ch; ++i) {
            const double zv = z.value[base + (size_t)i];
            const double gate = sig ? sigmoid(zv) : silu(zv);
            o.value[base + (size_t)i] =
                x.value[base + (size_t)i] * sc * w.value[(size_t)i] * gate;
        }
    }
    out->push_back(std::move(o));
}

/* Host adjoint for gdn_gated_rmsnorm, in f64. Frozen: w (2), the norm weight.
 *
 * With sc = rsqrt(mean(x^2) + eps) and, over the row,
 *   A = sum_e (x_e * w_e * g(z_e) * do_e),
 *   dx_i = w_i * g(z_i) * sc * do_i - x_i * sc^3 / channels * A
 *   dz_i = do_i * x_i * sc * w_i * g'(z_i).
 * A already carries the row's weights and gates, so the second term of dx has
 * them from A alone. */
void gdn_gated_rmsnorm_host_adjoint(const AdjointCall& call) {
    const OpRecord& rec = call.rec;
    if (call.fwd.size() < (size_t)nRmsnormOperands || call.cot.empty() ||
        call.dc->size() < (size_t)nRmsnormOperands)
        return;
    const AdValue& x = call.fwd[nX];
    const AdValue& z = call.fwd[nZ];
    const AdValue& w = call.fwd[nW];
    const AdValue& d_o = call.cot[0];
    const int64_t ch = int_param(rec, "channels", x.shape.empty() ? 0 : x.shape.back());
    if (ch <= 0 || w.value.size() < (size_t)ch || d_o.value.size() < x.value.size()) return;
    const int64_t rows = (int64_t)x.value.size() / ch;
    if (rows <= 0 || z.value.size() < x.value.size()) return;
    const bool sig = str_param(rec, "act", "silu") == "sigmoid";

    AdValue& dx = call.dc->at(nX);
    AdValue& dz = call.dc->at(nZ);
    if (dx.value.size() < x.value.size() || dz.value.size() < x.value.size()) return;
    for (int64_t r = 0; r < rows; ++r) {
        const size_t base = (size_t)(r * ch);
        double ss = 0.0;
        double dot = 0.0;
        for (int64_t i = 0; i < ch; ++i) {
            const size_t e = base + (size_t)i;
            const double xv = x.value[e];
            const double zv = z.value[e];
            const double wi = w.value[(size_t)i];
            const double gate = sig ? sigmoid(zv) : silu(zv);
            ss += xv * xv;
            dot += xv * d_o.value[e] * wi * gate;
        }
        const double sc = 1.0 / std::sqrt(ss / (double)ch + kRmsEps);
        const double inv_ch = 1.0 / (double)ch;
        for (int64_t i = 0; i < ch; ++i) {
            const size_t e = base + (size_t)i;
            const double xv = x.value[e];
            const double zv = z.value[e];
            const double wi = w.value[(size_t)i];
            const double gate = sig ? sigmoid(zv) : silu(zv);
            const double dgate = sig ? gate * (1.0 - gate) : silu_grad(zv);
            dx.value[e] += wi * gate * sc * d_o.value[e] - xv * sc * sc * sc * inv_ch * dot;
            dz.value[e] += d_o.value[e] * xv * sc * wi * dgate;
        }
    }
}

/* The three gdn ops that only prepare state for a later decode step. Nothing a
 * prefill loss reads depends on what they write -- docs/adjoints/gdn-math.md,
 * section 5, names each of them and says so -- so they are stops: the walk asks
 * them for nothing and this row adds nothing to any cotangent. Registering them
 * rather than leaving the op unserved is what tells a deliberate stop from a
 * family nobody has written yet. No twin is registered: no forward in the loss
 * path runs them. */
void gdn_decode_stop(const AdjointCall&) {}

R97_REGISTER_TWIN_IF("gdn_conv_prep", gdn_conv_prep_twin, reference_record)
R97_REGISTER_ADJOINT_IF_F("gdn_conv_prep", host, gdn_conv_prep_host_adjoint, reference_record, kW, kBias, kALog,
                       kDtBias, kCu, kSidx, kHasInit)
R97_REGISTER_TWIN_IF("gdn_gated_rmsnorm", gdn_gated_rmsnorm_twin, reference_record)
R97_REGISTER_ADJOINT_IF_F("gdn_gated_rmsnorm", host, gdn_gated_rmsnorm_host_adjoint, reference_record, nW)

R97_REGISTER_ADJOINT_IF("gdn_conv_update", host, gdn_decode_stop, reference_record)
R97_REGISTER_ADJOINT_IF("gdn_recurrent_update", host, gdn_decode_stop, reference_record)
R97_REGISTER_ADJOINT_IF("gdn_conv_recurrent_update", host, gdn_decode_stop, reference_record)

}  // namespace reference_gdn_prep
}  // namespace r97
