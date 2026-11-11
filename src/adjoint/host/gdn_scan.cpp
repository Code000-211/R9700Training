// src/adjoint/host/gdn_scan.cpp -- the gated delta net's chunked scan.
//
// gdn_chunk_scan carries the [H, V, K] state from chunk to chunk and writes the
// layer's output rows. Its forward is
//
//   W[j]  = beta_j (v_j - e^{g_j} S0^T k_j)
//   V'[i] = sum_{j<=i} A[i][j] W[j]
//   o[i]  = scale (e^{g_i} S0^T q_i + sum_{j<=i} e^{g_i-g_j} (q_i.k_j) V'[j])
//   S1    = e^{g_last} S0 + sum_j e^{g_last-g_j} k_j (x) V'[j],
//
// per value head and chunk, with the state held transposed, St[d][e], as the
// kernels hold it. The formulas and their derivation are docs/adjoints/gdn_scan.md
// and docs/adjoints/gdn-math.md, sections 2.2 and 4.
//
// The records this family serves are prefill rows: h0 is the zero initial state
// (frozen, no cotangent), and cu/state_idx are integer structure (frozen too).
// The adjoint is a reverse scan over the chunks with one state cotangent per
// head; at the first chunk that cotangent is discarded, because the state the
// sequence entered with is zero and nothing in the graph reads a cotangent for
// it. q, k, v, A, g and beta all receive cotangents, and A's feeds
// gdn_kkt_solve's adjoint.
#include "r97/adjoint.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace r97 {
namespace {

/* gdn_chunk_scan's schema positions, in schema order. */
enum : int {
    kQ = 0,      // [T, Hg, K] bf16
    kK = 1,      // [T, Hg, K] bf16
    kV = 2,      // [T, H, V] bf16
    kA = 3,      // [T, H, chunk] bf16, unit lower triangular within a chunk
    kG = 4,      // [T, H] f32, within-chunk gate cumsum
    kBeta = 5,   // [T, H] f32
    kH0 = 6,     // [Nslots, H, V, K] f32, zero at a prefill start, frozen
    kCu = 7,     // [S+1] i32, sequence boundaries, frozen
    kO = 8,      // [T, H, V] bf16, written
    kHt = 9,     // [Nslots, H, V, K] f32, written
    kSidx = 10,  // [S] i32, optional state slot per sequence, frozen
};

/* The read operands' slots, in schema order. `operands` holds only the reads, so these are not
 * the schema positions above: the writes sit between them there. */
enum : int {
    rQ = 0,
    rK = 1,
    rV = 2,
    rA = 3,
    rG = 4,
    rBeta = 5,
    rH0 = 6,
    rCu = 7,
    rSidx = 8,
};

/* The twin's writes, in the order it produces them and the walk hands their
 * cotangents. */
enum : int { wO = 0, wHt = 1, wScanWrites = 2 };

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

double param_f64(const OpRecord& rec, const char* name, double fallback) {
    for (const OpParam& p : rec.params) {
        if (p.first != name) continue;
        if (const auto* i = std::get_if<int64_t>(&p.second)) return (double)*i;
        const std::string& s = std::get<std::string>(p.second);
        char* end = nullptr;
        const double v = std::strtod(s.c_str(), &end);
        if (end && end != s.c_str() && *end == '\0' && std::isfinite(v)) return v;
        return fallback;
    }
    return fallback;
}

/* A written operand sized and typed as its schema row says. */
AdValue write_value(const OpRecord& rec, int pos) {
    AdValue v;
    v.dtype = rec.operands[(size_t)pos].dtype;
    v.shape = rec.operands[(size_t)pos].shape;
    v.value.assign((size_t)numel(v.shape), 0.0);
    return v;
}

/* What the scan's shapes resolve to. `cu` is where the token count and the
 * sequence boundaries come from, as they do for the kernel. A and the state
 * are read through their dense row-major layouts. */
struct ScanGeom {
    int64_t K = 0;       // head_k
    int64_t V = 0;       // head_v
    int64_t chunk = 0;   // tokens per gate chunk
    double scale = 1.0;  // the op's optional f64 scale, see scan_geom
    int64_t N = 0;       // sequences
    int64_t T = 0;       // tokens
    int64_t Hg = 0;      // q and k heads
    int64_t H = 0;       // v heads, the gate's and beta's width
    int64_t grp = 0;     // value heads per key head
    int64_t h0_slots = 0;
    int64_t ht_slots = 0;
};

bool scan_geom(const OpRecord& rec, const std::vector<AdValue>& f, ScanGeom* g) {
    if (rec.operands.size() < (size_t)kHt + 1 || f.size() <= (size_t)rCu) return false;
    const std::vector<int64_t>& qs = rec.operands[kQ].shape;
    const std::vector<int64_t>& vs = rec.operands[kV].shape;
    g->K = int_param(rec, "head_k", qs.empty() ? 0 : qs.back());
    g->V = int_param(rec, "head_v", vs.empty() ? 0 : vs.back());
    g->chunk = int_param(rec, "chunk", rec.operands[kA].shape.empty()
                                             ? 0
                                             : rec.operands[kA].shape.back());
    if (g->K <= 0 || g->V <= 0 || g->chunk <= 0) return false;
    /* The kernel's default is 1, but the architecture issues 1/sqrt(head_k) and the tape drops an
     * f64 parameter, so a record off the tape carries no `scale` at all. Fall back to the value
     * the model ran with, or every scan's output is sqrt(head_k) too large. */
    g->scale = param_f64(rec, "scale", 1.0 / std::sqrt((double)g->K));

    const AdValue& cu = f[(size_t)rCu];
    g->N = (int64_t)cu.value.size() - 1;
    if (g->N <= 0) return false;
    g->T = (int64_t)cu.value.back();
    if (g->T <= 0) return false;
    if (numel(qs) < g->T * g->K || numel(vs) < g->T * g->V) return false;
    g->Hg = numel(qs) / (g->T * g->K);
    g->H = numel(vs) / (g->T * g->V);
    if (g->Hg <= 0 || g->H <= 0 || g->H % g->Hg != 0) return false;
    g->grp = g->H / g->Hg;

    const std::vector<int64_t>& h0s = rec.operands[kH0].shape;
    const std::vector<int64_t>& hts = rec.operands[kHt].shape;
    g->h0_slots = h0s.empty() ? g->N : h0s.front();
    g->ht_slots = hts.empty() ? g->N : hts.front();
    if (g->h0_slots <= 0 || g->ht_slots <= 0) return false;
    /* state_idx may point several sequences at a larger cache, so the slot
     * count is not bounded below by the sequence count when it is present. */
    const bool has_state_idx = rec.operands.size() > (size_t)kSidx;
    if (!has_state_idx && (g->h0_slots < g->N || g->ht_slots < g->N)) return false;
    return true;
}

/* The state slot a sequence owns. Absent state_idx, row n owns slot n. `pos` is
 * state_idx's slot in the array handed in: the twin holds the reads alone (rSidx),
 * the adjoint the full record (kSidx). */
int64_t state_slot(const OpRecord& rec, const std::vector<AdValue>& f, int pos, int64_t n) {
    if (rec.operands.size() > (size_t)kSidx && f.size() > (size_t)pos &&
        (int64_t)f[(size_t)pos].value.size() > n)
        return (int64_t)f[(size_t)pos].value[(size_t)n];
    return n;
}

/* S0, transposed to St[d][e], from h0's [slot, head, e, d] block. */
void load_state(const ScanGeom& g, const AdValue& h0, int64_t slot, int64_t h,
                std::vector<double>* S) {
    S->assign((size_t)(g.K * g.V), 0.0);
    if (slot < 0) return;
    const int64_t per = g.V * g.K;
    const int64_t base = (slot * g.H + h) * per;
    if (base < 0 || base + per > (int64_t)h0.value.size()) return;
    for (int64_t e = 0; e < g.V; ++e)
        for (int64_t d = 0; d < g.K; ++d)
            (*S)[(size_t)(d * g.V + e)] = h0.value[(size_t)(base + e * g.K + d)];
}

/* The chunk's operand slices, widened to f64. A stores only j <= i. */
void load_chunk(const ScanGeom& g, const std::vector<AdValue>& f, int64_t bos, int64_t h,
                int64_t c0, int64_t L, std::vector<double>* qc, std::vector<double>* kc,
                std::vector<double>* vc, std::vector<double>* Ac, std::vector<double>* gc,
                std::vector<double>* bc) {
    const int64_t hk = h / g.grp;
    qc->assign((size_t)(L * g.K), 0.0);
    kc->assign((size_t)(L * g.K), 0.0);
    vc->assign((size_t)(L * g.V), 0.0);
    Ac->assign((size_t)(L * g.chunk), 0.0);
    gc->assign((size_t)L, 0.0);
    bc->assign((size_t)L, 0.0);
    for (int64_t j = 0; j < L; ++j) {
        const int64_t tj = bos + c0 + j;
        gc->at((size_t)j) = f[(size_t)rG].value[(size_t)(tj * g.H + h)];
        bc->at((size_t)j) = f[(size_t)rBeta].value[(size_t)(tj * g.H + h)];
        for (int64_t d = 0; d < g.K; ++d) {
            const size_t at = (size_t)((tj * g.Hg + hk) * g.K + d);
            qc->at((size_t)(j * g.K + d)) = f[(size_t)rQ].value[at];
            kc->at((size_t)(j * g.K + d)) = f[(size_t)rK].value[at];
        }
        for (int64_t e = 0; e < g.V; ++e)
            vc->at((size_t)(j * g.V + e)) = f[(size_t)rV].value[(size_t)((tj * g.H + h) * g.V + e)];
        for (int64_t i = 0; i <= j; ++i)
            Ac->at((size_t)(j * g.chunk + i)) =
                f[(size_t)rA].value[(size_t)((tj * g.H + h) * g.chunk + i)];
    }
}

/* One chunk's forward. W, V' and o are [L, V], S1 is [K, V] transposed. */
void chunk_forward(const ScanGeom& g, const std::vector<double>& qc,
                   const std::vector<double>& kc, const std::vector<double>& vc,
                   const std::vector<double>& Ac, const std::vector<double>& gc,
                   const std::vector<double>& bc, const std::vector<double>& S0,
                   std::vector<double>* W, std::vector<double>* Vp, std::vector<double>* o,
                   std::vector<double>* S1) {
    const int64_t K = g.K, V = g.V, L = (int64_t)gc.size();
    W->assign((size_t)(L * V), 0.0);
    Vp->assign((size_t)(L * V), 0.0);
    if (o) o->assign((size_t)(L * V), 0.0);
    S1->assign((size_t)(K * V), 0.0);

    for (int64_t j = 0; j < L; ++j) {
        const double ge = std::exp(gc[(size_t)j]);
        for (int64_t e = 0; e < V; ++e) {
            double acc = 0.0;
            for (int64_t d = 0; d < K; ++d)
                acc += kc[(size_t)(j * K + d)] * S0[(size_t)(d * V + e)];
            W->at((size_t)(j * V + e)) =
                bc[(size_t)j] * (vc[(size_t)(j * V + e)] - ge * acc);
        }
    }
    for (int64_t i = 0; i < L; ++i) {
        for (int64_t j = 0; j <= i; ++j) {
            const double a = Ac[(size_t)(i * g.chunk + j)];
            for (int64_t e = 0; e < V; ++e)
                Vp->at((size_t)(i * V + e)) += a * W->at((size_t)(j * V + e));
        }
    }
    if (o) {
        for (int64_t i = 0; i < L; ++i) {
            const double gi = gc[(size_t)i];
            const double ge = std::exp(gi);
            for (int64_t e = 0; e < V; ++e) {
                double acc = 0.0;
                for (int64_t d = 0; d < K; ++d)
                    acc += qc[(size_t)(i * K + d)] * S0[(size_t)(d * V + e)];
                o->at((size_t)(i * V + e)) = g.scale * ge * acc;
            }
            for (int64_t j = 0; j <= i; ++j) {
                double qk = 0.0;
                for (int64_t d = 0; d < K; ++d)
                    qk += qc[(size_t)(i * K + d)] * kc[(size_t)(j * K + d)];
                const double row = g.scale * std::exp(gi - gc[(size_t)j]) * qk;
                for (int64_t e = 0; e < V; ++e)
                    o->at((size_t)(i * V + e)) += row * Vp->at((size_t)(j * V + e));
            }
        }
    }
    const double glast = gc[(size_t)(L - 1)];
    const double el = std::exp(glast);
    for (int64_t d = 0; d < K; ++d)
        for (int64_t e = 0; e < V; ++e)
            S1->at((size_t)(d * V + e)) = el * S0[(size_t)(d * V + e)];
    for (int64_t j = 0; j < L; ++j) {
        const double gs = std::exp(glast - gc[(size_t)j]);
        for (int64_t d = 0; d < K; ++d) {
            const double gk = gs * kc[(size_t)(j * K + d)];
            for (int64_t e = 0; e < V; ++e)
                S1->at((size_t)(d * V + e)) += gk * Vp->at((size_t)(j * V + e));
        }
    }
}

/* One chunk's backward, docs/adjoints/gdn_scan.md's stages 1-4. dq, dk, dv, dA,
 * dg and dbeta are per-chunk scratch; dS0 is the state cotangent the previous
 * chunk consumes. */
void chunk_backward(const ScanGeom& g, const std::vector<double>& qc,
                    const std::vector<double>& kc, const std::vector<double>& vc,
                    const std::vector<double>& Ac, const std::vector<double>& gc,
                    const std::vector<double>& bc, const std::vector<double>& S0,
                    const std::vector<double>& W, const std::vector<double>& Vp,
                    const std::vector<double>& oc, const std::vector<double>& St1,
                    std::vector<double>* dq, std::vector<double>* dk, std::vector<double>* dv,
                    std::vector<double>* dA, std::vector<double>* dg, std::vector<double>* db,
                    std::vector<double>* dS0) {
    const int64_t K = g.K, V = g.V, L = (int64_t)gc.size();
    dq->assign((size_t)(L * K), 0.0);
    dk->assign((size_t)(L * K), 0.0);
    dv->assign((size_t)(L * V), 0.0);
    dA->assign((size_t)(L * g.chunk), 0.0);
    dg->assign((size_t)L, 0.0);
    db->assign((size_t)L, 0.0);
    dS0->assign((size_t)(K * V), 0.0);
    std::vector<double> dVp((size_t)(L * V), 0.0);
    std::vector<double> dW((size_t)(L * V), 0.0);

    const double glast = gc[(size_t)(L - 1)];
    const double el = std::exp(glast);

    /* Stage 1: S1 = e^{g_last} S0 + sum_j e^{g_last-g_j} k_j (x) V'[j]. */
    double gminuend = 0.0;
    for (int64_t d = 0; d < K; ++d)
        for (int64_t e = 0; e < V; ++e) {
            const size_t at = (size_t)(d * V + e);
            dS0->at(at) += el * St1[at];
            gminuend += St1[at] * (el * S0[at]);
        }
    dg->at((size_t)(L - 1)) += gminuend;
    for (int64_t j = 0; j < L; ++j) {
        const double gs = std::exp(glast - gc[(size_t)j]);
        double gw = 0.0;
        for (int64_t d = 0; d < K; ++d) {
            double w = 0.0;
            for (int64_t e = 0; e < V; ++e)
                w += gs * St1[(size_t)(d * V + e)] * Vp[(size_t)(j * V + e)];
            dk->at((size_t)(j * K + d)) += w;
            gw += w * kc[(size_t)(j * K + d)];
            for (int64_t e = 0; e < V; ++e)
                dVp[(size_t)(j * V + e)] +=
                    gs * St1[(size_t)(d * V + e)] * kc[(size_t)(j * K + d)];
        }
        dg->at((size_t)j) -= gw;
        dg->at((size_t)(L - 1)) += gw;
    }

    /* Stage 2: o's state read and the intra-chunk q.k / V' path. */
    std::vector<double> s0oi((size_t)K, 0.0);
    for (int64_t i = 0; i < L; ++i) {
        const double gi = gc[(size_t)i];
        const double egi = std::exp(gi);
        s0oi.assign((size_t)K, 0.0);
        for (int64_t d = 0; d < K; ++d)
            for (int64_t e = 0; e < V; ++e)
                s0oi[(size_t)d] += S0[(size_t)(d * V + e)] * oc[(size_t)(i * V + e)];
        double gstate = 0.0;
        for (int64_t d = 0; d < K; ++d) {
            const double qd = qc[(size_t)(i * K + d)];
            dq->at((size_t)(i * K + d)) += g.scale * egi * s0oi[(size_t)d];
            gstate += qd * s0oi[(size_t)d];
            for (int64_t e = 0; e < V; ++e)
                dS0->at((size_t)(d * V + e)) +=
                    g.scale * egi * qd * oc[(size_t)(i * V + e)];
        }
        dg->at((size_t)i) += g.scale * egi * gstate;
        for (int64_t j = 0; j <= i; ++j) {
            double qk = 0.0;
            for (int64_t d = 0; d < K; ++d)
                qk += qc[(size_t)(i * K + d)] * kc[(size_t)(j * K + d)];
            double vo = 0.0;
            for (int64_t e = 0; e < V; ++e)
                vo += Vp[(size_t)(j * V + e)] * oc[(size_t)(i * V + e)];
            const double phi = g.scale * std::exp(gi - gc[(size_t)j]);
            for (int64_t d = 0; d < K; ++d) {
                dq->at((size_t)(i * K + d)) += phi * vo * kc[(size_t)(j * K + d)];
                dk->at((size_t)(j * K + d)) += phi * vo * qc[(size_t)(i * K + d)];
            }
            for (int64_t e = 0; e < V; ++e)
                dVp[(size_t)(j * V + e)] += phi * qk * oc[(size_t)(i * V + e)];
            const double gpair = phi * qk * vo;
            dg->at((size_t)j) -= gpair;
            dg->at((size_t)i) += gpair;
        }
    }

    /* Stage 3: V' = A W, lower triangle only. */
    for (int64_t i = 0; i < L; ++i)
        for (int64_t j = 0; j <= i; ++j) {
            double s = 0.0;
            for (int64_t e = 0; e < V; ++e)
                s += dVp[(size_t)(i * V + e)] * W[(size_t)(j * V + e)];
            dA->at((size_t)(i * g.chunk + j)) += s;
        }
    for (int64_t j = 0; j < L; ++j)
        for (int64_t e = 0; e < V; ++e) {
            double s = 0.0;
            for (int64_t i = j; i < L; ++i)
                s += Ac[(size_t)(i * g.chunk + j)] * dVp[(size_t)(i * V + e)];
            dW[(size_t)(j * V + e)] = s;
        }

    /* Stage 4: W[j] = beta_j (v_j - e^{g_j} S0^T k_j). */
    for (int64_t j = 0; j < L; ++j) {
        const double egj = std::exp(gc[(size_t)j]);
        std::vector<double> sk2((size_t)V, 0.0);
        for (int64_t d = 0; d < K; ++d)
            for (int64_t e = 0; e < V; ++e)
                sk2[(size_t)e] += S0[(size_t)(d * V + e)] * kc[(size_t)(j * K + d)];
        double gterm = 0.0;
        for (int64_t e = 0; e < V; ++e) {
            const double r = dW[(size_t)(j * V + e)];
            db->at((size_t)j) += r * (vc[(size_t)(j * V + e)] - egj * sk2[(size_t)e]);
            dv->at((size_t)(j * V + e)) += bc[(size_t)j] * r;
            gterm += r * sk2[(size_t)e];
        }
        dg->at((size_t)j) -= bc[(size_t)j] * egj * gterm;
        const double be = bc[(size_t)j] * egj;
        for (int64_t d = 0; d < K; ++d) {
            for (int64_t e = 0; e < V; ++e) {
                const double r = dW[(size_t)(j * V + e)];
                dk->at((size_t)(j * K + d)) -= be * S0[(size_t)(d * V + e)] * r;
                dS0->at((size_t)(d * V + e)) -= be * kc[(size_t)(j * K + d)] * r;
            }
        }
    }
}

/* Twin: gdn_chunk_scan's forward in f64. The entering state is read from h0's
 * block for the slot `state_idx` names, the zero state a prefill starts from. */
void gdn_chunk_scan_twin(const OpRecord& rec, const std::vector<AdValue>& op,
                         std::vector<AdValue>* out) {
    ScanGeom g;
    if (!scan_geom(rec, op, &g) || rec.operands.size() < (size_t)kHt + 1) return;
    AdValue o = write_value(rec, kO);
    AdValue ht = write_value(rec, kHt);
    if (o.value.size() < (size_t)(g.T * g.H * g.V) ||
        ht.value.size() < (size_t)(g.ht_slots * g.H * g.V * g.K))
        return;

    std::vector<double> S0, S1, qc, kc, vc, Ac, gc, bc, W, Vp, orow;
    for (int64_t n = 0; n < g.N; ++n) {
        const int64_t bos = (int64_t)op[(size_t)rCu].value[(size_t)n];
        const int64_t slen = (int64_t)op[(size_t)rCu].value[(size_t)(n + 1)] - bos;
        if (slen <= 0) continue;
        const int64_t slot = state_slot(rec, op, rSidx, n);
        if (slot < 0 || slot >= g.ht_slots) continue;
        for (int64_t h = 0; h < g.H; ++h) {
            load_state(g, op[(size_t)rH0], slot, h, &S0);
            for (int64_t c0 = 0; c0 < slen; c0 += g.chunk) {
                const int64_t L = (c0 + g.chunk < slen) ? g.chunk : slen - c0;
                load_chunk(g, op, bos, h, c0, L, &qc, &kc, &vc, &Ac, &gc, &bc);
                chunk_forward(g, qc, kc, vc, Ac, gc, bc, S0, &W, &Vp, &orow, &S1);
                for (int64_t i = 0; i < L; ++i)
                    for (int64_t e = 0; e < g.V; ++e)
                        o.value[(size_t)(((bos + c0 + i) * g.H + h) * g.V + e)] =
                            orow[(size_t)(i * g.V + e)];
                S0.swap(S1);
            }
            const int64_t per = g.V * g.K;
            const int64_t base = (slot * g.H + h) * per;
            if (base >= 0 && base + per <= (int64_t)ht.value.size())
                for (int64_t e = 0; e < g.V; ++e)
                    for (int64_t d = 0; d < g.K; ++d)
                        ht.value[(size_t)(base + e * g.K + d)] =
                            S0[(size_t)(d * g.V + e)];
        }
    }
    out->clear();
    out->push_back(std::move(o));
    out->push_back(std::move(ht));
}

/* Host adjoint for gdn_chunk_scan, in f64. The state cotangent is per head and
 * flows from the last chunk to the first; h0, cu and state_idx are frozen, so
 * the cotangent leaving the first chunk is dropped. */
void gdn_chunk_scan_host_adjoint(const AdjointCall& call) {
    const std::vector<AdValue>& f = call.fwd;
    ScanGeom g;
    if (!call.dc || !scan_geom(call.rec, f, &g) || call.cot.size() < (size_t)wScanWrites) return;
    if (call.dc->size() < (size_t)kHt + 1) return;
    const AdValue& oc_all = call.cot[wO];
    const AdValue& st_all = call.cot[wHt];

    AdValue& dq = call.dc->at(kQ);
    AdValue& dk = call.dc->at(kK);
    AdValue& dv = call.dc->at(kV);
    AdValue& dA = call.dc->at(kA);
    AdValue& dg = call.dc->at(kG);
    AdValue& db = call.dc->at(kBeta);

    std::vector<double> S0, S1, qc, kc, vc, Ac, gc, bc, W, Vp, orow;
    std::vector<double> dqc, dkc, dvc, dAc, dgc, dbc, dS0, St1;
    std::vector<std::vector<double>> S0s;
    for (int64_t n = 0; n < g.N; ++n) {
        const int64_t bos = (int64_t)f[(size_t)rCu].value[(size_t)n];
        const int64_t slen = (int64_t)f[(size_t)rCu].value[(size_t)(n + 1)] - bos;
        if (slen <= 0) continue;
        const int64_t slot = state_slot(call.rec, f, kSidx, n);
        if (slot < 0) continue;
        for (int64_t h = 0; h < g.H; ++h) {
            /* Every chunk's entering state, one head at a time. The adjoint
             * needs them; the forward operands alone do not carry them. */
            S0s.clear();
            load_state(g, f[(size_t)rH0], slot, h, &S0);
            for (int64_t c0 = 0; c0 < slen; c0 += g.chunk) {
                const int64_t L = (c0 + g.chunk < slen) ? g.chunk : slen - c0;
                S0s.push_back(S0);
                load_chunk(g, f, bos, h, c0, L, &qc, &kc, &vc, &Ac, &gc, &bc);
                chunk_forward(g, qc, kc, vc, Ac, gc, bc, S0, &W, &Vp, nullptr, &S1);
                S0.swap(S1);
            }

            St1.assign((size_t)(g.K * g.V), 0.0);
            const int64_t per = g.V * g.K;
            const int64_t sbase = (slot * g.H + h) * per;
            if (sbase >= 0 && sbase + per <= (int64_t)st_all.value.size())
                for (int64_t e = 0; e < g.V; ++e)
                    for (int64_t d = 0; d < g.K; ++d)
                        St1[(size_t)(d * g.V + e)] = st_all.value[(size_t)(sbase + e * g.K + d)];

            const int64_t nchunks = (int64_t)S0s.size();
            for (int64_t ci = nchunks - 1; ci >= 0; --ci) {
                const int64_t c0 = ci * g.chunk;
                const int64_t L = (c0 + g.chunk < slen) ? g.chunk : slen - c0;
                load_chunk(g, f, bos, h, c0, L, &qc, &kc, &vc, &Ac, &gc, &bc);
                chunk_forward(g, qc, kc, vc, Ac, gc, bc, S0s[(size_t)ci], &W, &Vp, nullptr, &S1);
                orow.assign((size_t)(L * g.V), 0.0);
                for (int64_t i = 0; i < L; ++i)
                    for (int64_t e = 0; e < g.V; ++e)
                        orow[(size_t)(i * g.V + e)] =
                            oc_all.value[(size_t)(((bos + c0 + i) * g.H + h) * g.V + e)];
                chunk_backward(g, qc, kc, vc, Ac, gc, bc, S0s[(size_t)ci], W, Vp, orow, St1,
                               &dqc, &dkc, &dvc, &dAc, &dgc, &dbc, &dS0);
                for (int64_t j = 0; j < L; ++j) {
                    const int64_t tj = bos + c0 + j;
                    for (int64_t d = 0; d < g.K; ++d) {
                        const size_t at = (size_t)((tj * g.Hg + h / g.grp) * g.K + d);
                        dq.value[at] += dqc[(size_t)(j * g.K + d)];
                        dk.value[at] += dkc[(size_t)(j * g.K + d)];
                    }
                    for (int64_t e = 0; e < g.V; ++e)
                        dv.value[(size_t)((tj * g.H + h) * g.V + e)] +=
                            dvc[(size_t)(j * g.V + e)];
                    for (int64_t i = 0; i <= j; ++i)
                        dA.value[(size_t)((tj * g.H + h) * g.chunk + i)] +=
                            dAc[(size_t)(j * g.chunk + i)];
                    dg.value[(size_t)(tj * g.H + h)] += dgc[(size_t)j];
                    db.value[(size_t)(tj * g.H + h)] += dbc[(size_t)j];
                }
                St1.swap(dS0);
            }
            /* The first chunk's dS0 is h0's, and h0 is a frozen zero state. */
        }
    }
}

}  // namespace

R97_REGISTER_TWIN("gdn_chunk_scan", gdn_chunk_scan_twin)
R97_REGISTER_ADJOINT_F("gdn_chunk_scan", host, gdn_chunk_scan_host_adjoint, kH0, kCu, kSidx)

}  // namespace r97
