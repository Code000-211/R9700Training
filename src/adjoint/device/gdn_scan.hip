// src/adjoint/device/gdn_scan.hip -- the device adjoint of gdn_chunk_scan.
//
// The chunk scan's state cotangent is one [K, V] matrix per value head, and it is carried from the
// last chunk of a sequence to the first: over 4096 tokens that is 64 steps of f32 accumulation, so
// the cotangent lives in f32 in global memory and is never written to a bf16 buffer between chunks.
// Everything else in the kernel is f32 too -- the forward values the backward re-derives, the
// per-chunk gradients before they are rounded once -- and only the gradients that leave the kernel
// are bf16, the dtype the card's operands have.
//
// The adjoint needs the state that enters every chunk, which the forward does not hand over. One
// block per (sequence, value head) replays that sequence's chunks forward, writing each entering
// state to scratch, and then walks the chunks backward with the cotangent, so a head's replay is
// 4 MiB at 4096 tokens (f32, 64 chunks of 128x128) instead of the f64 the host row keeps. See
// docs/kernels/gdn_scan.md for the measurements and the memory.
//
// The reverse pass follows docs/adjoints/gdn_scan.md stage for stage: the state cotangent's
// update, the state's own contribution to o, the intra-chunk attention, and the W/V' pair the
// forward builds per chunk. h0, cu and state_idx are frozen, so the cotangent leaving the first
// chunk is dropped, exactly as the host row drops it.
#include "r97/adjoint.h"

#include "rad_device.h"
#include "scratch.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <variant>
#include <vector>

namespace r97 {
namespace {

/* gdn_chunk_scan's schema positions, in schema order (see docs/adjoints/gdn_scan.md). */
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

/* The read operands that come before the writes, at their schema positions: `fwd` is every
 * operand where the record puts it, so a read the schema lists after the writes -- state_idx,
 * position 10 -- is read at that position rather than here. */
enum : int { rQ = 0, rK = 1, rV = 2, rA = 3, rG = 4, rBeta = 5, rH0 = 6, rCu = 7 };

/* The twin's writes, in the order it produces them: o then ht. */
enum : int { wO = 0, wHt = 1, wScanWrites = 2 };

__host__ __device__ inline float bf16_to_f32(unsigned short h) {
    unsigned int u = (unsigned int)h << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

__host__ __device__ inline unsigned short f32_to_bf16(float f) {
    unsigned int u;
    std::memcpy(&u, &f, sizeof u);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x0040u);  // NaN
    const unsigned int lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (unsigned short)(u >> 16);
}

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return shape.empty() ? 0 : n;
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

/* What the scan's shapes resolve to, read the way the host row reads them. */
struct GdnGeom {
    int64_t K = 0;
    int64_t V = 0;
    int64_t chunk = 0;
    double scale = 1.0;
    int64_t N = 0;
    int64_t T = 0;
    int64_t Hg = 0;
    int64_t H = 0;
    int64_t h0_slots = 0;
    int64_t ht_slots = 0;
};

bool gdn_geom(const OpRecord& rec, const std::vector<AdValue>& f, GdnGeom* g) {
    if (rec.operands.size() < (size_t)kHt + 1 || f.size() <= (size_t)rCu) return false;
    const std::vector<int64_t>& qs = rec.operands[kQ].shape;
    const std::vector<int64_t>& vs = rec.operands[kV].shape;
    g->K = int_param(rec, "head_k", qs.empty() ? 0 : qs.back());
    g->V = int_param(rec, "head_v", vs.empty() ? 0 : vs.back());
    g->chunk = int_param(rec, "chunk", rec.operands[kA].shape.empty()
                                             ? 0
                                             : rec.operands[kA].shape.back());
    if (g->K <= 0 || g->V <= 0 || g->chunk <= 0) return false;
    /* An off-tape record carries no `scale`: the architecture issues 1/sqrt(head_k). */
    g->scale = param_f64(rec, "scale", 1.0 / std::sqrt((double)g->K));

    const AdValue& cu = f[(size_t)rCu];
    g->N = (int64_t)cu.value.size() - 1;
    if (g->N <= 0) return false;
    g->T = (int64_t)cu.value.back();
    if (g->T <= 0) return false;
    /* The head counts come from the operands' element counts, the way the host row takes them. */
    const int64_t qn = numel(qs), vn = numel(vs), kn = numel(rec.operands[kK].shape);
    if (qn <= 0 || vn <= 0 || qn % (g->T * g->K) != 0) return false;
    if (vn % (g->T * g->V) != 0) return false;
    g->Hg = qn / (g->T * g->K);
    g->H = vn / (g->T * g->V);
    if (g->Hg <= 0 || g->H <= 0 || g->H % g->Hg != 0) return false;
    if (kn < qn) return false;

    /* The state's slots, and the per-token widths the kernel indexes by hand. */
    const std::vector<int64_t>& h0s = rec.operands[kH0].shape;
    const std::vector<int64_t>& hts = rec.operands[kHt].shape;
    g->h0_slots = h0s.empty() ? g->N : h0s.front();
    g->ht_slots = hts.empty() ? g->N : hts.front();
    if (g->h0_slots <= 0 || g->ht_slots <= 0) return false;
    const bool has_state_idx = rec.operands.size() > (size_t)kSidx;
    if (!has_state_idx && (g->h0_slots < g->N || g->ht_slots < g->N)) return false;
    if (numel(h0s) != g->h0_slots * g->H * g->V * g->K) return false;
    if (numel(hts) != g->ht_slots * g->H * g->V * g->K) return false;

    if (numel(rec.operands[kA].shape) != g->T * g->H * g->chunk) return false;
    if (numel(rec.operands[kG].shape) != g->T * g->H) return false;
    if (numel(rec.operands[kBeta].shape) != g->T * g->H) return false;
    if (numel(rec.operands[kO].shape) != g->T * g->H * g->V) return false;

    /* The values the kernel uploads, at the element counts it walks. */
    if ((int64_t)f[(size_t)rQ].value.size() != qn) return false;
    if ((int64_t)f[(size_t)rK].value.size() < qn) return false;
    if ((int64_t)f[(size_t)rV].value.size() < vn) return false;
    if ((int64_t)f[(size_t)rA].value.size() < g->T * g->H * g->chunk) return false;
    if ((int64_t)f[(size_t)rG].value.size() < g->T * g->H) return false;
    if ((int64_t)f[(size_t)rBeta].value.size() < g->T * g->H) return false;
    if ((int64_t)f[(size_t)rH0].value.size() < g->h0_slots * g->H * g->V * g->K) return false;
    return true;
}

int64_t state_slot(const OpRecord& rec, const std::vector<AdValue>& f, int64_t n) {
    if (rec.operands.size() <= (size_t)kSidx) return n;
    if (f.size() <= (size_t)kSidx) return n;
    const std::vector<double>& s = f[(size_t)kSidx].value;
    if ((int64_t)s.size() <= n) return n;
    return (int64_t)s[(size_t)n];
}

RadStream adjoint_stream() {
    static RadStream s = nullptr;
    if (!s) {
        if (rad_dev_count() <= 0) return nullptr;
        if (rad_stream_create(&s, 0) != 0) s = nullptr;
    }
    return s;
}

/* The kernel's constant geometry. Passed by value: every block reads it many times. */
struct GdnDims {
    int k;
    int v;
    int hk;
    int hv;
    int chunk;
    int ncmax;
    int h0_slots;
    int ht_slots;
    float scale;
};

/* The per-block scratch pool: the replay's states, the state cotangent's two buffers, and the
 * per-chunk working set. The host and the kernel lay it out the same way; this is the one place
 * either counts it. */
int64_t pool_floats(const GdnDims& p) {
    const int64_t K = p.k, V = p.v, L = p.chunk;
    return (int64_t)p.ncmax * K * V     // states: the entering state of every chunk
           + 2 * K * V                  // St1 and dS0
           + 7 * L * V                  // W, Vp, dVp, dW, vc, oc, dvc
           + 4 * L * K                  // qc, kc, dqc, dkc
           + 5 * L * L                  // A, dA, qk, vo, phi
           + 6 * L                      // gc, bc, dgc, dbc, gs, eg
           + V + K;                     // sk2, s0oi
}

/* A sum across the block. Every thread takes part, so the barrier inside is the whole block's. */
__device__ inline float block_reduce(float x, float* red) {
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    red[tid] = x;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

/* One block per (sequence, value head): replay the sequence forward, then walk it backward. The
 * two phases share the per-chunk buffers, so the chunk's operands are loaded twice rather than
 * kept; a chunk of operands is a few tens of KiB against the state's 64 KiB. */
__global__ void __launch_bounds__(128) gdn_scan_adjoint_kernel(
    const unsigned short* __restrict__ qsrc, const unsigned short* __restrict__ ksrc,
    const unsigned short* __restrict__ vsrc, const unsigned short* __restrict__ Asrc,
    const float* __restrict__ gsrc, const float* __restrict__ betasrc,
    const float* __restrict__ h0, const int* __restrict__ cu, const int* __restrict__ slots,
    const unsigned short* __restrict__ ocot, const float* __restrict__ htc,
    float* __restrict__ dqout, float* __restrict__ dkout,
    unsigned short* __restrict__ dvout, unsigned short* __restrict__ dAout,
    float* __restrict__ dgout, float* __restrict__ dbout,
    float* __restrict__ pool, int64_t stride, GdnDims p) {
    const int K = p.k, V = p.v, L_max = p.chunk, hv = p.hv, hk = p.hk;
    const int blk = (int)blockIdx.x;
    const int n = blk / hv;
    const int h = blk - n * hv;
    const int len = cu[n + 1] - cu[n];
    if (len <= 0) return;
    const int sl = slots ? slots[n] : n;
    if (sl < 0) return;
    const int nchunks = (len + L_max - 1) / L_max;
    const int kh = h / (hv / hk);
    const int tok0 = cu[n];

    float* base = pool + (int64_t)blk * stride;
    float* states = base;
    float* st1 = states + (int64_t)p.ncmax * K * V;
    float* ds0 = st1 + K * V;
    float* W = ds0 + K * V;
    float* Vp = W + L_max * V;
    float* dVp = Vp + L_max * V;
    float* dW = dVp + L_max * V;
    float* vc = dW + L_max * V;
    float* oc = vc + L_max * V;
    float* dvc = oc + L_max * V;
    float* qc = dvc + L_max * V;
    float* kc = qc + L_max * K;
    float* dqc = kc + L_max * K;
    float* dkc = dqc + L_max * K;
    float* Ac = dkc + L_max * K;
    float* dAc = Ac + L_max * L_max;
    float* qkb = dAc + L_max * L_max;
    float* vo = qkb + L_max * L_max;
    float* phi = vo + L_max * L_max;
    float* gc = phi + L_max * L_max;
    float* bc = gc + L_max;
    float* dgc = bc + L_max;
    float* dbc = dgc + L_max;
    float* gs = dbc + L_max;
    float* eg = gs + L_max;
    float* sk2 = eg + L_max;
    float* s0oi = sk2 + V;

    __shared__ float red[128];
    const int tid = threadIdx.x;
    const int nt = blockDim.x;

    /* The chunk's operands, at f32: A only where the forward reads it (j <= i), the rest zero. */
    auto load_chunk = [&](int c, int L, int t0) {
        for (int t = tid; t < L * K; t += nt) {
            const int j = t / K, d = t - j * K;
            const int at = ((t0 + j) * hk + kh) * K + d;
            qc[t] = bf16_to_f32(qsrc[at]);
            kc[t] = bf16_to_f32(ksrc[at]);
        }
        for (int t = tid; t < L * V; t += nt) {
            const int j = t / V, e = t - j * V;
            const int64_t at = ((int64_t)(t0 + j) * hv + h) * V + e;
            vc[t] = bf16_to_f32(vsrc[at]);
            oc[t] = bf16_to_f32(ocot[at]);
        }
        for (int t = tid; t < L * L_max; t += nt) {
            const int i = t / L_max, j = t - i * L_max;
            float a = 0.f;
            if (j <= i) a = bf16_to_f32(Asrc[((int64_t)(t0 + i) * hv + h) * L_max + j]);
            Ac[t] = a;
        }
        for (int j = tid; j < L; j += nt) {
            const int64_t at = (int64_t)(t0 + j) * hv + h;
            gc[j] = gsrc[at];
            bc[j] = betasrc[at];
        }
        __syncthreads();
        for (int j = tid; j < L; j += nt) {
            eg[j] = expf(gc[j]);
            gs[j] = expf(gc[L - 1] - gc[j]);
        }
        __syncthreads();
    };

    /* The forward's per-chunk pair the adjoint re-derives from the entering state: W is v with the
     * state's contribution removed, V' is W under the intra-chunk (unit lower triangular) A. */
    auto forward_chunk = [&](const float* S, int L) {
        for (int t = tid; t < L * V; t += nt) {
            const int j = t / V, e = t - j * V;
            float acc = 0.f;
            for (int d = 0; d < K; ++d) acc += kc[j * K + d] * S[d * V + e];
            W[t] = bc[j] * (vc[t] - eg[j] * acc);
        }
        __syncthreads();
        for (int t = tid; t < L * V; t += nt) {
            const int i = t / V, e = t - i * V;
            float acc = 0.f;
            for (int j = 0; j <= i; ++j) acc += Ac[i * L_max + j] * W[j * V + e];
            Vp[t] = acc;
        }
        __syncthreads();
    };

    /* Phase 1: the state entering every chunk. The first chunk's is h0's block for the slot; h0 is
     * frozen and the host row drops the cotangent that would leave it. */
    {
        const int64_t per = (int64_t)V * K;
        const int64_t hb = ((int64_t)sl * hv + h) * per;
        const bool inb = sl < p.h0_slots && hb >= 0 && hb + per <= (int64_t)p.h0_slots * hv * per;
        for (int t = tid; t < K * V; t += nt) {
            const int d = t / V, e = t - d * V;
            states[t] = inb ? h0[hb + e * K + d] : 0.f;
        }
        __syncthreads();
        for (int c = 0; c < nchunks; ++c) {
            const int c0 = c * L_max;
            const int L = (c0 + L_max < len) ? L_max : len - c0;
            const float* S = states + (int64_t)c * K * V;
            load_chunk(c, L, tok0 + c0);
            forward_chunk(S, L);
            if (c + 1 < nchunks) {
                float* Sn = states + (int64_t)(c + 1) * K * V;
                const float el = eg[L - 1];
                for (int t = tid; t < K * V; t += nt) {
                    const int d = t / V, e = t - d * V;
                    float acc = el * S[t];
                    for (int j = 0; j < L; ++j) acc += gs[j] * kc[j * K + d] * Vp[j * V + e];
                    Sn[t] = acc;
                }
                __syncthreads();
            }
        }
    }

    /* The cotangent entering the last chunk: ht's block for the slot, at f32, e-major like the
     * state the forward writes. */
    {
        const int64_t per = (int64_t)V * K;
        const int64_t hb = ((int64_t)sl * hv + h) * per;
        const bool inb = htc && sl < p.ht_slots && hb >= 0 &&
                         hb + per <= (int64_t)p.ht_slots * hv * per;
        for (int t = tid; t < K * V; t += nt) {
            const int d = t / V, e = t - d * V;
            st1[t] = inb ? htc[hb + e * K + d] : 0.f;
        }
    }
    __syncthreads();

    /* Phase 2: the chunks backward, one f32 state cotangent carried between them. */
    for (int c = nchunks - 1; c >= 0; --c) {
        const int c0 = c * L_max;
        const int L = (c0 + L_max < len) ? L_max : len - c0;
        const int t0 = tok0 + c0;
        const float* S = states + (int64_t)c * K * V;

        for (int t = tid; t < L * K; t += nt) {
            dqc[t] = 0.f;
            dkc[t] = 0.f;
        }
        for (int t = tid; t < L * V; t += nt) {
            dVp[t] = 0.f;
            dvc[t] = 0.f;
        }
        for (int t = tid; t < L * L_max; t += nt) dAc[t] = 0.f;
        for (int j = tid; j < L; j += nt) {
            dgc[j] = 0.f;
            dbc[j] = 0.f;
        }
        __syncthreads();

        load_chunk(c, L, t0);
        forward_chunk(S, L);

        /* Stage 1: the state cotangent's own update, and what leaves through k and V'. */
        {
            const float el = eg[L - 1];
            float part = 0.f;
            for (int t = tid; t < K * V; t += nt) {
                ds0[t] = el * st1[t];
                part += st1[t] * (el * S[t]);
            }
            const float gmin = block_reduce(part, red);
            if (tid == 0) dgc[L - 1] += gmin;
            /* dk[j][d] = gs_j * (St1 . V'_j)[d], and the same product against k gives dg's
             * log-derivative. One thread per d: the reduction runs over the rows of St1. */
            for (int j = 0; j < L; ++j) {
                float pw = 0.f;
                for (int d = tid; d < K; d += nt) {
                    float acc = 0.f;
                    for (int e = 0; e < V; ++e) acc += st1[d * V + e] * Vp[j * V + e];
                    const float w = gs[j] * acc;
                    dkc[j * K + d] += w;
                    pw += w * kc[j * K + d];
                }
                const float gw = block_reduce(pw, red);
                if (tid == 0) {
                    dgc[j] -= gw;
                    dgc[L - 1] += gw;
                }
            }
            for (int t = tid; t < L * V; t += nt) {
                const int j = t / V, e = t - j * V;
                float acc = 0.f;
                for (int d = 0; d < K; ++d) acc += st1[d * V + e] * kc[j * K + d];
                dVp[t] += gs[j] * acc;
            }
            __syncthreads();
        }

        /* Stage 2: o's state term, then the intra-chunk attention's terms. */
        for (int i = 0; i < L; ++i) {
            const float sc = (float)p.scale * eg[i];
            for (int d = tid; d < K; d += nt) {
                float acc = 0.f;
                for (int e = 0; e < V; ++e) acc += S[d * V + e] * oc[i * V + e];
                s0oi[d] = acc;
            }
            __syncthreads();
            float p3 = 0.f;
            for (int d = tid; d < K; d += nt) {
                const float qd = qc[i * K + d];
                dqc[i * K + d] += sc * s0oi[d];
                p3 += qd * s0oi[d];
            }
            const float gstate = block_reduce(p3, red);
            if (tid == 0) dgc[i] += sc * gstate;
            for (int t = tid; t < K * V; t += nt) {
                const int d = t / V, e = t - d * V;
                ds0[t] += sc * qc[i * K + d] * oc[i * V + e];
            }
            __syncthreads();
        }

        for (int t = tid; t < L * L; t += nt) {
            const int i = t / L, j = t - i * L;
            phi[i * L_max + j] = (j <= i) ? (float)p.scale * expf(gc[i] - gc[j]) : 0.f;
            float a = 0.f, b = 0.f;
            if (j <= i) {
                for (int d = 0; d < K; ++d) a += qc[i * K + d] * kc[j * K + d];
                for (int e = 0; e < V; ++e) b += Vp[j * V + e] * oc[i * V + e];
            }
            qkb[i * L_max + j] = a;
            vo[i * L_max + j] = b;
        }
        __syncthreads();
        for (int t = tid; t < L * K; t += nt) {
            const int i = t / K, d = t - i * K;
            float acc = 0.f;
            for (int j = 0; j <= i; ++j)
                acc += phi[i * L_max + j] * vo[i * L_max + j] * kc[j * K + d];
            dqc[t] += acc;
        }
        for (int t = tid; t < L * K; t += nt) {
            const int j = t / K, d = t - j * K;
            float acc = 0.f;
            for (int i = j; i < L; ++i)
                acc += phi[i * L_max + j] * vo[i * L_max + j] * qc[i * K + d];
            dkc[t] += acc;
        }
        for (int t = tid; t < L * V; t += nt) {
            const int j = t / V, e = t - j * V;
            float acc = 0.f;
            for (int i = j; i < L; ++i)
                acc += phi[i * L_max + j] * qkb[i * L_max + j] * oc[i * V + e];
            dVp[t] += acc;
        }
        for (int j = tid; j < L; j += nt) {
            float s = 0.f;
            for (int i = j; i < L; ++i)
                s += phi[i * L_max + j] * qkb[i * L_max + j] * vo[i * L_max + j];
            dgc[j] -= s;
        }
        __syncthreads();
        for (int i = tid; i < L; i += nt) {
            float s = 0.f;
            for (int j = 0; j <= i; ++j)
                s += phi[i * L_max + j] * qkb[i * L_max + j] * vo[i * L_max + j];
            dgc[i] += s;
        }
        __syncthreads();

        /* Stage 3: through A into W's cotangent. */
        for (int t = tid; t < L * L; t += nt) {
            const int i = t / L, j = t - i * L;
            float acc = 0.f;
            if (j <= i) {
                for (int e = 0; e < V; ++e) acc += dVp[i * V + e] * W[j * V + e];
            }
            dAc[i * L_max + j] = acc;
        }
        __syncthreads();
        for (int t = tid; t < L * V; t += nt) {
            const int j = t / V, e = t - j * V;
            float acc = 0.f;
            for (int i = j; i < L; ++i) acc += Ac[i * L_max + j] * dVp[i * V + e];
            dW[t] = acc;
        }
        __syncthreads();

        /* Stage 4: W's cotangent back to v, beta, the gate and the state. */
        for (int j = 0; j < L; ++j) {
            const float egj = eg[j];
            const float bj = bc[j];
            for (int e = tid; e < V; e += nt) {
                float s = 0.f;
                for (int d = 0; d < K; ++d) s += S[d * V + e] * kc[j * K + d];
                sk2[e] = s;
            }
            __syncthreads();
            float pa = 0.f, pb = 0.f;
            for (int e = tid; e < V; e += nt) {
                const float r = dW[j * V + e];
                dvc[j * V + e] = bj * r;
                pa += r * (vc[j * V + e] - egj * sk2[e]);
                pb += r * sk2[e];
            }
            const float sdb = block_reduce(pa, red);
            const float sgt = block_reduce(pb, red);
            if (tid == 0) {
                dbc[j] += sdb;
                dgc[j] -= bj * egj * sgt;
            }
            const float be = bj * egj;
            for (int d = tid; d < K; d += nt) {
                float acc = 0.f;
                for (int e = 0; e < V; ++e) acc += S[d * V + e] * dW[j * V + e];
                dkc[j * K + d] -= be * acc;
            }
            for (int t = tid; t < K * V; t += nt) {
                const int d = t / V, e = t - d * V;
                ds0[t] -= be * kc[j * K + d] * dW[j * V + e];
            }
            __syncthreads();
        }

        /* The chunk's gradients leave the kernel rounded once; the token rows past this chunk's
         * length belong to the next sequence and are that block's to write. */
        for (int j = 0; j < L; ++j) {
            const int tj = t0 + j;
            for (int d = tid; d < K; d += nt) {
                const int at = ((tj * hk + kh) * K + d);
                atomicAdd(&dqout[at], dqc[j * K + d]);
                atomicAdd(&dkout[at], dkc[j * K + d]);
            }
            for (int e = tid; e < V; e += nt)
                dvout[((int64_t)tj * hv + h) * V + e] = f32_to_bf16(dvc[j * V + e]);
            for (int i = tid; i < L_max; i += nt)
                dAout[((int64_t)tj * hv + h) * L_max + i] =
                    f32_to_bf16(i <= j ? dAc[j * L_max + i] : 0.f);
            if (tid == 0) {
                dgout[(int64_t)tj * hv + h] = dgc[j];
                dbout[(int64_t)tj * hv + h] = dbc[j];
            }
        }
        __syncthreads();

        /* The first chunk's dS0 is h0's, and h0 is a frozen zero state. */
        float* tmp = st1;
        st1 = ds0;
        ds0 = tmp;
    }
}

/* Adds a bf16 gradient buffer into the tape's f64 cotangent: the tape's entries are added to, never
 * overwritten, because other ops share them. */
bool download_add_bf16(const void* dev, int64_t n, AdValue* dst, RadStream s) {
    if (n <= 0) return true;
    std::vector<unsigned short> h((size_t)n);
    if (rad_memcpy_async(h.data(), dev, n * (int64_t)sizeof(unsigned short), s) != 0) return false;
    if (rad_stream_sync(s) != 0) return false;
    if ((int64_t)dst->value.size() < n) return false;
    for (int64_t i = 0; i < n; ++i)
        dst->value[(size_t)i] += (double)bf16_to_f32(h[(size_t)i]);
    return true;
}

bool download_add_f32(const void* dev, int64_t n, AdValue* dst, RadStream s) {
    if (n <= 0) return true;
    std::vector<float> h((size_t)n);
    if (rad_memcpy_async(h.data(), dev, n * (int64_t)sizeof(float), s) != 0) return false;
    if (rad_stream_sync(s) != 0) return false;
    if ((int64_t)dst->value.size() < n) return false;
    for (int64_t i = 0; i < n; ++i) dst->value[(size_t)i] += (double)h[(size_t)i];
    return true;
}

/* Device adjoint for gdn_chunk_scan: one kernel, one block per (sequence, value head). */
void gdn_chunk_scan_device_adjoint(const AdjointCall& call) {
    const std::vector<AdValue>& f = call.fwd;
    GdnGeom g;
    if (!call.dc || !gdn_geom(call.rec, f, &g) || call.cot.size() < (size_t)wScanWrites) return;
    if (call.dc->size() < (size_t)kHt + 1) return;
    if (g.K > INT32_MAX || g.V > INT32_MAX || g.chunk > INT32_MAX) return;
    if (g.Hg > INT32_MAX || g.H > INT32_MAX) return;

    const AdValue& oc_all = call.cot[(size_t)wO];
    const AdValue& ht_all = call.cot[(size_t)wHt];
    if ((int64_t)oc_all.value.size() < g.T * g.H * g.V) return;
    if ((int64_t)ht_all.value.size() < g.ht_slots * g.H * g.V * g.K) return;

    /* The sequences' boundaries and slots, and the longest sequence's chunk count: every block's
     * pool is laid out for it, so one stride serves the whole grid. */
    std::vector<int> h_cu((size_t)g.N + 1), h_slot((size_t)g.N);
    int64_t ncmax = 0;
    for (int64_t n = 0; n <= g.N; ++n) {
        const int64_t v = (int64_t)f[(size_t)rCu].value[(size_t)n];
        if (v < 0 || v > g.T) return;
        if (n > 0 && v < (int64_t)h_cu[(size_t)n - 1]) return;
        h_cu[(size_t)n] = (int)v;
    }
    for (int64_t n = 0; n < g.N; ++n) {
        const int64_t slen = (int64_t)h_cu[(size_t)n + 1] - h_cu[(size_t)n];
        if (slen < 0) return;
        ncmax = std::max(ncmax, (slen + g.chunk - 1) / g.chunk);
        const int64_t slot = state_slot(call.rec, f, n);
        if (slot < 0 || slot > INT32_MAX) return;
        h_slot[(size_t)n] = (int)slot;
    }
    if (ncmax == 0) return;

    GdnDims dims{};
    dims.k = (int)g.K;
    dims.v = (int)g.V;
    dims.hk = (int)g.Hg;
    dims.hv = (int)g.H;
    dims.chunk = (int)g.chunk;
    dims.ncmax = (int)ncmax;
    dims.h0_slots = (int)g.h0_slots;
    dims.ht_slots = (int)g.ht_slots;
    dims.scale = (float)g.scale;

    RadStream stream = adjoint_stream();
    if (!stream) return;

    const int64_t qk_n = g.T * g.Hg * g.K;
    const int64_t v_n = g.T * g.H * g.V;
    const int64_t a_n = g.T * g.H * g.chunk;
    const int64_t gb_n = g.T * g.H;
    const int64_t h0_n = g.h0_slots * g.H * g.V * g.K;
    const int64_t ht_n = g.ht_slots * g.H * g.V * g.K;
    const int64_t nblocks = g.N * g.H;
    const int64_t stride = pool_floats(dims);

    /* The uploads are the operands at the dtype the card's op holds them in: q, k, v, A and o's
     * cotangent bf16, then g, beta, h0 and ht's cotangent at f32. */
    device::ScratchBlock d_q(qk_n * 2), d_k(qk_n * 2), d_v(v_n * 2), d_A(a_n * 2);
    device::ScratchBlock d_g(gb_n * 4), d_beta(gb_n * 4), d_h0(h0_n * 4);
    device::ScratchBlock d_cu(((int64_t)g.N + 1) * 4), d_slot(g.N * 4);
    device::ScratchBlock d_oc(v_n * 2), d_ht(ht_n * 4);
    device::ScratchBlock d_dq(qk_n * 4), d_dk(qk_n * 4), d_dv(v_n * 2), d_dA(a_n * 2);
    device::ScratchBlock d_dg(gb_n * 4), d_db(gb_n * 4);
    device::ScratchBlock d_pool(nblocks * stride * 4);
    if (!d_q || !d_k || !d_v || !d_A || !d_g || !d_beta || !d_h0 || !d_cu || !d_slot || !d_oc) {
        return;
    }
    if (!d_ht || !d_dq || !d_dk || !d_dv || !d_dA || !d_dg || !d_db || !d_pool) return;

    {
        std::vector<unsigned short> hq((size_t)qk_n), hk((size_t)qk_n), hv((size_t)v_n),
            ha((size_t)a_n), hoc((size_t)v_n);
        const std::vector<double>& fq = f[(size_t)rQ].value;
        const std::vector<double>& fk = f[(size_t)rK].value;
        const std::vector<double>& fv = f[(size_t)rV].value;
        const std::vector<double>& fa = f[(size_t)rA].value;
        for (int64_t i = 0; i < qk_n; ++i) {
            hq[(size_t)i] = f32_to_bf16((float)fq[(size_t)i]);
            hk[(size_t)i] = f32_to_bf16((float)fk[(size_t)i]);
        }
        for (int64_t i = 0; i < v_n; ++i) {
            hv[(size_t)i] = f32_to_bf16((float)fv[(size_t)i]);
            hoc[(size_t)i] = f32_to_bf16((float)oc_all.value[(size_t)i]);
        }
        for (int64_t i = 0; i < a_n; ++i) ha[(size_t)i] = f32_to_bf16((float)fa[(size_t)i]);
        std::vector<float> hg((size_t)gb_n), hb((size_t)gb_n);
        for (int64_t i = 0; i < gb_n; ++i) {
            hg[(size_t)i] = (float)f[(size_t)rG].value[(size_t)i];
            hb[(size_t)i] = (float)f[(size_t)rBeta].value[(size_t)i];
        }
        std::vector<float> h0((size_t)h0_n), ht((size_t)ht_n);
        for (int64_t i = 0; i < h0_n; ++i) h0[(size_t)i] = (float)f[(size_t)rH0].value[(size_t)i];
        for (int64_t i = 0; i < ht_n; ++i) ht[(size_t)i] = (float)ht_all.value[(size_t)i];

        if (rad_memcpy_async(d_q.p, hq.data(), qk_n * 2, stream) != 0) return;
        if (rad_memcpy_async(d_k.p, hk.data(), qk_n * 2, stream) != 0) return;
        if (rad_memcpy_async(d_v.p, hv.data(), v_n * 2, stream) != 0) return;
        if (rad_memcpy_async(d_A.p, ha.data(), a_n * 2, stream) != 0) return;
        if (rad_memcpy_async(d_g.p, hg.data(), gb_n * 4, stream) != 0) return;
        if (rad_memcpy_async(d_beta.p, hb.data(), gb_n * 4, stream) != 0) return;
        if (rad_memcpy_async(d_h0.p, h0.data(), h0_n * 4, stream) != 0) return;
        if (rad_memcpy_async(d_ht.p, ht.data(), ht_n * 4, stream) != 0) return;
        if (rad_memcpy_async(d_oc.p, hoc.data(), v_n * 2, stream) != 0) return;
        if (rad_memcpy_async(d_cu.p, h_cu.data(), ((int64_t)g.N + 1) * 4, stream) != 0) return;
        if (rad_memcpy_async(d_slot.p, h_slot.data(), g.N * 4, stream) != 0) return;
    }

    /* The q and k accumulators start at zero: the value heads of a key head add into them. */
    {
        std::vector<float> zeros((size_t)qk_n, 0.f);
        if (rad_memcpy_async(d_dq.p, zeros.data(), qk_n * 4, stream) != 0) return;
        if (rad_memcpy_async(d_dk.p, zeros.data(), qk_n * 4, stream) != 0) return;
    }

    hipLaunchKernelGGL(gdn_scan_adjoint_kernel, dim3((unsigned int)nblocks), dim3(128), 0,
                       (hipStream_t)stream, (const unsigned short*)d_q.p,
                       (const unsigned short*)d_k.p, (const unsigned short*)d_v.p,
                       (const unsigned short*)d_A.p, (const float*)d_g.p, (const float*)d_beta.p,
                       (const float*)d_h0.p, (const int*)d_cu.p, (const int*)d_slot.p,
                       (const unsigned short*)d_oc.p, (const float*)d_ht.p, (float*)d_dq.p,
                       (float*)d_dk.p, (unsigned short*)d_dv.p, (unsigned short*)d_dA.p,
                       (float*)d_dg.p, (float*)d_db.p, (float*)d_pool.p, stride, dims);
    if (rad_stream_sync(stream) != 0) return;

    if (!download_add_f32(d_dq.p, qk_n, &call.dc->at(kQ), stream)) return;
    if (!download_add_f32(d_dk.p, qk_n, &call.dc->at(kK), stream)) return;
    if (!download_add_bf16(d_dv.p, v_n, &call.dc->at(kV), stream)) return;
    if (!download_add_bf16(d_dA.p, a_n, &call.dc->at(kA), stream)) return;
    if (!download_add_f32(d_dg.p, gb_n, &call.dc->at(kG), stream)) return;
    if (!download_add_f32(d_db.p, gb_n, &call.dc->at(kBeta), stream)) return;
}

}  // namespace

R97_REGISTER_ADJOINT_F("gdn_chunk_scan", device, gdn_chunk_scan_device_adjoint, kH0, kCu, kSidx)

}  // namespace r97
