// src/adjoint/device/attention.hip -- the card's adjoint of paged causal attention, and of the
// fusion the card issues in its place, `attn_paged_gate_quant`.
//
// The forward never stores the tokens-by-tokens weights, so neither does the adjoint: it recomputes
// them in tiles, one warp to a (query row, head) and one warp to a pool row.
//
//   1. norm_dq_kernel, a warp to a (query row, head), three passes over the row's visible keys:
//      the first keeps the softmax's max and sum in registers, the second the row's mean value
//      cotangent under its own weights, delta = sum_j p_j G_j with G_j = v_descale * <ycot, V_j>,
//      and the third the query's cotangent,
//
//          ds_j = p_j (G_j - delta),
//          dq   += ds_j * scale * k_descale * K_j.
//
//      The row's m, z and delta go to scratch for the pool pass: delta is a mean over the row's
//      keys, and the pool's rows are accumulated key by key, so it cannot be recomputed there.
//
//   2. pool_dkv_kernel, a warp to a (sequence, key, key-value head): streams the query rows that
//      read the key and accumulates the pool row's cotangents in registers,
//
//          dK_j += ds_j * scale * k_descale * q,   dV_j += p_j * v_descale * ycot,
//
//      over the gqa query heads of that key-value head, and adds them with atomicAdd once.
//
// Both accumulate in f32 and round once, to the operand's dtype, on the way out.
//
// `attn_paged_gate_quant` is the forward card's fusion of the attention, the output gate
// (y = bf16(out * sigmoid(gate))) and the quantising producer for the output projection
// (codes = quant(y), scale = amax/127). Its adjoint is gate_ste_kernel, which turns the two output
// cotangents into the attention's out cotangent and the gate's, followed by the attention adjoint:
//
//     ytot    = ycot(out) + live * codes_cot / scale      (the straight-through step)
//     d(out)  = ytot * sigmoid(gate)                      (the gate's mul, into the attention)
//     d(gate) = ytot * y * (1 - sigmoid(gate))            (its sigmoid, with a = y / sigmoid(gate))
//
// The reference for both is the host adjoint of `attn_paged` (docs/adjoints/attention.md) and, for
// the fusion, that adjoint composed with the host adjoints of `quant_act_i8g`, `mul` and
// `sigmoid` (docs/adjoints/quant.md, docs/adjoints/elementwise.md). The doc for this kernel, with
// its measured times and its scratch, is docs/kernels/attention.md.
//
// The contract gives a device adjoint f64 operands, not device buffers, so this call owns every
// device buffer it touches. They are scratch, freed before it returns; the harness reports the
// high-water mark.
#include "r97/adjoint.h"

#include "scratch.h"

#include "rad_device.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace r97 {
namespace {

// bf16 is stored as the high half of an f32, with round-to-nearest-even on the way down: the same
// pair as src/adjoint/device/add.hip, so the file needs neither __bf16 nor a HIP bf16 type.
__host__ __device__ inline float bf16_to_f32(unsigned short b) {
    const unsigned int u = (unsigned int)b << 16;
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

/* ---------------------------------------------------------------- reading a record */

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (const int64_t d : shape) n *= d;
    return n;
}

/* The record's slot for a named operand, or its size when there is none. */
size_t slot_of(const OpRecord& rec, const char* name) {
    for (size_t i = 0; i < rec.operands.size(); ++i)
        if (rec.operands[i].buffer == name) return i;
    return rec.operands.size();
}

size_t read_count(const OpRecord& rec) {
    size_t n = 0;
    for (const Operand& o : rec.operands)
        if (!o.written) ++n;
    return n;
}

/* Reads sit at their schema positions 0..n_reads-1; an optional one the record leaves out sits past
 * the reads rather than taking a written operand's place. */
bool listed(const std::vector<AdValue>& fwd, size_t n_reads, size_t i) {
    return i < n_reads && i < fwd.size() && !fwd[i].shape.empty() &&
           fwd[i].value.size() == (size_t)std::max<int64_t>(numel(fwd[i].shape), 0);
}

/* A parameter as an integer, or a string that is one; a value the caller may set to zero is a
 * value, not a missing parameter. */
int64_t param_int(const OpRecord& rec, const char* name, int64_t fallback) {
    for (const OpParam& p : rec.params) {
        if (p.first != name) continue;
        if (const auto* i = std::get_if<int64_t>(&p.second)) return *i;
        if (const auto* s = std::get_if<std::string>(&p.second)) {
            char* end = nullptr;
            const long long v = std::strtoll(s->c_str(), &end, 10);
            if (end && end != s->c_str() && *end == '\0') return v;
        }
        return fallback;
    }
    return fallback;
}

double param_f64(const OpRecord& rec, const char* name, double fallback) {
    for (const OpParam& p : rec.params) {
        if (p.first != name) continue;
        if (const auto* i = std::get_if<int64_t>(&p.second)) return (double)*i;
        if (const auto* s = std::get_if<std::string>(&p.second)) {
            char* end = nullptr;
            const double v = std::strtod(s->c_str(), &end);
            if (end && end != s->c_str() && *end == '\0' && std::isfinite(v)) return v;
        }
        return fallback;
    }
    return fallback;
}

int64_t int_at(const AdValue& a, size_t i) {
    return i < a.value.size() ? (int64_t)std::llround(a.value[i]) : 0;
}

double f64_at(const AdValue& a, size_t i, double fallback) {
    return i < a.value.size() ? a.value[i] : fallback;
}

/* The attention's read operands, in schema order. The fusion reads the same seven and then the
 * gate; `fwd` holds the reads in schema order and then the writes, so the gate is the read at 7 and
 * the caller's cotangents -- out, codes, scale -- start at n_reads. `dc`, in contrast, is indexed
 * as the record is, so the gate's cotangent is found by name. */
enum Slot : size_t {
    kQ = 0,
    kKv = 1,
    kBlockTable = 2,
    kSeqUsed = 3,
    kKDescale = 4,
    kVDescale = 5,
    kCuSeq = 6,
    kGate = 7,  // the fusion's eighth read
};

/* One call's geometry, resolved the way the host adjoint resolves it (src/adjoint/host/attention.cpp
 * holds the reference and the reason for each rule). */
struct Geom {
    const AdValue* q = nullptr;
    const AdValue* kv = nullptr;

    int64_t head_dim = 0;
    int64_t kv_heads = 0;
    int64_t q_heads = 0;
    int64_t gqa = 1;
    int64_t block_size = 0;
    int64_t n_seq = 0;
    int64_t rows = 0;
    int64_t max_blocks = 0;
    int64_t jcap = 0;
    int64_t n_blocks = 0;
    int64_t causal = 1;
    int64_t window = 0;
    double scale = 1.0;

    std::vector<int64_t> ctx;    // [n_seq] the sequence's key count (seqused)
    std::vector<int64_t> cu;     // [n_seq + 1] the query rows of each sequence
    std::vector<int64_t> table;  // [n_seq][max_blocks]
    std::vector<double> k_des;   // [n_seq][kv head]
    std::vector<double> v_des;
};

/* False when the call is not one this kernel can compute, so the adjoint adds no cotangent; an
 * adjoint must not throw. */
bool resolve(const OpRecord& rec, const std::vector<AdValue>& fwd, Geom* c) {
    const size_t n_reads = read_count(rec);
    if (!listed(fwd, n_reads, kQ) || !listed(fwd, n_reads, kKv) ||
        !listed(fwd, n_reads, kBlockTable) || !listed(fwd, n_reads, kSeqUsed))
        return false;
    c->q = &fwd[kQ];
    c->kv = &fwd[kKv];
    if (c->q->shape.size() < 3 || c->kv->shape.size() != 4) return false;
    c->head_dim = c->q->shape.back();
    c->kv_heads = c->kv->shape[1];
    if (c->head_dim <= 0 || c->kv_heads <= 0) return false;
    if (c->kv->shape[3] < 2 * c->head_dim) return false;
    if (param_int(rec, "head_dim", c->head_dim) != c->head_dim) return false;

    const int64_t heads = c->q->shape[c->q->shape.size() - 2];
    c->gqa = param_int(rec, "gqa", 0);
    if (c->gqa <= 0) c->gqa = (heads % c->kv_heads == 0) ? heads / c->kv_heads : 1;
    c->q_heads = c->gqa * c->kv_heads;
    if (c->q_heads != heads) return false;
    if (c->q->value.size() % (size_t)(c->q_heads * c->head_dim) != 0) return false;
    c->rows = (int64_t)c->q->value.size() / (c->q_heads * c->head_dim);
    if (c->rows <= 0) return false;

    c->block_size = param_int(rec, "block_size", c->kv->shape[2]);
    if (c->block_size <= 0 || c->kv->shape[2] != c->block_size) return false;
    c->n_blocks = c->kv->shape[0];
    if (c->n_blocks <= 0) return false;

    const AdValue& su = fwd[kSeqUsed];
    c->n_seq = (int64_t)su.value.size();
    if (c->n_seq <= 0 || c->n_seq > c->rows) return false;
    c->ctx.resize((size_t)c->n_seq);
    for (int64_t s = 0; s < c->n_seq; ++s) {
        c->ctx[(size_t)s] = int_at(su, (size_t)s);
        if (c->ctx[(size_t)s] <= 0) return false;
    }

    const AdValue& bt = fwd[kBlockTable];
    c->max_blocks = bt.shape.empty() ? (int64_t)bt.value.size() / c->n_seq : bt.shape.back();
    if (c->max_blocks <= 0 || bt.value.size() < (size_t)(c->n_seq * c->max_blocks)) return false;
    c->jcap = c->max_blocks * c->block_size;
    c->table.resize((size_t)(c->n_seq * c->max_blocks));
    for (size_t i = 0; i < c->table.size(); ++i) c->table[i] = int_at(bt, i);

    c->cu.assign((size_t)c->n_seq + 1, 0);
    if (listed(fwd, n_reads, kCuSeq)) {
        const AdValue& cus = fwd[kCuSeq];
        if ((int64_t)cus.value.size() != c->n_seq + 1) return false;
        for (int64_t s = 0; s <= c->n_seq; ++s) c->cu[(size_t)s] = int_at(cus, (size_t)s);
        if (c->cu[0] != 0 || c->cu[(size_t)c->n_seq] != c->rows) return false;
        for (int64_t s = 0; s < c->n_seq; ++s)
            if (c->cu[(size_t)s + 1] < c->cu[(size_t)s]) return false;
    } else {
        if (c->rows % c->n_seq != 0) return false;
        for (int64_t s = 0; s <= c->n_seq; ++s) c->cu[(size_t)s] = (s * c->rows) / c->n_seq;
    }

    c->k_des.assign((size_t)(c->n_seq * c->kv_heads), 1.0);
    c->v_des.assign((size_t)(c->n_seq * c->kv_heads), 1.0);
    if (listed(fwd, n_reads, kKDescale))
        for (size_t i = 0; i < c->k_des.size(); ++i)
            c->k_des[i] = f64_at(fwd[kKDescale], i, 1.0);
    if (listed(fwd, n_reads, kVDescale))
        for (size_t i = 0; i < c->v_des.size(); ++i)
            c->v_des[i] = f64_at(fwd[kVDescale], i, 1.0);

    c->causal = param_int(rec, "causal", 1) != 0 ? 1 : 0;
    c->window = param_int(rec, "window", 0);
    c->scale = param_f64(rec, "scale", 1.0 / std::sqrt((double)c->head_dim));
    /* One warp holds a head row's dims in registers: head_dim / 32 of them per lane. */
    return c->head_dim <= 512;
}

/* ---------------------------------------------------------------- the host's index of the call */

/* What the kernels index: the pool block of every key, the query rows that see it, and, for the
 * query pass, the row's sequence, position and last key. It is the host adjoint's own key_slot
 * loop over the same operands, so the two cannot disagree about the mask. */
struct Index {
    std::vector<int32_t> row_seq;   // [rows]
    std::vector<int32_t> row_jlo;   // [rows] its first visible key
    std::vector<int32_t> row_jhi;   // [rows] one past its last
    std::vector<int32_t> keyblk;    // [n_seq][jcap], -1 where the key is hidden or unmapped
    std::vector<int32_t> qlo;       // [n_seq][jcap] the first query row that sees it
    std::vector<int32_t> qhi;       // [n_seq][jcap] one past the last
    std::vector<float> kdes;        // [n_seq][kv head]
    std::vector<float> vdes;
};

void build_index(const Geom& c, Index* ix) {
    const int64_t rows = c.rows;
    ix->row_seq.resize((size_t)rows);
    ix->row_jlo.resize((size_t)rows);
    ix->row_jhi.resize((size_t)rows);
    for (int64_t s = 0; s < c.n_seq; ++s) {
        const int64_t first = c.cu[(size_t)s];
        const int64_t qlen = c.cu[(size_t)s + 1] - first;
        const int64_t pos0 = c.ctx[(size_t)s] - qlen;
        const int64_t last = std::min(c.ctx[(size_t)s], c.jcap);
        for (int64_t i = 0; i < qlen; ++i) {
            const int64_t row = first + i;
            const int64_t pos = pos0 + i;
            /* The row's keys, as one interval: causal keeps j <= pos, a window keeps the last
             * `window` of those, and a non-causal call keeps |pos - j| < window. */
            int64_t jlo = 0, jhi = last;
            if (c.causal) {
                jhi = std::min(jhi, pos + 1);
                if (c.window > 0) jlo = std::max<int64_t>(0, pos - c.window + 1);
            } else if (c.window > 0) {
                jlo = std::max<int64_t>(0, pos - c.window + 1);
                jhi = std::min(jhi, pos + c.window);
            }
            ix->row_seq[(size_t)row] = (int32_t)s;
            ix->row_jlo[(size_t)row] = (int32_t)jlo;
            ix->row_jhi[(size_t)row] = (int32_t)std::max<int64_t>(jlo, jhi);
        }
    }

    const size_t keys = (size_t)(c.n_seq * c.jcap);
    ix->keyblk.assign(keys, -1);
    ix->qlo.assign(keys, 0);
    ix->qhi.assign(keys, 0);
    for (int64_t s = 0; s < c.n_seq; ++s) {
        const int64_t qlen = c.cu[(size_t)s + 1] - c.cu[(size_t)s];
        const int64_t pos0 = c.ctx[(size_t)s] - qlen;
        for (int64_t j = 0; j < c.jcap; ++j) {
            const size_t kk = (size_t)(s * c.jcap + j);
            int32_t blk = -1;
            if (j < c.ctx[(size_t)s]) {
                const int64_t b = c.table[(size_t)(s * c.max_blocks + j / c.block_size)];
                if (b >= 0 && b < c.n_blocks) blk = (int32_t)b;
            }
            ix->keyblk[kk] = blk;
            /* The rows of the sequence that see key j are one interval: causal keeps pos >= j, a
             * window keeps pos - j < window, and a non-causal call keeps |pos - j| < window. */
            int64_t lo = 0, hi = qlen;
            if (blk >= 0) {
                if (c.causal) {
                    lo = j - pos0;
                    if (c.window > 0) hi = j - pos0 + c.window;
                } else if (c.window > 0) {
                    lo = j - pos0 - c.window + 1;
                    hi = j - pos0 + c.window;
                }
            }
            lo = std::max<int64_t>(lo, 0);
            hi = std::min<int64_t>(hi, qlen);
            if (blk < 0) hi = lo;
            ix->qlo[kk] = (int32_t)lo;
            ix->qhi[kk] = (int32_t)hi;
        }
    }

    ix->kdes.assign((size_t)(c.n_seq * c.kv_heads), 1.0f);
    ix->vdes.assign((size_t)(c.n_seq * c.kv_heads), 1.0f);
    for (size_t i = 0; i < ix->kdes.size(); ++i) {
        ix->kdes[i] = (float)c.k_des[i];
        ix->vdes[i] = (float)c.v_des[i];
    }
}

/* ---------------------------------------------------------------- the kernels */

constexpr int kWarp = 32;
constexpr int kLane = 16;          // head_dim up to 512 through one warp
constexpr int kWarpsPerBlock = 8;  // 256 threads
constexpr float kTop = 127.0f;     // the quantiser family's grid, docs/adjoints/quant.md

/* The whole warp's sum, in every lane. */
__device__ __forceinline__ float warp_all(float v) {
#pragma unroll
    for (int off = kWarp / 2; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffffffffffull, v, off);
    return __shfl_sync(0xffffffffffffffffull, v, 0);
}

/* Lane 0's value, in every lane: for a total that every lane already holds in full. */
__device__ __forceinline__ float warp_one(float v) {
    return __shfl_sync(0xffffffffffffffffull, v, 0);
}

__device__ __forceinline__ int64_t pool_row(int64_t blk, int64_t kh, int64_t j, int64_t kv_heads,
                                            int64_t block_size, int64_t head_dim) {
    return ((blk * kv_heads + kh) * block_size + (j % block_size)) * (2 * head_dim);
}

/* The query's side: one warp to a (query row, head). Writes the row's m, z and delta for the pool
 * pass, and then the query's cotangent. */
__global__ void norm_dq_kernel(const float* __restrict__ q, const float* __restrict__ kv,
                               const int32_t* __restrict__ table,
                               const int32_t* __restrict__ row_seq,
                               const int32_t* __restrict__ row_jlo,
                               const int32_t* __restrict__ row_jhi,
                               const float* __restrict__ kdes, const float* __restrict__ vdes,
                               const float* __restrict__ ycot,
                               int64_t rows, int64_t q_heads, int64_t kv_heads, int64_t gqa,
                               int64_t head_dim, int64_t block_size, int64_t max_blocks,
                               int64_t n_blocks, float scale, float* __restrict__ m_out,
                               float* __restrict__ l_out, float* __restrict__ delta_out,
                               uint16_t* __restrict__ dq) {
    const int64_t w = (int64_t)blockIdx.x * (blockDim.x / kWarp) + (threadIdx.x / kWarp);
    if (w >= rows * q_heads) return;
    const int lane = (int)threadIdx.x % kWarp;
    const int64_t row = w / q_heads;
    const int64_t h = w % q_heads;
    const int64_t kh = h / gqa;
    const int64_t s = row_seq[row];
    const int64_t jlo = row_jlo[row], jend = row_jhi[row];
    const int64_t toff = s * max_blocks;
    const float kds = kdes[s * kv_heads + kh];
    const float vds = vdes[s * kv_heads + kh];
    const int64_t base = (row * q_heads + h) * head_dim;

    float qv[kLane], dqv[kLane];
    int nt = 0;
#pragma unroll
    for (int t = 0; t < kLane; ++t) {
        const int64_t d = lane + (int64_t)t * kWarp;
        if (d < head_dim) {
            qv[t] = q[base + d];
            dqv[t] = 0.0f;
            nt = t + 1;
        }
    }

    float mx = -INFINITY, z = 0.0f;
    for (int64_t j = jlo; j < jend; ++j) {
        const int32_t blk = table[toff + j / block_size];
        if (blk < 0 || blk >= n_blocks) continue;
        const int64_t slot = pool_row(blk, kh, j, kv_heads, block_size, head_dim);
        float acc = 0.0f;
        for (int t = 0; t < nt; ++t) acc += qv[t] * kv[slot + lane + (int64_t)t * kWarp];
        const float sc = warp_all(acc) * scale * kds;
        const float nm = fmaxf(mx, sc);
        z = z * __expf(mx - nm) + __expf(sc - nm);
        mx = nm;
    }
    if (!(z > 0.0f)) {
        mx = 0.0f;
        z = 1.0f;
    }

    /* delta = sum_j p_j G_j: the row's value cotangent under its own weights. A pass of its own,
     * so that the query's cotangent is the one the host's recomputation gives and not a value read
     * back out of a rounded buffer -- with the gate's cotangent on the value (the fusion below
     * multiplies it by one over the gate), those roundings would decide the result. */
    float gsum = 0.0f;
    for (int64_t j = jlo; j < jend; ++j) {
        const int32_t blk = table[toff + j / block_size];
        if (blk < 0 || blk >= n_blocks) continue;
        const int64_t slot = pool_row(blk, kh, j, kv_heads, block_size, head_dim);
        float acc = 0.0f, gacc = 0.0f;
        for (int t = 0; t < nt; ++t) {
            const int64_t d = lane + (int64_t)t * kWarp;
            acc += qv[t] * kv[slot + d];
            gacc += ycot[base + d] * kv[slot + head_dim + d];
        }
        const float sc = warp_all(acc) * scale * kds;
        const float gj = warp_all(gacc) * vds;
        gsum += __expf(sc - mx) / z * gj;
    }
    const float delta = warp_one(gsum);
    if (lane == 0) {
        m_out[w] = mx;
        l_out[w] = z;
        delta_out[w] = delta;
    }

    const float pscale = scale * kds;
    for (int64_t j = jlo; j < jend; ++j) {
        const int32_t blk = table[toff + j / block_size];
        if (blk < 0 || blk >= n_blocks) continue;
        const int64_t slot = pool_row(blk, kh, j, kv_heads, block_size, head_dim);
        float acc = 0.0f, gacc = 0.0f;
        for (int t = 0; t < nt; ++t) {
            const int64_t d = lane + (int64_t)t * kWarp;
            acc += qv[t] * kv[slot + d];
            gacc += ycot[base + d] * kv[slot + head_dim + d];
        }
        const float sc = warp_all(acc) * pscale;
        const float gj = warp_all(gacc) * vds;
        const float p = __expf(sc - mx) / z;
        const float ds = p * (gj - delta) * pscale;
        for (int t = 0; t < nt; ++t) dqv[t] += ds * kv[slot + lane + (int64_t)t * kWarp];
    }
    for (int t = 0; t < nt; ++t)
        dq[base + lane + (int64_t)t * kWarp] = f32_to_bf16(dqv[t]);
}

/* The pool's side: one warp to a (sequence, key, key-value head), over the query rows that read
 * the key. */
__global__ void pool_dkv_kernel(const float* __restrict__ q, const float* __restrict__ kv,
                                const float* __restrict__ ycot,
                                const int32_t* __restrict__ keyblk,
                                const int32_t* __restrict__ qlo,
                                const int32_t* __restrict__ qhi, const int32_t* __restrict__ cu,
                                const float* __restrict__ kdes, const float* __restrict__ vdes,
                                const float* __restrict__ m_in, const float* __restrict__ l_in,
                                const float* __restrict__ delta_in, int64_t q_heads,
                                int64_t kv_heads, int64_t gqa, int64_t head_dim,
                                int64_t block_size, int64_t n_seq, int64_t jcap, float scale,
                                float* __restrict__ dkv) {
    const int64_t w = (int64_t)blockIdx.x * (blockDim.x / kWarp) + (threadIdx.x / kWarp);
    if (w >= n_seq * jcap * kv_heads) return;
    const int lane = (int)threadIdx.x % kWarp;
    const int64_t kh = w % kv_heads;
    const int64_t rest = w / kv_heads;
    const int64_t s = rest / jcap;
    const int64_t j = rest % jcap;
    const int64_t kk = s * jcap + j;
    const int32_t blk = keyblk[kk];
    const int64_t first = qlo[kk], last = qhi[kk];
    if (blk < 0 || last <= first) return;

    const int64_t slot = pool_row(blk, kh, j, kv_heads, block_size, head_dim);
    const float kds = kdes[s * kv_heads + kh];
    const float vds = vdes[s * kv_heads + kh];

    float kr[kLane], vr[kLane], dkr[kLane], dvr[kLane];
    int nt = 0;
#pragma unroll
    for (int t = 0; t < kLane; ++t) {
        const int64_t d = lane + (int64_t)t * kWarp;
        if (d < head_dim) {
            kr[t] = kv[slot + d];
            vr[t] = kv[slot + head_dim + d];
            dkr[t] = 0.0f;
            dvr[t] = 0.0f;
            nt = t + 1;
        }
    }

    const float pscale = scale * kds;
    const int64_t head0 = kh * gqa, head1 = head0 + gqa;
    const int64_t row0 = cu[s];
    for (int64_t i = first; i < last; ++i) {
        const int64_t row = row0 + i;
        for (int64_t h = head0; h < head1; ++h) {
            const int64_t base = (row * q_heads + h) * head_dim;
            const int64_t wh = row * q_heads + h;
            float acc = 0.0f, gacc = 0.0f;
            for (int t = 0; t < nt; ++t) {
                const int64_t d = lane + (int64_t)t * kWarp;
                acc += q[base + d] * kr[t];
                gacc += ycot[base + d] * vr[t];
            }
            const float sc = warp_all(acc) * pscale;
            const float gj = warp_all(gacc) * vds;
            const float p = __expf(sc - m_in[wh]) / l_in[wh];
            const float ds = p * (gj - delta_in[wh]) * pscale;
            for (int t = 0; t < nt; ++t) {
                const int64_t d = lane + (int64_t)t * kWarp;
                dkr[t] += ds * q[base + d];
                dvr[t] += p * vds * ycot[base + d];
            }
        }
    }
    for (int t = 0; t < nt; ++t) {
        const int64_t d = lane + (int64_t)t * kWarp;
        atomicAdd(&dkv[slot + d], dkr[t]);
        atomicAdd(&dkv[slot + head_dim + d], dvr[t]);
    }
}

/* The fusion's two output cotangents, and with them the attention's: the quantiser's
 * straight-through step first (docs/adjoints/quant.md), then the gate's mul and sigmoid. One
 * element to a thread, its group's scale looked up. */
__global__ void gate_ste_kernel(const float* __restrict__ gate, const float* __restrict__ y,
                                const float* __restrict__ ycot,
                                const float* __restrict__ codes_cot,
                                const float* __restrict__ scale, int64_t n, int64_t head_dim,
                                int64_t group, int64_t ngrp, float top,
                                float* __restrict__ dattn, float* __restrict__ dgate) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t row = i / head_dim;
    const int64_t d = i % head_dim;
    const float s = 1.0f / (1.0f + __expf(-gate[i]));
    const float yv = y[i];
    const float sc = scale[row * ngrp + d / group];
    float x = 0.0f;
    /* The clamp's own derivative: zero where it engaged, one inside the grid. A group's largest
     * value sits on the top of the grid, where the quotient is 127 only to within the rounding of
     * the stored scale, so the comparison is made exactly as the host adjoint makes it -- the same
     * f64 quotient of the same stored f32 pair -- and the two sides agree on the boundary
     * element as well (docs/adjoints/quant.md). */
    if (sc > 0.0f && !((double)fabsf(yv) * (1.0 / (double)sc) > (double)top)) x = codes_cot[i] / sc;
    const float tot = ycot[i] + x;
    dattn[i] = tot * s;
    dgate[i] = tot * yv * (1.0f - s);
}

/* ---------------------------------------------------------------- one call's device memory */

/* One stream for every device adjoint call in this process: upload, launch and download share it,
 * so a call is one queue's worth of work. */
RadStream adjoint_stream() {
    static RadStream s = [] {
        RadStream created = nullptr;
        if (rad_stream_create(&created, 0) != 0) return (RadStream) nullptr;
        return created;
    }();
    return s;
}

int64_t grid_of(int64_t warps) { return (warps + kWarpsPerBlock - 1) / kWarpsPerBlock; }

struct Plan {
    std::deque<device::ScratchBlock> owned;
    std::deque<std::vector<float>> staging;  // the upload's host copies, alive until the sync

    float* q = nullptr;
    float* kv = nullptr;
    uint16_t* dq = nullptr;
    float* dkv = nullptr;
    const int32_t* table = nullptr;
    const int32_t* row_seq = nullptr;
    const int32_t* row_jlo = nullptr;
    const int32_t* row_jhi = nullptr;
    const int32_t* keyblk = nullptr;
    const int32_t* qlo = nullptr;
    const int32_t* qhi = nullptr;
    const int32_t* cu = nullptr;
    const float* kdes = nullptr;
    const float* vdes = nullptr;
    float* m = nullptr;
    float* l = nullptr;
    float* delta = nullptr;

    void* take(int64_t bytes) {
        owned.emplace_back(bytes);
        return owned.back().p;
    }
};

bool upload_f32(const std::vector<double>& src, float* dst, Plan* p, RadStream s) {
    p->staging.emplace_back(src.size());
    std::vector<float>& t = p->staging.back();
    for (size_t i = 0; i < src.size(); ++i) t[i] = (float)src[i];
    return rad_memcpy_async(dst, t.data(), (int64_t)(t.size() * sizeof(float)), s) == 0;
}

bool upload_i32(const std::vector<int32_t>& src, const int32_t* dst, RadStream s) {
    return rad_memcpy_async(const_cast<int32_t*>(dst), src.data(),
                            (int64_t)(src.size() * sizeof(int32_t)), s) == 0;
}

/* The pool's cotangent is f32 on the card and rounded once, to the operand's dtype, on the way
 * out; a pool whose dtype this file cannot round (fp8) keeps the f32 value. */
double round_out(double v, const std::string& dtype) {
    if (dtype == "bf16" || dtype == "bfloat16") return bf16_to_f32(f32_to_bf16((float)v));
    return v;
}

/* Every buffer a call writes, and the index the kernels read. The forward's values are uploaded by
 * the caller, which knows which of them the fusion has already put on the card. */
bool prepare(const Geom& c, const Index& ix, Plan* p) {
    const RadStream s = adjoint_stream();
    if (!s) return false;
    const int64_t qnum = (int64_t)c.q->value.size();
    const int64_t kvnum = (int64_t)c.kv->value.size();
    const int64_t heads_rows = c.rows * c.q_heads;
    p->q = (float*)p->take(qnum * (int64_t)sizeof(float));
    p->kv = (float*)p->take(kvnum * (int64_t)sizeof(float));
    p->dq = (uint16_t*)p->take(qnum * (int64_t)sizeof(uint16_t));
    p->dkv = (float*)p->take(kvnum * (int64_t)sizeof(float));
    p->table = (const int32_t*)p->take((int64_t)(c.table.size() * sizeof(int32_t)));
    p->row_seq = (const int32_t*)p->take(c.rows * (int64_t)sizeof(int32_t));
    p->row_jlo = (const int32_t*)p->take(c.rows * (int64_t)sizeof(int32_t));
    p->row_jhi = (const int32_t*)p->take(c.rows * (int64_t)sizeof(int32_t));
    p->keyblk = (const int32_t*)p->take((int64_t)(ix.keyblk.size() * sizeof(int32_t)));
    p->qlo = (const int32_t*)p->take((int64_t)(ix.qlo.size() * sizeof(int32_t)));
    p->qhi = (const int32_t*)p->take((int64_t)(ix.qhi.size() * sizeof(int32_t)));
    p->cu = (const int32_t*)p->take((int64_t)(c.cu.size() * sizeof(int32_t)));
    p->kdes = (const float*)p->take((int64_t)(ix.kdes.size() * sizeof(float)));
    p->vdes = (const float*)p->take((int64_t)(ix.vdes.size() * sizeof(float)));
    p->m = (float*)p->take(heads_rows * (int64_t)sizeof(float));
    p->l = (float*)p->take(heads_rows * (int64_t)sizeof(float));
    p->delta = (float*)p->take(heads_rows * (int64_t)sizeof(float));
    if (!p->q || !p->kv || !p->dq || !p->dkv || !p->table || !p->row_seq || !p->row_jlo ||
        !p->row_jhi || !p->keyblk || !p->qlo || !p->qhi || !p->cu || !p->kdes || !p->vdes ||
        !p->m || !p->l || !p->delta)
        return false;

    std::vector<int32_t> seq(c.rows), jlo(c.rows), jhi(c.rows), cu(c.cu.begin(), c.cu.end());
    std::vector<int32_t> table(c.table.begin(), c.table.end());
    for (int64_t i = 0; i < c.rows; ++i) {
        seq[(size_t)i] = ix.row_seq[(size_t)i];
        jlo[(size_t)i] = ix.row_jlo[(size_t)i];
        jhi[(size_t)i] = ix.row_jhi[(size_t)i];
    }
    if (!upload_i32(table, p->table, s)) return false;
    if (!upload_i32(seq, p->row_seq, s) || !upload_i32(jlo, p->row_jlo, s)) return false;
    if (!upload_i32(jhi, p->row_jhi, s) || !upload_i32(cu, p->cu, s)) return false;
    if (!upload_i32(ix.keyblk, p->keyblk, s)) return false;
    if (!upload_i32(ix.qlo, p->qlo, s) || !upload_i32(ix.qhi, p->qhi, s)) return false;
    if (rad_memcpy_async(const_cast<float*>(p->kdes), ix.kdes.data(),
                         (int64_t)(ix.kdes.size() * sizeof(float)), s) != 0)
        return false;
    if (rad_memcpy_async(const_cast<float*>(p->vdes), ix.vdes.data(),
                         (int64_t)(ix.vdes.size() * sizeof(float)), s) != 0)
        return false;
    /* The pool's cotangent is accumulated with atomics, so it starts at zero. */
    if (rad_memset_async(p->dkv, 0, kvnum * (int64_t)sizeof(float), s) != 0) return false;
    return upload_f32(c.q->value, p->q, p, s) && upload_f32(c.kv->value, p->kv, p, s);
}

void launch(const Geom& c, const Plan& p, const float* d_ycot) {
    const RadStream s = adjoint_stream();
    if (!s) return;
    const int64_t warps_q = c.rows * c.q_heads;
    if (warps_q > 0) {
        hipLaunchKernelGGL(norm_dq_kernel, dim3((unsigned int)grid_of(warps_q)),
                           dim3(kWarpsPerBlock * kWarp), 0, (hipStream_t)s, p.q, p.kv, p.table,
                           p.row_seq, p.row_jlo, p.row_jhi, p.kdes, p.vdes, d_ycot, c.rows,
                           c.q_heads, c.kv_heads, c.gqa, c.head_dim, c.block_size, c.max_blocks,
                           c.n_blocks, (float)c.scale, p.m, p.l, p.delta, p.dq);
    }
    const int64_t warps_kv = c.n_seq * c.jcap * c.kv_heads;
    if (warps_kv > 0) {
        hipLaunchKernelGGL(pool_dkv_kernel, dim3((unsigned int)grid_of(warps_kv)),
                           dim3(kWarpsPerBlock * kWarp), 0, (hipStream_t)s, p.q, p.kv, d_ycot,
                           p.keyblk, p.qlo, p.qhi, p.cu, p.kdes, p.vdes, p.m, p.l, p.delta,
                           c.q_heads, c.kv_heads, c.gqa, c.head_dim, c.block_size, c.n_seq, c.jcap,
                           (float)c.scale, p.dkv);
    }
}

/* Downloads both cotangents and adds them to the walk's. Nothing is written for a pool or a query
 * whose buffer does not match the walk's, and a failure leaves the walk's values alone. */
bool collect(const Geom& c, const Plan& p, const AdjointCall& call) {
    const RadStream s = adjoint_stream();
    if (!s) return false;
    const size_t qnum = call.dc->at(kQ).value.size();
    const size_t kvnum = call.dc->at(kKv).value.size();
    if (qnum != c.q->value.size() || kvnum != c.kv->value.size()) return false;
    std::vector<unsigned short> h_dq(qnum);
    std::vector<float> h_dkv(kvnum);
    if (rad_memcpy_async(h_dq.data(), p.dq, (int64_t)(qnum * sizeof(unsigned short)), s) != 0)
        return false;
    if (rad_memcpy_async(h_dkv.data(), p.dkv, (int64_t)(kvnum * sizeof(float)), s) != 0)
        return false;
    if (rad_stream_sync(s) != 0) return false;

    /* The tape's cotangents are f64 and are added to, never overwritten: other ops share them. */
    AdValue& dq = call.dc->at(kQ);
    AdValue& dkv = call.dc->at(kKv);
    for (size_t i = 0; i < qnum; ++i) dq.value[i] += bf16_to_f32(h_dq[i]);
    for (size_t i = 0; i < kvnum; ++i) dkv.value[i] += round_out(h_dkv[i], dkv.dtype);
    return true;
}

void attention_paged_device_adjoint(const AdjointCall& call) {
    if (call.cot.empty() || !call.dc || call.dc->size() < 2) return;
    Geom c;
    if (!resolve(call.rec, call.fwd, &c)) return;
    const size_t n_reads = read_count(call.rec);
    const size_t w_out = n_reads;
    if (call.fwd.size() <= w_out) return;
    if (call.fwd[w_out].value.size() != c.q->value.size()) return;
    const AdValue& ycot = call.cot[0];
    if (ycot.value.size() != c.q->value.size()) return;

    Index ix;
    build_index(c, &ix);
    Plan p;
    if (!prepare(c, ix, &p)) return;
    const RadStream s = adjoint_stream();
    float* d_ycot = (float*)p.take((int64_t)c.q->value.size() * (int64_t)sizeof(float));
    if (!d_ycot) return;
    if (!upload_f32(ycot.value, d_ycot, &p, s)) return;
    launch(c, p, d_ycot);
    collect(c, p, call);
}

/* The fusion: attention, the output gate and the quantising producer, in that order back to front.
 * Its record's reads are the attention's seven and the gate; its writes are the attention's out
 * (which the gate's mul overwrote with the gated product), the codes and the scale. */
void attention_gate_quant_device_adjoint(const AdjointCall& call) {
    const size_t gate_slot = slot_of(call.rec, "gate");
    if (call.cot.size() < 3 || !call.dc || call.dc->size() <= gate_slot) return;
    Geom c;
    if (!resolve(call.rec, call.fwd, &c)) return;
    const size_t n_reads = read_count(call.rec);
    const size_t w_out = n_reads, w_codes = n_reads + 1, w_scale = n_reads + 2;
    if (!listed(call.fwd, n_reads, kGate) || call.fwd.size() <= w_scale) return;

    const AdValue& gate = call.fwd[kGate];
    const AdValue& y = call.fwd[w_out];
    const AdValue& scales = call.fwd[w_scale];
    const AdValue& ycot = call.cot[0];
    const AdValue& codes_cot = call.cot[1];
    const int64_t n = (int64_t)c.q->value.size();
    const int64_t m = c.rows * c.q_heads;
    if (n <= 0 || (int64_t)y.value.size() != n || (int64_t)ycot.value.size() != n ||
        (int64_t)codes_cot.value.size() != n || (int64_t)gate.value.size() != n)
        return;
    if (scales.value.empty() || (int64_t)scales.value.size() % m != 0) return;
    const int64_t ngrp = (int64_t)scales.value.size() / m;
    int64_t group = param_int(call.rec, "group", 0);
    if (group <= 0 || group > c.head_dim) group = (c.head_dim + ngrp - 1) / ngrp;
    if (group <= 0 || (c.head_dim + group - 1) / group != ngrp) return;

    Index ix;
    build_index(c, &ix);
    Plan p;
    if (!prepare(c, ix, &p)) return;
    const RadStream s = adjoint_stream();
    p.owned.emplace_back(n * (int64_t)sizeof(float));
    float* d_gate = (float*)p.owned.back().p;
    p.owned.emplace_back(n * (int64_t)sizeof(float));
    float* d_y = (float*)p.owned.back().p;
    p.owned.emplace_back(n * (int64_t)sizeof(float));
    float* d_ycot = (float*)p.owned.back().p;
    p.owned.emplace_back(n * (int64_t)sizeof(float));
    float* d_codes_cot = (float*)p.owned.back().p;
    p.owned.emplace_back((int64_t)scales.value.size() * (int64_t)sizeof(float));
    float* d_scale = (float*)p.owned.back().p;
    /* The attention's cotangent of the value (the gate's product) and the gate's. */
    float* d_dattn = (float*)p.take(n * (int64_t)sizeof(float));
    float* d_dgate = (float*)p.take(n * (int64_t)sizeof(float));
    if (!d_gate || !d_y || !d_ycot || !d_codes_cot || !d_scale || !d_dattn || !d_dgate) return;
    if (!upload_f32(gate.value, d_gate, &p, s) || !upload_f32(y.value, d_y, &p, s)) return;
    if (!upload_f32(ycot.value, d_ycot, &p, s)) return;
    if (!upload_f32(codes_cot.value, d_codes_cot, &p, s)) return;
    if (!upload_f32(scales.value, d_scale, &p, s)) return;

    const int64_t grid = (n + 255) / 256;
    hipLaunchKernelGGL(gate_ste_kernel, dim3((unsigned int)grid), dim3(256), 0, (hipStream_t)s,
                       d_gate, d_y, d_ycot, d_codes_cot, d_scale, n, c.head_dim, group, ngrp, kTop,
                       d_dattn, d_dgate);
    launch(c, p, d_dattn);
    if (!collect(c, p, call)) return;
    AdValue& dgate = call.dc->at(gate_slot);
    if (dgate.value.size() != (size_t)n) return;
    std::vector<float> h_dgate((size_t)n);
    if (rad_memcpy_async(h_dgate.data(), d_dgate, n * (int64_t)sizeof(float), s) != 0) return;
    if (rad_stream_sync(s) != 0) return;
    for (size_t i = 0; i < (size_t)n; ++i)
        dgate.value[i] += round_out(h_dgate[i], dgate.dtype);
}

}  // namespace

/* Frozen, as the family's host rows are: the operands the forward reads and does not differentiate
 * -- the block table, the sequence lengths, the two descales and the offsets (2 to 6). The fusion's
 * gate is not among them: it is read, but its cotangent is written into its slot in `dc`. */
R97_REGISTER_ADJOINT_F("attn_paged", device, attention_paged_device_adjoint, 2, 3, 4, 5, 6)
R97_REGISTER_ADJOINT_F("attn_paged_gate_quant", device, attention_gate_quant_device_adjoint, 2, 3,
                       4, 5, 6)

}  // namespace r97
