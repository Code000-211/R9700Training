// src/adjoint/device/qk.hip -- the device adjoint of the gated attention prologue.
//
// The card prepares queries, keys and the gate with one fused op, `qk_norm_rope_gate`, and with
// `qk_norm_rope_gate_kv_store` when the step also writes keys and values into the paged pool
// (third_party/radiance/docs/OPS.md). The host issues the same block unfused -- rmsnorm on the
// queries and the keys, rope, kv_store, and sigmoid then mul for the gate (docs/adjoints/qk.md) --
// so this adjoint is the transpose of that chain, folded into one launch:
//
//   q lanes   q_gate[.][head][0 .. head_dim)    rmsnorm(q_norm, eps, wadd) -> rope -> q_out
//   gate      q_gate[.][head][head_dim ..]      copied to gate_out
//   k lanes   attn_k[.][kv head][0 .. head_dim) rmsnorm(k_norm, eps, wadd) -> rope -> k_out
//                                             -> kv_store -> kv_cache (kv_store form)
//
// A token's q head owns 2*head_dim lanes of the projection: its query lanes, then its gate lanes.
// The projection is `attn_qg`, `attn_k` is the key projection and `attn_v` the value projection, so
// the cotangents this kernel produces are those three, plus one per rotated output.
//
// What it reads and writes (the record's operand order, reads first):
//
//   [0] attn_qg    [tokens, n_head*2*head_dim]      fwd
//   [1] attn_k     [tokens, n_head_kv*head_dim]     fwd
//   [2] cos_sin    [table rows, rot]                fwd, constant
//   [3] positions  [tokens] i32                     fwd, constant
//   [4] q_norm     [head_dim] f32                   fwd, frozen (the gain is not trained)
//   [5] k_norm     [head_dim] f32                   fwd, frozen
//   [6] attn_v     [tokens, n_head_kv*head_dim]     fwd          (kv_store form only)
//   [7] slot_map   [tokens] i32                     fwd, frozen  (kv_store form only)
//   [8] kv_cache   [n_blocks, n_head_kv, block_size, 2*head_dim]  inout, written
//   [9] q          [tokens, n_head*head_dim]        written
//  [10] k          [tokens, n_head_kv*head_dim]     written
//  [11] gate       [tokens, n_head*head_dim]        written
//
// `qk_norm_rope_gate` is the same record without 6, 7 and 8, so its writes are 6, 7 and 8. The
// store's pool rows come back as the cotangent of `kv_cache`; they flow to `attn_k` and `attn_v`
// and not into a cotangent of the pool, because a store is where the pool's cotangent dies: the
// pool is seeded per step by the attention that reads it. Rows the store does not write still pass
// through the pool unchanged, and that identity is not differentiated.
//
// The kernel computes in f32: the tape's f64 operands are read at f32 precision, each token's row
// is reduced in f32, and a slot's cotangent is rounded once to bf16 on the way out, like
// src/adjoint/device/add.hip. bf16 is carried as unsigned short, so this file needs neither
// __bf16 nor a HIP bf16 type.
//
// `qk_norm_rope_gate_kv_store_streamed` is refused in training. Its pool is quantised -- fp8 rows
// plus a per-block residue row that the streaming reader reconstructs from (`res`, `host` operands
// of its own) -- while training keeps the pool in bf16, so there is no cotangent to hand it and no
// host chain to compare against. The row says so instead of leaving the caller with a missing
// adjoint.
//
// The contract gives a device adjoint f64 operands, not device buffers, so this call owns every
// device buffer it touches. They are scratch, freed before it returns; the harness reports the
// high-water mark. tests/gpu/adjoint_qk.cpp holds the kernel to the composition of the host
// adjoints of its parts within the bf16 tolerance docs/kernels/README.md states, and
// docs/kernels/qk.md carries the measurement.
#include "r97/adjoint.h"

#include "scratch.h"

#include "rad_device.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace r97 {
namespace {

// bf16 is stored as the high half of an f32, with round-to-nearest-even on the way down (the same
// two functions add.hip carries; each kernel's file owns its own).
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

/* ------------------------------------------------------------------ reading a record */

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return shape.empty() ? 0 : n;
}

/* A numeric parameter: a real one reaches a record as an int or as its decimal spelling. Zero is a
 * value here (`wadd=0` is the plain norm), so an unparsable string is what leaves the fallback. */
double param_num(const OpRecord& rec, const char* name, double fallback) {
    for (const OpParam& p : rec.params) {
        if (p.first != name) continue;
        if (const int64_t* i = std::get_if<int64_t>(&p.second)) return (double)*i;
        const std::string& s = std::get<std::string>(p.second);
        char* end = nullptr;
        const double v = std::strtod(s.c_str(), &end);
        return (end && end != s.c_str() && *end == '\0') ? v : fallback;
    }
    return fallback;
}

/* A flag parameter: present and not zero. A flag the record leaves out is off. */
bool param_flag(const OpRecord& rec, const char* name) {
    for (const OpParam& p : rec.params) {
        if (p.first != name) continue;
        if (const int64_t* i = std::get_if<int64_t>(&p.second)) return *i != 0;
        const std::string& s = std::get<std::string>(p.second);
        return !s.empty() && s != "0";
    }
    return false;
}

/* The op's geometry, as the record and its operands state it. A record that does not agree with
 * itself is not served: the caller returns with dc untouched, because an adjoint must not throw. */
struct Geometry {
    int64_t tokens = 0;
    int64_t head_dim = 0;
    int64_t n_head = 0;
    int64_t n_head_kv = 0;
    int64_t rot = 0;
    int64_t block_size = 0;
    int64_t n_blocks = 0;
    int64_t qg_len = 0;  // n_head * 2 * head_dim, one row of the projection
    int64_t k_len = 0;   // n_head_kv * head_dim
    int64_t q_len = 0;   // n_head * head_dim
    int64_t cs_rows = 0;
    int64_t pool_len = 0;  // numel of the whole kv_cache
    float eps = 1e-6f;
    float wadd = 0.0f;
    float k_scale = 1.0f;
    float v_scale = 1.0f;
    bool kv = false;
    bool cs_f32 = true;
};

bool read_geometry(const AdjointCall& call, const bool kv, Geometry* g) {
    const OpRecord& rec = call.rec;
    const size_t writes = kv ? 4u : 3u;
    const size_t operands = kv ? 12u : 9u;
    if (call.fwd.size() < operands || call.cot.size() < writes || !call.dc) return false;
    if (call.dc->size() < operands) return false;

    const AdValue& f_qg = call.fwd[0];
    const AdValue& f_k = call.fwd[1];
    const AdValue& f_cs = call.fwd[2];
    const AdValue& f_qw = call.fwd[4];
    if (f_qg.shape.size() != 2 || f_k.shape.size() != 2 || f_cs.shape.size() != 2) return false;

    g->kv = kv;
    g->tokens = f_qg.shape[0];
    g->head_dim = (int64_t)f_qw.value.size();
    if (const int64_t hd = (int64_t)param_num(rec, "head_dim", 0); hd > 0) {
        if (hd != g->head_dim) return false;
    }
    if (g->head_dim <= 0 || g->tokens <= 0) return false;

    g->n_head = (int64_t)param_num(rec, "n_head", 0);
    if (g->n_head <= 0) g->n_head = f_qg.shape[1] / (2 * g->head_dim);
    g->n_head_kv = (int64_t)param_num(rec, "n_head_kv", 0);
    if (g->n_head_kv <= 0) g->n_head_kv = f_k.shape[1] / g->head_dim;
    if (g->n_head <= 0 || g->n_head_kv <= 0) return false;

    g->qg_len = g->n_head * 2 * g->head_dim;
    g->k_len = g->n_head_kv * g->head_dim;
    g->q_len = g->n_head * g->head_dim;
    if (g->qg_len != f_qg.shape[1] || g->k_len != f_k.shape[1]) return false;
    if ((int64_t)f_qg.value.size() != g->tokens * g->qg_len) return false;
    if ((int64_t)f_k.value.size() != g->tokens * g->k_len) return false;

    g->rot = (int64_t)param_num(rec, "rot", 0);
    if (g->rot <= 0) g->rot = f_cs.shape[1];
    g->cs_rows = f_cs.shape[0];
    if (g->rot <= 0 || g->rot > g->head_dim || (g->rot & 1) != 0) return false;
    if (f_cs.shape[1] != g->rot) return false;
    if (g->cs_rows <= 0) return false;
    if ((int64_t)f_cs.value.size() != g->cs_rows * g->rot) return false;

    g->eps = (float)param_num(rec, "eps", 1e-6);
    g->wadd = (float)param_num(rec, "wadd", 0.0);
    g->cs_f32 = param_flag(rec, "cs_f32");

    if (kv) {
        const AdValue& f_v = call.fwd[6];
        const AdValue& f_pool = call.fwd[8];
        if ((int64_t)f_v.value.size() != g->tokens * g->k_len) return false;
        if (f_pool.shape.size() != 4) return false;
        g->block_size = (int64_t)param_num(rec, "block_size", f_pool.shape[2]);
        g->n_blocks = f_pool.shape[0];
        if (g->block_size <= 0 || g->n_blocks <= 0) return false;
        if (f_pool.shape[1] != g->n_head_kv || f_pool.shape[2] != g->block_size) return false;
        if (f_pool.shape[3] != 2 * g->head_dim) return false;
        g->pool_len = numel(f_pool.shape);
        if ((int64_t)f_pool.value.size() != g->pool_len) return false;
        g->k_scale = (float)param_num(rec, "k_scale", 1.0);
        g->v_scale = (float)param_num(rec, "v_scale", 1.0);
    }
    return true;
}

/* ------------------------------------------------------------------ the kernel */

// The kernel's own copy of the geometry: int32 is enough for every extent a step has.
struct KernelGeometry {
    int64_t head_dim;
    int64_t n_head;
    int64_t n_head_kv;
    int64_t rot;
    int64_t block_size;
    int64_t qg_len;
    int64_t k_len;
    int64_t q_len;
    int kv;
    float eps;
    float wadd;
    float k_scale;
    float v_scale;
};

// A block-wide sum: 32-lane groups reduce by shuffle, then one group folds the groups. `red` needs
// 33 floats: one per group, and one for the total.
__device__ float block_sum(float v, float* red) {
    for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor(v, m, 32);
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int group = threadIdx.x >> 5;
    if (lane == 0) red[group] = v;
    __syncthreads();
    const unsigned int groups = blockDim.x >> 5;
    float t = threadIdx.x < groups ? red[threadIdx.x] : 0.0f;
    for (int m = 16; m >= 1; m >>= 1) t += __shfl_xor(t, m, 32);
    if (threadIdx.x == 0) red[32] = t;
    __syncthreads();
    const float total = red[32];
    __syncthreads();
    return total;
}

// One block per (token, head) row of the prologue. The row's forward values and the cotangent of
// its output are both head_dim lanes wide, so the two reductions the norm needs -- the sum of
// squares and the sum the adjoint's shared term weights -- are rows of one block, and the rope
// adjoint between them is a pairing of lanes inside the same row.
__global__ void qk_adjoint_kernel(const float* qg, const float* fwd_k, const float* cs,
                                  const int* positions, const float* qw, const float* kw,
                                  const float* dpool, const int* slots, const float* dq,
                                  const float* dk, const float* dgate, unsigned short* dc_qg,
                                  unsigned short* dc_k, unsigned short* dc_v,
                                  const KernelGeometry p) {
    const int heads = (int)(p.n_head + p.n_head_kv);
    const int64_t row = blockIdx.x;
    const int64_t tok = row / heads;
    const int h = (int)(row % heads);
    const bool is_k = h >= (int)p.n_head;
    const int lh = is_k ? h - (int)p.n_head : h;
    const int hd = (int)p.head_dim;

    extern __shared__ float smem[];  // x[head_dim], then the cotangent of the normalised lanes
    float* sx = smem;
    float* sd = smem + p.head_dim;
    __shared__ float red[33];

    const float* src = is_k ? fwd_k + tok * p.k_len + (int64_t)lh * p.head_dim
                            : qg + tok * p.qg_len + (int64_t)lh * 2 * p.head_dim;
    const float* weight = is_k ? kw : qw;
    float acc = 0.0f;
    for (int i = (int)threadIdx.x; i < hd; i += (int)blockDim.x) {
        const float x = src[i];
        sx[i] = x;
        acc += x * x;
    }
    const float ss = block_sum(acc, red);
    const float inv = rsqrtf(ss / (float)hd + p.eps);

    // A store's row for this token and head: the pool cotangent is added to the key's before the
    // rope turns it back, and read again for the value below.
    int64_t pool = -1;
    if (is_k && p.kv) {
        const int slot = slots[tok];
        if (slot >= 0)
            pool = (((int64_t)(slot / p.block_size) * p.n_head_kv + lh) * p.block_size
                    + slot % p.block_size) * 2 * p.head_dim;
    }
    const float* dsrc = is_k ? dk + tok * p.k_len + (int64_t)lh * p.head_dim
                             : dq + tok * p.q_len + (int64_t)lh * p.head_dim;
    for (int i = (int)threadIdx.x; i < hd; i += (int)blockDim.x) {
        float d = dsrc[i];
        if (pool >= 0) d += p.k_scale * dpool[pool + i];
        sd[i] = d;
    }
    __syncthreads();

    // rope's adjoint: y = c*x - s*x_pair on the low lane and s*x + c*x_pair on the high one, so the
    // transpose turns the cotangent with the same table. The thread that owns the pair writes both
    // lanes, which keeps the pass in place.
    const float* csr = cs + (int64_t)positions[tok] * p.rot;
    const int half = (int)(p.rot / 2);
    for (int t = (int)threadIdx.x; t < half; t += (int)blockDim.x) {
        const float c = csr[t];
        const float s = csr[half + t];
        const float a = sd[t];
        const float b = sd[t + half];
        sd[t] = c * a + s * b;
        sd[t + half] = c * b - s * a;
    }
    __syncthreads();

    // The norm's shared term: dx_j = inv*c_j*dy_j - inv^3 x_j/n * sum_i(dy_i x_i c_i), with the
    // gain folded into c and the sum taken over the row after rope has turned the cotangent back.
    acc = 0.0f;
    for (int i = (int)threadIdx.x; i < hd; i += (int)blockDim.x)
        acc += sd[i] * sx[i] * (weight[i] + p.wadd);
    const float shared_sum = block_sum(acc, red);
    const float inv3 = inv * inv * inv;
    for (int i = (int)threadIdx.x; i < hd; i += (int)blockDim.x) {
        const float c = weight[i] + p.wadd;
        const float dx = inv * c * sd[i] - inv3 * sx[i] * shared_sum / (float)hd;
        if (is_k) {
            dc_k[tok * p.k_len + (int64_t)lh * p.head_dim + i] = f32_to_bf16(dx);
            if (pool >= 0)
                dc_v[tok * p.k_len + (int64_t)lh * p.head_dim + i] =
                    f32_to_bf16(p.v_scale * dpool[pool + p.head_dim + i]);
        } else {
            // The query lanes carry the norm's cotangent; the gate lanes carry the gate's through,
            // because the op copies them and the sigmoid above them hands their cotangent down.
            const int64_t base = tok * p.qg_len + (int64_t)lh * 2 * p.head_dim;
            dc_qg[base + i] = f32_to_bf16(dx);
            dc_qg[base + p.head_dim + i] =
                f32_to_bf16(dgate[tok * p.q_len + (int64_t)lh * p.head_dim + i]);
        }
    }
}

/* ------------------------------------------------------------------ the call */

// One stream for every device adjoint call in this process, as add.hip has it: upload, launch and
// download share the stream, so a call is one queue's worth of work.
RadStream adjoint_stream() {
    static RadStream s = [] {
        RadStream created = nullptr;
        if (rad_stream_create(&created, 0) != 0) return (RadStream) nullptr;
        return created;
    }();
    return s;
}

// The tape's f64 value of one operand, read at f32. A bf16 operand is rounded on the way in, so the
// kernel sees what the card's op would read.
std::vector<float> read_f32(const AdValue& v, const bool bf16) {
    std::vector<float> out(v.value.size());
    for (size_t i = 0; i < out.size(); ++i) {
        const float x = (float)v.value[i];
        out[i] = bf16 ? bf16_to_f32(f32_to_bf16(x)) : x;
    }
    return out;
}

// The tape's f64 value of an index operand, as int32: a position or a pool slot. A negative slot
// is a padded row and stores nothing.
std::vector<int> read_index(const AdValue& v) {
    std::vector<int> out(v.value.size());
    for (size_t i = 0; i < out.size(); ++i) {
        const double x = v.value[i];
        out[i] = std::isfinite(x) ? (int)x : -1;
    }
    return out;
}

void download_add(const std::vector<unsigned short>& src, AdValue* dst) {
    const size_t n = src.size() < dst->value.size() ? src.size() : dst->value.size();
    for (size_t i = 0; i < n; ++i) dst->value[i] += bf16_to_f32(src[i]);
}

// One device scratch buffer, copied up. The host vector outlives the copy: the uploads and the
// kernel share a stream, and every host buffer here is still alive at the sync after the launch.
bool upload(device::ScratchBlock& block, const void* host, int64_t bytes, RadStream stream) {
    if (bytes <= 0) return true;
    return rad_memcpy_async(block.p, host, bytes, stream) == 0;
}

void qk_gate_adjoint(const AdjointCall& call, const bool kv) {
    Geometry g;
    if (!read_geometry(call, kv, &g)) return;

    const size_t i_v = kv ? 6u : 0u;
    const size_t i_slot = kv ? 7u : 0u;
    // The cotangents come in the writes' order: the pool first when the op stores.
    const size_t i_pool_cot = 0u;
    const size_t i_q_cot = kv ? 1u : 0u;
    const size_t i_k_cot = i_q_cot + 1;
    const size_t i_gate_cot = i_q_cot + 2;

    RadStream stream = adjoint_stream();
    if (!stream) return;

    // Uploads: the forward values the kernel reads, the cotangents it turns back, and the pool.
    std::vector<float> h_qg = read_f32(call.fwd[0], true);
    std::vector<float> h_k = read_f32(call.fwd[1], true);
    std::vector<float> h_cs = read_f32(call.fwd[2], !g.cs_f32);
    std::vector<int> h_pos = read_index(call.fwd[3]);
    std::vector<float> h_qw = read_f32(call.fwd[4], false);
    std::vector<float> h_kw = read_f32(call.fwd[5], false);
    std::vector<float> h_dq = read_f32(call.cot[i_q_cot], true);
    std::vector<float> h_dk = read_f32(call.cot[i_k_cot], true);
    std::vector<float> h_dgate = read_f32(call.cot[i_gate_cot], true);
    std::vector<float> h_dpool;
    std::vector<int> h_slot;
    if (kv) {
        // Training keeps the pool in bf16, so its cotangent is read at bf16.
        h_dpool = read_f32(call.cot[i_pool_cot], true);
        h_slot = read_index(call.fwd[i_slot]);
        if ((int64_t)h_dpool.size() != g.pool_len) return;
        if ((int64_t)h_slot.size() != g.tokens) return;
    }
    if ((int64_t)h_qg.size() != g.tokens * g.qg_len) return;
    if ((int64_t)h_k.size() != g.tokens * g.k_len) return;
    if ((int64_t)h_pos.size() != g.tokens) return;
    if ((int64_t)h_dq.size() != g.tokens * g.q_len) return;
    if ((int64_t)h_dk.size() != g.tokens * g.k_len) return;
    if ((int64_t)h_dgate.size() != g.tokens * g.q_len) return;
    for (int p : h_pos) {
        if (p < 0 || p >= g.cs_rows) return;  // a position the table does not hold
    }

    device::ScratchBlock d_qg(g.tokens * g.qg_len * (int64_t)sizeof(float));
    device::ScratchBlock d_k(g.tokens * g.k_len * (int64_t)sizeof(float));
    device::ScratchBlock d_cs(g.cs_rows * g.rot * (int64_t)sizeof(float));
    device::ScratchBlock d_pos(g.tokens * (int64_t)sizeof(int));
    device::ScratchBlock d_qw(g.head_dim * (int64_t)sizeof(float));
    device::ScratchBlock d_kw(g.head_dim * (int64_t)sizeof(float));
    device::ScratchBlock d_dq(g.tokens * g.q_len * (int64_t)sizeof(float));
    device::ScratchBlock d_dk(g.tokens * g.k_len * (int64_t)sizeof(float));
    device::ScratchBlock d_dgate(g.tokens * g.q_len * (int64_t)sizeof(float));
    device::ScratchBlock d_dpool(kv ? g.pool_len * (int64_t)sizeof(float) : 0);
    device::ScratchBlock d_slot(kv ? g.tokens * (int64_t)sizeof(int) : 0);
    device::ScratchBlock d_dc_qg(g.tokens * g.qg_len * (int64_t)sizeof(unsigned short));
    device::ScratchBlock d_dc_k(g.tokens * g.k_len * (int64_t)sizeof(unsigned short));
    device::ScratchBlock d_dc_v(kv ? g.tokens * g.k_len * (int64_t)sizeof(unsigned short) : 0);
    if (!d_qg || !d_k || !d_cs || !d_pos || !d_qw || !d_kw || !d_dq || !d_dk || !d_dgate) return;
    if (!d_dc_qg || !d_dc_k) return;
    if (kv && (!d_dpool || !d_slot || !d_dc_v)) return;

    if (!upload(d_qg, h_qg.data(), (int64_t)h_qg.size() * 4, stream)) return;
    if (!upload(d_k, h_k.data(), (int64_t)h_k.size() * 4, stream)) return;
    if (!upload(d_cs, h_cs.data(), (int64_t)h_cs.size() * 4, stream)) return;
    if (!upload(d_pos, h_pos.data(), (int64_t)h_pos.size() * (int64_t)sizeof(int), stream)) return;
    if (!upload(d_qw, h_qw.data(), (int64_t)h_qw.size() * 4, stream)) return;
    if (!upload(d_kw, h_kw.data(), (int64_t)h_kw.size() * 4, stream)) return;
    if (!upload(d_dq, h_dq.data(), (int64_t)h_dq.size() * 4, stream)) return;
    if (!upload(d_dk, h_dk.data(), (int64_t)h_dk.size() * 4, stream)) return;
    if (!upload(d_dgate, h_dgate.data(), (int64_t)h_dgate.size() * 4, stream)) return;
    if (kv) {
        if (!upload(d_dpool, h_dpool.data(), (int64_t)h_dpool.size() * 4, stream)) return;
        if (!upload(d_slot, h_slot.data(), (int64_t)h_slot.size() * (int64_t)sizeof(int), stream))
            return;
    }

    KernelGeometry kp{};
    kp.head_dim = g.head_dim;
    kp.n_head = g.n_head;
    kp.n_head_kv = g.n_head_kv;
    kp.rot = g.rot;
    kp.block_size = g.block_size;
    kp.qg_len = g.qg_len;
    kp.k_len = g.k_len;
    kp.q_len = g.q_len;
    kp.kv = kv ? 1 : 0;
    kp.eps = g.eps;
    kp.wadd = g.wadd;
    kp.k_scale = g.k_scale;
    kp.v_scale = g.v_scale;

    const int64_t rows = g.tokens * (g.n_head + g.n_head_kv);
    const unsigned int threads = 128;
    const size_t shared = (size_t)(2 * g.head_dim) * sizeof(float);
    // One block per (token, head) row: a row is what the two reductions and the rope pairing work
    // inside, so the grid is the rows themselves.
    hipLaunchKernelGGL(qk_adjoint_kernel, dim3((unsigned int)rows), dim3(threads), shared,
                       (hipStream_t)stream, (const float*)d_qg.p, (const float*)d_k.p,
                       (const float*)d_cs.p, (const int*)d_pos.p, (const float*)d_qw.p,
                       (const float*)d_kw.p, kv ? (const float*)d_dpool.p : nullptr,
                       kv ? (const int*)d_slot.p : nullptr, (const float*)d_dq.p,
                       (const float*)d_dk.p, (const float*)d_dgate.p,
                       (unsigned short*)d_dc_qg.p, (unsigned short*)d_dc_k.p,
                       kv ? (unsigned short*)d_dc_v.p : nullptr, kp);
    if (rad_stream_sync(stream) != 0) return;

    std::vector<unsigned short> h_dc_qg((size_t)(g.tokens * g.qg_len));
    std::vector<unsigned short> h_dc_k((size_t)(g.tokens * g.k_len));
    std::vector<unsigned short> h_dc_v(kv ? (size_t)(g.tokens * g.k_len) : 0);
    if (rad_memcpy_async(h_dc_qg.data(), d_dc_qg.p, (int64_t)h_dc_qg.size() * 2, stream) != 0)
        return;
    if (rad_memcpy_async(h_dc_k.data(), d_dc_k.p, (int64_t)h_dc_k.size() * 2, stream) != 0) return;
    if (kv && rad_memcpy_async(h_dc_v.data(), d_dc_v.p, (int64_t)h_dc_v.size() * 2, stream) != 0)
        return;
    if (rad_stream_sync(stream) != 0) return;

    // The tape's cotangents are f64 and are added to, never overwritten: other ops share them.
    download_add(h_dc_qg, &call.dc->at(0));
    download_add(h_dc_k, &call.dc->at(1));
    if (kv) download_add(h_dc_v, &call.dc->at(i_v));
}

void qk_gate_device_adjoint(const AdjointCall& call) { qk_gate_adjoint(call, false); }

void qk_gate_kv_store_device_adjoint(const AdjointCall& call) { qk_gate_adjoint(call, true); }

// The streamed store keeps its pool quantised and reconstructs a row from a per-block residue the
// trainer does not hold, so training has no cotangent to give it. A row that says so is better than
// a missing one: the caller reads the reason instead of a lookup failure that names only the op.
void qk_gate_streamed_device_adjoint(const AdjointCall&) {
    throw std::runtime_error(
        "qk_norm_rope_gate_kv_store_streamed is refused in training: the streamed store keeps the "
        "pool in fp8 with a per-block residue row (`res`, `host`) that the trainer's bf16 pool "
        "does not carry, so its pool cotangent has no bf16 form; issue "
        "qk_norm_rope_gate_kv_store against the bf16 pool instead");
}

}  // namespace

// The gains (4, 5), the rope table (2) and the positions (3) are constants, and so is the store's
// slot map (7): none of them is asked for a cotangent.
R97_REGISTER_ADJOINT_F("qk_norm_rope_gate", device, qk_gate_device_adjoint, 2, 3, 4, 5)
R97_REGISTER_ADJOINT_F("qk_norm_rope_gate_kv_store", device, qk_gate_kv_store_device_adjoint, 2, 3,
                       4, 5, 7)
R97_REGISTER_ADJOINT("qk_norm_rope_gate_kv_store_streamed", device,
                     qk_gate_streamed_device_adjoint)

}  // namespace r97
