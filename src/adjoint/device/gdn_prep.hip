// src/adjoint/device/gdn_prep.hip -- the device adjoints of the gdn_prep family.
//
// The same two ops as src/adjoint/host/gdn_prep.cpp, on the card:
//
//   gdn_conv_prep      the short causal convolution over the projected row, its silu, the l2
//                      normalisation of q and k, and the per-chunk gate and beta;
//   gdn_gated_rmsnorm  the gated RMS norm on the recurrence's output.
//
// The difference from the host adjoints is where the arithmetic happens: the operands and the
// cotangents are read at f32 precision, every sum that runs over tokens accumulates in f32 (a
// chunk's gate tail, the window of the convolution, a head's l2 dot), and each result is rounded
// once to the op's bf16 dtype on the way out. tests/gpu/adjoint_gdn_prep.cpp holds both to the
// host adjoints within the bf16 tolerance docs/kernels/README.md states; docs/kernels/gdn_prep.md
// has the formulas, the scratch and the measured call times.
//
// The conv op is served at a prefill from position 0, the shape every current caller records and
// the shape the host twin implements: the carried convolution state is zero, what it writes is the
// tail of `x`, and the fused row of `x` carries the conv channels only.
//
// The contract gives a device adjoint f64 operands, not device buffers, so this call owns every
// device buffer it touches. They are scratch, freed before it returns; the harness reports the
// high-water mark. The host staging vectors are reused across calls: a 4096-token case moves
// hundreds of megabytes and a fresh allocation per call would show up in the time the doc reports.
#include "r97/adjoint.h"

#include "scratch.h"

#include "rad_device.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace r97 {
namespace {

/* gdn_conv_prep's schema positions, in schema order, as the host adjoint names them. */
enum : int {
    kX = 0,        // the fused projection row, [M, conv_dim] bf16
    kW = 1,        // the conv weight, [conv_dim, conv_width] bf16, frozen
    kBias = 2,     // the conv bias, [conv_dim] bf16, frozen
    kALog = 3,     // [H] f32, frozen
    kDtBias = 4,   // [H] f32, frozen
    kState = 5,    // conv_state, [S, conv_dim, conv_width-1] bf16, INOUT
    kCu = 6,       // sequence offsets, [S+1] i32, frozen
    kA = 7,        // the gate projection, [M, H] bf16
    kBGate = 8,    // the beta projection, [M, H] bf16
    kSidx = 9,     // conv-state slot per sequence, [S] i32, frozen
    kHasInit = 10, // cold (0) or warm (1) start per sequence, [S] i32, frozen
    kQ = 11,
    kK = 12,
    kV = 13,
    kG = 14,
    kBeta = 15,
    kConvPrepOperands = 16,
};

/* The writes, in the order the walk hands their cotangents. */
enum : int { wState = 0, wQ = 1, wK = 2, wV = 3, wG = 4, wBeta = 5, wConvPrepWrites = 6 };

/* gdn_gated_rmsnorm's schema positions. */
enum : int { nX = 0, nZ = 1, nW = 2, nO = 3, nRmsnormOperands = 4 };

/* The epsilons and the softplus threshold are the f64 parameters the op's schema allows and the
 * tape's parameter list cannot carry, so both sides use the op's defaults. */
constexpr float kL2Eps = 1e-6f;
constexpr float kRmsEps = 1e-6f;
constexpr float kSoftplusThr = 20.0f;

// bf16 is stored as the high half of an f32, with round-to-nearest-even on the way down. The
// kernels store and load it as unsigned short so the file needs neither __bf16 nor a HIP bf16
// type, and so the host side of this file can convert the same way.
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

__device__ inline float sigmoidf(float v) { return 1.0f / (1.0f + expf(-v)); }

__device__ inline float siluf(float v) { return v * sigmoidf(v); }

/* d/dv of v * sigmoid(v). */
__device__ inline float silu_gradf(float v) {
    const float s = sigmoidf(v);
    return s * (1.0f + v * (1.0f - s));
}

/* d/dv of softplus, cut off at the same threshold the forward uses: above it the forward's
 * softplus is the identity, so the derivative is 1. */
__device__ inline float softplus_gradf(float v) {
    return v > kSoftplusThr ? 1.0f : sigmoidf(v);
}

// -- parameters and geometry -------------------------------------------------

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

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

/* What the conv op's shapes resolve to, resolved the way the host adjoint resolves them. */
struct ConvGeom {
    int64_t K = 0;         // head_k
    int64_t V = 0;         // head_v
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
    if (g->N <= 0 || g->N > 65535) return false;
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

// One stream for every device adjoint call in this process, as src/adjoint/device/add.hip does: a
// call is one queue's worth of upload, launch and download.
RadStream adjoint_stream() {
    static RadStream s = [] {
        RadStream created = nullptr;
        if (rad_stream_create(&created, 0) != 0) return (RadStream) nullptr;
        return created;
    }();
    return s;
}

int64_t blocks(int64_t n) { return (n + 255) / 256; }

/* A device buffer may only go back to the allocator once the queue that reads it has drained: a
 * launch is asynchronous, so a stage's buffers outlive the end of its block by however long the
 * card takes. This syncs the call's stream when it leaves scope, so no path -- including the ones
 * that return early on a failed copy -- can free a buffer the queue still holds. */
class SyncOnExit {
public:
    explicit SyncOnExit(RadStream stream) : stream_(stream) {}
    SyncOnExit(const SyncOnExit&) = delete;
    SyncOnExit& operator=(const SyncOnExit&) = delete;
    ~SyncOnExit() {
        if (stream_) (void)rad_stream_sync(stream_);
    }

private:
    RadStream stream_;
};

/* Upload staging, reused across calls: a buffer is a name for a size, not for an operand, and no
 * two operands of one call share one (a copy is still in flight when the next operand is staged). */
constexpr int kPoolSlots = 16;
std::vector<float>& upload_buffer(int slot, int64_t n) {
    thread_local std::vector<std::vector<float>> pool((size_t)kPoolSlots);
    std::vector<float>& b = pool[(size_t)slot];
    if ((int64_t)b.size() < n) b.assign((size_t)n, 0.0f);
    return b;
}

std::vector<unsigned short>& download_buffer(int slot, int64_t n) {
    thread_local std::vector<std::vector<unsigned short>> pool((size_t)kPoolSlots);
    std::vector<unsigned short>& b = pool[(size_t)slot];
    if ((int64_t)b.size() < n) b.assign((size_t)n, 0);
    return b;
}

// An operand's first `n` values at f32, in the slot's staging buffer. A value the operand does not
// carry -- the optional conv bias -- stages as zero, which is what the forward reads there.
const std::vector<float>& stage(const AdValue& v, int slot, int64_t n) {
    std::vector<float>& b = upload_buffer(slot, n);
    const int64_t m = std::min<int64_t>(n, (int64_t)v.value.size());
    for (int64_t i = 0; i < m; ++i) b[(size_t)i] = (float)v.value[(size_t)i];
    for (int64_t i = m; i < n; ++i) b[(size_t)i] = 0.0f;
    return b;
}

// -- the conv op's kernels ---------------------------------------------------

/* Phase 1, per (token, channel): the causal convolution and its activation. `row_bos` and `row_t`
 * put a global row in its sequence, so the kernel needs no search. The convolution state is zero
 * at this prefill, so a source row before the sequence's first contributes nothing. */
__global__ void conv_acc_kernel(const float* x, const float* w, const float* bias,
                                const int* row_bos, const int* row_t, float* acc,
                                int64_t total, int64_t conv_dim, int64_t xw, int width) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t row = i / conv_dim;
    const int64_t c = i - row * conv_dim;
    const int64_t bos = row_bos[row];
    const int64_t t = row_t[row];
    float a = bias[c];
    const size_t wbase = (size_t)(c * width);
    for (int j = 0; j < width; ++j) {
        const int64_t src = t - (width - 1) + j;
        if (src < 0) continue;  // the carried state is zero at prefill
        a += x[(bos + src) * xw + c] * w[wbase + (size_t)j];
    }
    acc[i] = a;
}

/* Phase 2: q and k through the l2 norm, one block per (token, head). With u the pre-norm value and
 * inv = rsqrt(ss + eps), du_d = inv*dout_d - inv^3*u_d*dot, and the silu between it and the conv
 * row scales it by silu'(acc). Every channel of the conv row belongs to exactly one of q, k and v,
 * so the phases store into dacc rather than adding to it. */
__global__ void l2_backward_kernel(const float* acc, const float* dout, float* dacc, int64_t Hg,
                                   int64_t K, int64_t conv_dim, int64_t chan0, float eps) {
    const int64_t b = (int64_t)blockIdx.x;
    const int64_t row = b / Hg;
    const int64_t h = b - row * Hg;
    const int tid = (int)threadIdx.x;
    const int nthr = (int)blockDim.x;
    extern __shared__ float sh[];
    float* s_ss = sh;
    float* s_dot = sh + nthr;

    const int64_t base = row * conv_dim + chan0 + h * K;
    const int64_t obase = (row * Hg + h) * K;
    float ss = 0.0f;
    float dot = 0.0f;
    for (int64_t d = tid; d < K; d += nthr) {
        const float u = siluf(acc[base + d]);
        ss += u * u;
        dot += u * dout[obase + d];
    }
    s_ss[tid] = ss;
    s_dot[tid] = dot;
    __syncthreads();
    for (int stride = nthr / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_ss[tid] += s_ss[tid + stride];
            s_dot[tid] += s_dot[tid + stride];
        }
        __syncthreads();
    }
    const float inv = rsqrtf(s_ss[0] + eps);
    const float inv3 = inv * inv * inv;
    const float all = s_dot[0];
    for (int64_t d = tid; d < K; d += nthr) {
        const int64_t i = base + d;
        const float u = siluf(acc[i]);
        const float du = inv * dout[obase + d] - inv3 * u * all;
        dacc[i] = du * silu_gradf(acc[i]);
    }
}

/* Phase 2's other half: v straight through the activation. */
__global__ void v_backward_kernel(const float* acc, const float* dv, float* dacc, int64_t total,
                                  int64_t conv_dim, int64_t H, int64_t V, int64_t chan0) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t row = i / (H * V);
    const int64_t e = i - row * (H * V);
    const int64_t h = e / V;
    const int64_t c = row * conv_dim + chan0 + e;
    dacc[c] = dv[(row * H + h) * V + (e - h * V)] * silu_gradf(acc[c]);
}

/* Phase 3: the convolution's backward, one thread per (source row, channel). A source row receives
 * the window of tokens that read it, `width` terms summed in f32 and rounded once; at a cold start
 * the carried state holds the tail of x itself, so its cotangent lands on the same rows. */
__global__ void conv_backward_kernel(const float* dacc, const float* w, const float* d_state,
                                     const int* row_bos, const int* row_slen, const int* row_slot,
                                     unsigned short* dx, int64_t total, int64_t conv_dim,
                                     int64_t xw, int width, int64_t slots, int64_t keep) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t row = i / conv_dim;
    const int64_t c = i - row * conv_dim;
    const int64_t bos = row_bos[row];
    const int64_t s = row - bos;
    const int64_t slen = row_slen[row];

    float sum = 0.0f;
    const size_t wbase = (size_t)(c * width);
    for (int j = 0; j < width; ++j) {
        const int64_t t = s + (width - 1) - j;
        if (t >= slen) continue;  // t >= s >= 0, so only the sequence's end can cut the window
        sum += dacc[(bos + t) * conv_dim + c] * w[wbase + (size_t)j];
    }
    const int64_t slot = row_slot[row];
    if (d_state && slot >= 0 && slot < slots) {
        for (int64_t k = 0; k < keep; ++k) {
            if (slen - (width - 1) + k != s) continue;
            sum += d_state[(slot * conv_dim + c) * keep + k];
        }
    }
    dx[row * xw + c] = f32_to_bf16(sum);
}

/* Phase 4: the gate. g is a within-chunk running sum, so the cotangent of an increment is the sum
 * of its own and every later g in its chunk — a tail per (sequence, head, chunk), which is why one
 * thread per chunk is enough and the chunks need no ordering between them. The tail runs in f32
 * however long the chunk is. */
__global__ void gate_backward_kernel(const int* cu, const int* chunk_off, const float* a,
                                     const float* b_gate, const float* a_log, const float* dt_bias,
                                     const float* d_g, const float* d_beta, unsigned short* da,
                                     unsigned short* d_bgate, int64_t H, int64_t chunk,
                                     int64_t total_chunks) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * total_chunks) return;
    const int64_t h = idx / total_chunks;
    const int64_t cidx = idx - h * total_chunks;
    int64_t n = 0;
    while (chunk_off[n + 1] <= cidx) ++n;

    const int64_t bos = cu[n];
    const int64_t slen = cu[n + 1] - bos;
    const int64_t k = cidx - chunk_off[n];
    const int64_t t0 = k * chunk;
    const int64_t t1 = (k + 1) * chunk < slen ? (k + 1) * chunk : slen;
    const float decay = -expf(a_log[h]);
    const float bias = dt_bias[h];
    float tail = 0.0f;
    for (int64_t t = t1 - 1; t >= t0; --t) {
        const int64_t at = (bos + t) * H + h;
        tail += d_g[at];
        da[at] = f32_to_bf16(tail * decay * softplus_gradf(a[at] + bias));
        const float s = sigmoidf(b_gate[at]);
        d_bgate[at] = f32_to_bf16(d_beta[at] * s * (1.0f - s));
    }
}

void gdn_conv_prep_device_adjoint(const AdjointCall& call) {
    const OpRecord& rec = call.rec;
    if (!call.dc || call.fwd.size() < (size_t)kConvPrepOperands ||
        call.cot.size() < (size_t)wConvPrepWrites ||
        call.dc->size() < (size_t)kConvPrepOperands)
        return;
    const std::vector<AdValue>& f = call.fwd;
    ConvGeom g;
    if (!conv_geom(rec, f[kX], f[kCu], &g)) return;

    const AdValue& d_state = call.cot[wState];
    const AdValue& d_q = call.cot[wQ];
    const AdValue& d_k = call.cot[wK];
    const AdValue& d_v = call.cot[wV];
    const AdValue& d_g = call.cot[wG];
    const AdValue& d_beta = call.cot[wBeta];

    const int64_t qn = g.T * g.Hg * g.K;
    const int64_t vn = g.T * g.H * g.V;
    const int64_t gn = g.T * g.H;
    const int64_t keep = g.width - 1;
    const int64_t sn = keep > 0 ? g.slots * g.conv_dim * keep : 0;
    if ((int64_t)d_q.value.size() < qn || (int64_t)d_k.value.size() < qn ||
        (int64_t)d_v.value.size() < vn || (int64_t)d_g.value.size() < gn ||
        (int64_t)d_beta.value.size() < gn)
        return;
    if (sn > 0 && (int64_t)d_state.value.size() < sn) return;
    if ((int64_t)f[kA].value.size() < gn || (int64_t)f[kBGate].value.size() < gn) return;
    if ((int64_t)f[kALog].value.size() < g.H || (int64_t)f[kDtBias].value.size() < g.H) return;

    AdValue& dx_out = call.dc->at(kX);
    AdValue& da_out = call.dc->at(kA);
    AdValue& d_bgate_out = call.dc->at(kBGate);
    if ((int64_t)dx_out.value.size() < g.T * g.xw) return;
    if ((int64_t)da_out.value.size() < gn || (int64_t)d_bgate_out.value.size() < gn) return;

    RadStream stream = adjoint_stream();
    if (!stream) return;

    /* The rows and the sequences, resolved once on the host: a kernel that had to search for the
     * sequence of a row would do it for every channel. */
    std::vector<int32_t> row_bos((size_t)g.T), row_t((size_t)g.T), row_slen((size_t)g.T),
        row_slot((size_t)g.T);
    std::vector<int32_t> cu((size_t)(g.N + 1));
    std::vector<int32_t> chunk_off((size_t)(g.N + 1));
    int64_t total_chunks = 0;
    for (int64_t n = 0; n < g.N; ++n) {
        const int64_t bos = (int64_t)f[kCu].value[(size_t)n];
        const int64_t end = (int64_t)f[kCu].value[(size_t)(n + 1)];
        const int64_t slen = end - bos;
        if (slen < 0 || end > g.T) return;
        cu[(size_t)n] = (int32_t)bos;
        chunk_off[(size_t)n] = (int32_t)total_chunks;
        total_chunks += (slen + g.chunk - 1) / g.chunk;
        const int64_t slot = n < (int64_t)f[kSidx].value.size() ? (int64_t)f[kSidx].value[(size_t)n]
                                                               : -1;
        for (int64_t t = 0; t < slen; ++t) {
            row_bos[(size_t)(bos + t)] = (int32_t)bos;
            row_t[(size_t)(bos + t)] = (int32_t)t;
            row_slen[(size_t)(bos + t)] = (int32_t)slen;
            row_slot[(size_t)(bos + t)] = (int32_t)slot;
        }
    }
    cu[(size_t)g.N] = (int32_t)g.T;
    chunk_off[(size_t)g.N] = (int32_t)total_chunks;

    device::ScratchBlock d_w(g.conv_dim * g.width * (int64_t)sizeof(float));
    device::ScratchBlock d_bias(g.conv_dim * (int64_t)sizeof(float));
    device::ScratchBlock d_rows((int64_t)sizeof(int32_t) * 4 * g.T);
    device::ScratchBlock d_dacc(g.T * g.conv_dim * (int64_t)sizeof(float));
    // Declared after the buffers above so it drains the stream before they go back.
    const SyncOnExit drain(stream);
    if (!d_w || !d_bias || !d_rows || !d_dacc) return;
    int32_t* d_row_bos = (int32_t*)d_rows.p;
    int32_t* d_row_t = d_row_bos + g.T;
    int32_t* d_row_slen = d_row_t + g.T;
    int32_t* d_row_slot = d_row_slen + g.T;

    const std::vector<float>& h_w = stage(f[kW], 0, g.conv_dim * g.width);
    const std::vector<float>& h_bias = stage(f[kBias], 1, g.conv_dim);
    if (rad_memcpy_async(d_w.p, h_w.data(), g.conv_dim * g.width * (int64_t)sizeof(float), stream) !=
        0)
        return;
    if (rad_memcpy_async(d_bias.p, h_bias.data(), g.conv_dim * (int64_t)sizeof(float), stream) != 0)
        return;
    if (rad_memcpy_async(d_row_bos, row_bos.data(), g.T * (int64_t)sizeof(int32_t), stream) != 0)
        return;
    if (rad_memcpy_async(d_row_t, row_t.data(), g.T * (int64_t)sizeof(int32_t), stream) != 0) return;
    if (rad_memcpy_async(d_row_slen, row_slen.data(), g.T * (int64_t)sizeof(int32_t), stream) != 0)
        return;
    if (rad_memcpy_async(d_row_slot, row_slot.data(), g.T * (int64_t)sizeof(int32_t), stream) != 0)
        return;

    const float* w = (const float*)d_w.p;
    const float* bias = (const float*)d_bias.p;

    /* Stage 1 and 2 share acc; x is only stage 1's, so it goes back before the cotangents arrive. */
    {
        device::ScratchBlock d_acc(g.T * g.conv_dim * (int64_t)sizeof(float));
        {
            device::ScratchBlock d_x(g.T * g.xw * (int64_t)sizeof(float));
            if (!d_acc || !d_x) return;
            // Only the first T rows and the conv channels are read, but the rows are dense, so the
            // upload is the whole row width.
            std::vector<float>& h_x = upload_buffer(2, g.T * g.xw);
            for (int64_t row = 0; row < g.T; ++row)
                for (int64_t c = 0; c < g.conv_dim; ++c)
                    h_x[(size_t)(row * g.xw + c)] = (float)f[kX].value[(size_t)(row * g.xw + c)];
            if (rad_memcpy_async(d_x.p, h_x.data(), g.T * g.xw * (int64_t)sizeof(float), stream) != 0)
                return;
            const int64_t total = g.T * g.conv_dim;
            hipLaunchKernelGGL(conv_acc_kernel, dim3((unsigned int)blocks(total)), dim3(256), 0,
                               (hipStream_t)stream, (const float*)d_x.p, w, bias, d_row_bos,
                               d_row_t, (float*)d_acc.p, total, g.conv_dim, g.xw, (int)g.width);
            // x is this stage's alone; it goes back now rather than at the end of the call.
            (void)rad_stream_sync(stream);
        }

        /* Phase 2: q and k per head, then v. Each channel of the conv row has exactly one owner
         * among the three, so the phases never collide. */
        device::ScratchBlock d_dq(qn * (int64_t)sizeof(float));
        device::ScratchBlock d_dk(qn * (int64_t)sizeof(float));
        device::ScratchBlock d_dv(vn * (int64_t)sizeof(float));
        if (!d_dq || !d_dk || !d_dv) return;
        const std::vector<float>& h_q = stage(d_q, 3, qn);
        if (rad_memcpy_async(d_dq.p, h_q.data(), qn * (int64_t)sizeof(float), stream) != 0) return;
        const std::vector<float>& h_k = stage(d_k, 4, qn);
        if (rad_memcpy_async(d_dk.p, h_k.data(), qn * (int64_t)sizeof(float), stream) != 0) return;
        const std::vector<float>& h_v = stage(d_v, 5, vn);
        if (rad_memcpy_async(d_dv.p, h_v.data(), vn * (int64_t)sizeof(float), stream) != 0) return;

        int blk = 32;
        while ((int64_t)blk < g.K && blk < 512) blk <<= 1;
        const int64_t l2_blocks = g.T * g.Hg;
        hipLaunchKernelGGL(l2_backward_kernel, dim3((unsigned int)l2_blocks), dim3((unsigned)blk),
                           (size_t)(2 * blk) * sizeof(float), (hipStream_t)stream,
                           (const float*)d_acc.p, (const float*)d_dq.p, (float*)d_dacc.p, g.Hg,
                           g.K, g.conv_dim, (int64_t)0, kL2Eps);
        hipLaunchKernelGGL(l2_backward_kernel, dim3((unsigned int)l2_blocks), dim3((unsigned)blk),
                           (size_t)(2 * blk) * sizeof(float), (hipStream_t)stream,
                           (const float*)d_acc.p, (const float*)d_dk.p, (float*)d_dacc.p, g.Hg,
                           g.K, g.conv_dim, g.Hg * g.K, kL2Eps);
        const int64_t v_total = g.T * g.H * g.V;
        hipLaunchKernelGGL(v_backward_kernel, dim3((unsigned int)blocks(v_total)), dim3(256), 0,
                           (hipStream_t)stream, (const float*)d_acc.p, (const float*)d_dv.p,
                           (float*)d_dacc.p, v_total, g.conv_dim, g.H, g.V, 2 * g.Hg * g.K);
        // acc and the three cotangents are this stage's; the convolution's backward reads dacc.
        (void)rad_stream_sync(stream);
    }

    /* Phase 3: back through the convolution, and through the state at a cold start. */
    {
        device::ScratchBlock d_dstate(sn * (int64_t)sizeof(float));
        device::ScratchBlock d_dx(g.T * g.xw * (int64_t)sizeof(unsigned short));
        if (!d_dx || (sn > 0 && !d_dstate)) return;
        if (sn > 0) {
            const std::vector<float>& h_state = stage(d_state, 6, sn);
            if (rad_memcpy_async(d_dstate.p, h_state.data(), sn * (int64_t)sizeof(float), stream) !=
                0)
                return;
        }
        // The channels past conv_dim have no cotangent and must download as zero.
        if (hipMemsetAsync(d_dx.p, 0, (size_t)(g.T * g.xw * (int64_t)sizeof(unsigned short)),
                           (hipStream_t)stream) != hipSuccess)
            return;
        const int64_t total = g.T * g.conv_dim;
        hipLaunchKernelGGL(conv_backward_kernel, dim3((unsigned int)blocks(total)), dim3(256), 0,
                           (hipStream_t)stream, (const float*)d_dacc.p, w,
                           sn > 0 ? (const float*)d_dstate.p : nullptr, d_row_bos, d_row_slen,
                           d_row_slot, (unsigned short*)d_dx.p, total, g.conv_dim, g.xw,
                           (int)g.width, g.slots, keep);
        if (rad_stream_sync(stream) != 0) return;
        const int64_t n = g.T * g.xw;
        std::vector<unsigned short>& h_dx = download_buffer(0, n);
        if (rad_memcpy_async(h_dx.data(), d_dx.p, n * (int64_t)sizeof(unsigned short), stream) != 0)
            return;
        if (rad_stream_sync(stream) != 0) return;
        for (int64_t i = 0; i < n; ++i) dx_out.value[(size_t)i] += bf16_to_f32(h_dx[(size_t)i]);
    }

    /* Phase 4: the gate's within-chunk tail and beta's sigmoid. */
    {
        device::ScratchBlock d_a(gn * (int64_t)sizeof(float));
        device::ScratchBlock d_bgate(gn * (int64_t)sizeof(float));
        device::ScratchBlock d_alog(g.H * (int64_t)sizeof(float));
        device::ScratchBlock d_dtbias(g.H * (int64_t)sizeof(float));
        device::ScratchBlock d_dg(gn * (int64_t)sizeof(float));
        device::ScratchBlock d_dbeta(gn * (int64_t)sizeof(float));
        device::ScratchBlock d_cu((int64_t)sizeof(int32_t) * (g.N + 1));
        device::ScratchBlock d_chunk_off((int64_t)sizeof(int32_t) * (g.N + 1));
        device::ScratchBlock d_da(gn * (int64_t)sizeof(unsigned short));
        device::ScratchBlock d_dbg(gn * (int64_t)sizeof(unsigned short));
        if (!d_a || !d_bgate || !d_alog || !d_dtbias || !d_dg || !d_dbeta || !d_cu ||
            !d_chunk_off || !d_da || !d_dbg)
            return;
        if (rad_memcpy_async(d_a.p, stage(f[kA], 7, gn).data(), gn * (int64_t)sizeof(float),
                             stream) != 0)
            return;
        if (rad_memcpy_async(d_bgate.p, stage(f[kBGate], 8, gn).data(), gn * (int64_t)sizeof(float),
                             stream) != 0)
            return;
        if (rad_memcpy_async(d_alog.p, stage(f[kALog], 9, g.H).data(),
                             g.H * (int64_t)sizeof(float), stream) != 0)
            return;
        if (rad_memcpy_async(d_dtbias.p, stage(f[kDtBias], 10, g.H).data(),
                             g.H * (int64_t)sizeof(float), stream) != 0)
            return;
        if (rad_memcpy_async(d_dg.p, stage(d_g, 11, gn).data(), gn * (int64_t)sizeof(float),
                             stream) != 0)
            return;
        if (rad_memcpy_async(d_dbeta.p, stage(d_beta, 12, gn).data(), gn * (int64_t)sizeof(float),
                             stream) != 0)
            return;
        if (rad_memcpy_async(d_cu.p, cu.data(), (g.N + 1) * (int64_t)sizeof(int32_t), stream) != 0)
            return;
        if (rad_memcpy_async(d_chunk_off.p, chunk_off.data(),
                             (g.N + 1) * (int64_t)sizeof(int32_t), stream) != 0)
            return;
        const int64_t total = g.H * total_chunks;
        hipLaunchKernelGGL(gate_backward_kernel, dim3((unsigned int)blocks(total)), dim3(256), 0,
                           (hipStream_t)stream, (const int*)d_cu.p, (const int*)d_chunk_off.p,
                           (const float*)d_a.p, (const float*)d_bgate.p, (const float*)d_alog.p,
                           (const float*)d_dtbias.p, (const float*)d_dg.p,
                           (const float*)d_dbeta.p, (unsigned short*)d_da.p,
                           (unsigned short*)d_dbg.p, g.H, g.chunk, total_chunks);
        if (rad_stream_sync(stream) != 0) return;
        std::vector<unsigned short>& h_da = download_buffer(1, gn);
        std::vector<unsigned short>& h_dbg = download_buffer(2, gn);
        if (rad_memcpy_async(h_da.data(), d_da.p, gn * (int64_t)sizeof(unsigned short), stream) != 0)
            return;
        if (rad_memcpy_async(h_dbg.data(), d_dbg.p, gn * (int64_t)sizeof(unsigned short), stream) !=
            0)
            return;
        if (rad_stream_sync(stream) != 0) return;
        for (int64_t i = 0; i < gn; ++i) {
            da_out.value[(size_t)i] += bf16_to_f32(h_da[(size_t)i]);
            d_bgate_out.value[(size_t)i] += bf16_to_f32(h_dbg[(size_t)i]);
        }
    }
}

// -- the gated RMS norm's kernel ---------------------------------------------

/* One block per row. sc = rsqrt(mean(x^2) + eps) needs the row's sum of squares and dx needs
 *   A = sum_e (x_e * w_e * g(z_e) * do_e),
 * both reductions in f32, then the row is written once:
 *   dx_i = w_i*g(z_i)*sc*do_i - x_i*sc^3/ch*A,   dz_i = do_i*x_i*sc*w_i*g'(z_i). */
__global__ void gated_rmsnorm_adjoint_kernel(const float* x, const float* z, const float* w,
                                             const float* d_o, unsigned short* dx,
                                             unsigned short* dz, int64_t ch, int sig, float eps) {
    const int64_t row = (int64_t)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int nthr = (int)blockDim.x;
    extern __shared__ float sh[];
    float* s_ss = sh;
    float* s_dot = sh + nthr;
    const int64_t base = row * ch;

    float ss = 0.0f;
    float dot = 0.0f;
    for (int64_t i = tid; i < ch; i += nthr) {
        const int64_t e = base + i;
        const float xv = x[e];
        const float zv = z[e];
        const float gate = sig ? sigmoidf(zv) : siluf(zv);
        ss += xv * xv;
        dot += xv * d_o[e] * w[i] * gate;
    }
    s_ss[tid] = ss;
    s_dot[tid] = dot;
    __syncthreads();
    for (int stride = nthr / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_ss[tid] += s_ss[tid + stride];
            s_dot[tid] += s_dot[tid + stride];
        }
        __syncthreads();
    }
    const float sc = rsqrtf(s_ss[0] / (float)ch + eps);
    const float all = s_dot[0];
    const float inv_ch = 1.0f / (float)ch;
    for (int64_t i = tid; i < ch; i += nthr) {
        const int64_t e = base + i;
        const float xv = x[e];
        const float zv = z[e];
        const float wi = w[i];
        const float gate = sig ? sigmoidf(zv) : siluf(zv);
        const float dgate = sig ? gate * (1.0f - gate) : silu_gradf(zv);
        const float do_e = d_o[e];
        dx[e] = f32_to_bf16(wi * gate * sc * do_e - xv * sc * sc * sc * inv_ch * all);
        dz[e] = f32_to_bf16(do_e * xv * sc * wi * dgate);
    }
}

void gdn_gated_rmsnorm_device_adjoint(const AdjointCall& call) {
    const OpRecord& rec = call.rec;
    if (!call.dc || call.fwd.size() < (size_t)nRmsnormOperands || call.cot.empty() ||
        call.dc->size() < (size_t)nRmsnormOperands)
        return;
    const AdValue& x = call.fwd[nX];
    const AdValue& z = call.fwd[nZ];
    const AdValue& w = call.fwd[nW];
    const AdValue& d_o = call.cot[0];
    const int64_t n = (int64_t)x.value.size();
    const int64_t ch = int_param(rec, "channels", x.shape.empty() ? 0 : x.shape.back());
    if (ch <= 0 || (int64_t)w.value.size() < ch || (int64_t)d_o.value.size() < n) return;
    const int64_t rows = n / ch;
    if (rows <= 0 || (int64_t)z.value.size() < n) return;

    AdValue& dx_out = call.dc->at(nX);
    AdValue& dz_out = call.dc->at(nZ);
    if ((int64_t)dx_out.value.size() < n || (int64_t)dz_out.value.size() < n) return;

    RadStream stream = adjoint_stream();
    if (!stream) return;

    device::ScratchBlock d_x(n * (int64_t)sizeof(float));
    device::ScratchBlock d_z(n * (int64_t)sizeof(float));
    device::ScratchBlock d_w(ch * (int64_t)sizeof(float));
    device::ScratchBlock d_do(n * (int64_t)sizeof(float));
    device::ScratchBlock d_dx(n * (int64_t)sizeof(unsigned short));
    device::ScratchBlock d_dz(n * (int64_t)sizeof(unsigned short));
    // Declared after the buffers above so it drains the stream before they go back.
    const SyncOnExit drain(stream);
    if (!d_x || !d_z || !d_w || !d_do || !d_dx || !d_dz) return;

    if (rad_memcpy_async(d_x.p, stage(x, 0, n).data(), n * (int64_t)sizeof(float), stream) != 0) return;
    if (rad_memcpy_async(d_z.p, stage(z, 1, n).data(), n * (int64_t)sizeof(float), stream) != 0) return;
    if (rad_memcpy_async(d_w.p, stage(w, 2, ch).data(), ch * (int64_t)sizeof(float), stream) != 0)
        return;
    if (rad_memcpy_async(d_do.p, stage(d_o, 3, n).data(), n * (int64_t)sizeof(float), stream) != 0)
        return;

    const int sig = str_param(rec, "act", "silu") == "sigmoid" ? 1 : 0;
    const unsigned blk = 256;
    hipLaunchKernelGGL(gated_rmsnorm_adjoint_kernel, dim3((unsigned int)rows), dim3(blk),
                       (size_t)(2 * blk) * sizeof(float), (hipStream_t)stream,
                       (const float*)d_x.p, (const float*)d_z.p, (const float*)d_w.p,
                       (const float*)d_do.p, (unsigned short*)d_dx.p, (unsigned short*)d_dz.p, ch,
                       sig, kRmsEps);
    if (rad_stream_sync(stream) != 0) return;

    std::vector<unsigned short>& h_dx = download_buffer(0, n);
    std::vector<unsigned short>& h_dz = download_buffer(1, n);
    if (rad_memcpy_async(h_dx.data(), d_dx.p, n * (int64_t)sizeof(unsigned short), stream) != 0)
        return;
    if (rad_memcpy_async(h_dz.data(), d_dz.p, n * (int64_t)sizeof(unsigned short), stream) != 0)
        return;
    if (rad_stream_sync(stream) != 0) return;
    for (int64_t i = 0; i < n; ++i) {
        dx_out.value[(size_t)i] += bf16_to_f32(h_dx[(size_t)i]);
        dz_out.value[(size_t)i] += bf16_to_f32(h_dz[(size_t)i]);
    }
}

}  // namespace

R97_REGISTER_ADJOINT_F("gdn_conv_prep", device, gdn_conv_prep_device_adjoint, kW, kBias, kALog,
                       kDtBias, kCu, kSidx, kHasInit)
R97_REGISTER_ADJOINT_F("gdn_gated_rmsnorm", device, gdn_gated_rmsnorm_device_adjoint, nW)

}  // namespace r97
