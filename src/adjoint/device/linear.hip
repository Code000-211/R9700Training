// src/adjoint/device/linear.hip -- the frozen-linears family's device adjoints, gfx1201.
//
// gemm_nt_q writes y[M,N] = a[M,K] * b[N,K]^T with a int8, b int8 codes carrying one bf16 scale per
// output row and 128-wide block of K, y f32. Its adjoint, in the dequantised activation basis, is
// the transposed product
//
//     dx[m,k] = sum_n dy[m,n] * b[n,k] * s_b[n, k/128]
//
// and this is the kernel that carries the backward's arithmetic: bf16 dy against int8 codes, f32
// accumulation. The scale varies with n -- the reduction index here -- and with the block of K, so
// one CTA owns one 128-wide block of dx: it scales each 16-wide slice of dy by s_b[:, block] and
// multiplies that into the block's codes. int8 codes are exact in bf16, so the products run on the
// bf16 WMMA path; that is the decided precision (int8 forward, bf16 backward,
// docs/adjoints/linear.md).
//
// gemm_nt -- the gated-delta-net block's small bf16 projection -- is the same kernel with the codes
// already bf16 and no scale: `Q` picks the staging and whether dy is scaled.
//
// Tile: a 256-row band of M x one 128-wide block of K per CTA, eight waves, each wave two 16-row
// tiles and all eight k steps of the block in flight (16 accumulators of 8 f32 = 128 VGPRs). Every
// wave reads the same staged 16 x 128 weight slice, and the slice for one block is N*128 bytes
// (0.65 MB at the real model's 5120), so blocks are dispatched with M fastest and the slice stays
// in MALL across the bands in flight. dy is read once per block of K: K/128 times per call.
//
// The fragment layout is the one the forward's tiled kernel uses, idx = lane%16 and
// k = 8*(e>>2) + 4*(lane>>4) + (e&3), and the WMMA intrinsic is gfx12's
// __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12; docs/kernels/linear.md writes the layout down
// as this family measured it. A wants [M][K] and B wants [N][K], both K-contiguous. dy is already
// [M][N] in that order, so it is read straight from global. The weight is [N][K] too, but here N is
// the reduction and K is dx's column -- the fragment that WMMA wants for B has its rows on dx's
// columns -- so the staged tile is read with the eight scalar reads of a transposed fragment
// instead of the two of a contiguous one. Staging it transposed in shared memory instead would cost
// four times as many scattered LDS stores as those reads.
#include "r97/adjoint.h"

#include "rad_device.h"

#include "scratch.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// ------------------------------------------------------------------ fragment helpers

typedef __bf16 v8bf __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));

// Software f32 -> bf16 (gfx1201 has no convert), round-half-away, the same rule the host side
// rounds with below.
__device__ __forceinline__ unsigned short f2bf(float f) {
    return static_cast<unsigned short>((__builtin_bit_cast(unsigned, f) + 0x8000u) >> 16);
}

__device__ __forceinline__ unsigned f2bf2(float a, float b) {
    const unsigned ua = __builtin_bit_cast(unsigned, a) + 0x8000u;
    const unsigned ub = __builtin_bit_cast(unsigned, b) + 0x8000u;
    return __builtin_amdgcn_perm(ub, ua, 0x07060302u);
}

__device__ __forceinline__ float bf2f(unsigned short h) {
    return __builtin_bit_cast(float, static_cast<unsigned>(h) << 16);
}

// C += A @ B^T over one 16-wide k step, f32 accumulate.
__device__ __forceinline__ v8f mma(v8bf a, v8bf b, v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

// This lane's row of a fragment source, clamped to the last valid row: the rows past M hold nothing
// worth reading and their results are never stored.
__device__ __forceinline__ unsigned rowGo(int rowstride, int row0, int lim, int lane) {
    int row = row0 + (lane & 15);
    if (row >= lim) row = lim - 1;
    return static_cast<unsigned>(row) * static_cast<unsigned>(rowstride) +
           4u * static_cast<unsigned>(lane >> 4);
}

__device__ __forceinline__ v8bf fragGo(const unsigned short* base, unsigned off) {
    const uint2* pw = reinterpret_cast<const uint2*>(base + off);
    union {
        uint2 w[2];
        v8bf f;
    } u;
    u.w[0] = pw[0];
    u.w[1] = pw[2];
    return u.f;
}

// The same fragment from a source stored transposed, src[k][idx]: eight scalar reads.
__device__ __forceinline__ v8bf fragT(const unsigned short* src, int pitch, int base, int kbase,
                                      int lane) {
    const int idx = base + (lane & 15);
    const int k0 = kbase + 4 * (lane >> 4);
    union {
        unsigned short h[8];
        v8bf f;
    } u;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        u.h[e] = src[static_cast<size_t>(k0 + e) * pitch + idx];
        u.h[4 + e] = src[static_cast<size_t>(k0 + 8 + e) * pitch + idx];
    }
    return u.f;
}

// dy * s, column by column: this lane's eight scales are its eight fragment columns.
__device__ __forceinline__ v8bf scale8(v8bf x, const float* s) {
    union {
        v8bf b;
        unsigned short h[8];
    } in, out;
    in.b = x;
#pragma unroll
    for (int e = 0; e < 8; ++e) out.h[e] = f2bf(bf2f(in.h[e]) * s[e]);
    return out.b;
}

// ------------------------------------------------------------------ the kernel

#define LD_WAVES 8              // waves per CTA, 256 threads
#define LD_TM 2                 // 16-row tiles of dx a wave owns
#define LD_BK 128               // columns of dx per CTA: one w8a8 scale block
#define LD_KS (LD_BK / 16)      // k steps of the block
#define LD_BM (LD_WAVES * LD_TM * 16)
#define LD_BN 16                // output channels staged per reduction step
#define LD_PITCH 136            // staged weight row pitch in elements: 272 B, a multiple of 16
#define LD_TILE (LD_BN * LD_PITCH)  // staged weight tile, one reduction step

// `Q`: int8 codes with a bf16 scale per block, or a bf16 weight already dequantised.
template <bool Q>
__global__ __launch_bounds__(LD_WAVES * 32) void linear_dx_kernel(
    const unsigned short* __restrict__ dy, const void* __restrict__ w,
    const unsigned short* __restrict__ wscale, unsigned short* __restrict__ dx, int M, int N, int K,
    int G) {
    __shared__ __attribute__((aligned(16))) unsigned short sW[LD_TILE];

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int m0 = static_cast<int>(blockIdx.x) * LD_BM;
    const int k0 = static_cast<int>(blockIdx.y) * LD_BK;
    const int g = static_cast<int>(blockIdx.y);  // LD_BK is the block the scale grid uses

    // Staging: one eight-element chunk per thread, sixteen threads to a row of the tile.
    const int r = tid >> 4, e = (tid & 15) * 8;
    unsigned short* const dst = sW + r * LD_PITCH + e;

    unsigned raw[4];
    const auto load = [&](int n0) {
        const int nr = n0 + r < N ? n0 + r : N - 1;
        int ke = k0 + e;
        if (ke + 8 > K) ke = K - 8;
        if (Q) {
            const unsigned* p = reinterpret_cast<const unsigned*>(
                static_cast<const char*>(w) + static_cast<size_t>(nr) * K + ke);
            raw[0] = p[0];
            raw[1] = p[1];
        } else {
            const unsigned* p = reinterpret_cast<const unsigned*>(
                reinterpret_cast<const unsigned short*>(w) + static_cast<size_t>(nr) * K + ke);
            raw[0] = p[0];
            raw[1] = p[1];
            raw[2] = p[2];
            raw[3] = p[3];
        }
    };
    const auto store = [&]() {
        if (Q) {
            const auto code = [&](int i) {
                return static_cast<float>(static_cast<int>(
                    static_cast<signed char>((raw[i >> 2] >> (8 * (i & 3))) & 0xffu)));
            };
            *reinterpret_cast<uint2*>(dst) =
                make_uint2(f2bf2(code(0), code(1)), f2bf2(code(2), code(3)));
            *reinterpret_cast<uint2*>(dst + 4) =
                make_uint2(f2bf2(code(4), code(5)), f2bf2(code(6), code(7)));
        } else {
            *reinterpret_cast<uint2*>(dst) = make_uint2(raw[0], raw[1]);
            *reinterpret_cast<uint2*>(dst + 4) = make_uint2(raw[2], raw[3]);
        }
    };

    v8f acc[LD_TM][LD_KS];
#pragma unroll
    for (int i = 0; i < LD_TM; ++i)
#pragma unroll
        for (int j = 0; j < LD_KS; ++j) acc[i][j] = (v8f)(0.f);

    // This lane's eight scales for one reduction step: the scale is indexed by the output channel
    // the fragment column carries, and every column keeps its position across the loop.
    const unsigned kc = 4u * (lane >> 4);
    const bool have_scale = Q && wscale != nullptr;

    load(0);
    for (int n0 = 0; n0 < N; n0 += LD_BN) {
        __syncthreads();  // the previous step's fragments have all been read
        store();
        __syncthreads();
        if (n0 + LD_BN < N) load(n0 + LD_BN);

        v8bf bf[LD_KS];
#pragma unroll
        for (int kt = 0; kt < LD_KS; ++kt) bf[kt] = fragT(sW, LD_PITCH, kt * 16, 0, lane);

        float scl[8];
#pragma unroll
        for (int c = 0; c < 8; ++c)
            scl[c] = have_scale ? bf2f(wscale[static_cast<size_t>(n0 + 8 * (c >> 2) + kc + (c & 3)) * G + g])
                                : 1.0f;

#pragma unroll
        for (int mt = 0; mt < LD_TM; ++mt) {
            const unsigned off =
                rowGo(N, m0 + (wave * LD_TM + mt) * 16, M, lane) + static_cast<unsigned>(n0);
            v8bf af = fragGo(dy, off);
            if (Q) af = scale8(af, scl);
#pragma unroll
            for (int kt = 0; kt < LD_KS; ++kt) acc[mt][kt] = mma(af, bf[kt], acc[mt][kt]);
        }
        /* Each step's fragment reads, then its products: the same grouping the forward's tiled
         * kernel keeps, so the compiler does not spread the LDS reads among the previous step's
         * WMMAs. */
        __builtin_amdgcn_sched_group_barrier(0x100, 8 * LD_KS, 0);      /* DS reads */
        __builtin_amdgcn_sched_group_barrier(0x008, LD_TM * LD_KS, 0); /* WMMA */
    }

    /* Lane l holds column (l & 15) of a tile and rows 8 (l >> 4) + e: sixteen lanes write 32
     * contiguous bytes of one dx row per step. */
#pragma unroll
    for (int mt = 0; mt < LD_TM; ++mt)
#pragma unroll
        for (int kt = 0; kt < LD_KS; ++kt) {
            const int k = k0 + kt * 16 + (lane & 15);
            const int mb = m0 + (wave * LD_TM + mt) * 16 + 8 * (lane >> 4);
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const int m = mb + c;
                if (k < K && m < M) dx[static_cast<size_t>(m) * K + k] = f2bf(acc[mt][kt][c]);
            }
        }
}

static int linear_dx_launch(bool quant, const void* dy, const void* w, const void* wscale, void* dx,
                            int M, int N, int K, int G, hipStream_t stream) {
    const dim3 block(LD_WAVES * 32);
    const dim3 grid((M + LD_BM - 1) / LD_BM, (K + LD_BK - 1) / LD_BK);
    if (quant)
        hipLaunchKernelGGL((linear_dx_kernel<true>), grid, block, 0, stream,
                           static_cast<const unsigned short*>(dy), w,
                           static_cast<const unsigned short*>(wscale), static_cast<unsigned short*>(dx),
                           M, N, K, G);
    else
        hipLaunchKernelGGL((linear_dx_kernel<false>), grid, block, 0, stream,
                           static_cast<const unsigned short*>(dy), w,
                           static_cast<const unsigned short*>(nullptr),
                           static_cast<unsigned short*>(dx), M, N, K, G);
    return static_cast<int>(hipGetLastError());
}

// ------------------------------------------------------------------ the adjoint rows

namespace r97 {
namespace {

// f64 -> bf16, one rounding to nearest. The hardware's own conversion rounds a tie to even and this
// rounds it away from zero, so the two can differ by one bf16 ulp; staging adds no more than that.
unsigned short to_bf16(double x) {
    const float f = static_cast<float>(x);
    unsigned u;
    std::memcpy(&u, &f, sizeof(u));
    u += 0x8000u;
    return static_cast<unsigned short>(u >> 16);
}

double of_bf16(unsigned short h) {
    const unsigned u = static_cast<unsigned>(h) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return static_cast<double>(f);
}

// An int8 code carried as f64: a weight the IQR decode wrote is exact, and one that is not would
// quantise silently. A value outside the grid is a shape this row was not written for, so the call
// declines it rather than inventing a code.
bool to_code(double x, int8_t* out) {
    const double r = std::round(x);
    if (std::fabs(x - r) > 1e-3 || r < -128.0 || r > 127.0) return false;
    *out = static_cast<int8_t>(r);
    return true;
}

const OpParam* find_param(const std::vector<OpParam>& params, const char* name) {
    for (const OpParam& p : params)
        if (p.first == name) return &p;
    return nullptr;
}

bool param_is(const std::vector<OpParam>& params, const char* name, const char* value) {
    const OpParam* p = find_param(params, name);
    if (p == nullptr) return false;
    const std::string* s = std::get_if<std::string>(&p->second);
    return s != nullptr && *s == value;
}

bool param_int(const std::vector<OpParam>& params, const char* name, int64_t* out) {
    const OpParam* p = find_param(params, name);
    if (p == nullptr) return false;
    const int64_t* v = std::get_if<int64_t>(&p->second);
    if (v == nullptr) return false;
    *out = *v;
    return true;
}

int64_t int_param(const std::vector<OpParam>& params, const char* name, int64_t fallback) {
    int64_t v = 0;
    return param_int(params, name, &v) ? v : fallback;
}

int operand_pos(const OpRecord& rec, const char* name) {
    for (std::size_t i = 0; i < rec.operands.size(); ++i)
        if (rec.operands[i].buffer == name) return static_cast<int>(i);
    return -1;
}

// The scale grid the record carries: one value per output row and block of K, as the host adjoint
// reads it.
double scale_at(const AdValue* scale, int64_t row, int64_t group, int64_t rows, int64_t groups) {
    if (scale == nullptr || scale->value.empty()) return 1.0;
    const int64_t n = static_cast<int64_t>(scale->value.size());
    if (n == 1) return scale->value[0];
    if (n == rows * groups) return scale->value[static_cast<std::size_t>(row * groups + group)];
    if (n == groups) return scale->value[static_cast<std::size_t>(group)];
    if (n == rows) return scale->value[static_cast<std::size_t>(row)];
    const int64_t at = std::min<int64_t>(row * groups + group, n - 1);
    return scale->value[static_cast<std::size_t>(at)];
}

// One stream for every device adjoint call in this process, as add.hip keeps one: the call's
// upload, launch and download are one queue's worth of work. The stream belongs to the device
// rad_dev_set() chose; a test or a trainer selects one device before the first adjoint runs.
RadStream adjoint_stream() {
    static RadStream s = [] {
        RadStream created = nullptr;
        if (rad_stream_create(&created, 0) != 0) return (RadStream) nullptr;
        return created;
    }();
    return s;
}

// dx[m,k] = sum_n dy[m,n] * W[n,k], with W the int8 codes and their per-block scale (`quant`) or
// the bf16 weight as it stands.
void linear_dx_adjoint(const AdjointCall& call, bool quant) {
    if (!call.dc) return;
    const int ia = operand_pos(call.rec, "a");
    const int ib = operand_pos(call.rec, "b");
    const int ibs = operand_pos(call.rec, "b_scale");
    if (ia < 0 || ib < 0 || call.cot.empty()) return;

    const AdValue& a = call.fwd[static_cast<std::size_t>(ia)];
    const AdValue& b = call.fwd[static_cast<std::size_t>(ib)];
    if (static_cast<std::size_t>(ia) >= call.dc->size()) return;
    AdValue& dx = (*call.dc)[static_cast<std::size_t>(ia)];
    const AdValue* bs = ibs >= 0 ? &call.fwd[static_cast<std::size_t>(ibs)] : nullptr;
    const AdValue& dy = call.cot[0];

    const int64_t K = int_param(call.rec.params, "K", a.shape.size() >= 2 ? a.shape[1] : 0);
    const int64_t N = int_param(call.rec.params, "N", b.shape.size() >= 2 ? b.shape[0] : 0);
    const int64_t M = int_param(call.rec.params, "M", a.shape.empty() ? 0 : a.shape[0]);
    const int64_t group = int_param(call.rec.params, "group", K > 0 ? K : 1);
    const int64_t groups = group > 0 ? (K + group - 1) / group : 1;
    if (M <= 0 || N <= 0 || K <= 0) return;
    /* The operand extents, the write's shape and the record's M, N, K have to agree; a record that
     * does not is one this row was not written for, and an adjoint must not throw. */
    if (M * N != static_cast<int64_t>(dy.value.size()) ||
        M * K != static_cast<int64_t>(a.value.size()) ||
        N * K != static_cast<int64_t>(b.value.size()) ||
        static_cast<int64_t>(dx.value.size()) != M * K)
        return;
    /* Only the int8 grid has a scale grid, and the kernel's block of K is one of its 128-wide
     * blocks; a bf16 weight carries none, and `group` falls back to K there. */
    if (quant && group != LD_BK) return;
    if (K < 8 || K % 8 != 0) return;
    /* The reduction tile is 16 output channels wide, so a narrow projection -- the fixture's eight-
     * and twelve-channel ones -- gets zero rows of weight and zero dy columns out to 16. They add
     * nothing to the sum and every fragment read stays inside its buffer. */
    const int64_t Np = (N + LD_BN - 1) / LD_BN * LD_BN;

    /* The fragment offsets A reads with are 32-bit; the real model's largest band is 1.4e8
     * elements, so anything near the limit is a shape this kernel was not written for. */
    if (static_cast<double>(M) * static_cast<double>(Np) >= 2147483648.0 ||
        static_cast<double>(M) * static_cast<double>(K) >= 2147483648.0)
        return;

    RadStream stream = adjoint_stream();
    if (!stream) return;

    const std::size_t dy_n = static_cast<std::size_t>(M * Np);
    const std::size_t w_n = static_cast<std::size_t>(Np * K);
    const std::size_t dx_n = static_cast<std::size_t>(M * K);
    const std::size_t sc_n = static_cast<std::size_t>(Np * groups);

    /* Staging: the tape's dy and weight are f64, the kernel's are bf16 (the weight's codes int8),
     * so this call converts them on the way in and the answer on the way out. */
    std::vector<unsigned short> host_dy(dy_n, 0);
    for (int64_t m = 0; m < M; ++m) {
        const double* src = dy.value.data() + static_cast<std::size_t>(m * N);
        unsigned short* dst = host_dy.data() + static_cast<std::size_t>(m * Np);
        for (int64_t n = 0; n < N; ++n) dst[n] = to_bf16(src[n]);
    }
    std::vector<int8_t> host_code(quant ? w_n : 0, 0);
    std::vector<unsigned short> host_w(quant ? 0 : w_n, 0);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const std::size_t i = static_cast<std::size_t>(n * K + k);
            if (quant) {
                if (!to_code(b.value[i], &host_code[i])) return;
            } else {
                host_w[i] = to_bf16(b.value[i]);
            }
        }
    std::vector<unsigned short> host_sc(quant ? sc_n : 0, to_bf16(1.0));
    if (quant)
        for (int64_t n = 0; n < N; ++n)
            for (int64_t g = 0; g < groups; ++g)
                host_sc[static_cast<std::size_t>(n * groups + g)] =
                    to_bf16(scale_at(bs, n, g, N, groups));
    std::vector<unsigned short> host_dx(dx_n);

    device::ScratchBlock dev_dy(static_cast<int64_t>(dy_n * 2));
    device::ScratchBlock dev_w(static_cast<int64_t>(quant ? w_n : w_n * 2));
    device::ScratchBlock dev_sc(static_cast<int64_t>(quant ? sc_n * 2 : 0));
    device::ScratchBlock dev_dx(static_cast<int64_t>(dx_n * 2));
    if (!dev_dy || !dev_w || (quant && !dev_sc) || !dev_dx) return;

    if (rad_memcpy_async(dev_dy.p, host_dy.data(), static_cast<int64_t>(dy_n * 2), stream) != 0)
        return;
    if (rad_memcpy_async(dev_w.p, quant ? static_cast<const void*>(host_code.data())
                                       : static_cast<const void*>(host_w.data()),
                         static_cast<int64_t>(quant ? w_n : w_n * 2), stream) != 0)
        return;
    if (quant && rad_memcpy_async(dev_sc.p, host_sc.data(), static_cast<int64_t>(sc_n * 2), stream) != 0)
        return;

    const int rc = linear_dx_launch(quant, dev_dy.p, dev_w.p, quant ? dev_sc.p : nullptr, dev_dx.p,
                                    static_cast<int>(M), static_cast<int>(Np), static_cast<int>(K),
                                    static_cast<int>(groups), static_cast<hipStream_t>(stream));
    if (rc != 0) return;  // a failed launch leaves dc as the host row's would if it declined

    if (rad_stream_sync(stream) != 0) return;
    if (rad_memcpy_async(host_dx.data(), dev_dx.p, static_cast<int64_t>(dx_n * 2), stream) != 0)
        return;
    if (rad_stream_sync(stream) != 0) return;

    for (std::size_t i = 0; i < dx_n; ++i) dx.value[i] += of_bf16(host_dx[i]);
}

void gemm_nt_q_device(const AdjointCall& call) { linear_dx_adjoint(call, true); }
void gemm_nt_device(const AdjointCall& call) { linear_dx_adjoint(call, false); }

bool serves_i8a8h(const std::vector<OpParam>& params) {
    int64_t group = 0;
    return param_is(params, "dtype", "i8a8h") && param_int(params, "group", &group) && group == LD_BK;
}

bool serves_bf16(const std::vector<OpParam>& params) { return param_is(params, "dtype", "bf16"); }

}  // namespace
}  // namespace r97

// The frozen positions are the host rows' (docs/adjoints/linear.md): the activation scale, the codes
// and their scale for the int8 linear, the weight alone for the bf16 one.
R97_REGISTER_ADJOINT_IF_F("gemm_nt_q", device, r97::gemm_nt_q_device, r97::serves_i8a8h, 1, 2, 3);
R97_REGISTER_ADJOINT_IF_F("gemm_nt", device, r97::gemm_nt_device, r97::serves_bf16, 1);
