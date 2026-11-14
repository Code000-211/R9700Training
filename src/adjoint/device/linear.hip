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
//
// gather_rows -- the head's gather of the block's last hidden row, this family's other row -- has
// no product at all: its adjoint scatters dy's rows back to the rows of x they came from,
// dx[idx[i],:] += dy[i,:], and nothing else. One CTA per row of dy, the accumulation an f32
// atomic into a zeroed dx, which the row rounds to the op's dtype once on the way out.
#include "r97/adjoint.h"

#include "rad_device.h"

#include "scratch.h"
// plane_index.h -- the stored planes' addressing, shared with the IQR decode.
#include "plane_index.h"
#include "iqr_plane.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
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

// ------------------------------------------------------------------ the engine's stored layout
//
// A `w8a8_fwht128` container hands the tape its weight and its scale as plugin-private operands:
// no f64 form, no byte span, one pointer each. The pin stores them in two planes, and the forward
// GEMM reads exactly these (third_party/radiance/libr4d/r4d_fp8_layout.cpp):
//
//   codes  int8 [N][K], tag "r4d.i8h.w.frag64": a (16 row, 64 k) tile of 1024 bytes laid out as
//          32 lanes x 32 bytes. Lane l's 32 bytes are its four 8-byte fragments, bytes 8t of them
//          covering k = t*16 + (l>>4)*8 .. +7 of row l&15 (libr4d/r4d_fp8_frag.h, frag_src; the
//          host relayout walks the same permutation, r4d_rad_relayout_i8_block's frag_walk).
//   scale  bf16, tag "r4d.i8.s.t16": tile-major [N/16][K/128][16], the sixteen rows of a fragment
//          tile side by side for one 128-K block -- d[((r/16)*G + g)*16 + r%16] = s[r*G + g], the
//          loop in r4d_rad_relayout_i8_block.
//
// The kernel reads eight consecutive k of one weight row, which frag64 keeps contiguous inside one
// fragment, so one 8-byte group is one address; and the scale a fragment column carries is that
// row's, in the tile's row slot. K is a multiple of 128 (the scale grid's block) and N of 16 (the
// fragment's rows) for any weight the i8a8h GEMM runs, so neither index leaves its plane.

// The byte offset of the 8-byte fragment that starts at column `k` (k % 8 == 0) of `row`, and the
// element index of (row, block `g` of K) in the tile-major bf16 scale plane, are plane_index.h --
// shared with the IQR decode, which writes this same order into a staging plane.

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
// `STORED`: the weight and its scale are the engine's plugin-private planes -- frag64 codes and a
// t16 scale, the helpers above -- so `w` and `wscale` are the card's pointers into them, not
// row-major copies. Only `Q` has a stored form.
template <bool Q, bool STORED = false>
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
            const size_t off =
                STORED ? frag_go(K, nr, ke) : static_cast<size_t>(nr) * K + ke;
            const unsigned* p =
                reinterpret_cast<const unsigned*>(static_cast<const char*>(w) + off);
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
        for (int c = 0; c < 8; ++c) {
            const int row_c = n0 + 8 * (c >> 2) + kc + (c & 3);
            const size_t at = STORED ? t16_go(G, row_c, g) : static_cast<size_t>(row_c) * G + g;
            scl[c] = have_scale ? bf2f(wscale[at]) : 1.0f;
        }

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

static int linear_dx_launch(bool quant, bool stored, const void* dy, const void* w,
                            const void* wscale, void* dx, int M, int N, int K, int G,
                            hipStream_t stream) {
    const dim3 block(LD_WAVES * 32);
    const dim3 grid((M + LD_BM - 1) / LD_BM, (K + LD_BK - 1) / LD_BK);
    if (stored)
        hipLaunchKernelGGL((linear_dx_kernel<true, true>), grid, block, 0, stream,
                           static_cast<const unsigned short*>(dy), w,
                           static_cast<const unsigned short*>(wscale),
                           static_cast<unsigned short*>(dx), M, N, K, G);
    else if (quant)
        hipLaunchKernelGGL((linear_dx_kernel<true, false>), grid, block, 0, stream,
                           static_cast<const unsigned short*>(dy), w,
                           static_cast<const unsigned short*>(wscale),
                           static_cast<unsigned short*>(dx), M, N, K, G);
    else
        hipLaunchKernelGGL((linear_dx_kernel<false, false>), grid, block, 0, stream,
                           static_cast<const unsigned short*>(dy), w,
                           static_cast<const unsigned short*>(nullptr),
                           static_cast<unsigned short*>(dx), M, N, K, G);
    return static_cast<int>(hipGetLastError());
}

// ------------------------------------------------------------------ the gather's kernel

// One CTA per row of dy: the thread walks the width in stride and adds the row to the row of x
// the index names. The accumulation is the only arithmetic the backward asks of this op, and a
// row gathered by several gathers must be their sum, so it is an f32 atomic into a zeroed dx --
// there is no order to the sum to exploit, and the f32 of the atomic is the kernel's accumulation
// type, as the row's f64 -> bf16 staging and one bf16 rounding on the way out make the rest of it
// the same precision the host row's f64 sum is held to in the test. A negative index -- the zero
// row a short prefill leaves -- and a row past the end of x contribute nothing.
#define GR_THREADS 256

__global__ __launch_bounds__(GR_THREADS) void gather_rows_dx_kernel(
    const unsigned short* __restrict__ dy, const int* __restrict__ idx, float* __restrict__ dx,
    int rows, int width) {
    const int i = static_cast<int>(blockIdx.x);
    const int r = idx[i];
    if (r < 0 || r >= rows) return;
    const unsigned short* const src = dy + static_cast<size_t>(i) * width;
    float* const dst = dx + static_cast<size_t>(r) * width;
    for (int c = threadIdx.x; c < width; c += blockDim.x) atomicAdd(&dst[c], bf2f(src[c]));
}

int gather_rows_dx_launch(const void* dy, const void* idx, void* dx, int M, int rows, int width,
                          hipStream_t stream) {
    const dim3 block(GR_THREADS);
    const dim3 grid(M);
    hipLaunchKernelGGL(gather_rows_dx_kernel, grid, block, 0, stream,
                       static_cast<const unsigned short*>(dy), static_cast<const int*>(idx),
                       static_cast<float*>(dx), rows, width);
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

// A record the IQR path cannot serve names itself, and throws. Every return on that path leaves
// `dx` as it was, which is the shape of the 399-of-400 zero-gradient bug with a new cause: a reader
// of a run's JSON could not tell "the decode produced nothing" from "the row declined". The message
// carries the form, so the run's log says which guard fired rather than only that something did.
// Nothing is in flight when this is called: each of the call sites below is a check that failed
// before its launch, or a launch the driver refused.
void iqr_decline(const char* why, const OpRecord& rec, const AdValue& b, int64_t N, int64_t K,
                 int64_t groups) {
    char text[320];
    std::snprintf(text, sizeof text,
                  "r97: iqr dx declined: op %s: %s (N %lld K %lld G %lld form %s)",
                  rec.op.c_str(), why, static_cast<long long>(N), static_cast<long long>(K),
                  static_cast<long long>(groups), device::iqr_plane_form(b.enc));
    std::fprintf(stderr, "%s\n", text);
    std::fflush(stderr);
    throw std::runtime_error(text);
}

// dx[m,k] = sum_n dy[m,n] * W[n,k], with W the int8 codes and their per-block scale (`quant`) or
// the bf16 weight as it stands. `stored` is the plugin-private pair the w8a8_fwht128 trunk hands
// the tape: the weight and its scale have no f64 form, so the card's own frag64/t16 planes are
// read in place and frag_go/t16_go is the decode. The f64 path is untouched for a record whose
// operands do have values.
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

    /* The plugin-private pair: no f64 values, one pointer each into the engine's stored planes. A
     * bf16 weight with no values is a record this row does not serve. */
    const bool stored = quant && b.value.empty() && b.raw != nullptr;
    /* The plugin-private pair is one of two stored forms. The w8a8_fwht128 container's is the int8
     * pair above; the IQR trunk's (an IQR-V4 trellis at 4.81 bpw, so the codes plane is not N*K
     * bytes) is a trellis or a table whose form rides with the operand (AdValue::enc), and the row
     * decodes it with libr4d's own decoders into the two planes this kernel reads. Which one is the
     * encoding's business, never the op's name: `codes=i8` is the int8 pair, and every sub-byte
     * codes plane is an IQR form. A narrow encoding this build cannot decode is refused here, by
     * name -- falling through to the int8 reader would index N*K bytes into a 4.81-bpw plane and
     * fault, which is the loudest wrong answer available. */
    const bool narrow = stored && b.enc != nullptr && device::iqr_plane_narrow(b.enc);
    const bool iqr = narrow && device::iqr_plane_supported(b.enc);

    const int64_t K = int_param(call.rec.params, "K", a.shape.size() >= 2 ? a.shape[1] : 0);
    const int64_t N = int_param(call.rec.params, "N", b.shape.size() >= 2 ? b.shape[0] : 0);
    const int64_t M = int_param(call.rec.params, "M", a.shape.empty() ? 0 : a.shape[0]);
    const int64_t group = int_param(call.rec.params, "group", K > 0 ? K : 1);
    const int64_t groups = group > 0 ? (K + group - 1) / group : 1;
    if (M <= 0 || N <= 0 || K <= 0) return;
    if (narrow && !iqr)
        iqr_decline("the weight's encoding is an IQR form this build cannot decode", call.rec, b, N,
                    K, groups);
    /* The operand extents, the write's shape and the record's M, N, K have to agree; a record that
     * does not is one this row was not written for, and returns. A *stored plane* the row cannot
     * decode is the other case: it throws, because a zero dx there is indistinguishable from a
     * correct zero and that is exactly how #55 hid. */
    if (M * N != static_cast<int64_t>(dy.value.size()) ||
        M * K != static_cast<int64_t>(a.value.size()) ||
        static_cast<int64_t>(dx.value.size()) != M * K)
        return;
    if (stored) {
        /* Both stored forms keep the same geometry: a scale grid 128 wide, whole planes, and the
         * weight's own shape. Only the plane's layout differs -- the int8 pair's, or the IQR form
         * the encoding names -- so what a record has to satisfy is the same either way, and one
         * that does not fit is one this row was not written for. The IQR path names itself: a
         * silent return here is the 399-of-400 zero-gradient signature with a new cause. */
        const char* why = nullptr;
        if (group != LD_BK) why = "the scale grid is not 128 wide";
        else if (bs == nullptr || bs->raw == nullptr || !bs->value.empty())
            why = "the scale operand carries no stored plane";
        else if (b.shape.size() != 2 || b.shape[0] != N || b.shape[1] != K)
            why = "the weight's shape is not the record's [N, K]";
        else if (N % LD_BN != 0 || K % LD_BK != 0)
            why = "N or K is not a whole number of fragments";
        if (why != nullptr) {
            if (iqr) iqr_decline(why, call.rec, b, N, K, groups);
            return;
        }
    } else if (N * K != static_cast<int64_t>(b.value.size())) {
        return;
    }
    /* Only the int8 grid has a scale grid, and the kernel's block of K is one of its 128-wide
     * blocks; a bf16 weight carries none, and `group` falls back to K there. */
    if (quant && group != LD_BK) return;
    if (K < 8 || K % 8 != 0) return;
    /* The reduction tile is 16 output channels wide, so a narrow projection -- the fixture's eight-
     * and twelve-channel ones -- gets zero rows of weight and zero dy columns out to 16. They add
     * nothing to the sum and every fragment read stays inside its buffer. The stored planes cannot
     * be padded that way (the engine's geometry is the weight's), so they are required whole. */
    const int64_t Np = stored ? N : (N + LD_BN - 1) / LD_BN * LD_BN;

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

    /* Staging: the tape's dy is f64 and the kernel's bf16, so this call converts it on the way in
     * and the answer on the way out. The weight and its scale are staged the same way for the f64
     * path (f64 codes -> int8, f64 scale -> bf16) and read in place for the stored one. */
    std::vector<unsigned short> host_dy(dy_n, 0);
    for (int64_t m = 0; m < M; ++m) {
        const double* src = dy.value.data() + static_cast<std::size_t>(m * N);
        unsigned short* dst = host_dy.data() + static_cast<std::size_t>(m * Np);
        for (int64_t n = 0; n < N; ++n) dst[n] = to_bf16(src[n]);
    }
    std::vector<int8_t> host_code;
    std::vector<unsigned short> host_w;
    std::vector<unsigned short> host_sc;
    if (!stored) {
        host_code.assign(quant ? w_n : 0, 0);
        host_w.assign(quant ? 0 : w_n, 0);
        for (int64_t n = 0; n < N; ++n)
            for (int64_t k = 0; k < K; ++k) {
                const std::size_t i = static_cast<std::size_t>(n * K + k);
                if (quant) {
                    if (!to_code(b.value[i], &host_code[i])) return;
                } else {
                    host_w[i] = to_bf16(b.value[i]);
                }
            }
        host_sc.assign(quant ? sc_n : 0, to_bf16(1.0));
        if (quant)
            for (int64_t n = 0; n < N; ++n)
                for (int64_t g = 0; g < groups; ++g)
                    host_sc[static_cast<std::size_t>(n * groups + g)] =
                        to_bf16(scale_at(bs, n, g, N, groups));
    }
    std::vector<unsigned short> host_dx(dx_n);

    device::ScratchBlock dev_dy(static_cast<int64_t>(dy_n * 2));
    device::ScratchBlock dev_w(static_cast<int64_t>(stored ? 0 : (quant ? w_n : w_n * 2)));
    device::ScratchBlock dev_sc(static_cast<int64_t>(stored ? 0 : (quant ? sc_n * 2 : 0)));
    device::ScratchBlock dev_dx(static_cast<int64_t>(dx_n * 2));
    if (!dev_dy || (!stored && !dev_w) || (!stored && quant && !dev_sc) || !dev_dx) return;

    /* An IQR form's two planes, decoded on the card for this call: the codes into the int8
     * fragment order this kernel reads, and the form's scale into its tile-major bf16 plane. Both
     * are derived from the encoding, which is why a record whose weight is not this form never
     * reaches here. */
    device::ScratchBlock dev_iqr_w(iqr ? Np * K : 0);
    device::ScratchBlock dev_iqr_s(iqr ? Np * groups * 2 : 0);
    if (iqr) {
        if (!dev_iqr_w || !dev_iqr_s) {
            iqr_decline("the decode's scratch could not be allocated", call.rec, b, N, K, groups);
            return;
        }
        if (!device::iqr_plane_decode(b.enc, static_cast<uint32_t>(N), static_cast<uint32_t>(K),
                                      b.raw, bs->raw, dev_iqr_w.p, stream)) {
            iqr_decline("the codes plane did not decode", call.rec, b, N, K, groups);
            return;
        }
        if (!device::iqr_plane_scale(b.enc, static_cast<uint32_t>(N), static_cast<uint32_t>(K),
                                     bs->raw, dev_iqr_s.p, stream)) {
            iqr_decline("the scale plane did not decode", call.rec, b, N, K, groups);
            return;
        }
    }

    if (rad_memcpy_async(dev_dy.p, host_dy.data(), static_cast<int64_t>(dy_n * 2), stream) != 0)
        return;
    if (!stored) {
        if (rad_memcpy_async(dev_w.p, quant ? static_cast<const void*>(host_code.data())
                                           : static_cast<const void*>(host_w.data()),
                             static_cast<int64_t>(quant ? w_n : w_n * 2), stream) != 0)
            return;
        if (quant &&
            rad_memcpy_async(dev_sc.p, host_sc.data(), static_cast<int64_t>(sc_n * 2), stream) != 0)
            return;
    }

    /* The card's own planes for the stored pair -- the operand's for the int8 layout, the decode's
     * for an IQR form -- and the staged copies otherwise. */
    const void* const w_ptr = iqr ? dev_iqr_w.p : (stored ? b.raw : dev_w.p);
    const void* const sc_ptr =
        iqr ? dev_iqr_s.p : (stored ? bs->raw : (quant ? dev_sc.p : nullptr));
    const int rc = linear_dx_launch(quant, stored, dev_dy.p, w_ptr, sc_ptr, dev_dx.p,
                                    static_cast<int>(M), static_cast<int>(Np), static_cast<int>(K),
                                    static_cast<int>(groups), static_cast<hipStream_t>(stream));
    if (rc != 0) {
        if (iqr) iqr_decline("the dx kernel did not launch", call.rec, b, N, K, groups);
        return;  // a failed launch leaves dc as the host row's would if it declined
    }

    if (rad_stream_sync(stream) != 0) return;
    if (rad_memcpy_async(host_dx.data(), dev_dx.p, static_cast<int64_t>(dx_n * 2), stream) != 0)
        return;
    if (rad_stream_sync(stream) != 0) return;

    for (std::size_t i = 0; i < dx_n; ++i) dx.value[i] += of_bf16(host_dx[i]);
}

// The gather's row: dy staged bf16, the index staged i32, dx zeroed f32 and accumulated on the
// card, rounded to the op's dtype once on the way out and added into the slot as f64. The row
// serves the bf16 records only; another dtype keeps whatever the host row provides.
void gather_rows_dx_adjoint(const AdjointCall& call) {
    if (!call.dc) return;
    const int ix = operand_pos(call.rec, "x");
    const int ii = operand_pos(call.rec, "idx");
    if (ix < 0 || ii < 0 || call.cot.empty()) return;
    const AdValue& x = call.fwd[static_cast<std::size_t>(ix)];
    const AdValue& idx = call.fwd[static_cast<std::size_t>(ii)];
    if (static_cast<std::size_t>(ix) >= call.dc->size()) return;
    AdValue& dx = (*call.dc)[static_cast<std::size_t>(ix)];
    const AdValue& dy = call.cot[0];

    const int64_t width = int_param(call.rec.params, "n", dy.shape.size() >= 2 ? dy.shape[1] : 1);
    const int64_t M = dy.shape.empty() ? 0 : dy.shape[0];
    const int64_t rows = x.shape.empty() ? 0 : x.shape[0];
    if (M <= 0 || rows <= 0 || width <= 0) return;
    /* The operand extents and the slot have to agree, and the index is discrete: a record that
     * does not is one this row was not written for, and an adjoint must not throw. */
    if (M * width != static_cast<int64_t>(dy.value.size()) ||
        static_cast<int64_t>(idx.value.size()) < M ||
        rows * width != static_cast<int64_t>(x.value.size()) ||
        static_cast<int64_t>(dx.value.size()) != rows * width)
        return;
    /* The kernel's offsets are 32-bit, and an index that is not a whole row is a value this row
     * was not handed. */
    if (static_cast<double>(M) * static_cast<double>(width) >= 2147483648.0 ||
        static_cast<double>(rows) * static_cast<double>(width) >= 2147483648.0)
        return;
    for (int64_t i = 0; i < M; ++i) {
        const double v = idx.value[static_cast<std::size_t>(i)];
        if (v != std::floor(v) || v < -2147483648.0 || v > 2147483647.0) return;
    }

    RadStream stream = adjoint_stream();
    if (!stream) return;

    const std::size_t dy_n = static_cast<std::size_t>(M * width);
    const std::size_t idx_n = static_cast<std::size_t>(M);
    const std::size_t dx_n = static_cast<std::size_t>(rows * width);

    /* Staging: the tape's dy is f64 and the kernel's bf16, the index is f64 and the kernel's i32. */
    std::vector<unsigned short> host_dy(dy_n);
    for (std::size_t i = 0; i < dy_n; ++i) host_dy[i] = to_bf16(dy.value[i]);
    std::vector<int> host_idx(idx_n);
    for (std::size_t i = 0; i < idx_n; ++i) host_idx[i] = static_cast<int>(idx.value[i]);
    std::vector<float> host_dx(dx_n);

    device::ScratchBlock dev_dy(static_cast<int64_t>(dy_n * 2));
    device::ScratchBlock dev_idx(static_cast<int64_t>(idx_n * 4));
    device::ScratchBlock dev_dx(static_cast<int64_t>(dx_n * 4));
    if (!dev_dy || !dev_idx || !dev_dx) return;

    if (rad_memcpy_async(dev_dy.p, host_dy.data(), static_cast<int64_t>(dy_n * 2), stream) != 0)
        return;
    if (rad_memcpy_async(dev_idx.p, host_idx.data(), static_cast<int64_t>(idx_n * 4), stream) != 0)
        return;
    if (rad_memset_async(dev_dx.p, 0, static_cast<int64_t>(dx_n * 4), stream) != 0) return;

    const int rc = gather_rows_dx_launch(dev_dy.p, dev_idx.p, dev_dx.p, static_cast<int>(M),
                                         static_cast<int>(rows), static_cast<int>(width),
                                         static_cast<hipStream_t>(stream));
    if (rc != 0) return;  // a failed launch leaves dc as the host row's would if it declined

    if (rad_stream_sync(stream) != 0) return;
    if (rad_memcpy_async(host_dx.data(), dev_dx.p, static_cast<int64_t>(dx_n * 4), stream) != 0)
        return;
    if (rad_stream_sync(stream) != 0) return;

    /* The f32 the card accumulated is rounded to the op's dtype -- bf16, which the predicate
     * serves -- before it is added into the slot, the one rounding the host row's f64 sum is
     * compared against in the test. */
    for (std::size_t i = 0; i < dx_n; ++i)
        dx.value[i] += of_bf16(to_bf16(static_cast<double>(host_dx[i])));
}

void gemm_nt_q_device(const AdjointCall& call) { linear_dx_adjoint(call, true); }
void gemm_nt_device(const AdjointCall& call) { linear_dx_adjoint(call, false); }
void gather_rows_device(const AdjointCall& call) { gather_rows_dx_adjoint(call); }

bool serves_i8a8h(const std::vector<OpParam>& params) {
    int64_t group = 0;
    return param_is(params, "dtype", "i8a8h") && param_int(params, "group", &group) && group == LD_BK;
}

bool serves_bf16(const std::vector<OpParam>& params) { return param_is(params, "dtype", "bf16"); }

}  // namespace
}  // namespace r97

// The frozen positions are the host rows' (docs/adjoints/linear.md): the activation scale, the codes
// and their scale for the int8 linear, the weight alone for the bf16 one, the index for the gather.
R97_REGISTER_ADJOINT_IF_F("gemm_nt_q", device, r97::gemm_nt_q_device, r97::serves_i8a8h, 1, 2, 3);
R97_REGISTER_ADJOINT_IF_F("gemm_nt", device, r97::gemm_nt_device, r97::serves_bf16, 1);
R97_REGISTER_ADJOINT_IF_F("gather_rows", device, r97::gather_rows_device, r97::serves_bf16, 1);
