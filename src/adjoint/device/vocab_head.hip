// src/adjoint/device/vocab_head.hip -- the vocabulary head's backward, on the card.
//
// The forward (libr4d's r4d_logits_vocab) computes, per class, the logits the class's rows own:
// the hidden state rotated through the class's transform, against the class's IQR-decoded weight
// and its per-row scale and per-128 factor, scattered by the class's vocabulary ids. The adjoint
// this file runs is its transpose, accumulated into the hidden state's cotangent d_x [rows,
// n_embd] f32:
//
//   d_x  +=  H  [  sum over classes c:  (d_y[:, ids_c]  W_c)  *  scale.col_c  ]
//
// where H is the transform's basis applied per 128 columns when the encoding carries one, d_y is
// the cotangent the cross-entropy leaves [rows, vocab] bf16, and W_c is the class's decoded int8
// grid times its settled per-row, per-128 scale. The rows are independent, so the caller chunked
// the call's rows and d_x is added to, not assigned.
//
// The transform: the encoding's contract (abi/rad_encoding.h) is that the stored row is
// v = w . T with T = H/128 block-diagonal, and that the activation is rotated by the
// unnormalised H with nothing folded at runtime -- the 1/128 lives in the stored scale. The
// forward therefore computes (H x) . v^T, and the adjoint of that map is (d_y . v) . H^T. H is
// symmetric, so this file applies the same unnormalised H the forward's rotation is -- the
// seven butterfly stages, no factor. A 1/128 or a 1/sqrt(128) here would be a silent scale
// error: the normalisation is the scale's, and the scale is already on the weight side.
//
// The weight never comes from the tape: the engine serves it to the forward in the IQR word form
// its GEMM reads, not the stored plane. The plan the join built (src/vocab_head.cpp) holds the
// stored planes in device memory instead, and this kernel decodes them chunk by chunk -- the
// decode's int8 grid and its scales land in scratch the call owns, never in a 248320-wide
// buffer.
//
// The launch follows docs/kernels/README.md's discipline: it runs on the stream the caller hands
// in, it does not sync, and the caller's rad_stream_sync on that stream is what orders the read
// of d_x after it.

#include "../../iqr_decode.h"
#include "../../vocab_head.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <string>

#include "rad_device.h"

#include "scratch.h"

namespace {

constexpr int kThreads = 256;

// The decode's row range one sub-range is: the largest that stays a whole number of trellis runs
// and keeps the sub-range's int8 grid, its scales and the gather's buffer small beside the card.
// It must stay a multiple of the head form's run of 256 rows.
constexpr int64_t kDecodeRows = 4096;

// A device buffer may only go back to the allocator once the queue that reads it has drained: a
// launch is asynchronous, so the work this call queues outlives the end of the function by
// however long the card takes. Declared after the scratch blocks (so it is destroyed before them),
// it drains the stream on every path -- including the ones that return early on a failed copy,
// where an unsynchronised return would leave the decode writing into freed memory.
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

bool launch_check(const char* what, std::string* error) {
    const hipError_t e = hipGetLastError();
    if (e == hipSuccess) return true;
    if (error) *error = std::string("vocab head: ") + what + " failed: " + hipGetErrorString(e);
    return false;
}

// f32 -> bf16, the cast the join's loss wants its logits in.
__global__ void vocab_cast_kernel(const float* src, uint16_t* dst, int64_t total) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    dst[i] = r97_iqr_f32_to_bf16(src[i]);
}

// d_y's rows for the sub-range, gathered into the class's ids order, so the product below is a
// plain bf16 grid against the decode's int8 grid. An id outside this rank's vocabulary
// (a record and a container that disagree) contributes zero, as the forward's own guard does.
__global__ void vocab_gather_kernel(const uint16_t* dy, const int32_t* ids, uint16_t* a,
                                    int64_t rows, int64_t vocab, int64_t vocab_offset,
                                    int64_t ns) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * ns) return;
    const int64_t m = i / ns;
    const int64_t n = i - m * ns;
    const int64_t id = ids[n];
    a[i] = (vocab_offset <= id && id < vocab_offset + vocab)
               ? dy[m * vocab + (id - vocab_offset)]
               : (uint16_t)0;
}

// One CTA per token row: d_x[m, :] += A[m, :] W_sub, A the gathered cotangent [rows, ns] bf16,
// W_sub the sub-range's int8 grid [ns, K] with its settled per-row scale and the class's per-128
// factor. The CTA's 256 threads stride the (kb, j8) pairs of the row's 128-wide blocks, so each
// of d_x[m, :] is read and written by exactly one thread of the launch, and the sub-ranges and
// classes accumulate in stream order -- the read-modify-write needs no atomic.
__global__ void vocab_gemv_kernel(const uint16_t* a, const signed char* codes,
                                  const uint16_t* gscale, const uint8_t* cf, int32_t cf_dtype,
                                  float* d_x, int64_t K, int64_t ns) {
    const int64_t m = (int64_t)blockIdx.x;
    const int nkb = (int)(K >> 7);
    const uint16_t* am = a + m * ns;
    float* dx = d_x + m * K;
    for (int p = threadIdx.x; p < nkb * 8; p += 256) {
        const int kb = p >> 3;
        const int j8 = p & 7;
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
        for (int64_t n = 0; n < ns; ++n) {
            const float ag = r97_iqr_bf16_to_f32(am[n]) *
                             r97_iqr_bf16_to_f32(gscale[n * nkb + kb]);
            const signed char* c8 = codes + n * K + kb * 128 + j8 * 8;
            acc0 = fmaf(ag, (float)c8[0], acc0);
            acc1 = fmaf(ag, (float)c8[1], acc1);
            acc2 = fmaf(ag, (float)c8[2], acc2);
            acc3 = fmaf(ag, (float)c8[3], acc3);
            acc4 = fmaf(ag, (float)c8[4], acc4);
            acc5 = fmaf(ag, (float)c8[5], acc5);
            acc6 = fmaf(ag, (float)c8[6], acc6);
            acc7 = fmaf(ag, (float)c8[7], acc7);
        }
        const float c = r97_iqr_load_scalar(cf, cf_dtype, kb);
        float* o = dx + kb * 128 + j8 * 8;
        o[0] += c * acc0;
        o[1] += c * acc1;
        o[2] += c * acc2;
        o[3] += c * acc3;
        o[4] += c * acc4;
        o[5] += c * acc5;
        o[6] += c * acc6;
        o[7] += c * acc7;
    }
}

// The transform's basis, applied per (row, 128-block): the seven butterfly stages in shared
// memory, the stages quant.hip's mask_rotate_kernel runs. No scale factor: the file header's
// derivation, the 1/128 is the stored scale's and it is already on the weight side.
__global__ void vocab_unrotate_kernel(float* d_x, int64_t K) {
    __shared__ float v[128];
    const int tid = (int)threadIdx.x;
    const int64_t base = (int64_t)blockIdx.x * K + (int64_t)blockIdx.y * 128;
    v[tid] = d_x[base + tid];
    __syncthreads();  // the first stage reads lanes other than the one this thread wrote
    for (int h = 1; h < 128; h <<= 1) {
        if (tid < 64) {
            const int j = (tid / h) * (2 * h) + (tid % h);
            const float a = v[j], b = v[j + h];
            v[j] = a + b;
            v[j + h] = a - b;
        }
        __syncthreads();
    }
    d_x[base + tid] = v[tid];
}

}  // namespace

namespace r97 {
namespace {

bool vocab_head_cast_fn(void* dst, const void* src, int64_t values, RadStream stream) {
    if (dst == nullptr || src == nullptr || values <= 0 || stream == nullptr) return false;
    dim3 grid((unsigned)((values + kThreads - 1) / kThreads));
    hipLaunchKernelGGL(vocab_cast_kernel, grid, dim3(kThreads), 0, (hipStream_t)stream,
                       (const float*)src, (uint16_t*)dst, values);
    return hipGetLastError() == hipSuccess;
}

bool vocab_head_transpose_fn(const VocabHeadPlan& plan, const void* dy, int64_t rows, float* d_x,
                             RadStream stream, std::string* error) {
    if (dy == nullptr || d_x == nullptr || stream == nullptr || rows <= 0 || plan.classes <= 0 ||
        plan.n_embd <= 0 || plan.n_embd % 128 != 0 || plan.vocab <= 0 ||
        plan.vocab_offset < 0 || (plan.transform != 0 && plan.transform != 128)) {
        if (error) *error = "vocab head: the plan or the buffers are outside the kernel's rules";
        return false;
    }
    const int64_t K = plan.n_embd;
    const int64_t nkb = K >> 7;

    // The decode's scratch, one pair of buffers for every sub-range the call decodes: the
    // sub-range's int8 grid and its settled scales, the gather's buffer.
    device::ScratchBlock codes(kDecodeRows * K);
    device::ScratchBlock gscale(kDecodeRows * nkb * 2);
    device::ScratchBlock gather(rows * kDecodeRows * 2);
    // Declared after the blocks: destroyed before them, so every path out of this function drains
    // the stream while the scratch is still alive.
    SyncOnExit sync_on_exit(stream);
    if (!codes || !gscale || !gather) {
        if (error)
            *error = std::string("vocab head: the scratch will not allocate: ") +
                     rad_dev_last_error();
        return false;
    }

    for (int c = 0; c < plan.classes; ++c) {
        const VocabHeadClass& cls = plan.cls[c];
        if (cls.rows <= 0 || cls.ids == nullptr || cls.scale_col == nullptr ||
            cls.decode.rows_all != cls.rows) {
            if (error) *error = "vocab head: the plan's class " + std::to_string(c) +
                                " is incomplete";
            return false;
        }
        for (int64_t r0 = 0; r0 < cls.rows; r0 += kDecodeRows) {
            const int64_t ns = std::min(kDecodeRows, cls.rows - r0);
            R97IqrDecodeDesc d = cls.decode;
            d.row0 = r0;
            d.rows = ns;
            d.codes = d.codes + r0 * d.codes_row_bytes;
            d.scale = d.scale + r0 * d.scale_row_bytes;
            d.zero = d.zero ? d.zero + r0 * d.zero_row_bytes : nullptr;
            d.out_codes = (int8_t*)codes.p;
            d.out_scale = (uint16_t*)gscale.p;
            if (r97_iqr_decode(&d, stream) != 0) {
                if (error) *error = "vocab head: the plan's class " + std::to_string(c) +
                                    " decode is outside the kernel's rules";
                return false;
            }
            const int64_t gathered = rows * ns;
            dim3 ggrid((unsigned)((gathered + kThreads - 1) / kThreads));
            hipLaunchKernelGGL(vocab_gather_kernel, ggrid, dim3(kThreads), 0, (hipStream_t)stream,
                               (const uint16_t*)dy, (const int32_t*)cls.ids + r0,
                               (uint16_t*)gather.p, rows, plan.vocab, plan.vocab_offset, ns);
            if (!launch_check("the cotangent gather", error)) return false;
            dim3 grid((unsigned)rows);
            hipLaunchKernelGGL(vocab_gemv_kernel, grid, dim3(kThreads), 0, (hipStream_t)stream,
                               (const uint16_t*)gather.p, (const signed char*)codes.p,
                               (const uint16_t*)gscale.p, (const uint8_t*)cls.scale_col,
                               cls.scale_col_dtype, d_x, K, ns);
            if (!launch_check("the transposed product", error)) return false;
        }
    }

    if (plan.transform == 128) {
        dim3 grid((unsigned)rows, (unsigned)nkb);
        hipLaunchKernelGGL(vocab_unrotate_kernel, grid, dim3(128), 0, (hipStream_t)stream, d_x, K);
        if (!launch_check("the transform", error)) return false;
    }
    // The scratch above is freed on this function's return, so the work queued on it has to be
    // done by then: `sync_on_exit` drains the stream on the way out, before the blocks go back to
    // the allocator.
    return true;
}

// The file-scope initialiser, head.hip's pattern: loading the plugin registers the kernels into
// the table src/vocab_head.cpp dispatches through. In a process that dlopens the plugin without
// linking r97 the fallback in plugin.hip takes the call and registers nothing, which is right:
// the join that would run these kernels is not in that process.
struct Registration {
    Registration() {
        register_vocab_head_kernels(VocabHeadKernels{&vocab_head_transpose_fn,
                                                     &vocab_head_cast_fn});
    }
};
const Registration registration;

}  // namespace
}  // namespace r97
