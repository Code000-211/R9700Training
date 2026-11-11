// src/adjoint/device/head.hip -- the head's cross-entropy chunk, on the card.
//
// The loss the reference computes (src/loss.cpp, docs/losses/ce-objective.md) over a chunk of the
// head's logits: one block per row, the row's maximum and the sum of exp(x - max) reduced in f32,
// then the row's loss and its gradient -- softmax minus one-hot, times the mean's scale -- written
// as bf16. Rows are independent, so a chunk is the same arithmetic whatever its size, and the
// vocabulary is never held: a block strides a row instead of copying it.
//
// The contract is include/r97/head.h: device buffers in, device buffers out, no copy through the
// host. The host side of this file only queues it; the walk that fills and sinks the chunks is
// src/head.cpp. docs/kernels/head.md has the chunk's memory and the measured cost per position.
//
// The row's arithmetic is in head_row.h, shared with the checks that pin it, so the numbers a
// check computes serially are the numbers the block's threads compute.
#include "r97/head.h"

#include <hip/hip_runtime.h>

#include "head_row.h"

#include "rad_device.h"

#include <cstdint>

namespace r97 {
namespace {

constexpr int kBlock = 256;

// The block's reduction of each thread's partial: the maximum first, then the sum, both down a
// shared array. A row of 248320 values leaves every thread 970 of them, so the two passes are the
// whole cost and the reduction is noise.
template <int BLOCK>
__device__ inline float block_max(float* red, int tid, float v) {
    red[tid] = v;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] = fmaxf(red[tid], red[tid + stride]);
        __syncthreads();
    }
    const float out = red[0];
    __syncthreads();
    return out;
}

template <int BLOCK>
__device__ inline float block_sum(float* red, int tid, float v) {
    red[tid] = v;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        __syncthreads();
    }
    const float out = red[0];
    __syncthreads();
    return out;
}

// One block per row: row i's logits at `logits + i * vocab`, its target at `ids[i]` (-1 where the
// row is not scored), its loss to `loss[i]` and its gradient to `grad + i * vocab`.
__global__ void head_ce_kernel(const unsigned short* __restrict__ logits,
                               const int* __restrict__ ids, unsigned short* __restrict__ grad,
                               float* __restrict__ loss, int64_t vocab, float grad_scale) {
    __shared__ float red[kBlock];
    const int tid = int(threadIdx.x);
    const int64_t row = blockIdx.x;
    const unsigned short* x = logits + row * vocab;
    unsigned short* g = grad ? grad + row * vocab : nullptr;

    const float max = block_max<kBlock>(red, tid, head_row_max(x, vocab, tid, kBlock));
    const float sumexp = block_sum<kBlock>(red, tid, head_row_sumexp(x, vocab, max, tid, kBlock));

    HeadRowExp e;
    e.max = max;
    e.lse = logf(sumexp);

    const int32_t target = ids[row];
    if (target < 0) {
        // A row that is not scored contributes nothing to the loss and nothing to the gradient;
        // the zeros keep the chunk a complete cotangent, so the consumer need not know which rows
        // were scored.
        if (g)
            for (int64_t j = tid; j < vocab; j += kBlock) g[j] = 0;
        if (loss && tid == 0) loss[row] = 0.0f;
        return;
    }
    if (loss && tid == 0) loss[row] = head_row_loss(x, e, target);
    if (g) {
        for (int64_t j = tid; j < vocab; j += kBlock)
            g[j] = head_bf16_from_f32(head_row_grad(x, e, j, target, grad_scale));
    }
}

bool head_ce_chunk(const HeadChunk& chunk, const char** error) {
    if (!chunk.logits || !chunk.ids) {
        if (error) *error = "a chunk needs its logits and its ids";
        return false;
    }
    if (chunk.positions <= 0 || chunk.vocab <= 0) {
        if (error) *error = "a chunk needs rows and a vocabulary";
        return false;
    }
    if (!chunk.grad && !chunk.loss) return true;  // nothing was asked for

    hipStream_t stream = static_cast<hipStream_t>(chunk.stream);
    hipLaunchKernelGGL(head_ce_kernel, dim3(static_cast<unsigned int>(chunk.positions)),
                       dim3(kBlock), 0, stream,
                       static_cast<const unsigned short*>(chunk.logits),
                       static_cast<const int*>(chunk.ids),
                       static_cast<unsigned short*>(chunk.grad), chunk.loss, chunk.vocab,
                       chunk.grad_scale);
    const hipError_t queued = hipGetLastError();
    if (queued != hipSuccess) {
        if (error) *error = hipGetErrorString(queued);
        return false;
    }
    return true;
}

// The plugin's registration: it runs as the loader brings libr97 in, before a test or a trainer
// asks for the kernel. A process that loads the plugin without linking r97 -- radiance's
// converter -- resolves head_ce_register to plugin.hip's fallback and this row goes nowhere.
struct Registration {
    Registration() { head_ce_register(head_ce_chunk); }
};
const Registration registration;

}  // namespace
}  // namespace r97
