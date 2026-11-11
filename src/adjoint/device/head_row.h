// src/adjoint/device/head_row.h -- one logit row's arithmetic, shared by the head's chunk kernel
// and the checks that pin it.
//
// A row's stable log-sum-exp in f32, its loss and one gradient value: the kernel gives each of its
// threads a stride through the row and reduces what they found, and a serial caller -- a check, a
// chunk a host twin runs -- walks the same two passes with tid 0 and stride 1. Both compile the
// same expressions here, so the numbers a check sees are the kernel's numbers, and the loss keeps
// the reference's shape: `lse - (row[target] - max)`, never `max + log(sum(exp(x)))`.
#pragma once

#include <cstdint>
#include <cstring>
#include <math.h>

#if defined(__HIPCC__) || defined(__HIP__) || defined(__CUDACC__)
#define R97_HEAD_HD __host__ __device__
#else
#define R97_HEAD_HD
#endif

namespace r97 {

// bf16 is stored as the high half of an f32. The widening is exact; the narrowing rounds to
// nearest even, which is what the kernel stores its gradient rows with.
R97_HEAD_HD inline float head_bf16_to_f32(uint16_t b) {
    const uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

R97_HEAD_HD inline uint16_t head_bf16_from_f32(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x0040u);  // NaN stays NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}

// A row's stable log-sum-exp in f32: `max` is the row's largest value, `lse` is
// log(sum(exp(x - max))), so a row whose values are all near 1e30 still subtracts to something a
// float holds. The two passes are what the kernel's block runs: each thread strides through the
// row (`tid`, `stride`), and the block reduces the per-thread answers.
struct HeadRowExp {
    float max = 0.0f;
    float lse = 0.0f;
};

R97_HEAD_HD inline float head_row_max(const uint16_t* row, int64_t vocab, int64_t tid,
                                     int64_t stride) {
    float acc[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    int64_t j = tid;
    for (; j + 3 * stride < vocab; j += 4 * stride) {
        acc[0] = fmaxf(acc[0], head_bf16_to_f32(row[j]));
        acc[1] = fmaxf(acc[1], head_bf16_to_f32(row[j + stride]));
        acc[2] = fmaxf(acc[2], head_bf16_to_f32(row[j + 2 * stride]));
        acc[3] = fmaxf(acc[3], head_bf16_to_f32(row[j + 3 * stride]));
    }
    for (; j < vocab; j += stride) acc[0] = fmaxf(acc[0], head_bf16_to_f32(row[j]));
    return fmaxf(fmaxf(acc[0], acc[1]), fmaxf(acc[2], acc[3]));
}

R97_HEAD_HD inline float head_row_sumexp(const uint16_t* row, int64_t vocab, float max, int64_t tid,
                                         int64_t stride) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int64_t j = tid;
    for (; j + 3 * stride < vocab; j += 4 * stride) {
        acc[0] += expf(head_bf16_to_f32(row[j]) - max);
        acc[1] += expf(head_bf16_to_f32(row[j + stride]) - max);
        acc[2] += expf(head_bf16_to_f32(row[j + 2 * stride]) - max);
        acc[3] += expf(head_bf16_to_f32(row[j + 3 * stride]) - max);
    }
    for (; j < vocab; j += stride) acc[0] += expf(head_bf16_to_f32(row[j]) - max);
    return (acc[0] + acc[1]) + (acc[2] + acc[3]);
}

// The row's loss: minus the log softmax of its target.
R97_HEAD_HD inline float head_row_loss(const uint16_t* row, HeadRowExp e, int32_t target) {
    return e.lse - (head_bf16_to_f32(row[target]) - e.max);
}

// The row's gradient at `j`: softmax minus one-hot, times the scale of the mean.
R97_HEAD_HD inline float head_row_grad(const uint16_t* row, HeadRowExp e, int64_t j, int32_t target,
                                       float scale) {
    const float onehot = j == target ? 1.0f : 0.0f;
    return (expf((head_bf16_to_f32(row[j]) - e.max) - e.lse) - onehot) * scale;
}

}  // namespace r97
