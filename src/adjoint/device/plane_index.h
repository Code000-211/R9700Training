// src/adjoint/device/plane_index.h -- the stored planes' addressing, shared by the device rows.
//
// The forward's i8a8h layout (libr4d/r4d_fp8_layout.cpp: codes tag "r4d.i8h.w.frag64", scale tag
// "r4d.i8.s.t16") is what a quantised weight's plane is in, and two rows address it: the linear
// family's dx reads it (src/adjoint/device/linear.hip), and the IQR decode writes the same order
// into a staging plane so the linear row can read a trellis weight (iqr_plane.hip). One definition,
// because a second spelling of the same index is a second thing to get wrong.
#pragma once

#include <cstddef>

#include <hip/hip_runtime.h>

// The byte offset of the 8-byte fragment that starts at column `k` (k % 8 == 0) of `row`.
__device__ __forceinline__ size_t frag_go(int K, int row, int k) {
    const int kk = k & 63;
    const int lane = (row & 15) + 16 * ((kk & 15) >> 3);
    return ((static_cast<size_t>(row >> 4) * static_cast<size_t>(K >> 6) +
             static_cast<size_t>(k >> 6)) * 32u + static_cast<size_t>(lane)) * 32u +
           static_cast<size_t>((kk >> 4) * 8);
}

// The element index of (row, block `g` of K) in the tile-major bf16 scale plane.
__device__ __forceinline__ size_t t16_go(int G, int row, int g) {
    return (static_cast<size_t>(row >> 4) * static_cast<size_t>(G) + static_cast<size_t>(g)) * 16u +
           static_cast<size_t>(row & 15);
}
