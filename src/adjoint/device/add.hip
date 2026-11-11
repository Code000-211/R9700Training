// src/adjoint/device/add.hip -- the device adjoint of add, the worked example.
//
// The same op as src/adjoint/host/add.cpp, on the card: y = (a + b) / div, so the cotangent of y
// flows straight to a, and to b either element for element or as the row sum when b carries one
// row's width. The difference from the host adjoint is where the arithmetic happens: the
// cotangent is read at f32 precision, the reduction runs in f32, and the result is rounded once to
// the op's bf16 dtype on the way out. tests/gpu/adjoint_add.cpp holds it to the host adjoint
// within the bf16 tolerance docs/kernels/README.md states.
//
// The contract gives a device adjoint f64 operands, not device buffers, so this call owns every
// device buffer it touches. They are scratch, freed before it returns; the harness reports the
// high-water mark.
#include "r97/adjoint.h"

#include "scratch.h"

#include "rad_device.h"

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

// bf16 is stored as the high half of an f32, with round-to-nearest-even on the way down. The
// kernel stores and loads it as unsigned short so the file needs neither __bf16 nor a HIP bf16
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

__global__ void add_adjoint_kernel(const float* y_cot, unsigned short* a_cot,
                                   unsigned short* b_cot, int64_t n, float inv) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned short v = f32_to_bf16(y_cot[i] * inv);
    a_cot[i] = v;
    if (b_cot) b_cot[i] = v;
}

// b broadcasts along rows: one thread per column, summing in f32 before the one bf16 rounding.
__global__ void add_bias_adjoint_kernel(const float* y_cot, unsigned short* b_cot,
                                        int64_t rows, int64_t width, float inv) {
    const int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= width) return;
    float sum = 0.0f;
    for (int64_t r = 0; r < rows; ++r) sum += y_cot[r * width + j];
    b_cot[j] = f32_to_bf16(sum * inv);
}

// A numeric parameter as f64, as the host adjoint reads it: a missing or non-numeric parameter
// leaves the fallback in place, because an adjoint must not throw.
double param_num(const OpRecord& rec, const char* name, double fallback) {
    for (const OpParam& p : rec.params) {
        if (p.first != name) continue;
        if (auto* i = std::get_if<int64_t>(&p.second)) return *i == 0 ? fallback : (double)*i;
        const std::string& s = std::get<std::string>(p.second);
        char* end = nullptr;
        const double v = std::strtod(s.c_str(), &end);
        if (end && *end == '\0' && v != 0.0) return v;
        return fallback;
    }
    return fallback;
}

// One stream for every device adjoint call in this process. A kernel's work is not the only cost
// the harness times: upload, launch and download share the stream, so a call is one queue's worth
// of work. The stream belongs to the device rad_dev_set() chose; a test or a trainer selects one
// device before the first adjoint runs.
RadStream adjoint_stream() {
    static RadStream s = [] {
        RadStream created = nullptr;
        if (rad_stream_create(&created, 0) != 0) return (RadStream) nullptr;
        return created;
    }();
    return s;
}

int64_t blocks(int64_t n) { return (n + 255) / 256; }

void add_device_adjoint(const AdjointCall& call) {
    if (!call.dc || call.cot.empty() || call.dc->size() < 2) return;
    AdValue& a_cot = call.dc->at(0);
    AdValue& b_cot = call.dc->at(1);
    const AdValue& y_cot = call.cot[0];

    const int64_t n = (int64_t)a_cot.value.size();
    const int64_t bw = (int64_t)b_cot.value.size();
    if (n <= 0 || (int64_t)y_cot.value.size() < n) return;

    const bool elementwise = bw == n;
    const bool broadcast = !elementwise && bw > 0 && n % bw == 0;
    const int64_t writes_b = (elementwise || broadcast) ? bw : 0;
    const float inv = (float)(1.0 / param_num(call.rec, "div", 1.0));

    RadStream stream = adjoint_stream();
    if (!stream) return;

    device::ScratchBlock d_y(n * (int64_t)sizeof(float));
    device::ScratchBlock d_a(n * (int64_t)sizeof(unsigned short));
    device::ScratchBlock d_b(writes_b * (int64_t)sizeof(unsigned short));
    if (!d_y || !d_a || (writes_b > 0 && !d_b)) return;

    // Upload the cotangent at f32: the tape's values are f64, the kernel computes in f32, and
    // only the op's outputs go back to bf16.
    std::vector<float> h_y((size_t)n);
    for (int64_t i = 0; i < n; ++i) h_y[(size_t)i] = (float)y_cot.value[(size_t)i];
    if (rad_memcpy_async(d_y.p, h_y.data(), n * (int64_t)sizeof(float), stream) != 0) return;

    const int64_t grid = blocks(n);
    hipLaunchKernelGGL(add_adjoint_kernel, dim3((unsigned int)grid), dim3(256), 0,
                       (hipStream_t)stream, (const float*)d_y.p, (unsigned short*)d_a.p,
                       elementwise ? (unsigned short*)d_b.p : nullptr, n, inv);
    if (broadcast) {
        const int64_t rows = n / bw;
        const int64_t bias_grid = blocks(bw);
        hipLaunchKernelGGL(add_bias_adjoint_kernel, dim3((unsigned int)bias_grid), dim3(256), 0,
                           (hipStream_t)stream, (const float*)d_y.p, (unsigned short*)d_b.p, rows,
                           bw, inv);
    }
    if (rad_stream_sync(stream) != 0) return;

    std::vector<unsigned short> h_a((size_t)n);
    std::vector<unsigned short> h_b((size_t)writes_b);
    if (rad_memcpy_async(h_a.data(), d_a.p, n * 2, stream) != 0) return;
    if (writes_b > 0 && rad_memcpy_async(h_b.data(), d_b.p, writes_b * 2, stream) != 0) return;
    if (rad_stream_sync(stream) != 0) return;

    // The tape's cotangents are f64 and are added to, never overwritten: other ops share them.
    for (int64_t i = 0; i < n; ++i) a_cot.value[(size_t)i] += bf16_to_f32(h_a[(size_t)i]);
    if (broadcast) {
        for (int64_t j = 0; j < bw; ++j) b_cot.value[(size_t)j] += bf16_to_f32(h_b[(size_t)j]);
    } else if (elementwise) {
        for (int64_t i = 0; i < bw; ++i) b_cot.value[(size_t)i] += bf16_to_f32(h_b[(size_t)i]);
    }
}

}  // namespace

R97_REGISTER_ADJOINT("add", device, add_device_adjoint)

}  // namespace r97
