// src/adjoint/device/elementwise.hip -- the device adjoints of the elementwise family.
//
// The same eight ops as src/adjoint/host/elementwise.cpp, on the card:
//
//   mul      a, b -> y       y = a * b, b allowed to carry one row of a
//   scale    x, s -> y       y = x * s[0]                                   (s frozen)
//   sigmoid  x -> y          y = 1 / (1 + exp(-x))
//   silu     x -> y          y = x * sigmoid(x)
//   silu_mul gate_up -> y    gate_up is [rows, 2n], y is [rows, n]           (the gated MLP activation)
//   act_mul  a, b -> y       y = act(a) * b
//   cast     x -> y          y = x in the output's dtype
//   rmsnorm  x, w -> y       y = x / rms(x) * (w + wadd) * mult              (w frozen)
//
// silu_mul is the one member the forward fuses on the card: the plugin's gated MLP activation
// issues silu and mul as one op, so the adjoint of the pair is one kernel pass over the fused
// [rows, 2n] row rather than a silu adjoint followed by a mul adjoint. Its reference is the
// composition of those two: tests/gpu/adjoint_elementwise.cpp runs mul's and silu's host adjoints
// in the order the fused kernel undoes them, as docs/kernels/README.md asks of a fused op. The
// unfused pair stays available on the card too -- mul and silu are each registered here -- so a
// graph that issues the two ops separately has both.
//
// The difference from the host adjoint is where the arithmetic happens: forward values and
// cotangents are read at f32, every reduction runs in f32 (a row of 5120 adds, or a column of
// 4096 rows), and the result is rounded once, to the op's dtype, on the way out. The gain of
// rmsnorm is formed in f32 before that rounding, and act_mul rounds the activation the way the
// forward kernel does. tests/gpu/adjoint_elementwise.cpp holds each kernel to the host adjoint
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

// Where a kernel stores its cotangent. Every member of the family rounds once, here, to the dtype
// the record gives the operand: bf16 everywhere in the trainer's mixed-precision graph, f32 for a
// cotangent that really is f32 (cast f32 -> bf16 hands one back). The slot's buffer is as wide as
// its dtype says, so a bf16 cotangent costs one value per two bytes.
struct OutStore {
    void* p;
    int wide;  // 1 when the slot is f32, 0 when it is bf16

    __host__ __device__ void set(int64_t i, float v) const {
        if (wide) {
            static_cast<float*>(p)[i] = v;
        } else {
            static_cast<unsigned short*>(p)[i] = f32_to_bf16(v);
        }
    }
};

/* ------------------------------------------------------------------ the activations */

// The six activations act_mul's schema enumerates, in the host file's order.
enum Act { kSilu = 0, kGelu, kGeluTanh, kSigmoid, kQuickGelu, kRelu };

bool act_of(const std::string& name, Act* act) {
    static const char* kNames[] = {"silu", "gelu", "gelu_tanh", "sigmoid", "quick_gelu", "relu"};
    for (int i = 0; i < 6; ++i) {
        if (name != kNames[i]) continue;
        *act = (Act)i;
        return true;
    }
    return false;
}

__device__ inline float sigmoidf(float x) { return 1.0f / (1.0f + __expf(-x)); }

// The six as the host adjoint takes them: the derivative of the forward the tape's twin ran, in
// f32. The forward kernel on the card rounds its activation before the product; that rounding
// belongs to the store, and the one store this kernel does is its cotangent's.
__device__ inline float act_value(int act, float x) {
    switch (act) {
        case kSilu:
            return x * sigmoidf(x);
        case kGelu:
            return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
        case kGeluTanh: {
            const float k0 = 0.79788456080286535588f, k1 = 0.044715f;
            return 0.5f * x * (1.0f + tanhf(k0 * (x + k1 * x * x * x)));
        }
        case kSigmoid:
            return sigmoidf(x);
        case kQuickGelu:
            return sigmoidf(1.702f * x);
        case kRelu:
            return x > 0.0f ? x : 0.0f;
    }
    return 0.0f;
}

__device__ inline float act_slope(int act, float x) {
    switch (act) {
        case kSilu: {
            const float s = sigmoidf(x);
            return s * (1.0f + x * (1.0f - s));
        }
        case kGelu:
            return 0.5f * (1.0f + erff(x * 0.70710678118654752440f)) +
                   x * 0.39894228040143267794f * __expf(-0.5f * x * x);
        case kGeluTanh: {
            const float k0 = 0.79788456080286535588f, k1 = 0.044715f;
            const float t = tanhf(k0 * (x + k1 * x * x * x));
            return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * k0 * (1.0f + 3.0f * k1 * x * x);
        }
        case kSigmoid: {
            const float s = sigmoidf(x);
            return s * (1.0f - s);
        }
        case kQuickGelu: {
            const float s = sigmoidf(1.702f * x);
            return 1.702f * s * (1.0f - s);
        }
        case kRelu:
            return x > 0.0f ? 1.0f : 0.0f;
    }
    return 0.0f;
}

/* ------------------------------------------------------------------ the kernels */

__global__ void scale_adjoint_kernel(const float* x_cot, OutStore x_dc, int64_t n, float s) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x_dc.set(i, x_cot[i] * s);
}

// y = sigmoid(x) is on the tape, so the adjoint reads the forward value rather than the input.
__global__ void sigmoid_adjoint_kernel(const float* y, const float* y_cot, OutStore x_dc,
                                       int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x_dc.set(i, y_cot[i] * y[i] * (1.0f - y[i]));
}

__global__ void silu_adjoint_kernel(const float* x, const float* y_cot, OutStore x_dc, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x_dc.set(i, y_cot[i] * act_slope(kSilu, x[i]));
}

// The fused gated activation: one pass writes both halves of the [rows, 2n] cotangent, the gate's
// slope from the gate and the multiplier from the silu of the same gate.
__global__ void silu_mul_adjoint_kernel(const float* gate_up, const float* y_cot, OutStore up_dc,
                                        int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t row = blockIdx.y;
    const int64_t xb = row * 2 * n, yb = row * n;
    const float g = gate_up[xb + i], u = gate_up[xb + n + i], d = y_cot[yb + i];
    up_dc.set(xb + i, d * u * act_slope(kSilu, g));
    up_dc.set(xb + n + i, d * act_value(kSilu, g));
}

// b carries its own width: both cotangents are element for element.
__global__ void mul_pair_kernel(const float* a, const float* b, const float* y_cot, OutStore a_dc,
                                OutStore b_dc, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float d = y_cot[i];
    a_dc.set(i, d * b[i]);
    b_dc.set(i, d * a[i]);
}

// b carries one row of a: a's cotangent is still element for element.
__global__ void mul_row_a_kernel(const float* b, const float* y_cot, OutStore a_dc, int64_t na,
                                 int64_t nb) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= na) return;
    a_dc.set(i, y_cot[i] * b[i % nb]);
}

// ... and b's is the column sum over the rows: one thread per column over a slice of the rows, so
// several blocks share a column and the partial sums meet in f32 staging, never in bf16.
__global__ void mul_row_b_kernel(const float* a, const float* y_cot, float* partial, int64_t na,
                                 int64_t nb) {
    const int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nb) return;
    const int64_t rows = (na + nb - 1) / nb;
    const int64_t chunks = gridDim.y;
    const int64_t r0 = rows * blockIdx.y / chunks;
    const int64_t r1 = rows * (blockIdx.y + 1) / chunks;
    float sum = 0.0f;
    for (int64_t r = r0; r < r1; ++r) {
        const int64_t i = r * nb + j;
        if (i >= na) break;
        sum += y_cot[i] * a[i];
    }
    atomicAdd(&partial[j], sum);
}

// A staged f32 result to the slot's dtype: the column sum above, and cast's pass-through.
__global__ void store_f32_kernel(const float* src, OutStore out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out.set(i, src[i]);
}

__global__ void act_mul_adjoint_kernel(const float* a, const float* b, const float* y_cot,
                                       OutStore a_dc, OutStore b_dc, int64_t n, int act) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float d = y_cot[i];
    a_dc.set(i, d * act_slope(act, a[i]) * b[i]);
    b_dc.set(i, d * act_value(act, a[i]));
}

// One pass over the row sums both x^2 (the rms) and dy*x*c (the shared term), in f32; the row's
// gain is formed here, at f32, before the store's one rounding.
__global__ void rmsnorm_adjoint_kernel(const float* x, const float* w, const float* y_cot,
                                       OutStore x_dc, int64_t n, float eps, float wadd, float mult) {
    __shared__ float sum_sq[256];
    __shared__ float sum_t[256];
    const int64_t row = blockIdx.x;
    const int64_t base = row * n;
    const int t = (int)threadIdx.x;
    const int str = (int)blockDim.x;
    float ss = 0.0f, tt = 0.0f;
    for (int64_t i = t; i < n; i += str) {
        const float xi = x[base + i];
        const float c = w ? w[i] + wadd : 1.0f;
        ss += xi * xi;
        tt += y_cot[base + i] * xi * c;
    }
    sum_sq[t] = ss;
    sum_t[t] = tt;
    __syncthreads();
    for (int s = str / 2; s > 0; s >>= 1) {
        if (t < s) {
            sum_sq[t] += sum_sq[t + s];
            sum_t[t] += sum_t[t + s];
        }
        __syncthreads();
    }
    const float inv_n = 1.0f / (float)n;
    const float s = rsqrtf(sum_sq[0] * inv_n + eps);
    const float t_shared = sum_t[0];
    const float s3 = s * s * s;
    for (int64_t i = t; i < n; i += str) {
        const float xi = x[base + i];
        const float c = w ? w[i] + wadd : 1.0f;
        const float shared = s3 * xi * t_shared * inv_n;
        x_dc.set(base + i, mult * (s * c * y_cot[base + i] - shared));
    }
}

/* ------------------------------------------------------------------ reading a record */

// A numeric parameter, as the host file reads it: an int is a value, a string parses as the
// decimal spelling the record carries, and anything unparsable leaves the fallback. Zero is a
// value here -- `wadd=0` is the plain norm -- which is why this does not fall back on it.
double param_num(const OpRecord& rec, const char* name, double fallback) {
    for (const auto& p : rec.params) {
        if (p.first != name) continue;
        if (const int64_t* i = std::get_if<int64_t>(&p.second)) return (double)*i;
        const std::string& s = std::get<std::string>(p.second);
        char* end = nullptr;
        const double v = std::strtod(s.c_str(), &end);
        return (end && end != s.c_str() && *end == '\0') ? v : fallback;
    }
    return fallback;
}

const std::string* param_str(const OpRecord& rec, const char* name) {
    for (const auto& p : rec.params) {
        if (p.first != name) continue;
        if (const std::string* s = std::get_if<std::string>(&p.second)) return s;
    }
    return nullptr;
}

/* The row width: the record's `n`, else the last dimension of the shape given. */
int64_t width_of(const OpRecord& rec, const std::vector<int64_t>& shape) {
    const int64_t n = (int64_t)param_num(rec, "n", 0.0);
    if (n > 0) return n;
    return shape.empty() ? 0 : shape.back();
}

/* The rows of a [rows, n] read, or 0 when the read does not divide into rows of n. */
int64_t rows_of(const AdValue& v, int64_t n) {
    const int64_t total = (int64_t)v.value.size();
    if (n <= 0 || total <= 0 || total % n != 0) return 0;
    return total / n;
}

/* silu_mul's output width, which the record's `n` names; its input is the fused [rows, 2n] row. */
int64_t gate_width(const OpRecord& rec, const std::vector<int64_t>& in_shape) {
    const int64_t n = (int64_t)param_num(rec, "n", 0.0);
    if (n > 0) return n;
    return in_shape.empty() ? 0 : in_shape.back() / 2;
}

/* ------------------------------------------------------------------ the card */

constexpr int64_t f32_bytes(int64_t n) { return n * (int64_t)sizeof(float); }

/* A cotangent slot's buffer: one value per element, bf16 or f32, written through OutStore. */
int64_t slot_bytes(const AdValue& slot, int64_t n) {
    return n * (int64_t)(slot.dtype == "f32" ? sizeof(float) : sizeof(unsigned short));
}

OutStore store_in(const AdValue& slot, void* p) {
    return OutStore{p, slot.dtype == "f32" ? 1 : 0};
}

// One stream for every call: the plugin runs on the card's default device and the harness syncs
// this stream before it reads a result back.
RadStream adjoint_stream() {
    static RadStream s = [] {
        RadStream created = nullptr;
        if (rad_stream_create(&created, 0) != 0) return (RadStream) nullptr;
        return created;
    }();
    return s;
}

int64_t blocks(int64_t n) { return (n + 255) / 256; }

/* The tape's f64 operands reach the card at f32: the kernels compute in f32, and every reduction
 * they run is a sum of f32 -- a row of 5120 or a column of 4096 rows -- so the accumulate is the
 * card's f32 arithmetic, not bf16's. */
bool upload(const AdValue& v, void* d, int64_t n, RadStream s) {
    if (n <= 0) return true;
    std::vector<float> h((size_t)n);
    for (int64_t i = 0; i < n; ++i) h[(size_t)i] = (float)v.value[(size_t)i];
    return rad_memcpy_async(d, h.data(), f32_bytes(n), s) == 0;
}

/* Adds a kernel's result back into the tape: the contract adds to a cotangent, never overwrites it,
 * because several ops share the buffer. The one rounding of the call happened in the store. */
bool download_add(AdValue& slot, const void* d, int64_t n, RadStream s) {
    if (n <= 0) return true;
    if (slot.dtype == "f32") {
        std::vector<float> h((size_t)n);
        if (rad_memcpy_async(h.data(), d, f32_bytes(n), s) != 0) return false;
        if (rad_stream_sync(s) != 0) return false;
        for (int64_t i = 0; i < n; ++i) slot.value[(size_t)i] += (double)h[(size_t)i];
        return true;
    }
    std::vector<unsigned short> h((size_t)n);
    const int64_t bytes = n * (int64_t)sizeof(unsigned short);
    if (rad_memcpy_async(h.data(), d, bytes, s) != 0) return false;
    if (rad_stream_sync(s) != 0) return false;
    for (int64_t i = 0; i < n; ++i) slot.value[(size_t)i] += (double)bf16_to_f32(h[(size_t)i]);
    return true;
}

/* ------------------------------------------------------------------ mul: y = a * b */

void mul_device_adjoint(const AdjointCall& call) {
    if (!call.dc || call.cot.empty() || call.fwd.size() < 2 || call.dc->size() < 2) return;
    const AdValue& a = call.fwd[0];
    const AdValue& b = call.fwd[1];
    const AdValue& y_cot = call.cot[0];
    AdValue& a_dc = call.dc->at(0);
    AdValue& b_dc = call.dc->at(1);
    const int64_t na = (int64_t)y_cot.value.size(), nb = (int64_t)b.value.size();
    if (nb == 0 || nb > na || (int64_t)a.value.size() < na) return;

    RadStream stream = adjoint_stream();
    if (!stream) return;

    const bool pair = nb == na;
    device::ScratchBlock d_a(f32_bytes(na));
    device::ScratchBlock d_b(f32_bytes(nb));
    device::ScratchBlock d_y_cot(f32_bytes(na));
    device::ScratchBlock d_a_dc(slot_bytes(a_dc, na));
    device::ScratchBlock d_b_dc(slot_bytes(b_dc, nb));
    if (!d_a || !d_b || !d_y_cot || !d_a_dc || !d_b_dc) return;
    if (!upload(a, d_a.p, na, stream) || !upload(b, d_b.p, nb, stream) ||
        !upload(y_cot, d_y_cot.p, na, stream))
        return;

    const OutStore a_store = store_in(a_dc, d_a_dc.p);
    const OutStore b_store = store_in(b_dc, d_b_dc.p);

    if (pair) {
        hipLaunchKernelGGL(mul_pair_kernel, dim3((unsigned int)blocks(na)), dim3(256), 0,
                           (hipStream_t)stream, (const float*)d_a.p, (const float*)d_b.p,
                           (const float*)d_y_cot.p, a_store, b_store, na);
    } else {
        // b carries one row of a, so its cotangent is the column sum. A column's rows are split
        // across a few blocks and the partials meet in f32 staging: with a thousand blocks and one
        // thread per column, a single block per column would leave most of the card idle and an
        // atomic per element would crowd one address with every row.
        device::ScratchBlock d_partial(f32_bytes(nb));
        if (!d_partial) return;
        if (rad_memset_async(d_partial.p, 0, f32_bytes(nb), stream) != 0) return;
        const int64_t col_blocks = blocks(nb);
        const int64_t rows = (na + nb - 1) / nb;
        int64_t chunks = 2048 / col_blocks;
        if (chunks < 1) chunks = 1;
        if (chunks > rows) chunks = rows;
        hipLaunchKernelGGL(mul_row_a_kernel, dim3((unsigned int)blocks(na)), dim3(256), 0,
                           (hipStream_t)stream, (const float*)d_b.p, (const float*)d_y_cot.p, a_store,
                           na, nb);
        hipLaunchKernelGGL(mul_row_b_kernel,
                           dim3((unsigned int)col_blocks, (unsigned int)chunks), dim3(256), 0,
                           (hipStream_t)stream, (const float*)d_a.p, (const float*)d_y_cot.p,
                           (float*)d_partial.p, na, nb);
        hipLaunchKernelGGL(store_f32_kernel, dim3((unsigned int)blocks(nb)), dim3(256), 0,
                           (hipStream_t)stream, (const float*)d_partial.p, b_store, nb);
    }

    if (!download_add(a_dc, d_a_dc.p, na, stream)) return;
    if (!download_add(b_dc, d_b_dc.p, nb, stream)) return;
}

/* ------------------------------------------------------------------ scale: y = x * s[0] */

void scale_device_adjoint(const AdjointCall& call) {
    if (!call.dc || call.cot.empty() || call.fwd.size() < 2 || call.fwd[1].value.empty() ||
        call.dc->empty())
        return;
    const int64_t n = (int64_t)call.cot[0].value.size();
    if (n <= 0) return;
    AdValue& x_dc = call.dc->at(0);

    RadStream stream = adjoint_stream();
    if (!stream) return;

    device::ScratchBlock d_y_cot(f32_bytes(n));
    device::ScratchBlock d_x_dc(slot_bytes(x_dc, n));
    if (!d_y_cot || !d_x_dc) return;
    if (!upload(call.cot[0], d_y_cot.p, n, stream)) return;

    hipLaunchKernelGGL(scale_adjoint_kernel, dim3((unsigned int)blocks(n)), dim3(256), 0,
                       (hipStream_t)stream, (const float*)d_y_cot.p, store_in(x_dc, d_x_dc.p), n,
                       (float)call.fwd[1].value[0]);
    download_add(x_dc, d_x_dc.p, n, stream);
}

/* ------------------------------------------------------------------ sigmoid: y = 1/(1+e^-x) */

void sigmoid_device_adjoint(const AdjointCall& call) {
    if (!call.dc || call.cot.empty() || call.fwd.size() < 2 || call.dc->empty()) return;
    const int64_t n = (int64_t)call.cot[0].value.size();
    if (n <= 0 || (int64_t)call.fwd[1].value.size() < n) return;
    AdValue& x_dc = call.dc->at(0);

    RadStream stream = adjoint_stream();
    if (!stream) return;

    device::ScratchBlock d_y(f32_bytes(n));
    device::ScratchBlock d_y_cot(f32_bytes(n));
    device::ScratchBlock d_x_dc(slot_bytes(x_dc, n));
    if (!d_y || !d_y_cot || !d_x_dc) return;
    if (!upload(call.fwd[1], d_y.p, n, stream) || !upload(call.cot[0], d_y_cot.p, n, stream)) return;

    hipLaunchKernelGGL(sigmoid_adjoint_kernel, dim3((unsigned int)blocks(n)), dim3(256), 0,
                       (hipStream_t)stream, (const float*)d_y.p, (const float*)d_y_cot.p,
                       store_in(x_dc, d_x_dc.p), n);
    download_add(x_dc, d_x_dc.p, n, stream);
}

/* ------------------------------------------------------------------ silu: y = x * sigmoid(x) */

void silu_device_adjoint(const AdjointCall& call) {
    if (!call.dc || call.cot.empty() || call.fwd.empty() || call.dc->empty()) return;
    const int64_t n = (int64_t)call.cot[0].value.size();
    if (n <= 0 || (int64_t)call.fwd[0].value.size() < n) return;
    AdValue& x_dc = call.dc->at(0);

    RadStream stream = adjoint_stream();
    if (!stream) return;

    device::ScratchBlock d_x(f32_bytes(n));
    device::ScratchBlock d_y_cot(f32_bytes(n));
    device::ScratchBlock d_x_dc(slot_bytes(x_dc, n));
    if (!d_x || !d_y_cot || !d_x_dc) return;
    if (!upload(call.fwd[0], d_x.p, n, stream) || !upload(call.cot[0], d_y_cot.p, n, stream)) return;

    hipLaunchKernelGGL(silu_adjoint_kernel, dim3((unsigned int)blocks(n)), dim3(256), 0,
                       (hipStream_t)stream, (const float*)d_x.p, (const float*)d_y_cot.p,
                       store_in(x_dc, d_x_dc.p), n);
    download_add(x_dc, d_x_dc.p, n, stream);
}

/* --------------------------------------------------- silu_mul: the fused gated activation */

void silu_mul_device_adjoint(const AdjointCall& call) {
    if (!call.dc || call.cot.empty() || call.fwd.empty() || call.dc->empty()) return;
    const AdValue& gate_up = call.fwd[0];
    const AdValue& y_cot = call.cot[0];
    const int64_t n = gate_width(call.rec, gate_up.shape);
    const int64_t rows = rows_of(gate_up, 2 * n);
    if (n <= 0 || rows <= 0 || rows_of(y_cot, n) != rows) return;
    AdValue& up_dc = call.dc->at(0);
    if ((int64_t)up_dc.value.size() < rows * 2 * n) return;

    RadStream stream = adjoint_stream();
    if (!stream) return;

    device::ScratchBlock d_gate_up(f32_bytes(rows * 2 * n));
    device::ScratchBlock d_y_cot(f32_bytes(rows * n));
    device::ScratchBlock d_up_dc(slot_bytes(up_dc, rows * 2 * n));
    if (!d_gate_up || !d_y_cot || !d_up_dc) return;
    if (!upload(gate_up, d_gate_up.p, rows * 2 * n, stream) ||
        !upload(y_cot, d_y_cot.p, rows * n, stream))
        return;

    hipLaunchKernelGGL(silu_mul_adjoint_kernel,
                       dim3((unsigned int)blocks(n), (unsigned int)rows), dim3(256), 0,
                       (hipStream_t)stream, (const float*)d_gate_up.p, (const float*)d_y_cot.p,
                       store_in(up_dc, d_up_dc.p), n);
    download_add(up_dc, d_up_dc.p, rows * 2 * n, stream);
}

/* ------------------------------------------------------------------ act_mul: y = act(a) * b */

void act_mul_device_adjoint(const AdjointCall& call) {
    if (!call.dc || call.cot.empty() || call.fwd.size() < 2 || call.dc->size() < 2) return;
    const std::string* name = param_str(call.rec, "act");
    Act act;
    if (!name || !act_of(*name, &act)) return;

    const AdValue& a = call.fwd[0];
    const AdValue& b = call.fwd[1];
    const AdValue& y_cot = call.cot[0];
    AdValue& a_dc = call.dc->at(0);
    AdValue& b_dc = call.dc->at(1);
    const int64_t n = (int64_t)y_cot.value.size();
    if (n <= 0 || (int64_t)a.value.size() < n || (int64_t)b.value.size() < n) return;

    RadStream stream = adjoint_stream();
    if (!stream) return;

    device::ScratchBlock d_a(f32_bytes(n));
    device::ScratchBlock d_b(f32_bytes(n));
    device::ScratchBlock d_y_cot(f32_bytes(n));
    device::ScratchBlock d_a_dc(slot_bytes(a_dc, n));
    device::ScratchBlock d_b_dc(slot_bytes(b_dc, n));
    if (!d_a || !d_b || !d_y_cot || !d_a_dc || !d_b_dc) return;
    if (!upload(a, d_a.p, n, stream) || !upload(b, d_b.p, n, stream) ||
        !upload(y_cot, d_y_cot.p, n, stream))
        return;

    hipLaunchKernelGGL(act_mul_adjoint_kernel, dim3((unsigned int)blocks(n)), dim3(256), 0,
                       (hipStream_t)stream, (const float*)d_a.p, (const float*)d_b.p,
                       (const float*)d_y_cot.p, store_in(a_dc, d_a_dc.p),
                       store_in(b_dc, d_b_dc.p), n, (int)act);
    if (!download_add(a_dc, d_a_dc.p, n, stream)) return;
    download_add(b_dc, d_b_dc.p, n, stream);
}

/* ------------------------------------------------------------------ cast: y = x in to */

void cast_device_adjoint(const AdjointCall& call) {
    if (!call.dc || call.cot.empty() || call.dc->empty()) return;
    const int64_t n = (int64_t)call.cot[0].value.size();
    if (n <= 0) return;
    AdValue& x_dc = call.dc->at(0);

    RadStream stream = adjoint_stream();
    if (!stream) return;

    device::ScratchBlock d_y_cot(f32_bytes(n));
    device::ScratchBlock d_x_dc(slot_bytes(x_dc, n));
    if (!d_y_cot || !d_x_dc) return;
    if (!upload(call.cot[0], d_y_cot.p, n, stream)) return;

    hipLaunchKernelGGL(store_f32_kernel, dim3((unsigned int)blocks(n)), dim3(256), 0,
                       (hipStream_t)stream, (const float*)d_y_cot.p, store_in(x_dc, d_x_dc.p), n);
    download_add(x_dc, d_x_dc.p, n, stream);
}

/* ------------------------------------------------------------------ rmsnorm */

void rmsnorm_device_adjoint(const AdjointCall& call) {
    if (!call.dc || call.cot.empty() || call.fwd.empty() || call.dc->empty()) return;
    const AdValue& x = call.fwd[0];
    const AdValue* w = call.fwd.size() > 1 && !call.fwd[1].value.empty() ? &call.fwd[1] : nullptr;
    const AdValue& y_cot = call.cot[0];
    const int64_t n = width_of(call.rec, x.shape);
    const int64_t rows = rows_of(x, n);
    if (n <= 0 || rows <= 0 || rows_of(y_cot, n) != rows) return;
    if (w && (int64_t)w->value.size() < n) return;
    const float eps = (float)param_num(call.rec, "eps", 1e-6);
    const float wadd = (float)param_num(call.rec, "wadd", 0.0);
    const float mult = (float)param_num(call.rec, "mult", 1.0);
    if (!w && wadd != 0.0f) return;  // a gain offset without a gain is nothing to add it to
    AdValue& x_dc = call.dc->at(0);
    if ((int64_t)x_dc.value.size() < rows * n) return;

    RadStream stream = adjoint_stream();
    if (!stream) return;

    device::ScratchBlock d_x(f32_bytes(rows * n));
    device::ScratchBlock d_w(f32_bytes(n));
    device::ScratchBlock d_y_cot(f32_bytes(rows * n));
    device::ScratchBlock d_x_dc(slot_bytes(x_dc, rows * n));
    if (!d_x || !d_w || !d_y_cot || !d_x_dc) return;
    if (!upload(x, d_x.p, rows * n, stream) || !upload(y_cot, d_y_cot.p, rows * n, stream)) return;
    if (w && !upload(*w, d_w.p, n, stream)) return;

    hipLaunchKernelGGL(rmsnorm_adjoint_kernel, dim3((unsigned int)rows), dim3(256), 0,
                       (hipStream_t)stream, (const float*)d_x.p,
                       w ? (const float*)d_w.p : nullptr, (const float*)d_y_cot.p,
                       store_in(x_dc, d_x_dc.p), n, eps, wadd, mult);
    download_add(x_dc, d_x_dc.p, rows * n, stream);
}

}  // namespace

// The two weights are frozen, as on the host: the walk never asks them for a cotangent.
R97_REGISTER_ADJOINT("mul", device, mul_device_adjoint)
R97_REGISTER_ADJOINT_F("scale", device, scale_device_adjoint, 1)
R97_REGISTER_ADJOINT("sigmoid", device, sigmoid_device_adjoint)
R97_REGISTER_ADJOINT("silu", device, silu_device_adjoint)
R97_REGISTER_ADJOINT("silu_mul", device, silu_mul_device_adjoint)
R97_REGISTER_ADJOINT("act_mul", device, act_mul_device_adjoint)
R97_REGISTER_ADJOINT("cast", device, cast_device_adjoint)
R97_REGISTER_ADJOINT_F("rmsnorm", device, rmsnorm_device_adjoint, 1)

}  // namespace r97
