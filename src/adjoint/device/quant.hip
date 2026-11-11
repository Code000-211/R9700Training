// src/adjoint/device/quant.hip -- the device straight-through adjoint of the quant family.
//
// The same ops as src/adjoint/host/quant.cpp, on the card: had_quant_act_i8g and the fused
// producers the card issues (rmsnorm_had_quant_i8g, gated_had_quant_i8g, gate_had_quant_i8g).
// Each rotates an activation, quantises it per group of `group` values, and stores the int8 codes
// and one f32 scale per group; a fused producer also applies an elementwise op (an RMS norm, the
// gated activation's silu(gate) * up, the gate's sigmoid(gate) * x) before the rotation.
//
// The adjoint of one group is docs/adjoints/quant.md's straight-through rule: the cotangent of the
// codes is divided by the group's scale, dropped where the forward's value saturated the int8 grid,
// and rotated back with the transposed Hadamard transform -- the same transform, since H is its own
// transpose and the forward does not normalise. A fused producer's cotangent with respect to the
// rotated activation then continues through the elementwise part, whose cotangent lands on the
// record's read operands. One kernel does the mask and the rotation; a second does the elementwise
// backprop.
//
// Where the mask comes from. docs/adjoints/quant.md's rule is |v * (1/scale)| > 127, v the group's
// rotated activation, and this kernel applies exactly that: the codes the replayed forward wrote
// are read, and a code at the top of the grid (|code| == 127) keeps its cotangent only when the
// rotated value was inside the grid. Both halves are needed. A code of 127 is not by itself
// evidence of a clamp -- every value in (126.5, 127] rounds to it, and the group's amax reaches it
// by construction in either direction of the scale's rounding -- so the reference's mask and a mask
// read from the codes alone would disagree on the amax of roughly half the groups, which is what
// the test would catch. Reading the codes and confirming with v is the reference's rule exactly,
// and the codes are what tell a replay which side of the grid the forward's rounding landed on.
//
// v is formed in f64, stage by stage, the way the host adjoint's rotated() forms it, so a group
// whose amax sits an ulp from the grid's edge decides as the reference decides. The butterflies of
// one stage are independent, so threads may share a stage without changing a bit of the result.
//
// The contract gives a device adjoint f64 operands, not device buffers, so this call owns every
// device buffer it touches. They are scratch, freed before it returns; the harness reports the
// high-water mark. The forward values are uploaded at f64 where a decision depends on them (the
// rotated activation, the scale) and at f32 where only a smooth derivative does (the elementwise
// part's operands); gradients are read at f32, summed in f32, and rounded once to bf16 on the way
// out, as add.hip does. A fused producer's intermediate is not on the tape, so it is rebuilt
// through the parts' registered twin -- the same definition the host reference composes -- and a
// record whose activation this row does not serve is refused at the row lookup rather than served
// approximately.
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

// bf16 is stored as the high half of an f32, with round-to-nearest-even on the way down (add.hip's
// conversion, so both kernels round the tape's cotangents the same way).
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

/* ---------------------------------------------------------------- the family's parts */

// Which elementwise op a fused producer applies before the rotation. `quant` is the plain
// quantiser: the whole of had_quant_act_i8g.
enum class Part { quant, rmsnorm, gated, gate };

// The host twin and host adjoint of the elementwise part, by op name.
const char* part_op(Part part) {
    switch (part) {
        case Part::rmsnorm: return "rmsnorm";
        case Part::gated: return "silu_mul";
        case Part::gate: return "act_mul";
        case Part::quant: break;
    }
    return nullptr;
}

// How many operands of the record the adjoint answers for (every operand but the codes and scale).
size_t part_reads(Part part) {
    return part == Part::rmsnorm || part == Part::gate ? 2 : 1;
}

/* ---------------------------------------------------------------- parameters */

// A numeric parameter, as the host adjoint reads it: an int or a decimal string. Missing or
// non-numeric leaves false, because an adjoint must not throw.
bool num_param(const std::vector<OpParam>& params, const char* name, double* out) {
    for (const OpParam& p : params) {
        if (p.first != name) continue;
        if (const int64_t* i = std::get_if<int64_t>(&p.second)) {
            *out = (double)*i;
            return true;
        }
        const std::string* s = std::get_if<std::string>(&p.second);
        if (!s || s->empty()) return false;
        char* end = nullptr;
        const double v = std::strtod(s->c_str(), &end);
        if (!end || *end != '\0') return false;
        *out = v;
        return true;
    }
    return false;
}

double param_num(const OpRecord& rec, const char* name, double fallback) {
    double v = 0.0;
    return num_param(rec.params, name, &v) ? v : fallback;
}

// The rotation and the per-group scale are the record's `n` and `group`: a power of two that
// divides the width, no wider than one block's threads. A record outside that is refused at the
// row lookup rather than served approximately.
bool rotated(const std::vector<OpParam>& params) {
    double n = 0.0, group = 0.0;
    if (!num_param(params, "n", &n) || !num_param(params, "group", &group)) return false;
    if (n <= 0.0 || n != std::floor(n)) return false;
    if (group <= 0.0 || group != std::floor(group) || group > 1024.0) return false;
    const int64_t g = (int64_t)group;
    if ((g & (g - 1)) != 0 || (int64_t)n % g != 0) return false;
    return true;
}

// The gated producer's activation: silu, its documented first mode and the one the card's swiglu
// sites issue. The sigmoid form would take a different second half and is not served yet.
bool serves_gated(const std::vector<OpParam>& params) {
    if (!rotated(params)) return false;
    for (const OpParam& p : params) {
        if (p.first != "act") continue;
        const std::string* act = std::get_if<std::string>(&p.second);
        return act && *act == "silu";
    }
    return true;
}

// The gate's activation: sigmoid, the mode the card's narrow-head gate issues. It is required,
// because the record's default is the schema's to say, not this row's to guess.
bool serves_gate(const std::vector<OpParam>& params) {
    if (!rotated(params)) return false;
    for (const OpParam& p : params) {
        if (p.first != "act") continue;
        const std::string* act = std::get_if<std::string>(&p.second);
        return act && *act == "sigmoid";
    }
    return false;
}

/* ---------------------------------------------------------------- kernels */

// One block per (row, group), one thread per element of the group: form the group's rotated
// activation in f64, decide the mask against the replayed forward's codes and scale, then rotate
// the masked cotangent back in f32. The forward transform's stages are the host's rotated()'s, so
// a value one ulp inside or outside the grid lands on the side the reference puts it; the
// butterflies of a stage are independent, so sharing a stage across threads changes no bit.
__global__ void mask_rotate_kernel(const signed char* codes, const double* y, const double* scale,
                                   const float* cot, float* rotated_cot, int n, int group,
                                   int ngrp) {
    extern __shared__ char smem[];
    double* v = reinterpret_cast<double*>(smem);
    float* d = reinterpret_cast<float*>(smem + (size_t)group * sizeof(double));

    const int tid = (int)threadIdx.x;
    const int64_t row = (int64_t)blockIdx.x / ngrp;
    const int gi = (int)((int64_t)blockIdx.x - row * ngrp);
    const int64_t base = row * (int64_t)n + (int64_t)gi * (int64_t)group;

    v[tid] = y[base + tid];
    __syncthreads();
    for (int h = 1; h < group; h <<= 1) {
        if (tid < group / 2) {
            const int j = (tid / h) * (2 * h) + (tid % h);
            const double a = v[j], b = v[j + h];
            v[j] = a + b;
            v[j + h] = a - b;
        }
        __syncthreads();
    }

    const double s = scale[row * (int64_t)ngrp + gi];
    const double u = std::fabs(v[tid] * (1.0 / s));
    const signed char code = codes[base + tid];
    const bool top = code == 127 || code == -127;
    const bool live = top ? u <= 127.0 : true;
    d[tid] = live ? cot[base + tid] / (float)s : 0.0f;
    __syncthreads();
    for (int h = 1; h < group; h <<= 1) {
        if (tid < group / 2) {
            const int j = (tid / h) * (2 * h) + (tid % h);
            const float a = d[j], b = d[j + h];
            d[j] = a + b;
            d[j + h] = a - b;
        }
        __syncthreads();
    }
    rotated_cot[base + tid] = d[tid];
}

// The plain quantiser: the rotated cotangent is the activation's cotangent.
__global__ void quant_back_kernel(const float* rotated_cot, unsigned short* dx, int64_t total) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    dx[i] = f32_to_bf16(rotated_cot[i]);
}

// silu_mul: y = silu(g) * u over gate_up's two halves, so g's cotangent carries the slope and the
// second half's the activation alone.
__global__ void gated_back_kernel(const float* rotated_cot, const float* gate_up,
                                  unsigned short* d, int64_t total, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t row = i / n;
    const int64_t gb = row * 2 * n + (i - row * n);
    const float g = gate_up[gb], u = gate_up[gb + n];
    const float s = 1.0f / (1.0f + expf(-g));
    d[gb] = f32_to_bf16(rotated_cot[i] * u * (s * (1.0f + g * (1.0f - s))));
    d[gb + n] = f32_to_bf16(rotated_cot[i] * (g * s));
}

// The gate: y = sigmoid(g) * x.
__global__ void gate_back_kernel(const float* rotated_cot, const float* gate, const float* x,
                                 unsigned short* dgate, unsigned short* dx, int64_t total) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float s = 1.0f / (1.0f + expf(-gate[i]));
    dgate[i] = f32_to_bf16(rotated_cot[i] * s * (1.0f - s) * x[i]);
    dx[i] = f32_to_bf16(rotated_cot[i] * s);
}

// The RMS norm, one block per row: ss = sum x^2 and t = sum r x c share the row's reduction, c =
// w + wadd. dx_j = mult * (rs c_j r_j - rs^3 x_j t / n), the row's shared term carried by every
// element. ss and t accumulate in f32, per the family's tolerance; the gain is frozen (the host
// rmsnorm adjoint leaves it), so only x's cotangent comes back.
__global__ void rmsnorm_back_kernel(const float* rotated_cot, const float* x, const float* w,
                                    unsigned short* dx, int64_t n, float eps, float wadd,
                                    float mult) {
    __shared__ float sh_ss[256];
    __shared__ float sh_t[256];
    const int tid = (int)threadIdx.x;
    const int64_t base = (int64_t)blockIdx.x * n;

    float ss = 0.0f, t = 0.0f;
    for (int64_t i = tid; i < n; i += blockDim.x) {
        const float xi = x[base + i];
        ss += xi * xi;
        t += rotated_cot[base + i] * xi * (w[i] + wadd);
    }
    sh_ss[tid] = ss;
    sh_t[tid] = t;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sh_ss[tid] += sh_ss[tid + s];
            sh_t[tid] += sh_t[tid + s];
        }
        __syncthreads();
    }

    const float inv_n = 1.0f / (float)n;
    const float rs = rsqrtf(sh_ss[0] * inv_n + eps);
    const float shared = mult * rs * rs * rs * sh_t[0] * inv_n;
    for (int64_t i = tid; i < n; i += blockDim.x) {
        const float c = w[i] + wadd;
        dx[base + i] = f32_to_bf16(mult * (rs * c * rotated_cot[base + i]) - shared * x[base + i]);
    }
}

/* ---------------------------------------------------------------- the call */

// One stream for every device adjoint call in this process, as add.hip has it: upload, launch and
// download are one queue's worth of work, which is what the harness times.
RadStream adjoint_stream() {
    static RadStream s = [] {
        RadStream created = nullptr;
        if (rad_stream_create(&created, 0) != 0) return (RadStream) nullptr;
        return created;
    }();
    return s;
}

int64_t blocks(int64_t n) { return (n + 255) / 256; }

// The value the quantiser rotated: the record's own activation for the plain quantiser, and for a
// fused producer the intermediate the elementwise part wrote, rebuilt through its twin so this
// call and the host reference rotate the same numbers.
bool intermediate(const AdjointCall& call, Part part, std::vector<AdValue>* y) {
    const char* op = part_op(part);
    if (!op) {
        y->assign(1, call.fwd[0]);
        return true;
    }
    TwinFn twin = twin_find(op, call.rec.params, nullptr);
    if (!twin) return false;
    // The parts' twins reject a record they cannot serve by throwing; an adjoint must not throw.
    try {
        twin(call.rec, call.fwd, y);
    } catch (...) {
        return false;
    }
    return !y->empty();
}

// The mask and the rotation: uploads what the decision reads, launches the one kernel, and keeps
// nothing. `d_rotated` is the caller's `rows * n` f32 buffer.
bool mask_and_rotate(const AdjointCall& call, Part part, int64_t rows, int64_t n, int64_t group,
                     int64_t ngrp, const std::vector<double>& y, RadStream s, void* d_rotated) {
    const size_t reads = part_reads(part);
    const AdValue& codes = call.fwd[reads];
    const AdValue& scale = call.fwd[reads + 1];
    const AdValue& cot = call.cot[0];
    const int64_t total = rows * n;

    std::vector<double> h_y((size_t)total), h_scale((size_t)(rows * ngrp));
    std::vector<signed char> h_codes((size_t)total);
    std::vector<float> h_cot((size_t)total);
    for (int64_t i = 0; i < total; ++i) {
        h_y[(size_t)i] = y[(size_t)i];
        h_codes[(size_t)i] = (signed char)codes.value[(size_t)i];
        h_cot[(size_t)i] = (float)cot.value[(size_t)i];
    }
    for (int64_t i = 0; i < rows * ngrp; ++i) h_scale[(size_t)i] = scale.value[(size_t)i];

    device::ScratchBlock d_y(total * (int64_t)sizeof(double));
    device::ScratchBlock d_scale(rows * ngrp * (int64_t)sizeof(double));
    device::ScratchBlock d_codes(total);
    device::ScratchBlock d_cot(total * (int64_t)sizeof(float));
    if (!d_y || !d_scale || !d_codes || !d_cot) return false;
    if (rad_memcpy_async(d_y.p, h_y.data(), total * 8, s) != 0) return false;
    if (rad_memcpy_async(d_scale.p, h_scale.data(), rows * ngrp * 8, s) != 0) return false;
    if (rad_memcpy_async(d_codes.p, h_codes.data(), total, s) != 0) return false;
    if (rad_memcpy_async(d_cot.p, h_cot.data(), total * 4, s) != 0) return false;

    const size_t shared = (size_t)(group * ((int64_t)sizeof(double) + (int64_t)sizeof(float)));
    hipLaunchKernelGGL(mask_rotate_kernel, dim3((unsigned int)(rows * ngrp)),
                       dim3((unsigned int)group), shared, (hipStream_t)s,
                       (const signed char*)d_codes.p, (const double*)d_y.p,
                       (const double*)d_scale.p, (const float*)d_cot.p, (float*)d_rotated,
                       (int)n, (int)group, (int)ngrp);
    return rad_stream_sync(s) == 0;
}

// The tape's cotangents are f64 and are added to, never overwritten: other ops share them.
void add_back(unsigned short* d_out, int64_t count, AdValue& into, RadStream s) {
    std::vector<unsigned short> h((size_t)count);
    if (rad_memcpy_async(h.data(), d_out, count * 2, s) != 0) return;
    if (rad_stream_sync(s) != 0) return;
    for (int64_t i = 0; i < count; ++i) into.value[(size_t)i] += bf16_to_f32(h[(size_t)i]);
}

void quant_adjoint(const AdjointCall& call, Part part) {
    if (!call.dc || call.cot.size() < 2) return;
    size_t reads = 0;
    for (const Operand& o : call.rec.operands)
        if (!o.written) ++reads;
    if (reads != part_reads(part) || call.rec.operands.size() != reads + 2) return;
    if (call.fwd.size() != call.rec.operands.size() || call.dc->size() < reads) return;

    const double n_d = param_num(call.rec, "n", 0.0);
    const double group_d = param_num(call.rec, "group", 0.0);
    if (n_d <= 0.0 || group_d <= 0.0 || group_d > 1024.0) return;
    const int64_t n = (int64_t)n_d, group = (int64_t)group_d;
    if ((group & (group - 1)) != 0 || n % group != 0) return;

    const AdValue& codes = call.fwd[reads];
    const AdValue& scale = call.fwd[reads + 1];
    const AdValue& cot = call.cot[0];
    const int64_t ngrp = n / group;
    if (codes.shape.size() != 2 || codes.shape[1] != n) return;
    if (scale.shape.size() != 2 || scale.shape[1] != ngrp) return;
    const int64_t rows = codes.shape[0];
    const int64_t total = rows * n;
    if (rows <= 0 || (int64_t)codes.value.size() != total) return;
    if ((int64_t)scale.value.size() != rows * ngrp) return;
    if ((int64_t)cot.value.size() != total) return;

    // What the backprop reads: the elementwise operands' widths, and the slots it answers for.
    const int64_t arg_count = part == Part::quant ? 0
                              : part == Part::rmsnorm ? total + n
                                                      : 2 * total;
    // The gated producer answers for gate_up's two halves; the gate answers for two operands.
    const int64_t out_slot = part == Part::gated ? 2 * total : total;
    const int64_t out_count = part == Part::gated || part == Part::gate ? 2 * total : total;
    if (part == Part::rmsnorm && (int64_t)call.fwd[0].value.size() != total) return;
    if (part == Part::rmsnorm && (int64_t)call.fwd[1].value.size() < n) return;
    if (part != Part::rmsnorm && (int64_t)call.fwd[0].value.size() !=
                                     (part == Part::gated ? 2 * total : total))
        return;
    if (part == Part::gate && (int64_t)call.fwd[1].value.size() != total) return;
    if ((int64_t)call.dc->at(0).value.size() < out_slot) return;
    if (part == Part::gate && (int64_t)call.dc->at(1).value.size() < total) return;

    std::vector<AdValue> y;
    if (!intermediate(call, part, &y)) return;
    if ((int64_t)y[0].value.size() != total) return;

    RadStream s = adjoint_stream();
    if (!s) return;

    device::ScratchBlock d_rotated(total * (int64_t)sizeof(float));
    if (!d_rotated) return;
    if (!mask_and_rotate(call, part, rows, n, group, ngrp, y[0].value, s, d_rotated.p)) return;

    // The elementwise part's own operands, at f32: they carry a smooth derivative, not a decision.
    device::ScratchBlock d_arg(arg_count * (int64_t)sizeof(float));
    if (arg_count > 0 && !d_arg) return;
    if (arg_count > 0) {
        std::vector<float> h_arg((size_t)arg_count);
        for (int64_t i = 0; i < total; ++i) h_arg[(size_t)i] = (float)call.fwd[0].value[(size_t)i];
        if (part == Part::rmsnorm || part == Part::gate) {
            const AdValue& second = call.fwd[1];
            const int64_t count = part == Part::rmsnorm ? n : total;
            for (int64_t i = 0; i < count; ++i)
                h_arg[(size_t)(total + i)] = (float)second.value[(size_t)i];
        }
        if (rad_memcpy_async(d_arg.p, h_arg.data(), arg_count * 4, s) != 0) return;
    }

    device::ScratchBlock d_out(out_count * (int64_t)sizeof(unsigned short));
    if (!d_out) return;

    const float* rotated = (const float*)d_rotated.p;
    const float* arg = (const float*)d_arg.p;
    switch (part) {
        case Part::quant:
            hipLaunchKernelGGL(quant_back_kernel, dim3((unsigned int)blocks(total)), dim3(256), 0,
                               (hipStream_t)s, rotated, (unsigned short*)d_out.p, total);
            break;
        case Part::gated:
            hipLaunchKernelGGL(gated_back_kernel, dim3((unsigned int)blocks(total)), dim3(256), 0,
                               (hipStream_t)s, rotated, arg, (unsigned short*)d_out.p, total, n);
            break;
        case Part::gate:
            hipLaunchKernelGGL(gate_back_kernel, dim3((unsigned int)blocks(total)), dim3(256), 0,
                               (hipStream_t)s, rotated, arg, arg + total,
                               (unsigned short*)d_out.p, (unsigned short*)d_out.p + total, total);
            break;
        case Part::rmsnorm:
            hipLaunchKernelGGL(rmsnorm_back_kernel, dim3((unsigned int)rows), dim3(256), 0,
                               (hipStream_t)s, rotated, arg, arg + total,
                               (unsigned short*)d_out.p, n,
                               (float)param_num(call.rec, "eps", 1e-6),
                               (float)param_num(call.rec, "wadd", 0.0),
                               (float)param_num(call.rec, "mult", 1.0));
            break;
    }

    add_back((unsigned short*)d_out.p, out_slot, call.dc->at(0), s);
    if (part == Part::gate) add_back((unsigned short*)d_out.p + total, total, call.dc->at(1), s);
}

void had_device_adjoint(const AdjointCall& call) { quant_adjoint(call, Part::quant); }
void rmsnorm_device_adjoint(const AdjointCall& call) { quant_adjoint(call, Part::rmsnorm); }
void gated_device_adjoint(const AdjointCall& call) { quant_adjoint(call, Part::gated); }
void gate_device_adjoint(const AdjointCall& call) { quant_adjoint(call, Part::gate); }

}  // namespace

R97_REGISTER_ADJOINT_IF("had_quant_act_i8g", device, had_device_adjoint, rotated)
R97_REGISTER_ADJOINT_IF_F("rmsnorm_had_quant_i8g", device, rmsnorm_device_adjoint, rotated, 1)
R97_REGISTER_ADJOINT_IF("gated_had_quant_i8g", device, gated_device_adjoint, serves_gated)
R97_REGISTER_ADJOINT_IF("gate_had_quant_i8g", device, gate_device_adjoint, serves_gate)

}  // namespace r97
