// tests/gpu/adjoint_lora.cpp -- the adapter's adjoints on the card, held to the host rows.
//
// src/adjoint/device/lora.hip answers for `lora_shrink` and `lora_expand` on the coded trunk, and
// its reference is the host row of each op (src/adjoint/host/lora.cpp, docs/adjoints/lora.md):
//
//   * at the tiny fixture's geometry (hidden 128, ffn 256, docs/FIXTURES.md) and at the real
//     model's (hidden 5120, docs/DESIGN.md) the device adjoint matches the host adjoint on the
//     same random codes, scale, pool and cotangents within the harness's bf16 tolerance;
//   * the gradients -- dA over the tokens, dB over the tokens and the adapter's contribution to
//     dx -- are leaves of the walk's ParamSet, so they are not among the dc slots the shared
//     harness compares: this file runs both rows with a ParamSet and holds dA and dB to the same
//     tolerance, measured against the block's largest entry because a gradient entry is a sum that
//     cancels and an f32 sum's error follows the terms it summed, not the entry it lands in;
//   * a gradient is the step's, not the call's (src/adjoint/device/grad_stage.h): the row adds a
//     call's sum into an f32 accumulator on the card and touches no leaf, the accumulator is the
//     same f32 buffer for every token of every call of the step, and it is rounded to bf16 once --
//     at the step's end, when the walk copies it into the ParamSet. A step whose tokens arrive in
//     two calls is held to the host row over the sum of the two calls;
//   * the shapes the standard names: ranks 4 and 16 on the real widths, 1024 and 4096 tokens, one
//     sequence and a batch of unequal lengths (the adapter sees a batch of unequal lengths as its
//     total token count, so 448 + 320 + 256 is the 1024-token row at the ffn's widths);
//   * the adapter that has not moved off its start -- B = 0 -- gives dA = 0, and the harness
//     rejects a deliberately wrong reference rather than accepting anything plausible.
//
// Needs one R9700 (gfx1201, 32 GB) or similar; runs under the GPU test guard. No container is
// read: the real widths are numbers from the design, so this test carries the `gpu` label only.
#include "r97/adjoint.h"

#include "adjoint/device/grad_stage.h"

#include "support/bench.h"
#include "support/device_check.h"
#include "support/gpu_guard.h"
#include "support/test.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace r97;
using device_check::Case;
using device_check::Report;

// The device side rounds a gradient to bf16 once, at the step's end, after the f32 sum over all of
// the step's tokens: an entry carries about 2^-9 of relative rounding, and the sum's f32 noise
// follows the terms, not the entry. Both are measured against the block's largest entry, so the
// tolerance is the harness's bf16 relative bound on that scale.
constexpr double kGradScale = device_check::kRelTol;

// One step of bf16 rounding: what a claim that differs only in which side of a rounding tie a term
// fell on may be off by.
constexpr double kOneBf16 = 1.0 / 256.0;

/* ------------------------------------------------------------------ the sites */

/* One site of the adapter pool as an op call: a row band of M tokens, the trunk's K in, N out,
 * rank R (the pool's packed columns), S pool rows, and the int8 trunk's scale group width.
 * `packed` is the real packing: two modules of R/2 in one pool, each row of B carrying only its
 * own module's columns's halves and zero elsewhere (docs/ADAPTER.md 2.1). */
struct Site {
    const char* name;
    const char* module;  // the PEFT module path the gradients are named after
    int64_t M, K, N, R, S, group;
    bool packed;
};

const Site sites[] = {
    // The tiny fixture: hidden 128, ffn 256, ranks 4 and 16.
    {"tiny_q_proj r16", "model.layers.0.self_attn.q_proj", 6, 128, 128, 16, 2, 128, false},
    {"tiny_down_proj r4", "model.layers.0.mlp.down_proj", 4, 128, 256, 4, 1, 128, false},
    // K = 300 is three groups of 128, 128 and 44 wide: the shape that catches a width read as
    // ceil(K / groups) = 100, which would take the wrong scale over half the row.
    {"tiny_up_proj k300 r4", "model.layers.0.mlp.up_proj", 5, 300, 96, 4, 2, 128, false},
    // A pool wider than a token's row of shared memory: the row comes from global memory instead,
    // and the kernel still answers.
    {"tiny_q_proj r2048", "model.layers.0.self_attn.q_proj", 5, 128, 128, 2048, 1, 128, false},
    // The real model: hidden 5120, q_proj out 12288, ffn_down in 17408, ffn_gate_up out 34816.
    {"q_proj r16 1024 tok, one sequence", "model.layers.13.self_attn.q_proj", 1024, 5120, 12288, 16,
     1, 128, false},
    {"q_proj r16 4096 tok, one sequence", "model.layers.13.self_attn.q_proj", 4096, 5120, 12288, 16,
     1, 128, false},
    {"down_proj r16 448+320+256 tok", "model.layers.13.mlp.down_proj", 1024, 17408, 5120, 16, 1, 128,
     false},
    {"q_proj r4 1024 tok, one sequence", "model.layers.13.self_attn.q_proj", 1024, 5120, 12288, 4, 1,
     128, false},
    {"gate_up r32 1024 tok, two modules", "model.layers.13.mlp.gate_up_proj", 1024, 5120, 34816, 32,
     1, 128, true},
};

const Site& site_named(const char* name) {
    for (const Site& s : sites)
        if (std::strcmp(s.name, name) == 0) return s;
    throw std::runtime_error(std::string("no site named ") + name);
}

/* Row m's adapter. Row 1 is a base row spelled -1 and row 2 one spelled past the pool (S), the two
 * spellings a record uses, so the base-row rule is exercised wherever there is room for it. Real
 * rows all take the one pool row a delta layer's site carries. */
int32_t slot_of(const Site& s, int64_t m) {
    if (s.M > 16) return 0;
    if (m == 1) return -1;
    if (m == 2) return (int32_t)s.S;
    return (int32_t)(m % s.S);
}

/* ------------------------------------------------------------------ records */

/* `lora_shrink` in the coded form: a, a_scale, slot, lora_a, then the write t. The schema is five
 * operands wide (an optional operand keeps its position), and the coded form fills all five with a
 * real a_scale -- the float activation's is the absent one, which this test only asks the registry
 * about. */
OpRecord shrink_rec(const Site& s, bool coded) {
    OpRecord rec;
    rec.op = "lora_shrink";
    rec.params = {{"M", s.M},
                  {"K", s.K},
                  {"R", s.R},
                  {"S", s.S},
                  {"group", coded ? s.group : 0},
                  {"dtype", coded ? "i8" : "bf16"}};
    rec.operands = {{"a", coded ? "i8" : "bf16", {s.M, s.K}, 0, s.M, false}};
    if (coded)
        rec.operands.push_back(
            {"a_scale", "f32", {s.M, (s.K + s.group - 1) / s.group}, 0, s.M, false});
    else
        rec.operands.push_back({"a_scale", "invalid", {}, 0, 0, false});
    rec.operands.push_back({"slot", "i32", {s.M}, 0, s.M, false});
    rec.operands.push_back({"lora_a", "bf16", {s.S, s.R, s.K}, 0, s.S, false});
    rec.operands.push_back({"t", "f32", {s.M, s.R}, 0, s.M, true});
    return rec;
}

/* `lora_expand`: t, slot, lora_b, scale, then the inout write y. */
OpRecord expand_rec(const Site& s) {
    OpRecord rec;
    rec.op = "lora_expand";
    rec.params = {{"M", s.M}, {"N", s.N}, {"R", s.R}, {"S", s.S}, {"dtype", "bf16"}};
    rec.operands = {{"t", "f32", {s.M, s.R}, 0, s.M, false},
                    {"slot", "i32", {s.M}, 0, s.M, false},
                    {"lora_b", "bf16", {s.S, s.N, s.R}, 0, s.S, false},
                    {"scale", "f32", {s.S, s.N}, 0, s.S, false},
                    {"y", "bf16", {s.M, s.N}, 0, s.M, true}};
    return rec;
}

/* ------------------------------------------------------------------ values */

/* bf16 is the high half of an f32 with round-half-to-even, the same rounding radiance's kernels
 * and the trunk store do. */
double as_bf16(double v) {
    uint32_t u = 0;
    const float f = (float)v;
    std::memcpy(&u, &f, sizeof u);
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    float r = 0.0f;
    std::memcpy(&r, &u, sizeof r);
    return (double)r;
}

double urand(uint64_t* seed) { return device_check::urand(seed); }

/* The value a buffer name holds, in the shapes the real step gives it: codes as integers, the
 * scale positive f32, the pool and the factors bf16 exactly, t f32, y the trunk's bf16 base. */
AdValue value_of(const Operand& o, const Site& s, bool zero_b, uint64_t* seed) {
    AdValue v;
    v.dtype = o.dtype;
    v.shape = o.shape;
    int64_t n = 1;
    for (int64_t d : o.shape) n *= d;
    if (o.dtype == "invalid") return v;
    v.value.assign((size_t)n, 0.0);
    if (o.buffer == "slot") {
        for (int64_t i = 0; i < n; ++i) v.value[(size_t)i] = slot_of(s, i);
    } else if (o.buffer == "a") {
        for (int64_t i = 0; i < n; ++i)
            v.value[(size_t)i] = (double)(int64_t)std::llround(urand(seed) * 60.0);
    } else if (o.buffer == "a_scale") {
        // One positive f32 scale per group of the rotated stream.
        for (int64_t i = 0; i < n; ++i)
            v.value[(size_t)i] =
                (double)(float)(0.02 + 0.01 * (double)(i % 7) + 0.004 * std::fabs(urand(seed)));
    } else if (o.buffer == "lora_a") {
        for (int64_t i = 0; i < n; ++i) v.value[(size_t)i] = as_bf16(urand(seed));
    } else if (o.buffer == "lora_b") {
        if (!zero_b) {
            for (int64_t i = 0; i < n; ++i) v.value[(size_t)i] = as_bf16(urand(seed));
            if (s.packed) {
                // A row of B carries only its own module's half and zero outside it.
                const int64_t r = s.R / 2;
                for (int64_t nn = 0; nn < s.N; ++nn) {
                    const int64_t own = nn < s.N / 2 ? 0 : 1;
                    for (int64_t j = 0; j < s.R; ++j)
                        if (j / r != own) v.value[(size_t)(nn * s.R + j)] = 0.0;
                }
            }
        }
    } else if (o.buffer == "t") {
        for (int64_t i = 0; i < n; ++i) v.value[(size_t)i] = (double)(float)urand(seed);
    } else if (o.buffer == "scale") {
        // alpha/r, 2.0 by default (docs/ADAPTER.md 4), a row at a time so a row read from the
        // wrong pool entry shows.
        for (int64_t i = 0; i < n; ++i)
            v.value[(size_t)i] = (double)(float)(2.0 / (1 + (i / s.N) % 2));
    } else {  // y: the base the trunk's GEMM wrote
        for (int64_t i = 0; i < n; ++i) v.value[(size_t)i] = as_bf16(urand(seed));
    }
    return v;
}

uint64_t seed_of(const Site& s, const char* op) {
    return 0x1000ull + (uint64_t)(s.M * 31 + s.K * 7 + s.N * 3 + s.R * 11) * (op[0] == 'l' ? 1 : 3);
}

Case make_case(const Site& s, const char* op, bool zero_b, bool zero_t) {
    uint64_t seed = seed_of(s, op);
    Case c;
    c.rec = std::strcmp(op, "lora_shrink") == 0 ? shrink_rec(s, true) : expand_rec(s);
    for (const Operand& o : c.rec.operands) {
        if (o.written) {
            AdValue v = device_check::random_value(o.dtype, o.shape, &seed);
            if (zero_t && o.buffer == "t") {
                // A cotangent of zero: the step where the adapter's B is zero, so nothing above it
                // asks for its input's cotangent and dA comes out zero.
                for (double& x : v.value) x = 0.0;
            }
            if (o.dtype == "bf16")
                for (double& x : v.value) x = as_bf16(x);
            c.cot.push_back(std::move(v));
            continue;
        }
        c.in.push_back(value_of(o, s, zero_b && o.buffer == "lora_b", &seed));
    }
    c.label = std::string(op) + " " + s.name;
    return c;
}

/* ------------------------------------------------------------------ the gradients */

/* What a gradient check found: the block's largest entry and the worst difference against it. */
struct GradReport {
    bool ok = true;
    std::string note;
    double worst_rel = 0.0;  // the largest |device - host|, in units of the block's largest entry
};

const Param* param_named(const ParamSet& set, const std::string& name) {
    for (const Param& p : set)
        if (p.name == name) return &p;
    return nullptr;
}

/* Runs both rows with a ParamSet and holds the leaves the card wrote to the host row's, entry by
 * entry. `zero_a` asks that dA be exactly zero, which is what the adapter that has not moved off
 * its start gives (B = 0 above it, so nothing asks for its input's cotangent).
 */
GradReport check_grads(const Case& c, const char* prefix, bool zero_a) {
    GradReport g;
    std::string err;
    AdjointFn device = adjoint_find(c.rec.op, Side::device, c.rec.params, &err);
    if (!device) {
        g.ok = false;
        g.note = c.label + ": device adjoint: " + err;
        return g;
    }
    AdjointFn host = adjoint_find(c.rec.op, Side::host, c.rec.params, &err);
    if (!host) {
        g.ok = false;
        g.note = c.label + ": host adjoint: " + err;
        return g;
    }

    std::vector<AdValue> device_fwd, device_dc;
    AdjointCall device_call = device_check::make_call(c, &device_fwd, &device_dc);
    ParamSet device_grads;
    device_call.grads = &device_grads;
    device_call.prefix = prefix;
    device(device_call);

    // The call leaves the step's total on the card: the walk's ParamSet is untouched until the step
    // ends, and what the step is holding is the f32 accumulator, not a bf16 leaf.
    if (!device_grads.empty()) {
        g.ok = false;
        g.note = c.label + ": the call copied gradients into the ParamSet before the step ended";
        return g;
    }
    if (!device::grad_stage_flush(&device_grads)) {
        g.ok = false;
        g.note = c.label + ": the step's staged gradients did not copy out";
        return g;
    }
    if (device::grad_stage_bytes() != 0) {
        g.ok = false;
        g.note = c.label + ": the step's staging was not freed when it ended";
        return g;
    }

    // The reference sees the same forward values and the same cotangent as the device did, at zero
    // dc and at an empty ParamSet.
    std::vector<AdValue> host_dc = device_dc;
    for (AdValue& v : host_dc) std::fill(v.value.begin(), v.value.end(), 0.0);
    ParamSet host_grads;
    AdjointCall host_call{c.rec, device_fwd, c.cot, &host_dc, &host_grads, prefix};
    host(host_call);

    if (device_grads.size() != host_grads.size()) {
        g.ok = false;
        g.note = c.label + ": device wrote " + std::to_string(device_grads.size())
                 + " gradient leaves, host " + std::to_string(host_grads.size());
        return g;
    }
    for (const Param& h : host_grads) {
        const Param* d = param_named(device_grads, h.name);
        if (!d) {
            g.ok = false;
            g.note = c.label + ": device wrote no gradient named " + h.name;
            return g;
        }
        if (d->shape != h.shape || d->value.size() != h.value.size()) {
            g.ok = false;
            g.note = c.label + ": gradient " + h.name + " has the wrong shape";
            return g;
        }
        double scale = 0.0;
        for (float v : h.value) scale = std::max(scale, (double)std::fabs(v));
        const double tol = device_check::kAbsTol + kGradScale * scale;
        for (size_t i = 0; i < h.value.size(); ++i) {
            const double dv = d->value[i];
            const double hv = h.value[i];
            const double rel = scale > 0.0 ? std::fabs(dv - hv) / scale : std::fabs(dv - hv);
            g.worst_rel = std::max(g.worst_rel, rel);
            if (std::isfinite(dv) && std::fabs(dv - hv) <= tol) continue;
            g.ok = false;
            g.note = c.label + ": gradient " + h.name + " index " + std::to_string(i)
                     + ": device " + std::to_string(dv) + ", host " + std::to_string(hv)
                     + " (tolerance " + std::to_string(tol) + ")";
            return g;
        }
        if (zero_a) {
            size_t nonzero = 0;
            for (float v : d->value) nonzero += v != 0.0f ? 1 : 0;
            if (nonzero != 0) {
                g.ok = false;
                g.note = c.label + ": gradient " + h.name + " is not zero under B = 0 ("
                         + std::to_string(nonzero) + " entries)";
                return g;
            }
        }
    }
    return g;
}

/* The gradient is filled in, not sparse by accident: a zero code or a zero factor zeroes a column
 * of it honestly, so the statement is that most of the block is away from zero -- unless it is a
 * base row's pool entry or the adapter has not moved. */
void require_filled(const ParamSet& grads) {
    for (const Param& p : grads) {
        size_t nonzero = 0;
        for (float v : p.value) nonzero += v != 0.0f ? 1 : 0;
        R97_CHECK(nonzero * 2 >= p.value.size());
    }
}

void require(const GradReport& g) {
    if (!g.ok) throw std::runtime_error(g.note);
}

/* ------------------------------------------------------------------ timing */

/* The time of the whole call with the gradients in it: upload, the kernels and the f32 sum over
 * the tokens into the step's accumulator on the card. The harness's own bench times the call the
 * contract states (no ParamSet), which the doc quotes beside this. */
double time_with_grads_ms(const Case& c, int reps) {
    AdjointFn device = adjoint_find(c.rec.op, Side::device, c.rec.params, nullptr);
    if (!device) throw std::runtime_error("no device adjoint for '" + c.rec.op + "'");
    std::vector<AdValue> fwd, dc;
    AdjointCall call = device_check::make_call(c, &fwd, &dc);
    ParamSet grads;
    call.grads = &grads;
    device(call);  // warm-up: the kernel's module is finalized and the step's accumulator is made
    auto zero = [&dc] { for (AdValue& v : dc) std::fill(v.value.begin(), v.value.end(), 0.0); };
    zero();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        zero();
        device(call);
    }
    const auto t1 = std::chrono::steady_clock::now();
    device::grad_stage_discard();  // the timing does not want the step's gradients
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / std::max(reps, 1);
}

/* The step's end, as the walk pays for it: one call has put the step's f32 total on the card, and
 * grad_stage_flush rounds it once and copies it into the ParamSet. The copy is what the walk does
 * once per step, so the number is the whole of the step's gradient traffic besides the calls. */
double time_flush_ms(const Case& c, int reps) {
    AdjointFn device = adjoint_find(c.rec.op, Side::device, c.rec.params, nullptr);
    if (!device) throw std::runtime_error("no device adjoint for '" + c.rec.op + "'");
    std::vector<AdValue> fwd, dc;
    AdjointCall call = device_check::make_call(c, &fwd, &dc);
    ParamSet grads;
    call.grads = &grads;
    device(call);  // warm-up, and the leaf the flush will add to
    device::grad_stage_flush(&grads);
    double total = 0.0;
    for (int i = 0; i < reps; ++i) {
        for (AdValue& v : dc) std::fill(v.value.begin(), v.value.end(), 0.0);
        device(call);  // one call's tokens, staged on the card
        const auto t0 = std::chrono::steady_clock::now();
        if (!device::grad_stage_flush(&grads)) throw std::runtime_error("the flush failed");
        const auto t1 = std::chrono::steady_clock::now();
        total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return total / std::max(reps, 1);
}

/* Deliberately wrong: the value-space cotangent without the group scale, which a coded record's
 * adjoint must carry, and no scale cotangent at all. */
void wrong_shrink_reference(AdjointCall& call) {
    const OpRecord& rec = call.rec;
    const AdValue& slot = call.fwd[2];
    const AdValue& la = call.fwd[3];
    const AdValue& t_cot = call.cot[0];
    int64_t M = 0, K = 0, S = 0, R = 0;
    // The record's shapes, read the way the op does.
    for (const Operand& o : rec.operands) {
        if (o.buffer == "a") {
            M = o.shape[0];
            K = o.shape[1];
        } else if (o.buffer == "lora_a") {
            S = o.shape[0];
            R = o.shape[1];
        }
    }
    AdValue& a_cot = call.dc->at(0);
    for (int64_t m = 0; m < M; ++m) {
        const int64_t s = (int64_t)slot.value[(size_t)m];
        if (s < 0 || s >= S) continue;
        for (int64_t k = 0; k < K; ++k) {
            double g = 0.0;
            for (int64_t j = 0; j < R; ++j)
                g += t_cot.value[(size_t)(m * R + j)] * la.value[(size_t)((s * R + j) * K + k)];
            a_cot.value[(size_t)(m * K + k)] += g;  // the scale is missing
        }
    }
}

/* ------------------------------------------------------------------ the test */

void check_site(const Site& s, int reps) {
    for (const char* op : {"lora_shrink", "lora_expand"}) {
        const Case c = make_case(s, op, false, false);
        std::printf("%s: ", c.label.c_str());
        const Report r = device_check::check(c);
        device_check::require(r);
        R97_CHECK(r.scratch_peak > 0);
        std::printf("device == host, scratch peak %lld KiB\n", (long long)(r.scratch_peak / 1024));
        const GradReport g = check_grads(c, s.module, false);
        std::printf("%s: gradients device == host, worst %.2g of the block\n", c.label.c_str(),
                    g.worst_rel);
        require(g);
        if (r97test::bench()) {
            device_check::print_bench(c, reps);
            std::printf("%-44s %8.3f ms adjoint with dA/dB\n", c.label.c_str(),
                        time_with_grads_ms(c, reps));
            std::printf("%-44s %8.3f ms step's flush into the ParamSet\n", c.label.c_str(),
                        time_flush_ms(c, reps));
            std::fflush(stdout);
        }
    }
}

void run_fixture() {
    const auto options = device_check::guard_options(int64_t(1) << 30);
    const r97test::DeviceGuard guard(options);
    device_check::require_device(options.device);
    (void)device_check::load_plugin();

    // The rows this file's kernels sit behind: the coded shrink and the expand, with the operands
    // the host row freezes, and nothing registered for the float trunk's shrink.
    const std::vector<int> shrink_frozen = {2, 3};
    const std::vector<int> expand_frozen = {1, 2, 3};
    const OpRows shrink_rows = op_rows("lora_shrink");
    R97_CHECK(shrink_rows.device_adjoint);
    R97_CHECK(shrink_rows.device_frozen == shrink_frozen);
    const OpRows expand_rows = op_rows("lora_expand");
    R97_CHECK(expand_rows.device_adjoint);
    R97_CHECK(expand_rows.device_frozen == expand_frozen);

    {
        std::string err;
        const Site& s = site_named("tiny_q_proj r16");
        const OpRecord float_rec = shrink_rec(s, false);
        R97_CHECK(adjoint_find("lora_shrink", Side::device, float_rec.params, &err) == nullptr);
        R97_CHECK(err.find("lora_shrink") != std::string::npos);
        std::printf("the float trunk's shrink has no device row: %s\n", err.c_str());
    }

    const Site& tiny_r16 = site_named("tiny_q_proj r16");
    const Site& tiny_r4 = site_named("tiny_down_proj r4");
    const Site& k300 = site_named("tiny_up_proj k300 r4");
    const Site& wide = site_named("tiny_q_proj r2048");
    for (const Site* s : {&tiny_r16, &tiny_r4, &k300, &wide}) {
        for (const char* op : {"lora_shrink", "lora_expand"}) {
            const Case c = make_case(*s, op, false, false);
            const Report r = device_check::check(c);
            device_check::require(r);
            R97_CHECK(r.scratch_peak > 0);
            std::printf("%s: device == host, scratch peak %lld KiB\n", c.label.c_str(),
                        (long long)(r.scratch_peak / 1024));
            const GradReport g = check_grads(c, s->module, false);
            require(g);
            std::printf("%s: gradients device == host, worst %.2g of the block\n", c.label.c_str(),
                        g.worst_rel);
        }
    }

    // The gradient carries the walk's prefix, and it is filled in rather than left at zero.
    {
        const Case c = make_case(tiny_r16, "lora_shrink", false, false);
        std::vector<AdValue> fwd, dc;
        AdjointCall call = device_check::make_call(c, &fwd, &dc);
        ParamSet grads;
        call.grads = &grads;
        call.prefix = "model.layers.0.self_attn.q_proj";
        AdjointFn device = adjoint_find("lora_shrink", Side::device, c.rec.params, nullptr);
        device(call);
        R97_CHECK(grads.empty());
        R97_CHECK(device::grad_stage_bytes() == int64_t(tiny_r16.R) * tiny_r16.K * 4);
        R97_CHECK(device::grad_stage_flush(&grads));
        R97_CHECK(device::grad_stage_bytes() == 0);
        R97_CHECK(grads.size() == 1);
        R97_CHECK(grads[0].name == "model.layers.0.self_attn.q_proj.lora_A");
        R97_CHECK(grads[0].shape == std::vector<int64_t>({tiny_r16.R, tiny_r16.K}));
        require_filled(grads);
        std::printf("dA is named %s and is [%lld, %lld]\n", grads[0].name.c_str(),
                    (long long)grads[0].shape[0], (long long)grads[0].shape[1]);

        // A step's tokens in two calls: the second call adds into the accumulator the first one made
        // -- the same buffer, not a second one -- and nothing reaches the leaf until the walk ends
        // the step, which rounds the total once. So a step of two identical calls is twice the step
        // of one, to within that one rounding (the sum's order puts a term on either side of a
        // rounding tie).
        std::vector<AdValue> fwd2, dc2;
        AdjointCall again = device_check::make_call(c, &fwd2, &dc2);
        ParamSet doubled;
        call.grads = &doubled;
        again.grads = &doubled;
        again.prefix = call.prefix;
        const std::vector<float> one_call = grads[0].value;
        device(call);
        R97_CHECK(doubled.empty());                     // the step's first call copies nothing out
        R97_CHECK(grads[0].value == one_call);          // and the leaf the step before left is its own
        device(again);
        R97_CHECK(doubled.empty());                     // the second call copies nothing out either
        R97_CHECK(grads[0].value == one_call);
        R97_CHECK(device::grad_stage_bytes() == int64_t(tiny_r16.R) * tiny_r16.K * 4);
        R97_CHECK(device::grad_stage_flush(&doubled));
        R97_CHECK(doubled.size() == 1);
        for (size_t i = 0; i < grads[0].value.size(); ++i) {
            const double twice = 2.0 * grads[0].value[i];
            R97_CHECK(std::fabs(doubled[0].value[i] - twice) <= kOneBf16 * std::fabs(twice));
        }
        std::printf("a step of two calls sums them in f32 on the card and rounds the total once\n");
    }

    // The harness rejects a reference that drops the coded record's scale.
    {
        const Case c = make_case(tiny_r16, "lora_shrink", false, false);
        const Report wrong = device_check::check(c, nullptr, wrong_shrink_reference);
        R97_CHECK(!wrong.ok);
        R97_CHECK(wrong.first_slot == 0);
        std::printf("rejected the wrong reference: %s\n", wrong.note.c_str());
    }

    // The adapter that has not moved off its start: B = 0. Nothing asks for the expand's input's
    // cotangent, so dt is zero -- and dB is not, because dB is what moves B off zero. The shrink's
    // dA is zero when the cotangent of t it reads is zero, which is the dt the expand gave.
    {
        const Case expand = make_case(tiny_r16, "lora_expand", true, false);
        const Report r = device_check::check(expand);
        device_check::require(r);
        std::vector<AdValue> fwd, dc;
        AdjointCall call = device_check::make_call(expand, &fwd, &dc);
        ParamSet grads;
        call.grads = &grads;
        call.prefix = tiny_r16.module;
        adjoint_find("lora_expand", Side::device, expand.rec.params, nullptr)(call);
        R97_CHECK(device::grad_stage_flush(&grads));
        R97_CHECK(grads.size() == 1);
        size_t dt_nonzero = 0;
        for (double v : dc[0].value) dt_nonzero += v != 0.0 ? 1 : 0;
        R97_CHECK(dt_nonzero == 0);
        size_t db_nonzero = 0;
        for (float v : grads[0].value) db_nonzero += v != 0.0f ? 1 : 0;
        R97_CHECK(db_nonzero > 0);
        std::printf("%s: dt is zero and dB is filled with B = 0 (%zu of %zu)\n",
                    expand.label.c_str(), db_nonzero, grads[0].value.size());

        const Case shrink = make_case(tiny_r16, "lora_shrink", false, true);
        const GradReport g = check_grads(shrink, tiny_r16.module, true);
        require(g);
        R97_CHECK(g.worst_rel == 0.0);
        std::printf("%s: with a zero cotangent of t, dA is exactly zero\n",
                    shrink.label.c_str());
    }
}

void run_real() {
    const auto options = device_check::guard_options(int64_t(2) << 30);
    const r97test::DeviceGuard guard(options);
    device_check::require_device(options.device);
    (void)device_check::load_plugin();

    struct Real {
        const char* name;
        int reps;
    };
    const Real shapes[] = {
        {"q_proj r16 1024 tok, one sequence", 6},
        {"q_proj r16 4096 tok, one sequence", 3},
        {"down_proj r16 448+320+256 tok", 6},
        {"q_proj r4 1024 tok, one sequence", 6},
        {"gate_up r32 1024 tok, two modules", 3},
    };
    for (const Real& shape : shapes) check_site(site_named(shape.name), shape.reps);
}

}  // namespace

R97_TEST(adjoint_lora_fixture) { run_fixture(); }
R97_TEST(adjoint_lora_real) { run_real(); }
