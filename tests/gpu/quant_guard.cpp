// tests/gpu/quant_guard.cpp -- the mask pass's quotient guard, on the quantiser's own record.
//
// The mask pass divides each code's cotangent by its group scale, and that division is exact only
// while the quotient stays inside f32. A cotangent of 1e38 over a 1e-3 scale is 1e41: an Inf, which
// enters the inverse rotation and then the elementwise backprop, so the run's first non-finite
// value names a row that only did its own arithmetic. This pins three things about the guard that
// counts it:
//
//   * the count is exact, not a bound: it is taken where the division happens, so an element the
//     mask zeroed is not counted, and a group whose scale keeps the quotient finite is not either;
//   * the default is to report and carry on, because a container whose cotangents already divide
//     past f32 must not abort before the measurement that says why;
//   * `R97_QUANT_GUARD=1` turns the same count into a refusal that names the op, the position and
//     the count.
//
// Needs one R9700 (gfx1201, 32 GB) or similar; runs under the GPU test guard. No container is read.
#include "r97/adjoint.h"

#include "support/device_check.h"
#include "support/gpu_guard.h"
#include "support/test.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace r97;
using device_check::numel;

// The quantiser's record at its smallest: a read activation, the codes it writes, and the group
// scale. `codes` and `scale` are handed in, not recomputed, so a case can put the scale where the
// quotient overflows and the codes where the mask does not zero the element.
struct Run {
    bool threw = false;
    std::string what;
    std::vector<double> dc0;
};

Run run_quant(double cot_value, double scale_first_group, double scale_rest) {
    const int64_t rows = 4, n = 256, group = 128, ngrp = n / group, total = rows * n;
    OpRecord rec;
    rec.position = 1449;
    rec.op = "had_quant_act_i8g";
    rec.params = {{"M", rows}, {"n", n}, {"group", group}, {"dtype", std::string("bf16")}};
    rec.operands = {Operand{"x", "bf16", {rows, n}, 0, rows, false},
                    Operand{"q", "i8", {rows, n}, 0, rows, true},
                    Operand{"scale", "f32", {rows, ngrp}, 0, rows, true}};

    uint64_t seed = 0x51ee70001ull;
    std::vector<AdValue> fwd;
    fwd.push_back(device_check::random_value("bf16", {rows, n}, &seed));
    AdValue codes;
    codes.dtype = "i8";
    codes.shape = {rows, n};
    // No code at the mask's top (|code| == 127), so no element is zeroed by the mask: the count
    // this asserts is then the division's own, not the mask's.
    codes.value.assign((size_t)total, 0.0);
    fwd.push_back(std::move(codes));
    AdValue scale;
    scale.dtype = "f32";
    scale.shape = {rows, ngrp};
    scale.value.resize((size_t)(rows * ngrp));
    for (int64_t i = 0; i < rows * ngrp; ++i)
        scale.value[(size_t)i] = (i % ngrp) == 0 ? scale_first_group : scale_rest;
    fwd.push_back(std::move(scale));

    std::vector<AdValue> cot;
    for (const Operand& o : rec.operands) {
        if (!o.written) continue;
        AdValue c;
        c.dtype = "f32";
        c.shape = o.shape;
        c.value.assign((size_t)numel(o.shape), cot_value);
        cot.push_back(std::move(c));
    }
    std::vector<AdValue> dc;
    for (const Operand& o : rec.operands) {
        AdValue v;
        v.dtype = o.dtype;
        v.shape = o.shape;
        v.value.assign((size_t)numel(o.shape), 0.0);
        dc.push_back(std::move(v));
    }

    AdjointCall call{rec, fwd, cot, &dc, nullptr, ""};
    AdjointFn fn = adjoint_find(rec.op, Side::device, rec.params, nullptr);
    if (!fn) throw std::runtime_error("no device adjoint for had_quant_act_i8g");
    Run r;
    try {
        fn(call);
    } catch (const std::exception& e) {
        r.threw = true;
        r.what = e.what();
        return r;
    }
    r.dc0 = dc.at(0).value;
    return r;
}

void check_guard() {
    // 1e38 over a 1e-3 scale is 1e41, past f32's 3.4e38: every live element overflows.
    // 1e30 is 1e33, well inside it.
    const double kOverflow = 1e38, kFinite = 1e25, kTiny = 1e-3, kHuge = 1e30;

    // The default: report the count and carry on. A run whose cotangents already overflow must
    // reach the measurement that says why, so this must not throw.
    {
        const Run r = run_quant(kOverflow, kTiny, kTiny);
        R97_CHECK(!r.threw);
        std::printf("quant guard, default: reported and carried on\n");
    }

    ::setenv("R97_QUANT_GUARD", "1", 1);

    // Every element live, every scale 1e-3: the whole record overflows, and the count says so.
    {
        const Run r = run_quant(kOverflow, kTiny, kTiny);
        R97_CHECK(r.threw);
        R97_CHECK(r.what.find("had_quant_act_i8g") != std::string::npos);
        R97_CHECK(r.what.find("1449") != std::string::npos);
        R97_CHECK(r.what.find("1024 of 1024") != std::string::npos);
        std::printf("quant guard, all groups: %s\n", r.what.c_str());
    }

    // Half the groups keep the quotient finite: the count is exact, so exactly half is refused.
    {
        const Run r = run_quant(kOverflow, kHuge, kTiny);
        R97_CHECK(r.threw);
        R97_CHECK(r.what.find("512 of 1024") != std::string::npos);
        std::printf("quant guard, one group's scale large: %s\n", r.what.c_str());
    }

    // The quotient inside f32: no refusal, and the gradient the row hands back is finite.
    {
        const Run r = run_quant(kFinite, kTiny, kTiny);
        R97_CHECK(!r.threw);
        R97_CHECK(!r.dc0.empty());
        for (double v : r.dc0) R97_CHECK(std::isfinite(v));
        std::printf("quant guard, quotient inside f32: no refusal, %lld finite values\n",
                    (long long)r.dc0.size());
    }

    ::unsetenv("R97_QUANT_GUARD");
}

void run() {
    const auto options = device_check::guard_options(int64_t(1) << 30);
    const r97test::DeviceGuard guard(options);
    device_check::require_device(options.device);
    (void)device_check::load_plugin();
    check_guard();
}

}  // namespace

R97_TEST(quant_guard) { run(); }
