// tests/host/adjoint_quant_cot_slot.cpp -- the fused producers find their parts' cotangents by
// role, never by position.
//
// `AdjointCall::cot` holds one entry per write, in the record's own write order, and the live tape
// does not order a fused record the way the schema does: `rmsnorm_had_quant_i8g` writes the normed
// value beside the codes, and which of the two comes first is the record's. A row that reads
// `cot[0]` for the codes therefore takes a slot no consumer fills on one of the two orders, and its
// quantiser part contributes nothing -- on the real container that zeroed the residual stream's
// cotangent from the first block boundary down, and the walk's gradient came back 10x too small
// (4.905 against 48.616, issue #145).
//
// The two orders below are the same arithmetic with the writes swapped. Both must hand the
// residual the same cotangent; the only difference is which `cot` index the codes' slot lands in.
#include "r97/adjoint.h"

#include "support/test.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace r97;

constexpr int64_t kRows = 4;
constexpr int64_t kN = 128;
constexpr int64_t kGroup = 128;
constexpr int64_t kNgrp = kN / kGroup;

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

AdValue values(const char* dtype, std::vector<int64_t> shape, double seed) {
    AdValue v;
    v.dtype = dtype;
    v.shape = std::move(shape);
    v.value.resize(static_cast<size_t>(numel(v.shape)));
    for (size_t i = 0; i < v.value.size(); ++i)
        v.value[i] = std::sin(seed * 0.7 + static_cast<double>(i) * 0.13);
    return v;
}

// One fused rmsnorm record. `out_first` is the only difference between the two cases: the normed
// value's write before the codes, or after them.
struct Case {
    OpRecord rec;
    std::vector<AdValue> fwd;
    std::vector<AdValue> cot;
    std::vector<AdValue> dc;
    int64_t codes_at = -1;
};

Case build(bool out_first) {
    Case c;
    c.rec.position = 1449;
    c.rec.op = "rmsnorm_had_quant_i8g";
    c.rec.params = {{"M", kRows},
                    {"n", kN},
                    {"group", kGroup},
                    {"dtype", std::string("bf16")},
                    {"eps", std::string("1e-6")},
                    {"wadd", std::string("0.0")}};
    c.rec.operands = {
        Operand{"h", "bf16", {kRows, kN}, 0, kRows, false},
        Operand{"x", "bf16", {kRows, kN}, 0, kRows, false},
        Operand{"w", "bf16", {kN}, 0, 1, false},
    };
    c.fwd = {values("bf16", {kRows, kN}, 1.0), values("bf16", {kRows, kN}, 2.0),
             values("bf16", {kN}, 3.0)};
    const Operand out{"h_out", "bf16", {kRows, kN}, 0, kRows, true};
    const Operand codes{"h.q8", "i8", {kRows, kN}, 0, kRows, true};
    const Operand scale{"h.s8", "f32", {kRows, kNgrp}, 0, kRows, true};
    if (out_first) {
        c.rec.operands.push_back(out);
        c.rec.operands.push_back(codes);
    } else {
        c.rec.operands.push_back(codes);
        c.rec.operands.push_back(out);
    }
    c.rec.operands.push_back(scale);
    c.fwd.push_back(values("bf16", {kRows, kN}, 4.0));
    AdValue q = values("i8", {kRows, kN}, 5.0);
    for (double& v : q.value) v = std::round(v * 40.0);
    c.fwd.push_back(std::move(q));
    AdValue s = values("f32", {kRows, kNgrp}, 6.0);
    for (double& v : s.value) v = 0.01 + std::fabs(v) * 0.001;
    c.fwd.push_back(std::move(s));

    // One cotangent per write, in the record's write order. Only the codes' is non-zero: the
    // elementwise output's slot is the one a walk leaves empty for a producer whose output no
    // consumer reads separately, and a row that took it for the codes would read zeros.
    for (const Operand& o : c.rec.operands) {
        if (!o.written) continue;
        if (o.buffer == "h.q8") c.codes_at = static_cast<int64_t>(c.cot.size());
        c.cot.push_back(values("f32", o.shape, o.buffer == "h.q8" ? 7.0 : 8.0));
        if (o.buffer != "h.q8")
            for (double& v : c.cot.back().value) v = 0.0;
    }
    for (const Operand& o : c.rec.operands) {
        AdValue v;
        v.dtype = o.dtype;
        v.shape = o.shape;
        v.value.assign(static_cast<size_t>(numel(o.shape)), 0.0);
        c.dc.push_back(std::move(v));
    }
    return c;
}

double sum_abs(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += std::fabs(x);
    return s;
}

void run_case(bool out_first, double* dx_sum) {
    Case c = build(out_first);
    AdjointFn fn = adjoint_find(c.rec.op, Side::host, c.rec.params, nullptr);
    R97_CHECK(fn != nullptr);
    AdjointCall call{c.rec, c.fwd, c.cot, &c.dc, nullptr, ""};
    fn(call);
    *dx_sum = sum_abs(c.dc.at(0).value);
    std::printf("  writes %s the codes: codes cot at index %lld, residual dx |sum| %.6g\n",
                out_first ? "out before" : "out after", (long long)c.codes_at, *dx_sum);
}

void run() {
    double before = 0.0, after = 0.0;
    std::printf("rmsnorm_had_quant_i8g: the codes' cotangent by role\n");
    run_case(true, &before);
    run_case(false, &after);
    // The producer's part reached the residual: a positional read of cot[0] leaves this zero in the
    // case where the output's write comes first.
    R97_CHECK(before > 0.0);
    R97_CHECK(after > 0.0);
    // The same arithmetic either way: the order of two writes is not part of the op.
    R97_CHECK(std::fabs(before - after) <= 1e-9 * std::max(1.0, std::fabs(before)));
}

}  // namespace

R97_TEST(adjoint_quant_cot_slot) { run(); }
