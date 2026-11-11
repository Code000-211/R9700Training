// tests/gpu/adjoint_elementwise.cpp -- the elementwise family's device adjoints against the host
// adjoints.
//
//   * mul, scale, sigmoid, silu, silu_mul, act_mul, cast and rmsnorm are each registered on the
//     card, and each agrees with src/adjoint/host/elementwise.cpp within the harness's bf16
//     tolerance at the tiny fixture's width (128, docs/FIXTURES.md) and at the real model's (hidden
//     5120 and the MLP's 17408, docs/DESIGN.md);
//   * 1024 tokens as one sequence and 4096 as a batch of unequal lengths are both covered, because
//     the ops that reduce -- mul's row bias, rmsnorm's row sums -- only see rows;
//   * silu_mul, the one member the forward fuses on the card, is held to the composition of mul's
//     and silu's host adjoints, in the order the fused pass undoes them, as
//     docs/kernels/README.md asks of a fused op -- and then the composition's own pair is checked
//     on the card as well, so both forms are known good;
//   * a deliberately wrong kernel is rejected rather than accepted as plausible.
//
// Needs one R9700 (gfx1201, 32 GB) or similar; runs under the GPU test guard. No container is read:
// the real shapes are numbers from the design, so this test does not carry the `model` label.
#include "support/device_check.h"
#include "support/gpu_guard.h"
#include "support/test.h"

#include <cstdint>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace r97;

// The shape every case is built from: the lengths of the sequences in the batch, and the row
// width. One length is one sequence; more than one is a batch whose rows sum to the same total.
struct Shape {
    std::vector<int64_t> lengths;
    int64_t width;
};

int64_t rows_of(const Shape& s) {
    return std::accumulate(s.lengths.begin(), s.lengths.end(), int64_t(0));
}

std::string name_of(const Shape& s) {
    std::string out;
    for (size_t i = 0; i < s.lengths.size(); ++i) {
        if (i) out += "+";
        out += std::to_string(s.lengths[i]);
    }
    out += " x " + std::to_string(s.width);
    if (s.lengths.size() > 1) out += " batch";
    return out;
}

Operand read_operand(const char* buffer, const char* dtype, std::vector<int64_t> shape) {
    return Operand{buffer, dtype, std::move(shape), 0, 0, false};
}

Operand written_operand(const char* buffer, const char* dtype, std::vector<int64_t> shape) {
    return Operand{buffer, dtype, std::move(shape), 0, 0, true};
}

/* -- one builder per schema, in the schema's operand order ------------------ */

device_check::Case unary_case(const char* op, const Shape& s, uint64_t seed) {
    device_check::Case c;
    const std::vector<int64_t> rows{rows_of(s), s.width};
    c.rec.op = op;
    c.rec.params = {{"M", rows[0]}, {"n", s.width}, {"dtype", std::string("bf16")}};
    c.rec.operands = {read_operand("x", "bf16", rows), written_operand("y", "bf16", rows)};
    c.in = {device_check::random_value("bf16", rows, &seed)};
    c.cot = {device_check::random_value("bf16", rows, &seed)};
    c.label = name_of(s);
    return c;
}

// b whole, or the one row every row of a multiplies -- the op's schema allows both.
device_check::Case mul_case(const Shape& s, bool one_row, uint64_t seed) {
    device_check::Case c;
    const std::vector<int64_t> rows{rows_of(s), s.width};
    const std::vector<int64_t> b_shape = one_row ? std::vector<int64_t>{s.width} : rows;
    c.rec.op = "mul";
    c.rec.params = {{"M", rows[0]}, {"n", s.width}, {"dtype", std::string("bf16")}};
    c.rec.operands = {read_operand("a", "bf16", rows), read_operand("b", "bf16", b_shape),
                      written_operand("y", "bf16", rows)};
    c.in = {device_check::random_value("bf16", rows, &seed),
            device_check::random_value("bf16", b_shape, &seed)};
    c.cot = {device_check::random_value("bf16", rows, &seed)};
    c.label = name_of(s) + (one_row ? " row b" : "");
    return c;
}

// s is the op's weight: a single factor, frozen for the whole block.
device_check::Case scale_case(const Shape& s, uint64_t seed) {
    device_check::Case c;
    const std::vector<int64_t> rows{rows_of(s), s.width};
    c.rec.op = "scale";
    c.rec.params = {{"M", rows[0]}, {"n", s.width}, {"dtype", std::string("bf16")}};
    c.rec.operands = {read_operand("x", "bf16", rows), read_operand("s", "bf16", {1}),
                      written_operand("y", "bf16", rows)};
    c.in = {device_check::random_value("bf16", rows, &seed),
            device_check::random_value("bf16", {1}, &seed)};
    c.cot = {device_check::random_value("bf16", rows, &seed)};
    c.label = name_of(s);
    return c;
}

// The gate in the first half of the fused row, the up projection in the second.
device_check::Case silu_mul_case(const Shape& s, uint64_t seed) {
    device_check::Case c;
    const std::vector<int64_t> rows{rows_of(s), s.width};
    const std::vector<int64_t> fused{rows_of(s), 2 * s.width};
    c.rec.op = "silu_mul";
    c.rec.params = {{"M", rows[0]}, {"n", s.width}, {"dtype", std::string("bf16")}};
    c.rec.operands = {read_operand("gate_up", "bf16", fused),
                      written_operand("y", "bf16", rows)};
    c.in = {device_check::random_value("bf16", fused, &seed)};
    c.cot = {device_check::random_value("bf16", rows, &seed)};
    c.label = name_of(s);
    return c;
}

device_check::Case act_mul_case(const Shape& s, const char* act, uint64_t seed) {
    device_check::Case c;
    const std::vector<int64_t> rows{rows_of(s), s.width};
    c.rec.op = "act_mul";
    c.rec.params = {
        {"M", rows[0]}, {"n", s.width}, {"act", std::string(act)}, {"dtype", std::string("bf16")},
    };
    c.rec.operands = {read_operand("a", "bf16", rows), read_operand("b", "bf16", rows),
                      written_operand("y", "bf16", rows)};
    c.in = {device_check::random_value("bf16", rows, &seed),
            device_check::random_value("bf16", rows, &seed)};
    c.cot = {device_check::random_value("bf16", rows, &seed)};
    c.label = name_of(s) + " " + act;
    return c;
}

// The cotangent of a cast is the *input's* dtype, so f32 -> bf16 hands one back as f32 and the
// bf16 -> f32 direction as bf16: the two cases are the two ways the store's dtype can go.
device_check::Case cast_case(const Shape& s, const char* from, const char* to, uint64_t seed) {
    device_check::Case c;
    const std::vector<int64_t> rows{rows_of(s), s.width};
    c.rec.op = "cast";
    c.rec.params = {
        {"M", rows[0]}, {"n", s.width}, {"from", std::string(from)}, {"to", std::string(to)},
    };
    c.rec.operands = {read_operand("x", from, rows), written_operand("y", to, rows)};
    c.in = {device_check::random_value(from, rows, &seed)};
    c.cot = {device_check::random_value(from, rows, &seed)};
    c.label = name_of(s) + " " + from + "->" + to;
    return c;
}

// Both norms the block carries read bf16 rows with an f32 gain shifted by wadd 1.0 (RAD_F32
// attn_norm and the q/k norms, wadd set to 1.0), so a zero-centred gain reads as GemmaRMSNorm's
// 1 + w; `mult` weights the normed row before its one rounding. An empty gain is c = 1.
device_check::Case rmsnorm_case(const Shape& s, bool gain, const char* wadd, const char* mult,
                                uint64_t seed) {
    device_check::Case c;
    const std::vector<int64_t> rows{rows_of(s), s.width};
    c.rec.op = "rmsnorm";
    c.rec.params = {
        {"M", rows[0]},   {"n", s.width}, {"eps", std::string("1e-6")},
        {"dtype", std::string("bf16")}, {"wadd", std::string(wadd)},
        {"mult", std::string(mult)},
    };
    c.rec.operands = {read_operand("x", "bf16", rows), read_operand("w", "f32", {s.width}),
                      written_operand("y", "bf16", rows)};
    c.in = {device_check::random_value("bf16", rows, &seed),
            device_check::random_value("f32", {s.width}, &seed)};
    if (!gain) c.in[1].value.clear();  // the tape's way of saying there is no gain
    c.cot = {device_check::random_value("bf16", rows, &seed)};
    c.label = name_of(s) + (gain ? " gain" : " no gain") + " wadd " + wadd + " mult " + mult;
    return c;
}

/* -- the reference the fused activation is held to ------------------------- */

// The composition silu_mul's kernel undoes in one pass, run as the host rows it is made of: mul
// first -- dy splits into the activation's cotangent and the multiplier's -- then silu, from that
// cotangent to the gate's. Both forwards it needs (a = silu(g)) come from the host twin.
device_check::HostRef silu_mul_composition(int64_t n) {
    return [n](AdjointCall& call) {
        const AdjointFn mul_adjoint = adjoint_find("mul", Side::host, {}, nullptr);
        const AdjointFn silu_adjoint = adjoint_find("silu", Side::host, {}, nullptr);
        const TwinFn silu_twin = twin_find("silu", {}, nullptr);
        if (!mul_adjoint || !silu_adjoint || !silu_twin)
            throw std::runtime_error("silu_mul composition: a host row is missing");

        const AdValue& gate_up = call.fwd[0];
        const AdValue& y_cot = call.cot[0];
        const int64_t rows = (int64_t)y_cot.value.size() / n;
        const int64_t total = rows * n;

        AdValue g{"bf16", {rows, n}, std::vector<double>((size_t)total)};
        AdValue u{"bf16", {rows, n}, std::vector<double>((size_t)total)};
        for (int64_t r = 0; r < rows; ++r) {
            const int64_t xb = r * 2 * n, yb = r * n;
            for (int64_t i = 0; i < n; ++i) {
                g.value[(size_t)(yb + i)] = gate_up.value[(size_t)(xb + i)];
                u.value[(size_t)(yb + i)] = gate_up.value[(size_t)(xb + n + i)];
            }
        }

        OpRecord silu_rec;
        silu_rec.op = "silu";
        silu_rec.operands = {read_operand("x", "bf16", {rows, n}),
                             written_operand("y", "bf16", {rows, n})};
        std::vector<AdValue> act;
        silu_twin(silu_rec, {g}, &act);

        OpRecord mul_rec;
        mul_rec.op = "mul";
        mul_rec.operands = {read_operand("a", "bf16", {rows, n}),
                            read_operand("b", "bf16", {rows, n}),
                            written_operand("y", "bf16", {rows, n})};
        std::vector<AdValue> mul_dc(3);
        for (AdValue& v : mul_dc) {
            v.dtype = "bf16";
            v.shape = {rows, n};
            v.value.assign((size_t)total, 0.0);
        }
        const std::vector<AdValue> mul_fwd = {act[0], u};
        const std::vector<AdValue> mul_cot = {y_cot};
        AdjointCall mul_call{mul_rec, mul_fwd, mul_cot, &mul_dc, nullptr, ""};
        mul_adjoint(mul_call);

        std::vector<AdValue> silu_dc(1);
        silu_dc[0].dtype = "bf16";
        silu_dc[0].shape = {rows, n};
        silu_dc[0].value.assign((size_t)total, 0.0);
        const std::vector<AdValue> silu_fwd = {g};
        const std::vector<AdValue> silu_cot = {mul_dc[0]};
        AdjointCall silu_call{silu_rec, silu_fwd, silu_cot, &silu_dc, nullptr, ""};
        silu_adjoint(silu_call);

        AdValue& up_dc = call.dc->at(0);
        for (int64_t r = 0; r < rows; ++r) {
            const int64_t xb = r * 2 * n, yb = r * n;
            for (int64_t i = 0; i < n; ++i) {
                up_dc.value[(size_t)(xb + i)] += silu_dc[0].value[(size_t)(yb + i)];
                up_dc.value[(size_t)(xb + n + i)] += mul_dc[1].value[(size_t)(yb + i)];
            }
        }
    };
}

// Deliberately wrong: carries the cotangent into the gate half untouched and drops the up half.
// The harness must reject it, in the fused row's slot.
void wrong_silu_mul_adjoint(const AdjointCall& call) {
    const int64_t n = (int64_t)call.fwd[0].shape.back() / 2;
    const int64_t rows = (int64_t)call.cot[0].value.size() / n;
    AdValue& up_dc = call.dc->at(0);
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = r * 2 * n, yb = r * n;
        for (int64_t i = 0; i < n; ++i) up_dc.value[(size_t)(xb + i)] += call.cot[0].value[(size_t)(yb + i)];
    }
}

/* -- the run --------------------------------------------------------------- */

// A case and how many times --bench repeats it: the real-width cases are hundred-million-element
// calls, and a measured number needs no more reps than that to be steady.
struct Timed {
    device_check::Case c;
    int reps;
};

void require_registered() {
    for (const char* op : {"mul", "scale", "sigmoid", "silu", "silu_mul", "act_mul", "cast",
                           "rmsnorm"}) {
        const OpRows rows = op_rows(op);
        if (!rows.device_adjoint) throw std::runtime_error(std::string("no device adjoint for '") + op + "'");
    }
}

const char* const kActivations[] = {"silu", "gelu", "gelu_tanh", "sigmoid", "quick_gelu", "relu"};

std::vector<Timed> fixture_cases() {
    const Shape s{{4}, 128};
    std::vector<Timed> out = {
        {mul_case(s, false, 0xe1e00001ull), 20},
        {mul_case(s, true, 0xe1e00002ull), 20},
        {scale_case(s, 0xe1e00003ull), 20},
        {unary_case("sigmoid", s, 0xe1e00004ull), 20},
        {unary_case("silu", s, 0xe1e00005ull), 20},
        {silu_mul_case(Shape{{4}, 64}, 0xe1e00006ull), 20},
        {cast_case(s, "bf16", "f32", 0xe1e00007ull), 20},
        {cast_case(s, "f32", "bf16", 0xe1e00008ull), 20},
        {rmsnorm_case(s, true, "1.0", "1.0", 0xe1e00009ull), 20},
        {rmsnorm_case(s, true, "0.0", "0.5", 0xe1e0000aull), 20},
        {rmsnorm_case(s, false, "0.0", "1.0", 0xe1e0000bull), 20},
    };
    for (const char* act : kActivations) out.push_back({act_mul_case(s, act, 0xe1e00010ull), 20});
    return out;
}

// The real model: hidden 5120, the MLP's intermediate 17408 (whose gated activation is a fused
// [rows, 2n] row of 34816), 1024 tokens as one sequence, and 4096 as a batch of unequal lengths.
std::vector<Timed> real_cases() {
    const Shape one{{1024}, 5120};
    const Shape batch{{1024, 512, 2560}, 5120};
    const Shape wide_one{{1024}, 17408};
    const Shape wide_batch{{1024, 512, 2560}, 17408};
    return {
        {mul_case(one, false, 0xe1e10001ull), 5},
        {mul_case(one, true, 0xe1e10002ull), 5},
        {scale_case(one, 0xe1e10003ull), 5},
        {unary_case("sigmoid", one, 0xe1e10004ull), 5},
        {unary_case("silu", one, 0xe1e10005ull), 5},
        {cast_case(one, "bf16", "f32", 0xe1e10006ull), 5},
        {rmsnorm_case(one, true, "1.0", "1.0", 0xe1e10007ull), 5},
        {act_mul_case(one, "silu", 0xe1e10008ull), 5},
        {act_mul_case(one, "gelu_tanh", 0xe1e10009ull), 5},
        {silu_mul_case(Shape{{1024}, 17408}, 0xe1e1000aull), 5},
        {mul_case(wide_one, true, 0xe1e1000bull), 5},
        {act_mul_case(wide_one, "silu", 0xe1e1000cull), 5},
        // 4096 rows, as a batch of unequal lengths, at both widths.
        {mul_case(batch, false, 0xe1e1000dull), 5},
        {mul_case(batch, true, 0xe1e1000eull), 2},
        {unary_case("silu", batch, 0xe1e1000full), 5},
        {rmsnorm_case(batch, true, "1.0", "1.0", 0xe1e10010ull), 2},
        {silu_mul_case(wide_batch, 0xe1e10011ull), 2},
    };
}

void run() {
    const auto options = device_check::guard_options(int64_t(8) << 30);
    const r97test::DeviceGuard guard(options);
    device_check::require_device(options.device);
    (void)device_check::load_plugin();

    require_registered();

    std::vector<Timed> cases = fixture_cases();
    for (Timed& t : real_cases()) cases.push_back(std::move(t));

    for (const Timed& t : cases) {
        const device_check::Report r = device_check::check(t.c);
        device_check::require(r);
        R97_CHECK(r.scratch_peak > 0);  // the kernel took its device buffers from the allocator
        std::printf("%s %s: device == host, scratch peak %lld KiB\n", t.c.rec.op.c_str(),
                    t.c.label.c_str(), (long long)(r.scratch_peak / 1024));
        if (r97test::bench()) device_check::print_bench(t.c, t.reps);
    }

    // The fused gated activation against the composition it stands for, at the fixture's width and
    // at the real one. At 256 rows the composition's f64 buffers stay small; the fused kernel's own
    // shape at 4096 tokens is covered above, against the host row.
    const std::vector<device_check::Case> fused = {
        silu_mul_case(Shape{{4}, 64}, 0xe1e20001ull),
        silu_mul_case(Shape{{256}, 17408}, 0xe1e20002ull),
    };
    for (const device_check::Case& c : fused) {
        const int64_t n = c.rec.operands[0].shape.back() / 2;
        const device_check::Report r = device_check::check(c, nullptr, silu_mul_composition(n));
        device_check::require(r);
        std::printf("silu_mul %s: fused == mul then silu, %lld mismatches\n", c.label.c_str(),
                    (long long)r.mismatches);
        if (r97test::bench()) device_check::print_bench(c, 5);
    }

    // A kernel that keeps only the gate half must be rejected, in the fused row's slot.
    const device_check::Report wrong = device_check::check(fused.front(), wrong_silu_mul_adjoint);
    R97_CHECK(!wrong.ok);
    R97_CHECK(wrong.first_slot == 0);
    std::printf("rejected the wrong kernel: %s\n", wrong.note.c_str());
}

}  // namespace

R97_TEST(adjoint_elementwise_device) { run(); }
