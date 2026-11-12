// tests/host/adjoint_quant.cpp -- the quant family (src/adjoint/host/quant.cpp)
// through both checks of tests/support/adjoint_check.h: each op's f64 twin
// against radiance's reference kernel, and its host adjoint against finite
// differences of the twin.
//
// Rounding is a step function, so the finite differences are taken on the twin
// linearised at a base point (`<op>_ste`), which is the definition the
// straight-through estimator states: the rounding's residual goes straight
// through, the clamp's does not. A second test builds a group whose largest
// value the clamp takes and reads the mask off the adjoint itself; a third
// reads the transpose of the Hadamard rotation off a group the op does not
// touch. docs/adjoints/quant.md has the formulas.
#include "support/adjoint_check.h"
#include "support/test.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace r97;
using namespace r97::adjoint_check;

namespace {

/* bf16 carries 8 significand bits, so a kernel that rounds as it goes lands
 * within about 2^-9 relative of the f64 twin; 0.02 leaves the sum's rounding
 * and the bf16 store room without hiding an op that computes something else. */
const double bf16_tol = 0.02;

/* The twin is exact in f64 and its linearisation in the input, so central
 * differences reproduce the adjoint far inside 1e-6 relative. */
const double fd_tol = 1e-6;

const double top = 127.0; /* the top of the int8 code grid */

/* One call of the family: `rows` rows of `n` columns, one scale per `group`. */
struct Shape {
    const char* name;
    int64_t rows, n, group;
};

/* The group width and the groups per row, read the way the op reads them: a
 * group that is absent or wider than the row is one scale per row. */
struct Geom {
    int64_t n, group, ngrp;
};

Geom geom_of(bool rotated, const Shape& s) {
    Geom g;
    g.n = s.n;
    g.group = (s.group <= 0 || s.group > s.n) ? s.n : s.group;
    if (rotated) g.group = s.group; /* the rotated op requires group | n */
    if (g.group <= 0) g.group = 1;   /* a record the family refuses still needs a shape */
    g.ngrp = rotated ? s.n / g.group : (s.n + g.group - 1) / g.group;
    if (g.ngrp <= 0) g.ngrp = 1;
    return g;
}

std::vector<int64_t> dims(const Shape& s) { return {s.rows, s.n}; }

/* A record of the family. `linearised` asks for the twin the finite differences
 * run on: it reads the base point the estimator is taken at as a second
 * operand, and that operand is the one its adjoint row marks frozen. */
OpRecord quant_rec(bool rotated, const Shape& s, bool linearised) {
    const Geom g = geom_of(rotated, s);
    OpRecord rec;
    rec.op = rotated ? "had_quant_act_i8g" : "quant_act_i8g";
    if (linearised) rec.op += "_ste";
    rec.params = {{"M", s.rows}, {"n", s.n}, {"group", s.group}, {"dtype", "bf16"}};
    rec.operands = {{"x", "bf16", dims(s), 0, s.rows, false}};
    if (linearised) rec.operands.push_back({"x0", "bf16", dims(s), 0, s.rows, false});
    rec.operands.push_back({"codes", "i8", dims(s), 0, s.rows, true});
    rec.operands.push_back({"scale", "f32", {s.rows, g.ngrp}, 0, s.rows, true});
    return rec;
}

AdValue values(const char* dtype, std::vector<int64_t> shape, std::vector<double> v) {
    AdValue a;
    a.dtype = dtype;
    a.shape = std::move(shape);
    a.value = std::move(v);
    return a;
}

/* The written operands' dtypes and shapes, which is all the checks want of
 * them: the twin supplies the values. */
std::vector<AdValue> written_of(const Shape& s, const Geom& g) {
    std::vector<AdValue> out;
    out.push_back(values("i8", dims(s), std::vector<double>((size_t)(s.rows * s.n), 0.0)));
    out.push_back(values("f32", {s.rows, g.ngrp},
                         std::vector<double>((size_t)(s.rows * g.ngrp), 0.0)));
    return out;
}

/* The forward and the adjoint are asked the way the walk asks: through the
 * registry, by the record's op name and parameters. */
std::vector<AdValue> forward(const OpRecord& rec, const std::vector<AdValue>& in) {
    std::string err;
    TwinFn twin = twin_find(rec.op, rec.params, &err);
    if (twin == nullptr) fail(err);
    std::vector<AdValue> out;
    twin(rec, in, &out);
    return out;
}

/* One adjoint call over the whole schema: the reads, the twin's writes, the
 * cotangent of the codes and one value for the cotangent of every scale. The
 * adjoint's answer for the activation comes back. */
std::vector<double> backward(const OpRecord& rec, const std::vector<AdValue>& in,
                             std::vector<double> cot_codes, double cot_scale) {
    std::string err;
    AdjointFn adj = adjoint_find(rec.op, Side::host, rec.params, &err);
    if (adj == nullptr) fail(err);
    const std::vector<AdValue> out = forward(rec, in);

    std::vector<AdValue> fwd(rec.operands.size());
    std::vector<AdValue> dc(rec.operands.size());
    size_t r = 0, w = 0;
    for (size_t i = 0; i < rec.operands.size(); ++i) {
        if (rec.operands[i].written) {
            fwd[i] = out[w++];
        } else {
            fwd[i] = in[r];
            dc[i] = in[r];
            dc[i].value.assign(dc[i].value.size(), 0.0);
            ++r;
        }
    }
    std::vector<AdValue> cot;
    for (size_t i = 0; i < rec.operands.size(); ++i) {
        if (!rec.operands[i].written) continue;
        cot.push_back(fwd[i]);
        cot.back().value.assign(fwd[i].value.size(), cot_scale);
    }
    if (cot.size() != 2 || cot_codes.size() != cot[0].value.size())
        throw std::runtime_error("quant: the record writes the codes and the scale");
    cot[0].value = std::move(cot_codes);

    AdjointCall call{rec, fwd, cot, &dc, nullptr, ""};
    adj(call);
    return dc[0].value;
}

/* -- the data -------------------------------------------------------------- */

/* A bf16 value, the way the container stores one: an f32 with its low
 * significand bits cleared, so the twin and the kernel see the same number. */
double bf16(double x) {
    const float f = (float)x;
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof bits);
    bits &= 0xffff0000u;
    float r = 0.0f;
    std::memcpy(&r, &bits, sizeof r);
    return (double)r;
}

/* One bf16 ulp further from zero: still bf16, and far enough from where it was
 * to leave its bucket, which is what the no-tie nudge below needs. */
double bf16_ulp(double x) {
    const float f = (float)x;
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof bits);
    bits += 0x00010000u;
    float r = 0.0f;
    std::memcpy(&r, &bits, sizeof r);
    return (double)r;
}

/* An activation in [-1, 1), bf16 as the op's operand is. */
std::vector<double> activation(const Shape& s, uint64_t seed) {
    std::vector<double> x((size_t)(s.rows * s.n));
    for (double& v : x) v = bf16(urand(&seed));
    return x;
}

/* The values the quantiser sees, v = W x, and the scale of every group, asked
 * of the op itself. The linearised twin at the base point zero has c0 = 0,
 * s0 = 1 and nothing clamped, so its codes output is v exactly; the scale is
 * the forward's. The test therefore holds no second copy of W, and the scale it
 * reads is the one the op wrote. */
void quantiser_view(bool rotated, const Shape& s, const std::vector<double>& x,
                    std::vector<double>* v, std::vector<double>* scale) {
    const std::vector<double> zero(x.size(), 0.0);
    *v = forward(quant_rec(rotated, s, true),
                 {values("bf16", dims(s), x), values("bf16", dims(s), zero)})[0]
             .value;
    *scale = forward(quant_rec(rotated, s, false), {values("bf16", dims(s), x)})[1].value;
}

/* Is |u| close enough to a half-integer that the f64 twin and the f32 kernel
 * could round it to opposite sides? Both compute u to within a few ulps of
 * f32, so a value further than that from every boundary rounds the same way in
 * both. */
bool at_a_tie(double u) {
    const double a = std::fabs(u);
    return std::fabs(a - (std::floor(a) + 0.5)) <= 32.0 * 0x1p-23 * std::max(1.0, a);
}

/* An activation with no rounding tie in any group: the values are nudged one
 * bf16 ulp at a time until none sits at a boundary. A tie is arithmetic, not a
 * claim about the op -- the twin and the kernel round the last bit of the
 * reciprocal to opposite sides of it, one code apart -- so the kernel check runs
 * where it is exact and docs/adjoints/quant.md says where it is not. */
std::vector<double> without_ties(bool rotated, const Shape& s, std::vector<double> x) {
    const Geom g = geom_of(rotated, s);
    for (int pass = 0; pass < 64; ++pass) {
        std::vector<double> v, scale;
        quantiser_view(rotated, s, x, &v, &scale);
        bool any = false;
        for (int64_t r = 0; r < s.rows; ++r) {
            for (int64_t j = 0; j < s.n; ++j) {
                const size_t i = (size_t)(r * s.n + j);
                const double sc = scale[(size_t)(r * g.ngrp + j / g.group)];
                if (at_a_tie(v[i] * (1.0 / sc))) {
                    x[i] = bf16_ulp(x[i]);
                    any = true;
                }
            }
        }
        if (!any) return x;
    }
    throw std::runtime_error("quant: the activation would not leave every rounding tie");
}

/* The largest value of a group, chosen so that the reciprocal of its scale
 * lands just above 127: the clamp then takes the value that set the scale.
 * Whether it does is arithmetic, so the value is searched for, not assumed.
 * Every candidate is bf16-exact, and so are 0.75, 0.5, 0.25 and 0.125 of it,
 * which is what the constructed group below is made of. */
double scanned_amax() {
    for (int k = 0; k < 64; ++k) {
        const double a = 1.0 + k / 64.0; /* two mantissa bits: 0.75a is bf16 too */
        const double s = a * (1.0 / top);
        if (std::fabs(a * (1.0 / s)) > top) return a;
    }
    throw std::runtime_error("quant: found no group whose own scale clamps it");
}

/* -- (1) the twin against radiance's host kernel --------------------------- */

void kernel_check(bool rotated, const Shape& s) {
    const Geom g = geom_of(rotated, s);
    const OpRecord rec = quant_rec(rotated, s, false);
    const std::vector<double> x = without_ties(rotated, s, activation(s, 0x5eed));
    const KernelReport k =
        check_kernel(rec, {values("bf16", dims(s), x)}, written_of(s, g), bf16_tol);
    std::printf("%-20s %-28s kernel max err %.6g\n", rec.op.c_str(), s.name, k.max_err);
    require(k);
}

R97_TEST(adjoint_quant_twin_matches_the_host_kernel) {
    const Shape plain[] = {
        {"3 x 64, groups of 32", 3, 64, 32},
        {"2 x 48, groups of 32", 2, 48, 32}, /* a short last group: 32 + 16 */
        {"1 x 48, groups of 16", 1, 48, 16}, /* three whole groups, n not a multiple of 32 */
        {"8 x 128, groups of 128", 8, 128, 128},
        {"4 x 128, one scale per row", 4, 128, 0},
    };
    const Shape had[] = {
        {"2 x 96, groups of 32", 2, 96, 32},
        {"2 x 256, groups of 64", 2, 256, 64},
        {"3 x 128, groups of 4", 3, 128, 4},
        {"5 x 128, groups of 128", 5, 128, 128},
    };
    for (const Shape& s : plain) kernel_check(false, s);
    for (const Shape& s : had) kernel_check(true, s);
}

/* -- (2) the adjoint against finite differences of the twin ---------------- */

void fd_check(bool rotated, const Shape& s) {
    const Geom g = geom_of(rotated, s);
    const OpRecord ste = quant_rec(rotated, s, true);
    const std::vector<double> x = activation(s, 0x5eed);
    const std::vector<AdValue> in = {values("bf16", dims(s), x), values("bf16", dims(s), x)};

    /* The linearisation at its own base point is the forward: the same codes
     * and the same scale, to the last bit. */
    const std::vector<AdValue> fwd = forward(quant_rec(rotated, s, false), {in[0]});
    const std::vector<AdValue> lin = forward(ste, in);
    R97_CHECK(fwd.size() == lin.size());
    for (size_t o = 0; o < fwd.size(); ++o) {
        R97_CHECK(fwd[o].value.size() == lin[o].value.size());
        for (size_t e = 0; e < fwd[o].value.size(); ++e) R97_CHECK(fwd[o].value[e] == lin[o].value[e]);
    }

    /* How much of the grid this data saturates: the mask is what the adjoint
     * does with it, and data of this kind has saturated groups in it. */
    std::vector<double> v, scale;
    quantiser_view(rotated, s, x, &v, &scale);
    int64_t clamped = 0;
    for (int64_t r = 0; r < s.rows; ++r) {
        for (int64_t j = 0; j < s.n; ++j) {
            const size_t i = (size_t)(r * s.n + j);
            const double sc = scale[(size_t)(r * g.ngrp + j / g.group)];
            if (std::fabs(v[i] * (1.0 / sc)) > top) ++clamped;
        }
    }

    const FdReport fd = check_adjoint(ste, in, written_of(s, g), fd_tol);
    std::printf("%-20s %-28s %lld of %lld codes clamped, adjoint rel err %.3g\n",
                ste.op.c_str(), s.name, (long long)clamped, (long long)(s.rows * s.n), fd.rel);
    require(fd);
}

R97_TEST(adjoint_quant_adjoint_matches_finite_differences) {
    const Shape plain[] = {
        {"3 x 64, groups of 32", 3, 64, 32},
        {"1 x 48, groups of 16", 1, 48, 16},
        {"5 x 128, groups of 128", 5, 128, 128},
    };
    const Shape had[] = {
        {"2 x 96, groups of 32", 2, 96, 32},
        {"2 x 256, groups of 64", 2, 256, 64},
        {"5 x 128, groups of 128", 5, 128, 128},
    };
    for (const Shape& s : plain) fd_check(false, s);
    for (const Shape& s : had) fd_check(true, s);
}

/* -- (3) the mask, on a group the clamp takes ------------------------------ */

/* The group v = (a, a/2, -a/4, a/8) with a the scanned amax: the first value is
 * rounded and then clamped from a reciprocal just above 127, the rest are inside
 * the grid. The cotangent of the clamped code must not reach the activation; the
 * cotangent of an inside code must, divided by the scale and rotated back. */
void mask_check(bool rotated) {
    const Shape s = rotated ? Shape{"1 x 2, one group", 1, 2, 2}
                            : Shape{"1 x 4, one group", 1, 4, 4};
    const Geom g = geom_of(rotated, s);
    const OpRecord rec = quant_rec(rotated, s, false);
    const double a = scanned_amax();
    const double scale = a * (1.0 / top);
    /* W (0.75a, 0.25a) = (a, a/2): the rotated group holds the same pattern,
     * and 0.75 and 0.25 keep the activation bf16-exact. */
    const std::vector<double> x =
        rotated ? std::vector<double>{0.75 * a, 0.25 * a}
                : std::vector<double>{a, 0.5 * a, -0.25 * a, 0.125 * a};
    const std::vector<AdValue> in = {values("bf16", dims(s), x)};

    /* What the op wrote: the clamped code, and the scale that clamped it. */
    const std::vector<AdValue> fwd = forward(rec, in);
    R97_CHECK(fwd[0].value[0] == top);
    R97_CHECK(fwd[1].value[0] == scale);

    const size_t cols = (size_t)(s.rows * s.n);
    std::vector<double> cot(cols, 0.0);
    cot[0] = 1.0;
    std::vector<double> dx = backward(rec, in, cot, 7.5); /* a scale cotangent, dropped */
    for (double d : dx) R97_CHECK(d == 0.0);

    cot.assign(cols, 0.0);
    cot[1] = 1.0;
    dx = backward(rec, in, cot, 7.5);
    if (rotated) {
        R97_CHECK(dx[0] == 1.0 / scale);
        R97_CHECK(dx[1] == -1.0 / scale);
    } else {
        R97_CHECK(dx[0] == 0.0);
        R97_CHECK(dx[1] == 1.0 / scale);
        R97_CHECK(dx[2] == 0.0);
        R97_CHECK(dx[3] == 0.0);
    }

    /* And the same saturated base point through the finite differences. */
    const FdReport fd = check_adjoint(quant_rec(rotated, s, true), {in[0], in[0]},
                                      written_of(s, g), fd_tol);
    std::printf("%-20s saturated group of %lld: adjoint rel err %.3g\n", rec.op.c_str(),
                (long long)s.n, fd.rel);
    require(fd);
}

R97_TEST(adjoint_quant_masks_saturated_codes) {
    mask_check(false);
    mask_check(true);
}

/* -- (4) the transpose of the rotation, on a group the op does not touch ---- */

R97_TEST(adjoint_quant_zero_group_passes_through) {
    for (bool rotated : {false, true}) {
        const Shape s = rotated ? Shape{"2 x 8, groups of 8", 2, 8, 8}
                                : Shape{"2 x 6, groups of 3", 2, 6, 3};
        const Geom g = geom_of(rotated, s);
        const OpRecord rec = quant_rec(rotated, s, false);
        const std::vector<double> x((size_t)(s.rows * s.n), 0.0);
        const std::vector<AdValue> in = {values("bf16", dims(s), x)};

        /* The zero group has no direction to scale: codes zero, one scale. */
        const std::vector<AdValue> fwd = forward(rec, in);
        for (double c : fwd[0].value) R97_CHECK(c == 0.0);
        for (double sc : fwd[1].value) R97_CHECK(sc == 1.0);

        /* With the scale one and nothing clamped, the family is its rotation
         * alone. A cotangent at the first code of every group is e0 of each
         * group: the plain op hands it back, and the Hadamard op spreads it,
         * W e0 being all ones. */
        const size_t cols = (size_t)(s.rows * s.n);
        std::vector<double> first(cols, 0.0);
        for (int64_t r = 0; r < s.rows; ++r)
            for (int64_t j = 0; j < s.n; j += g.group) first[(size_t)(r * s.n + j)] = 1.0;
        const std::vector<double> dx = backward(rec, in, first, 3.0);
        for (int64_t r = 0; r < s.rows; ++r) {
            for (int64_t j = 0; j < s.n; ++j) {
                const double want = rotated || j % g.group == 0 ? 1.0 : 0.0;
                R97_CHECK(dx[(size_t)(r * s.n + j)] == want);
            }
        }

        /* One cotangent at every code: the plain op hands it back, and the
         * Hadamard op comes back as the group width at the first code and
         * nothing elsewhere, W (1, ..., 1) = (t, 0, ..., 0), so the adjoint
         * really is the transpose of the same transform. */
        const std::vector<double> all(cols, 1.0);
        const std::vector<double> dy = backward(rec, in, all, 3.0);
        for (int64_t r = 0; r < s.rows; ++r) {
            for (int64_t j = 0; j < s.n; ++j) {
                const double want =
                    rotated ? (j % g.group == 0 ? (double)g.group : 0.0) : 1.0;
                R97_CHECK(dy[(size_t)(r * s.n + j)] == want);
            }
        }
        std::printf("%-20s zero group of %lld: pass-through and transpose hold\n", rec.op.c_str(),
                    (long long)g.group);
    }
}

/* -- (5) the registry, and the records the family refuses ------------------- */

void refuses(const OpRecord& rec, const std::vector<AdValue>& in, const char* what) {
    bool threw = false;
    try {
        forward(rec, in);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    if (!threw) fail(std::string("quant: the twin accepted a record that ") + what);
}

R97_TEST(adjoint_quant_serves_both_ops) {
    for (const char* op : {"quant_act_i8g", "had_quant_act_i8g"}) {
        const OpRows rows = op_rows(op);
        R97_CHECK(rows.twin);
        R97_CHECK(rows.host_adjoint);
        R97_CHECK(rows.host_frozen.empty()); /* the scale is written, not read frozen */
    }
    /* The linearised twin's base point is its second read, and frozen. */
    const std::vector<OpParam> params = {{"n", 64}, {"group", 32}};
    R97_CHECK(adjoint_frozen("quant_act_i8g_ste", Side::host, params) == std::vector<int>{1});
    R97_CHECK(adjoint_frozen("had_quant_act_i8g_ste", Side::host, params) == std::vector<int>{1});
    /* The fused producers append their base point after the activation's write. */
    R97_CHECK(adjoint_frozen("rmsnorm_quant_fp8_ste", Side::host, params) == std::vector<int>{6});
    R97_CHECK(adjoint_frozen("gated_quant_fp8_ste", Side::host, params) == std::vector<int>{4});
    for (const char* op : {"rmsnorm_quant_fp8", "gated_quant_fp8", "rmsnorm_quant_fp8_ste",
                           "gated_quant_fp8_ste"}) {
        const OpRows rows = op_rows(op);
        R97_CHECK(rows.twin);
        R97_CHECK(rows.host_adjoint);
    }

    /* A record no kernel of the family takes is refused by name, not silently
     * computed. */
    const std::vector<double> x((size_t)96, 0.5);
    refuses(quant_rec(true, {"1 x 96, groups of 48", 1, 96, 48}, false),
            {values("bf16", {1, 96}, x)}, "has a group that is not a power of two");
    refuses(quant_rec(true, {"1 x 96, groups of 64", 1, 96, 64}, false),
            {values("bf16", {1, 96}, x)}, "has a group that does not divide the row");
    refuses(quant_rec(false, {"1 x 0", 1, 0, 0}, false), {values("bf16", {1, 0}, {})},
            "has no columns");

    OpRecord missing = quant_rec(false, {"1 x 4, one group", 1, 4, 4}, false);
    missing.params = {{"M", 1}, {"group", 4}};
    refuses(missing, {values("bf16", {1, 4}, {1.0, 0.5, 0.25, 0.125})}, "has no 'n'");
}

/* -- (6) the int8 trunk's fused producers ---------------------------------- */

/* The trunk replaces a residual add, a norm and an E4M3 quantiser with one
 * record, `rmsnorm_quant_fp8`, and a gate, its product and the same quantiser
 * with `gated_quant_fp8`. Both write the bf16 activation beside the codes, so
 * their adjoints have a path the unfused quantisers do not have: the bf16
 * write's own Jacobian. The record also writes its activation before the
 * quantiser reads it, so the linearised twin appends its base point after the
 * activation's write -- a schema position the unfused record does not use --
 * and the adjoint keeps the positions it has in the trunk. */
struct FusedShape {
    const char* name;
    int64_t rows, n, group;
};

std::vector<FusedShape> fused_shapes() {
    /* The trunk's shape (128 columns, one scale per row) and two groups of 32,
     * so the group loop and a short last group are both taken. */
    return {{"trunk-128", 5, 128, 128}, {"two-groups", 3, 64, 32}};
}

int64_t fused_groups(const FusedShape& s) {
    const int64_t g = (s.group <= 0 || s.group > s.n) ? s.n : s.group;
    return (s.n + g - 1) / g;
}

bool is_rmsnorm(const char* op) { return std::strcmp(op, "rmsnorm_quant_fp8") == 0; }

/* The record the trunk gives the op. `rmsnorm_quant_fp8` reads the activation,
 * the residual it adds in place and the gain, and writes the residual, the
 * codes, the scale and the bf16 activation; `gated_quant_fp8` reads the gated
 * projection, whose two halves are the gate and the up-projection, and writes
 * the codes, the scale and the product. */
OpRecord fused_rec(const char* op, const FusedShape& s, bool linearised) {
    const int64_t ngrp = fused_groups(s);
    OpRecord rec;
    rec.op = op;
    if (linearised) rec.op += "_ste";
    rec.n = s.rows;
    if (is_rmsnorm(op)) {
        rec.params = {{"M", s.rows},   {"n", s.n},     {"eps", "1e-6"},
                      {"group", s.group}, {"dtype", "bf16"}, {"wadd", int64_t{0}}};
        rec.operands = {{"x", "bf16", {s.rows, s.n}, 0, s.rows, false},
                        {"residual", "bf16", {s.rows, s.n}, 0, s.rows, true, true},
                        {"w", "f32", {s.n}, 0, 1, false},
                        {"codes", "fp8_e4m3", {s.rows, s.n}, 0, s.rows, true},
                        {"scale", "f32", {s.rows, ngrp}, 0, s.rows, true},
                        {"y", "bf16", {s.rows, s.n}, 0, s.rows, true}};
    } else {
        rec.params = {{"M", s.rows},   {"n", s.n},        {"group", s.group},
                      {"dtype", "bf16"}, {"act", "silu"}};
        rec.operands = {{"gate_up", "bf16", {s.rows, 2 * s.n}, 0, s.rows, false},
                        {"q", "fp8_e4m3", {s.rows, s.n}, 0, s.rows, true},
                        {"scale", "f32", {s.rows, ngrp}, 0, s.rows, true},
                        {"p", "bf16", {s.rows, s.n}, 0, s.rows, true}};
    }
    if (linearised) rec.operands.push_back({"base", "bf16", {s.rows, s.n}, 0, s.rows, false});
    return rec;
}

/* The gain the trunk stores: f32, near one, so the norm's Jacobian is not a
 * diagonal of zeros. */
std::vector<double> gain(int64_t n, uint64_t seed) {
    std::vector<double> w((size_t)n);
    for (double& v : w) v = 1.0 + 0.5 * urand(&seed);
    return w;
}

struct FusedCase {
    std::vector<AdValue> in;   /* reads, in schema order */
    std::vector<AdValue> out;  /* writes, in write order */
};

/* One call of the op. The base point, when asked for, is the forward's own
 * activation: the quantiser is linearised where the trunk's record sits. */
FusedCase fused_case(const char* op, const FusedShape& s, bool linearised, uint64_t seed) {
    const int64_t ngrp = fused_groups(s);
    FusedCase c;
    if (is_rmsnorm(op)) {
        const std::vector<double> x = activation(Shape{s.name, s.rows, s.n, s.group}, seed);
        const std::vector<double> res = activation(Shape{s.name, s.rows, s.n, s.group}, seed + 1);
        const std::vector<double> w = gain(s.n, seed + 2);
        c.in = {values("bf16", {s.rows, s.n}, x), values("bf16", {s.rows, s.n}, res),
                values("f32", {s.n}, w)};
        if (linearised) {
            const std::vector<AdValue> fwd = forward(fused_rec(op, s, false), c.in);
            c.in.push_back(fwd[3]); /* the bf16 activation the quantiser saw */
        }
        c.out.push_back(
            values("bf16", {s.rows, s.n}, std::vector<double>((size_t)(s.rows * s.n), 0)));
        c.out.push_back(values("fp8_e4m3", {s.rows, s.n},
                               std::vector<double>((size_t)(s.rows * s.n), 0)));
        c.out.push_back(values("f32", {s.rows, ngrp},
                               std::vector<double>((size_t)(s.rows * ngrp), 0)));
        c.out.push_back(
            values("bf16", {s.rows, s.n}, std::vector<double>((size_t)(s.rows * s.n), 0)));
        return c;
    }
    const std::vector<double> gu = activation(Shape{s.name, s.rows, 2 * s.n, s.group}, seed);
    c.in = {values("bf16", {s.rows, 2 * s.n}, gu)};
    if (linearised) {
        const std::vector<AdValue> fwd = forward(fused_rec(op, s, false), c.in);
        c.in.push_back(fwd[2]); /* the product the quantiser saw */
    }
    c.out.push_back(values("fp8_e4m3", {s.rows, s.n},
                           std::vector<double>((size_t)(s.rows * s.n), 0)));
    c.out.push_back(values("f32", {s.rows, ngrp}, std::vector<double>((size_t)(s.rows * ngrp), 0)));
    c.out.push_back(values("bf16", {s.rows, s.n}, std::vector<double>((size_t)(s.rows * s.n), 0)));
    return c;
}

/* The residual is written in place, which is the schema index the tape marks
 * and the adjoint reads back. */
std::vector<size_t> fused_in_place(const char* op) {
    return is_rmsnorm(op) ? std::vector<size_t>{1} : std::vector<size_t>{};
}

void fused_kernel_check(const char* op, const FusedShape& s) {
    const OpRecord rec = fused_rec(op, s, false);
    const FusedCase c = fused_case(op, s, false, 0x9e3779b9ull);
    const KernelReport k = check_kernel(rec, c.in, c.out, bf16_tol, fused_in_place(op));
    std::printf("%-20s %-28s kernel max err %.6g\n", rec.op.c_str(), s.name, k.max_err);
    require(k);
}

void fused_fd_check(const char* op, const FusedShape& s) {
    const OpRecord rec = fused_rec(op, s, true);
    const FusedCase c = fused_case(op, s, true, 0x5deece66dull);
    const FdReport fd = check_adjoint(rec, c.in, c.out, fd_tol, 0xc0ffeeull, fused_in_place(op));
    std::printf("%-20s %-28s adjoint rel err %.3g\n", rec.op.c_str(), s.name, fd.rel);
    require(fd);
}

R97_TEST(adjoint_fused_quant_twin_matches_the_host_kernel) {
    for (const FusedShape& s : fused_shapes()) {
        fused_kernel_check("rmsnorm_quant_fp8", s);
        fused_kernel_check("gated_quant_fp8", s);
    }
}

R97_TEST(adjoint_fused_quant_agrees_with_finite_differences) {
    for (const FusedShape& s : fused_shapes()) {
        fused_fd_check("rmsnorm_quant_fp8", s);
        fused_fd_check("gated_quant_fp8", s);
    }
}

}  // namespace
