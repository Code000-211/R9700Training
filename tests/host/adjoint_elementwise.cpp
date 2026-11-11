// tests/host/adjoint_elementwise.cpp -- the elementwise family
// (src/adjoint/host/elementwise.cpp) through both checks of
// tests/support/adjoint_check.h: each op's f64 twin against radiance's
// reference kernel, and its host adjoint against finite differences of that
// twin. Then the registry: every op of the family resolves to a twin and a host
// adjoint, and the two weights the family carries are named as frozen.
#include "support/adjoint_check.h"
#include "support/test.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

using namespace r97;
using namespace r97::adjoint_check;

namespace {

/* bf16 carries 8 significand bits, so a kernel that rounds as it goes lands
 * within about 2^-9 relative of the f64 twin; 0.02 leaves act_mul's rounding of
 * the activation before the product and the rms row's single rounding room
 * without hiding an op that computes something else. */
const double bf16_tol = 0.02;

/* The twin is exact in f64, so central differences reproduce its directional
 * derivative far inside 1e-6 relative. */
const double fd_tol = 1e-6;

/* Both tiny fixtures normalise at eps 1e-6 (docs/FIXTURES.md); a record carries
 * a real parameter as its decimal spelling. */
const char* const kEps = "1e-06";

/* A shape the family runs at: m rows of width n. The small one is the tiny
 * fixture's: 128 hidden, 4 heads of 32 (docs/FIXTURES.md). */
struct Shape {
    const char* name;
    int64_t m, n;
};

const Shape kShapes[] = {
    {"the tiny fixture's 5 x 128", 5, 128},
    {"3 x 64", 3, 64},
    {"7 x 17 (odd pitch)", 7, 17},
};

/* silu_mul's and act_mul's shape: the tiny fixture's feed-forward is 256 wide,
 * and silu_mul reads the fused gate and up rows, 2n of them. */
const Shape kFeedShapes[] = {
    {"the tiny fixture's 5 x 256", 5, 256},
    {"2 x 64", 2, 64},
    {"3 x 17 (odd pitch)", 3, 17},
};

/* What the checkers are given: the read operands' values in schema order, and
 * the written operands' dtypes and shapes. Values in [-1, 1). */
struct Case {
    std::vector<AdValue> in;
    std::vector<AdValue> out;
};

Case make_case(const OpRecord& rec, uint64_t seed) {
    Case c;
    for (const Operand& o : rec.operands) {
        AdValue v;
        v.dtype = o.dtype;
        v.shape = o.shape;
        int64_t n = 1;
        for (int64_t d : o.shape) n *= d;
        v.value.resize((size_t)n);
        for (double& x : v.value) x = urand(&seed);
        (o.written ? c.out : c.in).push_back(std::move(v));
    }
    return c;
}

/* -- one record per op, in the schema's operand order ---------------------- */

OpRecord unary_rec(const char* op, const Shape& s, const char* dtype) {
    OpRecord rec;
    rec.op = op;
    rec.params = {{"M", s.m}, {"n", s.n}, {"dtype", dtype}};
    rec.operands = {
        {"x", dtype, {s.m, s.n}, 0, s.m, false},
        {"y", dtype, {s.m, s.n}, 0, s.m, true},
    };
    return rec;
}

/* b whole, or one row of it that every row of a multiplies (the op's own schema
 * reads a one-row b that way). */
OpRecord mul_rec(const Shape& s, const char* dtype, bool one_row) {
    OpRecord rec;
    rec.op = "mul";
    rec.params = {{"M", s.m}, {"n", s.n}, {"dtype", dtype}};
    rec.operands = {
        {"a", dtype, {s.m, s.n}, 0, s.m, false},
        {"b", dtype, one_row ? std::vector<int64_t>{s.n} : std::vector<int64_t>{s.m, s.n}, 0,
         one_row ? 1 : s.m, false},
        {"y", dtype, {s.m, s.n}, 0, s.m, true},
    };
    return rec;
}

/* s is the op's weight, a single factor frozen for the whole block. */
OpRecord scale_rec(const Shape& s, const char* dtype) {
    OpRecord rec;
    rec.op = "scale";
    rec.params = {{"M", s.m}, {"n", s.n}, {"dtype", dtype}};
    rec.operands = {
        {"x", dtype, {s.m, s.n}, 0, s.m, false},
        {"s", dtype, {1}, 0, 0, false},
        {"y", dtype, {s.m, s.n}, 0, s.m, true},
    };
    return rec;
}

/* The gate in the first half of the fused row, the up projection in the second. */
OpRecord silu_mul_rec(const Shape& s, const char* dtype) {
    OpRecord rec;
    rec.op = "silu_mul";
    rec.params = {{"M", s.m}, {"n", s.n}, {"dtype", dtype}};
    rec.operands = {
        {"gate_up", dtype, {s.m, 2 * s.n}, 0, s.m, false},
        {"y", dtype, {s.m, s.n}, 0, s.m, true},
    };
    return rec;
}

OpRecord act_mul_rec(const Shape& s, const char* dtype, const char* act) {
    OpRecord rec;
    rec.op = "act_mul";
    rec.params = {{"M", s.m}, {"n", s.n}, {"act", std::string(act)}, {"dtype", dtype}};
    rec.operands = {
        {"a", dtype, {s.m, s.n}, 0, s.m, false},
        {"b", dtype, {s.m, s.n}, 0, s.m, false},
        {"y", dtype, {s.m, s.n}, 0, s.m, true},
    };
    return rec;
}

OpRecord cast_rec(const Shape& s, const char* from, const char* to) {
    OpRecord rec;
    rec.op = "cast";
    rec.params = {
        {"M", s.m}, {"n", s.n}, {"from", std::string(from)}, {"to", std::string(to)},
    };
    rec.operands = {
        {"x", from, {s.m, s.n}, 0, s.m, false},
        {"y", to, {s.m, s.n}, 0, s.m, true},
    };
    return rec;
}

/* Both tiny fixture rmsnorms are bf16 rows with an f32 gain shifted by wadd 1.0:
 * arch/common/rad_block_attn_gated.h declares attn_norm, attn_q_norm and attn_k_norm
 * RAD_F32 and passes wadd / wadd_qk, and arch/qwen35_bf16/qwen35_bf16.cpp sets both
 * to 1.0 (the final norm too), so a zero-centred gain reads as GemmaRMSNorm's
 * 1 + w. `mult` weights the normed row before its single rounding. */
OpRecord rmsnorm_rec(const Shape& s, const char* dtype, const char* gain_dtype, const char* wadd,
                     const char* mult) {
    OpRecord rec;
    rec.op = "rmsnorm";
    rec.params = {
        {"M", s.m},      {"n", s.n},  {"eps", std::string(kEps)},
        {"dtype", dtype}, {"wadd", std::string(wadd)}, {"mult", std::string(mult)},
    };
    rec.operands = {
        {"x", dtype, {s.m, s.n}, 0, s.m, false},
        {"w", gain_dtype, {s.n}, 0, 1, false},
        {"y", dtype, {s.m, s.n}, 0, s.m, true},
    };
    return rec;
}

/* -- the two checks, once per record --------------------------------------- */

struct Named {
    std::string label;
    OpRecord rec;
};

void check_family(const char* op, const std::vector<Named>& cases) {
    for (const Named& c : cases) {
        const Case v = make_case(c.rec, 0x5eed);
        const KernelReport k = check_kernel(c.rec, v.in, v.out, bf16_tol);
        const FdReport fd = check_adjoint(c.rec, v.in, v.out, fd_tol);
        std::printf("%s %s: kernel max err %.6g, adjoint rel err %.3g\n", op, c.label.c_str(),
                    k.max_err, fd.rel);
        require(k);
        require(fd);
    }
}

void elementwise_test() {
    /* (1) the twin against radiance's host kernel, (2) the adjoint against
     * finite differences of the twin -- three shapes each, one of them the tiny
     * fixture's, in the dtypes and modes the two fixtures issue. */

    std::vector<Named> mul;
    std::vector<Named> scale;
    std::vector<Named> sigmoid;
    std::vector<Named> silu;
    for (const Shape& s : kShapes) {
        mul.push_back({std::string(s.name) + ", bf16", mul_rec(s, "bf16", false)});
        mul.push_back({std::string(s.name) + ", bf16, b one row", mul_rec(s, "bf16", true)});
        mul.push_back({std::string(s.name) + ", f32", mul_rec(s, "f32", false)});
        scale.push_back({std::string(s.name) + ", bf16", scale_rec(s, "bf16")});
        sigmoid.push_back({std::string(s.name) + ", bf16", unary_rec("sigmoid", s, "bf16")});
        silu.push_back({std::string(s.name) + ", bf16", unary_rec("silu", s, "bf16")});
    }
    scale.push_back({std::string(kShapes[0].name) + ", f32", scale_rec(kShapes[0], "f32")});
    check_family("mul", mul);
    check_family("scale", scale);
    check_family("sigmoid", sigmoid);
    check_family("silu", silu);

    std::vector<Named> silu_mul;
    std::vector<Named> act_silu;
    std::vector<Named> act_sigmoid;
    for (const Shape& s : kFeedShapes) {
        silu_mul.push_back({std::string(s.name) + ", bf16", silu_mul_rec(s, "bf16")});
        act_silu.push_back({std::string(s.name) + ", bf16, act silu",
                            act_mul_rec(s, "bf16", "silu")});
        act_sigmoid.push_back({std::string(s.name) + ", bf16, act sigmoid",
                               act_mul_rec(s, "bf16", "sigmoid")});
    }
    check_family("silu_mul", silu_mul);
    check_family("act_mul", act_silu);
    check_family("act_mul", act_sigmoid);

    /* A cast only rounds, so every float pair the plugins issue is checked: the
     * identity row copy (from == to, rad_block_attn_fp8.h's q lift), and the
     * widening and narrowing between the bf16 activations and the f32 logits. */
    std::vector<Named> cast;
    for (const Shape& s : kShapes) {
        cast.push_back({std::string(s.name) + ", bf16 to bf16", cast_rec(s, "bf16", "bf16")});
        cast.push_back({std::string(s.name) + ", bf16 to f32", cast_rec(s, "bf16", "f32")});
        cast.push_back({std::string(s.name) + ", f32 to bf16", cast_rec(s, "f32", "bf16")});
        cast.push_back({std::string(s.name) + ", f32 to f32", cast_rec(s, "f32", "f32")});
    }
    cast.push_back({std::string(kShapes[0].name) + ", bf16 to f16",
                    cast_rec(kShapes[0], "bf16", "f16")});
    check_family("cast", cast);

    /* wadd 1.0 is the two fixtures' own mode, at the block norm's width and at the
     * per-head norm's (5 tokens x 4 heads of 32). The third case is the plain
     * norm with a bf16 gain and Gemma 4's residual mult, the other two modes the
     * kernel carries. */
    const Shape head{"the tiny fixture's 5 x 4 heads x 32", 5 * 4, 32};
    std::vector<Named> rmsnorm;
    rmsnorm.push_back({"the tiny fixture's 5 x 128, bf16, f32 gain, wadd 1",
                       rmsnorm_rec(kShapes[0], "bf16", "f32", "1", "1")});
    rmsnorm.push_back({std::string(head.name) + ", bf16, f32 gain, wadd 1",
                       rmsnorm_rec(head, "bf16", "f32", "1", "1")});
    rmsnorm.push_back({std::string(kShapes[2].name) + ", bf16 gain, wadd 0, mult 0.5",
                       rmsnorm_rec(kShapes[2], "bf16", "bf16", "0", "0.5")});
    check_family("rmsnorm", rmsnorm);
}

/* The family is what the tape and `r97train ops --registered` read: every op of
 * it must resolve to a twin and a host adjoint, and rmsnorm's gain and scale's
 * factor must be frozen rather than asked for a cotangent. */
void registry_test() {
    const Shape s = kShapes[0];
    const std::vector<OpRecord> ops = {
        mul_rec(s, "bf16", false),
        scale_rec(s, "bf16"),
        unary_rec("sigmoid", s, "bf16"),
        unary_rec("silu", s, "bf16"),
        silu_mul_rec(kFeedShapes[0], "bf16"),
        act_mul_rec(s, "bf16", "sigmoid"),
        cast_rec(s, "bf16", "f32"),
        rmsnorm_rec(s, "bf16", "f32", "1", "1"),
    };
    for (const OpRecord& rec : ops) {
        std::string err;
        const TwinFn twin = twin_find(rec.op, rec.params, &err);
        const AdjointFn adj = adjoint_find(rec.op, Side::host, rec.params, &err);
        std::printf("%s: twin %s, host adjoint %s\n", rec.op.c_str(), twin ? "yes" : "no",
                    adj ? "yes" : "no");
        R97_CHECK(twin != nullptr);
        R97_CHECK(adj != nullptr);

        const std::vector<int> frozen = adjoint_frozen(rec.op, Side::host, rec.params);
        const size_t want = (rec.op == "rmsnorm" || rec.op == "scale") ? 1 : 0;
        R97_CHECK(frozen.size() == want);
        if (want == 1) {
            std::printf("%s: operand %d frozen\n", rec.op.c_str(), frozen[0]);
            R97_CHECK(frozen[0] == 1);
        }
    }
}

}  // namespace

R97_TEST(adjoint_elementwise) {
    elementwise_test();
    registry_test();
}
