// tests/host/adjoint_add.cpp -- `add` (src/adjoint/host/add.cpp) through both
// checks of tests/support/adjoint_check.h: its f64 twin against radiance's
// reference kernel, and its host adjoint against finite differences of the
// twin. Then the deliberate failures: the checker must reject a wrong-signed
// adjoint and one that drops a route, and a lookup of an op nobody serves must
// name the op and its parameters.
#include "support/adjoint_check.h"
#include "support/test.h"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

using namespace r97;
using namespace r97::adjoint_check;

namespace {

/* bf16 carries 8 significand bits, so a kernel that rounds as it goes lands
 * within about 2^-9 relative of the f64 twin; 0.02 leaves the sum's rounding
 * and the bf16 store room without hiding an op that computes something else. */
const double bf16_tol = 0.02;

/* The twin is exact in f64, so central differences reproduce its directional
 * derivative far inside 1e-6 relative. */
const double fd_tol = 1e-6;

/* A record for `add` at M x n, with b reading the whole block or one row of it
 * (a per-channel bias), as the op's own schema reads b. */
OpRecord add_rec(int64_t m, int64_t n, bool bias) {
    OpRecord rec;
    rec.op = "add";
    rec.params = {{"M", m}, {"n", n}, {"dtype", "bf16"}};
    rec.operands = {
        {"a", "bf16", {m, n}, 0, m, false},
        {"b", "bf16", bias ? std::vector<int64_t>{1, n} : std::vector<int64_t>{m, n}, 0,
         bias ? 1 : m, false},
        {"y", "bf16", {m, n}, 0, m, true},
    };
    return rec;
}

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

/* -- the ops behind the deliberate failures -------------------------------- */

/* The same elementwise sum, written here rather than borrowed from add.cpp, so
 * the failure modes do not depend on the op under test. One function serves
 * both names: the registrar object takes its name from the registration line. */
void wrong_twin(const OpRecord&, const std::vector<AdValue>& op, std::vector<AdValue>* out) {
    AdValue y{op[0].dtype, op[0].shape, op[0].value};
    for (size_t i = 0; i < y.value.size(); ++i) y.value[i] += op[1].value[i];
    out->push_back(std::move(y));
}

/* Wrong sign: sends both routes the wrong way, so finite differences put the
 * directional derivative at twice the claimed value. */
void neg_adjoint(const AdjointCall& call) {
    const AdValue& y_cot = call.cot[0];
    for (size_t slot = 0; slot < 2; ++slot)
        for (size_t i = 0; i < y_cot.value.size(); ++i)
            call.dc->at(slot).value[i] -= 2.0 * y_cot.value[i];
}

/* Missing term: routes the cotangent to a and never to b, although b is read
 * and not frozen. */
void miss_adjoint(const AdjointCall& call) {
    const AdValue& y_cot = call.cot[0];
    for (size_t i = 0; i < y_cot.value.size(); ++i) call.dc->at(0).value[i] += y_cot.value[i];
}

R97_REGISTER_TWIN("add_neg", wrong_twin)
R97_REGISTER_TWIN("add_miss", wrong_twin)
R97_REGISTER_ADJOINT("add_neg", host, neg_adjoint)
R97_REGISTER_ADJOINT("add_miss", host, miss_adjoint)

void adjoint_add_test() {
    struct Shape {
        const char* name;
        int64_t m, n;
        bool bias;
    };
    const Shape shapes[] = {
        {"1 x 128", 1, 128, false},
        {"64 x 96 with a bias", 64, 96, true},
        {"37 x 129", 37, 129, false},
    };

    /* (1) the twin against radiance's host kernel, (2) the adjoint against
     * finite differences of the twin -- for three shapes. */
    for (const Shape& s : shapes) {
        const OpRecord rec = add_rec(s.m, s.n, s.bias);
        const Case c = make_case(rec, 0x5eed);

        const KernelReport k = check_kernel(rec, c.in, c.out, bf16_tol);
        const FdReport fd = check_adjoint(rec, c.in, c.out, fd_tol);
        std::printf("add %s: kernel max err %.6g, adjoint rel err %.3g\n", s.name, k.max_err,
                    fd.rel);
        require(k);
        require(fd);
    }

    /* The checker must reject an adjoint that is wrong in either way: a check
     * that passes a wrong adjoint checks nothing. */
    for (const char* name : {"add_neg", "add_miss"}) {
        OpRecord rec = add_rec(8, 16, false);
        rec.op = name;
        const Case c = make_case(rec, 0x5eed);
        const FdReport fd = check_adjoint(rec, c.in, c.out, fd_tol);
        std::printf("%s rejected: %s (rel %.3g)\n", name, fd.note.c_str(), fd.rel);
        R97_CHECK(!fd.ok);
    }

    /* An op with no adjoint says which op and which parameters it could not
     * serve, so a missing op reads as one in a failed run. */
    const OpRecord missing = add_rec(4, 8, false);
    const std::string err = lookup_error("not_an_op", Side::host, missing.params);
    std::printf("lookup error: %s\n", err.c_str());
    R97_CHECK(err.find("not_an_op") != std::string::npos);
    R97_CHECK(err.find("M=4") != std::string::npos);
    R97_CHECK(err.find("n=8") != std::string::npos);
    R97_CHECK(err.find("dtype=bf16") != std::string::npos);
}

}  // namespace

R97_TEST(adjoint_add) { adjoint_add_test(); }
