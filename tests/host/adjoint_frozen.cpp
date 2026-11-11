// tests/host/adjoint_frozen.cpp -- the frozen operand, through the checker that
// every adjoint test runs. A frozen operand is a weight or a scale: its
// cotangent is not requested, so the check must neither seed a cotangent for it
// nor perturb it. The same adjoint, registered once with the operand frozen and
// once without, tells the two readings apart: the frozen row passes although it
// routes nothing to the second operand, and the open row fails for exactly that
// reason. Label: host.
#include "support/adjoint_check.h"
#include "support/test.h"

#include <cstdio>
#include <utility>
#include <vector>

using namespace r97;
using namespace r97::adjoint_check;

namespace {

/* The twin is exact in f64, so central differences reproduce it far inside 1e-6
 * relative. */
const double fd_tol = 1e-6;

/* y = a * w, element for element. */
void scale_twin(const OpRecord&, const std::vector<AdValue>& op, std::vector<AdValue>* out) {
    AdValue y;
    y.dtype = op[0].dtype;
    y.shape = op[0].shape;
    y.value.resize(op[0].value.size());
    for (size_t i = 0; i < y.value.size(); ++i) y.value[i] = op[0].value[i] * op[1].value[i];
    out->push_back(std::move(y));
}

/* Adds a's share of the cotangent and nothing to w: w is the frozen operand. The
 * adjoint reads w's forward value, so it goes through call.fwd like any op. */
void scale_a_only(const AdjointCall& call) {
    const AdValue& y_cot = call.cot[0];
    const AdValue& w = call.fwd[1];
    AdValue& a_cot = call.dc->at(0);
    for (size_t i = 0; i < a_cot.value.size(); ++i) a_cot.value[i] += y_cot.value[i] * w.value[i];
}

R97_REGISTER_TWIN("scale_frozen", scale_twin)
R97_REGISTER_TWIN("scale_open", scale_twin)
/* Operand 1 is frozen: the walk does not ask for its cotangent. */
R97_REGISTER_ADJOINT_F("scale_frozen", host, scale_a_only, 1)
/* The same adjoint with nothing frozen: now dropping w's route is a mistake. */
R97_REGISTER_ADJOINT("scale_open", host, scale_a_only)

OpRecord scale_rec(const char* op, int64_t m, int64_t n) {
    OpRecord rec;
    rec.op = op;
    rec.params = {{"M", m}, {"n", n}, {"dtype", "bf16"}};
    rec.operands = {
        {"a", "bf16", {m, n}, 0, m, false},
        {"w", "bf16", {m, n}, 0, m, false},
        {"y", "bf16", {m, n}, 0, m, true},
    };
    return rec;
}

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

}  // namespace

R97_TEST(adjoint_check_honours_a_frozen_operand) {
    const OpRecord frozen = scale_rec("scale_frozen", 12, 20);
    const Case c = make_case(frozen, 0x5eed);
    const FdReport f = check_adjoint(frozen, c.in, c.out, fd_tol);
    std::printf("scale_frozen (w frozen): adjoint rel err %.3g\n", f.rel);
    require(f);

    /* Nothing frozen, the same adjoint: w is read and not routed to, so the
     * checker must reject it. This is what says the pass above came from the
     * frozen marker and not from a check that never looked at w. */
    const OpRecord open = scale_rec("scale_open", 12, 20);
    const FdReport o = check_adjoint(open, c.in, c.out, fd_tol);
    std::printf("scale_open (w not frozen) rejected: %s (rel %.3g)\n", o.note.c_str(), o.rel);
    R97_CHECK(!o.ok);
}
