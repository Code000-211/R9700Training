// tests/host/adjoint_quant_fused.cpp -- the fused quantised producers the int8 trunk uses:
// gated_quant_fp8 (silu(gate)*up -> bf16 -> code) and rmsnorm_quant_fp8 (residual in place, norm,
// bf16 -> code). Each is checked twice: the f64 twin against radiance's reference kernel, and the
// host adjoint against finite differences of a twin linearised at a base point, which is the
// straight-through definition docs/adjoints/quant.md states for the quantiser.
//
// The linearised records read the base point as extra operands after the schema (frozen, so the
// check neither perturbs them nor asks for their cotangent): the fused gate takes the previous
// gate_up's value, and the norm takes the previous x, residual and gain.
#include "support/adjoint_check.h"
#include "support/test.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

using namespace r97;
using namespace r97::adjoint_check;

namespace {

/* bf16 carries 8 significand bits; the kernel narrows as it goes, the twin's f64 does not, so
 * 0.03 leaves the rounding room without hiding an op that computes something else. */
const double bf16_tol = 0.03;

/* The twin's linearisation is exact, so central differences reproduce the adjoint well inside
 * 1e-6 relative. */
const double fd_tol = 1e-6;

AdValue values(const char* dtype, std::vector<int64_t> shape, std::vector<double> v) {
    AdValue a;
    a.dtype = dtype;
    a.shape = std::move(shape);
    a.value = std::move(v);
    return a;
}

AdValue random_value(const char* dtype, std::vector<int64_t> shape, uint64_t seed) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    std::vector<double> v((size_t)n);
    uint64_t s = seed;
    for (double& x : v) x = urand(&s);
    return values(dtype, std::move(shape), std::move(v));
}

/* The gate's record: gate_up [rows, 2n] in, codes [rows, n] and scales [rows, ngroups] out, and
 * with `out_bf16` the narrowed value beside them. */
OpRecord gated_rec(int64_t rows, int64_t n, int64_t group, bool linearised) {
    OpRecord rec;
    rec.op = linearised ? "gated_quant_fp8_ste" : "gated_quant_fp8";
    rec.params = {{"n", n}, {"group", group}, {"dtype", "bf16"}, {"act", "silu"}, {"M", rows}};
    rec.operands = {{"gate_up", "bf16", {rows, 2 * n}, 0, rows, false, false}};
    if (linearised)
        rec.operands.push_back({"gate_up0", "bf16", {rows, 2 * n}, 0, rows, false, false});
    rec.operands.push_back({"q", "fp8_e4m3", {rows, n}, 0, rows, true, false});
    rec.operands.push_back({"s", "f32", {rows, (n + group - 1) / group}, 0, rows, true, false});
    rec.operands.push_back({"out_bf16", "bf16", {rows, n}, 0, rows, true, false});
    return rec;
}

/* The norm's record: x and the in-place residual in, the gain [n] in, then codes, scales and the
 * narrowed value; a linearised record adds the base point's x, residual and gain after them. */
OpRecord rmsnorm_rec(int64_t rows, int64_t n, int64_t group, bool linearised,
                     bool in_place_res = true) {
    OpRecord rec;
    rec.op = linearised ? "rmsnorm_quant_fp8_ste" : "rmsnorm_quant_fp8";
    rec.params = {{"n", n}, {"group", group}, {"dtype", "bf16"}, {"eps", "0.000001"},
                  {"wadd", (int64_t)0},          {"M", rows}};
    rec.operands = {{"x", "bf16", {rows, n}, 0, rows, false, false},
                    {"res", "bf16", {rows, n}, 0, rows, in_place_res, in_place_res},
                    {"w", "f32", {n}, 0, 1, false, false},
                    {"q", "fp8_e4m3", {rows, n}, 0, rows, true, false},
                    {"s", "f32", {rows, (n + group - 1) / group}, 0, rows, true, false},
                    {"out_bf16", "bf16", {rows, n}, 0, rows, true, false}};
    if (linearised) {
        rec.operands.push_back({"x0", "bf16", {rows, n}, 0, rows, false, false});
        rec.operands.push_back({"res0", "bf16", {rows, n}, 0, rows, false, false});
        rec.operands.push_back({"w0", "f32", {n}, 0, 1, false, false});
    }
    return rec;
}

std::vector<AdValue> gated_in(const OpRecord& rec, int64_t rows, int64_t n, uint64_t seed) {
    std::vector<AdValue> in;
    in.push_back(random_value("bf16", {rows, 2 * n}, seed));
    if (rec.op == "gated_quant_fp8_ste") in.push_back(random_value("bf16", {rows, 2 * n}, seed + 1));
    return in;
}

std::vector<AdValue> rmsnorm_in(const OpRecord& rec, int64_t rows, int64_t n, uint64_t seed) {
    std::vector<AdValue> in;
    in.push_back(random_value("bf16", {rows, n}, seed));
    in.push_back(random_value("bf16", {rows, n}, seed + 1));
    in.push_back(random_value("f32", {n}, seed + 2));
    if (rec.op == "rmsnorm_quant_fp8_ste") {
        in.push_back(random_value("bf16", {rows, n}, seed + 3));
        in.push_back(random_value("bf16", {rows, n}, seed + 4));
        in.push_back(random_value("f32", {n}, seed + 5));
    }
    return in;
}

std::vector<AdValue> outs_of(int64_t rows, int64_t n, int64_t group) {
    std::vector<AdValue> out;
    out.push_back(values("fp8_e4m3", {rows, n}, {}));
    out.push_back(values("f32", {rows, (n + group - 1) / group}, {}));
    out.push_back(values("bf16", {rows, n}, {}));
    return out;
}

}  // namespace

R97_TEST(adjoint_quant_fused) {
    // The gate: the twin against the kernel, then the adjoint against its linearisation.
    {
        const int64_t rows = 3, n = 8, group = 4;
        const OpRecord rec = gated_rec(rows, n, group, false);
        const std::vector<AdValue> in = gated_in(rec, rows, n, 11);
        const KernelReport k = check_kernel(rec, in, outs_of(rows, n, group), bf16_tol);
        require(k);
    }
    {
        const int64_t rows = 3, n = 8, group = 4;
        const OpRecord rec = gated_rec(rows, n, group, true);
        const std::vector<AdValue> in = gated_in(rec, rows, n, 21);
        const FdReport r = check_adjoint(rec, in, outs_of(rows, n, group), fd_tol);
        require(r);
    }

    // The norm: same two checks, with the residual in place.
    {
        const int64_t rows = 2, n = 8, group = 8;
        // The kernel check reads the residual as a plain operand: its in-place write would be a
        // fourth written slot, and the twin writes only the code, the scale and the value.
        const OpRecord rec = rmsnorm_rec(rows, n, group, false, false);
        const std::vector<AdValue> in = rmsnorm_in(rec, rows, n, 31);
        const KernelReport k = check_kernel(rec, in, outs_of(rows, n, group), bf16_tol, {1});
        require(k);
    }
    {
        const int64_t rows = 2, n = 8, group = 8;
        // The residual is read as a plain operand here too: an in-place slot is one more written
        // operand in the check's counting, while the twin writes only the three outputs. The
        // adjoint's residual slot is the same either way.
        const OpRecord rec = rmsnorm_rec(rows, n, group, true, false);
        const std::vector<AdValue> in = rmsnorm_in(rec, rows, n, 41);
        const FdReport r = check_adjoint(rec, in, outs_of(rows, n, group), fd_tol);
        require(r);
    }
}
