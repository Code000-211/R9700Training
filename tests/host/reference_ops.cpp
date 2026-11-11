// tests/host/reference_ops.cpp -- the reference step's two private ops through the adjoint check.
//
// `ref_add_cols` and `ref_select_rows` are where a fused LoRA block lands in its module's columns
// and where the full logits row space meets the objective's scored rows. Each is a registered op,
// so the ordinary checker is all they need: a twin and adjoint found by name, finite differences.
// Label: host.
#include "support/adjoint_check.h"
#include "support/test.h"

#include <cstdint>
#include <vector>

using namespace r97;
using namespace r97::adjoint_check;

namespace {

AdValue make_value(const std::string& dtype, const std::vector<int64_t>& shape) {
    AdValue v;
    v.dtype = dtype;
    v.shape = shape;
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    v.value.assign(static_cast<std::size_t>(n), 0.0);
    return v;
}

}  // namespace

R97_TEST(reference_op_add_cols) {
    OpRecord rec;
    rec.op = "ref_add_cols";
    rec.params = {{"M", int64_t{2}}, {"off", int64_t{1}}};
    rec.operands = {
        {"y", "f64", {2, 4}, 0, 2, false},
        {"inc", "f64", {2, 2}, 0, 2, false},
        {"y", "f64", {2, 4}, 0, 2, true},
    };
    std::vector<AdValue> in = {make_value("f64", {2, 4}), make_value("f64", {2, 2})};
    for (std::size_t i = 0; i < in[0].value.size(); ++i) in[0].value[i] = (double)i - 3.0;
    in[1].value = {0.5, -1.5, 2.0, 0.25};
    const std::vector<AdValue> out = {make_value("f64", {2, 4})};
    const FdReport f = check_adjoint(rec, in, out, 1e-6, 0x7101);
    std::printf("ref_add_cols adjoint rel err %.3g\n", f.rel);
    require(f);
}

R97_TEST(reference_op_select_rows) {
    OpRecord rec;
    rec.op = "ref_select_rows";
    rec.params = {{"M", int64_t{2}}};
    rec.operands = {
        {"logits_full", "f64", {4, 3}, 0, 4, false},
        {"idx", "i32", {2}, 0, 2, false},
        {"logits", "f64", {2, 3}, 0, 2, true},
    };
    std::vector<AdValue> in = {make_value("f64", {4, 3}), make_value("i32", {2})};
    for (std::size_t i = 0; i < in[0].value.size(); ++i) in[0].value[i] = (double)i / 4.0 - 1.0;
    in[1].value = {3.0, 1.0};
    const std::vector<AdValue> out = {make_value("f64", {2, 3})};
    const FdReport f = check_adjoint(rec, in, out, 1e-6, 0x7102);
    std::printf("ref_select_rows adjoint rel err %.3g\n", f.rel);
    require(f);
}
