// tests/host/reference_linear.cpp -- the reference step's linear rows over a composed record.
//
// The fixture's composed graph keeps its own buffer names, so `operand_pos` falls back to the
// schema's read order. Read the order wrong and `b` is taken for `a_scale`; that is invisible to
// the family test, whose records use the schema names themselves. Build the composed record here
// and let the finite-difference check ask the twin's own gradient.
// Label: host.
#include "support/adjoint_check.h"
#include "support/test.h"

#include <cstdint>
#include <vector>

using namespace r97;
using namespace r97::adjoint_check;

namespace {

AdValue make_value(const std::string& dtype, const std::vector<int64_t>& shape, double fill) {
    AdValue v;
    v.dtype = dtype;
    v.shape = shape;
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    v.value.assign(static_cast<std::size_t>(n), fill);
    return v;
}

// A composed gemm_nt_q: reads are a, a_scale, b, b_scale, but the buffers carry the fixture's
// names, not the schema's. a_scale is given more elements than b needs so the wrong binding stays
// inside the array while the finite difference still sees it.
OpRecord composed_gemm_nt_q() {
    OpRecord rec;
    rec.op = "gemm_nt_q";
    rec.params = {{"M", 4}, {"N", 2}, {"K", 4}, {"group", 2}, {"dtype", "i8a8h"},
                  {"r97ref", 1}};
    rec.operands = {
        {"blk.0.attn_q", "i8", {4, 4}, 0, 4, false},
        {"blk.0.attn_q_scale", "f32", {4, 4}, 0, 4, false},
        {"blk.0.wq.weight", "i8", {2, 4}, 0, 2, false},
        {"blk.0.wq.scale", "bf16", {2, 2}, 0, 2, false},
        {"blk.0.attn_out", "f32", {4, 2}, 0, 4, true},
    };
    return rec;
}

}  // namespace

R97_TEST(reference_linear_gemm_nt_q_reads_in_schema_order) {
    const OpRecord rec = composed_gemm_nt_q();
    std::vector<AdValue> in = {
        make_value("i8", {4, 4}, 1.0),
        /* The family's int8 adjoint is in the dequantised basis and holds the activation scale at
         * one (tests/host/adjoint_linear.cpp), so do the same here: the read-order bug is visible
         * anyway, because the weight codes are not one. */
        make_value("f32", {4, 4}, 1.0),
        make_value("i8", {2, 4}, 0.5),
        make_value("bf16", {2, 2}, 1.0),
    };
    for (std::size_t i = 0; i < in[0].value.size(); ++i)
        in[0].value[i] = static_cast<double>(static_cast<int>(i % 5) - 2);
    for (std::size_t i = 0; i < in[2].value.size(); ++i)
        in[2].value[i] = static_cast<double>(static_cast<int>(i % 3) - 1);
    const std::vector<AdValue> out = {make_value("f32", {4, 2}, 0.0)};

    const FdReport f = check_adjoint(rec, in, out, 1e-6, 0x6201);
    std::printf("composed gemm_nt_q adjoint rel err %.3g\n", f.rel);
    require(f);
}
