// tests/host/backward_alloc.cpp -- the walk's cotangent allocation follows the row's contract.
//
// A dc slot the adjoint may not write -- a frozen operand, or a write the op does not read
// through -- holds no payload: the walk leaves it empty instead of zeroing a full copy of the
// operand. A record whose frozen operand is a giga-byte of weights therefore allocates no
// cotangent for it, and the report's peak cotangent bytes stays the walk's small working set.
// The per-op timing the report carries is checked on the same tapes.
#include "r97/backward.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "support/test.h"

namespace {

using r97::AdValue;
using r97::BoundaryNames;
using r97::OpParam;
using r97::OpRecord;
using r97::Operand;
using r97::Seed;

// y = s * (x * w), with `s` an integer parameter and `w` the frozen weight: its forward value is
// read by the row, its cotangent never is. The row writes x's slot and nothing else.
void alloc_scale_adjoint(const r97::AdjointCall& call) {
    double s = 1.0;
    for (const OpParam& p : call.rec.params)
        if (p.first == "s")
            if (const auto* i = std::get_if<int64_t>(&p.second)) s = static_cast<double>(*i);
    const AdValue& y_cot = call.cot.at(0);
    const AdValue& w = call.fwd.at(1);
    AdValue& x_cot = call.dc->at(0);
    for (size_t i = 0; i < x_cot.value.size(); ++i)
        x_cot.value[i] += s * y_cot.value[i] * w.value[i];
}
R97_REGISTER_ADJOINT_F("bwd_alloc_scale", host, alloc_scale_adjoint, 1)

Operand operand_of(const std::string& buffer, const std::vector<int64_t>& shape, bool written) {
    Operand o;
    o.buffer = buffer;
    o.dtype = "f32";
    o.shape = shape;
    o.row_lo = 0;
    o.rows = 0;
    o.written = written;
    return o;
}

OpRecord rec_of(const char* op, int64_t s, std::vector<Operand> operands, int64_t position) {
    OpRecord rec;
    rec.op = op;
    rec.params = {{"s", s}};
    rec.operands = std::move(operands);
    rec.position = position;
    return rec;
}

void set_snap(r97::StoredSnapshots* snaps, int64_t position, size_t operand, std::string buffer,
              const std::vector<int64_t>& shape, std::vector<double> value) {
    AdValue v;
    v.dtype = "f32";
    v.shape = shape;
    v.value = std::move(value);
    snaps->set(position, operand, std::move(v));
}

Seed seed_on(const char* buffer, int64_t row_hi, const std::vector<float>& cotangent) {
    Seed s;
    s.buffer = buffer;
    s.row_lo = 0;
    s.row_hi = row_hi;
    s.cotangent = cotangent;
    return s;
}

// One giga-byte of frozen weight: the walk allocates no cotangent for it, so the report's peak
// cotangent bytes is the walk's small working set -- not the weight's copy, which the old
// allocation zeroed on the call.
R97_TEST(backward_alloc_frozen_weight_holds_no_cotangent) {
    const size_t w_values = 1000000000 / sizeof(double);  // 10^9 bytes of f64
    const std::vector<int64_t> w_shape{static_cast<int64_t>(w_values)};
    const std::vector<OpRecord> records{
        rec_of("bwd_alloc_scale", 3,
               {operand_of("x", {4, 2}, false), operand_of("w", w_shape, false),
                operand_of("y", {4, 2}, true)},
               0)};

    r97::StoredSnapshots snaps;
    set_snap(&snaps, 0, 0, "x", {4, 2}, {0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.5, -0.25});
    set_snap(&snaps, 0, 1, "w", w_shape, std::vector<double>(w_values, 1.0));
    set_snap(&snaps, 0, 2, "y", {4, 2}, {0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.5, -0.25});

    BoundaryNames names;
    names.add("y", 0, 4);
    const std::vector<float> c{1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 0.25f, 2.0f, -0.5f};

    const r97::BackwardResult got =
        r97::backward_block(records, snaps, names, {seed_on("y", 4, c)}, {}, {});

    R97_CHECK(got.inputs.size() == 1);
    R97_CHECK(got.inputs[0].buffer == "x");
    std::vector<double> want;
    for (float v : c) want.push_back(3.0 * static_cast<double>(v));
    R97_CHECK(got.inputs[0].value == want);

    // The working set: the seed's version, the block's input, one op's cotangent and dc.
    R97_CHECK(got.report.peak_cotangent_bytes > 0);
    R97_CHECK(got.report.peak_cotangent_bytes < 16 * 1024 * 1024);

    // The row's call is in the report, named and counted.
    R97_CHECK(got.report.ops.size() == 1);
    R97_CHECK(got.report.ops[0].op == "bwd_alloc_scale");
    R97_CHECK(got.report.ops[0].calls == 1);
    R97_CHECK(got.report.ops[0].seconds >= 0.0);
}

// Two records of the same op: the report sums the calls and the seconds per op name, and the
// frozen weight's slot holds no payload in either.
R97_TEST(backward_alloc_report_sums_the_op) {
    const std::vector<int64_t> shape{4, 2};
    const std::vector<OpRecord> records{
        rec_of("bwd_alloc_scale", 2,
               {operand_of("x", shape, false), operand_of("w", shape, false),
                operand_of("y", shape, true)},
               0),
        rec_of("bwd_alloc_scale", 5,
               {operand_of("y", shape, false), operand_of("w", shape, false),
                operand_of("z", shape, true)},
               1),
    };

    r97::StoredSnapshots snaps;
    set_snap(&snaps, 0, 0, "x", shape, {0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.5, -0.25});
    set_snap(&snaps, 0, 1, "w", shape, {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0});
    set_snap(&snaps, 0, 2, "y", shape, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    set_snap(&snaps, 1, 0, "y", shape, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    set_snap(&snaps, 1, 1, "w", shape, {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0});
    set_snap(&snaps, 1, 2, "z", shape, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});

    BoundaryNames names;
    names.add("z", 0, 4);
    const std::vector<float> c{1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 0.25f, 2.0f, -0.5f};

    const r97::BackwardResult got =
        r97::backward_block(records, snaps, names, {seed_on("z", 4, c)}, {}, {});

    R97_CHECK(got.inputs.size() == 1);
    R97_CHECK(got.inputs[0].buffer == "x");
    std::vector<double> want;
    for (float v : c)
        want.push_back(2.0 * 5.0 * 2.0 * 2.0 * static_cast<double>(v));  // s0*s1*w*w
    R97_CHECK(got.inputs[0].value == want);

    R97_CHECK(got.report.ops.size() == 1);
    R97_CHECK(got.report.ops[0].op == "bwd_alloc_scale");
    R97_CHECK(got.report.ops[0].calls == 2);
    R97_CHECK(got.report.ops[0].seconds >= 0.0);
}

}  // namespace
