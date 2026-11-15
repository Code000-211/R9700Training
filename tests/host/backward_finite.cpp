// tests/host/backward_finite.cpp -- the walk's finiteness attribution: a non-finite cotangent is
// named by op, operand and value, and told apart from one the forward already had.
//
// A NaN gradient stops a step, and bisecting a step to find the op that made it costs a GPU
// window each time. The walk therefore counts the non-finite values it is about to hand on, per
// op, and keeps the first one it met with the operand it belongs to and where it came from: the
// row that produced it, or the walk that handed it in (a seed, the cotangent from above, a record
// later in the forward). The forward's own operands are asked the same question, because a NaN
// born in the forward is a different bug from one born in the walk.
//
// The blocks are hand-built, as backward_core's are, and the rows are stand-ins registered here:
// nothing here needs a model or a card. Each case plants exactly one non-finite value and reads
// the report back.
#include "r97/backward.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "support/test.h"

namespace {

using r97::AdValue;
using r97::OpRecord;
using r97::Operand;
using r97::Seed;

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// ------------------------------------------------------------------ stand-in rows

// x -> y: the cotangent of x is the cotangent of y.
void copy_adjoint(const r97::AdjointCall& call) {
    AdValue& x = call.dc->at(0);
    const AdValue& y = call.cot.at(0);
    for (size_t i = 0; i < y.value.size(); ++i) x.value[i] += y.value[i];
}
R97_REGISTER_ADJOINT("fin_copy", host, copy_adjoint)

// The same, and then one value of the read's cotangent is an inf: the row a walk has to name.
void inf_adjoint(const r97::AdjointCall& call) {
    AdValue& x = call.dc->at(0);
    const AdValue& y = call.cot.at(0);
    for (size_t i = 0; i < y.value.size(); ++i) x.value[i] += y.value[i];
    x.value[1] = kInf;
}
R97_REGISTER_ADJOINT("fin_inf", host, inf_adjoint)

// ------------------------------------------------------------------ a block's buffers

Operand read(const std::string& buffer, std::vector<int64_t> shape) {
    Operand o;
    o.buffer = buffer;
    o.dtype = "f32";
    o.shape = std::move(shape);
    o.row_lo = 0;
    o.rows = 0;  // every row of the buffer
    o.written = false;
    return o;
}

Operand written(const std::string& buffer, std::vector<int64_t> shape) {
    Operand o = read(buffer, std::move(shape));
    o.written = true;
    return o;
}

OpRecord op(const std::string& name, std::vector<Operand> operands, int64_t position) {
    OpRecord rec;
    rec.op = name;
    rec.operands = std::move(operands);
    rec.position = position;
    return rec;
}

using Values = std::map<std::string, std::vector<double>>;

// A block with the forward values its ops saw: one snapshot per operand, the buffer's values as
// given. `boundary` names the buffers a seed may enter at; each is a boundary over all its rows.
struct Block {
    std::vector<OpRecord> records;
    r97::StoredSnapshots snaps;
    r97::BoundaryNames boundaries;
};

Block block_of(std::vector<OpRecord> records, const Values& values,
               const std::vector<std::string>& boundary) {
    Block b;
    b.records = std::move(records);
    std::map<std::string, int64_t> rows;
    for (const OpRecord& rec : b.records)
        for (size_t j = 0; j < rec.operands.size(); ++j) {
            const Operand& o = rec.operands[j];
            rows[o.buffer] = o.shape.empty() ? 1 : o.shape[0];
            AdValue v;
            v.dtype = o.dtype;
            v.shape = o.shape;
            v.value = values.at(o.buffer);
            b.snaps.set(rec.position, j, std::move(v));
        }
    for (const std::string& name : boundary) b.boundaries.add(name, 0, rows.at(name));
    return b;
}

Seed seed_of(std::string buffer, int64_t row_lo, int64_t row_hi, std::vector<float> cotangent) {
    Seed s;
    s.buffer = std::move(buffer);
    s.row_lo = row_lo;
    s.row_hi = row_hi;
    s.cotangent = std::move(cotangent);
    return s;
}

const std::vector<int64_t> kShape{3, 2};
const std::vector<double> kFinite{0.5, -1.0, 2.0, 0.25, -0.75, 1.5};
const std::vector<float> kSeed{1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 0.25f};

const r97::OpTiming* timing_of(const r97::BackwardReport& report, const std::string& name) {
    for (const r97::OpTiming& t : report.ops)
        if (t.op == name) return &t;
    return nullptr;
}

const r97::InputCotangent& input_named(const r97::BackwardResult& got, const std::string& name) {
    for (const auto& in : got.inputs)
        if (in.buffer == name) return in;
    throw std::runtime_error("test: no input cotangent for " + name);
}

// ------------------------------------------------------------------ the cases

// The row itself made the inf: the walk names it, the operand it is the cotangent of, and the
// value's offset in that operand's payload.
R97_TEST(backward_finite_from_row) {
    Block b = block_of({op("fin_inf", {read("x", kShape), written("y", kShape)}, 0)},
                       {{"x", kFinite}, {"y", kFinite}}, {"y"});
    const std::vector<Seed> seeds{seed_of("y", 0, 3, kSeed)};

    const r97::BackwardResult got =
        r97::backward_block(b.records, b.snaps, b.boundaries, seeds, {}, {});

    R97_CHECK(got.report.first_non_finite_op == "fin_inf");
    R97_CHECK(got.report.first_non_finite_operand == "x");
    R97_CHECK(got.report.first_non_finite_operand_index == 0);
    R97_CHECK(got.report.first_non_finite_position == 0);
    R97_CHECK(got.report.first_non_finite_index == 1);  // inf_adjoint plants x's value 1
    R97_CHECK(got.report.first_non_finite_from_row);

    // The op's own count: the one slot the row may add to, counted once.
    const r97::OpTiming* t = timing_of(got.report, "fin_inf");
    R97_CHECK(t != nullptr);
    R97_CHECK(t->calls == 1);
    R97_CHECK(t->non_finite == 1);

    // The value the walk handed on is the one it named, and the forward was finite.
    R97_CHECK(!std::isfinite(input_named(got, "x").value[1]));
    R97_CHECK(std::isfinite(input_named(got, "x").value[0]));
    R97_CHECK(got.report.forward_all_finite);
    R97_CHECK(got.report.forward_first_non_finite_operand.empty());
}

// The seed carried the inf in: the row only passes it on, so the report says the walk handed it
// in rather than blaming the row, and names the operand the seed entered on.
R97_TEST(backward_finite_handed_in) {
    Block b = block_of({op("fin_copy", {read("x", kShape), written("y", kShape)}, 0)},
                       {{"x", kFinite}, {"y", kFinite}}, {"y"});
    std::vector<float> c = kSeed;
    c[4] = std::numeric_limits<float>::infinity();
    const std::vector<Seed> seeds{seed_of("y", 0, 3, c)};

    const r97::BackwardResult got =
        r97::backward_block(b.records, b.snaps, b.boundaries, seeds, {}, {});

    R97_CHECK(got.report.first_non_finite_op == "fin_copy");
    R97_CHECK(got.report.first_non_finite_operand == "y");  // the written operand, index 1
    R97_CHECK(got.report.first_non_finite_operand_index == 1);
    R97_CHECK(got.report.first_non_finite_position == 0);
    R97_CHECK(got.report.first_non_finite_index == 4);
    R97_CHECK(!got.report.first_non_finite_from_row);

    // Two slots carried it: the cotangent the walk handed in, and the read's slot the row copied
    // it into.
    const r97::OpTiming* t = timing_of(got.report, "fin_copy");
    R97_CHECK(t != nullptr);
    R97_CHECK(t->non_finite == 2);

    R97_CHECK(got.report.forward_all_finite);
}

// A chain: the inf is made at the block's first op and handed down to the second, whose row is
// clean. The walk meets the second first, must not blame it, and names the first.
R97_TEST(backward_finite_chain) {
    Block b = block_of({op("fin_inf", {read("x", kShape), written("h", kShape)}, 0),
                        op("fin_copy", {read("h", kShape), written("y", kShape)}, 1)},
                       {{"x", kFinite}, {"h", kFinite}, {"y", kFinite}}, {"y"});
    const std::vector<Seed> seeds{seed_of("y", 0, 3, kSeed)};

    const r97::BackwardResult got =
        r97::backward_block(b.records, b.snaps, b.boundaries, seeds, {}, {});

    R97_CHECK(got.report.first_non_finite_op == "fin_inf");
    R97_CHECK(got.report.first_non_finite_operand == "x");
    R97_CHECK(got.report.first_non_finite_operand_index == 0);
    R97_CHECK(got.report.first_non_finite_position == 0);
    R97_CHECK(got.report.first_non_finite_from_row);

    const r97::OpTiming* clean = timing_of(got.report, "fin_copy");
    R97_CHECK(clean != nullptr);
    R97_CHECK(clean->calls == 1);
    R97_CHECK(clean->non_finite == 0);
}

// The forward's own values were not finite and every cotangent was: the walk blames no row and
// says which operand of the forward carried the NaN.
R97_TEST(backward_finite_forward) {
    Values values{{"x", kFinite}, {"y", kFinite}};
    values["x"][2] = kNaN;
    Block b = block_of({op("fin_copy", {read("x", kShape), written("y", kShape)}, 0)}, values,
                       {"y"});
    const std::vector<Seed> seeds{seed_of("y", 0, 3, kSeed)};

    const r97::BackwardResult got =
        r97::backward_block(b.records, b.snaps, b.boundaries, seeds, {}, {});

    R97_CHECK(!got.report.forward_all_finite);
    R97_CHECK(got.report.forward_first_non_finite_operand == "x");
    R97_CHECK(got.report.forward_first_non_finite_operand_index == 0);
    R97_CHECK(got.report.forward_first_non_finite_position == 0);
    R97_CHECK(got.report.forward_first_non_finite_index == 2);

    // No row made anything non-finite, and the walk says so.
    R97_CHECK(got.report.first_non_finite_op.empty());
    R97_CHECK(got.report.first_non_finite_operand.empty());
    for (const r97::OpTiming& t : got.report.ops) R97_CHECK(t.non_finite == 0);
}

}  // namespace
