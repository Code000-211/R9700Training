// tests/host/snapshot_share.cpp -- the frozen operands a step widens once.
//
// A step's weights are read by every block and written by none of them, so every block widened the
// same bytes to the same f64 values: on the real container that is ~3.7 GiB of f64 a block, built
// again for each of the 64 blocks. `r97::SharedSnapshots` (include/r97/snapshot_share.h) holds
// each such value once, keyed by what makes two reads the same read, and answers every
// (position, operand) key of the walk's store that reads it. The device join binds it for the
// operands of a buffer the step never writes; a block's own store then holds only what that block
// sees, and the step's frozen share is one copy rather than one per block.
//
// This test is the join's accounting in miniature: a two-block tape whose blocks both read one
// frozen weight, walked once with a per-block widen (today's shape) and once with the shared
// store. The walk must return the same bits -- the weight's f64 values and the input cotangents
// the row computes from them -- and the peak store figure must fall by exactly the frozen share.
#include "r97/backward.h"
#include "r97/snapshot_share.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "support/test.h"

namespace {

using r97::AdValue;
using r97::BoundaryNames;
using r97::OpRecord;
using r97::Operand;
using r97::Seed;

// y = x * w, elementwise, with `w` frozen: the row reads w's forward value and never writes a
// cotangent for it, so the walk allocates no slot for it.
void mul_adjoint(const r97::AdjointCall& call) {
    const AdValue& y_cot = call.cot.at(0);
    const AdValue& w = call.fwd.at(2);
    AdValue& x_cot = call.dc->at(1);
    for (size_t i = 0; i < x_cot.value.size(); ++i) x_cot.value[i] += y_cot.value[i] * w.value[i];
}
R97_REGISTER_ADJOINT_F("snapshot_share_mul", host, mul_adjoint, 2)

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

OpRecord block_of(int64_t position, const std::string& y, const std::string& x,
                  const std::vector<int64_t>& shape) {
    OpRecord rec;
    rec.op = "snapshot_share_mul";
    rec.operands = {operand_of(y, shape, true), operand_of(x, shape, false),
                    operand_of("w", {64, 64}, false)};
    rec.position = position;
    return rec;
}

AdValue value_of(const std::string& dtype, const std::vector<int64_t>& shape,
                 std::vector<double> value) {
    AdValue v;
    v.dtype = dtype;
    v.shape = shape;
    v.value = std::move(value);
    return v;
}

Seed seed_of(const std::string& buffer, int64_t rows, std::vector<float> cotangent) {
    Seed s;
    s.buffer = buffer;
    s.row_lo = 0;
    s.row_hi = rows;
    s.cotangent = std::move(cotangent);
    return s;
}

// What one block's walk costs and returns, as the join's report reads it.
struct BlockFigures {
    int64_t store_bytes = 0;     // the block's own store
    int64_t op_bytes = 0;        // the one op the walk copies out of it
    std::vector<double> grad;    // the cotangent the block handed down on x
    std::vector<double> frozen;  // the weight's values the walk was answered with
};

// One block of the two-block tape: its record, the buffers it reads, and its seed.
struct BlockInput {
    const std::vector<OpRecord>* records = nullptr;
    std::string y;
    std::string x;
    std::vector<double> x_values;
    std::vector<float> cotangent;
};

BlockFigures walk_block(const BlockInput& in, const r97::SnapshotProvider& provider,
                        int64_t a_values, int64_t frozen_position) {
    BoundaryNames names;
    names.add(in.y, 0, 8);
    const r97::BackwardResult got = r97::backward_block(
        *in.records, provider, names, {seed_of(in.y, 8, in.cotangent)}, {}, {});
    R97_CHECK(got.inputs.size() == 1);
    R97_CHECK(got.inputs[0].buffer == in.x);
    R97_CHECK(got.inputs[0].value.size() == static_cast<size_t>(a_values));
    R97_CHECK(got.report.forward_all_finite);
    BlockFigures fig;
    fig.op_bytes = got.report.peak_snapshot_bytes;
    fig.grad = got.inputs[0].value;
    const AdValue* answered = provider.at(frozen_position, 2);
    R97_CHECK(answered != nullptr);
    R97_CHECK(answered->shape == std::vector<int64_t>({64, 64}));
    fig.frozen = answered->value;
    return fig;
}

}  // namespace

// A key names one read: the same operand gives the same key, and a different buffer, dtype, row
// window or view gives another. Two reads a store may answer with one value are exactly the ones
// whose keys match.
R97_TEST(snapshot_key_identifies_one_read) {
    const Operand w = operand_of("w", {64, 64}, false);
    R97_CHECK(r97::snapshot_key(w) == r97::snapshot_key(w));

    Operand other = w;
    other.buffer = "v";
    R97_CHECK(r97::snapshot_key(other) != r97::snapshot_key(w));
    other = w;
    other.dtype = "bf16";
    R97_CHECK(r97::snapshot_key(other) != r97::snapshot_key(w));
    other = w;
    other.row_lo = 1;
    R97_CHECK(r97::snapshot_key(other) != r97::snapshot_key(w));
    other = w;
    other.rows = 32;
    R97_CHECK(r97::snapshot_key(other) != r97::snapshot_key(w));
    other = w;
    other.view_shape = {64, 64};
    other.view_stride = {1, 64};
    R97_CHECK(r97::snapshot_key(other) != r97::snapshot_key(w));
    other = w;
    other.shape = {32, 128};
    R97_CHECK(r97::snapshot_key(other) != r97::snapshot_key(w));
}

// The store holds one payload per key, answers every binding of it, and falls through to the base
// provider for what it does not hold.
R97_TEST(snapshot_share_counts_one_value_per_key) {
    r97::SharedSnapshots shared;
    const std::shared_ptr<const AdValue> first =
        shared.insert("w", value_of("f32", {4, 4}, std::vector<double>(16, 2.0)));
    const std::shared_ptr<const AdValue> again =
        shared.insert("w", value_of("f32", {4, 4}, std::vector<double>(16, 3.0)));
    // The first widen wins: two reads of one operand cannot be answered with different bits.
    R97_CHECK(first == again);
    R97_CHECK(shared.values() == 1);
    R97_CHECK(shared.bytes() == 16 * int64_t(sizeof(double)));

    shared.bind(7, 0, first);
    shared.bind(9, 0, first);
    R97_CHECK(shared.size() == 2);
    R97_CHECK(shared.bytes() == 16 * int64_t(sizeof(double)));  // two keys, one payload
    R97_CHECK(shared.at(7, 0) == first.get());
    R97_CHECK(shared.at(9, 0) == first.get());
    R97_CHECK(shared.at(8, 0) == nullptr);

    r97::StoredSnapshots base;
    base.set(8, 1, value_of("f32", {2}, {1.0, 2.0}));
    shared.set_base(&base);
    R97_CHECK(shared.at(8, 1) == base.at(8, 1));
    R97_CHECK(shared.at(8, 0) == nullptr);  // neither store answers it
    R97_CHECK(shared.size() == 2);          // a base answer is not a binding

    bool refused = false;
    try {
        shared.bind(1, 0, nullptr);
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    R97_CHECK(refused);
}

// Two blocks that both read one frozen weight: the shared store's figures are the per-block ones
// less the frozen share, and every value the walk returns is bit-identical.
R97_TEST(snapshot_share_walks_two_blocks_bit_identically) {
    const std::vector<int64_t> a_shape{8, 8};
    const int64_t a_values = 8 * 8;
    std::vector<double> w_values(64 * 64);
    for (size_t i = 0; i < w_values.size(); ++i)
        w_values[i] = 0.5 + 0.25 * static_cast<double>(i % 17) - 1.0;
    std::vector<double> x0_values(static_cast<size_t>(a_values));
    std::vector<double> x1_values(static_cast<size_t>(a_values));
    std::vector<float> c0(static_cast<size_t>(a_values));
    std::vector<float> c1(static_cast<size_t>(a_values));
    for (int64_t i = 0; i < a_values; ++i) {
        x0_values[static_cast<size_t>(i)] = 0.125 * static_cast<double>(i) - 0.5;
        x1_values[static_cast<size_t>(i)] = -0.25 * static_cast<double>(i) + 1.0;
        c0[static_cast<size_t>(i)] = static_cast<float>(0.5 * static_cast<double>(i) - 3.0);
        c1[static_cast<size_t>(i)] = static_cast<float>(2.0 - 0.125 * static_cast<double>(i));
    }

    const std::vector<OpRecord> block0{block_of(1, "y0", "x0", a_shape)};
    const std::vector<OpRecord> block1{block_of(2, "y1", "x1", a_shape)};
    const std::vector<BlockInput> inputs{
        BlockInput{&block0, "y0", "x0", x0_values, c0},
        BlockInput{&block1, "y1", "x1", x1_values, c1},
    };

    // Today's shape: every block's own store holds the weight, widened again for that block.
    std::vector<BlockFigures> plain;
    int64_t plain_peak_store = 0;
    int64_t plain_peak_op = 0;
    for (const BlockInput& in : inputs) {
        r97::StoredSnapshots snaps;
        const OpRecord& rec = (*in.records)[0];
        snaps.set(rec.position, 0, value_of("f32", a_shape,
                                            std::vector<double>(static_cast<size_t>(a_values), 1.0)));
        snaps.set(rec.position, 1, value_of("f32", a_shape, in.x_values));
        snaps.set(rec.position, 2, value_of("f32", {64, 64}, w_values));
        BlockFigures fig = walk_block(in, snaps, a_values, rec.position);
        fig.store_bytes = snaps.bytes();
        plain.push_back(std::move(fig));
        plain_peak_store = std::max(plain_peak_store, plain.back().store_bytes);
        plain_peak_op = std::max(plain_peak_op, plain.back().op_bytes);
    }

    // The shared shape: the weight widened once for the step and bound into both blocks' walks,
    // each block's own store holding only that block's activations.
    r97::SharedSnapshots shared;
    const std::string key = r97::snapshot_key(operand_of("w", {64, 64}, false));
    std::vector<BlockFigures> shared_figures;
    int64_t shared_peak_store = 0;
    int64_t shared_peak_op = 0;
    for (const BlockInput& in : inputs) {
        r97::StoredSnapshots snaps;
        shared.set_base(&snaps);
        const OpRecord& rec = (*in.records)[0];
        std::shared_ptr<const AdValue> w = shared.find(key);
        if (!w) w = shared.insert(key, value_of("f32", {64, 64}, w_values));
        shared.bind(rec.position, 2, w);
        snaps.set(rec.position, 0, value_of("f32", a_shape,
                                            std::vector<double>(static_cast<size_t>(a_values), 1.0)));
        snaps.set(rec.position, 1, value_of("f32", a_shape, in.x_values));
        BlockFigures fig = walk_block(in, shared, a_values, rec.position);
        fig.store_bytes = snaps.bytes();
        shared_figures.push_back(std::move(fig));
        shared_peak_store = std::max(shared_peak_store, shared_figures.back().store_bytes);
        shared_peak_op = std::max(shared_peak_op, shared_figures.back().op_bytes);
        shared.set_base(nullptr);
    }

    // The weight's values are the ones the per-block widen produced, to the bit, in both blocks,
    // and so are the cotangents the row computed from them.
    R97_CHECK(plain[0].frozen.size() == w_values.size());
    R97_CHECK(shared_figures[0].frozen.size() == w_values.size());
    R97_CHECK(std::memcmp(plain[0].frozen.data(), w_values.data(),
                          w_values.size() * sizeof(double)) == 0);
    for (size_t b = 0; b < inputs.size(); ++b) {
        R97_CHECK(shared_figures[b].frozen.size() == plain[b].frozen.size());
        R97_CHECK(std::memcmp(shared_figures[b].frozen.data(), plain[b].frozen.data(),
                              plain[b].frozen.size() * sizeof(double)) == 0);
        R97_CHECK(shared_figures[b].grad == plain[b].grad);
    }

    // The figures: the shared store's block holds the activations, the step holds the weight once,
    // and the per-block peak falls by exactly the frozen share -- both the store and the peak the
    // report quotes (the store plus the one op the walk copies out of it).
    const int64_t frozen_bytes = static_cast<int64_t>(w_values.size()) * int64_t(sizeof(double));
    R97_CHECK(shared.bytes() == frozen_bytes);
    R97_CHECK(shared.values() == 1);
    R97_CHECK(shared_peak_store > 0);
    R97_CHECK(plain_peak_store == shared_peak_store + frozen_bytes);
    R97_CHECK(plain_peak_op == shared_peak_op);
    R97_CHECK(plain_peak_store + plain_peak_op ==
              shared_peak_store + shared_peak_op + frozen_bytes);

    std::printf("snapshot_share: per block %lld B store + %lld B op, against %lld B + %lld B "
                "before; the frozen %lld B is held once\n",
                (long long)shared_peak_store, (long long)shared_peak_op,
                (long long)plain_peak_store, (long long)plain_peak_op, (long long)frozen_bytes);
}
