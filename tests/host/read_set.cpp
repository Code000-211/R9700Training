// tests/host/read_set.cpp -- the two-way rule: the read set the walk proves from the tape, and what
// the rmsnorm adjoint does with it.
//
// A row of an op's output that no consumer reads is provably dead, so a non-finite forward value
// there is zeroed rather than refused; a non-finite row a consumer does read refuses. The set has
// to be proven from the tape, never inferred: the one structural case is a record whose every
// written output is read by gather_rows and nothing else, whose index operand then names exactly
// the rows anything reads (src/read_set.h). Everything else is unproven and keeps the loud refusal.
//
// The blocks are hand-built, as backward_core's and backward_finite's are: no model, no card.
#include "read_set.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "r97/backward.h"
#include "support/test.h"

namespace {

using r97::AdValue;
using r97::OpRecord;
using r97::Operand;
using r97::Seed;

constexpr double kInf = std::numeric_limits<double>::infinity();

Operand read(const std::string& buffer, std::vector<int64_t> shape) {
    Operand o;
    o.buffer = buffer;
    o.dtype = "f32";
    o.shape = std::move(shape);
    o.row_lo = 0;
    o.rows = 0;
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

// A snapshot for one operand: the value the walk will read for it.
void snap(r97::StoredSnapshots* s, int64_t position, size_t j, std::vector<int64_t> shape,
          std::vector<double> value) {
    AdValue v;
    v.dtype = "f32";
    v.shape = std::move(shape);
    v.value = std::move(value);
    s->set(position, j, std::move(v));
}

const std::vector<int64_t> kRow{2, 2};      // a record's [rows, n]
const std::vector<int64_t> kIdx{1};         // the gather's index list: one row
const std::vector<int64_t> kOut{1, 2};

// A norm whose only consumer is a gather: the gather's index list is the read set.
R97_TEST(read_set_from_a_lone_gather) {
    const std::vector<OpRecord> all{
        op("rmsnorm", {read("x", kRow), read("w", {2}), written("h", kRow)}, 0),
        op("gather_rows", {read("h", kRow), read("raw:1", kIdx), written("h_out", kOut)}, 1),
    };
    r97::StoredSnapshots snaps;
    snap(&snaps, 1, 1, kIdx, {1.0});  // the gather selects row 1
    const auto reads = r97::derive_read_sets(all, all, snaps, {});
    const auto it = reads.find(0);
    R97_CHECK(it != reads.end());
    R97_CHECK(it->second == std::vector<int64_t>{1});
    // The gather's own output is not read by anything in the tape: with no consumer its rows are
    // all provably unread, which is the correct empty set.
    const auto g = reads.find(1);
    R97_CHECK(g != reads.end());
    R97_CHECK(g->second.empty());
}

// A consumer that is not a gather leaves the record unproven: every row counts as readable.
R97_TEST(read_set_needs_every_consumer_to_be_a_gather) {
    const std::vector<OpRecord> all{
        op("rmsnorm", {read("x", kRow), read("w", {2}), written("h", kRow)}, 0),
        op("add", {read("h", kRow), read("z", kRow), written("y", kRow)}, 1),
    };
    r97::StoredSnapshots snaps;
    const auto reads = r97::derive_read_sets(all, all, snaps, {});
    R97_CHECK(reads.find(0) == reads.end());
}

// A seed reads a buffer outside the tape, so the record that writes it is never proven.
R97_TEST(read_set_an_external_read_is_never_proven) {
    const std::vector<OpRecord> all{
        op("rmsnorm", {read("x", kRow), read("w", {2}), written("h", kRow)}, 0),
        op("gather_rows", {read("h", kRow), read("raw:1", kIdx), written("h_out", kOut)}, 1),
    };
    r97::StoredSnapshots snaps;
    snap(&snaps, 1, 1, kIdx, {1.0});
    const auto reads = r97::derive_read_sets(all, all, snaps, {"h_out"});
    R97_CHECK(reads.find(1) == reads.end());
}

// The gather's index list is not in the walk's snapshots: unproven rather than guessed.
R97_TEST(read_set_a_missing_index_list_is_unproven) {
    const std::vector<OpRecord> all{
        op("rmsnorm", {read("x", kRow), read("w", {2}), written("h", kRow)}, 0),
        op("gather_rows", {read("h", kRow), read("raw:1", kIdx), written("h_out", kOut)}, 1),
    };
    r97::StoredSnapshots snaps;
    const auto reads = r97::derive_read_sets(all, all, snaps, {});
    R97_CHECK(reads.find(0) == reads.end());
}

// The row applies the rule: a non-finite row outside the set is zeroed, a row inside it is
// differentiated, and the two counts say which happened.
R97_TEST(read_set_adjoint_zeroes_an_unreadable_row) {
    std::vector<OpRecord> records{
        op("rmsnorm", {read("x", kRow), read("w", {2}), written("h", kRow)}, 0),
    };
    r97::StoredSnapshots snaps;
    snap(&snaps, 0, 0, kRow, {kInf, 1.0, 3.0, 4.0});  // row 0 is not finite
    snap(&snaps, 0, 1, {2}, {1.0, 1.0});
    snap(&snaps, 0, 2, kRow, {1.0, 1.0, 1.0, 1.0});
    r97::BoundaryNames names;
    names.add("h", 0, 2);
    const std::vector<Seed> seeds{[] {
        Seed s;
        s.buffer = "h";
        s.row_lo = 0;
        s.row_hi = 2;
        s.cotangent = {1.0f, 1.0f, 1.0f, 1.0f};
        return s;
    }()};

    std::map<int64_t, std::vector<int64_t>> reads{{0, {1}}};
    r97::BackwardOptions bo;
    bo.read_rows = &reads;
    const r97::BackwardResult got = r97::backward_block(records, snaps, names, seeds, {}, bo);

    const r97::OpTiming* t = nullptr;
    for (const r97::OpTiming& x : got.report.ops)
        if (x.op == "rmsnorm") t = &x;
    R97_CHECK(t != nullptr);
    R97_CHECK(t->rows_zeroed_unreadable == 1);
    R97_CHECK(t->rows_readable == 1);
    R97_CHECK(t->rows_refused_non_finite == 0);

    const r97::InputCotangent* x = nullptr;
    for (const auto& in : got.inputs)
        if (in.buffer == "x") x = &in;
    R97_CHECK(x != nullptr);
    R97_CHECK(x->value[0] == 0.0 && x->value[1] == 0.0);  // the dead row, left zero
    R97_CHECK(x->value[2] != 0.0 || x->value[3] != 0.0);  // the live row, differentiated
}

// A non-finite row inside the read set refuses: the row is live and its gradient would be a lie.
R97_TEST(read_set_refuses_a_readable_non_finite_row) {
    std::vector<OpRecord> records{
        op("rmsnorm", {read("x", kRow), read("w", {2}), written("h", kRow)}, 0),
    };
    r97::StoredSnapshots snaps;
    snap(&snaps, 0, 0, kRow, {kInf, 1.0, 3.0, 4.0});
    snap(&snaps, 0, 1, {2}, {1.0, 1.0});
    snap(&snaps, 0, 2, kRow, {1.0, 1.0, 1.0, 1.0});
    r97::BoundaryNames names;
    names.add("h", 0, 2);
    const std::vector<Seed> seeds{[] {
        Seed s;
        s.buffer = "h";
        s.row_lo = 0;
        s.row_hi = 2;
        s.cotangent = {1.0f, 1.0f, 1.0f, 1.0f};
        return s;
    }()};

    std::map<int64_t, std::vector<int64_t>> reads{{0, {0}}};
    r97::BackwardOptions bo;
    bo.read_rows = &reads;
    bool threw = false;
    try {
        (void)r97::backward_block(records, snaps, names, seeds, {}, bo);
    } catch (const std::exception&) {
        threw = true;
    }
    R97_CHECK(threw);
}

}  // namespace
