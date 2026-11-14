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
    // The gather's own output has no consumer in this tape. That is NOT proof that nothing reads
    // it -- it is only proof that this tape does not show the reader -- so the record stays
    // unproven and every row of it counts as readable. An empty set here would be a vacuous proof,
    // and a vacuous proof zeroes a live value's whole gradient.
    R97_CHECK(reads.find(1) == reads.end());
}

// The case the suite was missing: a record whose window contains no consumer at all is unproven,
// whether the window ended at a rewrite or at the end of the tape. Proven-empty is only reachable
// through a gather that reads nothing, which the index list is checked for separately.
R97_TEST(read_set_with_no_consumer_at_all_is_unproven) {
    const std::vector<OpRecord> ends_at_the_tape{
        op("rmsnorm", {read("x", kRow), read("w", {2}), written("h", kRow)}, 0),
    };
    const std::vector<OpRecord> ends_at_a_rewrite{
        op("rmsnorm", {read("x", kRow), read("w", {2}), written("h", kRow)}, 0),
        op("add", {read("h", kRow), read("z", kRow), written("h", kRow)}, 1),
        op("gather_rows", {read("h", kRow), read("raw:9", kIdx), written("h_out", kOut)}, 2),
    };
    r97::StoredSnapshots snaps;
    snap(&snaps, 2, 1, kIdx, {1.0});
    for (const auto& all : {ends_at_the_tape, ends_at_a_rewrite}) {
        const auto reads = r97::derive_read_sets(all, all, snaps, {});
        R97_CHECK(reads.find(0) == reads.end());
    }
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

// A head walk's externals are the objective's buffer (the logits), never the walk's own input. This
// is the bug the 512 step found: the device path marked the hidden state -- the final norm's
// output -- so the record that writes it could never be proven and the read set never fired (0
// zeroed, 0 readable, 0 refused over 1459 calls), while a forward inf in a row the gather never
// reads reached every gradient. The two paths disagreeing is how it survived, so the marking is one
// function and this pins it.
R97_TEST(read_set_head_externals_are_the_logits_not_the_hidden_state) {
    const std::vector<OpRecord> head{
        op("rmsnorm", {read("x", kRow), read("w", {2}), written("h", kRow)}, 0),
        op("gather_rows", {read("h", kRow), read("raw:1", kIdx), written("h_out", kOut)}, 1),
        op("logits_gemm_vocab", {read("h_out", kOut), read("W", {3, 2}), written("logits", {1, 3})},
           2),
    };
    r97::StoredSnapshots snaps;
    snap(&snaps, 1, 1, kIdx, {1.0});  // the gather selects row 1 of 2
    const std::set<std::string> external = r97::head_external_reads(head[2]);
    R97_CHECK(external == std::set<std::string>{"logits"});
    // With the objective's buffer marked, the norm's writer is proven and row 0 is dead.
    const auto proven = r97::derive_read_sets(head, head, snaps, external);
    const auto norm = proven.find(0);
    R97_CHECK(norm != proven.end());
    R97_CHECK(norm->second == std::vector<int64_t>{1});
    // With the walk's input marked instead -- the bug -- nothing is proven and every row counts as
    // readable, so a non-finite row 0 refuses instead of being zeroed.
    const auto degraded = r97::derive_read_sets(head, head, snaps, {"h"});
    R97_CHECK(degraded.find(0) == degraded.end());
    // A walk that wrote no logits marks nothing: unproven, the fail-safe direction.
    R97_CHECK(r97::head_external_reads(head[0]).empty());
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

// An EMPTY proof on the consumer side is unproven, not dead: every row counts as read, so a
// non-finite row refuses rather than being silently zeroed. This is the invariant's second end --
// the derivation must not produce an empty proof, and no row may act on one if it appears.
R97_TEST(read_set_an_empty_proof_is_unproven) {
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

    std::map<int64_t, std::vector<int64_t>> reads{{0, {}}};  // an empty proof
    r97::BackwardOptions bo;
    bo.read_rows = &reads;
    bool threw = false;
    try {
        (void)r97::backward_block(records, snaps, names, seeds, {}, bo);
    } catch (const std::exception&) {
        threw = true;
    }
    R97_CHECK(threw);  // row 0 counts as read, so it refuses rather than zeroing
}

// A gather that selects no rows is not a proof that nothing is read: a zero-row index list is far
// more likely to be a broken index operand than a real configuration, so the record stays
// unproven and a non-finite row refuses loudly.
R97_TEST(read_set_a_zero_row_gather_is_unproven) {
    const std::vector<OpRecord> all{
        op("rmsnorm", {read("x", kRow), read("w", {2}), written("h", kRow)}, 0),
        op("gather_rows", {read("h", kRow), read("raw:1", {0}), written("h_out", {0, 2})}, 1),
    };
    r97::StoredSnapshots snaps;
    snap(&snaps, 1, 1, {0}, {});
    const auto reads = r97::derive_read_sets(all, all, snaps, {});
    R97_CHECK(reads.find(0) == reads.end());
}

}  // namespace
