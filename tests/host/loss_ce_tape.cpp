// tests/host/loss_ce_tape.cpp -- CeObjective on a VectorTape: exactly one seed, on "logits",
// over exactly the window's scored rows, bit-identical to the gradient function's output, and its
// read recorded as differentiated. The chunked tape (a window) gives the same sum and cotangent
// rows as the whole call. Label: host.
#include "support/self_check.h"

#include "support/vector_tape.h"

#include "r97/loss.h"

#include <cstdio>
#include <vector>

using r97::Batch;
using r97::TokenSeq;

static Batch two_sequences() {
    // Ids fit the toy vocab of 3; scored rows are 0, 2 (token 1) and 4 (token 0).
    Batch b;
    b.push_back(TokenSeq{{0, 1, 2, 1}, {0, 1, 0, 1}});
    b.push_back(TokenSeq{{2, 1, 0}, {0, 0, 1}});
    return b;
}

// One sequence of six tokens, scored tokens 3 and 4 only: the five logits rows hold the scored
// rows 2 and 3, so neither end of a window [0, 5) is a scored row.
static Batch interior_sequence() {
    Batch b;
    b.push_back(TokenSeq{{0, 1, 2, 3, 4, 5}, {0, 0, 0, 1, 1, 0}});
    return b;
}

// One sequence of six tokens, scored tokens 3 and 5 only: the scored rows 2 and 4 of the five
// logits rows have the unscored row 3 between them.
static Batch gapped_sequence() {
    Batch b;
    b.push_back(TokenSeq{{0, 1, 2, 3, 4, 5}, {0, 0, 0, 1, 0, 1}});
    return b;
}

// Deterministic logits with no ties, so every cotangent entry is distinguishable.
static std::vector<float> small_logits(int64_t rows, int64_t vocab) {
    std::vector<float> out(size_t(rows * vocab));
    for (size_t i = 0; i < out.size(); ++i) out[i] = float(double(i) * 0.25 - 3.0);
    return out;
}

// Seed cotangent on the tape == ce_grad rows, bit for bit (both go through ce_position's own
// f64 arithmetic; rows are independent, so windowing cannot change them).
static void check_tape(const Batch& batch, const std::vector<int64_t>& shape,
                       const std::vector<float>& logits, const r97::ForwardTape& tape,
                       const r97::ObjectiveResult& got) {
    const int64_t vocab = shape[1];
    CHECK(got.seeds.size() == 1);
    const r97::Seed& seed = got.seeds.at(0);
    CHECK(seed.buffer == "logits");
    CHECK(seed.row_lo < seed.row_hi);

    // The same rows, recomputed by the gradient function with the tape's total_scored scale.
    std::vector<float> grad(size_t((seed.row_hi - seed.row_lo) * vocab), 0.0f);
    r97::ce_sum(logits.data(), vocab, r97::ce_grad_rows(batch, seed.row_lo, seed.row_hi),
                grad.data(), 1.0 / double(tape.total_scored()));
    CHECK(seed.cotangent.size() == grad.size());
    for (size_t i = 0; i < grad.size(); ++i) CHECK(seed.cotangent[i] == grad[i]);
}

static void test_loss_ce_tape() {
    const std::vector<int64_t> shape{5, 3};
    std::vector<float> logits{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};

    // The whole step, one window.
    r97::ParamSet deltas;
    r97test::VectorTape whole;
    whole.set_batch(two_sequences()).set_scored(3).set_buffer("logits", shape, logits);
    const r97::ObjectiveResult a = r97::CeObjective(whole, deltas);

    // One seed, on logits, over the scored rows (2 through 4); zeros nowhere needed here.
    CHECK(a.seeds.size() == 1);
    CHECK(a.seeds[0].row_lo == 0 && a.seeds[0].row_hi == 5);  // scored rows 0, 2 and 4
    CHECK(a.scored == 3);
    check_tape(two_sequences(), shape, logits, whole, a);

    // The read of logits is recorded as differentiated.
    CHECK(whole.reads_of("logits").size() == 1);
    CHECK(whole.reads_of("logits")[0].row_lo == 0 && whole.reads_of("logits")[0].row_hi == 5);
    CHECK(whole.reads_of("logits")[0].differentiated);

    // Chunked: three windows, same sum, cotangent rows that sum (concatenate, they are disjoint
    // or zero) to the whole call's.
    double loss_sum = 0.0;
    std::vector<float> merged(size_t(15), 0.0f);
    for (auto [lo, hi] : {std::pair<int64_t, int64_t>{0, 2}, {2, 4}, {4, 5}}) {
        r97test::VectorTape tape;
        tape.set_batch(two_sequences()).set_scored(3).set_buffer("logits", shape, logits);
        tape.set_window("logits", lo, hi);
        const r97::ObjectiveResult w = r97::CeObjective(tape, deltas);
        CHECK(tape.reads_of("logits").size() == 1);
        CHECK(tape.reads_of("logits")[0].differentiated);
        CHECK(tape.reads_of("logits")[0].row_lo == lo && tape.reads_of("logits")[0].row_hi == hi);
        if (w.seeds.empty()) continue;  // no scored rows in this window
        loss_sum += w.loss;
        const r97::Seed& s = w.seeds.at(0);
        CHECK(s.row_lo >= 0 && s.row_hi <= 5);  // scored rows live in [0, 5)
        const int64_t cols = 3;
        for (int64_t r = s.row_lo; r < s.row_hi; ++r)
            for (int64_t c = 0; c < cols; ++c) {
                // Disjoint scored ranges: each row comes from exactly one window.
                float& m = merged[size_t(r * cols + c)];
                CHECK(m == 0.0f);
                m = s.cotangent[size_t((r - s.row_lo) * cols + c)];
            }
    }
    CHECK(a.loss == loss_sum);  // row-independent f64 sums over disjoint scored sets
    for (int64_t r = 0; r < 5; ++r)      // every row: bits of the whole call's seed
        for (int64_t c = 0; c < 3; ++c)
            CHECK(merged[size_t(r * 3 + c)] ==
                  a.seeds[0].cotangent[size_t((r - a.seeds[0].row_lo) * 3 + c)]);

    std::puts("loss_ce_tape ok");
}

// One window of `batch`'s logits through CeObjective: a fresh tape, so the read record is this
// window's alone.
static r97::ObjectiveResult window_of(const Batch& batch, int64_t scored,
                                      const std::vector<int64_t>& shape,
                                      const std::vector<float>& logits, int64_t lo, int64_t hi) {
    r97test::VectorTape tape;
    tape.set_batch(batch).set_scored(scored).set_buffer("logits", shape, logits);
    tape.set_window("logits", lo, hi);
    return r97::CeObjective(tape, r97::ParamSet{});
}

// ---------------------------------------------------------------- the seed range
// The seed covers exactly the window's scored rows: the first of them through the last, in the
// buffer's own rows, so a caller merges window seeds by row. The scored rows here are 2 and 3 of
// a window [0, 5), which neither "the whole window" [0, 5) nor "the window's local rows" [0, 4)
// can be.
static void seed_range_is_the_scored_rows() {
    const std::vector<int64_t> shape{5, 6};
    const std::vector<float> logits = small_logits(5, 6);
    const int64_t vocab = 6;

    // The whole step, one window.
    r97::ParamSet deltas;
    r97test::VectorTape whole_tape;
    whole_tape.set_batch(interior_sequence()).set_scored(2).set_buffer("logits", shape, logits);
    const r97::ObjectiveResult whole = r97::CeObjective(whole_tape, deltas);
    CHECK(whole.seeds.size() == 1);
    CHECK(whole.scored == 2);
    CHECK(whole.seeds[0].buffer == "logits");
    CHECK(whole.seeds[0].row_lo == 2 && whole.seeds[0].row_hi == 4);
    CHECK(whole.seeds[0].cotangent.size() ==
          size_t((whole.seeds[0].row_hi - whole.seeds[0].row_lo) * vocab));

    // Bit-identical to the gradient function's rows.
    std::vector<float> grad(size_t(5 * vocab), 0.0f);
    r97::ce_grad(interior_sequence(), logits.data(), vocab, 1.0 / 2.0, grad.data());
    for (int64_t r = whole.seeds[0].row_lo; r < whole.seeds[0].row_hi; ++r)
        for (int64_t c = 0; c < vocab; ++c)
            CHECK(whole.seeds[0].cotangent[size_t((r - whole.seeds[0].row_lo) * vocab + c)] ==
                  grad[size_t(r * vocab + c)]);

    // A window whose own ends are unscored: [1, 5) keeps both scored rows, [0, 3) only row 2 and
    // [3, 5) only row 3. A seed over the window, or in local rows, is either too wide or off.
    const int64_t windows[3][4] = {
        {1, 5, 2, 4},  // unscored rows 1 and 4 around the scored ones
        {0, 3, 2, 3},  // scored row 2, unscored rows 0 and 1 before it
        {3, 5, 3, 4},  // scored row 3, scored row 2 outside the window
    };
    for (const auto& w : windows) {
        const r97::ObjectiveResult got =
            window_of(interior_sequence(), 2, shape, logits, w[0], w[1]);
        CHECK(got.seeds.size() == 1);
        CHECK(got.seeds[0].buffer == "logits");
        CHECK(got.seeds[0].row_lo == w[2] && got.seeds[0].row_hi == w[3]);
        CHECK(got.seeds[0].cotangent.size() == size_t((w[3] - w[2]) * vocab));
        for (int64_t r = w[2]; r < w[3]; ++r)
            for (int64_t c = 0; c < vocab; ++c)
                CHECK(got.seeds[0].cotangent[size_t((r - w[2]) * vocab + c)] ==
                      grad[size_t(r * vocab + c)]);
    }

    // A window with no scored row has no seed and no loss -- and is not read at all: a
    // differentiated read must be answered by a seed, so reading rows the loss does not depend on
    // would leave the tape with a read the backward cannot start from. Chunked by position, a
    // walk lands on such windows routinely (score[0] == 0, assistant-only score masks).
    {
        r97test::VectorTape tape;
        tape.set_batch(interior_sequence()).set_scored(2).set_buffer("logits", shape, logits);
        tape.set_window("logits", 0, 2);  // the scored rows of this batch are 2 and 3
        const r97::ObjectiveResult got = r97::CeObjective(tape, r97::ParamSet{});
        CHECK(got.seeds.empty() && got.loss == 0.0 && got.scored == 0);
        CHECK(tape.reads_of("logits").empty());  // no read, so no read left unseeded
    }

    // Window by window: the seeds are the scored rows, disjoint, and concatenate to the whole
    // call's cotangent bit for bit, with the same sum.
    {
        double loss = 0.0;
        std::vector<float> merged(size_t(5 * vocab), 0.0f);
        // A walk of three windows, the first of them holding no scored row: it adds nothing and
        // seeds nothing. The seed rows each of the other two must produce are in the last two
        // columns.
        const int64_t spans[3][4] = {{0, 2, 0, 0}, {2, 3, 2, 3}, {3, 5, 3, 4}};
        for (const auto& w : spans) {
            const r97::ObjectiveResult got =
                window_of(interior_sequence(), 2, shape, logits, w[0], w[1]);
            loss += got.loss;
            if (w[2] == w[3]) {  // no scored row in the window: nothing to seed
                CHECK(got.seeds.empty() && got.loss == 0.0 && got.scored == 0);
                continue;
            }
            CHECK(got.seeds.size() == 1);
            const r97::Seed& s = got.seeds[0];
            CHECK(s.row_lo == w[2] && s.row_hi == w[3]);
            for (int64_t r = s.row_lo; r < s.row_hi; ++r)
                for (int64_t c = 0; c < vocab; ++c) {
                    float& m = merged[size_t(r * vocab + c)];
                    CHECK(m == 0.0f);  // disjoint windows: no row twice
                    m = s.cotangent[size_t((r - s.row_lo) * vocab + c)];
                }
        }
        CHECK(loss == whole.loss);
        // ce_grad over every row: what the concatenated seeds must hold, bit for bit.
        std::vector<float> all(size_t(5 * vocab), 0.0f);
        r97::ce_grad(interior_sequence(), logits.data(), vocab, 1.0 / 2.0, all.data());
        for (size_t i = 0; i < merged.size(); ++i) CHECK(merged[i] == all[i]);
    }

    // Scored rows with an unscored row between them: the seed spans them, with a zero row inside
    // (the merge above relies on that row being exactly zero).
    {
        const r97::ObjectiveResult got =
            window_of(gapped_sequence(), 2, shape, logits, 0, 5);
        CHECK(got.seeds.size() == 1);
        CHECK(got.seeds[0].row_lo == 2 && got.seeds[0].row_hi == 5);
        CHECK(got.scored == 2);
        for (int64_t c = 0; c < vocab; ++c)
            CHECK(got.seeds[0].cotangent[size_t((3 - 2) * vocab + c)] == 0.0f);  // row 3 unscored
        std::vector<float> gapped(size_t(5 * vocab), 0.0f);
        r97::ce_grad(gapped_sequence(), logits.data(), vocab, 1.0 / 2.0, gapped.data());
        for (int64_t r = 2; r < 5; ++r)
            for (int64_t c = 0; c < vocab; ++c)
                CHECK(got.seeds[0].cotangent[size_t((r - 2) * vocab + c)] ==
                      gapped[size_t(r * vocab + c)]);
    }

    std::puts("loss_ce_tape: seed range ok");
}

R97_SELF_CHECK(test_loss_ce_tape)
R97_SELF_CHECK(seed_range_is_the_scored_rows)
