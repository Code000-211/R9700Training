// tests/host/loss_ce_tape.cpp -- CeObjective on a VectorTape: exactly one seed, on "logits",
// bit-identical to the gradient function's output, and its read recorded as differentiated.
// The chunked tape (a window) gives the same sum and cotangent rows as the whole call. Label: host.
#include "support/test_reg.h"

#include "support/vector_tape.h"

#include "r97/loss.h"

#include <cassert>
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

// Seed cotangent on the tape == ce_grad rows, bit for bit (both go through ce_position's own
// f64 arithmetic; rows are independent, so windowing cannot change them).
static void check_tape(const std::vector<int64_t>& shape, const std::vector<float>& logits,
                       const r97::ForwardTape& tape, const r97::ObjectiveResult& got) {
    const int64_t vocab = shape[1];
    const r97::Seed& seed = got.seeds.at(0);
    assert(got.seeds.size() == 1);
    assert(seed.buffer == "logits");
    assert(seed.row_lo < seed.row_hi);

    // The same rows, recomputed by the gradient function with the tape's total_scored scale.
    std::vector<float> grad(size_t((seed.row_hi - seed.row_lo) * vocab), 0.0f);
    r97::ce_sum(logits.data(), vocab, r97::ce_grad_rows(two_sequences(), seed.row_lo, seed.row_hi),
                grad.data(), 1.0 / double(tape.total_scored()));
    assert(seed.cotangent.size() == grad.size());
    for (size_t i = 0; i < grad.size(); ++i) assert(seed.cotangent[i] == grad[i]);
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
    assert(a.seeds.size() == 1);
    assert(a.seeds[0].row_lo == 0 && a.seeds[0].row_hi == 5);  // scored rows 0, 2 and 4
    assert(a.scored == 3);
    check_tape(shape, logits, whole, a);

    // The read of logits is recorded as differentiated.
    assert(whole.reads_of("logits").size() == 1);
    assert(whole.reads_of("logits")[0].row_lo == 0 && whole.reads_of("logits")[0].row_hi == 5);
    assert(whole.reads_of("logits")[0].differentiated);

    // Chunked: three windows, same sum, cotangent rows that sum (concatenate, they are disjoint
    // or zero) to the whole call's.
    double loss_sum = 0.0;
    std::vector<float> merged(size_t(15), 0.0f);
    for (auto [lo, hi] : {std::pair<int64_t, int64_t>{0, 2}, {2, 4}, {4, 5}}) {
        r97test::VectorTape tape;
        tape.set_batch(two_sequences()).set_scored(3).set_buffer("logits", shape, logits);
        tape.set_window("logits", lo, hi);
        const r97::ObjectiveResult w = r97::CeObjective(tape, deltas);
        assert(tape.reads_of("logits").size() == 1);
        assert(tape.reads_of("logits")[0].differentiated);
        assert(tape.reads_of("logits")[0].row_lo == lo && tape.reads_of("logits")[0].row_hi == hi);
        if (w.seeds.empty()) continue;  // no scored rows in this window
        loss_sum += w.loss;
        const r97::Seed& s = w.seeds.at(0);
        assert(s.row_lo >= 0 && s.row_hi <= 5);  // scored rows live in [0, 5)
        const int64_t cols = 3;
        for (int64_t r = s.row_lo; r < s.row_hi; ++r)
            for (int64_t c = 0; c < cols; ++c) {
                // Disjoint scored ranges: each row comes from exactly one window.
                float& m = merged[size_t(r * cols + c)];
                assert(m == 0.0f);
                m = s.cotangent[size_t((r - s.row_lo) * cols + c)];
            }
    }
    assert(a.loss == loss_sum);  // row-independent f64 sums over disjoint scored sets
    for (int64_t r = 0; r < 5; ++r)      // every row: bits of the whole call's seed
        for (int64_t c = 0; c < 3; ++c)
            assert(merged[size_t(r * 3 + c)] ==
                   a.seeds[0].cotangent[size_t((r - a.seeds[0].row_lo) * 3 + c)]);

    std::puts("loss_ce_tape ok");
}

R97_REGISTER_TEST("loss_ce_tape", test_loss_ce_tape)
