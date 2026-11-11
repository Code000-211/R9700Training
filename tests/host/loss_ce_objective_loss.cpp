// tests/host/loss_ce_objective_loss.cpp -- CeObjective's loss is the loss it wraps: the summed
// cross-entropy over the window's scored tokens, in f64, and the count that goes with it. Not the
// mean, not zero, and not scaled by the gradient's 1/total_scored. Label: host.
#include "../support/self_check.h"

#include "../support/vector_tape.h"

#include "r97/loss.h"

#include <cmath>
#include <cstdio>
#include <vector>

using r97::Batch;
using r97::TokenSeq;

// One sequence of six tokens, scored tokens 2, 3 and 3: three scored rows, one unscored row
// between the second and the third (rows 1, 2 and 4 of the five logits rows). Every scored token
// is inside the four-token vocab the tests below pass: a target at or past the vocab reads past
// its logits row, and with NDEBUG the assert in ce_position does not catch it.
static Batch scored_sequence() {
    Batch b;
    b.push_back(TokenSeq{{0, 1, 2, 3, 4, 3}, {0, 0, 1, 1, 0, 1}});
    return b;
}

// Deterministic logits with no ties and a positive loss at every scored position.
static std::vector<float> small_logits(int64_t rows, int64_t vocab) {
    std::vector<float> out(size_t(rows * vocab));
    for (size_t i = 0; i < out.size(); ++i) out[i] = float(double(i) * 0.25 - 3.0);
    return out;
}

// One sequence of two tokens, row {0, 0} predicting token 1: the scored position's cross
// entropy is ln 2, a value this file computes without the loss (p = 1/2).
static void hand_computed() {
    const std::vector<int64_t> shape{1, 2};
    const std::vector<float> logits{0.0f, 0.0f};
    Batch b;
    b.push_back(TokenSeq{{0, 1}, {0, 1}});

    r97test::VectorTape tape;
    // The step's own count is not the window's: if the loss were scaled by the gradient's
    // 1/total_scored, it would be ln 2 / 5 here and not ln 2.
    tape.set_batch(b).set_scored(5).set_buffer("logits", shape, logits);
    const r97::ObjectiveResult got = r97::CeObjective(tape, r97::ParamSet{});

    CHECK(got.scored == 1);
    CHECK(std::fabs(got.loss - std::log(2.0)) <= 1e-15);

    // And it is the sum, not the mean: with one scored token the two coincide, so a second
    // scored token shows the difference. Two rows of zeros, both scored: ln 2 + ln 2.
    const std::vector<int64_t> shape2{2, 2};
    const std::vector<float> logits2{0.0f, 0.0f, 0.0f, 0.0f};
    Batch two;
    two.push_back(TokenSeq{{0, 1, 0}, {0, 1, 1}});
    r97test::VectorTape tape2;
    tape2.set_batch(two).set_scored(2).set_buffer("logits", shape2, logits2);
    const r97::ObjectiveResult got2 = r97::CeObjective(tape2, r97::ParamSet{});
    CHECK(got2.scored == 2);
    CHECK(std::fabs(got2.loss - 2.0 * std::log(2.0)) <= 1e-15);
    std::puts("loss_ce_objective_loss: hand-computed ok");
}

// The whole step in one window: the objective's loss and count are ce_loss's on the same logits,
// bit for bit (same scored rows, same order, same f64 accumulation).
static void whole_call_is_the_loss() {
    const std::vector<int64_t> shape{5, 4};
    const std::vector<float> logits = small_logits(5, 4);
    const r97::CeSum want = r97::ce_loss(scored_sequence(), logits, 4);
    CHECK(want.scored == 3);
    CHECK(want.loss > 0.0);

    r97test::VectorTape tape;
    // total_scored is deliberately not the window's count: the loss is the raw sum, while only
    // the seed's cotangent carries the 1/total_scored scale.
    tape.set_batch(scored_sequence()).set_scored(7).set_buffer("logits", shape, logits);
    const r97::ObjectiveResult got = r97::CeObjective(tape, r97::ParamSet{});

    CHECK(got.scored == want.scored);
    CHECK(got.loss == want.loss);
    // The equality above already pins the sum; these name the two misreadings -- the mean and the
    // gradient's 1/total_scored scale -- directly.
    CHECK(got.loss != want.loss / double(want.scored));
    CHECK(got.loss != want.loss / 7.0);
    std::puts("loss_ce_objective_loss: whole call ok");
}

// Window by window: each window's loss is the sum over exactly the scored rows ce_grad_rows
// names for it, and the windows add up to the whole call.
static void windows_are_the_sum() {
    const std::vector<int64_t> shape{5, 4};
    const std::vector<float> logits = small_logits(5, 4);
    const Batch b = scored_sequence();
    const r97::CeSum whole = r97::ce_loss(b, logits, 4);

    double loss = 0.0;
    int64_t scored = 0;
    // The first window holds no scored row (score[1] is 0), the second two, the third one.
    const int64_t spans[][2] = {{0, 1}, {1, 4}, {4, 5}};
    for (const auto& w : spans) {
        r97test::VectorTape tape;
        tape.set_batch(b).set_scored(3).set_buffer("logits", shape, logits);
        tape.set_window("logits", w[0], w[1]);
        const r97::ObjectiveResult got = r97::CeObjective(tape, r97::ParamSet{});

        // The wrapped function over the same window's rows: the same f64 sum, in the same order.
        const r97::CeSum want =
            r97::ce_sum(logits.data(), 4, r97::ce_grad_rows(b, w[0], w[1]), nullptr, 1.0);
        CHECK(got.scored == want.scored);
        CHECK(got.loss == want.loss);

        loss += got.loss;
        scored += got.scored;
    }
    CHECK(scored == whole.scored);
    CHECK(loss == whole.loss);
    std::puts("loss_ce_objective_loss: windows ok");
}

static void test_loss_ce_objective_loss() {
    hand_computed();
    whole_call_is_the_loss();
    windows_are_the_sum();
    std::puts("loss_ce_objective_loss ok");
}

R97_SELF_CHECK(test_loss_ce_objective_loss)
