// tests/host/loss_ce.cpp -- the reference cross-entropy loss: hand-computed cases, the gradient
// against finite differences, a row of very large logits stays finite, and the chunked call
// equals the whole call. Label: host.
#include "support/test_reg.h"

#include "r97/loss.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using r97::Batch;
using r97::ScoredRow;
using r97::TokenSeq;

static bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

// ---------------------------------------------------------------- hand-computed cases
static void hand_computed() {
    // One scored token, two-way softmax. Row {0, 0}, target 0: p = 1/2, loss = ln 2.
    Batch b;
    b.push_back(TokenSeq{{0, 1}, {0, 1}});
    {
        const float row[2] = {0.0f, 0.0f};
        const r97::CeSum s = r97::ce_loss(b, row, 2);
        assert(s.scored == 1);
        assert(near(s.loss, std::log(2.0), 1e-15));
    }
    // Row {ln 2, 0}, target 0: p = 2/3, loss = ln(3/2).
    {
        const float row[2] = {std::log(2.0f), 0.0f};
        const r97::CeSum s = r97::ce_loss(b, row, 2);
        assert(near(s.loss, std::log(3.0), 1e-6));  // target is ids[1] = 1, logit 0: p = 1/3
    }
    // Two sequences concatenated: rows 0..2 predict tokens 1..3 of the first, rows 3..4 tokens
    // 1..2 of the second. The scored token 7 of the second sequence hangs off row 4.
    Batch two;
    two.push_back(TokenSeq{{0, 1, 2, 1}, {0, 1, 0, 1}});
    two.push_back(TokenSeq{{2, 1, 0}, {0, 0, 1}});
    const float rows5[5][3] = {
        {0, 1, 2},    // row 0 (sequence 1, position 0)
        {3, 4, 5},    // row 1
        {6, 7, 8},    // row 2 -> token ids[3] = 1 (scored)
        {9, 10, 11},  // row 3 (sequence 2, position 0)
        {12, 13, 14}, // row 4 -> token ids[2] = 0 (scored)
    };
    std::vector<float> flat;
    for (auto& r : rows5) flat.insert(flat.end(), std::begin(r), std::end(r));
    {
        const r97::CeSum s = r97::ce_loss(two, flat, 3);
        double want = 0.0;
        want += r97::ce_position(rows5[0], 3, 1);  // seq 1, token ids[1] = 1 (scored)
        want += r97::ce_position(rows5[2], 3, 1);  // seq 1, token ids[3] = 1 (scored)
        want += r97::ce_position(rows5[4], 3, 0);  // seq 2, token ids[2] = 0 (scored)
        assert(s.scored == 3);
        assert(near(s.loss, want, 1e-15));
    }
    std::puts("loss_ce: hand-computed ok");
}

// ---------------------------------------------------------------- gradient vs finite differences
static void grad_finite_differences() {
    // A batch of two sequences, several scored tokens, deterministic pseudo-logits. Ids fit the
    // toy vocab of 4: scored rows are 0 (token 1), 2 (token 2), 3 and 4 (seq 2).
    Batch b;
    b.push_back(TokenSeq{{0, 1, 3, 2}, {0, 1, 0, 1}});
    b.push_back(TokenSeq{{3, 2, 1, 0}, {0, 1, 1, 0}});
    const int64_t vocab = 4, rows = 6;
    std::vector<float> logits(rows * vocab);
    for (size_t i = 0; i < logits.size(); ++i)
        logits[i] = float(std::sin(double(i) * 0.37) * 1.7 + double(i % vocab) * 0.21);

    // Mean loss: sum / scored (scored = 4).
    const int64_t scored = 4;
    auto mean_loss = [&](const std::vector<float>& x) {
        return r97::ce_loss(b, x, vocab).loss / double(scored);
    };

    std::vector<float> grad(size_t(rows * vocab), 0.0f);
    r97::ce_grad(b, logits.data(), vocab, 1.0 / double(scored), grad.data());

    // Central differences on every entry of a few rows, scored or not.
    for (int64_t row = 0; row < rows; ++row)
        for (int64_t col = 0; col < vocab; ++col) {
            std::vector<float> xp = logits, xm = logits;
            const double h = 1e-2;
            xp[size_t(row * vocab + col)] += float(h);
            xm[size_t(row * vocab + col)] -= float(h);
            const double fd = (mean_loss(xp) - mean_loss(xm)) / (2.0 * h);
            const double g = grad[size_t(row * vocab + col)];
            assert(near(g, fd, 1e-4));
        }
    // Unscored rows are exactly zero.
    assert(grad[1 * vocab + 1] == 0.0f);  // row 1 predicts token 2, score 0
    assert(grad[5 * vocab + 2] == 0.0f);  // row 5 predicts nothing scored
    std::puts("loss_ce: gradient vs finite differences ok");
}

// ---------------------------------------------------------------- very large logits stay finite
static void large_logits() {
    Batch b;
    b.push_back(TokenSeq{{0, 1}, {0, 1}});
    // 1e30 dwarfs 5: softmax is {~1/2, ~1/2, ~0}... vocab 3, target 0: loss = ln 2, finite.
    const float row[3] = {1e30f, 1e30f, 5.0f};
    float grad[3] = {0, 0, 0};
    const double loss = r97::ce_position(row, 3, 0, grad);
    assert(std::isfinite(loss) && near(loss, std::log(2.0), 1e-6));
    assert(near(grad[0], -0.5, 1e-6) && near(grad[1], 0.5, 1e-6) && near(grad[2], 0.0, 1e-6));
    const r97::CeSum s = r97::ce_loss(b, row, 3);
    assert(std::isfinite(s.loss));
    std::puts("loss_ce: large logits ok");
}

// ---------------------------------------------------------------- chunked equals whole
static void chunked_equals_whole() {
    Batch b;
    b.push_back(TokenSeq{{0, 1, 2, 1}, {0, 1, 0, 1}});
    b.push_back(TokenSeq{{2, 1, 0}, {0, 0, 1}});
    const int64_t vocab = 3, rows = 5;
    std::vector<float> logits{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};

    // Whole call.
    const r97::CeSum whole = r97::ce_loss(b, logits, vocab);
    std::vector<float> gw(size_t(rows * vocab), 0.0f);
    r97::ce_grad(b, logits.data(), vocab, 1.0 / double(whole.scored), gw.data());

    // Chunked by windows of rows. Rows outside the scored set stay zero in both.
    double loss_sum = 0.0;
    std::vector<float> gc(size_t(rows * vocab), 0.0f);
    for (auto [lo, hi] : {std::pair<int64_t, int64_t>{0, 2}, {2, 4}, {4, 5}}) {
        const std::vector<r97::ScoredRow> in = r97::ce_grad_rows(b, lo, hi);
        const r97::CeSum s = r97::ce_sum(logits.data(), vocab, in, gc.data(),
                                         1.0 / double(whole.scored));
        loss_sum += s.loss;
    }
    assert(near(loss_sum, whole.loss, 1e-12 * std::max(1.0, std::fabs(whole.loss))));
    for (size_t i = 0; i < gw.size(); ++i) assert(gw[i] == gc[i]);  // rows are independent
    std::puts("loss_ce: chunked equals whole ok");
}

// ---------------------------------------------------------------- bf16 input
static void bf16_matches_f32() {
    Batch b;
    b.push_back(TokenSeq{{0, 1}, {0, 1}});
    const float row[2] = {0.25f, -1.0f};
    const uint16_t bf[2] = {0x3e80, 0xbf80};  // bf16 of 0.25 and -1.0
    const r97::CeSum sf = r97::ce_loss(b, row, 2);
    const r97::CeSum sb = r97::ce_loss(b, std::vector<uint16_t>(bf, bf + 2), 2);
    assert(sf.loss == sb.loss && sb.scored == 1);
    std::puts("loss_ce: bf16 input ok");
}

static void test_loss_ce() {
    hand_computed();
    grad_finite_differences();
    large_logits();
    chunked_equals_whole();
    bf16_matches_f32();
    std::puts("loss_ce ok");
}

R97_REGISTER_TEST("loss_ce", test_loss_ce)
