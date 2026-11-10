// src/loss.cpp -- CeObjective, the reference cross-entropy loss. See include/r97/loss.h and
// docs/losses/ce-objective.md. Own file: a new loss is a new Objective in its own file, never a
// branch in the forward driver or the backward core.
#include "r97/loss.h"

#include <cassert>
#include <cmath>
#include <limits>

namespace r97 {

double logsumexp(const float* row, int64_t n) {
    double m = -std::numeric_limits<double>::infinity();
    for (int64_t j = 0; j < n; ++j) m = std::max(m, double(row[j]));
    if (m == -std::numeric_limits<double>::infinity()) return m;
    double s = 0.0;
    for (int64_t j = 0; j < n; ++j) s += std::exp(double(row[j]) - m);
    return m + std::log(s);
}

double ce_position(const float* row, int64_t vocab, int32_t target, float* grad,
                   double grad_scale) {
    assert(target >= 0 && target < vocab);
    // Stable in two pieces: subtract the max first (exact for the max entries), then the log of
    // the sum. Adding m + log(s) instead would round log(s) away once m reaches ~1e30.
    double m = -std::numeric_limits<double>::infinity();
    for (int64_t j = 0; j < vocab; ++j) m = std::max(m, double(row[j]));
    double s = 0.0;
    for (int64_t j = 0; j < vocab; ++j) s += std::exp(double(row[j]) - m);
    const double ls = std::log(s);
    if (grad) {
        for (int64_t j = 0; j < vocab; ++j) {
            double g = std::exp((double(row[j]) - m) - ls) * grad_scale;  // softmax * scale
            if (j == target) g -= grad_scale;                             // minus one-hot
            grad[j] = float(g);
        }
    }
    return ls - (double(row[target]) - m);  // logsumexp(row) - row[target]
}

CeSum ce_sum(const float* logits, int64_t vocab, const std::vector<ScoredRow>& rows, float* grad,
             double grad_scale) {
    CeSum out;
    for (const ScoredRow& r : rows) {
        assert(r.id >= 0 && r.id < vocab);
        const float* row = logits + r.row * vocab;
        float* g = grad ? grad + r.row * vocab : nullptr;
        out.loss += ce_position(row, vocab, r.id, g, grad_scale);
        ++out.scored;
    }
    return out;
}

// Scored tokens of the batch whose prediction row lands in [lo, hi), as (row, id); `off` walks
// the concatenated rows, L-1 per sequence. `hi < 0` means "no window: all of them".
static std::vector<ScoredRow> scored_rows(const Batch& batch, int64_t lo, int64_t hi) {
    std::vector<ScoredRow> out;
    int64_t off = 0;
    for (const TokenSeq& s : batch) {
        const int64_t len = int64_t(s.ids.size());
        assert(s.score.size() == s.ids.size());
        for (int64_t i = 1; i < len; ++i) {
            if (s.score[i] != 1) continue;
            const int64_t row = off + i - 1;  // token i is predicted at position i - 1
            if (hi >= 0 && (row < lo || row >= hi)) continue;
            out.push_back(ScoredRow{row, s.ids[i]});
        }
        off += len - 1;
    }
    return out;
}

std::vector<ScoredRow> ce_grad_rows(const Batch& batch, int64_t lo, int64_t hi) {
    return scored_rows(batch, lo, hi);
}

static int64_t batch_rows(const Batch& batch) {
    int64_t rows = 0;
    for (const TokenSeq& s : batch) rows += int64_t(s.ids.size()) - 1;
    return rows;
}

CeSum ce_loss(const Batch& batch, const float* logits, int64_t vocab) {
    return ce_sum(logits, vocab, scored_rows(batch, 0, -1));
}

CeSum ce_loss(const Batch& batch, const std::vector<float>& logits, int64_t vocab) {
    assert(logits.size() == size_t(batch_rows(batch) * vocab));
    return ce_loss(batch, logits.data(), vocab);
}

CeSum ce_loss(const Batch& batch, const std::vector<uint16_t>& bf16_logits, int64_t vocab) {
    const int64_t rows = batch_rows(batch);
    assert(bf16_logits.size() == size_t(rows * vocab));
    std::vector<float> f32(bf16_logits.size());
    for (size_t i = 0; i < bf16_logits.size(); ++i) f32[i] = bf16_to_f32(bf16_logits[i]);
    return ce_loss(batch, f32.data(), vocab);
}

void ce_grad(const Batch& batch, const float* logits, int64_t vocab, double grad_scale,
             float* out) {
    const std::vector<ScoredRow> rows = scored_rows(batch, 0, -1);
    ce_sum(logits, vocab, rows, out, grad_scale);
}

ObjectiveResult CeObjective(const ForwardTape& fwd, const ParamSet&) {
    ObjectiveResult out;

    const std::vector<int64_t> shape = fwd.shape("logits");
    assert(shape.size() == 2);
    const int64_t vocab = shape[1];

    int64_t lo = 0, hi = 0;
    fwd.window("logits", &lo, &hi);
    if (lo >= hi) return out;

    // A differentiated read: the loss depends on these rows, so the backward gets a Seed on them.
    const std::vector<float> data = fwd.read("logits", lo, hi, /*differentiated=*/true);
    const std::vector<ScoredRow> rows = scored_rows(fwd.batch(), lo, hi);
    if (rows.empty()) return out;

    const double scale = fwd.total_scored() > 0 ? 1.0 / double(fwd.total_scored()) : 0.0;
    const int64_t span = hi - lo;
    std::vector<float> grad(size_t(span) * size_t(vocab), 0.0f);
    // Local rows: ce_sum indexes the buffer it reads, which here starts at window row lo.
    std::vector<ScoredRow> local = rows;
    for (ScoredRow& r : local) r.row -= lo;
    const CeSum s = ce_sum(data.data(), vocab, local, grad.data(), scale);

    // One seed over the window's scored rows, first through last; unscored rows inside stay 0.
    int64_t seed_lo = local.front().row, seed_hi = local.back().row + 1;
    Seed seed;
    seed.buffer = "logits";
    seed.row_lo = lo + seed_lo;
    seed.row_hi = lo + seed_hi;
    seed.cotangent.assign(grad.begin() + seed_lo * vocab, grad.begin() + seed_hi * vocab);

    out.loss = s.loss;
    out.scored = s.scored;
    out.seeds.push_back(std::move(seed));
    return out;
}

}  // namespace r97
