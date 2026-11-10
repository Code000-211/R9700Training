// r97/loss.h -- the reference loss: cross-entropy over scored tokens, in f64.
//
// The loss every later issue is checked against, as the first Objective (include/r97/objective.h).
// It is called CeObjective; nothing else in the engine defines it. Docs: docs/losses/ce-objective.md.
//
// One sequence of L tokens has L-1 logits rows (row p predicts token p+1); a batch's rows are its
// sequences concatenated, in order. Token i with score[i] == 1 is predicted by row i-1 of its
// sequence. The loss is the sum over scored tokens of -log softmax(row)[id], accumulated in f64
// with a stable log-sum-exp; the reported loss is sum/count.
//
// The position-by-position core (logsumexp / ce_position) lets a caller feed chunks of rows
// without holding every logit row. Rows may be bf16 or f32: the f32 form is the core, bf16
// callers convert with bf16_to_f32 (a ForwardTape's read already returns f32).
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "r97/batch.h"
#include "r97/objective.h"
#include "r97/params.h"

namespace r97 {

// A scored prediction: logits row `row` (local to the buffer ce_sum reads) predicts token `id`.
struct ScoredRow {
    int64_t row = 0;
    int32_t id = 0;
};

// bf16 bits (memory order: bf16_to_f32 of the two bytes little-endian) to f32.
inline float bf16_to_f32(uint16_t v) {
    const uint32_t u = uint32_t(v) << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

// Stable log-sum-exp of one row, in f64: max + log(sum(exp(x - max))).
double logsumexp(const float* row, int64_t n);

// The cross entropy of one scored position in f64: logsumexp(row) - row[target].
// If grad is not null, *grad (vocab entries) is softmax(row), minus one at target, times
// grad_scale, in f64 then cast to f32 -- the gradient of this position's share of
// (grad_scale * the summed loss) with respect to the row.
double ce_position(const float* row, int64_t vocab, int32_t target, float* grad = nullptr,
                   double grad_scale = 1.0);

// sum and count of one ce computation.
struct CeSum {
    double loss = 0.0;
    int64_t scored = 0;
};

// The sum over `rows` of the positions of `logits` (row-major, exactly the rows the
// ScoredRows index). If grad is not null it must hold a zeroed buffer of the same footprint;
// only the scored rows are written (softmax minus one-hot, times grad_scale), the rest stay 0.
CeSum ce_sum(const float* logits, int64_t vocab, const std::vector<ScoredRow>& rows,
             float* grad = nullptr, double grad_scale = 1.0);

// The batch loss: the sum over all scored tokens of the batch. The rows of `logits` are the
// batch's sequences concatenated (L-1 rows for a sequence of L tokens). The std::vector overloads
// require exactly that many rows.
CeSum ce_loss(const Batch& batch, const float* logits, int64_t vocab);
CeSum ce_loss(const Batch& batch, const std::vector<float>& logits, int64_t vocab);
CeSum ce_loss(const Batch& batch, const std::vector<uint16_t>& bf16_logits, int64_t vocab);

// The gradient of the summed loss scaled by `grad_scale` (pass 1.0/scored for the mean) with
// respect to every logits row: softmax minus one-hot on the scored rows, zero elsewhere.
// `out` must hold a zeroed buffer of rows*vocab; rows as in ce_loss.
std::vector<ScoredRow> ce_grad_rows(const Batch& batch, int64_t lo, int64_t hi);
void ce_grad(const Batch& batch, const float* logits, int64_t vocab, double grad_scale, float* out);

// The Objective: reads the boundary "logits" in the window the tape gives (a differentiated
// read), takes ids and score mask from fwd.batch(), and returns the window's summed loss, its
// scored count and one Seed on "logits" over the window's scored rows -- contiguous from the
// first to the last scored row, in the buffer's own rows (the ones window/read count in), with
// zeros on unscored rows inside -- whose cotangent is the gradient above scaled by
// 1/fwd.total_scored(). Called window by window it gives the same sum and the same cotangent
// rows as called once.
ObjectiveResult CeObjective(const ForwardTape& fwd, const ParamSet& deltas);

}  // namespace r97
