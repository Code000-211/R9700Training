// tests/support/sum_objective.h -- a test objective: a sum of the chosen rows of a chosen
// boundary, seeded back onto it. Registered by name from the test binaries only, by
// tests/host/test_registrations.cpp.
//
// The objective reads rows [lo, hi) of `buffer` as differentiated and, when the buffer has one,
// the next row as detached, so a test can see both kinds of read in the tape's record. Its seed
// covers [lo, hi) and its loss is the sum of the differentiated rows' values.
#pragma once

#include "r97/objective.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace r97test {

inline r97::ObjectiveResult sum_objective(const r97::ForwardTape& fwd, const r97::ParamSet&) {
    const std::string buffer = "logits";   // the boundary it seeds
    const int64_t lo = 0, hi = 1;          // the rows it sums, in the window the test sets

    r97::ObjectiveResult result;
    std::vector<float> rows = fwd.read(buffer, lo, hi, true);
    // The detached read of the next row, when the window has one: a test can see both kinds of
    // read in the tape's record, and the detached row is not part of the loss.
    if (hi + 1 <= fwd.shape(buffer)[0]) (void)fwd.read(buffer, hi, hi + 1, false);
    result.loss = std::accumulate(rows.begin(), rows.end(), 0.0);
    result.scored = hi - lo;

    std::vector<int64_t> shape = fwd.shape(buffer);
    int64_t cols = 1;
    for (size_t i = 1; i < shape.size(); ++i) cols *= shape[i];
    // The cotangent of a sum: ones over the rows it read.
    result.seeds.push_back(r97::Seed{
        buffer, lo, hi,
        std::vector<float>(static_cast<size_t>((hi - lo) * cols), 1.0f)});
    return result;
}

inline r97::Objective make_sum_objective(const r97::Config&) {
    return r97::Objective(sum_objective);
}

}  // namespace r97test
