// tests/support/reader_objective.h -- a test objective that reads a boundary and seeds nothing.
//
// It exists to make the reciprocity rule check_seeds enforces testable through a real Objective,
// not through a hand-built result: the differentiated variant reads `logits` as differentiated
// and returns no seed (refused), the detached variant makes the same read detached (passes).
// Both are forward-only objectives; neither is registered, so a test calls its factory directly
// the way the forward driver would.
#pragma once

#include "r97/config.h"
#include "r97/objective.h"

#include <cstddef>
#include <vector>

namespace r97test {

inline r97::ObjectiveResult reading_objective(const r97::ForwardTape& fwd, bool differentiated) {
    const std::vector<float> rows = fwd.read("logits", 0, 1, differentiated);
    r97::ObjectiveResult result;
    result.loss = rows.empty() ? 0.0 : static_cast<double>(rows.front());
    result.scored = 1;
    return result;   // no seeds: this objective never claims a gradient
}

inline r97::Objective make_differentiated_reader(const r97::Config&) {
    return [](const r97::ForwardTape& fwd, const r97::ParamSet&) {
        return reading_objective(fwd, true);
    };
}

inline r97::Objective make_detached_reader(const r97::Config&) {
    return [](const r97::ForwardTape& fwd, const r97::ParamSet&) {
        return reading_objective(fwd, false);
    };
}

}  // namespace r97test
