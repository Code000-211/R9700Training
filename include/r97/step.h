// r97/step.h -- one training step's shape.
//
// The forward driver and the joins take a Model and give a StepFn. Neither names a loss (the
// Objective owns it) nor an adapter kind (the Delta owns that).
#pragma once

#include <cstdint>
#include <functional>
#include <memory>

#include "r97/batch.h"
#include "r97/delta.h"
#include "r97/objective.h"
#include "r97/params.h"

namespace r97 {

struct StepResult {
    double loss;      // the sum over the batch's windows, as the reference loss defines it
    int64_t scored;   // how many scored tokens the loss covered
};

using StepFn = std::function<StepResult(const Batch&, const ParamSet& params, ParamSet* grads)>;
// grads == nullptr means forward only.

struct Model {
    Objective objective;          // against what is trained: the loss and where it seeds the tape
    std::shared_ptr<Delta> delta; // what is trained
};

}  // namespace r97
