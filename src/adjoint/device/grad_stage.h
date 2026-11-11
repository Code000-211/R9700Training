// src/adjoint/device/grad_stage.h -- where a step's trainable gradients wait, on the card.
//
// A gradient a device row produces is a sum over every token of every call of one step, and the
// gradient ParamSet it lands in is host memory (include/r97/params.h). Copying each call's sum out
// when the call ends would round the step's partial sums to the bf16 grid once per call and would
// put the leaf on the host for the whole backward walk. So the row adds the call's sum into an f32
// accumulator of this stage on the card, and the stage is copied out once, at the step's end, when
// the walk has walked every token:
//
//   * grad_stage_slot is what a row asks for, by the leaf's name and shape. The first ask of a step
//     allocates and zeroes the accumulator; every later ask, in this call or a later one, is the
//     same f32 buffer, so the card holds the step's running total and no bf16 rounding happens in
//     the accumulation at all;
//   * grad_stage_flush is the step's end: each accumulator is rounded once, from its f32 total, to
//     the bf16 grid a gradient has on the card, and added to the leaf of that name in the step's
//     gradient ParamSet, which is created when the step has not made it yet. The card's staging is
//     freed on the way out;
//   * grad_stage_discard ends a step whose gradients are not wanted.
//
// The walk calls one of the two at the end of every step: the stage's lifetime is the step's, not a
// call's, so a stage that is never flushed is added to by the next step's first call.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "r97/params.h"

namespace r97 {
namespace device {

// The step's f32 accumulator for the leaf `name`, zeroed on the first ask of the step.
// `shape` is the leaf's shape in the gradient ParamSet. nullptr when the card refuses it.
float* grad_stage_slot(const std::string& name, const std::vector<int64_t>& shape);

// The card bytes the step is holding, so a test or a measurement can say what the staging costs.
int64_t grad_stage_bytes();

// Copies the step's accumulators into `grads`, one bf16 rounding each, and frees the staging.
// False when a copy failed; the staging is freed either way, so the step is over.
bool grad_stage_flush(ParamSet* grads);

// The step is over and its gradients are not wanted: the staging is freed, nothing is copied out.
void grad_stage_discard();

}  // namespace device
}  // namespace r97
