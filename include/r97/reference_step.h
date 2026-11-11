// r97/reference_step.h -- one training step on the host, from the twins alone.
//
// The reference step runs the tiny fixture's recorded op graph in f64. The forward comes from the
// registered twins, the backward from the registered host adjoints, and the weights from the
// container through rad_core's reader: no radiance kernel computes a value and no card is used.
//
// `program` names an op list written by `r97train ops --record FILE`; with it the step runs no
// `rad_issue` at all. Empty, the plugin's host kernel chain runs once per batch and only its
// OpRecords are kept -- every value the twins compute is then a function of the Batch and the
// weights, not of the kernel output.
//
// The step exists so the training loop, the checkpoint, the batcher and the resume are exercised
// on a real graph shape on a CPU. It is not the int8 path and it is not the kernels.
#pragma once

#include <string>
#include <vector>

#include "r97/model.h"
#include "r97/step.h"

namespace r97 {

struct ReferenceStepOptions {
    std::string container;                 // the .rad the weights are read from
    std::string radiance_home;             // the plugin home Config takes (live trace only)
    std::vector<std::string> kernels;      // kernel hierarchy for the live trace
    std::string program;                   // recorded op list; empty = trace live
};

// The step. `model` supplies the objective and the trainable kind; `opts.container` supplies the
// weights. Throws std::runtime_error when the program or the container cannot be read.
StepFn make_reference_step(Model model, ReferenceStepOptions opts);

}  // namespace r97
