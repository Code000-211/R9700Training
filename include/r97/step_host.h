// r97/step_host.h -- the host join: one model's adapter gradient as a StepFn.
//
// The forward driver, the layer replay, the backward core and the live adapter meet here. The
// join names neither a loss nor an adapter kind: it takes a Model, asks the delta for the wire
// the live adapter reads, asks the objective for the seeds the backward starts from, and walks
// from the head down through every block, one replay at a time. docs/STEPS-HOST.md owns the
// path and the memory it holds.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "r97/forward.h"
#include "r97/step.h"

#if defined(RAD_ISSUE_OBSERVER) && defined(RAD_REPLAY_BLOCK)

namespace r97 {

struct HostStepOptions {
    // Keep the boundary values and cotangents in the report. A training loop may turn this off;
    // the issue's checks need them.
    bool keep_boundaries = true;
    // Mark a head-split adapter activation so the join reads it as [rows][K] with the reference
    // step's row. A check that wants the family row's reading of the live trace turns this off.
    bool reference_head_split = true;
};

// What one joined step measured. Boundary 0 is the input of block 0 and boundary `layers` is the
// final hidden state the head reads; each boundary is dense over the step's token rows, in the
// batch's sequence order, so a caller can split it by the batch's lengths.
struct HostStepReport {
    double loss = 0.0;
    int64_t scored = 0;
    int64_t tokens = 0;
    int64_t hidden = 0;
    int64_t layers = 0;
    int64_t residual_store_bytes = 0;    // the boundary residuals (grows with layers)
    // The most forward-value payload the join and the walks held at once: one chunk's snapshots
    // (the head's, or one block's) plus the copies of the one op the walk was adjointing. One
    // chunk's worth, so it does not grow with the number of layers. `peak_snapshot_store_bytes`
    // and `peak_snapshot_op_bytes` are its two parts at the chunk that peaked: the store, which is
    // held whole from the start of the walk, and the one op copied out of it.
    int64_t peak_snapshot_bytes = 0;
    int64_t peak_snapshot_store_bytes = 0;
    int64_t peak_snapshot_op_bytes = 0;
    // What every chunk's store adds up to, which is what holding them all at once would have cost
    // and what the join does not do; it grows with the number of layers, unlike the peak.
    int64_t all_snapshot_store_bytes = 0;
    // The most cotangent payload held at once: the walk's live versions, its cot/dc vectors and
    // input accumulators, plus the boundary cotangent the join keeps for the next block. Also one
    // chunk's worth.
    int64_t peak_cotangent_bytes = 0;
    // The report's own copies of the boundary values and cotangents, which do grow with the number
    // of layers and are zero unless keep_boundaries is set. The two peaks above leave them out.
    int64_t kept_boundary_bytes = 0;
    std::vector<std::string> boundary_name;           // the tape buffer of each boundary
    std::vector<std::vector<double>> boundary_value;  // boundary forward values, bf16 widened
    std::vector<std::vector<double>> boundary_cotangent;
};

// The join. The driver must outlive the returned StepFn; the StepFn keeps the Model and, for each
// call, feeds the live adapter from `params`, runs the batch, and fills `grads` with the delta's
// gradients when it is non-null. The report, when given, is overwritten on every call.
StepFn host_step(ForwardDriver& driver, const Model& model,
                 const HostStepOptions& options = {}, HostStepReport* report = nullptr);

}  // namespace r97

#endif  // RAD_ISSUE_OBSERVER && RAD_REPLAY_BLOCK
