// r97/step_device.h -- the device join: one model's adapter gradient on one card, as a StepFn.
//
// The forward driver runs the batch on the device kernel tier, the residual at every layer
// boundary is kept in one VRAM allocation, and each block is replayed from it on the card and
// walked in reverse with the device adjoints (Side::device). The forward values the walk reads
// are the operands' bytes: pulled back at the replay for a block, and copied at the issue for the
// head's records, whose buffers the engine reuses before the walk runs (capture_head_record in
// src/step_device.cpp); either is widened to the walk's f64 contract. The adapter's gradients are
// staged on the card by the device rows and flushed once, at the step's end. The head is walked
// through the chunked CE loop and never holds a logit row's cotangent: docs/STEPS-DEVICE.md owns
// the path and the memory it holds.
//
// The join names neither a loss nor an adapter kind, as the host join does: Model::objective
// supplies the seeds, Model::delta the wire and the gradient names.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "r97/backward.h"
#include "r97/forward.h"
#include "r97/head.h"
#include "r97/step.h"

#if defined(RAD_ISSUE_OBSERVER) && defined(RAD_REPLAY_BLOCK) && defined(RAD_CTX_STREAM)
#define R97_DEVICE_STEP 1

namespace r97 {

struct DeviceStepOptions {
    // Keep the boundary values and cotangents in the report, as the host join does. Off by
    // default: at 4096 tokens one boundary's f64 copy is 4096 x 5120 x 8 B = 168 MB, and the
    // report holds every boundary's value and cotangent -- 2 x 65 of them is ~21 GiB, past this
    // host's headroom beside the walk's own store. A training loop never asks for them; a check
    // does, on a batch that fits.
    bool keep_boundaries = false;
    // The VRAM the residual store may take. 0 means the card's free VRAM read when the engine is
    // up -- after the model's load, before the step's serve -- and the store refuses with the
    // plan's own numbers when the plan does not fit.
    int64_t residual_budget_bytes = 0;
    // The rows of one head chunk: the CE loop's chunk, and the transposed product's.
    int64_t head_chunk_rows = kHeadChunkRows;
    // Mark a head-split adapter activation so the join reads it as [rows][K] with the reference
    // step's row, the host join's flag with the same meaning.
    bool reference_head_split = true;
    // Walk the head and the blocks with the host rows instead of the device ones
    // (`Side::host`, no fallback): the A/B of one tape. A device row that is wrong shows as a
    // finite host gradient beside the device one; both non-finite means the plumbing -- the
    // seeds, the head's cotangent, the staging -- rather than a row. The host rows add to the
    // gradient ParamSet themselves, so nothing is staged on the card (grad_stage_bytes is 0), and
    // the walk costs whatever the host's f64 arithmetic costs -- hours, not minutes, on the real
    // container. A check runs it only when the device walk's gradient was not finite.
    bool host_walk = false;
};

struct DeviceStepReport {
    double loss = 0.0;
    int64_t scored = 0;
    int64_t tokens = 0;
    int64_t hidden = 0;
    int64_t layers = 0;

    // The card's residual store: the plan's copies, its bytes, and the free VRAM it was checked
    // against.
    int64_t residual_boundaries = 0;
    int64_t residual_extras = 0;
    int64_t residual_store_bytes = 0;
    int64_t residual_budget_bytes = 0;

    // What the walk held on the host at once, in bytes of f64 payload: one block's snapshots
    // (the store the join builds from the replay) plus the one op the walk copies out of it, and
    // the cotangents. One chunk's worth, as the host join's figures are.
    int64_t peak_snapshot_bytes = 0;
    int64_t peak_snapshot_store_bytes = 0;
    int64_t peak_snapshot_op_bytes = 0;
    int64_t peak_cotangent_bytes = 0;
    // The step's frozen operands -- the weights and tables no op of the step writes -- widened
    // once for the whole step and shared by every block (r97/snapshot_share.h). They are held
    // beside the per-block store, not in it, so the figures above describe what one block's walk
    // holds and this one what the step holds once: at a block the join holds
    // peak_snapshot_bytes + shared_snapshot_bytes.
    int64_t shared_snapshot_bytes = 0;

    // The card's gradient staging when the step ended (0 when the step ran without grads).
    int64_t grad_stage_bytes = 0;

    // The head: the transposed product's accumulator, the rows the CE loop walked, and its
    // chunks.
    int64_t d_hout_bytes = 0;
    int64_t head_rows = 0;
    int64_t head_chunks = 0;

    // The head's operands copied at their issue, before the engine reused the buffers: the records
    // captured and the bytes held (the head's, never the whole tape's). A forward-only call walks
    // nothing, so it installs no hook and both stay 0.
    int64_t head_capture_records = 0;
    int64_t head_capture_bytes = 0;

    // The report's own f64 copies of the boundaries, when keep_boundaries is set.
    int64_t kept_boundary_bytes = 0;

    // Where the step's time went: the engine run (the model's load and the prefill), the head's
    // loss and transposed product, and the replay-plus-walk of the blocks.
    double forward_seconds = 0.0;
    double head_seconds = 0.0;
    double backward_seconds = 0.0;

    // Where the backward's seconds went, per op name, summed over the head walk and every
    // block: the adjoint row calls (each walk's own report), and the replay's pull-back of the
    // op's operands -- one copy-and-sync per op, outside the walk. A long block can spend as
    // much of the step here as in the rows.
    std::vector<OpTiming> row_seconds;
    std::vector<OpTiming> pull_seconds;

    std::vector<std::string> boundary_name;           // the tape buffer of each boundary
    std::vector<std::vector<double>> boundary_value;  // boundary forward values, bf16 widened
    std::vector<std::vector<double>> boundary_cotangent;

    // The first non-finite cotangent the step's walks handed on, and whether the forward's operands
    // were finite: a NaN born in the forward is a different bug from one born in the walk, and a
    // gradient is only as good as the values it differentiates. Empty `first_non_finite_op` means
    // the walks saw none; the walks run from the head down through the layers, so "first" is the
    // first the step met, not the first in forward order. The per-op count is on row_seconds'
    // OpTiming.
    std::string first_non_finite_op;
    std::string first_non_finite_operand;
    int64_t first_non_finite_operand_index = -1;
    int64_t first_non_finite_position = -1;
    int64_t first_non_finite_index = -1;
    bool first_non_finite_from_row = false;
    bool forward_all_finite = true;
    // The op that wrote the first non-finite forward value the step's snapshot loop met. The walk
    // can only name the op that first read such a value; the device join also scans what each op
    // wrote, so a NaN born in the forward is named at its producer -- a layer's output, the
    // embedding, or the head's norm -- and not only at the op downstream of it. Empty when the
    // walk met no non-finite value, or when none of the walked ops wrote one (the value then
    // entered from a boundary no walked op writes).
    std::string forward_first_non_finite_op;
    std::string forward_first_non_finite_operand;
    int64_t forward_first_non_finite_operand_index = -1;
    int64_t forward_first_non_finite_position = -1;
    int64_t forward_first_non_finite_index = -1;
};

// The join. The driver must be configured for the device tier -- its ForwardConfig::kernels name
// the card's library and libr97, the plugin the device adjoints load from -- and must outlive the
// returned StepFn. One call forwards the batch with the live adapter, walks it back on the card
// and fills `grads` with the delta's gradients when it is non-null. `grads == nullptr` is a
// forward-only call: the forward runs with the adapter fed from `params` and the report is filled,
// but the head's gradient pass and the block walk are skipped -- the report's cotangent fields stay
// empty and nothing is staged. A training loop's evaluation pass and a check's pre-step reference
// are that call.
StepFn device_step(ForwardDriver& driver, const Model& model,
                   const DeviceStepOptions& options = {}, DeviceStepReport* report = nullptr);

}  // namespace r97

#endif  // RAD_ISSUE_OBSERVER && RAD_REPLAY_BLOCK && RAD_CTX_STREAM
