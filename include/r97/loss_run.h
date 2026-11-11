// r97/loss_run.h -- the reference loss over a set of tokenised sequences, one forward step per
// token-budget batch.
//
// This is the piece `r97train loss` and the gpu device-forward test share: the Batcher packs the
// sequences under the step's token budget, the ForwardDriver runs each sub-batch as one prefill
// step, and its CeObjective sums the loss window by window. The result is the loss the trainer
// reports -- sum of cross-entropy over scored positions divided by their count -- with the run's
// token and scored counts.
//
// Nothing here names a kernel tier: the caller puts `libavx,libref` in ForwardConfig::kernels for
// the host forward or `libr4d` for the device one, and the steps are the same. A sequence longer
// than the budget is not dropped silently: it is listed in LossRunResult::skipped.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "r97/batch.h"
#include "r97/forward.h"

namespace r97 {

struct LossRunConfig {
    ForwardConfig forward;       // model, plugin home, kernel tier, budgets and window
    int64_t token_budget = 0;    // a step's tokens; 0 takes forward.max_tok
    uint64_t seed = 0;           // the Batcher's seed; the mean does not depend on it
    bool sample_vram = false;    // sample the card after the first step, while its engine is up
};

struct LossRunResult {
    double loss = 0.0;           // sum / scored, the reference loss
    double sum = 0.0;            // the summed cross-entropy, in f64
    int64_t scored = 0;          // scored tokens the loss covered
    int64_t tokens = 0;          // tokens in the batches (the step's rows, L-1 a sequence)
    int64_t sequences = 0;       // sequences in the batches
    int64_t steps = 0;           // forward steps taken
    int64_t windows = 0;         // Objective calls, i.e. windows of positions read
    double seconds = 0.0;        // the forward steps alone, not the data's rendering
    std::vector<size_t> skipped; // indices into `sequences` over the token budget
    bool vram_sampled = false;   // sample_vram asked and the card answered
    int64_t vram_used = 0;       // card bytes in use after the first step (its engine is alive)
    int64_t vram_total = 0;      // the card's device-local bytes
};

// Run one epoch's loss over `sequences`. Returns false, and says why in *error, when the driver
// refuses the container or a step throws; a step that throws leaves *out holding what ran before.
bool run_loss(const LossRunConfig& config, const std::vector<TokenSeq>& sequences,
              LossRunResult* out, std::string* error);

}  // namespace r97
