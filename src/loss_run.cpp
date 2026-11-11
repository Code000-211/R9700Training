// src/loss_run.cpp -- the reference loss over a set of sequences, one forward step a batch.
#include "r97/loss_run.h"

#include "engine.h"           // radiance core/: Engine, the driver's own

#include <chrono>
#include <memory>
#include <stdexcept>
#include <utility>

#include "r97/batcher.h"
#include "r97/device_tier.h"
#include "r97/loss.h"

namespace r97 {
namespace {

// The loss model: the reference cross-entropy on the "logits" boundary. Named here rather than
// looked up by the registry's `ce` name, so a caller's loss is the one this function reports.
Model ce_model() {
    Model model;
    model.objective = CeObjective;
    return model;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

bool run_loss(const LossRunConfig& config, const std::vector<TokenSeq>& sequences,
              LossRunResult* out, std::string* error) {
    *out = LossRunResult();
    const int64_t budget = config.token_budget > 0 ? config.token_budget : config.forward.max_tok;
    if (budget < 1) {
        if (error) *error = "a step needs a token budget of at least one";
        return false;
    }

    std::unique_ptr<ForwardDriver> driver = ForwardDriver::open(config.forward, error);
    if (!driver) return false;

    BatcherConfig bc;
    bc.token_budget = budget;
    bc.devices = 1;
    bc.seed = config.seed;
    Batcher batcher(sequences, bc);
    out->skipped = batcher.skipped();

    const Model model = ce_model();
    std::vector<Batch> step_batches;
    while (batcher.next(&step_batches)) {
        for (const Batch& batch : step_batches) {
            if (batch.empty()) continue;
            const auto t0 = std::chrono::steady_clock::now();
            StepResult got;
            try {
                // A fresh StepFn per step: the driver releases the previous engine before this
                // one starts, and the StepFn hands back a step whose buffers are gone next time.
                got = driver->step(model)(batch, ParamSet(), nullptr);
            } catch (const std::exception& e) {
                if (error) *error = e.what();
                return false;
            }
            out->seconds += seconds_since(t0);
            out->steps += 1;
            out->windows += driver->last().windows();
            if (config.sample_vram && out->steps == 1) {
                DeviceVram v;
                std::string why;
                if (device_vram(&v, &why)) {
                    out->vram_sampled = true;
                    out->vram_used = v.used();
                    out->vram_total = v.total;
                }
            }
            out->sum += got.loss;
            out->scored += got.scored;
            for (const TokenSeq& s : batch) {
                out->tokens += int64_t(s.ids.size());
                out->sequences += 1;
            }
        }
    }
    out->loss = out->scored > 0 ? out->sum / double(out->scored) : 0.0;
    return true;
}

}  // namespace r97
