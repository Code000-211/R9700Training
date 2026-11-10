// r97/optim.h -- AdamW over a ParamSet, with warm-up + cosine schedule, global-norm
// clipping and gradient accumulation. Host only, multithreaded; the state serialises to
// and from a byte buffer so a run can stop and continue bit-identically.
//
// Built on include/r97/params.h only: the ParamSet holds f32 values, which are the
// master weights, updated in place. Moments and the accumulation buffer are private
// f32 copies of the same shapes.
//
// The update (per element, t = 1-based update index, g the clipped, accumulated gradient):
//   m  = b1*m + (1-b1)*g
//   v  = b2*v + (1-b2)*g*g
//   mh = m / (1 - b1^t)        vh = v / (1 - b2^t)
//   w -= lr(t) * (mh / (sqrt(vh) + eps) + wd * w)
// Weight decay is decoupled (AdamW) and applies to every parameter by default; a
// parameter whose name contains a substring of `no_decay` is excluded.
#pragma once

#include "r97/params.h"

#include <cstdint>
#include <string>
#include <vector>

namespace r97 {

struct AdamWConfig {
    float lr = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.0f;
    float clip_norm = 0.0f;             // <= 0 disables clipping
    int warmup_steps = 0;               // linear from 0 to lr over these updates
    int total_steps = 0;                // cosine horizon from warm-up's end; <= 0 holds lr
    float lr_floor_frac = 0.1f;         // cosine floor: lr * floor_frac at total_steps
    int accum_steps = 1;                // gradient accumulation over k step() calls
    std::vector<std::string> no_decay;  // name substrings excluded from weight decay
    int threads = 0;                    // 0: std::thread::hardware_concurrency()
};

// The learning rate at the `step`-th update (1-based): linear warm-up to `lr`, then cosine
// down to `lr * lr_floor_frac` at `total_steps` and flat at the floor beyond it.
float scheduled_lr(const AdamWConfig& cfg, int step);

// Rescale `grads` in place so its global norm is at most `max_norm`; returns the norm
// before clipping. `max_norm <= 0` is a no-op that still returns the norm.
float clip_global_norm(ParamSet& grads, float max_norm);

class Optimiser {
public:
    // `params` must outlive the optimiser; its values are the master weights and are
    // updated in place. Throws std::invalid_argument on an empty set, a non-2D-friendly
    // shape (an empty value with a non-empty shape) or a bad config.
    Optimiser(ParamSet& params, AdamWConfig cfg);
    ~Optimiser();
    Optimiser(const Optimiser&) = delete;
    Optimiser& operator=(const Optimiser&) = delete;

    // Accumulate `grads` (element-wise sum) into the buffer; every `accum_steps`-th call
    // the mean gradient is clipped, then applied as one AdamW step, and the buffer clears.
    // Returns true when this call applied an update. Throws std::invalid_argument when
    // `grads` names different parameters or shapes than the optimiser was built with.
    bool step(const ParamSet& grads);

    int updates_done() const { return step_; }
    int pending() const { return accum_count_; }
    int threads() const { return nthreads_; }

    // Bytes: magic "R97OPT" + u16 version 1 + u16 zero, i64 update step, i32 pending
    // count, i32 parameter count, then per parameter in order (u64 element count, then
    // that many f32 weights, then the same for m, v and the accumulation buffer).
    // Native endianness, plain memcpy of the float bits. Throwing on the way in
    // (`restore`) leaves the optimiser unchanged.
    std::vector<uint8_t> serialise() const;
    void restore(const std::vector<uint8_t>& bytes);  // std::runtime_error on bad bytes

private:
    ParamSet& params_;
    AdamWConfig cfg_;
    ParamSet m_;
    ParamSet v_;
    ParamSet g_;                        // accumulation buffer, zero when not pending
    std::vector<size_t> offsets_;       // params_.size() + 1 element prefix sums
    std::vector<uint8_t> decay_;        // per parameter: weight decay applies
    int step_ = 0;                      // applied AdamW updates
    int accum_count_ = 0;
    int nthreads_;
};

}  // namespace r97
