// r97/batcher.h -- token-budget batches over tokenised sequences.
//
// Given the training data (one `TokenSeq` per example) the Batcher plans the epoch: it groups
// similar lengths (sorted into buckets, each bucket shuffled under the seed) so a batch never
// mixes one long sequence with many short ones, and packs each sub-batch's total token count
// under the budget. Each step hands out `devices` sub-batches with balanced token counts, one per
// device; the budget is per sub-batch, so a step may spend up to `devices * token_budget` tokens.
// A sequence longer than the budget is reported (in `skipped()`) and never batched. A sub-batch
// holds its sequences in the order the epoch's shuffle gave them, so the seed and the epoch reach
// the batches even where the packing's outcome is fixed.
//
// Deterministic for a given seed: the same seed replays the same batches, each epoch is a new
// shuffle, and a position (epoch, step) saved and restored resumes with the same batches.
#pragma once

#include <cstdint>
#include <vector>

#include "r97/batch.h"

namespace r97 {

struct BatcherConfig {
    int64_t token_budget = 0;  // tokens per sub-batch, not per step
    int devices = 1;           // sub-batches per step
    uint64_t seed = 0;         // the shuffle's seed; same seed, same batches
};

class Batcher {
public:
    struct Position {
        int64_t epoch = 0;  // which shuffle
        int64_t step = 0;   // the step's index within that epoch
    };

    Batcher() = default;
    Batcher(std::vector<TokenSeq> sequences, const BatcherConfig& config);

    // The next step: one sub-batch per device. False at the epoch's end, consuming nothing (the
    // position keeps pointing at the epoch's end); `seek({epoch + 1, 0})` moves on to the next
    // epoch, `reset()` back to the first.
    bool next(std::vector<Batch>* step);

    // How many steps the named epoch plans (the shuffle decides the packing).
    int64_t steps_in_epoch(int64_t epoch) const;

    const Position& position() const;  // where the next next() reads
    void reset();                      // to (0, 0), same seed
    void seek(const Position& p);

    // Indices into the input's sequences that exceed the budget, in input order; the caller
    // reports them and they appear in no batch.
    const std::vector<size_t>& skipped() const;
    int64_t skipped_tokens() const;

    const std::vector<TokenSeq>& sequences() const;
    const BatcherConfig& config() const;

private:
    // One planned epoch: steps[s][d] holds the sequence indices of step s, device d, in the
    // epoch's shuffled order inside each sub-batch, so a tape reads them back in a stable order.
    struct EpochPlan {
        std::vector<std::vector<std::vector<size_t>>> steps;
    };
    const EpochPlan& plan(int64_t epoch) const;

    std::vector<TokenSeq> sequences_;
    BatcherConfig config_{};
    std::vector<size_t> skipped_;  // input indices over budget
    int64_t skipped_tokens_ = 0;
    mutable int64_t planned_epoch_ = -1;  // the epoch whose plan is cached, -1 when none
    mutable EpochPlan cached_plan_;
    Position position_{};
};

}  // namespace r97
