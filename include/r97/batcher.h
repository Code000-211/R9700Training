// r97/batcher.h -- token-budget batches over tokenised sequences.
//
// Given the training data (one `TokenSeq` per example) the Batcher plans the epoch: it groups
// similar lengths into buckets (sorted by token count, cut at power-of-two length classes) and
// shuffles the bucket order and each bucket's own order under the seed, so a batch never mixes one
// long sequence with many short ones, and packs each sub-batch's total token count under the
// budget. The epoch's first bucket advances by one every epoch, so two consecutive epochs cannot
// hand out the same order; inside a bucket the order is a shuffle as well, so the seed reaches
// every plan and not only the order of the buckets. A dataset that fits one bucket has no bucket
// order to shuffle, so that bucket's sequences are shuffled under the seed and rotated by the
// epoch; either way the seed and the epoch reach the batches. Each step hands out `devices`
// sub-batches with balanced token counts, one per device, and only the epoch's last tail hands out
// fewer -- a sequence of no tokens can also leave one short, since it is a class of its own and a
// step of one sub-batch has no spread to admit it under. The spread is a property of the step, not
// of every add: a step on its way to filling every device with the same number of fixed-length
// sequences passes through a spread of one maximal sequence and comes back inside it, so the
// packer admits an add that lands on the bound while the rest of the run can still raise the
// sub-batches below the longest, and refuses it -- starting the next step rather than splitting a
// step off after the fact -- when it cannot. A run of one length class is packed as a whole (its
// adjacent 32-sequence chunks are one run), and a dataset that is one length throughout has its
// step sizes planned to spend the budget on the fewest balanced steps; a run that ends with fewer
// sequences than devices carries them into the next bucket to fill the idle devices, which is how
// a short tail fills a device a long sequence left idle; the buckets' classes never meet inside
// one sub-batch. A step never hands out an empty sub-batch. The budget is per sub-batch, so a step
// may spend up to `devices * token_budget` tokens. A sequence longer than the budget is reported
// (in `skipped()`) and never batched. A sub-batch holds its sequences in the planned order, which
// the seed and the epoch choose both through the bucket order and through the shuffle inside a
// bucket.
//
// Deterministic for a given seed: the same seed replays the same batches, each epoch is a new
// shuffle -- two consecutive epochs never hand out the same plan -- and a position (epoch, step)
// saved and restored resumes with the same batches. How far apart two seeds are is bounded by how
// many plans there are: a dataset whose buckets each hold one sequence has `n!` of them for `n`
// buckets, so two seeds must agree on some of those datasets.
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
        int64_t epoch = 0;  // which shuffle; seek() clamps a negative one to the first
        int64_t step = 0;   // the step's index within that epoch; seek() clamps it to the epoch
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
    // Clamps the position into its valid range -- a negative epoch is the first, a step below
    // the epoch's first is its first, past its end is its end -- so a position restored from
    // stored data can never index outside a plan.
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
