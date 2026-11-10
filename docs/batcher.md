// The Batcher: token-budget batches over tokenised sequences.
//
// `r97/batcher.h` (`src/batcher.cpp`), over the `Batch`/`TokenSeq` contract of `r97/batch.h`.
// The trainer's loop asks a `Batcher` for one step at a time; each epoch is a new shuffle.

## The plan

An epoch is planned on demand from `(seed, epoch)` and cached (`Batcher::plan`):

1. **Shuffle.** The kept sequences (those not skipped) are Fisher-Yates shuffled with a
   splitmix64 stream derived from the config's seed and the epoch number. The same seed and
   epoch always give the same order; a new epoch is a new shuffle.
2. **Group similar lengths.** The shuffled list is cut into buckets of 32 sequences and each
   bucket is sorted descending by token count. So a batch never mixes one long sequence with
   many short ones, while the epoch's randomness is preserved across buckets.
3. **Pack steps.** Walking the order, a step closes when adding the next sequence would push
   its total token count over `token_budget`. The budget therefore holds for every step.
4. **Balance devices.** Each step is dealt into `devices` sub-batches: longest first, each
   sequence handed to the least-loaded sub-batch at that point (ties to the lowest device). This
   keeps the token spread between fullest and emptiest below one maximal sequence of the step. A
   spread of exactly one maximal sequence can be forced (one sequence for two devices, or three
   budget-length sequences into two devices) and is then the best any arrangement can do; the
   tests assert strictly below it for varied data.

## Skip, position, resume

A sequence longer than the budget is reported in `skipped()` (indices into the input, in input
order, with `skipped_tokens()`) and appears in no batch.

The iteration position is `(epoch, step)`. `position()` reads it, `seek()` moves to it, `reset()`
goes to `(0, 0)`. `next()` returns the step at the position and advances it; at an epoch's end it
returns false without consuming, and `reset()`/`seek()` moves on. Because a plan is a pure
function of `(seed, epoch, sequences)`, a resumed run that seeks to a saved position sees exactly
the batches the original run produced from there.

## Determinism

Everything rests on splitmix64 and a Fisher-Yates over it — no `rand`, no locale, no
unordered containers. Same seed, same batches, every run; another seed shuffles differently.
