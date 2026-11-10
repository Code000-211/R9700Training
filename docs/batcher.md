# The Batcher: token-budget batches over tokenised sequences

`r97/batcher.h` (`src/batcher.cpp`), over the `Batch`/`TokenSeq` contract of `r97/batch.h`.
The trainer's loop asks a `Batcher` for one step at a time; each epoch is a new shuffle.

## The plan

An epoch is planned on demand from `(seed, epoch)` and cached (`Batcher::plan`):

1. **Group similar lengths.** The kept sequences (those not skipped) are sorted descending by
   token count, ties by input index, and cut into buckets of 32. Each bucket is therefore a
   contiguous run of similar lengths, so a batch never mixes one long sequence with many short
   ones.
2. **Shuffle each bucket.** Every bucket is Fisher-Yates shuffled with a splitmix64 stream
   derived from the config's seed and the epoch number. The same seed and epoch always give the
   same order; a new epoch is a new shuffle. Grouping comes first and the shuffle second on
   purpose: a shuffle followed by a sort by length would erase the shuffle entirely whenever the
   whole dataset fits in one bucket (at most 32 sequences), leaving every seed and every epoch
   with the same batches.
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
returns false without consuming anything, and `seek({epoch + 1, 0})` moves on. Because a plan is
a pure function of `(seed, epoch, sequences)`, a resumed run that seeks to a saved position sees
exactly the batches the original run produced from there.

## Determinism

Everything rests on splitmix64 and a Fisher-Yates over it — no `rand`, no locale, no
unordered containers. Same seed, same batches, every run; another seed, and every new epoch,
shuffles each length bucket differently. Which sequences a bucket holds is fixed by the data
(the grouping); what the seed and the epoch choose is the order inside the buckets, and with it
the step boundaries and so the batches. For a dataset of at most 32 sequences there is a single
bucket, so the order is a plain shuffle.

## Tests

`tests/host/batcher.cpp` holds the host tier's checks: the budget holds for every sub-batch;
every kept sequence appears exactly once an epoch and the skipped one in no batch; the same seed
replays and another seed does not; a seek to a saved `(epoch, step)` continues identically; a
step's sub-batches differ by less than one maximal sequence; a dataset of 20 or 32 sequences
(the single-bucket case) still changes its batches with the seed and with the epoch; and a
dataset of two length classes 40x apart, 32 of each, comes out in steps that hold one class only,
the long ones first -- the grouping, which a plain shuffle over every sequence would lose.
`tests/headers/hdr_batcher.cpp` is the standalone-compile check of the header.

The host test binary has a single `main`, in `tests/host/vector_tape.cpp`; a second one cannot
link. Rather than edit that test or settle the repository's host-test entry point from here, the
batcher's test file carries its own runner and arms it with `std::atexit`, so it runs when the
binary's `main` returns and a failed assertion still fails `ctest`. It is the only part of the
file that has to change once one entry point is agreed.
