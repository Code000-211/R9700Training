# The Batcher: token-budget batches over tokenised sequences

`r97/batcher.h` (`src/batcher.cpp`), over the `Batch`/`TokenSeq` contract of `r97/batch.h`.
The trainer's loop asks a `Batcher` for one step at a time; each epoch is a new shuffle.

## The plan

An epoch is planned on demand from `(seed, epoch)` and cached (`Batcher::plan`):

1. **Group similar lengths.** The kept sequences (those not skipped) are sorted descending by
   token count, ties by input index, and cut into buckets: a bucket holds at most 32 sequences,
   all of one power-of-two length class (`std::bit_width` of the token count), and a change of
   class always starts a new bucket. Any two lengths inside a bucket therefore differ by less
   than a factor of two, which is what keeps a batch from mixing one long sequence with many
   short ones.
2. **Shuffle the buckets.** The bucket order is Fisher-Yates shuffled with a splitmix64 stream
   derived from the config's seed and the epoch number, then rotated by `(epoch + 3 * seed)`
   modulo the bucket count. The rotation matters when there are only a couple of buckets:
   Fisher-Yates can hand the same order to nearby seeds, and the rotation makes the seed shift
   the order and each epoch advance its starting bucket, which breaks the usual repeated order.
   A sequence inside a bucket stays in length order. The one exception is a dataset that fits
   one bucket: there is no bucket order to shuffle, so that bucket's sequences are shuffled
   under the seed and rotated by the epoch, which keeps the seed and the epoch reaching the
   batches even when the whole dataset is one length class.
3. **Pack each sub-batch under the budget.** Walking the buckets in their shuffled order, each
   sequence goes to the least-loaded of the `devices` sub-batches that still has room for it
   (`load + length <= token_budget`), ties to the lowest device. When no sub-batch can take the
   next sequence the step closes. A bucket that ends with fewer sequences than devices carries
   them into the next bucket to fill its idle devices instead of being handed out with an empty
   sub-batch. If the next bucket is another chunk of the same length class, it just continues
   filling the step. At a class edge the occupied devices are sealed for the carry, so the next
   class cannot join an occupied sub-batch and mix classes inside one `Batch`; every sub-batch
   still holds sequences of one length class. The budget is per sub-batch, not per step: every
   `Batch` stays under `token_budget`, so a step may spend up to `devices * token_budget` tokens.
   The packing keeps each sub-batch in the planned order; it does not re-sort it afterwards.
4. **Balance devices.** Least-loaded-first assignment keeps the token spread between the fullest
   and emptiest sub-batch of a step under one maximal sequence of that step. If a closed step
   still reaches that bound — an odd residue of equal-length sequences, say — the fullest
   sub-batch's last sequences spill into a step of their own until the remainder is strictly
   under the bound. What spills is one sub-batch's worth, so it is still under the budget, and a
   spill step has a single sub-batch and no spread to bound. So a step hands out fewer than
   `devices` sub-batches in two cases: it is a spill split off to keep another step's strict
   spread, or it is the epoch's last tail, which has no next bucket to carry into. Neither is
   handed out with an empty sub-batch: a single final sequence has no step to balance against, so
   its step has one sub-batch rather than an empty second one.

## Skip, position, resume

A sequence longer than the budget is reported in `skipped()` (indices into the input, in input
order, with `skipped_tokens()`) and appears in no batch.

The iteration position is `(epoch, step)`. `position()` reads it, `seek()` moves to it, `reset()`
goes to `(0, 0)`. `next()` returns the step at the position and advances it; at an epoch's end it
returns false without consuming anything, and `seek({epoch + 1, 0})` moves on. Because a plan is
a pure function of `(seed, epoch, sequences)`, a resumed run that seeks to a saved position sees
exactly the batches the original run produced from there. A position restored from stored data is
treated as data and clamped into the valid range: a negative epoch becomes the first, a step
below the epoch's first becomes its first, and a step past its end becomes its end (where
`next()` returns false). A stepped-back or corrupted `epoch`/`step` therefore never indexes a
plan out of bounds.

## Determinism

Everything rests on splitmix64 and a Fisher-Yates over it — no `rand`, no locale, no
unordered containers. Same seed, same batches, every run. Which sequences a bucket holds is
fixed by the data (the grouping); what the seed and the epoch choose is the bucket order, so
even a dataset whose buckets each hold one sequence changes its step order. For a dataset that
fits one bucket, the seed chooses a shuffle of it and the epoch rotates that shuffle, so
consecutive epochs differ even when there are only two sequences. The same seed and epoch replay
the same plan, so a saved position resumes identically.

## Tests

`tests/host/batcher.cpp` holds the host tier's checks: the budget holds for every sub-batch (each
`Batch`), not for a step over all of them, and on the varied data a step spends more than one
budget; a run of four long sequences, none of which can share a sub-batch with another, still
fills both devices of every step; every step's lengths are within a factor of two; every kept
sequence appears exactly once an epoch and the skipped one in no batch; the same seed replays and
another seed does not; a seek to a saved `(epoch, step)` continues identically; `reset()` returns
to `(0, 0)` and replays the first epoch, even after a traversal has run out; a non-positive budget
and fewer than one device are rejected; on the varied data a step with two sub-batches keeps the
strict spread, less than one maximal sequence; a short bucket tail — 33 sequences of 500 beside
32 of 12 on two devices — fills the idle device from the next bucket, so no step has an empty
sub-batch and the carried steps keep the strict bound. For that case a short step is allowed only
as the final one, because the shuffled bucket order may put the one-long remainder last where it
has nothing to carry into. An epoch's last odd tail — 5 sequences of 500 at budget 1200 on two
devices — has no next bucket, so its last step has one sub-batch rather than an empty one, again
keeping the strict bound on every step that has two; and the same holds for the smaller odd
residues (3 sequences of 500, 3 unit sequences) that a greedy hand-out would leave unbalanced. A
dataset whose buckets each hold one sequence — two sequences in different length classes, and one
per class across eight classes — does not come out identical across the seed runs or across three
epochs; so does a dataset that fits one bucket (2, 3 and 20 sequences of one length), by shuffling
and rotating that bucket. A dataset that fits one step per bucket — 20 sequences of 17..74 tokens,
100 sequences of one length, 1000 sequences of 50..80 tokens on two devices — still changes its
whole shape (steps, sub-batches and the order inside them) with the seed and with the epoch, at
the realistic budget of 4096 rather than a small one that forces the packing to end a step early. A dataset of two length classes 40x
apart, 32 long ones beside 32 shorts and then 33 long ones beside 32 shorts with one device, comes
out in steps that hold one class only; the shuffled bucket order decides which class comes first,
so no test fixes that order. The 32 case is exactly two buckets; the 33 case leaves a bucket that
would hold one long and the shorts after it in the sorted order, so it is the case that mixes
classes unless the bucket is cut at the length class and a step stops at the bucket edge — the
grouping that a plain shuffle, or buckets of 32 without the class cut, would lose. A saved
position with a negative step or epoch resumes at the epoch's first step or shuffle, and one past
the epoch's end sits at its end and hands out nothing, so a stepped-back or corrupted position
cannot read outside the plan. `tests/headers/hdr_batcher.cpp` is the standalone-compile check of
the header.

The host test binary has a single `main`, in `tests/host/vector_tape.cpp`; a second one cannot
link. Rather than edit that test or settle the repository's host-test entry point from here, the
batcher's test file carries its own runner and arms it with `std::atexit`, so it runs when the
binary's `main` returns and a failed assertion still fails `ctest`. It is the only part of the
file that has to change once one entry point is agreed.
