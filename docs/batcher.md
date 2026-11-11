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
2. **Shuffle each bucket.** Every bucket is Fisher-Yates shuffled with a splitmix64 stream
   derived from the config's seed and the epoch number. The same seed and epoch always give the
   same order; a new epoch is a new shuffle. Grouping comes first and the shuffle second on
   purpose: a shuffle followed by a sort by length would erase the shuffle entirely whenever the
   whole dataset fits in one bucket (at most 32 sequences), leaving every seed and every epoch
   with the same batches.
3. **Pack each sub-batch under the budget.** Walking one bucket's shuffled order, each sequence
   goes to the least-loaded of the `devices` sub-batches that still has room for it
   (`load + length <= token_budget`), ties to the lowest device. When no sub-batch can take the
   next sequence the step closes. At a bucket's end the step normally closes too, so its
   sub-batches hold one length class only. The exception is a bucket that ends with fewer
   sequences than devices: those sequences carry into the next bucket to fill its idle devices
   instead of being handed out with an empty sub-batch. The occupied devices are sealed for the
   carry, so the next bucket's shorter sequences cannot join an occupied sub-batch and mix
   classes inside one `Batch`; each sub-batch still holds sequences of one length class. Because
   the sort runs longest first, the carry crosses from the short end of one class to the long end
   of the next, so the two classes that meet at one carry are within a factor of two, and a carry
   repeats only when a class is too small to fill the remaining devices. The budget
   is per sub-batch, not per step: every `Batch` stays under `token_budget`, so a step may spend
   up to `devices * token_budget` tokens. A sub-batch keeps its sequences in the order the
   shuffle put them in; the packing does not re-sort them. A re-sort (longest first, say) would
   erase the epoch's shuffle wherever the packing's outcome is fixed by the data — a bucket that
   fits one step, sequences all of one length — leaving every seed and every epoch with
   byte-identical batches.
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
exactly the batches the original run produced from there.

## Determinism

Everything rests on splitmix64 and a Fisher-Yates over it — no `rand`, no locale, no
unordered containers. Same seed, same batches, every run; another seed, and every new epoch,
shuffles each length bucket differently. Which sequences a bucket holds is fixed by the data
(the grouping); what the seed and the epoch choose is the order inside the buckets — the order
the batches hand the sequences to a tape, and with it the step boundaries and so the batches. For
a dataset whose sequences all share one length class and fit one bucket (at most 32), the order is
a plain shuffle; for one small next to the budget, so that every step is one bucket wide, it is
still the whole of what changes between seeds and epochs.

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
sub-batch and the carried steps keep the strict bound; an epoch's last odd tail — 5 sequences of
500 at budget 1200 on two devices — has no next bucket, so its last step has one sub-batch rather
than an empty one, again keeping the strict bound on every step that has two; and the same holds
for the smaller odd residues (3 sequences of 500, 3 unit sequences) that a greedy hand-out would
leave unbalanced; a dataset that fits one step per bucket — 20 sequences of 17..74 tokens, 100
sequences of one length, 1000 sequences of 50..80 tokens on two devices — still changes its whole
shape (steps, sub-batches and the order inside them) with the seed and with the epoch, at the
realistic budget of 4096 rather than a small one that forces the packing to end a step early; and
a dataset of two length classes 40x apart, 32 long ones beside 32 shorts and then 33 long ones
beside 32 shorts with one device, comes out in steps that hold one class only, the long ones
first. The 32 case is exactly two buckets; the 33 case leaves a bucket that would hold one long
and the shorts after it in the sorted order, so it is the case that mixes classes unless the
bucket is cut at the length class and a step stops at the bucket edge — the grouping that a plain
shuffle, or buckets of 32 without the class cut, would lose.
`tests/headers/hdr_batcher.cpp` is the standalone-compile check of the header.

The host test binary has a single `main`, in `tests/host/vector_tape.cpp`; a second one cannot
link. Rather than edit that test or settle the repository's host-test entry point from here, the
batcher's test file carries its own runner and arms it with `std::atexit`, so it runs when the
binary's `main` returns and a failed assertion still fails `ctest`. It is the only part of the
file that has to change once one entry point is agreed.
