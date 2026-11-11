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
2. **Shuffle the buckets, and what is inside them.** The bucket order is Fisher-Yates shuffled
   with a splitmix64 stream derived from the config's seed and the epoch number, then rotated to
   start on the bucket the epoch names: the seed chooses where in the bucket cycle an epoch
   begins, and every epoch moves that start on by one. Two consecutive epochs therefore start on
   different buckets, which -- a bucket never being empty -- is a different order of the
   sequences, whatever the shuffle draws. Rotating a fresh shuffle by an offset drawn from the
   same `(seed, epoch)` cannot promise that: the shuffle and the offset are drawn independently,
   so an epoch's order lands on the previous epoch's exactly when the two shuffles are rotations
   of each other. Nor may the offset be a linear function of the seed, as `(epoch + 3 * seed)`
   was: that term vanishes whenever the bucket count divides three times the seed, and two seeds
   then hand out one order.

   The sequences inside a bucket are shuffled too, from a stream for `(seed, epoch, bucket)`. The
   sort did its work when it formed the bucket: inside one, every length is in one power-of-two
   class, so shuffling them costs the grouping nothing, and it is what gives the seed and the
   epoch room to differ. The bucket order alone is a small space for a dataset of a few classes --
   `n` buckets have `n!` orders and no more -- so two seeds would hand out the same batches one
   time in two or six on a dataset of two or three classes however the seed were hashed, and a
   fresh draw per epoch would repeat another epoch's order about as often. A dataset that fits one
   bucket is the exception: there is no bucket order to shuffle, so the same two steps are taken
   one level down -- the epoch starts on the sequence the seed's hash plus the epoch names, and the
   rest of the bucket is a fresh Fisher-Yates under `(seed, epoch)`. The start keeps consecutive
   epochs apart even when the whole dataset is one length class and has only two sequences; the
   fresh shuffle is what keeps the epoch from cycling, where rotating one seed-drawn shuffle by the
   epoch hands out the first epoch's order again after as many epochs as the bucket holds.
3. **Pack each sub-batch under the budget.** Adjacent buckets of one length class are merged
   into one run before packing: the 32-sequence chunks were the sort's unit, not the step's, so a
   step keeps filling across that seam instead of closing on an arbitrary chunk boundary. Walking
   the runs in their shuffled order, each sequence goes to the least-loaded of the `devices`
   sub-batches that can still take it: it has room under the budget (`load + length <=
   token_budget`) and adding it keeps the step's spread inside its bound (point 4). Ties go to the
   lowest device. When no sub-batch can take the next sequence the step closes and the next
   sequence starts a new one. When the kept data is a single run of one length -- fixed-length or
   packed data, the case the budget exists for -- the step sizes are planned exactly instead:
   every step before the tail is a multiple of `devices` sequences (so every device gets the same
   count and the spread is zero), and the tail is split into equal sub-batches that each fit the
   budget. That spends the budget on the fewest steps the rules allow; filling greedily can leave
   a tail (five sequences at four per device on three devices) that needs an extra step to hand
   out balanced. A run that ends with fewer sequences than devices carries them into
   the next run to fill its idle devices instead of being handed out with an empty sub-batch. At a
   class edge the occupied devices are sealed for the carry, so the next class cannot join an
   occupied sub-batch and mix classes inside one `Batch`; every sub-batch still holds sequences of
   one length class, while a step may hold two classes in different sub-batches. The budget is per
   sub-batch, not per step: every `Batch` stays under `token_budget`, so a step may spend up to
   `devices * token_budget` tokens. The packing keeps each sub-batch in the planned order; it does
   not re-sort it afterwards.
4. **Balance devices.** The spread is a property of the step the reader receives, not of every
   intermediate add. Walking a fixed-length run from one sequence per device to four of the same
   length passes through a spread of exactly one maximal sequence; a packer that bounds every add
   stops there and hands out a fraction of the budget the batcher exists to spend. So
   `keeps_spread` admits an add that lands exactly on the bound while the rest of the run can
   still raise the sub-batches at the bottom: it simulates the next sequences, at most `devices`
   of them, raising the least-loaded device, and allows the add only if they bring the spread
   strictly inside the bound. If the run ends first, or those sequences do not fit, the add is
   refused and the step closes on the state before it, which is still inside the bound. Thus a
   step closes with every device carrying a sub-batch -- the sequences it could not take are the
   first of the next step, and nothing is split off after the fact to keep the bound. A carried
   class edge is never admitted at the bound, because its sealed sub-batches at the bottom cannot
   be raised. Every step but an epoch's last tail therefore hands out `devices` sub-batches, and
   the tail hands out the one to `devices` it has. Neither is handed out with an empty sub-batch: a
   single final sequence has no step to balance against, so its step has one sub-batch rather than
   an empty second one. A sequence of no tokens is the case neither rule covers and the packing
   does not special-case: it is its own length class, so it cannot join a sub-batch that already
   holds one, and a step of one sub-batch has no bound that admits a second, so a zero-token
   sequence can leave a step short in the middle of an epoch. A tokenised sequence is never empty,
   and such a sequence has nothing to train on.

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
fixed by the data (the grouping); what the seed and the epoch choose is the bucket order and the
order inside each bucket, so even a dataset whose buckets each hold one sequence changes its plan
with both. A dataset whose buckets all hold a single sequence has only `n!` plans for its `n`
buckets, so two seeds must agree on some such datasets — two buckets leave two orders — and the
seed is hashed into the choice so that it reaches every one of them; every bucket that holds more
than one sequence multiplies the plans by its own orders. For a dataset that fits one bucket, the
seed hashes to the sequence the epoch starts on and the epoch moves that start on by one, so
consecutive epochs differ even when there are only two sequences, and the rest of the bucket is
reshuffled every epoch, so the plans do not cycle: one shuffle rotated by the epoch is back where
it started after `k` epochs and hands out `k` plans and no more. Two consecutive epochs never hand
out the same plan.
The same seed and epoch replay the same plan, so a saved position resumes identically.

## Tests

`tests/host/batcher.cpp` holds the host tier's checks: the budget holds for every sub-batch (each
`Batch`), not for a step over all of them, and on the varied data a step spends more than one
budget; a run of four long sequences, none of which can share a sub-batch with another, still fills
both devices of every step; every sub-batch's lengths are within a factor of two (a step may hold
two classes, one per sub-batch, where a short tail fills an idle device); every kept sequence
appears exactly once an epoch and the skipped one in no batch; the same seed replays and another
seed does not; a seek to a saved `(epoch, step)` continues identically; `reset()` returns to `(0,
0)` and replays the first epoch, even after a traversal has run out; a non-positive budget and fewer
than one device are rejected; on the varied data a step with two sub-batches keeps the strict
spread, less than one maximal sequence; a short bucket tail — 33 sequences of 500 beside 32 of 12 on
two devices — fills the idle device from the next bucket, so no step has an empty sub-batch and the
carried steps keep the strict bound. For that case a short step is allowed only as the final one,
because the shuffled bucket order may put the one-long remainder last where it has nothing to carry
into. An epoch's last odd tail — 5 sequences of 500 at budget 1200 on two devices — has no next
bucket, so its last step has one sub-batch rather than an empty one, again keeping the strict bound
on every step that has two; and the same holds for the smaller odd residues (3 sequences of 500, 3
unit sequences) that a greedy hand-out would leave unbalanced. Every step but the epoch's last fills
all of its devices — on three longs beside three shorts on two devices, the case where one long is
left for the next step and the three shorts then fill the device it left idle, and on 800 varied
lengths at two, three and four devices, over three seeds and three epochs each, and on 101 sequences
of one length at three devices; a packer that bounds every add hands out one sequence per device
there, a step of one sub-batch in the middle of the epoch.
`batcher_fills_the_budget_on_fixed_lengths` pins what a token-budget batcher spends on fixed-length
data: 101 of 500 at budget 2000 on three devices, 100 of 100 at 1000 on two, and 1000 of 100 and 800
of 128 at 4096 on four each stay within their step counts and spend at least three quarters of the
capacity of their steps — the old packer took 34 steps where this plan takes nine, and 250 and 512
where it takes seven. A dataset whose buckets each hold one sequence —
two sequences in different length classes, and one per class across eight classes — does not come
out identical across the seed runs or across three epochs; so does a dataset that fits one bucket
(2, 3 and 20 sequences of one length), by shuffling that bucket and starting it on the sequence the
epoch names. `batcher_single_bucket_does_not_cycle` looks past the neighbouring epoch, which is
where the earlier rotation went wrong: a dataset that fits one bucket, run for three times its size
in epochs, must not hand out the first epoch's plan again at epoch `k` (20 sequences of 50 tokens
at budget 500 on two devices did, and 8 of 12 tokens at 36 did; 12 of 7 tokens at 30 and the same
20 on one device are checked beside them) and must hand out more plans than it has sequences -- one
plan per rotation and no more is exactly the cycle. A dataset that fits
one step per bucket — 20 sequences of 17..74 tokens, 100 sequences of one length, 1000 sequences of
50..80 tokens on two devices — still changes its whole shape (steps, sub-batches and the order
inside them) with the seed and with the epoch, at the realistic budget of 4096 rather than a small
one that forces the packing to end a step early. A dataset of two length classes 40x apart, 32 long
ones beside 32 shorts and then 33 long ones beside 32 shorts with one device, comes out in steps
that hold one class only; the shuffled bucket order decides which class comes first, so no test
fixes that order. The 32 case is exactly two buckets; the 33 case leaves a bucket that would hold
one long and the shorts after it in the sorted order, so it is the case that mixes classes unless
the bucket is cut at the length class and a step stops at the bucket edge — the grouping that a
plain shuffle, or buckets of 32 without the class cut, would lose. A saved position with a negative
step or epoch resumes at the epoch's first step or shuffle, and one past the epoch's end sits at its
end and hands out nothing, so a stepped-back or corrupted position cannot read outside the plan.
Two shapes the last round's review reproduced — twelve sequences across four length classes at a
budget that keeps every sequence under it, where two consecutive epochs handed out the same steps,
and eight sequences where two seeds did — are checked epoch by epoch against each other, and the
two seeds' first epochs against one another. Because "each epoch is a new shuffle" is a promise
over ordinary data and not over the shapes a test happens to pick, a few hundred generated datasets
(20..49 sequences, every one under the budget, two to four devices, lengths in a handful of
classes) compare four epochs each against the epoch before, and check each step on the way for the
budget, the spread, the one-class sub-batches and the filled devices; and over sixty datasets
spread across eight length classes or more, three seeds must give three different first epochs.
`tests/headers/hdr_batcher.cpp` is the standalone-compile check of the header.

The host test binary has a single `main`, in `tests/host/vector_tape.cpp`; a second one cannot link.
The batcher's checks therefore register themselves with `R97_TEST` and fail through `R97_CHECK` from
`tests/support/test.h`, which runs them once that `main` returns, so a failed check still fails
`ctest`.
