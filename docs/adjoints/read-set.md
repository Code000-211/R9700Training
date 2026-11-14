# The read set: zeroing only the rows nothing reads

A step's forward can leave a non-finite value in a buffer the backward must differentiate. The
backward has three honest answers, and only three:

1. **Differentiate** the row. Right when the row's value is finite.
2. **Refuse**, naming the op, the position and the count. Right when a consumer reads the row: its
   gradient cannot be computed from a non-finite value, and a zero would be a silent lie.
3. **Zero** the row. Right only when the tape proves no consumer reads it: then the row is dead,
   nothing downstream depends on it, and zero is the true gradient.

There is no fourth answer. In particular, zeroing a row a consumer reads -- which is what the
rmsnorm rows used to do -- turns a loud forward defect into a quiet wrong gradient, and the 27B
device step's zero gradient for 399 of 400 adapter leaves was exactly that failure mode reached
through a different door. So the rule is two-way, and the default is refusal.

## The rule

`AdjointCall::read_rows` (`include/r97/adjoint.h`) carries the rows of a record's output that a
consumer reads, or null. `row_readable(call, row)` is the only question a row asks:

| `read_rows` | meaning | a non-finite row |
|---|---|---|
| `nullptr` | unknown | **refuses** |
| empty vector | proven: no consumer reads any row | zeroed |
| non-empty, row absent | proven: nothing reads this row | zeroed |
| non-empty, row present | proven: a consumer reads this row | **refuses** |

**Unknown is never unreadable.** A row set the caller cannot prove is null, and null keeps the
refusal. That direction is the whole safety argument: a bug in the analysis costs a run that stops,
never a gradient that is wrong.

The row counts what it did (`AdjointCall::RowCount`, summed into `OpTiming` and out to the step's
`row_seconds` JSON as `rows_zeroed_unreadable`, `rows_readable`, `rows_refused_non_finite`). The
counts are the check on the analysis: for a 512-token step the head's norm must report **511
zeroed and 1 readable** -- the head's `gather_rows` selects one row of the 512 -- and a run that
reports 512 or 0 has a wrong read set, not a different forward.

## Deriving it: `derive_read_sets` (`src/read_set.h`)

The join, not the walk, derives the set, because it is the only layer that holds the whole tape.
`derive_read_sets(records, all_ops, snaps, external_reads)` scans **`all_ops`, the whole step**, not
the block: a consumer can sit in another block or in the head, and a block-local scan would call a
live row dead.

For each record in the walk it:

1. Takes the record's **one** written output. Two written outputs with different row spaces cannot
   share a read set, so such a record is left unproven.
2. Leaves the record unproven when the buffer is named in `external_reads` -- a seed's buffer, the
   cotangent from above, the caller's own use. A read outside the tape is a consumer the analysis
   cannot see, so the value is not dead.
3. Walks `all_ops` forward from the record. Every operand of the buffer in that window is a
   consumer; **a write of the buffer ends the window**, because a later write is a different value
   and its readers are not this value's consumers. This is the in-place case: the block's residual
   add rewrites the buffer it read, and the reads after it belong to the new version.
4. Accepts the window only when **every** consumer is `gather_rows` with a readable index operand
   (`operands[1]`) whose values are finite, integral and in `[0, rows)`. The gather's index list is
   then exactly the rows anything reads, and every other row of the output is provably unread.
5. Anything else -- another op, a gather whose index list the snapshots do not answer, an index
   value out of range -- leaves the record **out of the map**, and the walk hands it null.

The result is `record position -> sorted, unique row indices`, which is `BackwardOptions::read_rows`.

### Why only a gather

Every other op in the tape may read any row of its input: a GEMM reads all of them, a norm reads all
of them, a residual add reads all of them. Proving a row dead for those shapes needs a consumer
graph over the whole vocabulary, and one misclassified op zeroes a live row's gradient silently.
`gather_rows` needs no classification at all: its index list is in hand, in the tape's own operand
order, and it is the only op in the step whose read set is a literal list.

This is deliberately narrow, and it is enough: the head's `rmsnorm` output is read by exactly one
`gather_rows` and nothing else, which is the case the 27B step hits. When a second op shape needs
the rule, the analysis widens with a second case to design and test against, not with a fallback.

## Where it is threaded

- `src/step_host.cpp` and `src/step_device.cpp` call `derive_read_sets` once for the head's walk
  (with the seed buffers as `external_reads`) and once per block (with the block's incoming buffer
  as `external_reads`), then point `BackwardOptions::read_rows` at the map before that walk.
- `src/backward.cpp` looks the record's position up in the map and sets `call.read_rows` -- the
  record's list, or null when the map does not name it -- plus `call.row_count`.
- The rmsnorm rows apply the rule: `rmsnorm_host_adjoint` (`src/adjoint/host/elementwise.cpp`) and
  the `rmsnorm` part of `quant_adjoint` (`src/adjoint/device/quant.hip`, through `add_back`'s
  `rows_per`/`read_rows`). The twin path -- `rmsnorm_twin`, which serves the fused
  `rmsnorm_had_quant_i8g` record -- honours it too: a record whose non-finite rows are all
  unreadable is servable, and the twin no longer refuses it.

## The twin's gain

`rmsnorm_twin` and `rmsnorm_host_adjoint` take the gain as **the first non-written operand after
index 0** (`norm_gain`), not `operands[1]`. The plain record is `x, w, y`; the fused producer is
`x, residual:inout?, w, q, scale, out?`. On the fused shape `operands[1]` is the residual, and the
old code used a present residual's first row as the gain -- a silent wrong intermediate that the
read set would have had no reason to catch. `norm_gain` is the same answer on both shapes, and it
refuses (returns empty, so the record refuses) when a gain exists and does not cover a row.

## Tests

- `tests/host/read_set.cpp` (host tier, no card): a lone gather proves `{1}`; a non-gather consumer
  leaves the record unproven; an external read leaves it unproven; a missing index list leaves it
  unproven; the walk zeroes an unreadable non-finite row and differentiates the readable one with
  the counts at 1/1/0; a non-finite row inside the read set refuses.
- `tests/gpu/device_step.cpp`'s `device_step_tape` (512 tokens, the real container) is the
  acceptance: the step's `row_seconds` must carry `rmsnorm` with `rows_zeroed_unreadable == 511`
  and `rows_readable == 1`.
