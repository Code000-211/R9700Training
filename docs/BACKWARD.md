# The backward core

`include/r97/backward.h`, `src/backward.cpp`. One recorded block, walked in reverse from a
list of seeds, with the cotangents kept per buffer version. It knows nothing about the model:
the caller gives it the block's `OpRecord`s, a snapshot provider, the `Seed`s of the objective
and the cotangent the block above handed down, and gets back the cotangent of the block's
inputs and the additions to a gradient `ParamSet`.

## What it is given

- `records` — the `OpRecord`s of one block, in append order, `block` and `position` as the tape
  recorded them. The snapshot provider is asked by `position` and the operand's schema index.
- `snapshots` — a `SnapshotProvider`. `at(position, operand)` answers the forward value the op
  saw: a read's value before the record's write, the written operand's value after it. A buffer
  written twice in a block has two values; this is why the key is the op, not the buffer.
  `StoredSnapshots` is the in-memory implementation tests use; a replay provider answers the
  same two keys without holding the block.
- `boundaries` — `BoundaryNames` from `include/r97/model.h`, filled by the caller. A seed names
  a boundary and the table resolves it to the buffer's row window. A seed on a name the table
  does not know, on rows outside the window, or on rows the block never wrote is refused and
  the message names the buffer.
- `seeds` — the objective's `Seed`s. Their cotangent is f32; the core widens it and tags the
  cotangent of the writing op with that operand's dtype, which is what the adjoint expects.
- `incoming` — `IncomingCotangent`, the cotangent the block above handed down: f64 and
  already carrying its values. It enters the same store a seed does and adds to whatever a
  seed on the same rows adds. Passing one seed on the output as a `Seed` and passing the same
  values as an `IncomingCotangent` give bit-identical walks.

## Versions

A block writes the same buffer more than once: a transient reused, an op in place, one op
writing rows `[0, 4)` and the next `[4, 8)`. Every write creates a **version** of the buffer,
and a cotangent belongs to `(buffer, rows, version)`:

- it is created on first use — a seed, the cotangent from above, or the first reader's `dc`;
- it is summed over every reader that saw that version;
- it is released once the writer's adjoint has run, and its bytes leave the report's count.

The first pass, in append order, records per row which version is live at each read. Reads are
taken before writes within one record, so an op that writes the buffer it reads still reads the
version that was live before it. The reverse pass then never hands a cotangent to a version
that is already gone: every reader of a version sits after its writer.

## Stops

`BackwardOptions::stops` names the ops whose path ends: their adjoint is not asked for, their
written operands' cotangents are consumed, and nothing reaches their reads. The registry has
no stop kind (`docs/adjoints/gdn_prep.md` §5), so the marker lives with the caller: an empty
adjoint that adds nothing is not enough, because the walk would keep going with zeros. The
core does not skip an op whose cotangent happens to be all zeros; a cotangent that arrived is
a cotangent, and only a stop or the end of the block ends the path.

## Errors

Every refusal is a `std::runtime_error` whose text begins `r97: backward:` and names the thing
that could not be placed:

- a seed or an incoming cotangent on a name that is not a boundary, or on rows the block never
  wrote;
- an op on the path with no adjoint — the error comes from `adjoint_find` and carries the op's
  name and parameters;
- an operand with no snapshot, or a snapshot with another shape than the record's operand;
- a buffer recorded with two shapes, or an operand whose row range leaves its buffer.

## The report

`BackwardReport` gives the peak bytes of tensor payload the walk held at once: the forward
values of one op (`peak_snapshot_bytes`, the only snapshots the core holds itself) and the
cotangents (`peak_cotangent_bytes`: the live version store, the block-input accumulators, and
the `cot`/`dc` vectors of the op being adjointed). Releasing a version after its adjoint is
what keeps the second number below the sum of every write in the block.

## The block's input

A read whose rows go back past the block is a block input. Its cotangent is accumulated dense
over the buffer: `InputCotangent` carries the whole shape, the values of the rows the block
read, and zeros elsewhere, so the block above can consume it as an `IncomingCotangent` without
re-indexing. The result lists the inputs sorted by buffer name.

## The tests

`tests/host/backward_core.cpp` builds its blocks from `add`, which already has an adjoint, and
from stand-in adjoints registered in the test; nothing there needs a model. A chain, a diamond,
an in-place op, a reused transient and a partial row range each check a hand derivation and a
central difference of the same program. Two seeds give the sum of the two single-seed walks and
agree with finite differences; one seed on the output is bit-identical to the same cotangent
handed down from above. The missing-adjoint error is checked for the op's name and parameters,
the report for its two peaks, and a stop for ending the path on its inputs.
