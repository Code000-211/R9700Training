# Objectives, seeds, deltas, and the model

What a run trains against, how a loss seeds the backward, and the check the
forward driver runs before any walk.

## Objective

An `Objective` (in `include/r97/objective.h`) is `std::function<ObjectiveResult(const
ForwardTape&, const ParamSet&)>`: the loss. It reads the forward tape, accumulates the
loss (summed over the batch's windows), and returns an `ObjectiveResult`:

- `loss` — the sum, as the loss defines it;
- `scored` — how many scored tokens the loss covered;
- `seeds` — the cotangents that seed the backward, one `Seed` per chunk of buffer:
  `{buffer, row_lo, row_hi, cotangent}`, with `cotangent` of size `(row_hi - row_lo) *
  cols` of that buffer, laid out like the buffer.

A read on the tape either claims its gradient as differentiated (the last argument of
`ForwardTape::read`) or is detached. A differentiated read of a boundary must be answered
by a seed of the same result; a seed exists only for what the loss actually read as
differentiated. A loss with no differentiated read returns no seeds: it is forward-only.
A detached read always passes the check.

## Seed

A `Seed` names a boundary buffer, the rows of it, and the cotangent of those rows. The
backward core resolves it through `BoundaryNames` (in `include/r97/model.h`), the table
from a boundary's name to the buffer id and the rows the forward driver recorded for it,
and adds it under the cotangent handed down from the block above. A seed never overlaps
the upstream cotangent: they add.

## Delta

A `Delta` (in `include/r97/delta.h`) is the trainable kind — LoRA, or a later kind. It
maps the container's weight names to the trained parameters it starts (`init`), names the
gradients it contributes (`grad_prefix`), and crosses a step's parameters onto the cards
(`to_wire`, including the sites it drives). A delta never names a loss.

## Model

A `Model` (in `include/r97/step.h`) is one `Objective` and one `shared_ptr<Delta>`: the
loss a run trains against and the kind it trains. `make_model(const Config&)` (in
`src/model.cpp`) builds it from `Config::objective` and `Config::delta.kind`, both by
name. An unknown name is refused with the list of the registered ones.

The forward driver and the joins take a `Model` and never name a loss or a delta
themselves — the `Objective` owns the loss, the `Delta` owns the kind.

## What check_seeds enforces

`check_seeds(const ForwardTape&, const ObjectiveResult&)` runs before any walk. It reads
the tape's record of what the objective read (`RecordedTape` in `include/r97/model.h`;
`RecordedVectorTape` in `tests/support/` wraps the test `VectorTape` with it). It
refuses, naming the buffer:

1. a seed on a buffer that is not one of `ForwardTape::boundaries`;
2. a seed whose row range lies outside the window the forward driver recorded for the
   boundary, or whose cotangent does not have the size of its rows;
3. two seeds of one result that overlap on a buffer — the per-result rule; a seed and the
   cotangent handed down from the block above do not overlap, they add;
4. a differentiated read of a boundary that no seed of the result answers (a seed answers
   a read when their row ranges share a row; it need not cover the read — a loss may read
   its whole window and seed only the rows it scores, so a window whose ends are unscored
   is answered by the seed over its scored middle, and one wide read may be answered by
   several adjacent seeds);
5. a seed on a boundary the objective never read as differentiated.

A result with no seed at all is valid: it is a forward-only objective, so long as the
objective made no differentiated read.

The tape must expose its read record: a tape that does not derive from `RecordedTape` is
refused, so a driver cannot pass the check unrecorded.

## Adding an objective or a delta in three steps

1. **Write the file.** A new objective is one `.cpp` (plus its `.h`, as small as you can)
   under your issue's area: a function `ObjectiveResult(const ForwardTape&, const
   ParamSet&)` and its seeds. A new trainable kind is one file implementing `Delta`. Never
   branch on the new name in the forward driver, the backward core, or any existing file.
2. **Register it.** One line at file scope,
   `R97_REGISTER_OBJECTIVE(name, factory);` or `R97_REGISTER_DELTA(kind, factory);` — the
   factory takes `const Config&` and gives an `Objective` or a `std::shared_ptr<Delta>`.
   The `name`/`kind` must be a bare identifier; registering the same one twice is refused
   when the binary starts. The file goes in one of the three directories the library's glob
   collects — `src/`, `src/adjoint/host/` or `src/adjoint/device/` — so no existing file
   changes. `r97train objectives` (or `r97train deltas`) is where you see it.
3. **Document it.** Add the name and one line on what it does to the list below, and, for
   a new trainable kind, its meaning to `docs/ADAPTER.md`. Nothing else changes:
   `r97train objectives` and `r97train deltas` print whatever is registered, so the new
   name shows up there with no edit to the tools.

### Where a registration runs

A registration is a static initialiser in the registering file, so it runs only in a binary
whose link kept that file. A static archive drops a member nothing references, and a file
whose only export is its registration is exactly such a member: `r97` is a static archive,
so a loss or a kind that ships from `src/` is reachable only once the targets that list it
link the library whole (an object library, or `$<LINK_LIBRARY:WHOLE_ARCHIVE>`). That is a
decision about every target in the build, so it belongs to a build issue and not to the file
that adds a loss. What the registries in this repository exercise today is registered from
files their own binary compiles: the test objective and the test delta in the host test
binary, the subcommands in `r97train`.

## Registered today

- `lora` -- the trainable kind `docs/ADAPTER.md` describes: PEFT's A and B factors for each
  targeted linear (`src/delta_lora.cpp`).

`r97train objectives` and `r97train deltas` print the live list, and the issue that registers a
loss or a kind adds its line here and changes nothing else.

The registries in `r97train` and in a test binary are separate: the test objective and
test delta under `tests/support/` are registered only in the host test binary.
