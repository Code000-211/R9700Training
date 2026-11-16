# The five in-place ops, FD-checked with their output aliased over their input

2026-11-16. Issue 144: `lora_expand`, `rmsnorm_had_quant_i8g`, `gdn_conv_prep`, `rope` and
`kv_store` are the five ops the live tape marks `written=1 in_place=1` (288, 127, 48, 32 and 16
of one 512-token step's records), and none of them had a finite difference with the output
aliased over the input. The one aliased FD the suite had was `rmsnorm_quant_fp8` via `in_place`
`{1}`; the rest of the aliasing was checked only in its unaliased shape.

## The gap

An in-place operand is one schema slot holding two values: the op reads the pre-op one and
writes the post-op one. To express the aliased case, a record marks the operand `written` and
`in_place`, the check is handed the pre-op value among its reads and the slot's post-op value
among the twin's outputs, and `check_adjoint` is told the slot is in place. The harness already
had the shape of this (`tests/support/adjoint_check.h`, `R97_FD_PREOP`): the finite differences
run at the twin's output (the default) and at the pre-op value the walk serves into the slot
(`step_device.cpp:1251` serves `before`), because those are two genuinely different questions
and production uses the second.

Two things stood between that shape and the five ops:

1. **The harness could not build the pre-op forward.** In `check_adjoint`, the forward is
   assembled write by write from the twin's outputs, and the in-place write's output was not
   consumed when the pre-op value took the slot's place. Every write after an in-place one
   therefore read the output that *preceded* it. Any record whose in-place write is not the
   last write -- all of them here, since the aliased slot is the op's own input -- got a
   shifted forward at the walk's point, before any row ran.

2. **The rows did not read the mark.** A row that ignores `in_place` differentiates the slot
   as if the write were a fresh buffer: the twin has no post-op value to write, the write's
   cotangent has no slot to land in, and the cotangents of the other writes are indexed as if
   the write were not there.

## What was added

### The harness (`tests/support/adjoint_check.h`)

The forward now consumes the twin's output for every write, in write order, in-place write
included: the pre-op value takes the slot's place in `fwd`, but the writes after it take
their outputs from past it.

### The rows (`src/adjoint/host/`)

Each of the five now reads the mark, the way the record the live tape gives it carries it.

- **`lora.cpp`** -- the twin adds the branch to the buffer's old value (a base row keeps its
  old value unrounded, the kernel's rule), and the row fills the slot's direct path: the
  buffer the write replaces reads its own cotangent back, unchanged. The walk passes that on
  when a row leaves the slot empty, so filling it says the same thing -- and a check that
  calls the row without the walk sees it either way. The layout check takes the slot's value
  as the write's own rows, dense in `M*N`: the walk's own records carry it in the lane split
  (`[M, N/lanes]`), so the rows and the element count are the check, not the second
  dimension.
- **`kv_store.cpp`** -- the twin starts the pool from the old value, and the row routes the
  pool's cotangent only to the rows the store does not overwrite; the rows the write covers
  take the k's and v's cotangents, which the row already computes.
- **`gdn_prep.cpp`** -- the twin addresses its reads by schema position, skipping the writes
  the caller's read list leaves out: a plain write always, and the in-place state only when
  the caller did not carry it between `dt_bias` and `cu`. The forward served is cold, so the
  carried value is positioned, not used.
- **`quant.cpp`** -- the fused twin writes the part's output into the aliased slot, where the
  schema's write order puts it (the buffer's bytes are the int8 codes over a bf16
  allocation, a reinterpretation nothing differentiates; the part's output is the value the
  backward may read the slot for). The row counts the slot among the part's reads, indexes
  the codes' and scale's cotangents by the writes' schema order rather than assuming the
  slot is not one of them, and routes the slot's own cotangent *through the part*: the slot
  holds `part(x)`, not `x`, so a passthrough would differentiate it at the identity.

### The checks (`tests/host/adjoint_in_place_fd.cpp`)

One test per op, each stating the aliased record the tape itself gives -- the mark on the
record, not a test's convenience -- and running what can be run at it: the twin against
radiance's host kernel, where the kernel reads the pre-op value the way the engine does
(`lora_expand` adds to `y`, `rope` rotates in place, `kv_store` overwrites its rows,
`gdn_conv_prep`'s cold start overwrites the state), and the host adjoint against central
differences of the twin, at both points the slot can hold. The rows run on the twins they
themselves serve, so a row that differentiates at the wrong point, or forgets the value an
aliased slot carries, fails here the way it would fail in the walk.

`rmsnorm_had_quant_i8g` has no host kernel (device only), and its row reads the slot's value,
so it is checked at the walk's point only: at the twin's output the row would differentiate
the norm at its own output, a point production never serves. Its record is also stated for
what the aliasing does not move: the frozen slots keep their schema positions, the gain at 1
and the base point at 4 under `_ste`.

The measured errors, at the points the rows are held:

```
lora_expand         post-op 1e-10 .. 3e-9   pre-op 1e-11 .. 3e-9   kernel max err 0.015
rope                post-op 4e-11 .. 4e-9   pre-op 3e-10 .. 6e-9   kernel max err 0.0038
kv_store            post-op 3e-10 .. 3e-10  pre-op 5e-11 .. 2e-8   kernel max err 0
gdn_conv_prep       post-op 4e-11 .. 7e-10  pre-op 4e-11 .. 3e-9   kernel max err 0.0039
rmsnorm_had_quant_i8g_ste   pre-op 5e-10 .. 1e-9
```

All host tests pass (295), including the walk's own: the rows' new paths are keyed on the
tape's mark, so a record without it -- the program-built graphs the reference step replays
from -- takes the old shape, and the walk's passthrough and a row that fills the slot agree
where the write's direct path is the identity.

## What this does not cover

The device rows do not read the mark. The device backward serves the same aliased records
(`step_device.cpp` sets the same passthrough), and the device `quant.hip` fused rows index
their writes the way the host rows did: a record the tape marks in place, served on the card,
still gets the unaliased treatment. The host rows are now what the aliased record asks of
them; the device rows are the next gap in the same line.
