# Device walk: an in-place read with no earlier writer is served the block's stored copy

2026-11-16. Issue 148. The device walk snapshots an operand **after** the op issues, so for an
in-place read (`h` read, `h` written by the same op) the snapshot held the op's **own output**, and
the row differentiated at its output. `R97_DUMP_DEVICE_PULL=1` named the case: on the live
container's 512-token walk, **111** reads have no earlier writer in their block, and **63** of them
are the defect -- every one `rmsnorm_had_quant_i8g` reading `h` (`named=1`, 5242880 bytes = 512 x
5120 x 2), the first at `block=63 pos=1429`, the position issue 145 identified independently. The
other 48 are `gdn_chunk_scan` reading `kv:2` (`named=0`), which the walk already zeroes: the engine
cleared the recurrent state before the replay, so that read enters cold.

The walk now serves each in-place read from the block's stored copy, as the host walk does (issue
145), and refuses by name when no stored copy covers the read. On this container the copy is the
plan's second boundary input, not the residual stream buffer -- the `h` the norm reads is the
block's second input. The count is held to the tape's own count, and the gradient's non-finite
leaves fall from 318 of 400 to 2 of 400 (A) and from 318 to 1 (B).

## The defect

`src/step_device.cpp`'s block walk keeps a per-block `block_wrote` set and fills it **after** the
read loop of each record. The branch `!block_wrote.count(read.buffer)` therefore means "no
**earlier** op in this block wrote it", and an op that writes the buffer it is reading falls into
it. The live pull in that branch was taken after the op issued, so it holds the op's own output:
for position 1429 the norm reads the bf16 `h` and writes its int8 codes over the same buffer, and
the pull read the codes as bf16. The walk then differentiated the row at its output.

The host walk's snapshotter (`src/step_host.cpp`) had the same gap and was fixed in issue 145: when
the read is in place and nothing earlier in the block wrote it, it reads the stored boundary back
from the `ResidualStore` and serves those bytes. The device walk had the naming (`5147d1b`) but not
the fix.

## The fix

`src/step_device.cpp`, in the `!block_wrote.count(read.buffer)` branch:

- The op's own write is detected by scanning the record for a written operand with the same buffer
  (the same rule the host uses; the tape does not always mark the read slot itself).
- `kv:` reads keep the existing cold-state zero.
- A non-`kv:` read with no earlier writer and no write of its own keeps the live pull: a frozen
  input or the boundary, where the pull is the value the op read.
- An in-place read is served from the stored copy the replay restored before the block issued:
  - the plan's second copy at the boundary, matched by `source_extra(layer)` exactly, read with
    `read_extra`;
  - the boundary residual, matched by the host's test (`named`, the boundary's byte count, and the
    `[tokens, hidden]` shape), read with `read(layer, host)` and offset when the operand's pointer
    lies inside the residual buffer.

  The second copy is tested first: on this container the `h` read is the block's second input, and
  the residual's own bytes would be the wrong value for it. The recorder already requires the copy
  at a boundary to be the buffer the block at it reads (`src/replay.cpp:348-352`), so the copy is
  the value the op consumed.
- A read matching neither refuses by name: no stored copy exists, and differentiating at the op's
  output is silent and wrong.
- `DeviceStepReport::in_place_restores` counts the reads served from a stored copy. The `kv:`
  reads are not counted (they enter zero, not restored).
- `R97_DUMP_RESTORE=1` prints `DEVRESTORE` with the block, position, op, buffer and byte count,
  and `extra=1` for the plan's second copy, mirroring the host's `RESTORE` line.
  `R97_DUMP_DEVICE_PULL=1` keeps its line: the pull still holds the op's output; the restore is
  what the walk serves instead.

## The check

`tests/gpu/device_step.cpp`'s `device_step_tape` derives the expected count from the probe's own
tape, not from a figure pinned to this container: per block, in record order, a read operand the
record itself also writes, with no earlier op in the block writing the buffer and not a `kv:`
state. It then holds `DeviceStepReport::in_place_restores` to that count (and requires it positive)
in the child, and checks the same two fields from the results JSON in the parent. The parent's
check runs immediately after the results headline, before the `all_finite` check this container
still fails without `R97_QUANT_GUARD=1`: issue 140's guard reports the overflowing quotient and
carries on by default.

## What was measured

Both runs are `device_step_tape` (512 tokens, the adapter seeded over both factors) on
`/srv/models/rad/qwen3.8-27b-iq5r-s-v4.rad`, one card, in one held GPU window; the second run is the
tree with this change and nothing else. The loss is identical to the last digit, so the forward is
the same run; every difference below is the walk's.

| figure | baseline (147afda) | with the fix |
| --- | --- | --- |
| `DEVPULL` lines | 111 | 111 |
| `DEVPULL` with `named=1` | 63 | 63 |
| `DEVRESTORE` lines | -- | 63 (`extra=1`) |
| `in_place_restores` of `in_place_expected` | -- | **63 of 63** |
| loss | 7371.591348378961 | 7371.591348378961 |
| loss_after (one AdamW step) | 6347.883966664209 | 7206.33404700259 |
| step (fwd / head / bwd) | 314.5 s (47.1 / 0.8 / 264.4) | 324.8 s (46.7 / 0.8 / 275.1) |
| `all_finite` (no `R97_QUANT_GUARD`) | false | false |
| first non-finite cotangent | `gdn_gated_rmsnorm`, `gdn_o`[0] at position **1143** | `gdn_gated_rmsnorm`, `gdn_o`[0] at position **9** |
| `n_a_finite_nonzero` / `n_b_finite_nonzero` | **82 / 400**, **82 / 400** | **398 / 400**, **399 / 400** |
| `n_zero_a` / `n_zero_b` | 0, 0 | 0, 0 |
| residual store | 640 MiB, 65 boundaries, 63 second copies | same |

The fixed tree was run twice. The two runs agree on every figure above except `loss_after`
(7206.33404700259 and 7198.305414118746), `a_changed` (283 and 286) and the step's wall split; the
counts, the first non-finite cotangent and the per-op non-finite counts are identical. That is the
device walk's run-to-run arithmetic on the finite leaves, not the fix. The table carries the run
with the `extra=1` diagnostic; the other run is the same tree without that one print.

The non-finite cotangent counts per op collapse with it: `rmsnorm_had_quant_i8g` 342/128 -> 4/128,
`gdn_chunk_scan` 273/48 -> 7/48, `gdn_gated_rmsnorm` 115/48 -> 1/48, `gdn_conv_prep` 312/48 ->
8/48. The first non-finite the walk meets moves from a mid-trunk block (position 1143) to position
9, in the last block the walk reaches: the snapshot at 1429 was poisoning every block the walk took
below it, and the restore removes that source.

The gradient is therefore **not** inert on the device, unlike the host branch (which the host-site
substitution never reached): the wrong snapshot was a large part of the non-finite gradient. The
descent after one AdamW step is smaller (-165.257 against -1023.71), which is consistent with a
gradient that is less poisoned rather than with a scale change; the loss still falls and no leaf
has a zero gradient (`n_zero_a`/`n_zero_b` = 0). Whether the remaining gradient is *correct* is the
row-level question issue 144's checks own, not this measurement's.

## What this does not claim

This does not fix the cotangent growth: issue 145 explains the growth as the rows' own Jacobians,
and issue 140 owns the quotient overflow. #140's fix is the mask pass's guard, not a numerical one:
its default reports an overflowing `cot / s` and carries on, so the unguarded case still fails
`all_finite` (two A leaves and one B leaf are non-finite, and the walk names
`gdn_gated_rmsnorm`/`gdn_o` at position 9); `R97_QUANT_GUARD=1` turns that report into a named
refusal, not into a finite gradient. What this measurement shows is that the walk no longer serves
the op's own output for any of the 63 reads, that the count is held to the tape's own count rather
than a figure pinned to this container, and that the fix moves the gradient's finiteness by two
orders of magnitude on the live container.

## Reproducing

```sh
/home/juup/ai-setups/aijuus/looping/local-radiance-window.sh open
/home/juup/ai-setups/aijuus/looping/local-radiance-window.sh run -- env \
    R97_MODEL_DIR=/srv/models/rad/qwen3.8-27b-iq5r-s-v4.rad \
    R97_TEST_ONLY=device_step_tape R97_DUMP_DEVICE_PULL=1 R97_DUMP_RESTORE=1 \
    ctest -V --test-dir build -R gpu_model_tests
/home/juup/ai-setups/aijuus/looping/local-radiance-window.sh close
```

The case still fails its `all_finite` check on this container without `R97_QUANT_GUARD=1`: #140's
default reports the overflowing quotient and carries on. With `R97_QUANT_GUARD=1` the same run
refuses by name at that quotient; it does not make it finite. That is #140, not this change.
