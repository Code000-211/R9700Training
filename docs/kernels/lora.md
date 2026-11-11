# The lora family on the card

`src/adjoint/device/lora.hip` is the device adjoint of the adapter radiance serves over the
rotated int8 trunk: `lora_shrink` and `lora_expand`, on the coded form of the first and the only
form of the second. This one is the family after `qk` in `src/adjoint/device` (`add`,
`elementwise`, `qk`, `lora`; `docs/kernels/README.md` for how a family is added), it answers
through the same plugin registry as the host rows, and it is the family whose backward pass
produces the *trained* things: `dA` and `dB`, accumulated in f32 over all of the step's tokens,
plus the adapter's own contribution to `dx`.

This family has no fused op to compose: radiance issues the adapter as the two ops it names
(`ref_lora.cpp`), so each row here is held to its own host row (`src/adjoint/host/lora.cpp`,
`docs/adjoints/lora.md`) on the same inputs, and the composition — the expand's cotangent of `t`
becoming the shrink's — is the trainer's two calls in that order.

## The record

The shrink, in schema order (`ref_registry.cpp`):

| # | buffer    | dtype | shape              |                                                                     |
|---|-----------|-------|--------------------|---------------------------------------------------------------------|
| 0 | `a`       | i8    | `[M, K]`           | the rotated codes, read                                             |
| 1 | `a_scale` | f32   | `[M, ceil(K/group)]` | the group scale, read                                             |
| 2 | `slot`    | i32   | `[M]`              | frozen: the pool row of each token, or a base row                   |
| 3 | `lora_a`  | bf16  | `[S, R, K]`        | frozen                                                              |
| 4 | `t`       | f32   | `[M, R]`           | written                                                             |

and the expand:

| # | buffer   | dtype | shape       |                                                                  |
|---|----------|-------|-------------|------------------------------------------------------------------|
| 0 | `t`      | f32   | `[M, R]`    | read                                                             |
| 1 | `slot`   | i32   | `[M]`       | frozen                                                           |
| 2 | `lora_b` | bf16  | `[S, N, R]` | frozen                                                           |
| 3 | `scale`  | f32   | `[S, N]`    | frozen: `alpha / r` per adapter row and output row               |
| 4 | `y`      | bf16  | `[M, N]`    | inout: the adjoint leaves `y`'s base value alone                 |

`M` tokens (`K` in, `N` out for the expand), `R` the pool's packed columns, `S` the pool's rows,
`group` the trunk's scale group width. The frozen sets are the ones the host rows declare, `{2, 3}`
and `{1, 2, 3}`: `slot` is the tape's, and the pool's factors and per-row scale are constants of
the step.

## The coded trunk

The value the shrink differentiates is `x = code · scale`, so a token's row is
`x[m, k] = a[m, k] · a_scale[m, k / group]`, and a base row — a `slot` of `-1`, or a value outside
`0 .. S-1` — carries nothing and contributes nothing (`fill_slots` in the source decides that once
on the host, so no kernel has to).

The group width is the one the stream was written with, the power of two whose ceiling division
gives `a_scale`'s column count (`group_width` in `src/adjoint/host/lora.cpp`, and `docs/adjoints/lora.md`
§1 for why `K = 300` is `128, 128, 44` and not three groups of 100). The record's `group`
parameter is the consistency statement radiance's row treats it as: a count that disagrees with
the stream's columns is a record the kernel would refuse with `RAD_E_SHAPE`, and the row lays it
out for no one.

`A` and `B` are the *packed* buffers the pool holds: `R` columns per row even where one site's
module owns only its own `r` of them (`docs/ADAPTER.md` §2, `PackedSite`). Neither kernel assumes
otherwise — a zero region is a zero, and `dA`/`dB` come out zero on the columns no module reads
and the rows no module writes.

## What the kernels do

`shrink_dx_kernel`, one block per token, one thread per column of a 256-wide tile:

    g[m, k] = sum_j cot_t[m, j] * A[s, j, k]
    dx[m, k] = g[m, k] * scale[m, k / group]          the cotangent of the code
    dscale[m, grp] += sum_{k in grp} g[m, k] * a[m, k]

The token's row of `cot_t` goes to shared memory when the pool is no wider than a shared row
(1024 columns, far past the model's 32); a wider pool reads it from global instead, so no pool
width is refused. The group split is a run of columns, so a tile holds the groups it overlaps
whole or in part, and the parts are summed by one thread per group and added into an f32 staging
buffer with `atomicAdd` — the staging is zeroed first, which is what lets a group span tiles.

`shrink_da_kernel` is `dA[j, k] = sum_{m valid} cot_t[m, j] · x[m, k]`: one thread per column,
its R row sums in a register array (`kMaxJ` at a time, so any `R` works), a chunk of 64 token rows
per block, and `atomicAdd` into an f32 staging buffer for the chunks that share a column.

`expand_dt_kernel` is the cotangent of the expand's input:

    dt[m, j] = sum_n cot_y[m, n] * scale[s, n] * B[s, n, j]        dt = 0 on a base row

`expand_db_kernel` is the pool's:

    dB[n, j] = sum_{m valid} cot_y[m, n] * scale[s, n] * t[m, j]

`shrink_da_kernel` and `expand_db_kernel` write into the *step's* accumulator rather than into a
buffer of the call's: the shrink's `atomicAdd` and the expand's one-thread-per-entry add both
accumulate into the f32 buffer `grad_stage.h` holds for that leaf, so a later call of the same step
adds to the earlier call's total on the card instead of to a zeroed buffer, and nothing is rounded
between them.

## The gradients

`dA` is the leaf `<prefix>.lora_A`, shape `{R, K}`; `dB` is `<prefix>.lora_B`, shape `{N, R}`.
Those are the names and shapes the host rows use (`docs/adjoints/lora.md` §4), and each token's
contribution lands in its own pool row's block — for the model's pools, which have one slot, that
is the whole leaf.

The step's sum runs on the card, and the leaf stays off it. `dA` and `dB` are host memory by the
contract (`include/r97/params.h`), so a row that copied a call's sum into the leaf when the call
ended would sum the step in f32 only after rounding every call to bf16. Instead the row asks the
step's stage for the leaf's accumulator (`src/adjoint/device/grad_stage.h`), which is one f32
buffer per leaf for as long as the step lasts: the first ask of a step allocates and zeroes it,
every later call — this site's next chunk of tokens, or another call that shares the leaf — adds
into the same buffer, and no rounding happens in between.

The walk ends the step with `grad_stage_flush(&grads)`: each accumulator is then rounded once, from
its f32 total, to bf16 — the dtype a gradient has on the card — and added to the leaf of that name,
which the flush creates from the name and shape the stage recorded. `grad_stage_discard` is the
other way to end a step, for a walk whose gradients nobody wants. Both belong to the walk, not to
the row: the stage's lifetime is the step's, so a walk that ends no step would have the next step's
first call add to the old total.

The staging costs the card `4 · R · K` and `4 · N · R` bytes — 320 KiB and 768 KiB for the model's
`q_proj` at rank 16 — and that is the same for every call of the step. It is not the call's scratch:
a call's scratch is freed when the call ends, while the stage lives until the step does, so it has
its own count rather than a line in the scratch ledger.

With `B = 0` — the adapter at its start (`docs/ADAPTER.md` §4) — nothing above the expand asks
for its input's cotangent, so `dt` is zero, and the shrink, handed that zero, gives `dA = 0`. `dB`
is *not* zero there: it is what moves `B` off zero. The test asserts both halves.

## bf16 and f32

- `A`, `B`, `y` are bf16 as the trunk stores them, and the rounded values are what the kernel
  reads; `t`, `cot_t`, `cot_y`, the group scale and `alpha / r` are read as f32, the width the
  tape records them in.
- The cotangent of a *code* is an f32 number, not a code: it is `g · scale`, and rounding it to i8
  or to bf16 would be a quantisation the reference does not make. It leaves as f32.
- The gradients are the opposite case: a real gradient is bf16 on the card, so the step's f32
  sum is rounded once, when the walk ends the step. A call of a step rounds nothing, which is the
  whole point of the staging: `docs/kernels/README.md`'s "The harness" says why the room the
  harness's tolerance gives a kernel is the room a kernel that rounds to bf16 and accumulates in
  f32 has.
- The sums are f32 throughout, including the atomics; the reference is f64, so a gradient entry
  carries both the bf16 rounding of its own value and the f32 noise of the terms that made it.

## The test

`tests/gpu/adjoint_lora.cpp` (`R97_DEVICE`, the `gpu` tier; no container is read — the real
widths are numbers from the design, so it carries no `model` label) runs both rows against the host
rows on the same random records and holds them to:

- the harness's bf16 tolerance, `1e-3 + 0.02 · |host|`, on every cotangent slot —
  `kAbsTol`, `kRelTol` in `tests/support/device_check.h`;
- for `dA`/`dB`, the same bound measured against the block's *largest* entry. A gradient entry is
  a sum that cancels, so the error of the sum follows the terms it summed and not the entry it
  landed in; the entry-by-entry relative bound would fail on the entries that cancel to near zero
  in both sides. The test prints the worst difference in units of the block (`0.002`–`0.003` at
  every site). The host's own gradient check reads it the same way, against the block's largest
  entry, at `1e-5` — its leaf is f32 and its reference f64, where the card's block is bf16
  (`docs/adjoints/lora.md` §4).

The sites: the tiny fixture's `q_proj` at rank 16 with a two-row pool and `down_proj` at rank 4
(hidden 128, ffn 256), `up_proj` at `K = 300` — three groups of 128, 128 and 44, the shape that
catches a width read as `ceil(K / groups) = 100` — a pool of 2048 columns (wider than the shared
row, so the global path is covered), and the real model's `q_proj` (K 5120, N 12288) at ranks 16
and 4 with 1024 and 4096 tokens, `down_proj` (K 17408, N 5120) with a batch of unequal lengths —
448 + 320 + 256, which the adapter sees as its total token count of 1024, since it has no
per-sequence structure — and `ffn_gate_up` (N 34816, rank 32: two modules of 16 in one pool, with
the zero regions `docs/ADAPTER.md` §2 describes).

It also asserts the teeth: `dA` is named, shaped and filled in (not zero by accident); a step's
tokens in two calls leave the `ParamSet` untouched until the walk ends it, put one accumulator on
the card (the second call does not make a second one) and come out as twice the one-call step, to
within that step's one rounding; a `lora_shrink` record of the float trunk
(`dtype` with no `a_scale`) has no device row; a deliberately wrong reference that drops the coded
record's scale is rejected rather than accepted; and with `B = 0`, `dt` and `dA` are exactly zero
while `dB` is filled.

## Measurement

`./build/r97_gpu_tests --bench` on one R9700 (gfx1201, 32 GB), 6 reps at 1024 tokens and 3 at 4096,
mean of the reps. This machine is shared — other work holds the other card and the CPU — and
another job's run only adds to a row's time, so each number below is the fastest that column came
out in the passes taken for this doc. The rows still agree with each other, which is what a pass
is read for: rank does not move them, `N` scales them, four times the tokens cost 3.8 to 3.9 times
the call. A pass a neighbour's run was on comes out a third to two thirds above these.

"call" is the adjoint as the contract hands it (the number the harness's own bench prints); "with
dA/dB" is the same call with the gradient `ParamSet` handed to it, so the kernels accumulate `dA`
and `dB` into the step's f32 buffers on the card and nothing is rounded or copied out; "step's
flush" is `grad_stage_flush`, the end of a step — one copy of each accumulated block into a leaf,
one rounding to bf16 per entry, one add — which a step pays once, not once per call; "forward
twin" is the op's forward in f64 on the host (`src/adjoint/host/lora.cpp`), the reference the
gradient is graded against.

| shape                                  | op     | call (ms) | with dA/dB (ms) | step's flush (ms) | forward twin (ms) | scratch peak |
|----------------------------------------|--------|-----------|-----------------|------------------|-------------------|--------------|
| `q_proj` r16, 1024 tokens              | shrink |      46.2 |            47.0 |             0.37 |               393 |     40.7 MiB |
| `q_proj` r16, 1024 tokens              | expand |      57.7 |            58.7 |             0.45 |               230 |     48.9 MiB |
| `q_proj` r16, 4096 tokens              | shrink |     178.0 |           176.7 |             0.37 |              1569 |    161.8 MiB |
| `q_proj` r16, 4096 tokens              | expand |     222.7 |           310.2 |             0.88 |               952 |    193.3 MiB |
| `down_proj` r16, 448+320+256 tokens    | shrink |     157.8 |           156.4 |             0.80 |              1414 |    138.2 MiB |
| `down_proj` r16, 448+320+256 tokens    | expand |      13.8 |            14.0 |             0.24 |              96.4 |     20.5 MiB |
| `q_proj` r4, 1024 tokens               | shrink |      46.0 |            47.4 |             0.17 |              97.9 |     40.4 MiB |
| `q_proj` r4, 1024 tokens               | expand |      57.6 |            57.2 |             0.20 |               107 |     48.3 MiB |
| `ffn_gate_up` r32, 1024 tokens         | shrink |      47.3 |            47.8 |             0.62 |               783 |     41.1 MiB |
| `ffn_gate_up` r32, 1024 tokens         | expand |     169.5 |           173.0 |             2.21 |              1093 |    140.6 MiB |

The reading: the call is bounded by moving the operands, not by the MACs. A 1024-token `q_proj`
call hands over 42 MB of f32 operands and takes back 21 MB, and costs 46 ms; the adapter's
gradient arithmetic — `dA` is a 16 × 5120 block summed over 1024 rows, `dB` a 12288 × 16 block —
is the call's own arithmetic into a buffer bought at the step's first ask, and it stays inside the
passes' noise: −1.3 to +3.5 ms across the six shapes at 1024 tokens and the shrink at 4096. The
one row the gradient does cost is the expand at 4096 tokens, 87 ms above its call: `dB`'s kernel
gives one thread each (n, j) and each of them walks every token of `cot y` with a stride of `N`, so
the walk touches the whole 201 MB of it, past the card's cache, where the 50 MB of the 1024-token
row stays in it. What a step pays on top of its calls is the flush: 0.2 to 0.9 ms for one module's
block — 320 KiB of `dA` or 768 KiB of `dB` at `q_proj` rank 16 — and 2.2 ms for `ffn_gate_up`'s
two (640 KiB and 4.3 MiB).

Four times the tokens cost 3.8 to 3.9 times the call, which is what a copy-bound call does. The
rank barely moves it: rank 4 and rank 32 are within a few percent of rank 16 at the same widths,
and the expand follows `N` (`ffn_gate_up`'s 34816 against `q_proj`'s 12288 is 2.8 times the money
and 2.9 times the time). The forward twin is 2× to 17× the call because it is the same arithmetic
in f64 on one host thread.

The scratch is the price of that: a token-major operand and its cotangent as f32, which is
`2 · 4 · M · K` bytes for the shrink and `4 · M · N` for the expand (161.8 MiB and 193.3 MiB at
4096 tokens), plus the pool's buffers — `A` and `dA`, `B` and `dB`, the group scale and its
cotangent, `t` and its cotangent, the slots. The codes are uploaded as f32, the value their
recorded dtype denotes (for `i8` the integer, for `fp8` the decoded value), so one float path
serves both coded forms; for the `i8` form the codes are the largest single piece of that scratch
and the one place a later change would save 4×. `R` costs the shrink's scratch nothing now that
`dA` is the step's staging: rank 16 and rank 4 come out 0.3 MiB apart, the staging they hold. The
staging is not in that ledger — `4 · R · K` bytes for `dA` and `4 · N · R` for `dB`, on the card
for the step's length and counted by `grad_stage_bytes()` — because a call's scratch is freed when
the call returns and this outlives it.

No speed target is set. The measurement is the deliverable.

## What it does not serve

- The **float trunk's** `lora_shrink` — the same op over a bf16 activation, with `a_scale` an
  absent operand. Serving it would mean a second row and a second predicate; the trainer's trunk
  is the coded one (`docs/ADAPTER.md` §1), so the row is registered through `coded_act` and the
  host row answers for the other (`adjoint_find` reports *no device adjoint for op 'lora_shrink'*
  with that record's `dtype` in the reason).
- **`lora_expand_mm`**, the same op with `t` rounded to bf16 before the sum. Its adjoint is the
  same function on the host (`rounding to bf16 is a step function, zero derivative almost
  everywhere`, `docs/adjoints/lora.md` §3), so registering this row under that name would serve
  it too; it is left on the host because this issue names the two ops, and one line registers it
  when a trainer issues it.
- **The walk that ends a step**: the backward core is issue 33 and is not written yet, so nothing
  under `src/` calls `grad_stage_flush`; the gpu test calls it where the core will, and the core must
  call it (or `grad_stage_discard`) at the end of every step, or the next step's gradients are added
  to the step before it.

## Files

- `src/adjoint/device/lora.hip` — the family's kernels and its two rows, in `libr97`.
- `src/adjoint/device/grad_stage.h`, `src/adjoint/device/grad_stage.cpp` — the step's f32
  accumulators and the copy into the gradient `ParamSet` that ends a step.
- `tests/gpu/adjoint_lora.cpp` — the fixture and real-width cases, the gradients, the teeth.
- `docs/kernels/lora.md` — this file.
