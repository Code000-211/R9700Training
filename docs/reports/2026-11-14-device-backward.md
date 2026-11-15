# The device backward: a real step on the 27B container

2026-11-14. One AdamW step on `qwen3.8-27b-iq5r-s-v4.rad` with the adapter's gradient
computed entirely on the card (tracker issue #55, plan R97 #53). Two cases, and their
pass conditions are **opposite** by design: the 512-token case starts with the adapter
seeded over both factors, the 4096-token case at the seed-0 start where `lora_B` is
exactly zero.

Both cases are `tests/gpu/device_step.cpp`; the assertions quoted below are that file's
(`:965-978`), not a reviewer's.

> **Superseding summary (2026-11-15).** Several sections below were written as the
> measurements arrived and quote rates that later runs replaced. **The authoritative
> numbers are these**, and any conflicting figure further down is superseded:
>
> - The container has **64 decoder layers** and the walk covers **1458 positions**, so one
>   layer is **22.78 records**. The dump corroborates it: 48 `gdn_gated_rmsnorm` (linear)
>   + 16 `rope` layers (full) = 64, a 3:1 hybrid.
> - The walk's cotangent grows **~0.140 decades per record (~1.38x)**, flat across both the
>   21- and the 49-record gaps: **~1.55e3 per decoder layer** (3.190 decades).
> - The true gradient grows **1.24x per decoder layer**, so the walk carries **~1.25e3 too
>   much per layer**. That is #55's defect. Compounded over the 11 layers from block 40 to
>   block 51 it gives 1.19e34 against a measured over-statement of 6.9e31 -- the same order.
> - The walk clips at f32's ceiling `3.4e38` **11.1 layers** below the window's top record
>   (measured: 12.0); everything below reads NaN because it saturated, not because growth
>   stopped.
> - **Retracted, do not reintroduce:** `~984x per block`, `~2.70x per block`,
>   `~3.9x per block`, `~250x per-block excess`, `365x`, `2e33`, `~76`, "no per-block factor
>   exists", and the brief `~1.8e13 per layer`. The `~984x` came from dividing a logit-space
>   cotangent by a weight-space gradient; the `~1.8e13` from calling a 4-layer `rope` period
>   one layer. `~1.6e3` was right; the derivations behind it were not.
>
> The measurement itself is in "The per-layer factor, measured: ~1.6e3, and it is real".

## 512 tokens, seeded start

`/tmp/r97-dstep-device_step_tape-23nofj/device_step_tape.json`, written 2026-11-14
18:35:36 by the 18:03 window.

**The case's own criteria — passed:** `a_all_nonzero true`, `b_all_nonzero true`,
`a_changed 400 == n_a`, `b_changed 400 == n_b`, `names_match true`, `shapes_match true`.
Equivalently `n_zero_a == n_zero_b == 0`: the adapter gradient exists on every one of the
800 leaves of the real container, which is what the ticket asks for.

The step: `loss 7371.5913` over `scored 511` tokens, `loss_after 6347.8839`,
`updates_done 1`, `opt_applied true`, `step_seconds 622.2` (forward 46.8, backward 572.2,
head 0.8 across `head_chunks 8` for `head_rows 511`), `tok_s 0.82`. Memory:
`vram_after_load 19750043648` (18.39 GiB), `vram_peak 21014626304` (19.57 GiB),
`peak_snapshot_bytes 2187526160` (2.04 GiB, of which `peak_snapshot_store_bytes` 1.838 GiB
and `shared_snapshot_bytes` 1.394 GiB), `peak_cotangent_bytes 490864656` (0.46 GiB),
`grad_stage_bytes 628097024` (0.59 GiB), `residual_store_bytes 671088640` (0.63 GiB of a
14.51 GiB budget, over `residual_boundaries 65` and `residual_extras 63`). The gate record
`gate_had_quant_i8g` served the tape's real layout (codes `[512, 6144]`, a 48-group scale,
the bf16 output), which is the row the earlier runs stopped on.

### The non-finite values: not the forward, the pull

`all_finite false`, with
`non_finite {op "rmsnorm", position 1458, operand "x", forward_operand "h",
forward_operand_index 2, forward_value_index 62, from_row true}`, and
`first_nonzero_a_finite false` / `first_nonzero_b_finite false`.

Neither branch of the test asserts anything about finiteness (`:965-978`), so this was an
observation rather than a case failure — but it was a real defect, and it is **not in the
forward**. A scan of every written operand at its issue, on the issuing lane, finds the head's
values finite:

```
DUMPFWD probe op=add pos=1457 buffer=x     count=2621440 maxabs=14720   nonfinite_at=-1
DUMPFWD probe op=rmsnorm pos=1458 buffer=h count=2621440 maxabs=11.0625 nonfinite_at=-1
DUMPFWD probe op=gather_rows pos=1459 buffer=h_out maxabs=11.0625      nonfinite_at=-1
DUMPFWD probe op=logits_gemm_vocab pos=1460 buffer=logits maxabs=13.4599 nonfinite_at=-1
```

The pull the join performs **after `serve()` returns**, reading the same pointers and byte
counts, finds something else:

```
DUMPPULL pos=1458 op=rmsnorm buffer=x operand=0 card=[-0.125 -0.160156 -7.86957e+33 ...] host=[same]
DUMPPULL pos=1458 op=rmsnorm buffer=h operand=2 card=[-37.75 3.01562 42.25 ...]         host=[same]
```

Card and host agree with each other and disagree with the forward. **The engine overwrites the
head's buffers after their ops issue and before the join reads them** — the block boundaries
survive because the residual store keeps them, and the head has no such store. The host join is
unaffected because `tape.keep_contents(true)` copies at issue.

That also bounds the defect: the descent check passed (`loss 7371.5913` -> `6347.8839`, finite),
which a globally NaN gradient could not do, so the staleness reaches the head's rows and part of
the adapter rather than breaking the walk.

**It is cured by capturing the head's operands at issue**, not by containing the symptom, and the
criterion for "the head's records" is structural — every record issued after the tape has recorded
`meta.layers` blocks — so it depends on no position and no count.

### Which run this is, and which one is still to come

This JSON is the **zero-gradient acceptance**, and it was measured on a binary that
predates the read-set fix (`e9eae50`, `src/step_device.cpp:737`: the buffer read outside
the head walk is the logits, not the hidden state). On that binary the head norm's writer
was unprovable, so `derive_read_sets` degraded to everything-readable and the `rmsnorm`
row counters read `0/0/0` — which is why `all_finite false` here is the inf flowing into
every gradient instead of being confined to the rows a consumer reads.

**So the two 512 results are not the same result with different `all_finite`.** This run
proves the ticket's core claim — the adapter gradient exists on all 800 leaves — and
nothing about the read set.

**The read-set acceptance's expected numbers were wrong, and the diagnostic is what
overturned them.** The device step prints the head walk's decision before walking it:

```
r97: head read set: 1 of 2 records proven [rmsnorm@1458 512 rows] [gather_rows@1459 unproven]
r97: head read set: 1 of 2 records proven [rmsnorm@1458 4096 rows] [gather_rows@1459 unproven]
```

The derivation **does** prove the head norm, and the gather's index list covers **every
row** — 512 of 512, 4096 of 4096. So no row of the final norm is unreadable, nothing may
be zeroed, and the inf at `h` row 0 sits in a row the objective reads. `all_finite false`
is then the honest outcome of the rule, not a failure of it: **the containment the read set
offers is not available on this path.** The earlier expectation of `511 zeroed / 1 readable`
came from assuming the gather selects only the last row; that premise is wrong, and the
counters said so before the criteria could hide it.

What the read set is worth here is therefore the *loud refusal* it produces for a
non-finite row a consumer reads — which is the correct behaviour, and which is why the
forward inf below is a defect to fix rather than one to contain.

### What the oracle does not say

`docs/kernels/README.md`'s host reference asserts the boundary and adapter floors, which
throw; the issue's 0.999 correlation / 2% criterion is a *separate* measurement and is
**not claimed** by this report or by the case. The step's own evidence is that the
gradient is non-zero on every leaf, with the right names and shapes, and that one AdamW
step moves the loss.

## 4096 tokens, zero-B start

**The case's contract is the opposite of the 512's:** `a_all_zero == true`,
`b_all_nonzero == true`, `a_changed == 0`, `b_changed == n_b`. At the seed-0 start
`lora_B` is exactly zero, so no cotangent reaches `lora_A`'s factor
(`tests/gpu/device_step.cpp:33-44`): **`n_zero_a == 400` is the expected result at 4096,
not a regression**, and the walk must not be changed to make it non-zero.

### The memory projection, before the run

| term | 512 measured | ×8 at 4096 | where |
|---|---|---|---|
| forward peak, incl. 15.13 GiB of weights | — | 20.2 GiB (`2026-11-12-device-forward.md`, this batch) | VRAM |
| residual store | 0.63 GiB | 5.0 GiB (128 × 4096 × 5120 × 2 = 5.37 GB) | VRAM, budget 14.51 GiB |
| grad stage | 0.59 GiB | 0.59 GiB (dA/dB are `[R,K]`: token-independent) | VRAM |
| snapshot store | 1.84 GiB | 14.70 GiB (shared 1.39 → 11.15) | **host RAM** |
| cotangent | 0.46 GiB | 3.66 GiB | host RAM |

**VRAM 25.8 GiB of 31.86 GiB; host 18.4 GiB of ~55 GiB free.** The distinction matters:
`peak_snapshot_store_bytes` is `snaps.bytes() + before_bytes` — host-side `AdValue` copies
(`src/step_device.cpp:912-925`), which the case's own comment says at `:50-51` — while
`vram_peak` comes from `rad_mem_get_info`. Only the second can OOM the card, so a ×8 on
the store is a host-RAM cost, not a VRAM one. Counting it as VRAM gives a spurious ~40 GiB
and a prediction of OOM that the card never sees.

**Two findings carried out of that projection:**

1. **The guard's 26 GiB declaration omits the grad stage** (0.59 GiB). At 4096 it sits
   inside the margin, but the declaration is under-stated, and the next token count may not
   be as forgiving. A budget that is right by luck is worth naming.
2. **`R97_NO_SHARED_SNAPSHOTS=1` is not a memory lever.** It exists to exercise the
   unshared path — widening every operand per block (`src/step_device.cpp:631-640`) — which
   increases work rather than shrinking the store. The store is re-readable from the card
   because it is host-side; the lever for host RAM is not keeping it, not widening it twice.

### The run

**First attempt (2026-11-14 20:24, 6000 s window): the case failed its descent check and
left no results file.** The window ran to completion — prefill of 4096 positions in 3 s, the
step itself, then `device_step_seq4096: threw: check failed: after.loss < result.loss` — and
the JSON was never written, because `card_case` threw at the descent checks
(`tests/gpu/device_step.cpp:640-642`) *before* it wrote the results file (`:746`) and before
the `non-finite:` / `walk by op:` attribution prints (`:781-846`). The one line that says
where a non-finite value was born was therefore unreachable exactly when it was needed, and
the run's 90 minutes produced a single line.

**That is fixed, not worked around** (`27fdcc2`): the three descent checks now run at the end
of the function, after the file is on disk and the attribution has printed, and a
`descent: loss A -> B (+delta)` line joins them; the parent prints a one-line summary of the
JSON's headline numbers before its own checks. The checks still fail the case — they no longer
take the evidence with them.

**What the failure does and does not say.** `after.loss` was *finite* and merely not lower
(`std::isfinite(after.loss)` passed, `after.loss < result.loss` did not), so the 4096
gradient was not NaN and the step did not ascend into a non-finite parameter. Three
explanations were open: a wrong-but-finite gradient, lr overshoot on 400 rank-16 leaves at
1e-3, and the case's own zero-`A` contract at this start.

**The first of those now has a mechanism.** The device join was reading buffers the engine had
already reused — the pull after `serve()` described in the 512 section above — so the head's
cotangent was computed from values the forward never produced. A finite but wrong gradient is
exactly the shape of a step that does not descend. If the at-issue capture cures the 512 case,
this case is its confirmation; if it descends afterwards, the lr and zero-`A` explanations are
both dead and should not be re-raised.

**The next run** is this case against its own contract — `a_all_zero`, `b_all_nonzero`,
`a_changed == 0`, `b_changed == n_b` — with the wall clock and the descent delta.

## The rows this run depended on

- **The quant family's roles come from the record** (`docs/kernels/quant.md`): the live
  `gate_had_quant_i8g` declares `n = 256 group = 128` while its codes are 6144 wide with a
  48-group scale, so every role is discovered by dtype and shape and the params are only a
  cross-check. Every decline in `quant_adjoint`, `mask_and_rotate` and `add_back` names
  itself — the operand and both counts, the scratch block's byte count, or the failed copy
  with `hipGetErrorString(hipGetLastError())` — because a silent return there is a zero
  gradient that agrees with nothing. `tests/gpu/adjoint_quant.cpp`'s `live_gate_shape`
  pins the live record and four of those refusals.
- **The read set** (`r97/adjoint.h`'s `read_rows`): a row no consumer reads is provably
  dead and its values are skipped, with two guards — a row must not be *proved* empty by an
  empty read set, and an empty read set at the consumer means *unknown*, not dead.
- **The head**: `logits_gemm_vocab`'s 248320-wide cotangent can never be a dense `AdValue`,
  so the head walks in chunks (`head_chunks 8`) through `head_ce_loop`, and the transposed
  product reuses the row's own trellis decoder.

## After this report: the block walk had the same defect, and it is now closed

The section above says the pull after `serve()` was reading reused buffers. That was one layer. The
block replay had the same class, and closing it is what this branch's tail is:

- **`df11fe6`** -- the head walk's operands are captured at their issue.
- **`d1ac737`** -- a block read whose buffer a prior op wrote is served from the writer's `before`
  capture at issue rather than the live, already-reused arena. A strided view and the `kv:` pool keep
  the live pull.
- **`2a66311`** -- `writer_view`: a strided read is gathered out of the writer's capture through the
  operand's `view_off`/`view_stride`/`view_shape`. **This is what made `forward_all_finite` true**;
  it had been non-finite at `attn_qg[0]` of position 1172.
- **`fea6371`** -- an uncaptured write drops the capture (`before.erase`) instead of leaving a stale
  one. Measured a no-op on this path (`512h` identical to `512g`): standing protection, not a fix.
- **`e96d281`** -- reverts an experiment that widened the capture gate to `rows == 0` writes. That
  regressed the case to four zero-gradient `lora_A`/`lora_B` leaves and moved the non-finite to
  `lora_shrink` at 1150. The gate stays
  `view_shape.empty() && rows > 0 && rows == shape[0]`.

Measured on the 512 case with those in place: **`forward_all_finite true`**, head `rmsnorm`
`512 readable / 0 zeroed / 0 refused`, `n_zero_a == n_zero_b == 0`, both `all_nonzero true`,
`a_changed == b_changed == 378`, descent holding `7371.59135 -> 6347.88397` (-1023.71), step
309.3 s (forward 46.7, head 0.8, backward 259.7). **`all_finite` is still false.**

**Re-measured on 2026-11-15** (`/tmp/r97-dstep-device_step_tape-i9Nlis`, the same case with
`R97_CENTRAL=1`): `forward_all_finite true`, head `rmsnorm` `512 readable / 0 zeroed / 0 refused`,
`n_zero_a == n_zero_b == 0`, both `all_nonzero true`, `a_changed == b_changed == 375`, descent
holding `7371.59135 -> 6347.88397`, step 316.2 s (forward 46.6, head 0.8, backward 266.6). **The
counts the run adds:** `n_a_finite_nonzero == n_b_finite_nonzero == 82` of 400, so 318 leaves of each
factor hold a **non-finite** gradient and none is exactly zero. The first non-finite the accounting
sees is now `lora_shrink` at position 1150, operand `h.s8`, index 6 -- moved down from
`rmsnorm_had_quant_i8g`@1176. **`all_finite` is still false.** The leaf gradient and the central
difference are in "The verdict" below.

## The open defect: measured, not inferred

The first non-finite the accounting sees is the cotangent of `rmsnorm_had_quant_i8g`'s in-place
output `h` at position 1176, non-finite in **every** element -- 2621440 of 2621440, all 20480
groups. What has been measured about it, each with the instrument named:

| question | measurement |
|---|---|
| did it arrive from upstream? | **No.** `R97_DUMP_POS=1176` prints every operand's incoming cotangent. At 1176 the two live ones -- the codes' `h.q8` (2621440 values) and the scale's `h.s8` (20480) -- both report no non-finite anywhere (`inc_nonfinite_at=-1`); `x` and `h` carry no version. |
| was the forward to blame? | **No.** `forward_all_finite true`. |
| is some group's scale tiny? | **No.** `SCALERANGE` over every quant row: the smallest group scale is **7.5e-3**, and none is below 1e-32. |
| is the zero-`amax` convention wrong? | **No.** `libr4d/r4d_fused_quant_fp8.hip:100` stores `s = 1.0f` for a zero-`amax` group, matching the reference twin (`src/adjoint/host/quant.cpp:163`) and `docs/adjoints/quant.md`. The `fmaxf(m, 1e-8f)` sites in `r4d_rmsnorm_quant_i8.hip` serve a **different op** that this container does not execute for this record. |

So the row was handed finite data, with ordinary scales, and produced a wholly non-finite output.

### Why "wholly", and why a per-group cause is not excluded

`rmsnorm_back_kernel` (`src/adjoint/device/quant.hip:274-278`) forms a **row-wide** share:

```cpp
const float rs = rsqrtf(sh_ss[0] * inv_n + eps);
const float shared = mult * rs * rs * rs * sh_t[0] * inv_n;    // whole row, 5120 columns
dx[base + i] = f32_to_bf16(mult * (rs * c * rotated_cot[base + i]) - shared * x[base + i]);
```

`sh_t[0]` sums `rotated_cot * x * (w + wadd)` over the entire row, and `rotated_cot` is the inverse
rotation of `d`, where `d[tid] = cot[base + tid] / (float)s`. So **one group** whose quotient
crosses f32's 3.4e38 ceiling puts an `Inf` into `sh_t[0]`, `shared` becomes `Inf` (or `NaN`, if
`sh_ss[0]` also went `Inf` so that `rs == 0` and `shared = 0 * Inf`), and `dx = finite - Inf * x[i]`
makes **every element of that row** `-Inf` (or `NaN`). 512 such rows is the whole buffer.

An earlier draft of this section claimed the opposite -- that a per-group mechanism "cannot produce
that buffer" because all 20480 groups would have to be degenerate. **That was wrong and is
retracted**: the row-wide reduction carries a single group's overflow across the whole row.

### The measurement: ~984x per block on average, and accelerating  [SUPERSEDED — see the summary at the top]

> **Confirmed in substance, corrected in number.** The curve is real: tracking one buffer across blocks
> (`gdn_in`, blocks 62 to 58) measures **~1.6e3 per block**, and that is why the walk saturates to
> non-finite around block 50. What the curve's own reading got wrong is the *value* -- `~984x` from a
> logit-space division against `~1.6e3` measured directly -- and the "accelerating" framing. See "The
> per-block factor, measured at last: ~1.6e3, and it is real" below.

`R97_DUMP_POS=all` prints the largest finite magnitude of every record's incoming cotangent in walk
order (63 -> 0). The curve is neither a spike nor a range limit -- it is **monotone** but **not
uniform**, so it is worth reading as a sequence of intervals rather than as one rate:

| position | approx block | largest finite max abs cot |
|---|---|---|
| 1459 (head) | -- | 1.9e-4 |
| 1449 | 63 | 2.7e-3 |
| 1400 | 59 | 2.0e4 |
| 1358 | 56 | 4.3e10 |
| 1309 | 54 | 3.4e17 |
| 1267 | 52 | 3.2e24 |
| **1176** | **51** | **1.214e37** -- and the next row produces non-finite |

Read as intervals, the acceleration is explicit -- and the "~2500 per layer" an earlier draft quoted
was an average of them, neither the net nor a constant:

| interval | blocks | ratio | x per block |
|---|---|---|---|
| head -> 63 | 1 | 14.2 | 14 |
| 63 -> 59 | 4 | 7.4e6 | 52 |
| 59 -> 56 | 3 | 2.2e6 | 129 |
| 56 -> 54 | 2 | 7.9e6 | 2812 |
| 54 -> 52 | 2 | 9.4e6 | 3068 |
| 52 -> 51 | 1 | **3.8e12** | **3.8e12** |

The amplification **accelerates**: ~52x per block early, ~3e3 in the middle, then 3.8e12 in the single
step before the non-finite. Averaged over the **13 steps** from the head to the block-51 leaf the net
is **~984x per block** (1377x to block 51's last finite value), against a **true ~3.9x per block**
(9.07e3 / 1.9e-4 = 4.8e7 over the same 13 steps). That is a **~250x per-block excess**, and
250^13 = 1.9e31 -- the over-statement, reached without assuming any uniform factor. (This paragraph
quoted "~2.70x per block / 365x / 2e33" until the `R97_CENTRAL` run; that pair came from the ~75.9
secant, which is not a derivative -- see "The verdict" below.)

Inside a block the *largest single record-level jump* is `gdn_gated_rmsnorm`, whose Jacobian grows
~1e4 when a row's `mean(x^2)` approaches `eps`; it both reads and writes `gdn_o` (opnd 0 and opnd 3),
which is what made that jump look like the scan's. **That is a same-buffer measurement across one
record, not the block's amplification**: it is offset by shrinks over the rest of the block, so it
does not multiply with the ~1e2 a quantised row adds -- the block net is the ~984x above.

From position **1175** onward, **817 records** report a non-finite *incoming* cotangent -- the
non-finite values propagated from 1176 and never became finite again.

`maxabs` counts finite values only (`src/backward.cpp:610`), so a fully non-finite vector reports the
largest surviving finite magnitude rather than `inf`. That is why the curve appears to plateaus after
1176: it does not, the vector simply stops being finite.

### What that means, and what it rules out

The growth is **not** an unbalanced quant `1/s`. The quant row divides the code's cotangent by the
group scale (`d[tid] = cot / s`, both twins -- in f64 on the host, `src/adjoint/host/quant.cpp:299`),
and the *consumer* of the code multiplies it back by `s`: `gemm_nt_q`'s adjoint works "in the
dequantised activation basis" and applies `dy * s` column by column
(`src/adjoint/device/linear.hip:121,129`). So the `1/s` round-trip is balanced, and the growth is
either the product of the **weight matrices' spectral norms** -- the backward's genuine
straight-through Jacobian -- or a scale bug that one block's norm would expose. The growth curve
cannot tell those apart; only a per-op finite difference can, and that is what closed it below.

One thing the curve does settle, and where the device is wrong on its own: `rmsnorm_back_kernel`
accumulates its row reduction `t` in **f32** (`src/adjoint/device/quant.hip:256-260`, 5120 terms)
while the host twin accumulates `drs` in **f64** (`src/adjoint/host/quant.cpp:628-632`). The quotient
`cot / s` is *safe* -- `1.25e34 / 0.19 ~ 6.6e34`, under f32's 3.4e38 -- so it is the **5120-term f32
sum**, not the division, that crosses: 5120 terms of order 1e34 reach ~1e38. The f32 accumulator
overflows there; the f64 one does not. **That is a measured device/reference divergence, and fixing
the device accumulator to f64 makes the device match the reference** -- it is not a range hack to
forbid. **The fix is in the tree and measured** (`src/adjoint/device/quant.hip:248-290`): with
`ss`/`t`/`rs`/`shared` in f64 and only the store `f32_to_bf16`, the 512-token run's first
non-finite **moved from `rmsnorm_had_quant_i8g`@1176 to `lora_expand`@1151** (`a_changed` 375 ->
373; loss and the descent unchanged). So the accumulator was a real defect and it is fixed -- and it
is **not sufficient**: `all_finite` is still false, because the cotangent is still ~1e37 at 1151 and
is amplified again before the bf16 store. Whether *any* widening of the device's arithmetic can make
`all_finite` true depends on whether that growth continues below block 51, which the walk can no
longer show -- everything below is non-finite. That is what the loss's central difference settles,
and it is the only open question now.

### The verdict: the true gradient is O(1e4), and the walk reports 6.28e35

> **Superseded in part.** The `-9073.8` reading stands -- it is re-measured below and unchanged -- but
> "the walk reports 6.28e35" is true of only *one* leaf: at block 40 the walk is **NaN**, because the
> cotangent has already crossed the f32 ceiling by block 50. See the per-block section below.

`R97_CENTRAL=1` was run on the 512-token case with a control proving the forward reads the params it
is handed. The run of 2026-11-15 (`/tmp/r97-dstep-device_step_tape-i9Nlis`) is the first with
`eps_bf16` and `central_bf16` in the record:

```
name            model.layers.51.self_attn.q_proj.lora_A
w               0.013567836955189705
eps             1.3567836955189706e-06   -> central 0.0          (bit-identical losses)
eps_bf16        1.3567836955189706e-04   -> central_bf16 -9073.830365103388
loss            7371.591348378961
loss_plus_bf16  7370.903491856901        loss_minus_bf16 7373.365736875957
grad            6.2826791988271414e+35   (the walk's leaf gradient)
params_live     true, control_loss 7371.694270937236
```

**The small-eps `central == 0` is a bf16 resolution artifact**, not a zero gradient: a 1.36e-6 move
on 0.0135678 is 1e-4 relative, below the forward's own ~0.4% bf16 mantissa, so the two losses are
bit-identical by construction. At `eps_bf16` the loss does move, in both directions:

| estimate | value |
|---|---|
| forward secant, `+1%` | **-5070.1** |
| backward secant, `-1%` | **-13078.0** |
| **central difference, `±1%`** | **-9073.8** |
| the walk's leaf gradient, same weight | **+6.2827e35** |
| **over-statement** | **~6.9e31** |

**The `~75.9` an earlier draft of this section quoted is wrong, and this run is what shows it.** That
figure was a **10% one-sided secant** (`0.102922558275 / 1.3567836955189706e-03`); over 10% the loss
is nowhere near linear, so it is not a derivative. At 1% the two secants already differ by a factor
of **2.6** (-5070 against -13078), which is why the central difference is the one to trust -- and why
even ±1% is not a linear interval, so -9074 is an estimate of the derivative at `w` with the second
derivative's contribution in it, not a converged value. Every estimator tried lands in **O(1e4)**;
none lands near 1e2.

**So the backward's leaf gradient is wrong by ~7e31** -- not a range limit, not an accumulator width,
not a bf16 store, and not something any widening fixes. With the true `dL/dw` at ~9e3 and the head
cotangent at 1.9e-4 (the CE gradient of a 511-token mean, which is right), the true amplification over
the 13 steps from the head to the block-51 leaf is **~3.9x per block**; the walk measures **~984x**, a
**~250x per-block excess**, and 250^13 = 1.9e31. That compounds to the 6.9e31 the two leaf values
give. **`descent` held only because AdamW is scale-free** (`lr*mhat/(sqrt(vhat)+eps)` is
`lr*sign(g)` in the first step).

**The tip's structural reading, which the run also now records** (`n_a_finite_nonzero`, added the
same day): of the 400 `A` leaves, **82 have a finite non-zero gradient, 318 have a non-finite one, and
0 have an exactly-zero one** -- the same 82/82 split on `B`, `a_changed == b_changed == 375`. The
first non-finite value is `lora_shrink` at position 1150, operand `h.s8`, index 6, i.e. the first
non-finite has moved down from `rmsnorm_had_quant_i8g`@1176 since the f64 row reduction landed.
`forward_all_finite` is true and the head `rmsnorm` reads `512 readable / 0 zeroed / 0 refused`.

### The growth is one global per-layer factor, not one op

> **Confirmed in substance, restated in number.** This section names a per-layer factor from the growth
> curve. The curve is real and now measured directly (**~1.6e3 per block**, one buffer across four
> blocks), and the truth grows 1.24x per block, so the error per block is ~1.3e3. Only the figure
> drawn from the logit-space division is superseded.

An earlier reading of the curve named `gdn_chunk_scan` (x1e4 per layer) the largest single
contributor. **That is wrong, and the measurement that refutes it is the same buffer tracked across
blocks.** Following `gdn_o`'s cotangent at the `gdn_chunk_scan` record from block to block:

| block boundary | `gdn_o` cotangent maxabs | ratio |
|---|---|---|
| pos 1415 | 4.160e3 | -- |
| pos 1394 | 7.602e6 | 1.8e3 |
| pos 1373 | 7.751e9 | 1.0e3 |
| pos 1324 | 5.432e16 | 3.2e3 |
| pos 1303 | 1.718e20 | 5.1e3 |

Every buffer grows at the same ~1e3 rate between consecutive blocks (`h` 2.3e3, `lane0` 2.7e3, `q8`
1.7e3, `s8` 1.3e3, `gdn_o` 1.8e3) -- the factor is **global to the layer**, not an op's. The x1e4 at
`gdn_chunk_scan` is real but it is a *within-layer* sub-op jump (the same `gdn_o` buffer grows 1.5e4
across that record), and it is offset by shrinks across the rest of the layer, so it is neither the
per-layer factor nor the primary suspect. A per-op verdict needs the op's Jacobian applied to the
real cotangent (`R97_DUMP_DC`), not the ratio of two adjacent records' peaks.

**A ~1e3 per block is the right size for ~5 quantised ops per block each contributing `1/s ~ 5`**
(`s ~ 0.2`, so 5^5 ~ 3125), so the working hypothesis at the time was that one or more
quant-consuming adjoints convert the code's cotangent with the wrong derivative factor, letting `1/s`
compound on every pass through the block's quantisers. **Superseded**: the two contributors that
carry the most of the curve -- `gdn_gated_rmsnorm` and `gdn_chunk_scan` -- are both exact (see "Where
the factor is not"), and `gemm_nt_q` (`src/adjoint/device/linear.hip:121,129`) and `lora_shrink`
(`src/adjoint/host/lora.cpp:255-258`) both handle `1/s` correctly. What survives is that the three
i8g fused producers cannot be reached by any finite difference at all.

### One real scale bug, found and fixed -- and measured NOT to be the 1.9e31

**`gemm_nt_q`'s adjoint returned the cotangent of the dequantised activation while the walk's code
operands carry the codes' cotangent.** Every quantiser's adjoint divides by the group scale `s`
(`src/adjoint/host/quant.cpp:299`, `src/adjoint/device/quant.hip:199`), so the code cotangent has to
come back multiplied by it. A standalone probe (`tests/host/scale_factor_probe.cpp`) measured the
omission as **`1/a_scale`, exactly**: `gemm_nt_q` at M=3, N=64, K=384, group=128, int8 codes,
`a_scale` in {0.125, 0.25, 0.375, 0.5}, central-differencing the op's own forward against its adjoint
for 15 single elements of `a`.

| row | adj/fd | adj/fd x a_scale |
|---|---|---|
| tagged (`gemm_nt_q_code_adjoint`) | **1.0000** (all 15) | 0.2667 |
| untagged (the family row, i.e. the device row's rule) | 8.0 / 4.0 / 2.667 / 2.0 | **1.000000** (all 15) |

The host already had the right rule, but only behind a tag: `gemm_nt_q_code_adjoint`
(`src/adjoint/host/linear.cpp:373-389`, registered at `:398` under `host_step_record`) multiplied
`dx` by `scale_at(as, m, k/group, M, groups)`. The device row and the host gated row never got it.
The contradiction was visible in the file: `gemm_nt_q_twin:208` multiplies `a` by `a_scale`,
`gemm_nt_q_adjoint:242` did not. All three rows now apply it, and the tag plumbing is deleted.

**The fix is live, and it does not explain the 1.9e31.** Re-measured on the card after the change
(`/tmp/r97-dstep-postfix.log`):

| | before | after |
|---|---|---|
| the leaf gradient | -1.5512e35 | **+6.41249e35** |
| `all_finite` | false | **false** |
| `gdn_o` at `gdn_chunk_scan`, block 62 -> 61 | 4.160e3 -> 7.602e6 | 1.128e3 -> 1.655e6 |

The gradient moved by ~4x and the growth curve's rate is unchanged (~984x per block). **A missing
`a_scale` worth ~5x per hop cannot account for a ~1.9e31 over-statement**; this is a genuine defect on
its own and it stays fixed, but the amplification that makes `all_finite` false is still
unexplained. The strongest remaining lead was the *within-layer* jump the post-fix curve still
shows: `gdn_o`'s cotangent grows ~1e4 as the walk passes the `gdn_chunk_scan` record, at every block
(`pos 1416 -> 1415` 1.07e4, `1395 -> 1394` 1.08e4, `1234 -> 1233` 1.13e4, ...). **That lead is now
closed, and the site it pointed at is cleared** -- see "Where the factor is not" below. The x1e4
straddles `gdn_gated_rmsnorm`, which *reads* `gdn_o` at opnd 0 and *writes* it at opnd 3, and an
op-level central difference reproduces it as that op's own conditioning when `mean(x^2)` sits near
`eps = 1e-6`. The scan's own `dy -> max|dc|` ratio is 0.03..0.12: it shrinks.

**The producers are not all checkable, and that is the gap.** All four quantisers that *have* a host
twin divide by `s` correctly on both sides, and `lora_shrink` multiplies by `a_scale` unconditionally
on both (`host/lora.cpp:312`, `device/lora.hip:296`). But `rmsnorm_had_quant_i8g`,
`gated_had_quant_i8g` and `gate_had_quant_i8g` have **no host twin and no host adjoint**, so no finite
difference reaches them at all; they are 208 of the walk's 1459 records and they produce every code
plane the `gemm_nt_q` consumers read.

Separately, the `all`-mode walk **aborted at position 565** (~block 24) with the card free:

```
r97: iqr dx declined: op gemm_nt_q: the decode's scratch could not be allocated
     (N 5120 K 17408 G 136 form trellis.mad b5 L16 lrb0 lrun8 scale=row-f32)
```

It did not occur in the earlier runs (`512n`/`512o`/`512p` all completed the backward in 257-259 s),
so it is its own defect, not part of the magnitude question.

### The accounting, for the record

`cot_non_finite` covering *written* operands only (`src/backward.cpp:512`) is **correct**, not a
blind spot: `cot[]` is what a row consumes and `dc[]` what it produces, and a read-only operand has
no incoming cotangent to scan. The one real over-claim is the comment at `:724-725`, which is false
on the `options.in_place_passthrough` path (`:709-718`): a slot the row left empty is filled with the
cotangent the row was *handed*, after the pre-call scan.

### Where the factor is not, and what is left

An op-level probe (`tests/host/gdn_gradient_probe.cpp`, CPU only) central-differences each op's own
host twin against its own host adjoint -- the method that found the `gemm_nt_q` scale, and one that a
device-vs-host *relative* comparison cannot replace. Every op it reached is the exact derivative of
its own forward (`adj/fd = 1.000000` at every probed element, worst relative error 1e-9..1e-7), and
it was shown to see a factor of 2 when one is injected, so it would see 1.9e31.

| op | fixtures | result |
|---|---|---|
| `gdn_chunk_scan` | 6, incl. ragged, `ht` live, and `scale` **absent** | `adj/fd = 1.000000` x12 each; worst 1.8e-7 |
| `gdn_gated_rmsnorm` | live conditioning + ordinary rows | every x ratio **exactly 1.000000** |

**The two largest jumps in the growth curve are therefore not defects.** The x1e4 at
`pos 1416 -> 1415` straddles `gdn_gated_rmsnorm` (it *reads* `gdn_o` at opnd 0 and *writes* it at
opnd 3), so it is the norm's Jacobian, not the scan's, and the probe reproduces it as the op's own
conditioning: **amp 1.543e4** against the dump's 1.07e4, with `mean(x^2)` walking down to
`eps = 1e-6`. A 1e4 cotangent step at a gated RMS norm whose input sits near `eps` is what that op
does. The scan's own `dy -> max|dc|` ratio is 0.03..0.12 -- it *shrinks*. `scale` is absent from the
tape (it holds ints and strings, no double), and both sides fall back to `1/sqrt(head_k)`, which is
the value the engine passes anyway (`third_party/radiance/arch/common/rad_block_gdn.h:209`); the
absent run reproduces the explicit one value for value.

**The gap is named.** `rmsnorm_had_quant_i8g`, `gated_had_quant_i8g` and `gate_had_quant_i8g` have
**no host twin and no host adjoint** -- the only registration is the device row
(`src/adjoint/device/quant.hip:874-876`), and the existing test checks that row against a host
*composition* of the same parts (`tests/gpu/adjoint_quant.cpp:535-543`), which is exactly the
shared-factor comparison that cannot see a factor both sides carry. **208 of the walk's 1459 records
(14%) are these three ops**, and they produce every code plane the `gemm_nt_q` consumers read. They
are the only op group on the path with no finite-difference check on either side.

Auditing that row against its forward accepts the `/s` (it matches `src/adjoint/host/quant.cpp:299`),
the `+ cot_h` fused-output term and the norm backward, so no factor is visible by inspection. An
earlier draft of this section then flagged a range hazard here -- `cot / s` in f32
(`quant.hip:199`) against the 5.96e-05 smallest live scale -- **and that arithmetic is wrong for this
row**: the quotient is safe where it is taken, because at position 1176 the code cotangent is 1.25e34
against a group scale of 0.19, i.e. `6.6e34`, an order of magnitude under f32's 3.4e38. The
5.96e-05 and 7.5e-3 scales belong to rows whose cotangents are far smaller. **What WAS a real
device-versus-reference divergence on this path is the 5120-term f32 row reduction** in
`rmsnorm_back_kernel`, fixed in the tree (`src/adjoint/device/quant.hip:248-290`, all-f64
accumulators, only the store narrowing). Issue #140 is re-scoped to match.

**That gap is now closed.** The three producers have host twins and host adjoints
(`src/adjoint/host/quant.cpp`), built as the composition the GPU tier already had in its test TU, and
each has a linearised `<op>_ste` row so that a finite difference can differentiate it -- these are
quantisers, so their forward is a step unless the row is linearised at a base point. Held to their
own differences at two shapes they read 1e-9 to 1e-11, against 1e-12 to 1e-9 for the fp8 control in
the same run. So the three producers' **host** side is now excluded.

**What is still unprobed is the device side of every row on this path, not any one row.** A device row
is checked against its host twin, and the host is now exact by finite difference -- but that
transitivity holds only if the device-versus-host check exercises the magnitudes that matter. Those
checks run on synthetic fixtures with small values, while the real container's cotangents reach
1e37 by block 51; a clamp, an `rsqrtf` near zero, an f32 accumulate or a bf16 store that engages only
at 1e34..1e37 would never fire on the fixture. **An earlier draft named the device row's `cot / s`
here and that is wrong** -- the quotient is safe where it is taken (`1.25e34 / 0.19 = 6.6e34`, under
f32's 3.4e38); see the re-scoped #140.

### The `1/s` round-trip, audited and refuted

The obvious candidate was the `1/s` round-trip: a quantiser's adjoint divides the codes' cotangent by
the group scale, its consumer must multiply it back, and at `s ~ 0.19` that is `5` per hop -- with
~5 hops per block, a few unbalanced hops would be the measured order. **It is balanced, on both
families, at every hop.** Audited against the retained `R97_DUMP_POS=all` dump of a contiguous
895-record walk (`/tmp/r97-dstep-512q.log`), so the hop inventory is measured rather than assumed:

| | |
|---|---|
| ops that write a code plane | `rmsnorm_had_quant_i8g` 78, `gated_had_quant_i8g` 39, `had_quant_act_i8g` 29, `gate_had_quant_i8g` 10 |
| ops that read one | `gemm_nt_q` **177**, `lora_shrink` **177** -- and each reads the matching scale 177 times |
| not a hop | `lora_expand` (its input `t` is f32); `gemm_nt` (bf16 family, no `a_scale`) |

Four producers, one divide site (`device/quant.hip:199`, all four share `mask_rotate_kernel`), two
multiply sites (`device/linear.hip:657`, `device/lora.hip:296`). **All eight producer x consumer pairs
have product exactly 1.0**, and the multiplier is *the operand the producer wrote* -- the same values,
not merely the same rule. The grids agree at all four live widths (5120/40, 5120/40, 4096/32,
17408/136 -- all 128), which is where a "balanced rule, wrong grid" bug would live. The host twins are
the same rule in f64 (`host/quant.cpp:365` divides, `host/linear.cpp:244` and `host/lora.cpp:312`
multiply).

### The per-layer factor, measured: ~1.6e3, and it is real

**Two measurements, and they say different things -- which is the whole point.**

**The walk's cotangent in one buffer through one op** (`gdn_o` at the `gdn_gated_rmsnorm` record,
`R97_DUMP_DC=1`), with every ratio recomputed from the dump rather than transcribed:

| position | maxabs | x since previous | decades | record gap |
|---|---|---|---|---|
| 1416 | 1128 | -- | -- | -- |
| 1395 | 1.66298e+06 | 1474 | 3.169 | 21 |
| 1374 | 2.03004e+09 | 1221 | 3.087 | 21 |
| 1325 | 9.57015e+15 | **4.714e+06** | 6.673 | 49 |
| 1304 | 7.96236e+18 | 832.0 | 2.920 | 21 |
| 1283 | 1.30603e+22 | 1640 | 3.215 | 21 |
| 1234 | 3.46623e+28 | **2.654e+06** | 6.424 | 49 |
| 1213 | 4.97553e+31 | 1435 | 3.157 | 21 |
| 1192 | 1.32404e+35 | 2661 | 3.425 | 21 |
| **1143** | **3.38953e+38** | 2560 | 3.408 | 49 |
| 1122 | 0 (all non-finite) | -- | -- | 21 |

**Per record the rate is 1.2x to 1.5x, and the per-layer rate is the robust quantity.** 0.1506
decades/record over the six 21-record gaps, 0.1336 over the first two 49-record gaps (the third is
clipped, see below): **~0.140 decades per record, i.e. ~1.38x.** It is not perfectly flat -- on the `h`
trace below the per-record rate runs **1.23x on the 20-record segments, 1.37x on the 13-record ones and
1.51x on the 8-record ones**, i.e. shorter segments carry more per record -- so the per-record figure is
an average over a within-period modulation, and **the per-layer figure is the one to quote.**

**One decoder layer is 22.78 records, so the per-layer rate is ~1.6e3.** The container's own artifact
reports `layers 64` (`device_step_tape.json`, the same 800-leaf `ParamSet` this report is about), and
the walk covers 1458 positions, so **1458/64 = 22.78 records per layer**. The dump corroborates the
count independently: `gdn_gated_rmsnorm` appears 48 times and `rope` 32 times, i.e. 48 linear-attention
plus 16 full-attention layers -- **48 + 16 = 64**, a 3:1 hybrid, and the 91-record `rope` period divided
by 22.78 records per layer is **3.99**.

So:

| | |
|---|---|
| per record | 0.140 decades = **1.38x** |
| per decoder layer (0.140 x 22.78) | 3.190 decades = **1.55e3** |

and a 21-record gap is one layer (3.162 decades, mean of six) while a 49-record gap is two (3.274,
mean of two) -- consistent to 4%, which is the second independent check on the layer size.

**Against the truth, that is an excess of ~1.25e3 per layer.** The true gradient grows **1.24x** per
layer (876.06 at block 40 to 9073.83 at block 51, 10.4x over 11 layers), so the walk's excess is
1.55e3/1.2373 = **1.25e3 per layer, 3.098 decades**. Compounded over the 11 layers from block 40 to
block 51 that is **1.19e34**, against a measured over-statement of **6.9e31** at the block-51 leaf --
the same order. **Three independent facts agree on the same per-layer rate**, which is what the earlier
`~984x` never had.

**And a fourth check, from a different instrument on a different run.** The retained
`R97_DUMP_POS=all` walk of the 512 case (`/tmp/r97-dstep-512q.log`) prints each record's *incoming*
cotangent magnitude in walk order. Over its 392 records -- positions 1459 down to 1157, **13.26
layers** -- the magnitude runs `1.86e-04` to `1.96e35`, i.e. **39.023 decades**:

| | this dump | the DC trace |
|---|---|---|
| per record | 0.0998 decades = **1.26x** | 0.140 decades = 1.38x |
| per decoder layer | 2.944 decades = **878x** | 3.190 decades = 1.55e3 |

Same order, within a factor of 1.8 per layer and 1.4 per record, from a completely independent
instrument. **So the defensible statement is: the walk multiplies by ~1.3x per record and ~1e3 per
decoder layer**, and the two instruments bracket it.

**The hops are balanced, and that rules the scale bookkeeping out.** Pairing every record that touches a
buffer with the next one that does (within two positions), on the same dump:

| buffer | adjacent pairs | equal to all printed digits | differ |
|---|---|---|---|
| `h` | 28 | **28** | **0** |
| `gdn_in` | 10 | 8 | 2 |
| `gdn_o` | 5 | 0 | 5 |

`h`'s producer -> consumer pairs are **exactly** equal: `rmsnorm_had_quant_i8g`@1449 reads
`7.24196e-06` and `lora_expand`@1448 reads `7.24196e-06`; @1429/@1428 both read `9.23157e-04`. The
quantiser's `1/s` and its consumer's `s` cancel exactly, as the audit above predicted. **The two
`gdn_o` columns differ by a steady ~1.07e4** (`had_quant_act_i8g`@1417 `0.105` against
`gdn_gated_rmsnorm`@1416 `1128`), but **that ratio is constant across layers** -- 1457x, 1446x, 1267x
between consecutive `had_quant_act_i8g` values, i.e. both operands grow at the same per-layer rate. A
constant offset between two operands is a scale difference, not an amplification.

**So the amplification is applied by the ops, not by a hop.** `h`'s own cotangent, taken at one op
(`rmsnorm_had_quant_i8g`) across 27 occurrences spanning positions 1449 to 1148 -- 301 records,
**13.2 layers** -- runs `7.24196e-06` to `2.72076e36`: **42.575 decades**, i.e. **0.1414 decades per
record (1.384x)** and **3.22 decades per layer (1.67e3)**. Its hops are exactly balanced and it still
grows at that rate, so **the factor is inside the per-op adjoints on the `h` path, not in the scale
bookkeeping.** That is a third independent measurement and it agrees with the `gdn_o` trace to within
1% per record and 1% per layer.

**And it is uneven within a period.** The 27 occurrences fall into segments of 8, 13 and 20 records
(period 91, which is the four-layer `rope` period again, with eight occurrences per period):

| segment length | n | decades/record | x/record |
|---|---|---|---|
| 8 records | 13 | 0.1787 | **1.51x** |
| 13 records | 9 | 0.1353 | **1.37x** |
| 20 records | 4 | 0.0895 | **1.23x** |

**Shorter segments carry more per record**, so the per-record figure is an average over a modulation and
the per-layer figure is the one to quote. What the modulation does rule out is a single record doing all
the work: no one step in any segment accounts for the decade, so the factor is smeared across the ops
rather than concentrated.

**And the segments differ in a way that names the suspect.** Every segment shares the same opening
chain, and the short ones are almost nothing but it:

| segment | op chain |
|---|---|
| 8 records | `rmsnorm_had_quant_i8g -> lora_expand -> lora_shrink` (+ `gemm_nt_q -> gated_had_quant_i8g`) |
| 13 records | the same, then `gdn_gated_rmsnorm -> gdn_chunk_scan -> gdn_kkt_solve -> gdn_conv_prep` |
| 20 records | the same, then `rope -> rmsnorm` (the full-attention layer) |

It is tempting to read the higher per-record rate of the short segments as the shared core carrying the
factor. **It does not survive the arithmetic.** Per 91-record period there are four 8-record segments,
three 13-record and one 20-record:

| segment | decades per period | share of decades | share of records |
|---|---|---|---|
| 8 records | 5.718 | **44.7 %** | 35 % |
| 13 records | 5.277 | **41.3 %** | 43 % |
| 20 records | 1.790 | **14.0 %** | 22 % |

**12.785 decades per period**, and the core's 44.7 % over 35 % of the records is close to proportional.
The per-segment products are also not what a first reading suggests: `1.51^8` is **27.0** and `1.23^20`
is **62.8**, so a single long segment still outgrows a single short one.

**Normalising the jump counts by occurrences kills the op-level suspect too.** In the `R97_DUMP_POS=all`
walk, jumps above 1.2x per occurrence: `kv_store` 4/10 (40 %), `gated_had_quant_i8g` 26/78 (33 %),
`lora_shrink` 57/177 (32 %), `gdn_chunk_scan` 9/29 (31 %), `gate_had_quant_i8g` 4/20 (20 %),
`had_quant_act_i8g` 10/58 (17 %), `gdn_conv_prep` 24/145 (17 %), `rmsnorm_had_quant_i8g` 26/186 (14 %),
`lora_expand` 13/177 (7 %). **The top four are 40/33/32/31 per cent -- clustered, with no standout.**

**So both attempts to name a single op fail, and that is itself the finding: the ~1.38x is smeared across
the ops rather than concentrated in one.** Naming it needs a per-op finite difference -- each op's own
adjoint against its own central difference at the real magnitudes -- not a jump census.

**And the per-op adjoints are already checked at the container's own widths.** `tests/gpu/` has
finite-difference checks for every op on the `h` path -- `adjoint_lora`, `adjoint_quant`,
`adjoint_linear`, `adjoint_elementwise`, `adjoint_gdn_prep`, `adjoint_gdn_scan`, `adjoint_gdn_kkt` --
and they run at the real shapes, not only at toy ones: `adjoint_lora.cpp:103-112` covers `M = 1024` and
`4096` at `K = 5120`, `N = 12288` and `K = 17408`; `adjoint_elementwise.cpp:303-320` covers hidden 5120
and 17408 at 1024 and 4096 tokens; `adjoint_linear.cpp:728-730` covers `4096 x 5120 x 6144` and
`4096 x 1024 x 5120`; and `adjoint_gdn_scan.cpp:103` covers 4096 tokens. **They all pass.**

**That is a real constraint, and it rules the obvious explanation out.** A linear op's adjoint is linear
in its cotangent, so a per-op factor cannot be magnitude-dependent; and the widths are already right. So
**no single op on this path is wrong**, which is consistent with the jump census above finding four ops
clustered at 31-40 % rather than one standout. The factor is in the **composition** -- how the ops'
contributions accumulate into one buffer -- not in any one op's own derivative. That is where the next
probe belongs.

**The specific composition to suspect is the saved forward, not the arithmetic.** If every op's adjoint
is FD-correct at the real shapes and every accumulation is correct, then the walk is the **exact**
gradient of *some* function -- and the only way for that to miss the real gradient by 1e31 is for the
function it linearises to be a different one. That is what a **stale or mismatched snapshot** produces,
and it is the one failure mode that no per-op FD can catch, because each op is being differentiated
correctly at the values it was handed.

It also fits the shape of the evidence: a per-record factor of ~1.38 applied uniformly, with the jump
census finding no op, is what a small systematic difference in the saved operands would look like
compounded 90 times per layer. The report already carries the two instruments for this -- `R97_DUMP_FWD`
(the forward's own operands as they were computed) and `R97_DUMP_SNAP` (what the walk was handed).
**Both flags must run in one process**, because `DUMPSNAP` printed the pointer, the view and a `nonzero`
count but **not** `maxabs` or a position, while `DUMPFWD` printed `maxabs`, `nonfinite_at` and `pos`;
comparing across two retained logs is not possible, and the retained ones confirm it
(`/tmp/r97-dump-window.log` has 46 `DUMPSNAP` lines and no `DUMPFWD`; `/tmp/r97-fwd-hunt.log` has 168
`DUMPFWD` and no `DUMPSNAP`).

**That gap is now closed** (commit `a3fb369`'s successor). `snapshot()` takes the record's position and
prints it, and computes `maxabs` over the same span `DUMPFWD` reports it over -- skipping non-finite
values so the two are comparable rather than both being `inf`. Both dumps gained a `=2` form
(`R97_DUMP_FWD=2`, `R97_DUMP_SNAP=2`) that prints a line for **every** operand of every record instead
of the first few, so the join is not limited to `gemm_nt_q` and the lora rows. The join itself is
`/tmp/r97-snapjoin.py`, which keys on `(op, pos, buffer)` and reports two things per key: whether the
tape's recorded `data=` pointer equals the pointer the forward actually wrote, and whether the two
`maxabs` agree. **A stale or mismatched snapshot shows as a different pointer or a disagreeing
`maxabs`; a correct one shows as neither.** Run both at `=2` on one layer and read the join.

The analyzer's negative controls were run against the retained pair and behave as the split predicts:
`/tmp/r97-dump-window.log` gives `0` forward lines and `44` snapshot lines with `28` keys present in
neither dump, and `/tmp/r97-fwd-hunt.log` gives `36` forward and `0` snapshot -- `NO JOIN` both times,
which is exactly why the pair proves nothing. It also confirms the old build printed no `pos` (every
`snap-only` key has `pos=None`).

**The probe was run. Both readings I published from it are wrong and are retracted here; what it
establishes is narrower and negative.**

`/tmp/r97-snap-probe.log` and `/tmp/r97-snap-probe2.log`, two 490 s windows, the 512-token case with
`R97_DUMP_FWD=2 R97_DUMP_SNAP=2` in one process: 4558 forward probe lines, 6775 snapshot lines, 2274
`(op, pos, buffer)` keys present in both.

**Retracted reading 1 -- "the snapshot agrees with the forward, 2530 to 0, so it is faithful".** The
join is circular: `dump_fwd_scan` skips every operand the op did not write (`src/forward.cpp:142`) and
reads the buffer at the issue, so for a *written* key both instruments read the same post-op
contents. **Read operands -- the values the walk consumes -- have no `DUMPFWD` line at all.** Two
things in that reading do stand: the pointer is not comparable (device `0x7448...` against host
`0x64d4...`), and the forward runs two or three times per step.

**Retracted reading 2 -- "an in-place read is recorded post-op, so the walk is handed the write".**
This one had a hand-verified case and a code path and was still wrong:

| | op | pos | `gdn_o` (`w.o`) | pointer |
|---|---|---|---|---|
| writer | `gdn_chunk_scan` | 1415 | 0.423828 | `0x64d46a82fb30` |
| reader | `gdn_gated_rmsnorm` | 1416 | 278 | `0x64d46a82fb30` |

278 is the reader's own output, the op is in-place (`rad_block_gdn.h:216`, `{w.o, w.z} -> {w.o}`), and
across all 192 `gdn_o` snapshot lines `0.423828` appears exactly once -- as the writer's own output.
Generalising that to every operand: of **512** in-place operands, **511** have a recorded read slot
equal to the op's own output and **0** equal the value it read (`lora_expand` 288,
`rmsnorm_had_quant_i8g` 127, `gdn_conv_prep` 48, `rope` 32, `kv_store` 16).

**All of that is true, and none of it is a defect, because the walk does not use the tape's copies
for this.** `Tape::record` does record post-op for an in-place operand -- `include/r97/tape.h:65-69`
says so outright ("The observer runs after the op does, so a written operand's copy is what the op
left"). But the device walk keeps its **own** capture: `std::map<std::string, AdValue> before` at
`src/step_device.cpp:1075`, filled at each write's issue (`:1274-1276`) and served back to a read at
`:1251`, `snapped.now[j] = writer_rows(prev->second, snapped.now[j])`. **`snapshot()` -- where
`DUMPSNAP` prints -- runs before that overwrite**, so the probe was measuring the value the walk
discards. The comment at `:1067-1071` names the very case: "An in-place read is the same defect with
the writer being this op's own earlier state ... the adjoint then sees the value the op read, not what
the arena holds a moment later (**the gated norm's gdn_o, above all**)".

**And the compensation covered this run completely.** The walk's capture is conditional -- only a write
with `view_shape.empty() && rows > 0 && rows == shape[0]` can be held (`src/step_device.cpp:1274`),
and a read the writer's shape does not cover falls back to the live pull with a printed line
(`:1243-1250`) -- so its gap is measurable rather than argued. In `/tmp/r97-snap-probe2.log` that gap
is **0**: no `live pull kept` line and no `refuses by name`, for any operand of the step. The two
instruments do disagree, and the disagreement is the tape's documented behaviour, which the walk does
not depend on.

**So the composition hypothesis is still unexplained and the search resumes**, now with two candidate
explanations eliminated rather than added. Two counts from intermediate models are withdrawn as
unusable rather than merely wrong, and are named here so they cannot be reintroduced: **"2335
differences"** (every snapshot against the highest position-below writer) and **"471 of 2579
read-only, 441 of 509 in-place"** (the same model). A buffer written in place records its post-op value
as that writer's copy, so the "last writer" model is wrong for exactly the buffers in question; and
buffer names repeat across the 64 layers, so a lower position of the same name may be another layer's.
The own-output test needs neither assumption and gave 511 of 512 -- and that property turned out to be
one the walk does not depend on.

**Four things the retraction turned up, all recorded rather than assumed.**

**#138 is fixed on this branch, not open.** `src/adjoint/device/lora.hip:117-154` now resolves each
pool row's destination through `factor_grad_name`/`grad_dest`, honouring `AdjointCall::fused_slots`
and `slot_prefix` exactly as the host row does (`src/adjoint/host/lora.cpp:212-215,271-273,299`), so a
record with several fused slots writes each slot's `dA` to the leaf the caller named instead of
summing them into one. `tests/gpu/adjoint_lora.cpp:309-342` builds the device and host calls under
both namings and compares them, which is the two-slot device-versus-host case the issue asked for.
It arrived in `64a437d`; the issue predates that commit and stayed open.

**#144: five in-place ops are never FD-checked with their output aliased over their input.** An
in-place operand is one schema slot holding two values, and only one can go in `fwd[i]`. This report's
walk puts the **pre-op** value there (`src/step_device.cpp:1251` serves `before`); `check_adjoint` puts
the twin's **post-op** output there (`tests/support/adjoint_check.h:389-408`). The convention is
deliberate and asserted -- `tests/host/backward_core.cpp:342-367`, "the adjoint receives what it read
from, not what it wrote" -- **but that test's op is `bwd_scale`, whose adjoint is value-independent,
so it cannot distinguish the two points.** Of the ops the probe found aliased, exactly one is ever
aliased in a test (`tests/host/adjoint_quant.cpp:682`, `{1}` for `rmsnorm_quant_fp8`);
`i8g_fd_check` (`:796-803`) passes no `in_place` at all. `R97_FD_PREOP=1` moves the harness to the
production point and is live for the one case that exists.

**And `gdn_gated_rmsnorm` is not one of them.** It is absent from the 511: its record keeps four
separate schema slots (`nRmsnormOperands = 4`, `src/adjoint/device/gdn_prep.hip:69`) and
`tests/host/adjoint_gdn_prep.cpp:170-175` models exactly that, so its FD test is faithful. Its
aliasing is between **buffers**, not slots, which is why the earlier reading of it as a slot defect
was wrong.

**#143: the `all`-mode walk aborts at position 565** with the card free -- `iqr dx declined: op
gemm_nt_q: the decode's scratch could not be allocated (N 5120 K 17408 G 136 ...)`. A VRAM-headroom
refusal on a path the 4096 case does not take, absent from the earlier `all` runs.

**The oracle is now a ladder.** `R97_CENTRAL_BLKS` measures several blocks in one run, and the
evaluations are cheap because `step_device.cpp:413` returns early on a null `grads`, so the oracle's
five calls per block are forward-only.

### The oracle as a ladder: the walk is already wrong at the LAST layer

`R97_CENTRAL_BLKS` measures several blocks in one run (`/tmp/r97-bisect.log`, window 10:09-10:47, 2256 s,
42 oracle evaluations). The control is live at every block (`params are live`), and the truth is
well-behaved at every block -- so this is not an FD artefact:

| block | leaf | walk's `grad` | FD `central_bf16` | \|walk/FD\| | decades |
|---|---|---|---|---|---|
| **63** (last) | `self_attn.q_proj.lora_A` | **0.00744629** | **948.917** | 7.85e-6 | **-5.11** |
| 60 | `linear_attn.in_proj_qkv.lora_A` | **9.43718e+07** | **701.006** | 1.35e+05 | **+5.13** |
| 55 | `self_attn.q_proj.lora_A` | -1.04482e+23 | 1351.07 | 7.73e+19 | +19.89 |
| 51 | `self_attn.q_proj.lora_A` | 6.59422e+35 | -9073.83 | 7.27e+31 | +31.86 |
| 45 | `linear_attn.in_proj_qkv.lora_A` | **-nan** | 2394.96 | -- | -- |
| 40 | `linear_attn.in_proj_qkv.lora_A` | **-nan** | 876.065 | -- | -- |
| 20 | `linear_attn.in_proj_qkv.lora_A` | **-nan** | 1262.9 | -- | -- |
| 0 | `linear_attn.in_proj_qkv.lora_A` | **-nan** | -74358.9 | -- | -- |

**The truth is O(1e2..1e4) at every one of the eight blocks** -- 949, 701, 1351, -9074, 2395, 876, 1263,
-74359. That is the strongest confirmation yet that the gradient is a well-behaved number at every
depth, and that the walk's 6.28e35 is not a range artefact of a genuinely huge gradient.

**One rate fits all four finite points, and no family term is needed.** Taking the two `self_attn`
points -- block 63 at -5.11 decades and block 55 at +19.89, eight layers apart -- gives **3.125
decades per layer**:

| block | leaf | observed | predicted from block 63 | residual |
|---|---|---|---|---|
| 63 | `self_attn` | -5.11 | -5.11 | 0.00 |
| 60 | `linear_attn` | **+5.13** | +4.26 | **+0.87** |
| 55 | `self_attn` | +19.89 | +19.89 | 0.00 |
| 51 | `self_attn` | +31.86 | +32.39 | -0.53 |

The one point that crosses the family boundary -- block 60, a gated-delta layer -- sits on the same
line as the three full-attention ones, within 0.87 decades against a per-layer rate of 3.1. The
observed rates are 3.413 over 63->60, 2.970 over 60->51 and 3.125 over 63->55. **An earlier reading of
this table called the block-63 and block-60 errors "equal and opposite to within 6% -- a property that
differs between the two attention families"; that is withdrawn.** It was an artefact of picking two
points three layers apart, where the sign happens to flip because the error passes through zero
between them. The rate is 3.125 decades/layer, against the 3.190 the growth curve gave independently
-- a 2% agreement from two instruments that share nothing but the container.

**And the walk is wrong at block 63, the first layer it touches.** `self_attn.q_proj.lora_A` of the
last decoder layer reads **0.00744629** where the loss's own central difference is **948.917** -- five
decades too *small*. **That is 1.64 layers' worth of the same rate**, so the error does not start in
the layers: it is already present when the walk enters the last block, at a size consistent with
**one to two layers of it originating at or above block 63** -- the head or the objective. The earlier
framing ("the walk grows ~1.6e3 per layer where the truth grows 1.24x") is consistent with this but
does not localise it; this does.

**So the next probe is the head and the objective, not the layers.** The walk's error is one rate all
the way down, already ~1.6 layers deep at the top, and the layers themselves add nothing but that same
rate. Measuring the cotangent where the loss hands it to the head -- `R97_DUMP_DC` on the objective's
own boundary -- is the narrowest remaining question, and it is upstream of everything the bisect
measured.

**The curve then clips, and that is the whole of `all_finite false`.** At position 1143 the magnitude
reaches `3.38953e38`, f32's ceiling, and 1196561 of its 3145728 values are already non-finite; from
position 1122 down, `gdn_o` is non-finite in every value (3145728 of 3145728). **The clip is visible in
the rate itself**: the last transition spans 3.408 decades where a 49-record gap had been running 6.4
to 6.7, i.e. it ran out of exponent range roughly halfway through. And the position agrees: at 3.190
decades per layer, f32's `3.4e38` is **11.1 layers** below position 1416, and the measured clip at
position 1143 is **12.0 layers** below it. The residual stream's own cotangent (`h`) shows the same
growth and the same clipping.

**And the same rate accounts for the block-51 over-statement end to end**, which is the check that ties
the two halves of this section together. Block 51 is **12 layers** below the last layer (64 layers, so
63 - 51). At 3.190 decades per layer the walk grows **38.3 decades** over that span while the truth
grows **1.11**, predicting an excess of **37.2 decades**; the measured over-statement at that leaf is
**31.84 decades** (6.2827e35 against -9073.83). Same order, within a factor of 5.

Two independent consequences of one rate therefore agree: the clip lands at **12.1 layers** below the
top by prediction and **13.8** by position, and the leaf twelve layers down is over-stated by 31.8
decades against 37.2 predicted. Over the whole model the rate gives **64 x 3.190 = 204 decades** of walk
growth against f32's 38.5, which is why everything below the twelfth layer reads NaN.

**The true gradient, from the central difference at two leaves:**

| leaf | walk's gradient | true (central, bf16 step) | over-statement |
|---|---|---|---|
| `model.layers.51.self_attn.q_proj.lora_A` | 6.2827e35 | **-9073.83** | 6.9e31 |
| `model.layers.40.linear_attn.in_proj_qkv.lora_A` | **NaN** | **876.06** | unbounded |
| `model.layers.0.linear_attn.in_proj_qkv.lora_A` | NaN | -- | (first leaf in param order) |

The truth is a well-behaved O(1e3) number at every depth, against a walk that multiplies by ~1.6e3 per
layer. That is the defect, and it is now measured on one buffer through one op rather than divided out
of a logit-space ratio.

**Which explains the NaN leaves without contradicting the rate.** At 3.190 decades per layer, f32's
`3.4e38` is 11 layers below the window's top record -- and the measured clip is 12 layers down. Leaves
at block 40 and block 0 are far *below* that crossing, so they read NaN -- not because the growth
stops, but because it saturated long before. An earlier draft of this section read that saturation as
evidence that no per-layer factor existed; that reading was wrong, and the buffer trace above is what
corrects it.

**What remains unexplained is why it is ~1.38x per record.** A quantiser hop contributes `1/s ~ 5`;
the `1/s` audit above shows all eight producer x consumer pairs balanced at product exactly 1.0, so it
is not the quantiser scales. **It is a per-record factor of ~1.38x made somewhere the scale audit does
not cover**, and `R97_DUMP_DC` at real magnitudes is the instrument that narrows it.

**And it is not inside a single record's row.** Within one layer the cotangents are ordinary: `gdn_in`
reads 0.527 at `gdn_conv_prep`@1413, 0.652 at `gdn_gated_rmsnorm`@1416 and 0.652 at
`lora_expand`@1411. The `gdn_o` span of 0.105 at `had_quant_act_i8g`@1417 to 1128 at
`gdn_gated_rmsnorm`@1416 is **not** a per-call factor -- those are different roles, a read and a write,
so the two numbers are not one row's input and output. What the cross-layer trace establishes is the
rate; what it does not yet establish is which record applies it.

**That instrument is now permanent.** `R97_DUMP_DC`'s op list (`src/backward.cpp:751-768`) covers
`rope`, `lora_expand`, `lora_shrink`, `rmsnorm`, `rmsnorm_had_quant_i8g`, `add`, `gdn_chunk_scan`,
`gdn_kkt_solve`, `gdn_conv_prep` and `gdn_gated_rmsnorm`, and its window is positions 1300..1458, so
complete layers sit inside it. The table above is that run.

**The next narrowing is a per-record handed-vs-left comparison at one layer boundary**, not a wider
window: the growth is visible in `gdn_o`, `gdn_in` and `h` at every layer, and no single record's own
`dx`/`dy` ratio stands out.

**Two mechanical findings came out of the audit and are acted on.** The f32 row reductions: the
report's own f64 fix landed at one of three sites on this path, and the other two
(`gated_rmsnorm_adjoint_kernel`'s `float dot` over 5120, `rmsnorm_adjoint_kernel`'s `float tt`) plus
`l2_backward_kernel`'s `float dot` over 128 are now f64 with their shared memory widened to match,
their host twins' widths. And the fused producers' device row and host row are *different functions*
(`device/quant.hip:743-795` against `host/quant.cpp:982,1003`), so device-vs-host plus host-vs-FD does
not exclude them -- filed as #142, measured negligible here (live in 1 of 78 records, one part in
8.0e5), and it is the last place a factor could hide.

**These fixes explain the non-finite marker; they do not explain the amplification.** Each one removes a
real f32 overflow from a row reduction, and together they moved the first non-finite from `lora_shrink`@1150
to `gdn_gated_rmsnorm`@1143 -- which is what a range fix buys: the instrument keeps working further down
the walk. **The ~1.6e3 per layer is untouched by them** (section "The per-layer factor, measured"), and
that is #55's actual defect. What is also not explained is the one finite leaf: 6.9e31 over its own
truth.

### The first per-hop measurement at the container's own magnitudes

No retained log had a single `R97_DUMP_DC` line, so every device row's verdict until now rested on
synthetic fixtures with small values. One `R97_DUMP_DC=1` pass on the 512-token case produced **4018
lines**, and the answer is unambiguous. `maxabs` is the largest magnitude the row left in that
operand, and `3.40178e+38` is f32's own ceiling (`FLT_MAX` = 3.4028235e38):

| position | op | operand | maxabs | non-finite | forward |
|---|---|---|---|---|---|
| **1150** | `lora_shrink` (blk.50 `ffn_gate_up`) | `h.s8` -- the **scale** cotangent | **3.40178e+38** | 10228 of 20480 | finite |
| 1150 | `lora_shrink` | `h.q8` -- the codes' cotangent | 1.60425e+36 | 0 | finite |
| 1146 | `lora_shrink` (blk.50 `ssm_out`) | `gdn_o.s8` | 3.35119e+38 | 163 | finite |
| 1138 | `lora_expand` (blk.50 `ssm_inz`) | `gdn_in` | 7.24429e+37 | 5230196 of 8388608 | finite |

**Every `fwd_at` is `-1`: the forward is finite at all of them.** So the non-finite is made in the
backward, by the row, not inherited from a poisoned forward.

**The first one is saturation, not a multiplier.** `lora_shrink`'s scale cotangent is a sum of
`g * code` over a group's 128 columns, where `g` is the value-space cotangent (`~1e36` at block 50,
seen here as `h.q8`) and `code` is the int8 code (up to 127). One product is already `1e38`; summing
128 of them in f32 lands on the ceiling and stays there. The host row does this sum in `double`
(`host/reference_lora.cpp:292`, `s_cot->value[ge] += g * a->value[e]`). **This is the fourth instance
of the same class on this path, and it is the site of the first non-finite in the whole walk.**

### The fix, and what it is and is not

Two rows from the same table are now f64 end to end, each against an f64 host twin:

- `shrink_dx_kernel` -- `g`'s reduction over `R`, the per-column product, the shared staging and the
  per-group sum, with only the store narrowing.
- `expand_dt_kernel` -- its sum over the projection's `N` columns, **17408 of them on the live
  trunk**. At position 1138 the whole of `lora.t.r32.lane0` came out non-finite (16384 of 16384),
  which is this reduction landing on the ceiling; the host accumulates it in double
  (`host/reference_lora.cpp:435`).

Both have their scratch buffers and their download widened to match. They build clean on gfx1201
(`atomicAdd(double*)` is available) and the GPU tier stays green.

**What it fixes is the non-finiteness, and it is worth being exact about that.** The magnitude at
block 50 is genuinely `1e36` where the leaf's true gradient is `9073.83` -- the walk is carrying a
cotangent about `1e33` too large well before any reduction overflows. Removing the f32 ceiling removes
a *symptom* that was masking the search: with the first non-finite pinned at a known range defect, the
position of the next one stops being a diagnostic. With it widened, the first non-finite becomes a
measurement again.

**The table also says the amplification is real, large, and upstream of this row.** `lora_shrink`
*received* `1e36` at position 1150 and *wrote* `1.6e36`; the row is not where the factor is made.
What the pass does establish is that the device rows do see the container's real magnitudes, and that
at those magnitudes f32 staging is not enough anywhere on this path.

### Defects and divergences found in passing

- **A device/host divergence in the fused producers, filed as #142.** The device row folds the fused
  bf16 output's own cotangent into its result (`src/adjoint/device/quant.hip:743-795`); `fused_adjoint`
  hands on only the codes' and the scale's (`src/adjoint/host/quant.cpp:982,1003`) and never reads it.
  The two rows are therefore **different functions**, so the device-vs-host plus host-vs-FD argument
  that clears these three ops does not hold for them. Measured negligible on this run (the term is
  live in 1 of 78 `rmsnorm_had_quant_i8g` records, and there it is one part in 8.0e5), and not fixed
  here -- closing it needs the `_ste` fixture to consume the bf16 output, or the new term ships
  untested.

- `src/adjoint/host/gdn_scan.cpp:446-455`: the twin's `unit()` computes `n = u/H`, `h = u%H`, then
  **shadows `h`** with the loop `for (int64_t h = 0; h < g.H; ++h)`, so each of the `N*H` units
  replays every head of its sequence -- values unchanged, `H`x the work (`H=48` live). The adjoint is
  free of it.
- `src/adjoint/device/quant.hip:359-377`: the `SCALERANGE ...` instrumentation used to print 256 lines
  per window unconditionally; it is now gated behind `R97_DUMP_QUANT=1` like the two existing sites
  in that file. The range hazard it reports is real and is #140.

## The 4096-token step, run to completion

The run the issue asks for was made on one card against `/srv/models/rad/qwen3.8-27b-iq5r-s-v4.rad`.
It completes: 4096 tokens, one AdamW update, 800 leaves (400 `lora_A`, 400 `lora_B`), 4095 scored
positions, `names_match` and `shapes_match` both true and `n_other` zero, so the join hands back
exactly the schema's leaves.

| | |
|---|---|
| step | **2946.41 s** (49 min) |
| forward | 104.17 s |
| backward | **2832.82 s** (47 min -- 96 % of the step) |
| head | 6.33 s, 64 chunks, 4095 rows |
| after (the AdamW update) | 105.47 s |
| throughput | 1.39 tok/s |
| `loss` -> `loss_after` | 34682.09 -> 50870.03 |
| VRAM at load / peak | 27.83 / **30.08 GiB** of 31.86 |
| peak snapshot / cotangent | 17.50 / 3.93 GB |

**The forward is finite; the backward is not.** `non_finite.forward_all_finite` is `true`, so every
non-finite value in the result is produced on the way back. That is the same shape the 512-token case
reported, and it rules the container, the loader and the forward out as the source.

### The start-split contract fails, and only `all_finite` can be read on its own

The case's structural check is meant to be independent of magnitude: a perturbation confined to the
start must leave the `A` leaves untouched and move every `B` leaf. **On this run it cannot be read
that way**, and the reason is in the test's own definitions:

- `a_all_zero` is `part != 0.0` over the leaf's gradient (`tests/gpu/device_step.cpp:529`), and
  **`NaN != 0.0` is true**, so one non-finite element sets it false with no real cotangent involved;
- `a_changed`/`b_changed` count leaves whose parameter **moved after the AdamW step** (`:692-697`),
  which a NaN gradient also causes;
- `n_zero_a`/`n_zero_b` count gradients that are **exactly zero** (`:524`, `:536`) -- a NaN gradient
  is not counted.

The artifact settles which case this is: **`first_nonzero_a_finite` is `false`**, so the first non-zero
`A` gradient is itself non-finite, and the test's own comment at `:496` says a non-finite leaf "says
only that the walk reached the leaf".

| check | required | measured | read |
|---|---|---|---|
| `n_grads == n_params`, `names_match`, `shapes_match`, `n_other` | 800, true, true, 0 | 800, true, true, 0 | **pass** |
| `b_all_nonzero` | true | true | pass |
| `n_zero_a` | 400 | **273** | **FAIL** -- 127 `A` leaves hold a gradient, and it is not finite |
| `n_zero_b` | 0 | 0 | pass |
| `all_finite` | true | **false** | **FAIL** |
| `a_all_zero`, `a_changed`, `b_changed` | — | false, 127, 330 | **confounded by the rows above** |

**What this run does establish.** The step completes at the issue's own size and hands back the
schema's 800 leaves with matching names and shapes; the forward is all finite, so the container, the
loader and the forward are ruled out as the source of any non-finite value; the first non-finite value
is at `model.layers.0.linear_attn.in_proj_qkv.lora_A`; and **96 % of the 2946 s step is the backward**
(2832.8 s), which is what an adjoint amplifying by ~1.6e3 per decoder layer costs in time as well as
correctness.

**What it does not establish, and an earlier draft of this section claimed.** That "127 of the 400 `A`
leaves moved when the contract says none may", and that the split is "structurally wrong at both
ends". Both are unsupported: 127 is exactly what poisoning produces, because a non-finite gradient
both moves the parameter and defeats `part != 0.0`.

**The field that settles it now exists.** `tests/gpu/device_step.cpp` records
`n_a_finite_nonzero`/`n_b_finite_nonzero` -- the count of leaves whose sum of squares is finite *and*
non-zero, so `n_*_finite_nonzero + n_zero_* == n_*` holds iff no leaf gradient went non-finite. On the
512-token case it reads **82 of 400** (`A` and `B` alike, with `n_zero_a == n_zero_b == 0`, so 318 of
each factor are non-finite).

**The 4096 case has now been re-run with the field** (`/tmp/r97-dstep-device_step_seq4096-sR5Lxx`,
2026-11-15 09:00, `R97_TEST_ONLY=device_step_seq4096`, 2996.85 s: forward 103.93, backward 2883.52,
head 6.32; `vram_peak` 28.01 GiB):

| field | value |
|---|---|
| `n_a_finite_nonzero` | **0 of 400** |
| `n_b_finite_nonzero` | **274 of 400** |
| `n_zero_a` / `n_zero_b` | 273 / 0 |
| `a_changed` / `b_changed` | 127 / 331 |
| `all_finite` | false |
| `descended` | **false** (loss 34682.089 -> 50870.029, **+16187.9**) |

**`273 + 0 + 127 = 400` exactly**: every `A` leaf is either exactly zero (273) or non-finite (127), and
**not one has a finite non-zero gradient at the issue's own size**. That is the cleanest statement of the
`A` side yet, and it also explains `a_changed == 127` -- the 127 that moved are precisely the 127 that
went non-finite, since a non-finite gradient both moves the parameter and defeats `part != 0.0`.

**The `B` side is the opposite shape and is the better diagnostic**: 274 of 400 finite and non-zero,
none exactly zero. So the 4096 case is not "everything is poisoned" -- it is a walk that stays usable
for most of `B` and has saturated completely on `A`.

**And the case now fails its own descent check.** `device_step_seq4096: threw: check failed: descended`,
`1 tests, 1 failed` -- the loss rises by 16187.9 after one AdamW step, where the 512-token case's
descent held. The first non-finite is `gdn_chunk_scan`'s operand `gdn_k`[1] at position 463, from the
row. So on this case the acceptance failures are three, not one: `all_finite`, `a_changed == 0`, and
`descended`.

**One oddity worth naming on its own, and it holds at both sizes.** On the 512-token case 70 of the 400
`B` leaves have a non-zero gradient (`n_zero_b == 0`) yet their value did not move under the step; on
the 4096 case the same reading is 69 (`b_changed` 331 of 400 with `n_zero_b == 0`). An AdamW update
should leave no parameter with a non-zero gradient bit-identical, so this is unexplained; it is not a
zero-gradient result and should not be read as one. **It is not an underflow threshold either**: at
step 1 with bias correction AdamW's update is `lr * m_hat / (sqrt(v_hat) + eps) = lr * 1 = 1e-3`,
against a parameter near 0.0136 whose ulp is 9.3e-10 -- eight orders of magnitude apart.

**The `A` side of the 4096 case is the one that closes exactly**, and it is worth stating on its own:
273 exactly-zero + 127 non-finite + 0 finite-non-zero = 400, and `a_changed == 127`. The `B` side does
not close the same way -- 274 finite-non-zero, 0 zero, 126 non-finite, but 331 moved -- so 57 of the
non-finite `B` leaves moved and 69 did not. That asymmetry is unexamined and should not be read as
either result until it is.

**A separate defect worth its own line: the case is not run-to-run deterministic.** Four runs of the
512-token case with the same seed and the same build give `a_changed` 374, 374, 375, 375 and
`b_changed` 375, 375, 374, 377, while `loss -> loss_after` is bit-identical across all four
(7371.59 -> 6347.88, -1023.71). The earlier `512d`/`512e`/`512f` runs likewise named different first
non-finite ops, and the 4096 re-run gives `b_changed` 331 against the earlier run's 330 at the same
`loss`. **A bit-identical loss with a differing changed-leaf count says the non-finite values differ
run to run** -- which is what a NaN/Inf entering a sum whose order is not fixed would look like, and
it is not the same question as the amplification. It is recorded as a separate defect, not folded
into the factor.

**The acceptance criterion is still not met**, and `all_finite` is the check that decides it: the
adapter gradient exists and is the right shape, but it is non-finite at block 0. That is the same
defect the 512-token case localised to a ~1.9e31 excess growth over the backward walk, now shown to
hold at the issue's own size.

**What the 4096 case adds beyond the 512:** the failure is not an artefact of a short sequence. With
4096 positions the backward takes 47 minutes and produces the same non-finite leaf at the same
place, so the defect is in the walk's arithmetic and not in a corner the short case happened to take.
It also puts a number on what the defect costs: **96 % of the step is the backward**, which is what
an adjoint amplifying by ~1.6e3 per decoder layer costs in time as well as in correctness.
