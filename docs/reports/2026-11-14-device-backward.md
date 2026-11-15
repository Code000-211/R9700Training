# The device backward: a real step on the 27B container

2026-11-14. One AdamW step on `qwen3.8-27b-iq5r-s-v4.rad` with the adapter's gradient
computed entirely on the card (tracker issue #55, plan R97 #53). Two cases, and their
pass conditions are **opposite** by design: the 512-token case starts with the adapter
seeded over both factors, the 4096-token case at the seed-0 start where `lora_B` is
exactly zero.

Both cases are `tests/gpu/device_step.cpp`; the assertions quoted below are that file's
(`:965-978`), not a reviewer's.

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

### The measurement: ~984x per block on average, and accelerating

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

### Where the factor is not, and the one group it can still be in

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
the `+ cot_h` fused-output term and the norm backward, so no factor is visible by inspection. What
*is* visible is a range hazard: the row computes `cot / s` in f32 (`quant.hip:199`), and this record
carries code cotangents of 2.44e30..2.62e33 while `gdn_o`'s reach 1e35..1e38 -- with live scales as
small as 5.96e-05, the quotient crosses f32's 3.4e38. That is consistent with the report's
first-non-finite being this row's in-place output, and it is a real defect on its own, though an
overflow gives Inf rather than a large finite multiplier.

**That gap is now closed.** The three producers have host twins and host adjoints
(`src/adjoint/host/quant.cpp`), built as the composition the GPU tier already had in its test TU, and
each has a linearised `<op>_ste` row so that a finite difference can differentiate it -- these are
quantisers, so their forward is a step unless the row is linearised at a base point. Held to their
own differences at two shapes they read 1e-9 to 1e-11, against 1e-12 to 1e-9 for the fp8 control in
the same run. So the three producers' **host** side is now excluded, and the remaining side no
finite difference has reached is the **device** row's `cot / s`, which is #140.

### Two defects found in passing

- `src/adjoint/host/gdn_scan.cpp:446-455`: the twin's `unit()` computes `n = u/H`, `h = u%H`, then
  **shadows `h`** with the loop `for (int64_t h = 0; h < g.H; ++h)`, so each of the `N*H` units
  replays every head of its sequence -- values unchanged, `H`x the work (`H=48` live). The adjoint is
  free of it.
- `src/adjoint/device/quant.hip:359-377`: the `SCALERANGE ...` instrumentation used to print 256 lines
  per window unconditionally; it is now gated behind `R97_DUMP_QUANT=1` like the two existing sites
  in that file. The range hazard it reports is real and is #140.

## The 4096-token step, run to completion

The run the issue asks for was made on one card against `/srv/models/rad/qwen3.8-27i-iq5r-s-v4.rad`.
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
(2832.8 s), which is what a per-op adjoint amplifying by ~250x per block costs in time as well as
correctness.

**What it does not establish, and an earlier draft of this section claimed.** That "127 of the 400 `A`
leaves moved when the contract says none may", and that the split is "structurally wrong at both
ends". Both are unsupported: 127 is exactly what poisoning produces, because a non-finite gradient
both moves the parameter and defeats `part != 0.0`.

**The field that settles it now exists.** `tests/gpu/device_step.cpp` records
`n_a_finite_nonzero`/`n_b_finite_nonzero` -- the count of leaves whose sum of squares is finite *and*
non-zero, so `n_*_finite_nonzero + n_zero_* == n_*` holds iff no leaf gradient went non-finite. On the
512-token case it reads **82 of 400** (`A` and `B` alike, with `n_zero_a == n_zero_b == 0`, so 318 of
each factor are non-finite). **The 4096 record predates the field**, so its own value is still
unmeasured; the qualitative answer there is already known to be "substantially fewer than 400", since
`first_nonzero_a_finite` is false.

**One oddity worth naming on its own.** 70 of the 400 `B` leaves have a non-zero gradient
(`n_zero_b == 0`) yet their value did not move under the step. No AdamW update should leave a
parameter with a non-zero gradient bit-identical, so this is unexplained; it is not a zero-gradient
result and should not be read as one.

**The acceptance criterion is still not met**, and `all_finite` is the check that decides it: the
adapter gradient exists and is the right shape, but it is non-finite at block 0. That is the same
defect the 512-token case localised to a ~1.9e31 excess growth over the backward walk, now shown to
hold at the issue's own size.

**What the 4096 case adds beyond the 512:** the failure is not an artefact of a short sequence. With
4096 positions the backward takes 47 minutes and produces the same non-finite leaf at the same
place, so the defect is in the walk's arithmetic and not in a corner the short case happened to take.
It also puts a number on what the defect costs: **96 % of the step is the backward**, which is what
a per-op adjoint that amplifies by ~250x per block costs in time as well as in correctness.
