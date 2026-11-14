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
gradient was not NaN and the step did not ascend into a non-finite parameter. It is
consistent with three things that are not yet distinguished: a wrong-but-finite gradient, lr
overshoot on 400 rank-16 leaves at 1e-3, and the case's own zero-`A` contract at this start.
Attributing it needs the numbers, which the fix above now guarantees.

**The next run** is the 512 confirmation first (the forward inf below is the live defect),
then this case again against its own contract — `a_all_zero`, `b_all_nonzero`,
`a_changed == 0`, `b_changed == n_b` — with the wall clock.

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
