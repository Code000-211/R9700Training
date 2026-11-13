# The device join: one step's adapter gradient on one card

`r97::device_step` (`include/r97/step_device.h`) is the card's `StepFn`: it forwards a batch
through the real container on the device kernel tier, walks the step back with the device adjoints
and returns the adapter's gradient. It is the device twin of `r97::host_step`
([`STEPS-HOST.md`](STEPS-HOST.md)), with the same block-walk structure and the same pure helpers
(`src/join_common.h`), and it differs in three things: the forward values the walk reads are pulled
back from the card, the head is never held whole, and the gradient is staged on the card.

## The tier it needs

The driver must name the card's library **and** `libr97`, the plugin the device rows load from
(`kernels = {"libr4d", "libr97", "libavx", "libref"}`). `libr97` serves no forward kernel; it is
in the hierarchy so the engine `dlopen`s it, and its static initialisers register the device rows
(and the head's chunk kernel, and the vocabulary head's kernels) into this process's registry.
A driver opened on the host tier has none of them and the walk refuses by name.

The join also needs the training patch set: the issue observer (0101) for the tape, the replay
block (0102) for a block's operands, and the context stream (0122) so the residual store keeps its
boundaries on the stream the step's ops issue on. `R97_DEVICE_STEP` is defined only when all three
are in the build.

## The path

1. **Forward, with the adapter live.** The join installs the driver's engine hook, opens the live
   adapter from the delta's `WireBuffers` and fills it from `params` before `serve()`, exactly as
   the host join does. The step's tape is the engine's own record of the issue.
2. **The residual store.** After the model's load and before the serve, the free VRAM is read
   (`rad_dev_props(0).vram_free`, or `DeviceStepOptions::residual_budget_bytes`) and the store is
   opened on the step's stream: `L+1` boundaries, plus the second copy at every boundary a block
   after the first starts at (`L-1` of them, the int8 trunk reads the previous block's output
   beside the residual). `2L` copies of `tokens x hidden x 2 B` — 5 GiB at 4096 tokens, 128
   copies. The store refuses with the plan's numbers when it does not fit.
3. **Each block, replayed and walked.** In reverse order: the replay re-issues the block's ops
   from the stored residual on the card, and the walk reads that block's operands through the
   tape, widens them to the f64 `AdValue` contract (`snapshot`), and runs `backward_block` with
   `BackwardOptions::side = Side::device`. The cotangents are the block above's; the block's
   device rows add into the step's card staging. One block's snapshots live on the host at a
   time, as in the host join.
4. **The head, chunk by chunk.** A logit row is 248320 wide, so its cotangent is never
   materialised. The scored rows go through `head_ce_loop` in chunks of `kHeadChunkRows` (64):
   the fill copies the engine's f32 logits chunk into the chunk's bf16 (`vocab_head_cast`), the
   CE kernel writes `(softmax - onehot) / scored`, and the sink consumes that gradient with the
   transposed product (`vocab_head_transpose`): per class, `dy`'s columns gathered by the class's
   ids, the class's IQR weight decoded to its int8 grid chunk by chunk, the int8 x bf16 product,
   `scale.col`, the sum over the four classes, and the inverse fwht per row — accumulated into
   `d_hout` `[tokens, hidden]` f32. `d_hout` seeds the head's remaining records (the final
   `rmsnorm` and the `gather_rows` before the logits op), which are walked with the device rows.
5. **The flush.** The card's staging holds one f32 buffer per leaf the step's rows named. It is
   copied out once, rounded to bf16, and split into the delta's leaves by the naming the walk
   used (`fused_slots` / `slot_prefix` / the record's prefix), then ordered like `params`. Any
   error path discards it, so the next step does not add to a dead total.

`grads == nullptr` is a forward-only call: the forward runs with the adapter fed and the report is
filled, but the head's gradient pass and the block walk are skipped and nothing is staged. That is
the evaluation pass of a training loop, and the pre-step reference of a check.

## What it holds, and what it does not

| | at 4096 tokens |
| --- | --- |
| residual store (card) | 5 GiB (`2 x 64` copies of 4096 x 5120 x 2 B) |
| engine (weights, KV, arena) | 20.2 GiB, measured ([`reports/2026-11-12-device-forward.md`](reports/2026-11-12-device-forward.md)) |
| `d_hout` (card) | 84 MB |
| the CE chunk (card) | 64 x 248320 x 2 B for the logits and the same for the gradient |
| one block's f64 snapshots (host) | ~2.5 GB |
| the card's gradient staging | one f32 buffer per leaf, 320 KiB + 768 KiB per `q_proj` at rank 16 |

`DeviceStepOptions::keep_boundaries` is off by default. On, the report keeps every boundary's f64
value **and** cotangent: 65 boundaries x 4096 x 5120 x 8 B is ~10 GiB each way, ~21 GiB for both.
A check turns it on
for a batch that fits (512 tokens is 2.7 GiB); a training loop never asks.

Not in this step: the narrow-fold `gemm_nt_q_gated` (a 4096-token step takes the `gemm_nt_q` +
`gated_had_quant_i8g` pair, whose rows exist), the `rope_table` op (off the walk's path), and the
logits op itself (the join handles it; no adjoint row exists for it, by design).

## Where a non-finite gradient is attributed

A step whose gradient holds a NaN or an inf reports where the walk first met one, instead of
leaving it to a bisect. `BackwardReport` (and, summed over the step's walks, `DeviceStepReport`)
carries `first_non_finite_op` / `first_non_finite_operand` / `first_non_finite_operand_index` /
`first_non_finite_position` / `first_non_finite_index` and `first_non_finite_from_row`: the op whose
row was being walked, the tape buffer of the operand whose cotangent held the value, its index in
the record and the value's offset in that operand's payload, and whether the row produced it or the
walk handed it in (a seed, the cotangent from above, or a record later in the forward). Each
`OpTiming` in `row_seconds` counts the slots of its op that carried one (`non_finite`). The same
question is asked of the forward's own operands (`forward_all_finite` and `forward_first_*`): a NaN
born in the forward is a different bug from one born in the walk. The scan is one pass over each
slot the walk already holds, stopping at the slot's first non-finite value, and changes no value,
order or row.

`device_step_tape` prints it as one line, after the walk's per-op split:

```
device_step_tape: non-finite: cotangent op gemm_nt_q, operand 'blk.3.ffn_gate_up.weight'[2] at position 812, value 4097, from the row | forward all finite | per op: gemm_nt_q 2/64
```

`cotangent none` and `per op: (none)` mean the walk handed on only finite values; `from the walk's
input` instead of `from the row` means the op merely passed the value on. The results JSON carries
the same numbers under `non_finite` and per op under `row_seconds[].non_finite`.

## How it is checked

`tests/gpu/device_step.cpp`, model tier, a child process per case (ROCR is initialised once):

- `device_step_tape` (512 tokens) — the op-coverage probe: every op the 64 blocks issued has a
  device row, and the head's `rmsnorm`/`gather_rows` do; the exceptions are named. Then a full
  step with the report's boundary copies on.
- `device_step_seq4096` — the ticket: one 4096-token step. The loss equals the forward-only call's,
  the gradient names every leaf of the delta, every `lora_A` leaf's gradient is exactly zero (the
  init leaves B at zero) and every `lora_B` leaf's is not, one AdamW step moves only the `lora_B`
  leaves, and a second forward with the updated adapter is strictly lower — the gradient's
  direction, not just its shape.

The row-level checks are the kernel families' own: `tests/gpu/adjoint_{linear,lora,attention,...}.cpp`
and `tests/gpu/vocab_head.cpp`.

## Reproducing

```sh
cmake -B build -G Ninja && cmake --build build -j8
ctest --test-dir build -L host                      # no card
ctest --test-dir build -L gpu                       # one card, serial; the model cases skip without the container
```

The numbers this file's tables carry are in
[`reports/2026-11-13-device-backward.md`](reports/2026-11-13-device-backward.md).
