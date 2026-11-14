# Device backward: one 4096-token step on one card — 2026-11-13

The device join (`r97::device_step`, `include/r97/step_device.h`) over the container on disk:
`/srv/models/rad/qwen3.8-27b-iq5r-s-v4.rad` (`qwen35`, 64 layers, hidden 5120, vocab 248320,
`fwht128`), on one AMD Radeon AI PRO R9700 (gfx1201, 31.86 GiB VRAM). The step runs the forward
through the device kernel tier, keeps the residual at every layer boundary in one VRAM allocation,
walks all 64 blocks back with the device adjoints (`Side::device`, `libr97.so`'s rows), takes the
head's gradient through the chunked CE loop and the transposed vocabulary product, and returns the
LoRA adapter's gradient as a `ParamSet`. The path and its memory are `docs/STEPS-DEVICE.md`'s.

The tier is `--kernels libr4d,libr97,libavx,libref`: libr4d's fused HIP kernels, libr97 the plugin
the device rows and the head's kernels register from (it serves no forward kernel), libavx/libref
as the per-op fallback.

Every number below comes from the revision this report ships with, run under the shared GPU lock
(`/tmp/kilo/gpu.lock` and `/tmp/gpu.lock`) with prod `radiance` stopped and both cards released
(57 MiB used of 32624 MiB each when the window opened), so nothing is contended. The model cases
ran with `R97_MODEL_DIR` naming the container: the build's `R97_MODEL_RAD` macro resolves to
`data/qwen3.8-27b-iq4r-s-v4.rad`, which does not exist on this host, and the default directory
`/srv/models/qwen3.8-27b-iqr-v4` does not exist either, so without the variable a model case skips
(exit 77) rather than runs.

## device_step_tape — the op coverage, and a full 512-token step

The case runs and descends; **it does not pass**, and the failure is the gradient's, not the loss's.

| figure | value |
| --- | --- |
| scored | 511 of 512 |
| loss → loss_after (one AdamW step) | 7371.591348 → 6347.883967 |
| forward / head / backward | 46.72 s / 0.82 s / 581.66 s |
| residual boundaries / store | 65 / 640 MiB |
| peak snapshot store / op / cotangent | 1899 MiB / 204 MiB / 490.9 MB |
| shared (frozen) snapshot bytes | 0 (the run holds `R97_NO_SHARED_SNAPSHOTS=1`) |
| `all_finite` | **false** — the forward's own inf, contained by nothing yet (below) |
| `n_zero_a`, `n_zero_b` | **0 of 400, 0 of 400** — every leaf moves (`a_all_nonzero`/`b_all_nonzero` true) |
| `n_lora_a`, `n_lora_b`, `n_params` | 400, 400, 800 — `names_match` true, 64 blocks probed |
| `first_nonzero_a_finite` / `_b_finite` | **false** / **false** — `model.layers.0.linear_attn.in_proj_qkv.lora_A` / `…lora_B` |
| row counters (`rmsnorm`, the whole walk) | **0 zeroed / 0 readable / 0 refused** over 1459 calls |

Measured twice, and **the two runs differ in exactly one figure**: `/tmp/r97-dstep-device_step_tape-23nofj`
(18:35, the device path marking the head walk's own input as external) and
`…-PLpzBN` (18:54, marking the objective's logits, `e9eae50`). Same loss to the last digit
(7371.591348378961 → 6347.883966664209), same `scored` 511, same 400/400 leaves, same
`all_finite: false`. The counters are what separates them from a protected run: **`0/0/0` means the
rule never fired** — this gradient was produced with no read set at all, not one that examined the
rows and found nothing to zero. The 512-token expectation is `511 zeroed / 1 readable` on the head's
norm (`gather_rows` selects one row of 512), and **that has not been observed yet**; the head walk
now prints its decision (`r97: head read set: N of M records proven [...]`) so the next run names
which condition fails instead of leaving it to be guessed.

The phase split is the same in both: forward 46.8 s, head 0.82 s, backward 572.2 s, the optimizer
step 46.5 s, and the step is **host-bound** — 388% CPU against 6% `GFX_ACTIVITY`, because every
replayed operand is pulled to the host and joined in f64 with the pulls serialised against the
arithmetic. That is a property of this harness, not of the model: it is a correctness harness, and
its wall clock is not a throughput claim.

The NaN's producer is the forward's, and it is named:

    device_step_tape: non-finite: cotangent op rmsnorm, operand 'x'[0] at position 1458, value 0,
      from the row | forward not finite: rmsnorm wrote operand 'h'[2] at position 1458, value 62

i.e. the `rmsnorm` record at position 1458 wrote element 62 of its output `h` non-finite — row 0,
column 62 of the final hidden — and the walk's first bad cotangent is that record's input. **This is
a known forward defect, not a cure**: the containment the read set provides only reaches a row no
consumer reads, and only when the derivation proves the record. A non-finite value *inside* a row a
consumer reads is refused by name, which is the correct outcome and the reason `all_finite` is
false rather than silently zero.


Measured with the IQR decode live (the stored path is the default, `R97_STORED_WEIGHTS` retired) and
the plugin-private planes read from the tape's own pointers. The backward is **572.2 s**, against
the 1176.9 s the same case took when every stored operand was refused and the walk fell back to the
f64 rows, because the 288 `gemm_nt_q` records no longer pull 1.4 GB of weight each. The pull's own
largest entries are `gemm_nt_q` 1.9 s and `lora_expand` 1.4 s.

**Reading the plugin-private weight was necessary and is not the whole story**: it is what makes
`dx` exist at all, and the 399-of-400 zero leaves it used to explain are gone — with the quant row
reaching the tape's pointers and the read set threaded through, **all 400 `lora_A` and all 400
`lora_B` leaves carry a gradient** (`n_zero_a`/`n_zero_b` 0, `names_match` true). What remains is
finiteness, and that is the forward's.

The per-op non-finite counts from the same run: `add 3/1`, `gated_had_quant_i8g 2/64`,
`gdn_conv_prep 4/48`, `gemm_nt 8/48`, `gemm_nt_q 2/288`, `lora_expand 3/288`, `lora_shrink 3/288`,
`rmsnorm 1/33`, `rmsnorm_had_quant_i8g 5/128`.

An absent operand (no pointer) and an operand the tape does not name (the raw adapter-slot array,
whose zero is a valid slot index) keep the zeros. `tests/host/lora_shrink_step_order.cpp` and
`tests/host/snapshot_view.cpp` hold the cases.

The op coverage probe is separate and still runs: the tape declares 224 `gemm_nt_q` and 64
`gemm_nt_q_gated` calls, and `probe_missing` lists the ops served by the host fallback rather than
a device row.

## device_step_seq4096 — the ticket

**Running, and its figures are not in yet.** A 6000 s window (`/tmp/r97-inner-4096.sh`, opened
18:54:23) holds the card for it. At the 512 case's measured rate the backward is ~2.6 h at 4096
tokens, so a window that ends before the case does reports nothing rather than a partial figure;
the section is filled from the case's JSON when it lands, and not before. Its figures go here: loss
(cross-checked against
`docs/reports/2026-11-12-device-forward.md`'s 34682.089228 / 8.469375 for the same batch),
`loss_after`, the gradient norm, the store and VRAM peaks, the phase split, and tokens a second.

## The device rows and the head's kernels

`ctest -R '^gpu_tests$'` — the container-free device suite, 751.67 s, all cases passed. The rows
this change adds or extends, as the harness printed them:

| row | case | result |
| --- | --- | --- |
| `gather_rows` | 4x17x128 … 4099x4096x5120, `every element` | rel 0.0015-0.0022, scratch peak 9 KiB … 122906 KiB |
| `gather_rows` | 7x4x37, the edges | rel 0.0021, every element |
| `gather_rows` | the accumulation dropped (negative) | rejected, 3727 mismatches, first at `dc[0]` |
| `gather_rows` | f32 operands | left to the host row: no device adjoint for `gather_rows` f32 |
| `lora_shrink`/`lora_expand` | r4, r16, r2048; 1024 and 4096 tokens, one sequence; 448+320+256 tokens | `device == host`, gradients worst 0.0020-0.0033 of the block's largest |
| `lora` pooled | pool3, pool2, `gate_up` r32 two modules | fused and per-slot names both `device == host` (fused worst 0.0022, per-slot worst 0.0035) |
| `lora` seeds | `dt` zero with `B = 0`; zero cotangent of `t` | `dB` filled (2048 of 2048); `dA` exactly zero |
| `qk` | 4096 tokens, 1 sequence, with and without the pool | `device == host`, scratch peak 574498 / 533522 KiB |

## What the step does not do

The narrow-fold `gemm_nt_q_gated` is not on this path (a 4096-token step takes the `gemm_nt_q` +
`gated_had_quant_i8g` pair, whose rows exist); `rope_table` is off the walk's path; the logits op
itself has no adjoint row by design, the join's transposed product replaces it. A driver opened on
the host kernel tier is not refused up front — it surfaces at the first operand pull or at
`adjoint_find`, naming the operand or the op. The join's gradient path is the vocabulary-weighted
head's by construction: a container with a plain `logits_gemm` head runs the forward-only call and
refuses the gradient call with the plan's own message.

## Reproducing

```sh
cmake -B build -G Ninja && cmake --build build -j8
R97_MODEL_DIR=/srv/models/rad/qwen3.8-27b-iq5r-s-v4.rad \
  ctest --test-dir build -R '^gpu_tests$' -j1 --output-on-failure
R97_MODEL_DIR=/srv/models/rad/qwen3.8-27b-iq5r-s-v4.rad \
  ctest --test-dir build -R '^gpu_model_tests$' -j1 --output-on-failure
```
