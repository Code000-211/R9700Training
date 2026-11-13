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
| forward / head / backward / step | 46.9 s / 0.8 s / 1176.9 s / 1227.0 s |
| residual boundaries / store | 65 / 640 MiB |
| peak cotangent / VRAM peak | 490.9 MB / 20.28 GB |
| `all_finite` | **false** — `grad norm nan` |
| `a_all_zero`, `b_all_nonzero` | false, false |

The walk's split (288 `gemm_nt_q` calls and 288 `lora_*`, 48 `gdn_chunk_scan`, 48
`had_quant_act_i8g`): `gemm_nt_q` 76.3 s, `lora_expand` 5.5 s, `gdn_chunk_scan` 5.4 s,
`lora_shrink` 5.4 s, `had_quant_act_i8g` 1.6 s — **~95 s of the 1177 s backward**. The rest is the
walk's own per-op work (the f64 snapshot of every operand, the distribute, the store), which is why
the parallelisation ticket exists and why the card sat at 3–5 % busy throughout.

The NaN is born in the **forward**, not the walk: `forward_all_finite` is false, the first
non-finite operand is `h` (position 1459, value index 62), and the walk's first bad cotangent is
`rmsnorm`'s `x` at 1458 — the reader, not the producer. `b_all_nonzero` false is a second,
independent gap: the check sums squares, so a NaN cannot make it read zero; one `lora_B` leaf
really gets an exactly-zero gradient.

The op coverage probe is separate and still runs: the tape declares 224 `gemm_nt_q` and 64
`gemm_nt_q_gated` calls, and `probe_missing` lists the ops served by the host fallback rather than
a device row.

## device_step_seq4096 — the ticket

**Not run.** At the 512 case's measured rate the backward is ~2.6 h at 4096 tokens (~2.8 h for the
step), which is a window nobody wants to hold while the forward's NaN is still unexplained: a
longer run of a step whose gradient is not finite proves nothing. It runs once the forward is
finite, and its figures go here: loss (cross-checked against
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
