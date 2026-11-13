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

**Pending.** The case is in the tree and its op-coverage probe runs; the 512-token walk has not
completed a measured run yet, so this section carries no figures. The run that fills it is the
one this report's window is for: the walk's per-op split (`row_seconds`, `pull_seconds`) is now
printed and written to the case's JSON, so a single window yields both the total and the ops that
hold it. What is already known, from the record and the refused runs:

- The 512-token tape declares 224 `gemm_nt_q` and 64 `gemm_nt_q_gated` calls; the fold's host
  adjoint now exists (`src/adjoint/host/gemm_nt_q_gated.cpp`), so the walk no longer refuses there.
- Before the walk's cotangent slots were sized only for the operands a row may write, every call
  zero-filled a cotangent for the fold's `[34816, 5120]` weight (1.4 GB) and the head's four class
  weights (~10 GB); `peak_cotangent_bytes` is what the walk actually held.

## device_step_seq4096 — the ticket

**Pending.** Not yet run: the 4096-token step needs a window sized by the 512 case's measured walk,
and the case refuses before that measurement exists. The numbers this section will carry are the
loss (to be cross-checked against `docs/reports/2026-11-12-device-forward.md`'s 34682.089228 /
8.469375 for the same batch), `loss_after`, the gradient norm, the store and VRAM peaks, the
forward/head/backward split, and tokens a second.

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
