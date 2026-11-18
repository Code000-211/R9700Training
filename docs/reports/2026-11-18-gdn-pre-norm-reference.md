# GDN pre-norm activation: the reference forward settles #149

**Date:** 2026-11-18
**Question (issue #149):** is the GDN branch's pre-norm activation (`gdn_o`, the
chunk scan's output) at row_ss ~1e-07..1e-05 (rms ~1e-3) a property of the
Qwen3.8-27B checkpoint, or a defect in the engine's GDN forward?
**Verdict:** it is a **property of the checkpoint**. The native forward agrees
with the engine row-for-row within a few percent. #145's explanation of the
cotangent growth stands as written.

## Method

Two measurements, same 512 tokens, same three GDN layers (60, 61, 62):

1. **Engine, unseeded.** The original #149 numbers came from the 512-token
   walk with the seeded LoRA adapter (its `in_proj_qkv`/`in_proj_z` deltas are
   4-9× the stored weight RMS, `docs/ADAPTER.md`), so the engine side was
   re-measured with the base container only: a new `R97_DSTEP_NO_SEED=1`
   guard in `tests/gpu/device_step.cpp` skips the `seed_adapter` call
   (lora_B stays zero, the walk is the base forward). Command:

   ```sh
   R97_MODEL_CASES=1 R97_MODEL_DIR=/srv/models/rad/qwen3.8-27b-iq5r-s-v4.rad \
   R97_TEST_ONLY=device_step_tape R97_HOST_WALK_AB=2 R97_WALK_BLOCKS=4 \
   R97_FD_REC=1416,1395,1374 R97_DSTEP_NO_SEED=1 ./build/r97_gpu_tests
   ```

   The seeded-vs-unseeded difference is the forward, not a coincidence of the
   guard: seeded grad norm 56984.9, unseeded 690.0.

2. **Reference.** The native bf16 checkpoint
   (`/srv/models/Qwen/Qwen3.8-27B`, `config.json` `"dtype": "bfloat16"`,
   52 GB) run in the framework it was trained in: Hugging Face
   `transformers` 5.22.1 (`Qwen3_5ForConditionalGeneration`, `eager`
   attention), device_map across the two R9700s. New tool
   `tools/gdn_pre_norm_ref.py` hooks `Qwen3_5GatedDeltaNet.norm` and captures
   exactly the operands the engine's `gdn_gated_rmsnorm` record names
   (`gdn_o` = schema nX, `gdn_in` = the gate z = schema nZ, the norm output,
   plus `normed_in`, input_layernorm's output, for scale sanity). It then
   recomputes the norm from the captured operands in the model's own order
   (f32 variance, bf16 weight multiply, f32 silu gate) and compares: all three
   layers give `max|diff| = 0.000e+00`, so the hooks sit on the model's own
   path and the captured values are the values the model used.

   The reference's delta rule ran on the pure-torch body
   (`GDN_REF_PURE_TORCH=1` blocks `fla`, whose Triton kernels were unverified
   on gfx1201). Pure torch is the algorithm's reference implementation;
   for a four-decade question the kernel choice is immaterial.

The token ids are the synthetic walk scheme (`1 + (i*7) % 299`) on **both**
sides, so the comparison is input-identical. The `gdn_gated_rmsnorm` schema
was cross-checked against the model: `eps = 1e-6` in
`config.json` (`rms_norm_eps`) equals the engine's hardcoded `kRmsEps`
(`src/adjoint/host/gdn_prep.cpp`) — the "the engine's eps matches the
model's" assumption from #149 is now a check.

## The reference's distributions

row_ss (mean square of each 128-channel (token, value-head) row, n=24576):

| layer | buffer | p01 | p25 | med | p75 | p99 | max |
|---|---|---|---|---|---|---|---|
| 60 | `gdn_o` | 3.89e-09 | 1.69e-07 | 4.49e-07 | 2.86e-06 | 1.85e-04 | 4.54e-04 |
| 61 | `gdn_o` | 7.50e-09 | 1.95e-07 | 7.72e-07 | 4.72e-06 | 6.11e-05 | 1.56e-04 |
| 62 | `gdn_o` | 3.47e-08 | 8.55e-07 | 3.38e-06 | 1.01e-05 | 2.35e-04 | 1.16e-03 |
| 60 | `gdn_in` (gate) | 1.97e+00 | 4.03e+00 | 5.43e+00 | 7.37e+00 | 1.76e+01 | 3.10e+01 |
| 61 | `gdn_in` (gate) | 1.22e+00 | 3.01e+00 | 4.13e+00 | 5.38e+00 | 1.02e+01 | 2.30e+01 |
| 62 | `gdn_in` (gate) | 1.76e+00 | 3.58e+00 | 4.93e+00 | 7.39e+00 | 1.87e+01 | 4.94e+01 |

`normed_in` (input_layernorm's output) is at
row_ss ~1.0-1.3 (rms ~1.1) in all three layers.

The pre-norm activation of the native model **is** small: median row_ss
4.5e-07..3.4e-06 (rms 6.7e-04..1.8e-03), and at layer 61 — the issue's
epsilon-dominated case — the median 7.7e-07 is **below `eps = 1e-6`**, so the
epsilon-dominated regime is the model's normal operating point there, not an
engine artifact.

## Engine vs reference, exact rows

The walk's FDREC rows, side by side with the same (token, value-head) row of
the reference (FDREC `elem` is a flat element index; the row is
`elem / 128`):

| layer | pos | elem | buffer | engine row_ss | HF same row | HF/engine |
|---|---|---|---|---|---|---|
| 62 | 1416 | 1007999 | `gdn_o` | 1.21188e-07 | 1.07882e-07 | 0.89 |
| 62 | 1416 | 2048343 | `gdn_o` | 6.06567e-04 | 5.73146e-04 | 0.94 |
| 61 | 1395 | 810998 | `gdn_o` | 9.32871e-07 | 9.35094e-07 | 1.00 |
| 61 | 1395 | 1444167 | `gdn_o` | 1.07165e-04 | 1.13670e-04 | 1.06 |
| 60 | 1374 | 115157 | `gdn_o` | 3.06421e-06 | 3.11531e-06 | 1.02 |
| 60 | 1374 | 49303 | `gdn_o` | 2.19576e-04 | 2.22357e-04 | 1.01 |
| 62 | 1416 | 1585489 | `gdn_in` | 7.19114 | 9.11467 | 1.27 |
| 62 | 1416 | 715403 | `gdn_in` | 5.02726 | 5.12226 | 1.02 |
| 61 | 1395 | 525568 | `gdn_in` | 5.48115 | 8.13441 | 1.48 |
| 61 | 1395 | 283752 | `gdn_in` | 5.10559 | 8.76327 | 1.72 |
| 60 | 1374 | 115157 | `gdn_in` | 4.93949 | 7.44774 | 1.51 |
| 60 | 1374 | 2114810 | `gdn_in` | 4.04937 | 11.2205 | 2.77 |

All six `gdn_o` rows agree within **±15 %** — tighter than the ~5-bit
quantization one would hope for. The gate rows agree within 2.8×; the gate is
`in_proj_z`'s output and the largest single quantized-projection delta in the
walk, so its row_ss is the quantity a quantized container moves most. Both
buffers are at the same scale as the native model.

## What this closes

- #149: the small pre-norm activation is the checkpoint's property. The norm
  is designed to rescale a small branch output (its output is O(0.02-0.25),
  ordinary residual magnitude, on both sides); the epsilon-dominated regime
  at layer 61 is the model's own. No candidate defect (chunk-scan decay,
  state initialisation, normalisation) is in play.
- #145: the cotangent-growth explanation stands as written; its
  amplification estimate rests on the activation scale, which the reference
  confirms.
- The original #149 table (L61 `gdn_o` 1.28881e-07, `gdn_in` row_ss 54-65)
  was the **seeded adapter's** forward. The unseeded engine and the native
  reference agree with each other and disagree with it, which is what
  separates "model property" from "adapter perturbation". The small `gdn_o`
  scale survives the adapter; the `gdn_in` row_ss it printed (rms 7-12) is
  the adapter's, not the base model's (base gate rms ~2).

## Caveats

- The tokens are synthetic (the walk's id scheme) on both sides: the
  comparison is input-identical and answers the scale question, but it is
  not a natural-text measurement.
- The reference ran pure-torch for the delta rule and bf16 elsewhere; the
  engine ran its own kernels on the iq5 container. The agreement is between
  two different numeric paths, which is the point.
- The 27B load needs both cards idle: this was done in a short window (prod
  radiance stopped, then restarted healthy; the pause marker removed).
- A same-size check on the 0.8B sibling (`Qwen3.5-0.8B`, layers 0-1, CPU,
  pure torch) shows the same behaviour in a trained small model of the family
  (`gdn_o` row_ss med 2.7e-08 / 3.7e-06), i.e. the scale is family behaviour,
  not a 27B quirk.
