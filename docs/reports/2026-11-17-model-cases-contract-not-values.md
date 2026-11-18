# The model cases assert the contract, not the walk's values

Issue #146 asks for an automated test of the real-container step. That coverage exists on
`main` since 64a437d: `device_step_tape` (512 tokens) and `device_step_seq4096` (4096 tokens) in
`tests/gpu/device_step.cpp`, the `model` tier. The tier was red because the cases asserted
gradient *values* the live-container walk does not produce today -- the walk's cotangents are
non-finite by the rows' own mathematics
([`2026-11-15-device-backward-cotangent-growth.md`](2026-11-15-device-backward-cotangent-growth.md),
#145's verdict). This report records what the cases now assert, what they report, and the numbers
from the window that verified the change.

## What the cases assert

Hard -- issue #146's contract: the `StepFn` runs; the returned `ParamSet` has the params' names
and shapes (`n_grads == n_params`, `names_match`, `shapes_match`, `n_other == 0`); the forward is
finite (`non_finite.forward_all_finite`); the container's hidden/layers match the metadata; the
report fields are consistent (`report_loss`, `report_scored`); the residual store's plan fits its
numbers (`residual_boundaries == layers + 1`, `residual_extras == layers - 1`); the probe ran and
the brackets cover the trunk (`probe_blocks == layers`, `probe_ops > 0`); the head's `rmsnorm`,
`gather_rows` and `logits_gemm_vocab` are on the tape; the optimiser applied exactly one update;
the post-step scored count matches; VRAM peak >= after-load; `tok_s > 0`.

Reported always, asserted when the walk's gradient is finite (`finite || R97_DSTEP_ASSERT_FINITE=1`
-- so the checks re-arm with the walk): `all_finite`; the zero / non-zero leaf split;
`a_changed` / `b_changed`; the post-step forward's finiteness; the descent (`after < loss`).

The 512-token case's row coverage is its own check, not a value check: a device row is an
optimisation and the walk falls back to the host row where `libr97` has none (`allow_host_fallback`
in `src/step_device.cpp:1094`). The ops that took the host path are named always (`no device
adjoint row: ...`), and only an op outside the case's `host_served_op()` list -- today `rope`,
whose imrope rows are the live container's one host-served op -- fails the case. The coverage does
not ride `all_finite`: a walk could be finite with `rope` still host-served, and the design
permits that (`R97_DSTEP_ASSERT_FINITE=1` forces the coverage too, naming even `rope`).

## Two pre-existing failures the gate unmasked

Neither is introduced here; both sit *after* the `all_finite` check in `device_step_case`, which
died first in PR #159's window:

1. The row-coverage probe throws `no device adjoint row for rope` (`mode=imrope, rotary_dim=64,
   sections 11 11 10, n_head_kv=0`, for the 24-head and 4-head tape rows). The engine's own
   `allow_host_fallback` documents the fallback as by design, so the case now names it and
   tolerates it (`host_served_op`).
2. The parent's reporting line read `j.at("grad_norm").get<double>()`, and nlohmann writes a
   non-finite number as `null`: on the live container the walk's gradient norm is NaN, so the
   4096-token case died with `[json.exception.type_error.302] type must be number, but is null`
   instead of reporting. `report_number()` now reads a null as NaN (`loss_after` too), so a
   non-finite value stays a finding for the checks and not a parse error.

## The window

`local-radiance-window.sh` held window, card 0, the container named through `R97_MODEL_DIR`:
`/srv/models/rad/qwen3.8-27b-iq5r-s-v4.rad` (19.6 GiB, qwen3.8-27B, iq5r-s, 64 layers, hidden
5120, 262144 ctx). The build's own container -- `R97_MODEL_RAD` resolves to
`data/qwen3.8-27b-iq4r-s-v4.rad` and the default directory is `/srv/models/qwen3.8-27b-iqr-v4` --
is not on this host, as [`2026-11-13-device-backward.md`](2026-11-13-device-backward.md) records;
the numbers are the iq5r container's.

| run | result | loss | all_finite | first non-finite cotangent | step |
| --- | --- | --- | --- | --- | --- |
| `device_step_tape` (512) | pass | 7371.59135 -> 6347.88397 (-1023.71) | 0 | `gdn_gated_rmsnorm` / `gdn_o[0]` / position 1143 | 322.8 s (fwd 47.3, bwd 272.2, head 0.8) |
| `ctest -L gpu` | pass (1/1) | | | | |
| the two step cases in `gpu_model_tests`' binary + env (`R97_MODEL_CASES=1`, one process) | pass (2/2) | 7371.59135 -> 6347.88397 and 34682.0892 -> 50870.029 | 0 | `gdn_gated_rmsnorm` / `gdn_o[0]` / 1143 and `gdn_chunk_scan` / `gdn_k[1]` / 463 | 325.7 s (fwd 47.7, bwd 275.0, head 0.8) and 3054.3 s (fwd 105.9, bwd 2939.0, head 6.3) |
| `device_step_seq4096` (4096) | pass | 34682.0892 -> 50870.029 (+16187.9) | 0 | `gdn_chunk_scan` / `gdn_k[1]` / position 463 | 3043.3 s (fwd 106.8, bwd 2927.0, head 6.3) |
| `R97_DSTEP_ASSERT_FINITE=1` tape | fail, by design (`check failed: finite`) | 7371.59135 -> 6347.88397 (-1023.71) | 0 | `gdn_gated_rmsnorm` / `gdn_o[0]` / position 1143 | 323.5 s (fwd 47.7, bwd 272.8, head 0.8) |

The four runs in one binary, in that order, in one held window: `TAPE_RC` / `GPU_TIER_RC` /
`SEQ4096_RC` / `TAPE_FORCED_RC`. The model tier's own entry, `gpu_model_tests`, runs **ten**
model cases -- the four `device_forward_*`, the two `device_step_*`, `iqr_decode`,
`replay_layers_card`, `serving_container_prefill_runs_on_the_card` and `vocab_head` (and
`model_tests` is the tier's other entry) -- two of which are 4096-token cases, so a full tier run
is hours of window and eight of the ten are not this PR's. This PR's scoped tier-entry proof is
the two step cases in that entry's own binary and environment (`R97_MODEL_CASES=1`), one process
(`TIER_PAIR_RC=0`); the other eight are untouched by this PR and were not run. The 512 case's
first attempt in that batch refused on the VRAM
guard (`card0: free VRAM 16372 MiB, the test needs 21504 MiB, no holder listed`): a killed
mid-walk run of the *previous* batch still held ~15 GiB of card 0 when it started, and the driver
released it a few minutes later (card 0 was down to 299 MiB at 20:20Z, and the runs after it saw
the card free). It was rerun alone in the same held window, which is the row above.
In the pair run the 4096 case cross-checks its own step's loss against the device forward's
recorded figure for the same batch and container -- 34682.089228 both -- and reports the match,
which ties the step to the forward [`2026-11-12-device-forward.md`](2026-11-12-device-forward.md)
measured.

On the 512-token run the descent did descend (-1023.71) and no leaf's gradient was exactly zero
(`zero-gradient leaves: lora_A none, lora_B none`); the value the checks withhold is `all_finite`.
The forward was all-finite (`forward all finite`), which the contract check now asserts. Both
cases report the same first non-finite site as PR #159's window
(`gdn_gated_rmsnorm` / `gdn_o[0]` / 1143 and `gdn_chunk_scan` / `gdn_k[1]` / 463), so the walk's
behaviour is unchanged by this PR -- only the reading of it.

## Not asserted, by design

- `rope`'s imrope rows have no device adjoint: the walk's coverage names them and the case
  tolerates them (`host_served_op`), because the fallback is by design
  (`src/step_device.cpp`'s `allow_host_fallback`). A device row for them is separate work.
- The walk's cotangent values are non-finite on the live container (#142/#149, PR #157's in-place
  mark). The cases print them and assert them only when the walk is finite, so nothing needs
  deleting or re-adding when that lands.
- The model tier's other eight model cases (`device_forward_*`, `iqr_decode`, `replay_layers_card`,
  `serving_container_prefill_runs_on_the_card`, `vocab_head`): untouched by this PR and not run
  here. `gpu_model_tests` runs all ten in one process, which is hours of window.
