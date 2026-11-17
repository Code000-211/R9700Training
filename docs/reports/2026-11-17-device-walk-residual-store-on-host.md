# Device walk: the residual store off the card, onto one pinned host block

`2026-11-17`, issue #143.

## What the issue says

The `all`-mode device walk aborts partway through the container's 64 blocks with
`iqr dx declined: op gemm_nt_q: the decode's scratch could not be allocated` at tape position 565,
with the step's own VRAM peak at 28.01 GiB of the R9700's 32 GiB, and it is absent from the
earlier `all` runs. The issue calls it what it is: a **headroom regression**, not a missing path.

The walk's VRAM at 4096 tokens is the engine (~20.2 GiB: weights, KV, arena) plus the residual
store — `(65 boundaries + 63 second copies) x 4096 x 5120 x 2` = **5.00 GiB**, one allocation held
for the whole walk — plus each block's per-op scratch, the largest being the int8 decode's weight
plane `N x K` = 5120 x 17408 = 85 MiB. That leaves the last few GiB for the walk's own growth
(snapshots, the head's chunks, the cotangents) and for whatever else the driver holds. The
reporter's run did not have the last few; this one, window 1 below, did — barely.

## Window 1 (the pre-fix build, card free)

`local-radiance-window.sh` window, `R97_MODEL_DIR=/srv/models/rad/qwen3.8-27b-iq5r-s-v4.rad`,
`R97_MODEL_CASES=all`.

- **Counter probe** (`vram_counter_probe`, 2 GiB block on card 0): the KFD available-memory figure
  the `iqr dx declined` check tests — `AMDKFD_IOC_AVAILABLE_MEMORY` — moved the full amount on
  `hipFree`: 32552 -> 30504 -> 32552 MiB, 2048 of 2048 MiB returned. So a refusal at the walk's
  numbers means the VRAM is genuinely held, not a stale counter. (The counter *does* track frees;
  the drift the issue hit is real residency.)
- **`device_step_seq4096`, all mode**: the walk **completed all 64 blocks** — no refusal this
  time. Peak card0 usage **28,676 MiB** (28.0 GiB, the issue's own figure), 3.8 GiB free at the
  peak. It failed later, on descent: `grad norm -nan`, `check failed: descended`.
- **`device_step_tape`, 512 tokens (the light control)**: completed, peak 19,222 MiB; failed the
  same way, `all_finite 0`.

The non-finite cotangent is **not this issue**: the forward is all finite in both runs, and the
walk names the rows (`gdn_chunk_scan`, `gdn_gated_rmsnorm`, `gate_had_quant_i8g`, ...). That is
the tracked device-backward gradient work — #142 (int8 fused producers' cotangent) and #149 (GDN
pre-norm in the epsilon-dominated regime). It is what makes `device_step_seq4096` red on the tree
today, with or without this change.

So the refusal at position 565 is a **marginal** condition: the same 28.01 GiB peak that refused
in the reporter's environment (different container, driver-held state) did not refuse here. What
#143 asks for — "the `all` mode's headroom is tracked rather than assumed" — is the fix below, and
the diagnostic that names the headroom left and its holders when the check does fire.

## The fix

**The store leaves the card.** The device walk's `ResidualStore` now keeps its boundaries in one
**pinned host allocation** (`RAD_MEM_HOST_PINNED`), the same size and the same slot arithmetic as
the card's one: `keep()` and `restore()` are stream-ordered copies on the step's lane, a copy out
after each block and a copy back before each replay. 5 GiB of the card's VRAM is gone from the
walk's peak, and only the per-op scratch competes for the last few GiB beside whatever else the
card holds — which in production is radiance itself.

- **Pinned, not pageable.** A copy to or from pageable host memory stages through the engine's
  8 MiB bounce buffer and drains the stream per chunk (`pageable_copy`,
  `third_party/radiance/core/device/aql.cpp`) -- a host wait at every boundary. Pinned memory is a
  blit packet: ordered on the lane, no drain. (The engine's own mover staging is pinned for the
  same reason.)
- **One block, shared by copies.** The walk copies the store (`const ResidualStore store =
  recorder->store()`) and `LayerReplay` takes its by-value. With per-boundary host vectors, at the
  issue's numbers that is 5 GiB deep-copied twice per step, 15 GiB host peak. The allocation is a
  `shared_ptr` exactly as the card's is, so a copy is O(1) and sees the same bytes.
- **The pre-flight stays.** The card constructor's `plan.fits(budget)` refusal is now mirrored in
  the walk before the store is allocated, with the plan's numbers: an oversized plan is refused
  before the step runs, not in the middle of it. The budget is the host's available memory
  (`/proc/meminfo`'s `MemAvailable`), since that is what the store now fits in.

**The refusal says its number.** `iqr dx declined` now appends the driver's own refusal —
`rad_dev_last_error()`, which carries how much the driver had left to give and the pids holding
the rest — to the message it throws. The issue's log was missing exactly this figure; the next
run that hits the check will print it.

## What it costs

Each step now moves the store over the lane: `plan.store_bytes()` device-to-host during the
forward and host-to-device during the walk — 5 GiB each way at 4096 tokens, stream-ordered, so the
forward's keeps overlap its compute. The card's peak drops by the store; the step's wall time pays
the copies. The store also pins 5 GiB of host RAM — one store at a time, on any call that builds
the recorder (a forward-only call builds it too: it is created before the `grads == nullptr`
return) — beside the engine's own 768 MiB pinned pool, 5.75 GiB pinned in the process, and host
available memory at 41.27 GiB while the walk runs, 47.08 GiB without it. The copies' cost: the
step took 3,080.4 s in window 2, 7.0 s (0.23%) above window 1's 3,073.4 s — the forward's keeps
overlap its compute.

## Verification

- Host tier, the Fix C tree: `ctest -L host` 25/25, the binary's own tally 295 tests, 0 failed.
- `residual_store_host_round_trip` (new, container-free): keep/restore/read of a small plan over
  the lane, the source buffers clobbered, the extra's restore into its own buffer, and a copy of
  the store seeing the kept bytes.
- Window 2, the Fix C build: `device_step_seq4096` all-mode under the VRAM poller. The walk
  completes; the card's peak is 23,544 MiB, down from window 1's 28,676 — the store's 5,120 MiB.
  The 512-token control peaks 18,582 MiB, down from 19,222 — its 640 MiB store. The arithmetic is
  identical between the windows: same loss (34682.089228 -> 50870.029048), same first non-finite
  (`gdn_chunk_scan`, `gdn_k[1]`, position 463, value 1280), same per-op walk counts, same
  zero-gradient leaf, same descent — only the timings differ. The boundary copies are exact.
- Window 3, the rebuilt binary: `replay_layers_card` — the card's own store path after the
  branch-order change: 65 boundaries plus 63 second copies (321,126,400 bytes) in VRAM, every
  layer replayed forward and reverse from them, bit-identical; 1 test, 0 failed. `device_step_tape`
  runs under its lowered 19 GiB declaration, and its arithmetic matches the other windows (loss
  7371.591348 -> 6347.883967, the non-finite at `gdn_gated_rmsnorm`/`gdn_o[0]`/1143, the descent
  -1023.71).

## Container

`/srv/models/rad/qwen3.8-27b-iq5r-s-v4.rad` (qwen3.8-27B, iq5r-s). The issue names no container;
the 27B-class container on this host. All figures above are from it.

## Not this change

The `all`-mode walk still fails on descent with the non-finite cotangent named in window 1 —
#142 and #149's territory, the forward all finite, the rows the walk names. This change removes
5 GiB of the card's peak and names the headroom when the scratch check fires; it does not change
what the backward computes.

The issue names a second cause: the decode scratch is sized for the whole `N x K` weight plane
(85 MiB) rather than the rows in flight. That is `src/adjoint/device/linear.hip`'s iqr-decode
granularity, and a row-wise decode is a rework of the same int8 rows #142's cotangent work
touches. This change does not need it: the store move returns 5 GiB of margin (the Verification
above), so the 85 MiB request fits with room to spare, and no arithmetic changes. The granularity
stays a lever on the decode's own cost, not on the walk's headroom.
