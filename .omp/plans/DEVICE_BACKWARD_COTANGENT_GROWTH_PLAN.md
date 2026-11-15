# Device backward: pin the per-block cotangent growth, guard the f32 quotient, prove both on the 27B tape

Issue: #145. Branch `J-Nova/device-backward-the-cotangent-grows-3.2-decades`, main `6b3df13`
(PR #137 merged). Container `/srv/models/rad/qwen3.8-27b-iq5r-s-v4.rad`, case
`device_step_tape`, 512 tokens.

## Context

The device backward grows the residual `h` cotangent **3.2 decades per block** over 13 blocks
(1.31e-7 -> 2.66e36 per element, 4.27e41 summed), while the forward `h` is flat to +-0.03
decades/block. The rate is type-independent (GDN 3.200, attention 3.193). Within block 62:

| position | where | cot max | factor |
|---|---|---|---|
| 1428 | block add | 9.232e-4 | -- |
| 1421 | FFN norm | 0.013855 | x15.0 |
| 1412 | mixer norm | 0.478516 | x34.5 |
| 1408 | attn norm | 0.980469 | x2.05 |

The first all-non-finite is at 1134; the per-element cotangent peaks at **2.65846e36** (finite,
pos 1148, below f32's 3.4e38). The f32 divide at `src/adjoint/device/quant.hip:199`
(`cot / (float)s`) overflows for any group with `s < cot/3.4e38`; the run's smallest scale is
**5.95831e-5**, and `s < 1` for every group whose amax is under 448, so the divide necessarily
fires before the f64->f32 cast at `:346-350`. The divide is a *site*, not the cause.

The FD says the gradient is wrong: blk63 0.007446 vs 1.857 (249x **small**), blk60 9.437e7 vs
1.372, blk55 -1.045e23 vs 2.644, blk51 6.594e35 vs -17.76. The FD's leaf gradients are flat
across blocks, so the **true** per-block factor is ~1; the device's is ~1600.

The decisive experiment already exists: `R97_HOST_WALK_AB=1`
(`tests/gpu/device_step.cpp:555-617`) walks **the same tape with the f64 host rows**. It
separates the composition (`src/backward.cpp`) from the device rows. Today it **fails at the
first `gemm_nt_q`**:

```
r97: host adjoint: op 'gemm_nt_q': operand 'b' is stored in a plugin-private encoding the host row cannot read
```

(`src/adjoint/host/linear.cpp:116-120`). Before it failed, host and device agreed to **0.2%** on
the top block's rows (add 1.3159e-7 vs 1.31316e-7; lora_expand 1.22004e-6 vs 1.22032e-6), so the
top-block plumbing is not the growth.

Ruled out by the evidence already in the issue and logs:

- **The in-place passthrough as a multiplier on this tape.** The `add`'s write is a plain write
  (its read is a separate operand and the write is not `inout`); the fused norm's opnd 0 is a
  single `inout` operand carrying its own read slot, so the passthrough skips it (the row wrote
  it). The only separate-read + in-place-write pairs are `gate_had_quant_i8g` (~2x) and
  `gdn_gated_rmsnorm` (~1.001x), not 1600x. (The #137 report's "cannot multiply" argument is
  still wrong in general, but the tape does not exercise it.)
- The block-boundary handoff (`pick_boundary`/`first_rows`): block 63's exit equals block 62's
  entry exactly.
- The f64->f32 downcast at `:345-349` as the mechanism (2.65846e36 < 3.4e38), the "ratio exactly
  10.0" amp signal, and the O(1)-cotangent explanation of the harness's blindness (the host never
  makes `cot` f32; the 1e39/1e45 arms are clean by construction).

## The open question and the gate

The growth is a **multiplicative factor of ~1600 per block applied to the whole cotangent
buffer**, in both branch types. Two families remain:

- **(A) the composition** -- `src/backward.cpp`'s `distribute`/`read_segments`/passthrough adds a
  cotangent twice or routes an output cotangent to an input version. The host walk on the same
  tape would grow the same way.
- **(B) the device rows** -- one row's Jacobian is wrong at the walk's magnitudes, shapes, or
  sequence length. The host walk would be flat.

The host A/B decides. It is one weight-decode away.

## Phase 1 -- unblock the host A/B

Goal: the host `gemm_nt_q` / `logits_gemm` rows read the stored IQR weight, so
`R97_HOST_WALK_AB=1` walks the whole tape.

1. **Extract the desc builder.** `tests/host/iqr_decode.cpp:90-186` builds a `R97IqrDecodeDesc`
   from a `RadEncoding` plus the plane pointers. Move it to a host-only header
   `src/iqr_plane_desc.h`:

   ```cpp
   // true: `out` decodes this form; false: a form this build does not implement.
   bool r97_iqr_desc_from_stored(const RadEncoding* enc, const void* joined,
                                 int64_t rows, int64_t cols, R97IqrDecodeDesc* out);
   ```

   `joined` is the operand's `raw` (the `RadPlanesHeader`); the plane pointers come from
   `rad_planes_header(joined)`'s offsets, the same resolution `src/adjoint/device/iqr_plane.cpp`
   does for the device row. The form classification reuses `iqr_plane_supported`/`narrow`'s
   rules, host side.

2. **Decode in the host row.** In `src/adjoint/host/linear.cpp`, replace the `refuse_stored` for
   the `b` operand in `gemm_nt_twin`, `gemm_nt_adjoint`, `gemm_nt_q_twin`, `gemm_nt_q_adjoint`,
   `logits_gemm_twin`, `logits_gemm_adjoint`:

   - `b.raw != nullptr && b.enc != nullptr` and `r97_iqr_desc_from_stored(...)` succeeds:
     allocate `codes[N*K]` int8 and `scale[N*(K/128)]` bf16, call `r97_iqr_decode_host(desc)`
     (`src/iqr_decode.h:353`, already the plain `[N,K]` grid), build the `AdValue`s -- `b` from
     the codes (as doubles, shape `[N,K]`), `bs` from the scale packed `[N, K/128]` -- and
     continue the existing loops unchanged. The existing `scale_at`'s packed branch
     (`n == rows*groups`) then reads the decoded scale.
   - Otherwise `refuse_stored` unchanged.
   - Cache the decoded weight by `b.raw` pointer (one decode per weight; one record per weight
     per step).

3. **Sanity gate.** `R97_TEST_ONLY=iqr_decode` (host, with `R97_LIBQUANT` set to the radiance
   build's `radiance_home/quantizers/libquant.so`) stays 0-failed, and the A/B's first records
   still agree with the device's to ~0.2%.

## Phase 2 -- the gate: measure the host walk

Under a GPU window:

```sh
R97_HOST_WALK_AB=1 R97_DUMP_DC=1 R97_DUMP_QUANT=1 R97_TEST_ONLY=device_step_tape ./r97_gpu_tests
```

Read the host walk's `DUMPDC op=rmsnorm_had_quant_i8g ... buffer=h ... maxabs=...` at the
boundary positions (1458, 1429, 1408, 1400, ...) and compute the per-block rate.

- **Gate A -- the host walk also grows ~3.2 decades/block:** the composition is the defect ->
  Phase 2A.
- **Gate B -- the host walk is flat (<=0.5 decades/block) and finite:** the device rows are the
  defect -> Phase 2B.

The host walk is slow (the test's own comment: hours on the real container). Run it detached
under the window (`timeout 0`, extend the hold past `PAUSE_MAX_HOLD` per the host's window
skill). If it is too slow, add `R97_WALK_BLOCKS=<n>` to `src/backward.cpp`'s reverse loop (stop
after the top `n` block boundaries); the rate is per-block and the top four transitions
distinguish 3.2 from ~0.

### Phase 2A -- if the composition grows

1. **Instrument.** Add `R97_DUMP_VER=1` in `src/backward.cpp`'s `distribute` path: after each
   record, print per written version `(op, position, buffer, version, row_lo..hi, maxabs,
   nonzero)` and the record's incoming `cot` maxabs. The ratio out/in is the record's effective
   Jacobian; the first record whose effective Jacobian is not the row's own `dc`/`cot` ratio is
   the defect.
2. **Fix.** In the `in_place_passthrough` loop (`:716-731`), skip operand `j` when the record has
   a **separate** read operand `k != j` with the same `buffer` and `!rec.operands[k].written`
   whose `dc[k]` is nonzero: the row answered the read, and the write's slot names the same prior
   version, so filling it adds the cotangent twice. Keep the passthrough for records with no such
   read (`lora_expand`, `rope`, the state updates). If a row leaves both slots zero, fill the
   **read's** slot only, never both.
3. **Test.** A host-tier `backward_block` test (`tests/host/backward_compose.cpp`) with two
   records -- an `add`-shaped one (read `x`, plain write `x`) and a fused-norm-shaped one
   (separate read `h`, in-place write `h`) -- a hand-built cotangent, asserting the prior version
   receives the read's `dc` exactly once and equals a hand-computed value; plus a
   `lora_expand`-shaped record (only an in-place write) that still gets its direct path.

### Phase 2B -- if the device rows are the defect

1. **Device FD harness.** Add `tests/gpu/adjoint_device.cpp` (gpu tier) for each row on the
   growth's path -- `gemm_nt_q`, `had_quant_act_i8g`, `rmsnorm_had_quant_i8g`,
   `gated_had_quant_i8g`, `gate_had_quant_i8g`, `gdn_gated_rmsnorm`, `gdn_chunk_scan`,
   `attn_paged`, `lora_shrink`, `lora_expand`. Build each record with the walk's **shapes**
   (M=512, real K/N, real group sizes), call the device row with the cotangent scaled at
   1 / 1e6 / 1e12 / 1e18 (the walk's per-element range), and compare the device `dc` to the host
   twin's f64 `dc`. This is the device analogue of `adjoint_check.h`, and the only instrument
   that can see a device row's magnitude/shape dependence. No `tests/gpu/*.cpp` uses
   `adjoint_check.h` today.
2. **Sequence-length arm.** For `attn_paged` and `gdn_chunk_scan`, run at 512 tokens (not the
   short FD fixture) against the host twin at the same length; a per-token error compounds into
   the 15-35x branch gains.
3. **Candidate rows.** `gemm_nt_q`'s `dx` scale (`src/adjoint/device/linear.hip:648-657`) against
   the host `linear.cpp:216-247`; the quantiser's `1/s` (`quant.hip:199`) against the host
   `quant.cpp`; the sequential rows' state handling. Fix the failing row in its `.hip`, host twin
   as the reference.
4. **Test.** The harness's assertions at all four cotangent scales for the fixed row, plus the
   `device_step_tape` checks in Phase 4.

## Phase 3 -- the f32 quotient guard (independent of the gate)

In `src/adjoint/device/quant.hip`'s `mask_and_rotate` (`:338-413`):

1. Pass the record's `op` and `position` in (two parameters from `quant_adjoint`), so the refusal
   names the record.
2. In the `:346-350` fill loop, after `h_cot[i] = (float)cot.value[i]`: count
   `!std::isfinite(h_cot[i])` (the cast), and compute
   `q = std::fabs(cot.value[i]) / h_scale[row*ngrp + group]` in f64, counting
   `q > (double)FLT_MAX` (the divide), tracking the first index, the worst quotient, and the
   group's scale.
3. Before the launch, if either count is nonzero, throw the `:686-697`-shaped refusal:

   ```
   r97: quant adjoint: op '<op>' at <position>: <n> of <total> elements' cot/scale quotient
   exceeds f32 (largest <worst> at index <i>, group scale <s>); refusing rather than writing Inf
   ```

   and the analogous message for the cast.
4. The kernel's `:199` divide stays (correct while representable); the guard is on the quotient,
   not the incoming cotangent, per the issue's final ordering.
5. Test: `tests/gpu/quant_guard.cpp` -- a quant record with `cot` 1e36 and group scale 1e-3
   (quotient 1e39) throws with that message; with `cot` 1e30 / scale 1e-3 (quotient 1e33) the row
   returns finite.

## Phase 4 -- device step assertions and the report

- `tests/gpu/device_step.cpp`: after the walk, assert the block-boundary `h` cotangent's
  per-block rate is below 1 decade, and that the final gradient is finite and the `R97_CENTRAL`
  FD agrees within 2x at `model.layers.63.`, `.60.`, `.55.`, `.51.`. The existing
  `all_finite` / `n_a_finite_nonzero` / `descended` stay.
- `docs/reports/2026-11-15-device-backward-cotangent-growth.md`: the gate's result (the host
  walk's rate), the pinned record or row, the fix, the before/after numbers, and the issue's
  existing retractions (no re-litigating them).

## Verification

1. Host: `flock /tmp/r97-cpu.lock cmake --build /tmp/r97-issue145-build -j8 --target
   r97_host_tests`; `R97_LIBQUANT=<radiance_home>/quantizers/libquant.so ctest --test-dir
   /tmp/r97-issue145-build -L host` -> 281/0.
2. The host A/B (the gate): the Phase 2 command; record the host walk's per-block rate.
   Expected 3.2 decades (A) or <=0.5 (B).
3. After the fix, the same `device_step_tape` with
   `R97_DUMP_FWD=2 R97_DUMP_DC=1 R97_DUMP_QUANT=1`: `all_finite` true; `n_a_finite_nonzero` the
   full finite-nonzero set (the failure had 82 of 400); the boundary `h` maxabs < 1e3 over 13
   blocks; per-block rate < 1 decade.
4. The FD: `R97_CENTRAL=1
   R97_CENTRAL_BLKS=model.layers.63.,model.layers.60.,model.layers.55.,model.layers.51.` -> the
   walk within 2x of the central difference at every block (the failure: 249x, 6.9e7x, 4e22x,
   3.7e34x).
5. The guard: `ctest -L gpu -R quant_guard` passes; a deliberately tiny-scale fixture throws the
   named message instead of producing `Inf`.

## Files

- `src/adjoint/host/linear.cpp` -- the stored-weight decode (Phase 1).
- `src/iqr_plane_desc.h` (new) -- the `RadEncoding` -> desc builder, extracted from
  `tests/host/iqr_decode.cpp`.
- `src/adjoint/device/quant.hip` -- the quotient guard (Phase 3).
- `src/backward.cpp` -- `R97_WALK_BLOCKS`, `R97_DUMP_VER`, the passthrough rule (Phase 2A,
  conditional).
- `src/adjoint/device/linear.hip` (or the failing row's `.hip`) -- the row fix (Phase 2B,
  conditional).
- `tests/gpu/adjoint_device.cpp` (new), `tests/gpu/quant_guard.cpp` (new),
  `tests/gpu/device_step.cpp` (assertions), `tests/host/backward_compose.cpp` (new, Phase 2A).
- `docs/reports/2026-11-15-device-backward-cotangent-growth.md` (new).

## Risks and assumptions

- The host A/B is the gate: flat -> the fix is in the device rows and the harness is the first
  PR; growing -> the fix is in the composition.
- The host decode is exact: `iqr_decode` proves it against `libquant`, and the A/B's top-block
  agreement is the live check.
- The host walk is slow; `R97_WALK_BLOCKS` is the mitigation and the top four blocks measure the
  rate.
- The f32 guard is independent and lands regardless.
- No submodule edits, no contract-header changes; the new test files register by glob; no
  `tools/commands.h` change.

## Out of scope

- The 249x-small top-block gradient if the gate shows the host walk flat and the top block's
  rows correct -- a separate seed/head issue, to be filed separately.
- #142 (the int8 fused producers' dropped bf16-output cotangent) -- known adjacent defect, not
  the growth.
