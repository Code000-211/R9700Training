# Device backward: the walk is finite at 512 and 4096 with the right sign; the magnitude is still wrong at `gemm_nt_q`

`2026-11-18`, issue #55.

## What the issue asks

The device backward as a `StepFn` on one R9700: one step over the real container, the adapter's
gradient finite on every leaf, and one AdamW step that lowers the loss. The join is #137; #146
made the model cases assert the contract and report the value checks; this report is the
gradient itself.

## The state on current main before this branch

`device_step_tape` (512 tokens): `all_finite 0`, first non-finite `gdn_gated_rmsnorm`/`gdn_o[0]`
at tape position **1143** (11-17), 398/400 A leaves and 399/400 B leaves finite.
`device_step_seq4096`: `gdn_chunk_scan`/`gdn_k[1]` at 463, step 3054 s. #145's verdict was that
the cotangent growth is the checkpoint's mathematics; #149 settled the GDN pre-norm's ~1e-3 as
the checkpoint's property. The walk's in-place reads were served from the op's own output (#148),
and #157 put the in-place mark on the tape and restored the 63 reads with no earlier writer from
the block's stored boundary.

## What changed here

1. **The device rows' in-place handling.** `kv_store` now fills only the pool rows the store
   does not overwrite (mirroring `host/kv_store.cpp`); `lora_expand` fills its in-place slot with
   the write's cotangent. A new `tests/gpu/adjoint_in_place.cpp` holds both at their aliased and
   plain-write records against the host row, and `tests/support/device_check.h`'s `make_call` now
   places twin outputs at their schema positions, which an aliased record needs.
2. **A fills signal in the registry.** The walk's in-place passthrough replaced a row's all-zero
   slot with the whole incoming cotangent, which cannot be told from "the row left it empty". An
   adjoint row now registers a `FillsFn` (`R97_REGISTER_ADJOINT_FF`), and the walk skips the
   passthrough for a slot the row owns. `kv_store` and `lora_expand` register on both sides.
3. **A readable-host range fallback.** A device row's kernels take their operands in f32 and
   write the cotangent back through the record's dtype; a record handed a cotangent past f32's
   range hands on Inf. `src/backward.cpp` now falls back to the op's host row (the same
   arithmetic in f64) when the incoming cotangent exceeds 1e30 and every forward operand is
   host-readable — a plugin-private weight (`raw != nullptr`) keeps the device row, because the
   host row refuses it by name.
4. **`clip_impl`'s norm is double.** `const float norm = float(std::sqrt(sq))` became Inf for the
   7.5e38 gradient, the scale became 1/Inf = 0, and a clip zeroed the whole gradient. The compare
   and the scale are double now; the scale is a representable f32 subnormal.
5. **The descent probes.** The test prints one AdamW step at the default 1e-3, unclipped 3e-4 and
   1e-4, and clipped (norm 1) 1e-3 and 1e-4; the value check is the direction at a step the
   gradient's scale admits, and the default's overshoot is reported. The changed-leaf checks hold
   that both factors move and the gradient reaches every leaf, not that every leaf moves: the
   unclipped AdamW's f32 `v = (1-b2)g^2` is Inf above |g| ~1.8e19, so the dominant leaves take no
   update at the walk's scale.
6. **The slice instrument's forward** (`src/slice_reference.h`, the f64 reference that is the
   magnitude arbiter). Five defects, each measured against the device's own dumped values: the
   dump and both program writers carry the operand views (`view_shape`/`view_stride`/`view_off`,
   the buffer's own pitch) and both parsers gather through them; the cold `kv:` rule covers a
   plain write as well as an in-place read (the walk zeroes a written `kv:` operand's snapshot
   when the block has not written it, `src/step_device.cpp:1590-1594`); the dump's first-read
   rule is mirrored, so `SNAPDIFF` compares like-for-like; and the fused norm's residual input is
   the post-update sum `bf16(x + residual_old)`. The block-37 forward now agrees to bf16 noise.
7. **`tests/fixtures/programs/oracle.json`** regenerated through its own
   `R97_RECORD_ORACLE_FIXTURE` gate: the program writers now emit the view fields and
   `reference_step_program_matches_the_live_trace_without_kernels` compares the live recording
   field-exactly, so the fixture and the writer must land together. The diff is the 8
   view-carrying operands and nothing else.


## The 512-token measurement

`device_step_tape`, one held `local-radiance-window.sh` window per run, prod restored by `close`.

| | 11-17 (pre-#157) | this branch |
|---|---|---|
| `all_finite` | 0 | **1** |
| finite leaves | 398/400 A, 399/400 B | **400/400 A, 400/400 B** |
| first non-finite | `gdn_gated_rmsnorm`/`gdn_o[0]`@1143 | **none** |
| in-place restores | — | **63 of 63** |
| grad norm | nan | 7.63e38 |
| step | 325.7 s | 342.4 s (fwd 47.0, bwd 292.3, head 0.8) |
| VRAM peak | 19,222 MiB | 19,489 MiB |

The walk's cotangent is finite on every op (`non-finite: cotangent none | per op: (none)`); the
forward is finite. The first non-finite before the range fallback was at position 9, and
`R97_DUMP_POS=9` printed the incoming cotangent's `maxabs = 4.23e34` — above the 1e30 threshold,
so the host row served it and the walk became finite.

The descent, same run (`descent_lrs` in the results JSON):

| lr | clip | loss after | delta |
|---|---|---|---|
| 1e-3 | 0 | 7403.47027 | +31.88 |
| 3e-4 | 0 | 7289.34732 | −82.24 |
| 1e-4 | 0 | 7317.57839 | −54.01 |
| 1e-3 | 1 | 7221.76164 | −149.83 |
| 1e-4 | 1 | 7299.87113 | −71.72 |

The direction is right at every step the gradient's scale admits; the unclipped default 1e-3
overshoots. **The magnitude is not.** The same-tree FD (`R97_CENTRAL`, the bf16-sized eps) at the
two ends of the stack:

| leaf | walk g (mean-loss) | central FD (sum-loss) | FD / 511 | walk vs true |
|---|---|---|---|---|
| `model.layers.63.self_attn.q_proj.lora_A` | 0.00744629 | 948.917 | 1.857 | **250x too small** |
| `model.layers.0.linear_attn.in_proj_qkv.lora_A` | -6.66657e35 | -74358.9 | -145.5 | **4.6e33x too large** |

The last layer is already wrong — 250x too small, above the in-place reads #157 fixed — and the
error compounds downward, crossing 1 and reaching 4.6e33x at the bottom. The true leaf gradients
are O(1)..O(150) mean-loss; the walk's ~1e38 norm is the compounding, not the checkpoint's
Jacobian. The clipped probes are a **first-order estimate**, not direction-only: `clip_impl` clips
the gradient before the moments (`src/optim.cpp:264`) and AdamW's `mh/sqrt(vh)` cancels the clip's
scale (`:289-293`), so a dominant leaf's step is ~`lr*sign(g)`. The FD run's clipped 1e-3 loss
change is −175.09, i.e. `sum(g_true*sign(g_walk)) = 1.75e5` sum-loss (342 mean over 511); the FD's
two leaves — block 0 74,358.9 and block 63 948.9 — are 43% of it. The 1e-4 clipped probe gives
4.39e5 (859 mean); the 1e-3 step is past the linear regime, so the two bracket the nonlinearity,
and both are O(1e2)..O(1e3) mean-loss, not 1e38. The *unclipped* probes are the direction-only
ones: the dominant leaves' f32 `v` is Inf above |g| ~1.8e19, so they move only the moderate
leaves. The two runs differ — the `R97_CENTRAL` run's walk norm 7.92e38 against the gate run's
7.63e38, its default-loss after 7400.96052 against 7403.47027, its clipped probes −175.09/−43.93
against −149.83/−71.72 — consistent with the f32 accumulation's order dependence at the walk's
~1e38 scale, while the FD's central differences agree to the digit (948.917 and −74358.9 in both).
#55 stays open on the magnitude; the section below carries what the error is — a uniform ~x3.7 per
block gain, not a top-block offset — and the candidate site.

## The 4096-token ticket case

`device_step_seq4096` (zero-B start), same window harness, re-run on the fixed tree: step
**3465.7 s** (fwd 105.6, bwd 3350.6, head 6.4), loss 34,682.089 -> **19,154.462** (mean NLL 8.469 ->
4.678), `all_finite 1`, `non-finite: cotangent none | forward all finite | per op: (none)`. The
case **passes**. It was non-finite before the residual-operand and boundary fixes -- `all_finite 0`,
first non-finite `gemm_nt_q`/`gdn_o.q8`[0] at position **690**, and the loss moved *up* to 50,870.

**The magnitude does not.** `grad norm 1.25528e+36` -- the #145 growth artifact at 4096's larger
cotangent -- and the descent ladder says so directly: lr=1e-3 gives -15,527.6 and lr=3e-4 gives
-12,840.1, but lr=1e-4 gives **+2,913.2** and clipped lr=1e-4 gives **+1,437.9**. A gradient at 1e36
is not a direction the loss can follow at any step; a smaller step going *up* while a larger one goes
down is what a meaningless magnitude looks like. This is the 4096 face of the open magnitude
question, not a new defect.

The `gemm_nt_q` row's range remains the recorded piece for the *overflow* -- an f64 carry or a
per-row scale for the activation cotangent and the staged weight gradient -- but the overflow no
longer fires on this case, so that work is now about the magnitude rather than about finiteness.

## The 4096 leg: the walk's own leaves, and a cotangent that arrives already amplified

One held window (`/tmp/r97-55-window38.sh`, 4303.19 s, `rc=0`, `pause` verified absent after):

```sh
env R97_MODEL_DIR=/srv/models/rad/qwen3.8-27b-iq5r-s-v4.rad R97_TEST_ONLY=device_step_seq4096 \
    R97_DUMP_LEAF=layers.37. R97_DUMP_BLOCK=37 \
    R97_DUMP_BLOCK_DIR=/tmp/r97-55-dump-block37-4096 ctest -V -R gpu_model_tests
```

The step: loss 34,682.089 -> 19,091.980, `all_finite 1`, grad norm 1.25519e+36, 4095 scored,
3442.9 s (fwd 106.3, bwd 3326.7, head 6.6); the ladder descends at 1e-3 (-15,590.1) and at the
clipped 1e-3 (-25,137.1) and rises at 1e-4 and below, the same shape as the ticket case above.

The dump: 40 constants (7.16 GB of JSON), the boundary 805 MB ([4096, 5120] f64, two buffers),
the seeds 817 MB, the program 38 KB, 8 constants refused -- the plugin-private weights, the same
set as 512. The instrument's memory at this size: nlohmann's DOM is ~3x the file transiently, then
~2.9 GB of f64 live, and the run peaks near 29 GB of host RAM.

**The walk's own block-37 leaves at 4096** (`R97_DUMP_LEAF`):

| leaf | walk l2 |
|---|---|
| `linear_attn.in_proj_qkv.lora_B` | 1.04067e+15 |
| `linear_attn.in_proj_z.lora_B` | 7.50543e+14 |
| `linear_attn.out_proj.lora_B` | 5.52932e+14 |
| `mlp.gate_proj.lora_B` | 5.34199e+14 |
| `mlp.up_proj.lora_B` | 7.58150e+14 |
| `mlp.down_proj.lora_B` | 3.46554e+14 |
| all six `.lora_A` | **exactly 0** |
| `LEAFSET layers.37.` | **1.71239755e+15** |

**The zero A leaves are structural, not a defect**: `device_step_seq4096` runs the container's
seed-0 start, where `lora_B == 0` (`tests/gpu/device_step.cpp`, `if (probe) seed_adapter(params)`
-- only the 512 case seeds both factors), so no cotangent reaches A's factor. It also means the
512 and 4096 walks are in **different parameter regimes** (512: seeded, both factors nonzero,
leaf-set norm 5.16; 4096: zero-B start), so a ratio between those two norms mixes regimes and is
not a token-count statement. The valid comparison is within each run -- slice against walk on the
same parameters -- and a token-count claim would need a matched control (a 512 dump at the seed-0
start, or a seeded 4096) that has not been run.

**The block-boundary cotangent is already amplified.** The cotangent the walk hands block 37 is
the seeds file -- the cotangent at the block's two output buffers, written by the head and the
blocks above it:

| | 512 | 4096 |
|---|---|---|
| seeds, L1 | 0.13915 | **2.2508e+16** |
| seeds, L2 | 7.90011e-05 | 4.88804e+12 |
| seeds, max | 2.90842e-07 | 3.42722e+10 |

At 4096 block 37 receives a cotangent whose largest value is 3.4e10 and whose L1 is 2.25e16; its
own leaves come out at 1.71e15, i.e. the block's transfer is *below* its input, not above it.
**The amplification is upstream of block 37** -- in the blocks above it or the head -- and the
1.7e15 leaf set is inherited, not produced here.

Do not confuse this with the host re-quantisation line, which prints a different quantity:
`DUMPQUANT side=host op=gated_had_quant_i8g pos=17` gives `codes_cot=7.5478e+13` (512: 0.841852)
for the cotangent at op 17's *codes slot*, one step inside the block's last op, after the
quantiser's own codebook step. It is not the boundary cotangent and it is not a block-level
measurement.

**The 4096 slice clears block 37's rows** (window 39's second arm, `rc=0`, 22:24; `pause` absent
after). Its forward matches the device's own values to bf16 noise on every honest line --
`h.q8` 0.00516, `gdn_q` 0.00389, `gdn_k` 0.00424, `gdn_v` 0.00400, `gdn_g` 0.00027, `gdn_beta`
0.00075, `gdn_kkt` 0.00195, `gdn_o.q8` 0.01189, `gate_up` 0.00686, `ffn.q8` 0.01417, the
adapter weights exactly 0 -- the same floor as 512. The lines that are not at that floor are the
documented pollution: `gdn_o`/`h` (the dump captured them post-op) and `kv:2`/`kv:3` (the dump's
outgoing state against the walk's cold zero).

Its leaves, against the walk's, grouped the way the 512 table groups them:

| weight | modules in the slice's leaf | walk l2 | slice l2 | slice / walk |
|---|---|---|---|---|
| `ssm_out` | one (`out_proj`) | 5.52932e+14 | 5.40199e+14 | **0.977** |
| `ffn_down` | one (`down_proj`) | 3.46554e+14 | 3.26531e+14 | **0.942** |
| `ssm_inz` | two (`in_proj_qkv` + `in_proj_z`) | 1.28309e+15 | 1.43157e+15 | 1.116 |
| `ffn_gate_up` | two (`gate_proj` + `up_proj`) | 9.27448e+14 | 1.22987e+15 | 1.326 |
| all six | A | 0 | 0 | -- |
| `LEAFSET layers.37.` | | 1.71239755e+15 | 1.99007734e+15 | **1.162** |

Read it the way the 512 table has to be read. The two single-module weights are like-for-like and
land in the same 2-6% band as 512's (`ssm_out` 1.049, `ffn_down` 1.041 there; here 0.977 and
0.942): the block's rows agree with the f64 reference to a few percent. The two two-module groups are not a pure row comparison -- the
slice's `ssm_inz`/`ffn_gate_up` leaves are single dense-fused factors (`[N_total, R_total]`) while
the walk's are per-module diagonal blocks (the same accounting difference the 512 table records at
1.5-1.7), so 1.116/1.326 mix the rows with that structure. The leaf-set ratio 1.162 is the
summary, and element-wise agreement is not established by it.

**The magnitude is inherited.** The seeds file hands block 37 a cotangent whose largest value is
3.4e10 and whose L1 is 2.25e16 (L2 4.888e12); whatever the block's rows do with it, that scale was
created above the block. The block hands the device walk's own leaves back at 1.162 of their norm
and the f64 slice reproduces them, so the rows are not where the amplification enters. Together
with the 512 result (1.0475) this clears block 37's rows at both token counts, each within its own
parameter regime, and moves the hunt to the blocks above 37 or the head. The next measurement is
the same dump-and-slice at a higher block (the seeds at block 38's output are what block 37
received, so block 38 is where to look first).

That arm runs in the same window, second, from the same dump:

```sh
./build/r97train slice-serve \
    --program   /tmp/r97-55-dump-block37-4096/layers.37.json \
    --boundary  /tmp/r97-55-dump-block37-4096/layers.37-boundary.json \
    --constants /tmp/r97-55-dump-block37-4096/layers.37-constants.json \
    --seeds     /tmp/r97-55-dump-block37-4096/layers.37-seeds.json \
    --model /srv/models/rad/qwen3.8-27b-iq5r-s-v4.rad --prefix model.layers.37. \
    --out /tmp/r97-55-serve-out-4096.txt
```

`/tmp/r97-55-compare-block.py --block 37` groups both sides by weight (`ssm_inz` = `in_proj_qkv` +
`in_proj_z`, `ffn_gate_up` = `gate_proj` + `up_proj`) and prints the per-weight ratios and the
leaf-set norms, against the walk's lines in `/tmp/r97-55-walk4096-leaves.txt`.

### Block 38: the boundary gain is 1.09, so the amplification is above it too

Window 40's dump of block 38 at 4096 (`/tmp/r97-55-dump-block38-4096/`, the same shape: 40
constants 7.14 GB, boundary 811 MB, seeds 816 MB) gives both sides of block 38's boundary, and
this comparison *is* like-for-like (L1 of a boundary cotangent against L1 of the next one):

| boundary | L1 | L2 | max |
|---|---|---|---|
| 39 -> 38 (block 38's seeds) | 2.05964e+16 | 4.50211e+12 | 2.2019e+10 |
| 38 -> 37 (block 37's seeds) | 2.2508e+16 | 4.88804e+12 | 3.42722e+10 |
| **block 38's own gain** | **1.093** | **1.086** | 0.643 |

So block 38 hands down very nearly what it receives, and **the growth is above block 38**.

What the 1.09 gains have to explain is a gap *within this run*: the boundary the head hands the
top block (the first line `print_boundary_scale` prints, below) against 2.25e16 at 38 -> 37. If
every block above 38 gained 1.09, the head's boundary would have to be ~2.06e16 / 1.09^25 ~ 2.5e15
for the chain to close; the head's own printed value settles whether that is so or whether one
block carries the jump. The 512 and 4096 boundary cotangents are **not** the comparison here --
those runs differ in both token count and adapter regime (see above), so their ratio is not a
quantity any per-block factor has to reproduce.

Block 37's like-for-like boundary gain is computable the same way once window 41 prints the
37 -> 36 line, which also says whether ~1.09 holds below 38 -- nothing so far makes 38 special
beyond being the second block measured.

### The amplification is the full-attention blocks: ~70x each, every fourth block

Window 42's per-block boundary line (`print_boundary_scale`) over the top of the 4096 walk, head
down. The container is Qwen3.8-27B with `full_attention_interval 4` (`docs/ARCHITECTURES.md`):
layers 3, 7, ..., 63 are the 16 gated-attention layers, the other 48 the gated delta net.

| boundary | L1 | L2 | max | L1 gain |
|---|---|---|---|---|
| 64 -> 63 (the head's hand-off) | 12.4045 | 0.00441766 | 1.24574e-05 | -- |
| 63 -> 62 (**full attention**) | 963.78 | 0.216353 | 0.000579399 | **77.7** |
| 62 -> 61 (delta net) | 1335.05 | 0.34031 | 0.0043915 | 1.385 |
| 61 -> 60 (delta net) | 1472.57 | 0.385137 | 0.00471636 | 1.103 |
| 60 -> 59 (delta net) | 1601.22 | 0.423023 | 0.00532786 | 1.087 |
| 59 -> 58 (**full attention**) | 113815 | 24.456 | 0.0555127 | **71.1** |
| 58 -> 57 (delta net) | 118535 | 25.2687 | 0.0583533 | 1.041 |
| 57 -> 56 (delta net) | 123534 | 26.2228 | 0.0708623 | 1.042 |
| 56 -> 55 (delta net) | 132522 | 30.1493 | 0.352937 | 1.073 |
| 55 -> 54 (**full attention**) | 8.17566e+06 | 1660.79 | 2.36872 | **61.7** |
| 54 -> 53 (delta net) | 8.37739e+06 | 1707.35 | 8.08446 | 1.025 |
| 53 -> 52 (delta net) | 8.72841e+06 | 1825.24 | 15.8091 | 1.042 |

The pattern is exact, not statistical: **every full-attention block amplifies its incoming
cotangent by ~60-80x in both L1 and L2, and the delta-net blocks pass it through at ~1.04-1.10**.
The L1 and L2 columns agree, so this is not a sign-cancelling structure hiding inside an L1
summary. Three jumps in eleven blocks, at 63, 59 and 55 -- the layers `layer % 4 == 3`.

This is where the 1e36 comes from, and it is **not** a uniform per-block factor: it is one kind of
block, repeated sixteen times. The next question is whether that gain is the model's own Jacobian
at 4096 tokens or the attention row's defect, and the 512 arms in the same window answer it: the
same lines at 512 (both regimes) put the same boundaries against a 512-token chain.

### The matched control: 512 zero-B against 4096 zero-B

`device_step_tape_zerob` runs 512 tokens at the same seed-0 start the 4096 ticket runs, so the
two are the same adapter regime and the only difference is the token count. The batch is
`1 + (i*7) % 299` (`tests/gpu/device_step.cpp:366`), so the 512 sequence is a strict prefix of the
4096 one and per-position difficulty is stationary.

| case | loss | mean NLL | grad norm | step |
|---|---|---|---|---|
| 512 zero-B | 3585.32 | 7.016 | **8.26632e+21** | 318.4 s |
| 4096 zero-B | 34682.09 | 8.469 | **1.27995e+36** | 3393.5 s |

The forward moves by 1.45 nats for eight times the context, while the gradient's norm moves by
**1.5e14**. The forward is therefore roughly stationary with length and the amplification is the
walk's: per full-attention block the boundary gain is ~7-25 at 512 and ~39-326 at 4096, i.e.
roughly proportional to the token count, and sixteen of those give the n^16-ish growth in the
norm. That is the shape to explain -- and the per-record dump (`R97_DUMP_POS=all`) names the op
inside block 63 that does it.

### The amplification is the zero-B start, not the token count

Three runs, one container, one build, the same 512 tokens; only the adapter's state differs
(`seed_adapter` seeds both factors for the 512 ticket, the 4096 ticket and `device_step_tape_zerob`
start where init leaves them, `lora_B` exactly zero):

| boundary | 512 seeded | 512 zero-B |
|---|---|---|
| 64 -> 63 (the head's hand-off) | 0.0284541 | 6.52524 |
| 63 -> 62 | 0.0586117 (x2.06) | 43.7711 (**x6.71**) |
| 62 -> 61 | 0.060787 (x1.04) | 52.712 (x1.20) |
| 59 -> 58 | 0.0694519 (x1.05) | 1519.45 (**x24.5**) |
| 55 -> 54 | 0.0821795 (x1.03) | 14785.5 (**x9.0**) |
| 51 -> 50 | -- (flat) | 106905 (**x7.1**) |
| 37 -> 36 | **0.13915** (the block-37 dump's seeds) | **6.34723e+09** |

The seeded chain is flat: every boundary passes its cotangent through at 1.03-1.05, the top block
at 2.06, and block 37's boundary is 0.139. The zero-B chain starts 229x higher at the head and
multiplies by 7-35 at each full-attention block, reaching 6.3e9 at the same boundary -- **a factor
of 4.6e10 between two runs that differ only in `lora_B == 0`**.

That is the whole defect, and it also explains why the 4096 norm is 1.28e36: the 4096 ticket runs
the zero-B start too. The per-attention-block gain grows with the token count (~7-25 at 512,
~39-326 at 4096) and sixteen of them carry the norm from O(1) to 1e21-1e36.

`lora_B == 0` makes `lora_expand`'s `dt = sum_n cot y * scale * B` exactly zero, so the adapter
contributes no term to the hidden state's cotangent -- the zero-B walk is the trunk's path alone.
That is not by itself a mechanism: removing a term from a sum cannot multiply the sum by 1e10. What
differs between the two runs is therefore the **forward's activations** (the seeded adapter
perturbs every block's output), and the cotangent's conditioning against them -- the block's
boundary is a difference of terms ~1e4 larger than itself (block 63's seeded intermediates reach
2.5e-3 while its boundary max is 1.2e-7), which is the regime in which the device's f32 arithmetic
and the host's f64 diverge. The one-knob control -- `device_step_tape_zerob` with
`R97_SEED_ADAPTER=1`, the same case, tokens and probe setting -- separates the two.

### The seeded chain is length-invariant; the zero-B chain is not

`R97_SEED_ADAPTER=1` seeds the 4096 ticket's adapter (a debug knob, gated on the truncation flag),
so the two axes -- token count and adapter state -- are separable. With the adapter seeded, the
4096 chain is **the same chain as the 512 one**:

| boundary | 512 seeded | 4096 seeded |
|---|---|---|
| 64 -> 63 | 0.0284541 | 0.0284606 |
| 63 -> 62 | 0.0586117 (x2.06) | 0.0586442 (x2.06) |
| 62 -> 61 | 0.060787 (x1.04) | 0.0607953 (x1.04) |
| 61 -> 60 | 0.0635817 (x1.05) | 0.0635051 (x1.04) |
| 60 -> 59 | 0.0662389 (x1.04) | 0.0661487 (x1.04) |

Four to five digits on every line, at eight times the tokens. **The backward is length-invariant
when the adapter is seeded** -- the rows handle 4096 correctly, and the attention-block amplification
does not exist in that regime at either length.

The zero-B chain is the one that amplifies, at both lengths (x6.7-35 at 512, x39-326 at 4096), so
the defect lives in the zero-B start's own cotangent, not in the token count and not in the rows'
length handling. `device_step_tape_zerob` with `R97_SEED_ADAPTER=1` is the one-knob control (the
same case, the same tokens, the same probe setting, the adapter seeded) that isolates it.

### The host rows (f64) give the same zero-B chain as the device (f32)

The device walk's cotangents are f32; the host walk's are f64. At 512 zero-B over the top two
blocks they agree:

| boundary | device (f32) | host (f64) |
|---|---|---|
| 64 -> 63 | 6.52524 | 6.52526 |
| 63 -> 62 | 43.7711 (x6.708) | 43.738 (x6.703) |
| 62 -> 61 | 52.712 (x1.204) | 52.6677 (x1.204) |

So at 512 zero-B the amplification is not the device's precision: the f64 host rows give the same
chain. (The host arm was run at zero-B 512 only -- there is no host arm for the seeded regime, so
this clears precision, not the rows: the device kernels are translations of the same host rows, and
both walks share the join, the head's seeds and `src/backward.cpp`.)

Where it is *created* is the head's junction, before any block row runs: the branch cotangents into
the head's residual add are the same size in both regimes (`gather_rows` h_out 1.712e-4 zero-B
versus 1.860e-4 seeded), while the add's own output -- the 64 -> 63 boundary -- is 4.053e-5 versus
1.313e-7. The boundary is a difference of terms ~1e4 larger than itself, and the seeded start sits
much closer to the cancellation than the zero-B one. That difference then rides down the chain.

### Real text: the same split, and the shape of a power iteration

`R97_STEP_CHAT=<jsonl>` steps a real conversation instead of the synthetic cycle (the container's
own tokeniser, `tests/fixtures/`-style JSONL). On a 308-token prose conversation the base model
scores **mean NLL 2.563** -- in distribution -- and the two regimes separate exactly as before:

| case | mean NLL | grad norm |
|---|---|---|
| prose, zero-B | 2.563 | **2.35124e+14** |
| prose, seeded | 14.559 | **1001.86** |

So the synthetic 299-id cycle is not the driver: it makes the top blocks worse (x6.7 against prose's
x2.6) but the zero-B amplification is there on in-distribution text too, at 1e14. The chain's shape
is the informative part:

| boundary | prose zero-B | prose seeded |
|---|---|---|
| 64 -> 63 | 7.01671 | 0.025778 |
| 63 -> 62 | 18.3806 (x2.62) | 0.0532303 (x2.07) |
| 51 -> 50 | 811.232 (**x7.49**) | flat (x1.03-1.05) |
| 47 -> 46 | 14200.5 (**x11.81**) | flat |
| 31 -> 30 | 7.12126e+07 (**x10.57**) | flat |
| 15 -> 14 | 4.62652e+12 (**x8.78**) | flat |
| 3 -> 2 | 2.50267e+15 (**x4.52**) | 3.29762 (x1.48) |

Every full-attention block jumps, the delta-net blocks pass through at 1.0-1.2, and **the jump
grows with depth** (x2.6 at the top, x7-12 in the middle, x4.5 at the bottom). That is the shape of
a **power iteration**: the walk applies each block's transposed Jacobian to the cotangent, the
cotangent's direction converges onto that block's dominant singular vector, and the per-boundary
gain converges to the block's largest singular value. The zero-B start's seed has a component
along those directions; the seeded start's seed -- which its own head attenuates by ~x270, the RMS
norm's projection removing a cotangent nearly parallel to its input -- does not, and its chain stays
at 1.04.

Under that reading the numbers are the model's own backward conditioning, not a row's error: the
device f32 rows and the host f64 rows agree to four to five digits on both regimes, the seeded
chain is identical at 512 and 4096 tokens, and the gains are a smooth function of the activations.
The practical consequence is the one a training step already has an answer for: **clip** -- the
optimiser's `clip_norm` is what a per-block gain of 2-10 over sixty-four blocks needs, and the
clipped 4096 step is the one that descends (-25,137 at 1e-3 in the 4096 ladder).

### The ladder cannot arbitrate the magnitude, in either regime

The prose arms ran the full descent ladder. AdamW's first step with bias correction on and decay 0
is `lr * g / (|g| + eps)` = `lr * sign(g)` per leaf, so the first-order `dL = -lr * sum|g|` -- which
looks like a magnitude arbiter. It is not one here, and the two regimes show why, in opposite
directions:

| lr | zero-B dL/lr (norm 2.35e14) | seeded dL/lr (norm 1.00e3) |
|---|---|---|
| 1e-3 | 5.4e+04 | 2.7e+05 |
| 3e-4 | 3.2e+05 | 1.7e+06 |
| 1e-4 | 5.6e+05 | 4.8e+06 |
| 1e-5 | 2.8e+05 | 3.1e+07 |
| 1e-6 | 6.2e+05 | 1.4e+08 |
| 1e-7 | 1.3e+06 | 3.3e+08 |

Neither plateaus (both grow as lr falls). The **seeded** arm's `dL/lr` reaches 3.3e8 while its
gradient's L1 can be at most its norm, 1e3 -- a first-order violation of five orders in the
*impossible* direction. The cause is the bf16 parameters: the loss is piecewise constant, and a
step that crosses a rounding boundary moves it by far more than the linear term. The **zero-B**
arm's `dL/lr` (1.3e6) sits eight orders *below* its norm, which the same piecewise constancy
explains from the other side (most leaves never move).

So the ladder measures the parameters' rounding, not the gradient's magnitude. The ceiling
`|dL| <= lr * L1` is one-sided (an undershoot is allowed), so the zero-B arm cannot falsify its own
norm; the seeded arm is the calibration, and its `|dL|` at 3e-4 (517) overshoots its own first-order
ceiling (`lr * L1 <= 3e-4 * sqrt(800) * 1000.15 = 8.5`) by 61x. Even applying that 61x calibration,
the zero-B deficit stands: at 3e-4 its claimed norm predicts `|dL| >= 7e10` and it measures 94.8,
so the magnitudes are off by at least 1e7. **Criterion two is not met at the zero-B start.** What it does say is that the *signs* are right (every unclipped probe
descends at every lr) and that the clipped 1e-3 step -- which moves every leaf by ~lr regardless of
its gradient -- **raises** the loss (+149.2 zero-B, -272.1 seeded: the seeded one descends), so the
zero-B gradient's *relative* leaf magnitudes do not support a uniform step.

**Criterion two therefore stays open, with the magnitude question untouched by the ladder.** The
next arbiter is a formula audit of the rows the amplification passes through, against
`docs/adjoints/attention.md` and the host rows, on the block whose gain is largest.

### The trigger is the activations, not the exact zero

`lora_B == 0` makes `lora_expand`'s `dt` exactly zero, so a plausible trigger was a cotangent of
exactly zero being handled differently from a tiny one. `R97_SEED_ADAPTER=1e-8` (a B small enough
to leave the forward's loss at the zero-B value) refutes it:

| boundary | prose zero-B | prose, B = 1e-8 |
|---|---|---|
| 64 -> 63 | 7.01671 | 7.01671 |
| 63 -> 62 | 18.3806 | 18.3805 |

Identical to five digits -- which is also what the geometry predicts (at B = 1e-8, `dt` is ~1e-8 of
the cotangent, far below any row's contribution). So this arm refutes exactly one thing, the
**exact-zero-slot** family (a zero cotangent mistaken for an empty slot in the join's fills or the
in-place path), and nothing more.

The dose arms do not carry the claim I first read into them. `seed_adapter`
(tests/gpu/device_step.cpp:316-325) overwrites BOTH factors with `mul`-scaled values, and the leaves'
geometry makes the norm scale with the seed: `dL/dB ~ A.x` and `dL/dA ~ B`, so a seed 1e4 larger
gives leaves ~1e4 larger. The measured norms are 5.30e7 (mul 1e-8) and 4.95e11 (mul 1e-4) -- a
ratio of 9.4e3 against a mul ratio of 1e4, textbook leaf geometry. The boundary chain, meanwhile,
is invariant: bottom L1 4.45e15 (zero-B), 4.37e15 (1e-8), 4.08e15 (1e-4). **The Jacobian did not
move**, so the dose arms neither convict nor clear the magnitude; what they do show is that the
amplification is a property of the sane forward, present at every seed from 0 to 1e-4, and absent
only in the wrecked seeded-1.0 forward. That reinforces the activation-regime reading: the chain is
flat only when the adapter has wrecked the forward (NLL 14.56 against 2.56). Criterion two is
therefore as open as it was before the dose arms, and the open question is exactly the x2-2.7 per
block at the sane forward.

### The arbiter's tooling, and what its self-check does and does not say

The block-local directional-derivative arbiter needs the slice's twin to run a block's forward, and
the block-47 slice now does: the dump tightens its records (`own_raws`, `tighten_rows`) and rewrites
`M`/`q_len` to the batch's own token count, carries the batch ids, and the runner derives the
attention raw tables (positions, block table, seqused, cu_seqlens) from the operands' own rows.
`slice-serve --out-values` writes the buffers the finite difference needs.

The base run's `CONSTDIFF` lines compare the twin's final environment against each buffer's
first-read capture, so they are not all twin error: `h` (1.07) is the block's own residual update,
`x` (0.21) is the caller's normed copy overwritten in place at op 0, and `lora.t.r16.lane0` (1.02)
is one projection's `t` against another's. The one unclassified outlier is `attn_k` at 0.22, and it
is K-specific: `attn_q`, sharing the same derived positions through the same rope twin, agrees at
0.005, and the rest of the block sits at quantisation noise (`attn_v` 0.005, `kv:1` 0.0047,
`gate_up` 0.0067, `attn_out` 0.0077). So the suspect is the rope's `imrope`/`sections` twin over
the `[308, 1024]` four-head layout, not the row counts: dump53 and dump54 give bit-identical
`CONSTDIFF` across `M` 512 -> 308, because the twins derive their rows from the operand values.
A truncated program splits it exactly, and the split is a capture-timing ladder, not a twin error:
through the K-norm (ops 0..11) `attn_k` agrees at 0.0045; adding the two rope ops (0..13) moves it
to 0.224. The capture for `attn_k` is the K-norm record's own write, so each cut measures the op in
between -- cut 9's 0.341 is the norm's rescale against the pre-norm capture (`SNAPDIFF attn_k`
0.340910749 is the same number), cut 11's 0.0045 is post-norm against post-norm, and cut 13's 0.224
is post-rope against the pre-rope capture, the rotation's own footprint on 64 of 256 lanes. The
like-for-like pair that settles it is `kv:1`: `kv_store` copies the twin's POST-rope K into the
pool, and the pool matches the device at 0.0047 -- a 22% rope error would force at least 0.12
there. So every twin is verified correct wherever a like-for-like pair exists, `attn_k` is the
fourth capture-timing artifact beside `h`/`x`/`lora.t.r16`, and the finite difference's
precondition is met.
The finite difference differences the twin against itself and the K twin is verified by `kv:1`, so
the precondition is met.

### The first independent measurement: the identity holds at the finest step, and the secant is the limit

With the twins verified where a like-for-like pair exists, the block-local directional derivative
runs. For block 47, `u = g_out(h)/|g_out(h)|` from the block-48 seeds, `g_in` from the block-47
seeds, and `J u` measured by perturbing the slice's boundary `h` by `eps * u` and differencing the
`--out-values` runs. The identity `<g_in, u> = <g_out, J u>` holds for any `u` when `g_in = J^T
g_out`:

| eps (fraction of `|x|`) | `<g_out, J u>` | ratio to `<g_in, u>` = 2.05172 |
|---|---|---|
| 0.00025 | 2.3011 | **1.1215** |
| 0.0005 | 1.2486 | 0.6086 |
| 0.001 | 0.7925 | 0.3863 |
| 0.0025 | 0.6104 | 0.2975 |
| 0.005 | 0.5971 | 0.2910 |
| 0.01 | 0.5926 | 0.2888 |
| 0.02 | 0.5924 | 0.2887 |

The ratio **rises** as `eps` shrinks (0.29 at the largest step, 1.12 at the finest) -- it approaches
1 from above -- so the adjoint's identity holds to about 12% at the finest step, and the small
ratios at larger `eps` are the secant averaging over the bf16/int8 staircase, not an adjoint error.
The method lesson: an inner-product plateau without a norm plateau does not certify a directional
derivative on a piecewise-affine forward (`|J u|` collapses 5.15 -> 0.90 over the same window while
`<g_out, J u>` looks flat), so the acceptance bar is both plateaus. A first reading of the four large-`eps` points
alone suggested a 3.45x violation; the small-`eps` points refute it. The residual 12% is within what
this instrument can resolve (the composition of the perturbation is the slice's own: `h` moves and
the `x` buffer is held, while the model derives `x` from `h`; that difference is of this order).
The composed path closes the gap. The walk's boundary cotangent hands the *same* vector to `h` and
to `x` (`max|g_h - g_x| = 0.0`), so a perturbation of `h` in the model's own function moves `x`
through the norm's Jacobian, and the composed lhs is `<g_in, u + J_norm u>`. With the gain from the
dump's `blk.47.attn_norm.weight` (`x_i = (h_i/rms)(1+w_i)`, eps 1e-6), that term is 7.9239, and the
composed ratio is `(rhs_hold + 7.9239)/(2.0517 + 7.9239)`:

| eps (fraction of `|x|`) | composed rhs | ratio |
|---|---|---|
| 0.00025 | 10.225 | **1.0250** |
| 0.0005 | 9.1725 | 0.9195 |
| 0.001 | 8.7164 | 0.8738 |
| 0.0025 | 8.5343 | 0.8555 |
| 0.02 | 8.5163 | 0.8537 |

**The identity holds to 2.5% at the finest step on the composed path**, and the residual decline
with eps is the same staircase. So block 47's adjoint passes this instrument at f64 resolution
along this direction: the x2-2.7 per block is not a per-block adjoint error of this size, and the
remaining candidates are the join (the block-to-block boundary bookkeeping), the truncated walk's
zero-fill below block 0, or a many-direction effect. The composed ratio is dominated by the 7.92
term, so its discriminating power is in the *difference* from 1.025, not the value itself.

### The dead boundary buffer: a per-block double count candidate

The composed-path experiment turned up the first concrete candidate for the x2 gain. In blocks 46,
47 and 48 the fused producer's normed copy `x` is **written by op 0 and read by nobody** (the
program's reads name `h`, `h.q8`, `h.s8` and the projections' buffers, never `x`), yet the walk's
boundary cotangents carry an `x` entry that is **exactly the residual's** (`max|g_h - g_x| = 0.0`
over 1.58M values). Perturbing the boundary along the composed path -- `x` dropped, so the twin
derives it -- changes the forward response by nothing at the finest step (`rhs` 2.3011, identical to
the hold-`x` value), which confirms `x` is dead: it cannot contribute to the block's output.

**It is a per-block double count, and the walk's own boundary line proves it.** The line
`r97: boundary 47 -> 46: entries=2 n=3153920 l1=14197.9 l2=10.6004` sums two entries, and the dump's
`layers.47-seeds.json` gives `l1(g_h) = l1(g_x) = 7102.289` and `l2 = 7.49899` for each: `14197.9 =
2 x 7102.289` and `10.6004 = sqrt(2) x 7.49899`, the sum of two *identical* vectors. So the walk
hands the residual's cotangent down twice at every block boundary. **That the block below folds
both into the residual's gradient is not yet proven, and two of the walk's own numbers bound it:**
the gated-delta boundaries carry the same two entries (`25 -> 24: entries=2 n=3153920`) yet their
per-block gains are 1.02, and since the chain's gain ratios are taken on already-doubled L1s, a
uniform duplicate cancels out of them. Even if all sixteen full-attention blocks folded the
duplicate, that is 2^16 = 6.6e4 -- taking the zero-B norm from 2.35e14 to 3.6e9, still more than
1e2 above the ladder's between-regime estimate. So the duplicate is a contributing factor at most, not the cause of criterion two.

**The decisive check overturns the duplicate reading: the two entries are the same cotangent
because that is the row's rule, and the boundary line's x2 is the sum of two identical, correct
entries.** The seeds' `h` and `x` entries are elementwise identical (1.5M values, `max|g_h - g_x| =
0.0`) while the boundary *values* differ (`max|diff| = 253.2`), so the residual and the accumulator
are different buffers that the norm's backward gives the same cotangent. Block 47's slice routing
matrix confirms it is the row, not the join: seeds h-only -> hand-down `h = x = 265.307`, seeds
x-only -> `h = x = 619.541`, both -> `h = x = 626.862` -- **identical on both slots under every
seed**, which is `adjoint_rmsnorm_residual`'s `add` rule (`dv + g_r` on each summand), pinned there
at rel 1e-5 against a finite difference of the same decomposition. So the boundary line's `l1 =
14197.9 = 2 x 7102.289` is two identical entries summed, **not a doubled gradient: the per-block
gain is 1.33x, not 2.6x.** The gain chain is still the defect, but the duplicate is not its
mechanism and no fix to the norm's row or the join is warranted by this evidence.

**The first same-block, same-seed comparison separates the walk's backward from the f64
reference, and the walk over-amplifies by 11.3x.** Feeding block 47's slice backward the walk's own
boundary-48 cotangent (`l1 = 600.796`) gives a 47 -> 46 hand-down of `l1 = 626.862` -- a block gain
of **1.043x** in f64. The walk's own chain over the same boundary pair is `600.796 -> 7098.95` --
**11.81x**, and the boundary lines' x7-12 per-block gains are the same number. So the per-block
amplification is not the model's Jacobian: the f64 reference on the same block and seeds is flat,
and the device walk's block backward multiplies the cotangent by ~11x where f64 multiplies it by
~1.04x. Compounded over the depth, that is the 2.35e14 norm.

**The per-op profile names the row: the fused norm's backward (`rmsnorm_had_quant_i8g`, op 0)
amplifies on the walk and is flat in f64.** The f64 slice's op 0 receives `l1 = 613.055` and hands
down `626.862` -- a row gain of **1.023x**. The walk's op 0 at block 47 (512-synthetic cycle,
`R97_DUMP_COT` on the gpu model test) receives `106982` and the boundary below it carries
`1.695e6`/entry -- **~16x through that one row**, and the same row at block 63 gains only 2.2x. So
the divergence is inside the norm's fused adjoint (host `src/adjoint/host/quant.cpp`'s
`fused_adjoint` or the device `plugin.hip`/`quant.hip`), not in the block's other rows: the
slice's profile from op 27 down is smooth and its hand-down is 1.02x. Note the two chains are
different regimes and must not be conflated: the 11.3x and the 1.078x block-46 figure come from
dump53's 308-token chat chain, the op-0 16x from the 512-synthetic cycle; the row's gain grows
with M, which is itself the M-scaling cross-check the report records.

**The host arm reproduces the device walk exactly, so the 11.8x is in the walk's shared setup,
not the device row.** `R97_HOST_WALK_AB=2` on the same 308-chat walk (blocks 63..47) gives boundary
47 -> 46 `l1 = 14169.8` against the device walk's `14197.9` -- 0.2% apart -- with the same block
gain (1198.9 -> 14169.8 = 11.8x) and the same chain shape (flat through 52 -> 51, then 7.5x at
51 -> 50, then 11.8x at 47 -> 46). Both arms run the same `backward_block`; the slice runs it too
and gives 1.043x on the same block and seeds. So the divergence is in what the walk feeds the
block, not in a row: the slice seeds only the boundary-48 cotangent, the walk also runs its
`enter` (the residual store's credit) and the objective's seeds. That is the next measurement:
dump the walk's block-47 `enter` input and the slice's seeds and diff them.

**The per-op diff exists now and names the op: the QK-norm/rope pair (ops 13 and 11), 42.4x.**
Extracting block 47's cot lines from the host-arm log between `boundary 48 -> 47` and
`boundary 47 -> 46` and putting them against the slice's file:

| op | name | slice l1 | walk l1 | ratio |
|---|---|---|---|---|
| 27 | lora_expand | 600.796 | 599.435 | 1.00 |
| 24 | gated_had_quant_i8g | 6.909 | 6.894 | 1.00 |
| 20 | rmsnorm_had_quant_i8g | 606.376 | 604.996 | 1.00 |
| 16 | gate_had_quant_i8g | 2.633 | 2.627 | 1.00 |
| 15 | attn_paged | 173.398 | 172.882 | 1.00 |
| 14 | kv_store | 87.231 | 86.985 | 1.00 |
| **13** | **rope** | **44.840** | **1899.71** | **42.4x** |
| 12 | rope | 69.807 | 69.541 | 1.00 |
| **11** | **rmsnorm** | **44.845** | **1899.71** | **42.4x** |
| 10 | rmsnorm | 69.715 | 69.541 | 1.00 |
| **9** | **lora_expand** | **42.391** | **1555.88** | **36.7x** |
| 7 | gemm_nt_q | 42.391 | 1555.88 | 36.7x |
| **6** | **lora_expand** | **41.712** | **1996.63** | **47.9x** |
| 4 | gemm_nt_q | 41.712 | 1996.63 | 47.9x |
| 3 | lora_expand | 128.436 | 128.046 | 1.00 |
| 1 | gemm_nt_q | 128.436 | 128.046 | 1.00 |
| 0 | rmsnorm_had_quant_i8g | 613.055 | 776.733 | 1.27 |

So the fused norm's row is nearly right (1.00 at op 20, 1.27 at op 0), and the divergence is the
**QK-norm and rope pair** -- ops 13 and 11, 42.4x each -- which then propagates into the qkv
LoRAs' expansions below them (ops 9, 6). That is where the fix goes, and it is a row or a pair of
rows, not the walk's setup. The two chains' op 13/11 values are identical in the slice (44.84 for
both) and identical in the walk (1899.71 for both), so the pair shares one input: the walk's
`kv_store`/`rope` path carries a cotangent 42x the slice's into the same two rows.

The rope's row itself is a correct inverse rotation (`rope_adjoint_f64`, src/adjoint/host/qk.cpp:
the transpose `[[c, -s], [s, c]]`, norm-preserving), so the row is not manufacturing the 42x. One
caveat on the table: the per-op line sums only the operands the record *writes* (the dump skips a
read operand's version), so `rope`'s 1899.71 is the post-rope `attn_k` version's cotangent, the
same version the `kv_store` reads. The next measurement is that reconciliation: dump the post-rope
`attn_k` version's cotangent at each reader and compare it with the attention's output.

**Two candidate mechanisms were checked against the records and cleared.** The device row's
`mult` (`param_num(call.rec, "mult", 1.0)`, src/adjoint/device/quant.hip) is 1.0 -- op 0's params
on the real container are `M=308, eps, group=128, n=5120, wadd=1` with no `mult` -- so the device
row's outer `mult` is not a scale the host lacks. And the host A/B does not separate the row from
the walk's bookkeeping: both arms run the same `src/backward.cpp` rows, so their 0.2% agreement
confirms the rows are backend-independent, which was never in doubt. What the slice's 1.023x
versus the walk's 16x does establish is that the row plus its *inputs* differ, and the slice's
inputs are the dump's boundary values with its own f64 twins for the intermediates. The next
measurement is therefore the intermediates: the walk's `attn_k`/`attn_q` snapshots against the
slice's twins' own outputs (`--out-values`), with the quantiser's codebook edges counted in each.

**The rows either side of the divergence are also clean.** `rope_adjoint_f64` is the transpose
rotation and `kv_store_adjoint_f64` (src/adjoint/host/qk.cpp) is a pure scatter -- `d_k[dst+d] +=
d_pool[base+d]`, gain one, with the pool's two lanes split back into k and v. So neither the rope
nor the kv_store manufactures the 42x, and the k path's cotangent arrives inflated before either
row runs. **The post-row dump brackets the mint between op 14 and op 13, and corrects the earlier reading
of `h.q8`.** A row never mutates its own *write* version -- it writes into the versions its
read_segments name -- so the cotout line's `v=` is the input's version, and `h.q8`'s 169.28 is a
downstream carrier of the already-inflated k path, not the mint. The bracket the data does
support: op 14's kv_store pre-row is 86.98 in the walk against the slice's 87.23 (the whole pool
version, k and v lanes), and op 13's rope pre-row is 1899.71 against 44.84 -- measured before any
qkv-gemm row has run, since the walk runs 27 down to 0. The kv_store's scatter is gain <= 1 and
the slot table is the same on both sides, so a second writer feeds the post-rope `attn_k`
version (or the walk's plan resolves op 13's read to a different version). The prime suspects are
the `in_place_passthrough`/FillsFn bookkeeping on that chain (a correct all-zero fill is
indistinguishable from an empty slot, and passthrough firing where the row owns the slot adds the
cotangent twice) and a crossed version id between op 11's and op 13's writes of `attn_k`. The
version ids the cotout line now prints (`v=`) settle which: the slice resolves op 13's write to
v=3, op 11's to v=2, op 12's to v=1, and the walk's plan is the comparison.

**The mint is found, and it is the kv slot table: every block folds the pool's *covered* rows by
exactly the token count.** The walk's own `dc` lines settle it, block by block, once the pool's
cotangent is split into the rows the store overwrote and the rows it left alone (`dc[3]` is the
uncovered remainder):

| block | pool l1 | uncovered | **covered** | dk+dv | ÷ covered |
|---|---|---|---|---|---|
| 63 | 1.8312 | 1.8051 | 0.02604 | 8.021 | **308.02** |
| 61 | 3.4394 | 3.3559 | 0.08358 | 25.74 | **308.00** |
| 59 | 7.0349 | 6.9556 | 0.07932 | 24.43 | **307.99** |
| 57 | 10.718 | 9.9695 | 0.74827 | 230.5 | **308.02** |
| 47 | 86.985 | 75.765 | 11.22 | 3455.6 | **308.00** |

308 is the token count to two decimals in every block. A gain-<=1 scatter over distinct slots
cannot fold by 308; a slot table that is all one value folds by exactly tokens. The cause is the
walk's *served* slot operand: `snapshot()` deliberately returns zeros for an unnamed raw operand
("the raw slot's zero is a valid slot index" -- true for a decode's one slot, fatal at 308), so
every token scatters from pool row 0. The slice fabricates `raw:2` as the identity table
(`src/slice_reference.h`: `v.value[i] = i`), which is the reference step's own formula
(`src/reference_step.cpp: block*kKvBlock + p%kKvBlock` for a one-sequence batch from block 0), and
gets 44.84. **The fix, and its verification.** The walk rebuilds the table from the batch's block size,
at the point `snap_record` serves the operand and ahead of the frozen-sharing branch (the first
block to read it freezes the value every later block binds, so a rebuild placed after that branch
only ever sees the shared zeroed copy -- the first attempt there did exactly nothing, and the walk
reproduced 14169.8 unchanged). It replaces only a serve with no usable table (empty or all one
value), leaves a tape-named table alone, and refuses loudly for a multi-sequence batch whose
per-sequence block tables do not share a layout.

Verified on the same 308-token chat walk: the kv_store row's fold is gone in **every** block --
`(dk+dv)/pool = 1.0000` at blocks 63, 61, 59, 57 and 47, where the pre-fix values were 4.38, 7.48,
3.47, 21.50 and 39.73 -- and the boundary chain collapses:

| boundary | pre-fix | post-fix | ratio |
|---|---|---|---|
| 63 -> 62 | 18.358 | 13.558 | 1.35x |
| 55 -> 54 | 87.856 | 21.966 | 4.00x |
| 51 -> 50 | 809.14 | 29.230 | 27.7x |
| 48 -> 47 | 1198.87 | 37.531 | 31.9x |
| **47 -> 46** | **14169.8** | **39.571** | **358x** |

The step-change at 51 -> 50 and the cliff at 47 -> 46 are both gone; the chain is monotone and
smooth (7.0, 13.6, 14.7, 15.5, 16.7, 18.1, 19.0, 19.9, 20.9, 22.0, 23.5, 25.2, 26.8, 29.2, 32.1,
34.5, 37.5, 39.6). Host tier 303/303 with the fix in. A loud refusal in the kv_store rows was
tried and **removed**: `step_host_int8`'s tape legitimately carries a constant slot table on one
record, and the guard failed two host tests (A/B: 2 failures with it, 303/303 without).

**The slice's own version resolution (with `v=` and `rd=` now printed) fixes the graph the mint
must fit.** Block 47's attn_k has four versions: v0 the qkv gemm (op 4), v1 the LoRA expand
(op 6), v2 the k-norm (op 11), v3 the rope (op 13). The slice resolves exactly that -- op 13
writes v3 reading v2 (`v=3 rd=2`), op 6 writes v1 reading v0, op 12 writes v1 reading v0. So v3's
only contributor is the kv_store (op 14), whose attn_k read maps to v3; and v3's cotangent is
therefore the scatter's `dk` alone. Slice: 44.84, the k-lane half of the pool's 87.23 -- which is
what a gain-<=1 scatter must emit (308*4*256 = 315,392 values from a 1,048,576-value pool whose
whole cotangent is 87). Walk: 1899.71 on the same 315,392 values. Both the record and the slot
geometry are identical between the dump and the walk (`op 14: block_size 16, head_dim 256,
n_head_kv 4`; op 13 declares `n_head_kv` 0 and only passes its width check because
`(4 + 0) * 256 = 1024` happens to equal the k tensor's width). Since the row, its input and the
plan agree, the remaining possibility is that the walk's *other* reader of v3 -- the rope's own
in-place read -- contributes to it a second time, or that `nv`/heads in the walk's live record
differ from the dumped one; `R97_DUMP_DC` on `kv_store` (now in the filter) plus the walk's
`v=`/`rd=` lines settle it in the one running window.

The row's inputs and formula match the host twin on inspection; the A/B above is what settled
that the row is not the site. `rmsnorm_back_kernel` (src/adjoint/device/quant.hip) takes
the narrowed sum's snapshot as `x` when the record has a residual (`res_at >= 0 ? call.fwd[res_at]
: call.fwd[0]`, and the comment says so), and its body is the host twin's `dx[i] += d[i]*scale*gain;
dx[i] -= drs*rs3*r[i]/n` term for term (`shared = mult*rs^3*t/n`). What is not yet tested is whether
the *host arm* of the same walk gives the slice's 1.02x at op 0: `R97_HOST_WALK_AB=2` with
`R97_DUMP_COT` on the same block separates "the device row is the defect" (host arm 1.02x) from
"the walk's op-0 handling is" (host arm also 16x). One window, the same test, and it is the
measurement the fix should wait on.

**The routing matrix also says the accumulator's cotangent reaches the gradient, and the forward
says it should not.** With block 47's slice backward seeded h-only, x-only and both: hand-down
`265.307`, `619.541`, `626.862` -- both is 2.36x h-only, so the x slot's cotangent routes into the
block's input cotangent; the leaves move 87.57 -> 80.50 -> 112.59. Yet the native composed run
(without `x`) reproduces the forward's `rhs` exactly, so the forward does not read `x`. A backward
that credits a forward-unread input is the defect, confirmed at one block. The next measurement is
inside the block: which op's row multiplies by ~11x on the device but not in the f64 slice (the arbiter's
finest-eps ratio of 1.12 says the block's Jacobian is right in the *directional-derivative*
sense at that point, so the suspect is a row whose contribution is small for the probe's direction
but large for the walk's actual cotangent -- the fused norm's `dv + g_r` split and the
quantised paths' straight-through terms are the first two). The fix has two candidate sites, and the second is the one the evidence points at:

1. `boundary_cotangents` (src/join_common.h) not handing down a buffer the block below only
   writes. The broad form of this -- skip any write-never-read entry -- **breaks
   `step_host_int8`'s whole-model cosine (0.8)** against the bf16 container, so the rule must be
   narrow: the accumulator that is written in place by the block's own fused producer AND read by
   nothing else. Note the scope of "read": `x` is read by nothing *in the block's op list*, but the
   residual store's `enter` does credit it at the boundary ("the branch's entry the mlp's write",
   src/join_common.h:378), so whether the fold is credited there or by the block's own backward is
   the first thing to instrument (`add_input_full`, src/backward.cpp).
2. The walk's `add_input_full` (src/backward.cpp) minting an input cotangent for op 0's in-place
   write in the first place. `add_input_full` accumulates (`at[k] += src[k]`), so a cotangent that
   enters twice sums -- which is right for two *readers* of one buffer and wrong for one buffer
   that is only written.

The verification is one device run: the per-block boundary gains should fall toward ~1 (the top
block's x2.6 -> ~1.3) and the gradient norm should fall by roughly the product of the removed
factors (2.4e14 -> a scale the ladder can test). `step_host_oracle` is the CPU gate for the rule,
because its reference is independent of the walk.

### Where the backward time goes

The results JSON carries `row_seconds` and `pull_seconds` per op (`src/step_device.cpp:1906-1912`);
this is the first report that reads them.

| | 512 | 4096 |
|---|---|---|
| `backward_seconds` | 267.6 | 3326.7 |
| `row_seconds`, all ops | 56.6 (21.2%) | 728.5 (21.9%) |
| `pull_seconds`, all ops | 5.9 (2.2%) | 78.7 (2.4%) |
| **neither** | **205.1 (76.6%)** | **2519.5 (75.7%)** |

The proportions are the same at both token counts, so the cost is not a token-count effect. At
4096 the largest rows are `attn_paged` (182.1 s over 16 calls), `lora_expand` (116.6 s / 288),
`gated_had_quant_i8g` (103.2 s / 64), `lora_shrink` (85.6 s / 288), `rmsnorm_had_quant_i8g`
(81.1 s / 128), `gdn_chunk_scan` (72.7 s / 48) and `gemm_nt_q` (69.3 s / 288); the largest pulls
are `gemm_nt_q` (22.0 s) and `lora_expand` (18.2 s).

**Three quarters of the backward is neither the rows nor the per-op pull**, and that is the
answer to "the rows or the pull-back?" — neither. The remaining time is in the walk's own
bookkeeping (the snapshot store, the tape's contents, the cotangent copies and the join), which
has no counter yet; a per-phase timer there is the next step for the cost question, not a row
optimisation.

### Two instruments added from this review

- **One line per block boundary** (`src/step_device.cpp`, `print_boundary_scale`): `r97: boundary
  <from> -> <to>: entries= n= l1= l2= max=` for the head and after every block, always on. It is
  the cheapest way to see whether the growth is one jump or a uniform per-block factor, and it
  needs no dump, no slice and no extra window beyond the walk itself.
- **`device_step_tape_zerob`** (`tests/gpu/device_step.cpp`): 512 tokens at the same seed-0 start
  the 4096 ticket runs (`lora_B == 0`), which is the matched control the 512/4096 regime
  difference needs. The 512 case above seeds both factors; this one does not.


## The per-block gain, against the full device walk

`R97_WALK_BLOCKS` limits **the host walk only** (`src/step_device.cpp:1220-1231`, deliberate: a
truncated walk zero-fills what it skipped, which would silence the very A/B it exists for). A device
arm run with `R97_WALK_BLOCKS=14` therefore still walks all 64 blocks, and its `grad norm 8.10e38`
is a full-walk norm — it is **not** comparable with a host 14-block `6.25e8`. Reading those two as an
A/B is a scope error, not a device/host disagreement.

What *is* comparable is the cotangent table at the positions both arms reach. The host 14-block
(blocks 50..63) and the device full walk agree at all 28 of those positions, within 1.25-1.45x — the
device slightly the larger, as its bf16 operands predict:

| pos | block | host | device |
|---|---|---|---|
| 1449 | 63 ffn | 7.25985e-06 | 7.24196e-06 |
| 1429 | 63 attn | 2.62647e-06 | 2.563e-06 |
| 1366 | 60 attn | 4.81279e-05 | 4.98295e-05 |
| 1358 | 59 ffn | 2.0516e-03 | 2.2583e-03 |
| 1148 | 50 ffn | 180.998 | 262 |
| 1135 | 50 attn | 40.267 | 72.5 |

The device walk continues below block 50 with no host counterpart: 1127 (49 ffn) 1344, 1106 (48
ffn) 9792, 1023 9.37e04, 924 5.58e08, 812 5.03e11, 693 6.33e14, 560 2.83e18, 448 1.78e21, 357
2.98e23, 238 1.56e27, 126 3.69e30, **14 (block 0 ffn) 7.05e33**. Head to foot is 40.7 decades over
64 blocks: the walk multiplies the `h` cotangent by **~4.3 per block** (median per-block ratio
4.2; 60->59 is x7.3, 50->49 x2.8).

**The true cotangent is flat.** The FD ladder puts the true leaf gradients at O(1)..O(150)
*mean-loss* across all 64 blocks, so the true per-block gain is ~x1.0, and the walk's ~x4.3 is the
whole error: block 63 too small by 249x and block 0 too large by 4.6e33 is a *pure* per-block gain
error of (4.6e33 x 249)^(1/63) = **x3.7 per block**, with no separate top-block offset needed. The
walk's measured x4.0-4.3 per block is the same number to within the FD's two-point estimate.

The 11-15 report's signature survives: the residual `x` path stays flat at 1.31e-07 while the `h`
path grows. `h` crosses exactly **two** fused norms per block (its `ffn_norm` and `attn_norm`), so a
factor of two at each is a x4/block gain on the `h` path alone, and the in-place `h` fill is the
candidate. The fused norms are **not** amplifying in their own right: `DUMPNORM`'s `dy_sum` against
`DUMPQUANT`'s `quant_dx + dx` is 1.0 within 3% at every record (1429: 181.144 vs 181.57; 1408:
391.66 vs 387.92; 1400: 485.412 vs 498.63; 1387: 950.038 vs 940.05), so the rows are unit-gain.

That measurement does **not** clear the walk, which is why the in-place `h` fill stays the candidate:
a double-count at the walk level would hand each norm an already-doubled cotangent, so `dy_sum`
inflates with it and the ratio above stays ~1.0. The sites to separate are the row's own write to
the `h` slot beside the passthrough fill, the record's two `h` slots being summed instead of one
being taken, and the distribution of the write's cotangent to the replaced version.

A `gdn_gated_rmsnorm` instance amplifies x1.5e4 in one step (pos 1416 leaves `dc[gdn_o] = 3.14662`
against the `2.09146e-04` handed to it at 1417), but that is not the per-block gain: the `dc[gdn_o]`
sequence across gated norms climbs x2-8 per block (3.15, 6.50, 23.4, 299, 925, 7304, 6.28e4,
1.88e5, 4.48e5, 1.52e7), the same x3.8/block as the `h` path, so something inside each block
compensates it by ~x1e4. The row's internal `dc[gdn_o]/dc[gdn_in]` is ~1800-3000 ~= `sc` at every
instance.

### The gain is the container's weight, and the norm is the only row family that disagrees

Four readings close every element-level candidate on the fused norm's gain:

- **The operand is rank 1.** `DUMPDC` prints `buffer=blk.63.ffn_norm.weight shape=[5120]` for all 28
  fused-norm records, so the record declares the weight itself, not a `[rows, n]` per-row tensor.
  `R97_GAIN_ROW=1` is therefore inert (`FDRMS ... w_rank=1 w0=5120 per_row=0`), and "no collapse"
  in that A/B says nothing about the kernel. The `elementwise.cpp:387-390` comment describing a
  `[512,5120]` arrival is stale symptom text of the #157 mis-taken-intermediate bug that
  `quant.cpp:1376-1403` fixed.
- **Its values are the container's**, read offline by `tools/dump_weight.cpp` (`rad::RadFile`, mmap,
  no engine, no GPU): `blk.63.ffn_norm.weight` `0.2451171875, 0.146484375, 0.2099609375,
  0.2138671875`, `blk.63.attn_norm.weight` `0.07568359375, 0.171875, 0.11328125, 0.080078125`,
  `output_norm.weight` `0.9609375, 0.75, 0.9375, 0.828125` -- all `scheme=plain`, bf16, one plane.
  The gain the walk took is the same to the bf16 digit (`GAINPOS pos=1449 ... vals=[0.245117,
  0.146484, 0.209961, 0.213867]`, and the same line for the twin and the adjoint).
- **Gain-after-rotation is closed.** `y2_rel` tracks `deq_rel` at every norm (1449: 1.362 vs 1.364;
  1379: 1.479 vs 1.490), and at the gates `y2_rel` is 1e4 while `deq_rel` is 0.004, so the shape test
  does not discriminate.
- **The twin's elementwise output is not the kernel's value.** `deq_vs_write` of 3e-4 / 1e-5 proves
  `dq` -- reconstructed from the kernel's own codes and scale -- *is* the value the kernel stored,
  and the twin's `y` differs from that same tensor by 1.36 at every fused norm, while all 14
  `gated_had_quant_i8g` and 6 `gate_had_quant_i8g` records match at 0.0034-0.0078. The difference is
  not a per-row scale (`row_ratio` 0.988-1.056) and sorting does not remove it (`sort_rel` 0.33-0.52,
  against 0.0037-0.0045 at the gates), so it is not a permutation either.

With `w` correct and `s = 1/rms(x)` derived from `x`, `y = x*s*(w+1)`: a *scaled* wrong `x` cannot
explain the difference, because `s` cancels the scale. What is left is the `h` snapshot's **vector**
or a structural difference in the kernel's norm.

### The fused norm's `residual` operand: the row differentiates the wrong vector

`rmsnorm_had_quant_i8g`'s operands are `x`, `residual`:inout?, `w`w -> `q`, `scale`, `out_bf16?`
(`third_party/radiance/docs/OPS.md:739`), and `third_party/radiance/libref/ref_quant.cpp:149-206` --
the fused op's own reference implementation, "what a plugin asks for first" -- defines the forward:

```cpp
const float v = res ? ldt(x, xb + i*xs) + ldt(res, rb + i*rst) : src(i);
if (res) stt(res, rb + i*rst, v);      // residual = x + residual
ss += v * v;                            // the variance is over the SUM
...
auto normed = [&](int64_t i) { return src(i) * sc * (ldt(w, i*ws) + wadd); };
```

with `src(i)` reading the residual *after* the update. Its comment states it outright: "With
`residual` it is the fused_add_rms_norm form: the sum is written back to `residual` (INOUT; the next
layer's shortcut reads it) and the norm is over the sum."

**The row takes operand 0 alone.** `src/adjoint/host/elementwise.cpp:461` binds `x = op[0]` and the
loop at `:518-527` never reads operand 1, so on every fused norm whose record carries a residual --
every one on this container -- the row differentiates `rmsnorm(x)` where the kernel computed
`rmsnorm(x + residual)`. That single wrong operand accounts for every exclusion above:

| observation | explained by normalising `x` instead of `x + residual` |
|---|---|
| `deq_rel` 0.99-1.62 at all 28 fused norms, 0.0034-0.0078 at all 20 gate records | gates have no residual operand |
| `row_ratio` 0.988-1.056 | `x` and `x + residual` are both residual-scaled vectors |
| `sort_rel` 0.33-0.52, `y2_rel` tracking `deq_rel` | neither a permutation nor a rotation placement |
| `bestrow` offsets scattered | not a row shift |
| plain `rmsnorm` at 1458 matches to 0.28% (`NORMELEM`) | it has no residual operand |

**Measured, not inferred.** The fused norm rebuilt from each candidate operand, against the
kernel's own reconstructed activation (`DEQELEM ... operandN_rel` = `max|y_N - dq| / max|dq|`):

| pos | `operand0_rel` (what the row uses) | `operand1_rel` |
|---|---|---|
| 1449 | 1.364 | **0.0080** |
| 1429 | 1.046 | **0.0054** |
| 1421 | 1.394 | **0.0087** |
| 1408 | 1.203 | **0.0062** |
| 1400 | 1.363 | **0.0088** |
| 1387 | 1.426 | **0.0071** |
| 1379 | 1.490 | **0.0087** |
| 1366 | 0.986 | **0.0060** |

and at a gate record the relationship inverts as the operands predict (operand 0 right at 0.0078,
operand 1 meaningless at 31, because a gate has no residual operand). 0.5-0.9% is the same floor the
gates and the plain `rmsnorm` sit at, so operand 1's snapshot *is* the norm's input: the tape
snapshots an operand after the op issues, so the in-out residual's value is the narrowed sum
`x + residual_old`, exactly what the reference's `src(i)` reads. The row therefore does not need to
sum anything -- it needs to take operand 1 instead of operand 0, and split the input cotangent
across the two summands.

No test caught it: `tests/gpu/adjoint_quant.cpp:154-187` builds the fused shape **without** the
in-place mark on operand 1, so the residual path is unexpressible in the fixture, and the host
fixtures carry no residual at all.

The backward has the matching hole: the sum's cotangent must reach *both* `x` and the residual's
prior value, and the residual's own in-place write cotangent must be added to its slot. Today the
residual slot is filled by the direct-path passthrough (`src/backward.cpp:803-812`), which is why
the earlier trace of that slot found no row involvement.

### The fix

Both sides now take the residual operand's snapshot as the norm's input `v` -- no summation in the
row, because the tape's post-issue snapshot already holds the narrowed sum -- and split the input
cotangent as `add` does, one copy of each term per slot:

- `src/adjoint/host/elementwise.cpp` -- `rmsnorm_residual` (by-role, with a named refusal on a size
  mismatch); `rmsnorm_twin` and `rmsnorm_host_adjoint` take `v` from that operand; the row adds
  `J_norm(dy) + cot_res` to **both** `dc[0]` and `dc[residual]` (`dv = 0` reproduces `add`); a
  `NORMSCALE` print under `R97_GAIN_POS`.
- `src/adjoint/host/quant.cpp` -- `fused_twin`'s residual slot carries the sum; `fused_adjoint`
  builds `ecot = {g_r, qdc[0] + out_cot}` and **skips the residual in the fold**, which used to
  mis-fold it into the intermediate's cotangent -- a second defect on the same record.
- `src/adjoint/device/quant.hip` -- the mirror: kernel unchanged, `v` uploaded in place of operand
  0, by-role `res_at` with a named refusal, the residual's cotangent kept off the rotation, both
  slots receiving `dv + g_r` once (`:977-1006`), pure superset with no residual.
- `tests/host/adjoint_rmsnorm_residual.cpp` (new) -- forward exact (`residual rel 0, normed rel 0`)
  and gradient `rel 1.33e-10` on both slots against a finite difference of the same decomposition.
  The FD zeroes the quantiser's `codes`/`scale` cotangents, which are piecewise constant in the
  input and cannot be finite-differenced; with them live the test failed at 5.5% while measuring a
  different function from the one the row differentiates.

Two defects sat on this one record, and they are separable in kind. The **wrong input operand**
changes *which function* the row differentiates -- the direction. The **fold in `fused_adjoint` that
leaked the residual's cotangent `g_r` into the elementwise cotangent** is a walk-level double count
*outside* the row: the residual stream's dominant cotangent was added to the norm's input cotangent
at every fused norm. Only the second can explain the magnitude, and only the second is invisible to
the in-row check -- `DUMPNORM`'s `dy_sum` against `DUMPQUANT`'s `quant_dx + dx` compares the row's
output L1 with its input, both computed from whatever input the row was handed, so a gain error
outside the row and a wrong input both leave it at ~1.0. That is also why that measurement, and the
in-place passthrough's insensitivity, are consistent rather than contradictory.

The growth's disappearance (below) was measured with **both** fixes in place. Which of the two
removed it is the mechanism above -- consistent with the insensitivity, and the only one of the two
that can produce a per-block factor -- but it has not been A/B'd against the input change alone.

Two follow-ups carried, not fixed: `rmsnorm_quant_fp8` is a separate row in `quant_fp8.cpp` and is
not in this container's program; and `R97_REGISTER_ADJOINT_IF_F(..., rotated, 1)` freezes operand 1,
which is the gain in the no-residual layout and the residual in the residual layout -- it works
because the walk sizes an in-place frozen slot anyway, but it is misaligned with the op's schema.

### The remaining per-block factor is the residual handoff, not the row

After the residual-operand fix the walk's cotangents decay ~x0.3 per block instead of growing ~x3.8
(the FD ladder at blocks 63 and 0 then read 2.2e7x and ~1e39x too small, a uniform 3.2x/block;
**both ratios are retracted** in "The reference is not a derivative" below -- the FD reference on the
real container is not a derivative, so no ratio can be formed from it).
The row is not the cause: its split is proven against a finite difference of `v = x + residual_old`
(both slots 1.33e-10) and the same-term A/B is refuted by that test -- `dc[0]` reads 0.247 with `g_r`
off the branch slot. The live walk confirms the rule too: `l1_dx == l1_dx_res` at all eight fused
norms (`R97_GAIN_POS=1`, `SPLIT`).

What is broken is which terms reach the row (`SPLIT` at the eight fused norms):

| pos | block | `l1_dy` | `l1_cot_res` | `l1_dx` |
|---|---|---|---|---|
| 1449 | 63 ffn | 15.71 | 0.02845 | 0.02926 |
| 1429 | 63 attn | 3.994 | 0.02926 | 0.02931 |
| 1421 | 62 ffn | 16.88 | **0** | 0.007908 |
| 1408 | 62 attn | 0.5828 | 0.007908 | 0.007912 |
| 1400 | 61 ffn | 4.745 | **0** | 0.002366 |
| 1387 | 61 attn | 0.1752 | 0.002366 | 0.002367 |
| 1379 | 60 ffn | 1.295 | **0** | 0.000647 |
| 1366 | 60 attn | 0.04298 | 0.000647 | 0.000648 |

`l1_cot_res` -- the residual write's incoming cotangent -- is **exactly zero at every ffn norm**, so
the row's output there is `dv` alone (0.007908 against an input of 0.02926): the accumulated
residual cotangent is discarded as the chain crosses each block. That is the x0.27/block, measured
rather than inferred.

The mechanism is in the distribution (`src/backward.cpp:1021-1047`): an in-place write's cotangent
travels to the version its **read segments** name, and `DEVRESTORE` shows the attn norm's residual
read being served from the **block-boundary store** rather than from the version the previous
block's ffn norm wrote. With `seg.version < 0` that cotangent leaves the block via `add_input_full`
instead of feeding the previous writer's version, so the ffn norm's write has no reader in the
version graph, `cot_res` is 0, and its `dc[residual]` carries only the norm's own small term. The
block handoff still works -- the cotangent does reach the next block's attn norm through `incoming`
-- but the ffn norm's slot, and with it the feedback the `add` half of the fusion owes that write,
is dropped every block.

**The constraint on any fix, stated before one is attempted.** The boundary store is not a mistake to
be removed: it exists because the engine reuses its arena between ops, so an in-place read genuinely
cannot be served from the live buffer and the stored copy is what the op consumed. The attn norm's
residual read *is* served from that store, correctly. So the fix is **not** to repoint those read
segments at the previous writer's version -- that would delete the boundary path and reintroduce
whatever it was built for. It is to credit the writer's version **in addition** to
`add_input_full`: the cotangent must both leave the block through `incoming` (as it does now) and
land on the version the previous block's ffn norm wrote, so that write's slot carries its consumers'
cotangent. Anything that moves the term rather than duplicating it will trade this decay for the
growth the boundary path was introduced to stop.

This is walk-side, so it is reachable from the host tier: `SPLIT` is a host-row print and reproduces
the same `l1_cot_res = 0` on the host arm, and `tests/host/backward_compose.cpp` drives the walk core
with a synthetic tape (`BoundaryNames::from(VectorTape)` + `StoredSnapshots` + `Seed`) in seconds. A
fixture there should reproduce the missing back-credit, so the fix can be iterated without a GPU
window each time; the GPU arm is then only for confirming it on the real container.

**That fixture now exists and corrects the mechanism.** `tests/host/backward_resid_handoff.cpp` (own
rows, `BoundaryNames` + `StoredSnapshots` + `Seed`, two `backward_block` calls joined by the join's
own pick rule) reproduces the live pattern in seconds:

```
JOIN below=m l1=0.203692 | the residual input x left block B with l1=0.203692
SPLIT pos=1 role=ffn l1_cot_res=0 (want the next block's residual cot l1=0.203692)
SPLIT pos=0 role=attn l1_cot_res=0.0676224 (the in-block reader; control)
```

`JOIN below=m` is the defect in one token. **Both** boundary-sized buffers of the upper block leave
it -- the residual `x` through `add_input_full` and the branch `m` -- and `pick_boundary` keeps the
first in **name order** (`src/backward.cpp:1080-1083`: the inputs map iterates in op-name order). The
block's carried cotangent therefore lands on the branch's write version, the ffn norm's residual
write has no reader in the version graph, and its slot is exactly zero. The earlier reading of this
section put the defect in the distribution's `seg.version < 0` routing; that routing is correct --
the residual *does* leave the block -- and the loss is one step later, in which of the qualifying
outputs the join chooses to carry.

The test asserts the defect's observable (the ffn slot exactly zero, the attn slot non-zero, the
boundary input non-zero), so it passes now and **fails once the pick is fixed**: the zero assertion
becomes an equality against the block's residual input cotangent, printed as `want` above. That is
the fix's acceptance test.

**How the fix chooses, as landed.** The carry is not a name or shape preference. The set is the
upper walk's own input list (`BackwardResult::inputs`) filtered by the boundary-shape gate, and each
entry is then checked against the block below: if the lower block **writes** that buffer, the carry
is refused by name. That list is tape-derived and is exactly "the buffers the block above consumed
from below", because every read with no earlier writer in the block leaves through
`add_input_full` (`src/backward.cpp:1014-1032`). On this container family it is the residual **and**
the branch.

My first instruction -- "the buffers the next block's records read" -- was wrong, and the agent
found it: the lower block never reads its own mlp output, it **writes** it; the branch is read by
the upper block. A literal read-of-the-lower predicate yields `{residual}` alone and starves the
branch, which is the regression the two-buffer rule exists to avoid.

Both paths are now pinned by the test, which is what makes this safe to land: a reproducer that
asserted only the residual would go green while the branch was zeroed.

```
JOIN below=m,x l1_m=0.203692 l1_x=0.203692
SPLIT pos=1 role=ffn l1_dy=0.203692 l1_cot_res=0.203692 (want the residual cot l1=0.203692)
SPLIT pos=0 role=attn l1_cot_res=0.271315 (the in-block reader; control)
SPLIT pos=2 role=mlp l1_cot_m=0.203692 (want the branch cot l1=0.203692)
```

The ffn norm's residual write now receives block B's residual cotangent **element-wise** and its own
`dy` is non-zero again; the mlp receives the branch cotangent; the attn control stays non-zero. The
equalities are bit-exact f64 copies, no tolerance.

Two shared-interface notes: `in_place_read` moved from `src/backward.cpp`'s anonymous namespace into
`include/r97/backward.h:41-50`, so the walk's version graph and the join derive from one rule; and
`store_boundary` now takes the whole vector and selects from it, since the boundary vector holds
`IncomingCotangent` rather than `InputCotangent`.

That selection matters more than it looks. `report.boundary_cotangent` is **defined** as the
residual's cotangent: `tests/host/step_host_boundary.cpp:245` finite-differences it against residual
perturbations, `tests/host/step_host_oracle.cpp:261` matches it to `residual_grad.<i>`, and the
criterion-two table above reads it. With two entries carried, "the first one" is the branch as often
as not, which would have silently redefined all three.

It selects the entry named by the **store's own record** (`ResidualStore::set_name`, filled at
`keep()` time from the buffer `residual_bytes` recognised), refusing by name when the walk left no
cotangent for it. Two earlier attempts were wrong and the host tier said so: the block's first
in-place read is a rope's write on every model with a rope -- eleven cases refused, naming
`attn_qg` -- and `BoundaryNames` is name-keyed, so a residual repeated at every boundary collapses
to one id and answers `logits`. The store cannot disagree with itself about what it holds.

### After the handoff fix: the boundary moves, the top block does not

The 512 gate improved by 143x in magnitude, stayed finite, and every descent probe still descends
and descends harder:

| | before the handoff fix | after |
|---|---|---|
| grad norm | 5.98934 | 858.248 |
| descent probes | -272.13, -387.26, -214.04, -334.80, -219.37 | -257.82, -741.63, -780.25, -258.80, -783.34 |
| loss after one step | 7099.46 | 7113.77 |
| step | 328.5 s | 311.3 s |

The FD ladder separates two things the fix could not both address.

**Block 0** (`linear_attn.in_proj_qkv.lora_A`): the walk's gradient moved from `7.6652e-32` to
`0.898438`, so the handoff fix reaches the walk. Its central difference is not a usable arbiter
though: at eps=1e-4 the loss moves 7381.51 -> 7396.38, a discrete 15-unit jump, so
`central=-74358.9` is a quantisation or token flip rather than a derivative; at eps=1e-6 both
central differences are exactly 0, below bf16's resolution. The walk's `0.898438` is neither
confirmed nor refuted by this ladder.

**Block 63** (`self_attn.q_proj.lora_A`): the walk is **unchanged** at `4.33922e-05` while the FD
reference reads `948.917`. That the value did not move is expected and diagnostic: block 63 is the
top block, so its leaf gradient never crosses a boundary below it and no handoff change can move it.
Whether `4.33922e-05` is *wrong* is not established by this ladder -- see below.

**The reference is not a derivative.** The third point settles it:

| eps | central |
|---|---|
| 1.14e-6 | 0 (the bf16 floor) |
| 1.14e-4 | 948.917 |
| 4.56e-4 | 152.981 |

A linear response gives the same central at every step. This one falls 6.2x when eps grows 4x, so the
finite difference is dominated by something that shrinks with eps and `948.917` is not a usable
derivative. The real container's forward is bf16 weights through an int8 path: it is piecewise
constant at this scale, and no FD on it can arbitrate a magnitude.

**The ratio does not survive, and it must not drive the next step.** I first wrote that because the
central falls as eps grows, the true gradient must be at least 948.917 and the walk therefore
>=2.2e7x too small. That inference assumes a smooth, concave response; for a piecewise-constant
forward it fails outright, because a single quantisation flip inside the interval scales the finite
difference as 1/eps and says nothing about the derivative. So **the 2.2e7x shortfall at block 63 is
retracted**: neither point is a derivative, no ratio can be formed from them, and criterion two is
open and unmeasured at this leaf rather than refuted.

That does mean the magnitude needs a different arbiter: a finite difference on a quantised forward
cannot supply one. The candidate already in the tree is the **reference step** (`src/reference_step.cpp`),
the f64 exact step the host tier's oracle test compares against -- a gradient-to-gradient comparison
there would be a far better instrument than FD on the real container.

The host walk is still the right discriminator, and it does not depend on the FD: the same batch's
block-63 leaf through `R97_HOST_WALK_AB=2 R97_WALK_BLOCKS=1`, the path the host tier already
finite-differences on models where FD is valid. Agreement with the device walk means the walk is not
what differs and the whole magnitude question is in the instrument; disagreement hands over a value
to bisect the device walk against, which is far cheaper than probing it blind.

The arbiter is `R97_DUMP_LEAF=<name>`: each arm prints its own leaf's gradient, labelled `(walk arm)`
or `(host arm)`. Its failure mode is worth recording, because it is silent. `R97_HOST_WALK_AB=2`
returns from its own arm -- before the device walk's block -- and its gradients live in `host_grads`,
never `grads`. A print placed after that return therefore reports the *device* value in both runs:
two identical numbers, indistinguishable from agreement. The labels exist so that cannot recur.

### The device rows agree with the host path at block 63 -- and that is all it says

`R97_DUMP_LEAF` prints the same leaf from both arms in one run:

| arm | `model.layers.63.self_attn.q_proj.lora_A` |
|---|---|
| device (walk arm, 306.0 s) | 4.33921814e-05 |
| host (host arm, top block only, 172.9 s) | 4.31380649e-05 |

They agree to 0.59%. The scope of that is narrow, and it matters more than the number:

- **It is one leaf in the top block.** Block 63 never crosses a boundary below it, so it was never
  where the per-block decay localised. Blocks 0..62 are **unarbitrated** by this comparison, and the
  host arm cannot reach them cheaply: `R97_WALK_BLOCKS` walks the top n blocks only.
- **Both arms share the join, the head's seeds and `src/backward.cpp`** (`boundary_cotangents`,
  `enter`, the version graph). A defect in any of those reproduces identically in both arms, so
  agreement cannot clear the walk -- only the **device rows** at that leaf.
- **The FD's 948.917 now disagrees with both arms.** Either the FD is unusable (established: it
  swings 948.917 -> 152.981 for a 4x larger eps) or both arms share an error. This comparison cannot
  separate those two possibilities.

So the honest statement is: **the device rows agree with the host path at block 63.** The magnitude
is not thereby right, and criterion two is not thereby cleared. The separator is the f64 reference
step (`src/reference_step.cpp`), not the host walk.

**The separator I named does not exist at this scale.** `src/reference_step.cpp` is the *tiny
fixture's* f64 step (`include/r97/reference_step.h:3`): its value path is `std::vector<double>` twins
over a recorded program or one live host-kernel trace, with the container supplying the weights.
Architecturally that is exactly the right instrument -- it is not the int8 path and not the kernels
-- but it takes a whole model's program, so pointing it at this container means a 64-layer 27B f64
walk on the CPU. It cannot arbitrate criterion two as it stands.

So criterion two currently has **no available arbiter**: the FD ladder is invalid (a quantised
forward), the host walk shares the machinery under suspicion, and the reference step is
fixture-scale. What is needed is a **single-block f64 reference at the real container's shapes** --
a new instrument, and the honest next step rather than a number.

**But the descent ladder is a scale check that needs neither.** AdamW's first step at a small enough
lr is `~lr*sign(g)` whatever the forward does, so `|dL|/lr` estimates the gradient's L1 directly.
The ladder is monotone in the wrong direction so far -- 7.80e6 at lr=1e-4, 2.47e6 at 3e-4, 2.58e5 at
1e-3 -- which says every measured point is past the linear regime and the smallest still overshoots.
The ladder now runs to 1e-7. If it plateaus below `sqrt(N)*858.248` the reported scale is consistent
with the loss; if it keeps growing past that bound, the reported norm cannot be the gradient's. That
is a **scale check, not a correctness check**: it can refute the magnitude, it cannot confirm the
gradient.

### The scale ladder: no plateau, so no estimate -- confirmed on two runs

Run on grad norm 860.732 (the scale-probe tree) and again on window27 (norm 859.419, current head
with the writer fix + regenerated fixture). Neither run reaches a plateau:

| lr | run1 dL | run1 `abs(dL)/lr` | run27 dL | run27 `abs(dL)/lr` |
|---|---|---|---|---|
| 1e-3 | -257.8 | 2.58e5 | -269.9 | 2.70e5 |
| 3e-4 | -748.0 | 2.49e6 | -750.2 | 2.50e6 |
| 1e-4 | -786.6 | 7.87e6 | -792.4 | 7.92e6 |
| 1e-5 | -422.0 | 4.22e7 | -426.1 | 4.26e7 |
| 1e-6 | -171.4 | **1.71e8** | -160.2 | **1.60e8** |
| 1e-7 | -0.519 | 5.20e6 | **+10.59** | (ascent: the probe's loss goes UP) |

Both non-monotone past 1e-6; the second crosses sign. No point is in the linear regime -- a
tenfold smaller step from 1e-4 loses only ~1.9x of its loss change, and 1e-7 loses or *gains* by
less than 1e-5 of it. The reason is the same one that invalidates the FD -- the forward is
piecewise constant, so the loss does not respond proportionally to the weight change -- and it is
not fixable by choosing a smaller step. The ladder therefore cannot estimate the gradient's L1,
and the `sqrt(N)*~860` bound cannot be applied: there is no plateau to compare. Two runs agreeing
on non-monotonicity makes this robust across trees rather than a single-run artifact.

So **four instruments have failed to arbitrate the magnitude, each for its own reason**: the FD (a
quantised forward), the host walk (shared machinery), the reference step (fixture-scale), and the
scale ladder (no linear regime). The remaining one is the only one that avoids all four -- a
single-block f64 reference at the real container's shapes.

### What that instrument needs, and what already exists

- **Exists**: the twins and the host adjoints run a program in f64 with the container's weights
  (`make_reference_step`, `src/reference_step.cpp`); `tools/dump_weight.cpp` reads any container
  tensor without a card; the walk's tape names every op, its operands and its buffer versions.
- **Missing (1)** -- *checked, and it is not missing*: a block's op list is
  `ops[block.begin .. block.end)`, because `Tape::Block` already carries the record indices
  (`src/join_common.h:357-359`, `block_wrote` walks exactly that range). What the reference step
  lacks is a way to *accept* a sub-program, not a way to find one.
- **Missing (2)** -- *also checked, also not missing*: `R97_DUMP_LEAF` takes a **substring** of the
  leaf name, and a leaf is named `layers.N.self_attn...`, so `R97_DUMP_LEAF=layers.0.` prints block
  0's leaves. The device walk reaches all 64 blocks (`R97_WALK_BLOCKS` truncates the *host* arm
  only), so any block's leaves can be dumped today.
- **Missing (3)**: the f64 block must start from the *device walk's* incoming cotangent for that
  block, or the comparison is confounded by every layer above it. The SPLIT instrumentation printed
  only the cotangent's L1; **`R97_SPLIT_VALUES=<n>` now dumps its first n values at `%.17g`**
  (`src/adjoint/host/elementwise.cpp`, `R97_SPLIT_POS` to pick the position) -- a *sample*, which
  checks a seed but is not one (see below). It carries the same scope limit as the host-walk arbiter:
  it tests this block's rows given the same input, not the whole walk.
- **The first slice**: one block below 63, one leaf, device vs f64. The leaf and the block's op list
  come from what exists; `parse_program` reads a **JSON** program (`program_json(prompts, ops)`), so
  a block's op slice is a filtered JSON file, not C++.
- **The closed gap, corrected: the slice cannot run through `ReferenceStep::run` at all.** Three
  reasons, verified in the source before any code was written: (1) `make_layout` throws "the graph
  writes no logits" unless some op wrote a `logits` buffer (`:611-615`) -- a single block's slice
  never does; (2) `run` builds the objective over `RefTape(batch, run.logits, vocab)` and seeds the
  backward from `obj.seeds` (`:572-579`) unconditionally -- a slice has no head, no loss, no seeds;
  (3) `resolve_read` serves any buffer no slice-op wrote from `constants_` and then from the
  *container weights by name* (`:972-982`) -- a block boundary's activations have no injection path
  and would throw, or silently load a name-colliding weight as an activation. So the slice needs
  **three inputs** the reference step does not have: boundary cotangents (planned), boundary
  *forward values* (the twins must see the block's incoming activations to replay its ops), and an
  objective-free run mode. Cleanest shape: a separate slice runner -- a tool or test-side harness --
  that reuses `parse_program`'s JSON, the twins and the host-adjoint backward core directly, rather
  than threading options through the whole-graph contract in `ReferenceStep`. As previously
  written ("two pieces"), the first GPU-window run of a filtered program would have thrown at
  `make_layout` before any comparison -- a whole window spent on an exception.

**The runner is landed** (`tests/host/slice_reference.cpp`, commits `d50f10d`, `0e5dff3`,
`52ffce1`): objective-free f64 replay of a filtered r97-program slice through the registered
twins and `backward_block`, with boundary forward values and `std::vector<r97::Seed>`
(`SliceInput::seeds`) injected by buffer name and any other read refused by name. Both
acceptance cases pass standalone (`R97_TEST_ONLY=... r97_host_tests`) and the `host_tests`
ctest entry runs the binary unfiltered to exit 0. Red-run proof: with
`options.in_place_passthrough = false`, case (b) fails on `dx`; with `true` both pass -- the
case pins the option by construction, not by accident.

**Serialization trap — landed, and the gate is green again.** `program_json` originally wrote
NO in-place field, so a slice recorded through it silently lost every in-place mark and would
have parsed `in_place=false` everywhere. The writer fix landed (`tools/ops.cpp` now emits
`"inout": o.in_place`), the trace test is green again (301/0), and the checked-in fixture
(`tests/fixtures/programs/oracle.json`) was regenerated through the new `R97_RECORD_ORACLE_FIXTURE`
env gate in `reference_step.cpp` — it now carries 6 real `inout: true` marks (all
`gdn_conv_prep`'s kv:3 state buffer), matching the live trace bit-for-bit. The runner reads
`inout` first, then `in_place` (a hand-written record convenience), so both forms parse.

### The remaining GPU-side run, mapped (2026-11-19, machine free after the foreign window was
cleared on the owner's instruction at 11:38, logged in /tmp/window-handclose.log)

Everything below is an existing API except one small env-gated dump in `step_device.cpp` (item 3); no new walk code is needed beyond that:

1. **The block's op slice, tape-serialized.** The walk already has the tape: `driver.last().tape()`
   (`src/step_device.cpp:474-476`), and a block's records are `ops[blocks[layer].begin ..
   blocks[layer].end)` (`:1573-1575`). **Landed** — the writer now emits `"inout": o.in_place`
   (`90f7193`), so dump the slice with the fixed writer and re-index positions 0..n-1; the snapshot
   store keys on (position, operand slot) and the walk's backward uses original positions, so a
   dumped slice that keeps tape positions 0..1463 would mis-key every snapshot against the record
   the walk used.
2. **Boundary FORWARD values.** NOT in the report's `measured.*` fields: `store_boundary` returns
   immediately when `options.keep_boundaries` is false (`src/step_device.cpp:1213`), and the
   window's `device_step_tape` runs with it **off** (`tests/gpu/device_step.cpp:74-75, :404-406` —
   the f64 copies are ~2.7 GiB per boundary at 512 tokens, and the case does not ask for them), so
   `measured.boundary_value` / `boundary_cotangent` / `boundary_name` are **empty vectors** in a
   dump run. The dump must read the store directly, the same way `store_boundary` does when it is
   on: `store.read(boundary, host.data())` then `widen_bf16(host, &out)` (`:1232-1234`), on the
   `reuse::` boundaries the `ResidualStore` keeps (`include/r97/replay.h:66+` — every boundary,
   bf16, `tokens*hidden*2` bytes each; `store.name(boundary)` for the buffer name;
   `store.extra_at`/`read_extra` where the plan holds a second copy). Gated on its own env var at
   the `store_boundary` call site (`:1211-1237`) or beside `:1620-1623` so it costs nothing when
   unset. For the slice runner's `SliceInput::boundary`, each boundary becomes one f64
   `AdValue{buffer=name, dtype, shape={tokens,hidden}}`.
3. **Boundary COTANGENTS (full-range Seed).** `boundary_cotangents(...)` at
   `src/step_device.cpp:1620-1623` produces the `std::vector<IncomingCotangent>` for each block as
   the walk descends; `store_boundary` (`:1211-1235`) already copies the selected one into
   `measured.boundary_cotangent`. To dump ALL of them per block, print each `IncomingCotangent`
   (buffer, row_lo, row_hi, value) at that call site gated on an env flag -- that is a few lines
   in `step_device.cpp`. **Type note:** `IncomingCotangent::value` is f64 full-range
   (`include/r97/backward.h:82-86`); `Seed::cotangent` is f32 (`include/r97/objective.h`), so the
   dump must `as_f32`-cast each value -- the same conversion the harness's own cases do — to
   yield `std::vector<r97::Seed>` for `SliceInput::seeds`.
3b. **The CONSTANTS the slice's ops read — the corrected split (host vs window vs not servable).**
   A real block's slice reads weight operands (`blk.N.attn_norm.weight`, `blk.N.ssm_in.weight`,
   ...) that no slice op writes, and the runner refuses them by name (`src/slice_reference.h:273-277`)
   unless `SliceInput::constants` carries them. The earlier claim that "all constants are host-side
   prep, orthogonal to the window" was WRONG for the real container: it only held for bare weights
   readable from the container file. The corrected split, three classes:
   - **HOST-side, servable as constants** (`dump-weight --full`; 7 bare container weights):
     `blk.37.attn_norm.weight`, `blk.37.ffn_norm.weight`, `blk.37.ssm_ab.weight`,
     `blk.37.ssm_conv1d.weight`, `blk.37.ssm_a_log`, `blk.37.ssm_dt.bias` — already in
     `/tmp/r97-55-dump-block37/const.json`. These resolve via `r97train dump-weight
     <file.rad> <name>` with **no card, no engine, no window** (`tools/dump_weight.cpp:18-19`).
   - **WINDOW-side, servable as constants**: the frozen `lora.blk.37.*` read snapshots — the
     walk's own shared snapshots, decoded f64, dumped by the extended hook into
     `layers.37-constants.json`. These exist only at window time; no host-side prep can produce
     them from the container alone.
   - **NOT servable, not constants**: the 8 plugin-private gemm_nt_q weight/scale operands
     (`ssm_inz`, `ssm_out`, `ffn_gate_up`, `ffn_down`, each weight + scale). Their
     `AdValue::value` is empty and only the device row's raw/enc decode serves them; the slice
     runner's constants path cannot carry them, so the first honest comparison records the
     slice-serve refusal chain naming these 8 — not a fabricated constant.
   - **FIFTH dump file.** The block dump is now five files, not three: the slice, the boundary
     forward values, the seeds, the constants, and the constants-refused list.
   - **Leaf-name reconciliation.** The walk prints `model.layers.37.<module>.lora_A/B`; the
     slice-serve runner emits `<bo.prefix><module>.lora_A/B`. `--prefix` must be set to
     `model.layers.37` and the module spellings verified against the walk's leaf names before
     diffing, or the leaf names will not line up even when the values do.
   - **Runner fix that followed (commit 22ca128's sweep).** `run_forward` handed twins a
     compacted reads vector (non-written slots only), while every registered twin indexes its
     operands schema-positionally — the fused-norm family reads `op[0]=x`, `op[1]=residual`,
     `op[2]=weight`, so the compaction silently reordered the schema under them and the 5120-value
     gain reached the twin where the 2621440-value activation belonged. The fix gives the twin
     every operand slot in schema order (written-but-not-in-place slots ride along as empty
     `AdValue`s, dtype "invalid"), matching `src/reference_step.cpp:886-927`'s convention; both
     slice acceptance cases pass with `in_place_passthrough` on, and the passthrough contract is
     unchanged.
4. **Device-side leaf gradients for the same block.** The walk stages per-leaf f32 accumulators on
   the card (`src/adjoint/device/grad_stage.{h,cpp}`); `grad_stage_flush(grads)` at
   `src/step_device.cpp:1674` materialises them into the `ParamSet` at step end, named by
   `factor_grad_name`/`split_adapter_grads` (`:1678`). The existing `R97_DUMP_LEAF=<substring>`
   already prints them (`LEAF <name> grad=<v>`), so `R97_DUMP_LEAF=layers.N.` is the comparison
   side; no new code needed there.
5. **The comparison itself** is host-side and needs no window. **Driver shape, decided:** a tool
   subcommand, not a test. The runner currently lives in the test TU's anonymous namespace, so
   lift `namespace slice` into a new header (e.g. `tests/support/slice_reference.h` or a
   `tools/` internal header) and have both the test and the new subcommand include it. The
   subcommand takes `--program slice.json --boundary fwd.json --seeds seeds.json --out report.txt`
   and prints the same `BUF`/`LEAF` lines; then diff its `LEAF` line against the device walk's
   `LEAF` line for the same leaf name. One block below 63, one leaf. (A test cannot be the driver:
   AGENTS.md's no-skip rule blocks an env-gated test that no-ops when the dump files are absent,
   and both existing cases are inline-synthetic.)

Window discipline, unchanged: one held window per batch via local-radiance-window.sh, launch
detached with a STARTED line and a trap-EXIT close writing a DONE marker, and after any long job
check `model-state/pause` by hand even on DONE rc=0 (a trap can leave it behind -- seen twice).
- **Scope, stated once -- and corrected once by review**: the slice's `backward_block`
  (`src/backward.cpp:208`) is the SAME core the reference step drives (`:1053`) AND the device walk
  drives (`src/step_device.cpp:1170`, `:1595`). Of the three components that scoped the host-walk
  arbiter's 0.59% (the join, the head's seeds, `src/backward.cpp`), caller-filled snapshots drop
  the first TWO but RETAIN the third: a defect in the core's in-place serving, passthrough or slot
  ordering reproduces identically on both sides and stays unarbitrated. So slice-vs-device agreement
  clears that block's ROWS and rules out the join's boundary handling there -- it does NOT rule
  out a backward-core defect. What the instrument ADDS over the walk-vs-walk arbiter is real:
  blocks 0..62 (no `R97_WALK_BLOCKS` top-n limit), no join, no head seeds. Same discipline as the
  0.59% scoping: the claim that survives is "the rows at that block", never "the walk".

The work is a real instrument, not a rerun, and it should start in a fresh context: the four
negative results above are the specification, and re-deriving them costs a GPU window each.

## The slice run: the instrument serves the real block, and the gap is in its forward

Three windows on 2026-11-19 (29: the five-file dump, 30: the walk's per-leaf norms, 31: the same
dump carrying the in-place `kv:` state priors) and the host-side runner now take block 37 of the
real container end to end: `r97train slice-serve` returns `rc=0` over the block's 21 ops and walks
them in f64, in ~160 s.

**What the runner needed, in the order the refusals named it** (each is a committed fix):

1. `prepare_adjoint_record` on the twin call. The gemm twin finds its operands by schema name
   (`operand_pos(rec, "b")`), the tape names them `blk.37.ssm_inz.weight`, and the adjoint guard
   turns the miss into a silent no-op -- zero trunk propagation with plausible-looking leaf lines.
2. The schema-renaming pass, `own_raws` and `record_weight` moved to `src/adjoint_schema.h`,
   unguarded: the runner's translation units never open `join_common.h`'s
   `RAD_ISSUE_OBSERVER` guard.
3. The raw position tables, rebuilt the way the reference step names and fills them
   (`src/reference_step.cpp:288-302`, `:438-486`): positions `0..M-1`, `cu_seqlens` `[0, M]` for
   one sequence, the sequence's state slot, `gdn_conv_prep`'s zero `[n_seq]`, `chunk_scan`'s
   `[[s]]` arange -- and, on the lora records, the pool's row map (one slot: all zeros).
4. One name per raw/absent operand (`own_raws`): position 10 is `[1]` in `gdn_conv_prep` and
   `[1,1]` in `chunk_scan`, and the backward core records one shape per buffer name.
5. A write that covers the record's whole buffer replaces it: the dump's activation leak puts a
   stale same-named constant in the environment, and it must not veto the twin's production.
6. The 8 plugin-private operands: `slice-serve --model` widens the weight from the container
   exactly as the host walk's `register_stored_operand` does; the scale has no container entry
   (the engine fuses it into the joined weight and `r97_iqr_decode_planes` folds it in), so it
   becomes the walk's own valueless, shaped, pointer-carrying snapshot -- never a fabrication.
7. One leaf per record: with `fused_slots` unset, all four lora records' factors landed under the
   bare `<prefix>.lora_A`, summed positionally into one buffer with the first record's shape. The
   walk names the factor from the record's weight (`src/step_device.cpp:1145-1160`); the slice has
   no container module map, so `record_prefix` returns the record's own weight and the leaves read
   `model.layers.37.blk.37.<weight>.lora_A`.
8. The in-place `kv:` state: the walk's adjoint zeroes a carried state this block has not written
   (`src/step_device.cpp:1582-1588`), but its forward values are the device's. Window 31's dump
   carries the real prior -- `kv:3`, 30,720 values, max |44.25|, none zero -- and the runner feeds
   the twin that value. (The zero rule moved the numbers by 1e-4: not the gap.)
   The rule is not read-only: `:1590-1594` zeroes a **written** `kv:` operand's snapshot as well,
   with no in-place test, which is `gdn_chunk_scan`'s plain write of `kv:2` (op 7, position 9 —
   plain because the record also *reads* `kv:2`, so `r97::in_place_read` is false for that write,
   include/r97/backward.h:42-50). The runner stored the twin's production there; it now mirrors
   the walk (`src/slice_reference.h`'s `cold_write`) while the environment still receives the
   production, so a later in-block read of the pool sees the live state exactly as the walk's
   store does. **This is a contract gap closed, not a measured difference**: the host
   `gdn_chunk_scan` row reads `call.fwd` only through its read slots (`f[rCu]`, `f[rH0]`,
   `load_chunk`/`load_state`, src/adjoint/host/gdn_scan.cpp:487-578) and takes the writes'
   cotangents from `call.cot`, so it never reads a written slot's snapshot — no other block-37 row
   reads a plain `kv:` write's snapshot either, and the leaf numbers after the fix are expected to
   be **identical**. Pinned by `slice_reference_cold_kv_write_zeroes_the_snapshot`, which asserts
   the snapshots themselves (`run_forward` + `StoredSnapshots::at`) on an `add(kv:s, b) -> kv:s`
   record — the cold write's snapshot is zero, the warm one is the production, and the
   environment holds the production either way. The case has to be that shape: a `kv:` write with
   *no* read of the same buffer is classified in-place by `in_place_read` and was already zeroed,
   and a gradient proxy would say things this rule is not about (a plain re-write of one buffer
   also pulls in the join's passthrough). With the zeroing removed the case fails on
   `kv:s (snapshot, cold write)` (0.1918, the production, where it wants 0).

**The first comparison** (window 30's walk, per-module leaves grouped by weight, against the
slice's per-weight leaves):

| weight | factor | walk l2 | slice l2 | ratio |
|---|---|---|---|---|
| `ssm_inz` | A | 0.43418 | 0.481192 | 1.108 |
| `ssm_inz` | B | 0.0279081 | 0.0395932 | 1.419 |
| `ssm_out` | A | 0.386223 | 0.40887 | 1.059 |
| `ssm_out` | B | 0.0213476 | 0.0227835 | 1.067 |
| `ffn_gate_up` | A | 3.20103 | 3.39008 | 1.059 |
| `ffn_gate_up` | B | 0.172384 | 0.293737 | 1.704 |
| `ffn_down` | A | 4.00152 | 4.16907 | 1.042 |
| `ffn_down` | B | 0.125314 | 0.130397 | 1.041 |

Set norm: walk 5.16169891 (12 module leaves) against the slice's 5.4201372 (8 weight leaves),
ratio 1.0501. The magnitudes are right everywhere -- the earlier arbiters were off by 1e33x and
249x -- the A factors and the two *unfused* B factors agree to 4-11%, and the two *fused* B
factors are 42-70% high: the engine's fused B is one dense `[N_total, R_total]` tensor, while the
walk's per-module leaves are its diagonal blocks (`split_adapter_grads`), so the two sides report
different tensors for exactly those two weights. Element-wise agreement is NOT established: the
`first=` elements differ by factors of 0.19x to 1.0x while the norms agree.

**Where the difference comes from: the slice's forward, not its adjoint.** `print_report` now
compares the slice's own forward values against the dump's device values for every constant the
block also writes (`CONSTDIFF <name> rel=<l2 of the difference / l2 of the device's>`):

```
CONSTDIFF h.q8 rel=0.205625338 n=2621440      (op 0's quantised output)
CONSTDIFF gdn_q rel=0.194039072 n=1048576     (op 5)
CONSTDIFF gdn_k rel=0.186912897 n=1048576
CONSTDIFF gdn_v rel=0.172616862 n=3145728
CONSTDIFF gdn_beta rel=0.548078349 n=24576
CONSTDIFF gate_up rel=0.187712154 n=17825792  (op 16)
CONSTDIFF gdn_o rel=1.22584308 n=3145728
```

**What that set says, and the two lines that say nothing.** The dump stores a buffer at its
*first read* (`src/step_device.cpp:1390-1415`), so a line is a measurement only when the slice's
final environment holds the same tensor. `h.q8`/`h.s8` are written at op 0 **and** op 13: the
dump holds op 0's stream (first read, op 1), the slice's environment op 13's -- the 0.2056
compares the attention norm's quantised stream against the feed-forward one and is not a
divergence. `gdn_o` is *not* in that class, and the proof is the quantiser itself: re-running
op 9's own Hadamard-and-quantise on the dump's `gdn_o` reproduces **99.9996%** of the dumped
`gdn_o.q8` codes exactly (all within 1 LSB) and `s8` to 2.06e-7 relative, so the dumped value
*is* op 8's output -- the same tensor the slice's environment holds, and `rel=1.2258` is real.
The honest set is `gdn_q/k/v/g/beta/kkt` (0.09-0.55), `gate_up` (0.188), `ffn.q8/s8`
(0.29/0.12), `gdn_o` (1.226) and `gdn_o.q8/s8` (1.206/0.470).

Three things were checked before writing that lead down, and two of them correct it:

- **A numpy reconstruction of rmsnorm + quantise reproduces 0.7% of the dumped codes -- and that
  is a fact about the reconstruction, not the twin.** The `had` in the op's name is an
  unnormalised Walsh-Hadamard per 128-group between the norm and the quantiser
  (`src/adjoint/host/quant.cpp:225`, `:182`; the family is described at `:826-832`), and a
  transform the reconstruction omits scrambles every code. Do not quote the 0.7% as evidence
  against the twin.
- **The tape's snapshot of the in-place residual is the post-op value** (`rmsnorm_residual`'s
  comment, `src/adjoint/host/elementwise.cpp:470-480`), which suggested the slice's pre-write read
  was the wrong norm input. It was tried: `reads[1] = bf16(x + h)`. It left `h.q8`'s rel at 0.207
  and moved the residual's own match from 0 to 0.020, so the device does not accumulate that
  operand at this op and the pre-write read is what the kernel consumed. Reverted; the trial is
  recorded here so it is not repeated.
- **The authoritative op-0 check already exists, and it runs in the slice, not the walk.**
  `R97_DUMP_QUANT=1` re-quantises the twin's intermediate through the same Hadamard and compares
  it with the codes the kernel wrote (`quant.cpp:1146-1230`), printing `code_err` and a
  code-distance histogram. The diagnostic lives in the **host** fused-adjoint row, and the slice
  walks `Side::host` (`src/slice_reference.h:561`) -- so `slice-serve` fires it in ~160 s, not
  the ~1 h host walk (`R97_HOST_WALK_AB=2`) the earlier reading of the code implied. Window 32's
  result is below: the twin is exact.

Two smaller corrections that came out of the same runs: `h`'s `rel=279` and `lora.t.*`'s `rel=1.3`
are artifacts of comparing a multi-write buffer's *final* value against the dump's first-read
value, not measurements; and the run-to-run self-consistency of the walk's own set norm is
5.16143265 (window 29) against 5.16169891 (window 30), 5e-5 relative -- the tolerance any later
comparison must respect.

### The slice is self-consistent; the dump was dropping the operand views

Window 32 ran the same slice with `R97_DUMP_QUANT=1`. Every fused producer's codes came back
identical to the twin's recomputation, and at the two norm ops the twin's intermediate matches
the write exactly:

```
NORMTWIN pos=0 n=5120 rows=512 w=yes eps=1e-06 wadd=1 mult=1 xmax=6560 ss_mean=3.36743e+06 s=0.000544942
DUMPTWIN pos=0  op=rmsnorm_had_quant_i8g  twin=1 codes=2621440 code_err=0 code_bad=0/2621440 scale_rel=0 act_at=5 act_err=0 act_rel=0 hist=[2621440,0,0,0,0,0,0,0,0,0]
DUMPTWIN pos=13 op=rmsnorm_had_quant_i8g  twin=1 codes=2621440 code_err=0 code_bad=0/2621440 scale_rel=0 act_at=5 act_err=0 act_rel=0 hist=[2621440,0,0,0,0,0,0,0,0,0]
DUMPTWIN pos=17 op=gated_had_quant_i8g    twin=1 codes=8912896 code_err=0 code_bad=0/8912896 scale_rel=0 act_at=1 act_err=1112.28 act_rel=8.75809 hist=[8912896,0,0,0,0,0,0,0,0,0]
```

`code_err=0` and one histogram bucket over 2.6 M and 8.9 M codes, `scale_rel=0`, and
`act_at=5` = the record's `h` write, `act_err=0`. **This is a self-consistency check, not a
device check, and the distinction matters**: the slice's forward *is* the twin chain
(`run_forward`), so the diagnostic compares the adjoint's twin against codes that same twin
wrote. It rules out an internal inconsistency -- the forward the adjoint differentiates is
exactly the forward it recomputes -- and it says nothing about the device's codes. The
twin-vs-device comparison needs the **host** walk (`R97_HOST_WALK_AB=2`, `R97_DUMP_QUANT=1`,
`R97_WALK_BLOCKS=27`; ~1 h at the measured ~2 min/block), where the twin's recomputation meets
the tape's own captured codes. (Op 17's `act_rel` is not a measurement even in that run: the op
writes only `ffn.q8`/`ffn.s8`, so no operand holds its bf16 intermediate to compare against.)

**The gap the dump *was* introducing is the operand view.** `snapshot` packs a narrowed read
over its own view shape (`src/join_common.h:264-295`), and the block's fused projection is read
three ways: `gdn_in` is `[512,16384]` (q|k|v|z) and its readers take `[512,10240]`@0 (the conv's
q|k|v) and `[512,6144]`@10240 (the gate's z); `gdn_ab` is `[512,96]` and `gdn_conv_prep` reads
its two halves as `[512,48]`@0 and `[512,48]`@48. The dump's own constants prove the view was
there -- `gdn_in` was stored as `[512,10240]` against a record that declares `[512,16384]`, and
`gdn_ab` as `[512,48]` against `[512,96]` -- but the program JSON's writer emitted only
`buffer/dtype/shape/row_lo/rows/written/inout` (`src/step_device.cpp:1713-1721`), so the slice
replayed the whole buffer: both `gdn_ab` reads became the same operand, and op 8's gate read
flat across q|k|v|z. That is the mechanism behind op 5's 0.17-0.55 and, through the gate, op 8's
`gdn_o rel=1.226` (the slice's `gdn_o` decorrelates from the device's at cos 0.268 while its norm
is 1.026x -- the signature of a wrong gate, not a wrong scale).

The fix is the round trip: the block dump and both program writers now emit
`view_shape`/`view_stride`/`view_off`, both parsers read them, `operand_at` gathers a read
through its view (a buffer whose numel already is the view's -- the dumped constant -- is
returned reshaped, not gathered twice), and `print_report` gathers the slice's buffer through
the same first-read view before `CONSTDIFF`, so the two narrowed buffers are compared instead of
silently skipped by the size check. `tests/host/slice_reference.cpp`'s
`slice_operand_view_gathers_the_named_half` pins the parser and the gather on a `[2,4]` buffer
read as two `[2,2]` halves.

**The last forward gap was the fused norm's residual input.** The elementwise part of
`rmsnorm_had_quant_i8g` norms the *post-update* residual, and the tape captures that in-place
operand after the op (`src/adjoint/host/quant.cpp:859-862`, the device's own test at `:863-869`).
The slice fed the twin the pre-write value. An offline reconstruction of op 0 from the dump --
`rmsnorm(h + x)` with the weight's `wadd` (`w' = w + 1`), then the per-128-group unnormalised
Hadamard and `s = amax/127` -- reproduces **91.5% of the dumped `h.q8` codes exactly** and
`h.s8` to 2e-3; with the residual alone it reproduces 0.9% and misses the scale by 95%. So
`run_forward` now forms `bf16(x + residual_old)` for the fused norm's residual slot and hands it
to the twin as the snapshot too, because the walk's own snapshot of that read is the live post-op
pull. (The earlier trial of this rule was measured against the polluted `h.q8` line and recorded
as refuted; that reading was wrong, and this section replaces it.)

**The forward is now the device's, to bf16 noise.** Each instrument fix moved the comparison, in
this order -- views, the cold `kv:` rule, the first-read rule, `SNAPDIFF`, the residual sum:

| line | before the views | after the views | after the residual sum |
|---|---|---|---|
| `gdn_in` | *skipped by size* | 0.1772 | **0.00585** |
| `gdn_q` | 0.1940 | 0.1940 | **0.00677** |
| `gdn_k` | 0.1869 | 0.1869 | **0.00666** |
| `gdn_v` | 0.1726 | 0.1726 | **0.00601** |
| `gdn_beta` | 0.5481 | 0.0628 | **0.000776** |
| `gdn_kkt` | 0.1775 | 0.0711 | **0.00257** |
| `gdn_o` | 1.2258 | 0.2986 | **0.0110** |
| `gdn_o.q8` | 1.2060 | 0.3420 | **0.0155** |
| `gate_up` | 0.1877 | 0.1877 | **0.00617** |
| `ffn.q8` | 0.2895 | 0.2895 | **0.0132** |
| `SNAPDIFF h.q8` | -- | 0.1812 | **0.00550** |

Every honest line is now at bf16's own level (the quantiser's dequantisation error alone is
0.9%). The lines that stay large are the ones the dump's capture explains, not divergence:
`h` (283.7) and `lora.t.*` (9.0/1.3) are same-record reads captured post-op, `gdn_o`'s snapshot
(0.999) is op 7's output against the dump's op-8 output, and `kv:2`/`kv:3` (rel=1) are the
slice's zeros against the dump's outgoing state.

**With the forward matched, the remaining leaf gap is the ROWS.** The slice's leaf set norm moved
5.41991 -> 5.40683 (the walk's 5.16169891: ratio 1.0475, was 1.0501) -- the forward fix moved the
leaves by 0.2%, not 5%. Per leaf:

| weight | factor | walk l2 | slice l2 | ratio |
|---|---|---|---|---|
| `ssm_inz` | A | 0.43418 | 0.479440 | 1.104 |
| `ssm_inz` | B | 0.0279081 | 0.0423961 | 1.519 |
| `ssm_out` | A | 0.386223 | 0.408531 | 1.058 |
| `ssm_out` | B | 0.0213476 | 0.0223883 | 1.049 |
| `ffn_gate_up` | A | 3.20103 | 3.389926 | 1.059 |
| `ffn_gate_up` | B | 0.172384 | 0.293761 | 1.704 |
| `ffn_down` | A | 4.00152 | 4.169112 | 1.042 |
| `ffn_down` | B | 0.125314 | 0.130389 | 1.041 |

The two A factors of the *fused* weights (`ssm_inz` = qkv+z, `ffn_gate_up` = gate+up) and the
unfused B factors differ by 4-10%; the two fused B factors by 52-70%, which is the accounting
difference already described (the engine's dense `[N_total, R_total]` against the walk's
per-module diagonal blocks). The slice walks the **host** rows and the walk the **device** rows,
over the same `backward_block` and now the same forward, so what is left is a host-vs-device row
difference at block 37 -- the same comparison that agreed to 0.59% at block 63's
`q_proj.lora_A`. That, not the forward, is the next instrument's target.

Scope, unchanged and repeated because it is the easy thing to overclaim: the slice shares
`backward_block` with both walks, so agreement clears block 37's ROWS at that token count given
the walk's incoming cotangent -- never "the walk", never another token count's magnitude. At 512
the rows are 4.75% apart and the forward agrees; at 4096 the forward agrees and the rows are
within 14% (see "The 4096 leg"), which clears this block at both counts and points the hunt
upstream.

The cold-`kv:`-write fix was re-measured after it landed (window 39's first arm): the 512 slice's
`LEAF` and `LEAFSET` lines are **byte-identical** to the pre-fix run (norm 5.40682538), which is
what the row's code predicts -- `gdn_chunk_scan`'s host adjoint reads `call.fwd` only through its
read slots (`f[rCu]`, `f[rH0]`, `load_chunk`/`load_state`, `src/adjoint/host/gdn_scan.cpp:487-578`)
and takes the writes' cotangents from `call.cot`, so a written slot's snapshot is inert for it. The
fix closes a contract gap; it moves no number in this block.

## Tests

- Host tier serial (`ctest --test-dir build -L host`): **25/25** ctest entries, the binary's own
  `303 tests, 0 failed` (150 s; the 301-test run predates the view case, the 302-test run the
  cold-write case); the new
  `adjoint_registry_fills_follow_the_served_row` and the huge-norm clip check in
  `tests/host/optim.cpp`. One full-tier run at reduced priority (`nice -n 10`) failed
  `host_step_int8` and `host_step_int8_gradient_points_the_way_bf16_does` (both at layer 0's
  `in_proj_qkv.lora_A`); both pass isolated 5/5, and the same suite at normal priority is green,
  so it is recorded as a scheduling-sensitive flake, not a regression — the same shape as the
  earlier `train_host_cli_serves_the_int8_trunk` flake. The two runs were back to back at
  ~19:2x, after window 31 had closed and with no GPU window active.
- The slice runner's own cases (`slice_reference_linear_scale`,
  `slice_reference_residual_two_consumers`), `slice_operand_view_gathers_the_named_half` and
  `slice_reference_cold_kv_write_zeroes_the_snapshot` pass after every change to
  `src/slice_reference.h` (`R97_TEST_ONLY` runs, `1 tests, 0 failed` each). The view case pins the
  parser's view fields and the gather: a `[2,4]` buffer read as two `[2,2]` halves must give
  `{0,1,4,5}` and `{2,3,6,7}`, not the whole buffer twice. The cold-write case asserts the
  snapshots themselves (`run_forward` + `StoredSnapshots::at`) on an `add(kv:s, b) -> kv:s`
  record -- the shape `gdn_chunk_scan` has, where the write is plain because the record also reads
  the buffer -- and fails on `kv:s (snapshot, cold write)` (0.1918, the production) with the
  zeroing removed, which is the bracket that proves it pins the rule.
- The five slice runs after the instrument fixes (windows 33-37) all returned `rc=0` and are the
  measurements quoted above; each ran in its own held window with `pause` verified absent
  afterwards.
- `tests/fixtures/programs/oracle.json` was regenerated through its own gate
  (`R97_RECORD_ORACLE_FIXTURE`) because the program writer now emits the view fields and
  `reference_step_program_matches_the_live_trace_without_kernels` compares the live recording
  field-exactly. The diff is 76 added lines: the 8 view-carrying operands
  (`view_shape`/`view_stride`/`view_off` and their arrays) and nothing else. The first full-tier
  run after the writer change failed on that case alone — recorded here rather than passed over,
  because it was a real regression of that comparison, not a load flake.
- GPU tier in one held window: `adjoint_in_place_rows` (4 cases), `adjoint_lora_fixture`,
  `adjoint_quant_device`, `adjoint_qk_fixture` — all pass, 27.8 s.

## Files changed

`include/r97/adjoint.h`, `src/adjoint_registry.cpp`, `src/backward.cpp`, `src/optim.cpp`,
`src/adjoint/device/{kv_store,lora,plugin}.hip`, `src/adjoint/host/{kv_store,lora}.cpp`,
`tests/support/device_check.h`, `tests/gpu/adjoint_in_place.cpp` (new), `tests/gpu/device_step.cpp`,
`tests/host/{adjoint_registry_precedence,optim}.cpp`, `docs/STEPS-DEVICE.md`, and for the slice
runner: `src/adjoint_schema.h` (new), `src/slice_reference.h`, `tools/slice_serve.cpp`,
`src/join_common.h` (its copies of the schema helpers moved out), `src/step_device.cpp` (the
`R97_DUMP_BLOCK` constants pass now carries the in-place `kv:` priors and the program writer
carries `view_shape`/`view_stride`/`view_off`), `src/reference_step.cpp` (its program writer and
parser carry the same view fields), `tools/ops.cpp` (the recorder's writer too),
`tests/gpu/device_step.cpp` (the leaf print carries `l1`/`l2`), `tests/host/slice_reference.cpp`
(the view round-trip and cold-write cases), `tests/fixtures/programs/oracle.json` (regenerated
with the view fields).
