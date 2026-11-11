# `attn_paged` and `attn_paged_gate_quant` on the card: the adjoints

Reference: the host adjoints of `attn_paged` (`src/adjoint/host/attention.cpp`,
`docs/adjoints/attention.md`) and, for the fusion, the host adjoints of its parts --
`quant_act_i8g` (`docs/adjoints/quant.md`), `mul` and `sigmoid`
(`docs/adjoints/elementwise.md`) -- composed in the order the card issues them. Kernel source
`src/adjoint/device/attention.hip`, test `tests/gpu/adjoint_attention.cpp`.

## 1. The two rows

The plugin registers

    R97_REGISTER_ADJOINT_F("attn_paged", device, attention_paged_device_adjoint, 2, 3, 4, 5, 6)
    R97_REGISTER_ADJOINT_F("attn_paged_gate_quant", device, attention_gate_quant_device_adjoint,
                           2, 3, 4, 5, 6)

both frozen at the operands the forward reads but does not differentiate: `block_table`,
`seqused`, `k_descale`, `v_descale`, `cu_seqlens` (`docs/adjoints/attention.md` §4). The contract
gives a device adjoint f64 operands, not device buffers, so a call uploads the values it reads,
allocates the scratch it writes, and frees all of it before it returns; the harness reports the
high-water mark.

The second row is the card's fusion: the attention, the output gate (`y = bf16(out · gate)`,
`gate` already passed through the sigmoid the producer applies) and the quantising producer for
the output projection (`codes = quant(y)`, `q8_scale = amax/127`). It writes three cotangents --
the query's and the pool's, into slots 0 and 1, and the gate's, into the slot named `gate` -- and
none for `q8_scale`, which is the constant the quantiser's adjoint treats it as.

## 2. The adjoint, tile by tile

The forward never stores the tokens-by-tokens weights and neither does the adjoint. Both passes
recompute them from the stored `q` and `kv_cache`, one warp to a tile, and never materialise a
`tokens × tokens` matrix: what a pass keeps is one warp's registers.

With `y = out[tok, h, ·]`, `ȳ = ∂L/∂y`, `kds`/`vds` the pool's descales, `G_j = vds · Σ_d ȳ[d]·V_j[d]`
and `Ḡ = Σ_j p_j G_j`:

    score_cot_j = p_j (G_j − Ḡ)                     the softmax's Jacobian, without Z
    dq[tok, h, d]  += Σ_j score_cot_j · scale · kds · K_j[d]
    dK_j[d]        += score_cot_j · scale · kds · q[tok, h, d]
    dV_j[d]        += p_j · vds · ȳ[d]

**`norm_dq_kernel`: a warp to a (query row, head)**, three passes over the row's visible keys, kept
in ascending order:

1. the dot products `q·K_j`, and in registers the running softmax maximum `mx` and sum `z`, so the
   row needs no second read of the keys to normalise;
2. `G_j` for each key, summed in f32 into `Ḡ`;
3. `score_cot_j` and `dq`, a tile of `head_dim` per lane: lane `l` holds dims `l + 32t` of the
   query, of `dq`, and of each key as it streams past.

`mx`, `z` and `Ḡ` go to scratch: the pool's rows are accumulated key by key, so `Ḡ` -- a mean over
a query row's keys -- cannot be recomputed there, and a row the mask leaves empty writes `mx = 0`,
`z = 1`.

**`pool_dkv_kernel`: a warp to a (sequence, key, key-value head)**, which streams the query rows
that read the key (`qlo`/`qhi` from the mask, the `gqa` query heads of the key-value head) and
keeps one pool row's `dK` and `dV` in registers across all of them, adding them with `atomicAdd`
once. A pool row no query read -- a block a table entry leaves at −1, a row past a sequence's last
position -- keeps the zero it started with, which is how the quantised pool's unused rows stay
untouched.

`Ḡ` is recomputed from the keys rather than read back as `<ȳ, out>`. The two are equal in exact
arithmetic, but the fusion multiplies the value's cotangent by `1/sigmoid(gate)` before it reaches
this kernel, and `out` is stored at bf16: at the fixture's geometry a bf16 `out` carries the fused
cotangent to about 0.2 % and `ȳ·out` then decides `score_cot` outright for a row whose weights are
nearly uniform. Recomputing keeps the kernel on the same arithmetic the host adjoint does.

Everything accumulates in f32 -- `dq` and the pool's rows alike -- and is rounded once to bf16 on
the way out: the query's in the kernel, whose accumulator is bf16 as the forward's query is, and
the pool's per row in `collect`, where the operand's dtype decides. The pool is accumulated with
atomics, so it is zeroed before the pass.

## 3. The fusion's adjoint

`gate_ste_kernel` runs first, one element to a thread, and turns the fusion's two output
cotangents into the attention's and the gate's. With `s = sigmoid(gate)`, `a = y/s` the attention's
own output, `q = y/q8_scale` the quantiser's grid position, and `live` the clamp's mask (zero where
`|q| > 127`, the top of the grid):

    ytot    = ȳ(out) + live · cot(q8) / q8_scale       the straight-through step
    d(a)    = ytot · s                                 the gate's mul, into the attention
    d(gate) = ytot · y · (1 − s)

then the attention adjoint above runs with `d(a)` in place of `ȳ`. Two details come from the host
composition and are reproduced exactly:

- The clamp's mask is decided by the *stored* pair: the f64 quotient of the stored bf16 `y` and the
  stored f32 `q8_scale`, compared with 127 as the host adjoint compares it. A group's largest value
  sits on the top of the grid to within the rounding of its scale, so an f32 comparison with slack
  picks the boundary element differently from the host and puts a full-sized cotangent, or none,
  on it -- 9 of the fixture's 1536 elements, and 2.7 % of every query row those 9 touch.
- `q8_scale`'s own cotangent is not asked for: the plan makes a scale a constant
  (`docs/adjoints/quant.md`), so this kernel writes nothing for it.

## 4. What the test covers

`tests/gpu/adjoint_attention.cpp` runs `device_check::check` -- the card against the host on the
same random inputs, the `2 % + 1e-3` bf16 tolerance of `tests/support/device_check.h`, every slot,
and the scratch fully freed -- for both rows:

| case | q / kv heads | head_dim | sequences | block size | rows |
|---|---|---|---|---|---|
| fixture, one sequence | 4 / 2 | 32 | 12 | 8 | 12 |
| fixture, three sequences | 4 / 2 | 32 | 5, 9, 14 | 8 | 28 |
| fixture, full context | 4 / 2 | 32 | 4096 | 8 | 4096 |
| real, one sequence | 24 / 4 | 256 | 1024 | 16 | 1024 |
| real, unequal batch | 24 / 4 | 256 | 128, 256, 640 | 16 | 1024 |
| real, full context | 24 / 4 | 256 | 4096 | 16 | 4096 |

The first three and the two 1024-token cases run against the fused row as well. The fixture's
geometry is `tests/support/fixture.h`; the real one is a Qwen3.8-27B block's grouped query
attention (hidden 5120, 24 query heads over 4 key-value heads of dimension 256). Each sequence's
pages are its own blocks, laid out one sequence after another, with one block the step never filled
left at −1 in its table entry.

4096 tokens is a full context in one sequence: at the fixture's head geometry, and at the real one
for `attn_paged`, whose reference there takes minutes (its f64 twin alone is 103 s a pass). The
fusion at the real geometry and 4096 tokens is timed under `--bench` rather than compared, since
its reference is that same host adjoint again; its own work is the gate's straight-through pass,
linear in the tokens, and is compared at the shapes above.

A run of the test takes a minute and a half for the ten cases up to 1024 tokens and five more for
the 4096-token real-geometry case, which is minutes of f64: the twin fills the forward's writes and
the reference runs it again, 103 s a pass. `--bench` adds the fusion's 4096-token timing, one more
pass of that twin.

The fusion has no host row of its own, so its reference is `fused_host_reference` in the test, a
`HostRef`: the host adjoints of `quant_act_i8g`, `mul` and `sigmoid` around the host adjoint of
`attn_paged`, on the fusion's own record, and its forward is the twin the test registers
(`fused_twin`, the three host twins in the same order). The twin stores what the record stores --
`out` at bf16, `q8_scale` at f32 -- because the adjoint's mask is decided by those values.

Every case also runs against a deliberately wrong kernel: the honest one with the pool's cotangent
left empty, which the harness must reject in the pool's slot.

## 5. Time and scratch

One R9700 (gfx1201), `R97_DEVICE=1`, from `./build/r97_gpu_tests --bench` on an otherwise idle
card, one call per measurement (`reps` is 1 above 64 rows; the small shapes use 20):

| shape | rows | adjoint | forward twin | adjoint / twin | scratch peak |
|---|---|---|---|---|---|
| fixture, one sequence | 12 | 2.97 ms | 0.02 ms | 122.9x | 31 KiB |
| fixture, three sequences | 28 | 4.35 ms | 0.05 ms | 89.8x | 77 KiB |
| fixture, full context | 4096 | 100.2 ms | 3713.2 ms | 0.03x | 9.3 MiB |
| real, one sequence, 1024 | 1024 | 167.9 ms | 5177.9 ms | 0.03x | 76.3 MiB |
| real, unequal batch, 1024 | 1024 | 112.1 ms | 1988.4 ms | 0.06x | 76.3 MiB |
| real, one sequence, 4096 | 4096 | 1250.1 ms | 103199.1 ms | 0.01x | 305.2 MiB |
| real, one sequence, 1024, fused | 1024 | 246.9 ms | 5906.0 ms | 0.04x | 196.5 MiB |
| real, unequal batch, 1024, fused | 1024 | 218.5 ms | 2444.8 ms | 0.09x | 196.5 MiB |
| real, one sequence, 4096, fused | 4096 | 1748.5 ms | 138185.2 ms | 0.01x | 785.9 MiB |

The two small shapes are fixed cost: one call there is a few hundred kilobytes of upload, a launch
per row-group and one synchronisation, about 2.3 ms that the forward twin does not pay. From the
fixture's 4096-token row upwards the card wins, and at the real geometry it wins by 30x; the
adjoint's cost is linear in the tokens times their average reach, which is why the 1024-token
sequence (one full causal block) costs the card more than the 4096-token fixture (24 query heads of
256 against 4 of 32). The fusion is the plain row plus the gate's straight-through pass and three
more operands to upload: 50 to 110 ms at these shapes, and at 4096 tokens 1.75 s against the plain
row's 1.25 s. Its scratch is the three extra operands wide: 196.5 MiB against 76.3 MiB at 1024
tokens, 785.9 MiB against 305.2 MiB at 4096.

The line is `tests/support/bench.h`'s: the adjoint's time per call and, beside it, the same
harness's forward twin on the same shape. That twin is the f64 CPU reference, not the card's
forward kernel -- this worktree has no card forward to reach from a device-adjoint test -- so the
measurement the issue asks for is the first column, and the ratio only says how the adjoint's cost
grows with the shape.

Scratch is the high-water mark of one call, and it is the operands and the two passes' tables, not
a weights matrix: `q` and `kv_cache` widened to f32 (the contract hands the adjoint f64 values and
the kernels read f32), the pool's f32 cotangent, the record's integers -- the block table, the
sequence lengths, the descales -- the row's visible-key interval, the key-to-page map and the query
interval per key, and per (query row, head) the softmax's max and sum and the row's mean value
cotangent. The fusion adds its gate, the gated output and the codes' cotangent.

## 6. What it does not cover

- The forward's real geometry is `24/4/256` at up to 4096 tokens; the test compares that geometry
  at 1024 tokens as one sequence and as an unequal batch, and as 4096 tokens in one sequence, and
  the fusion of the 4096-token case is timed rather than compared (section 4). No other length is
  run, and 4096 tokens as a batch of unequal lengths is not run at all.
- The pool the test hands the kernel is bf16, and both cotangents are rounded to it once, on the
  way out: the query's in the kernel (its accumulator is bf16) and the pool's in `collect`. For any
  other pool dtype the pool's cotangent comes back at f32, as the host adjoint's does -- an fp8
  pool's own rounding is the `qk` family's, not this one's.
- `attn_paged`'s sliding window and non-causal variants are in the host adjoint and in `resolve`;
  the test exercises the causal, unmasked path only, which is what training issues.
- A row with no visible key is a zero row here as it is in the host adjoint; the test's sequences
  all have at least one visible key per row.
