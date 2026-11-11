# `r97_iqr_decode`: IQR codes to the int8 grid the transposed linear reads

The forward IQR GEMMs decode a weight where they read it, inside `libr4d`
(`r4d_gemm_iqr.hip`, `r4d_iqr_dec.h`, `r4d_iqr.h`); there is no host kernel row for them. The
backward needs the same weight as plain int8 for the transposed product `dX = dY · W`. This kernel
turns one stored weight's IQR codes into that grid and its scales, into caller-provided scratch,
for a range of output rows.

- Device: `src/adjoint/device/iqr_decode.hip`, entry point `r97_iqr_decode` in `libr97.so`.
- Host reference: `r97_iqr_decode_host` in `src/iqr_decode.h`, the same arithmetic as the kernel
  inline, for tests and the host tier.
- Tests: `tests/host/iqr_decode.cpp` (synthetic forms of both kinds against libquant's
  `grid_decode` / `grid_tcq_decode`) and `tests/gpu/iqr_decode.cpp` (the real container, every IQR
  form, and the GEMM check).
- Reference for the arithmetic: libquant `libquant/lq_grid.h`: `grid_tcq_decode` for the trellis
  forms and `grid_decode` for the affine one.

## What it reads and writes

A weight is stored as the container's planes in the layout `[out, in]` the linear's own matrix
has: row `r` is one output channel, column `j` one input. The kernel reads the `codes` plane and
the scale planes, and writes

- `out_codes`: `rows × cols` int8, one code per element, the **same row-major orientation** as the
  stored plane. The transposed linear reads it as `[cols, rows]` with the indexing its transpose
  needs; no transpose is materialised here.
- `out_scale`: bf16, one per row per 128 columns (`rows × cols/128`), the scale the int8 code
  multiplies. A row-wide scale (`scale` with a whole-row block) is replicated into each 128-column
  group, which is what the `w8a8` GEMM reads.

`R97IqrDecodeDesc` carries the plane pointers, their formats and blocks, the run length (`T`, the
trellis's block rows), `rb`, the code width and the trellis state width, and the row range
`[row0, row0 + rows)`. `codes`, `scale` and (in the affine form) `zero` **start at `row0`**, so a
chunked caller offsets those pointers by `row0 × row_bytes`; `lut`, `trellis.mad` and `scale.1`
stay at the plane's base, and the affine form has none of the three.
`row0` must be a multiple of `T`, and `rows` must be a multiple of `T` unless the range ends at the
weight's last row, so every workgroup decodes whole trellis runs and the tail-biting seam is inside
its own run. The affine form's `T` is 1, so any range is a whole number of runs.

The vocabulary projection is `248320 × 5120`; the head is decoded in chunks (the test uses 256
rows). A chunk is independent: the seam is taken from the last `ceil(state_bits / bits)` codes of
the run, which are in the chunk's codes.

## The two forms

radiance stores two kinds of IQR grid in the first container, and one descriptor decodes both:

- **Trellis** (the trunk, `transform=fwht128`): `codes` (u3..u6) with `scale` and an optional
  `scale.1`, and a `trellis` table or a `trellis.mad` pair to compute the table from. A column is a
  tail-biting trellis of `state_bits` bits down its run of `T` rows: each row's code is looked up
  from the state and the row's code, and the state carries across the run. libquant's
  `grid_tcq_decode`.
- **Affine** (`dflash.draft_head.weight`, the container's one affine tensor): u2 `codes` and a u8
  `zero` on the same 128-column block as an f16 `scale`, and nothing above a row. An element is
  `(code - zero) * scale` — libquant's `grid_decode` for an asymmetric grid with no table — and the
  difference is already on the int8 grid, so the scale is all that is left. A row is a complete
  run: the descriptor sets `affine`, `zero`, `zero_bits` and `zero_row_bytes`, `run_rows` and `rb`
  are 1, and `state_bits`, `lut`, `mad` and `scale2` stay clear, so the kernel's launch needs no
  second shape.

## What it does not do

- **No transform.** The container stores `transform=fwht128`; the codes already decode to the grid
  in the rotated basis. The caller applies the transform to the activation (the `i8a8h` row does),
  exactly as the forward IQR GEMM does. Applying `fwht128` here would rotate the weight twice.
- **No `scale.col`.** The per-128-column factor is applied by the linear's epilogue, not folded
  into the int8 code. The kernel only reads its presence: for `rb > 1` it switches the rb rounding
  to the magic-number form libquant uses whenever a column factor or a trellis mad is present.
- **No `scale.1` folding into the code.** The second-level scale is applied to the row/128-block
  scale and rounded to bf16 before the code multiplies it, as libquant does.
- **The affine form has no `gemm_iqr_*` to compare against.** `R4D_IQR_FORMS` in
  `libr4d/r4d_iqr.h` carries a `zero` plane only for its 6-bit form, so radiance's forward GEMM
  never runs a 2-bit affine weight and there is no forward decode of the drafter's head to compare
  with. The affine form is therefore checked device against the host (libquant's `grid_decode`),
  not against a forward GEMM over the same codes; the GEMM check below is the trellis path's.

For `rb > 1`, a block of `rb` rows shares the largest row scale in the block (libquant's
`rb_round`); the kernel carries that maximum across the block.

## Interface

```cpp
R97IqrDecodeDesc d{};
// fill rows_all, cols, row0, rows, bits, state_bits, run_rows, rb,
//      scale_dtype, scale_block_rows/cols, scale_row_bytes, scale2_*,
//      codes, codes_row_bytes, scale, scale2, lut, mad/mad_a/mad_b,
//      out_codes, out_scale
// the affine form instead sets affine = 1, zero, zero_bits and zero_row_bytes,
// leaves state_bits, lut, mad and scale2 clear, and has run_rows = rb = 1
r97_iqr_decode(&d, stream);   // asynchronous on `stream`; 0, or 1 outside the rules
rad_stream_sync(stream);      // the caller's wait, on the stream it handed in
```

`r97_iqr_validate` is the rule list, and both the host and device entry points call it before
touching memory. The device descriptor is a **host** struct whose pointers are device pointers;
the kernel copies it by value.

The kernel launches on the `RadStream` the caller hands in (`R97IqrStream`, radiance's opaque
`void*`), with `hipLaunchKernelGGL`, and does not sync: the caller's `rad_stream_sync` on that
same stream orders its read of the outputs after the launch, so the uploads, the decode and the
downloads are one queue's work on the caller's stream rather than the whole device's. A caller
with the codes on another stream hands that stream in; there is no hidden legacy stream.
`tests/gpu/iqr_decode.cpp` is the caller that shows it: one stream for its uploads, the launch
and its downloads.

## Layout of the launch

One thread per column, one workgroup per trellis run: `grid = (ceil(cols/256), ceil(rows/T))`,
block 256. Threads of a workgroup cover consecutive columns, so at each row they read consecutive
packed codes of the same row; a column walks its run sequentially, seeded from the run's tail. An
affine weight is the same launch with `T = 1`: a workgroup is one row and a thread its columns.
There is no shared memory and no scratch beyond the two output buffers: the caller's scratch is
exactly `rows × cols` int8 plus `rows × cols/128` bf16.

## Measured

AMD Radeon AI PRO R9700 (gfx1201, 32 GB), one card and one queue; the numbers are the launch
and the caller's `rad_stream_sync` on that queue, best of five, with the planes uploaded once.
`tests/gpu/iqr_decode.cpp` prints them.

| weight | shape `[out, in]` | out | scales | decode |
|---|---|---:|---:|---:|
| `blk.0.ffn_gate_up.weight` (largest) | 5120 × 34816 | 170.0 MiB | 2.66 MiB | 1.47 ms (113 GiB/s) |
| `blk.0.ffn_down.weight` | 5120 × 17408 | 85.0 MiB | 1.33 MiB | 0.97 ms (86 GiB/s) |
| `blk.0.ssm_inz.weight` | 5120 × 16384 | 80.0 MiB | 1.25 MiB | 0.97 ms (81 GiB/s) |
| `blk.0.ssm_out.weight` | 5120 × 6144 | 30.0 MiB | 0.47 MiB | 0.39 ms (75 GiB/s) |
| `blk.3.attn_qg.weight` | 5120 × 12288 | 60.0 MiB | 0.94 MiB | 0.63 ms (92 GiB/s) |
| `blk.3.attn_k.weight` | 5120 × 1024 | 5.0 MiB | 0.08 MiB | 0.138 ms (35 GiB/s) |
| `blk.3.attn_v.weight` | 5120 × 1024 | 5.0 MiB | 0.08 MiB | 0.138 ms (35 GiB/s) |
| `blk.3.attn_output.weight` | 5120 × 6144 | 30.0 MiB | 0.47 MiB | 0.38 ms (78 GiB/s) |
| `output.k3.weight`, 256-row head chunk | 256 × 5120 | 1.2 MiB | 0.02 MiB | 0.137 ms (8.9 GiB/s) |
| `token_embd.weight`, 256-row head chunk | 256 × 5120 | 1.2 MiB | 0.02 MiB | 0.201 ms (6.1 GiB/s) |
| `dflash.draft_head.weight`, 8192-row chunk (affine) | 8192 × 5120 | 40.0 MiB | 0.62 MiB | 0.85 ms (46 GiB/s) |

A Mamba layer (blocks 0-2) decodes `ssm_inz`, `ssm_out`, `ffn_gate_up` and `ffn_down` in about
3.8 ms; an attention layer (block 3 onward) decodes `attn_qg`, `attn_k`, `attn_v`,
`attn_output`, `ffn_gate_up` and `ffn_down` in about 3.7 ms. Decoded one weight at a time, which
is what the backward does, the largest needs 173 MiB of scratch (170.0 + 2.66 MiB). The whole
vocabulary projection would need 1.184 GiB + 18.96 MiB; a 256-row head chunk needs 1.25 MiB +
20 KiB, which is why it is decoded in chunks. The small chunks are launch-bound, not
bandwidth-bound.

The drafter's head is the same `248320 × 5120` shape in the affine form, so its whole width needs
the same 1.184 GiB + 18.96 MiB and it is chunked the same way: the 8192-row chunk above is 40.0
MiB + 0.62 MiB and runs at about half the trellis rate, because a 2-bit affine row is a short run
(a row, decoded by one workgroup) whose codes are two bits an element, so the int8 write dominates
and there is no trellis walk to spread a thread's work over. The last 512 rows are the short tail
of the range, checked as its own chunk.

## The tests

`tests/host/iqr_decode.cpp` covers the forms synthetically: code widths 3-6, trellis table and
computed (`mad`) values, f32/bf16/e4m3 scales, a second-level scale, `rb` 8 and 16, L=12 and
16, a shorter last run, a partial workgroup, and a chunk against the whole decode; every case is
also checked against libquant's host decode when `R97_LIBQUANT` points at `libquant.so`. The
affine form has its own cases: u2 codes under a u8 zero with an f16 scale (the container's
shape), a u4 zero to exercise a packed zero plane, and bf16, a 100-row chunk at `row0 = 128`, and
refusal of a descriptor without its zero. Each affine case is compared value by value with
libquant's `grid_decode` over a constructed `Grid{asym}` and `BlockScale{s, zero}`.

`tests/gpu/iqr_decode.cpp` reads the real container (label `model`, `R97_MODEL_DIR`, default
`/srv/models/qwen3.8-27b-iqr-v4`) and, for a tensor of each of the container's trellis encodings
and each linear kind, decodes a chunk starting inside the tensor and the short last chunk, and
requires the device bytes to equal the host bytes exactly. The container's one affine tensor,
`dflash.draft_head.weight` (u2 codes, u8 zero, f16 scale), gets the same treatment for an 8192-row
chunk and its last 512 rows, and a timed pass over the chunk. It then runs radiance's
`gemm_iqr_nt_m64` over `blk.0.ssm_out.weight`'s original planes and `gemm_i8a8_m16` over the
decoded grid with the same activation, and requires the outputs to agree within bf16 tolerance
(worst relative error 0.000977 on the reference card); both rows use `i8a8h`, because the stored
grid is in the rotated basis.

The GEMM check needs `libr4d.so`. radiance makes that module only for a HIP build whose targets
hold RDNA4 (`libr4d/CMakeLists.txt`), so the build asks for it on those same terms: when it does,
`libr4d` joins radiance's `BUILD_COMMAND`, and the test finds it beside the device plugin, where
the radiance home puts both modules (`R97_LIBR4D` overrides the path); when it does not, no target
is named and the test fails by name rather than silently comparing nothing.

Because it needs real tensors it is a `model` test, `R97_MODEL_TEST(iqr_decode)`, registered from
`tests/support/model_case.h`: it resolves the container before it takes the card and skips with
exit 77 when the container is absent. The gpu binary is started twice, and the two runs differ by
one environment variable. ctest runs it as `gpu_model_tests` (`LABELS gpu;model`, `R97_MODEL_CASES=1`
and `SKIP_RETURN_CODE 77`), never as the plain `gpu_tests` run, which is the card's own cases and
must not skip. Directly: `R97_MODEL_CASES=1 ./build/r97_gpu_tests`, which is also where the timings
above come from. The forms a card can check without the container are covered synthetically in the
host tier, above.

That the second label needs the card as well has a consequence for the shared machine: `ctest -L
model` includes `gpu_model_tests`, so on a machine with the container and no card it fails on the
card guard. Run `ctest -L model -E gpu_model_tests` there, or take the whole thing on the card.
