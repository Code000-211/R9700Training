# `r97_iqr_decode`: IQR codes to the int8 grid the transposed linear reads

The forward IQR GEMMs decode a weight where they read it, inside `libr4d`
(`r4d_gemm_iqr.hip`, `r4d_iqr_dec.h`, `r4d_iqr.h`); there is no host kernel row for them. The
backward needs the same weight as plain int8 for the transposed product `dX = dY · W`. This kernel
turns one stored weight's IQR codes into that grid and its scales, into caller-provided scratch,
for a range of output rows.

- Device: `src/adjoint/device/iqr_decode.hip`, entry points `r97_iqr_decode` and
  `r97_iqr_decode_wait` in `libr97.so`.
- Host reference: `r97_iqr_decode_host` in `src/iqr_decode.h`, the same arithmetic as the kernel
  inline, for tests and the host tier.
- Tests: `tests/host/iqr_decode.cpp` (synthetic forms against libquant's `grid_decode` /
  `grid_tcq_decode`) and `tests/gpu/iqr_decode.cpp` (the real container, every form, and the GEMM
  check).
- Reference for the arithmetic: libquant `libquant/lq_grid.h`, `grid_decode` and
  `grid_tcq_decode`.

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
`[row0, row0 + rows)`. `codes` and `scale` **start at `row0`**, so a chunked caller offsets those
two pointers by `row0 × row_bytes`; `lut`, `trellis.mad` and `scale.1` stay at the plane's base.
`row0` must be a multiple of `T`, and `rows` must be a multiple of `T` unless the range ends at the
weight's last row, so every workgroup decodes whole trellis runs and the tail-biting seam is inside
its own run.

The vocabulary projection is `248320 × 5120`; the head is decoded in chunks (the test uses 256
rows). A chunk is independent: the seam is taken from the last `ceil(state_bits / bits)` codes of
the run, which are in the chunk's codes.

## What it does not do

- **No transform.** The container stores `transform=fwht128`; the codes already decode to the grid
  in the rotated basis. The caller applies the transform to the activation (the `i8a8h` row does),
  exactly as the forward IQR GEMM does. Applying `fwht128` here would rotate the weight twice.
- **No `scale.col`.** The per-128-column factor is applied by the linear's epilogue, not folded
  into the int8 code. The kernel only reads its presence: for `rb > 1` it switches the rb rounding
  to the magic-number form libquant uses whenever a column factor or a trellis mad is present.
- **No `scale.1` folding into the code.** The second-level scale is applied to the row/128-block
  scale and rounded to bf16 before the code multiplies it, as libquant does.

For `rb > 1`, a block of `rb` rows shares the largest row scale in the block (libquant's
`rb_round`); the kernel carries that maximum across the block.

## Interface

```cpp
R97IqrDecodeDesc d{};
// fill rows_all, cols, row0, rows, bits, state_bits, run_rows, rb,
//      scale_dtype, scale_block_rows/cols, scale_row_bytes, scale2_*,
//      codes, codes_row_bytes, scale, scale2, lut, mad/mad_a/mad_b,
//      out_codes, out_scale
r97_iqr_decode(&d);       // asynchronous; 0, or 1 outside the rules
r97_iqr_decode_wait();    // hipDeviceSynchronize()
```

`r97_iqr_validate` is the rule list, and both the host and device entry points call it before
touching memory. The device descriptor is a **host** struct whose pointers are device pointers;
the kernel copies it by value.

## Layout of the launch

One thread per column, one workgroup per trellis run: `grid = (ceil(cols/256), ceil(rows/T))`,
block 256. Threads of a workgroup cover consecutive columns, so at each row they read consecutive
packed codes of the same row; a column walks its run sequentially, seeded from the run's tail.
There is no shared memory and no scratch beyond the two output buffers: the caller's scratch is
exactly `rows × cols` int8 plus `rows × cols/128` bf16.

## Measured

AMD Radeon AI PRO R9700 (gfx1201, 32 GB), `R97_DEVICE=0`, one queue; the numbers are the launch
and `r97_iqr_decode_wait`, best of five, with the planes uploaded once. `tests/gpu/iqr_decode.cpp`
prints them.

| weight | shape `[out, in]` | out | scales | decode |
|---|---|---:|---:|---:|
| `blk.0.ffn_gate_up.weight` (largest) | 5120 × 34816 | 170.0 MiB | 2.66 MiB | 1.41 ms (117.5 GiB/s) |
| `blk.0.ffn_down.weight` | 5120 × 17408 | 85.0 MiB | 1.33 MiB | 0.94 ms (88.7 GiB/s) |
| `blk.0.ssm_inz.weight` | 5120 × 16384 | 80.0 MiB | 1.25 MiB | 0.93 ms (84.0 GiB/s) |
| `blk.0.ssm_out.weight` | 5120 × 6144 | 30.0 MiB | 0.47 MiB | 0.36 ms (82.0 GiB/s) |
| `blk.3.attn_qg.weight` | 5120 × 12288 | 60.0 MiB | 0.94 MiB | 0.59 ms (98.8 GiB/s) |
| `blk.3.attn_k.weight` | 5120 × 1024 | 5.0 MiB | 0.08 MiB | 0.120 ms (40.8 GiB/s) |
| `blk.3.attn_v.weight` | 5120 × 1024 | 5.0 MiB | 0.08 MiB | 0.120 ms (40.8 GiB/s) |
| `blk.3.attn_output.weight` | 5120 × 6144 | 30.0 MiB | 0.47 MiB | 0.36 ms (81.2 GiB/s) |
| `output.k3.weight`, 256-row head chunk | 256 × 5120 | 1.2 MiB | 0.02 MiB | 0.116 ms (10.5 GiB/s) |
| `token_embd.weight`, 256-row head chunk | 256 × 5120 | 1.2 MiB | 0.02 MiB | 0.181 ms (6.7 GiB/s) |

A Mamba layer (blocks 0-2) decodes `ssm_inz`, `ssm_out`, `ffn_gate_up` and `ffn_down` in about
3.6 ms; an attention layer (block 3 onward) decodes `attn_qg`, `attn_k`, `attn_v`,
`attn_output`, `ffn_gate_up` and `ffn_down` in about 3.5 ms. Decoded one weight at a time, which
is what the backward does, the largest needs 173 MiB of scratch (170.0 + 2.66 MiB). The whole
vocabulary projection would need 1.184 GiB + 18.96 MiB; a 256-row head chunk needs 1.25 MiB +
20 KiB, which is why it is decoded in chunks. The small chunks are launch-bound, not
bandwidth-bound.

## The tests

`tests/host/iqr_decode.cpp` covers the forms synthetically: code widths 3-6, trellis table and
computed (`mad`) values, f32/bf16/e4m3 scales, a second-level scale, `rb` 8 and 16, L=12 and
16, a shorter last run, a partial workgroup, and a chunk against the whole decode; every case is
also checked against libquant's host decode when `R97_LIBQUANT` points at `libquant.so`.

`tests/gpu/iqr_decode.cpp` reads the real container (label `model`, `R97_MODEL_DIR`, default
`/srv/models/qwen3.8-27b-iqr-v4`) and, for one tensor of each of the container's ten IQR encodings
and each linear kind, decodes a chunk starting inside the tensor and the short last chunk, and
requires the device bytes to equal the host bytes exactly. It then runs radiance's
`gemm_iqr_nt_m64` over `blk.0.ssm_out.weight`'s original planes and `gemm_i8a8_m16` over the
decoded grid with the same activation, and requires the outputs to agree within bf16 tolerance
(worst relative error 0.000977 on the reference card); both rows use `i8a8h`, because the stored
grid is in the rotated basis.

The GEMM check needs `libr4d.so`. radiance makes that module only for a HIP build whose targets
hold RDNA4 (`libr4d/CMakeLists.txt`), so the build asks for it on those same terms: when it does,
`libr4d` joins radiance's `BUILD_COMMAND` and the gpu test gets its path as `R97_LIBR4D`; when it
does not, `R97_LIBR4D` is left unset and the test looks beside the device plugin, where the radiance
home puts both modules. The gpu test runs without a model container by skipping the container cases
with a note; the synthetic host tier still runs.
