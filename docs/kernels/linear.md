# Device adjoints: `linear`

The frozen linears on the card. The rows are registered by `src/adjoint/device/linear.hip`, which
compiles into `libr97` as HIP when `R97_WITH_HIP` is not `OFF`; the checks are
`tests/gpu/adjoint_linear.cpp` and the arithmetic they implement is stated in
[`docs/adjoints/linear.md`](../adjoints/linear.md), whose host rows they stand beside.

| op | reads | writes | frozen | served |
| --- | --- | --- | --- | --- |
| `gemm_nt_q` | `a`, `b`, `b_scale` | the cotangent of `a` | `a_scale`, `b`, `b_scale` | `dtype` `i8a8h` at `group` 128 |
| `gemm_nt` | `a`, `b` | the cotangent of `a` | `b` | `dtype` `bf16` |

Each row writes only the cotangent of the activation operand, and adds it into the slot the loader
hands over (`call.dc[ia]`), so a tape walk sees the same `+=` the host row does. A record the
predicate declines -- another `dtype`, another `group` -- gets no device row at all and keeps
whatever row the host or another plugin provides; nothing falls back silently inside the kernel.

`logits_gemm` is not covered here. Its arithmetic is `gemm_nt`'s, but the vocabulary edge is not one
of the linears this issue names, and its own operands (`x`, `lm_head`) would need a second pair of
schema positions; a later change can register the name against the same kernel.

## The arithmetic

`gemm_nt_q` at `i8a8h`: the forward is `y[m,n] = sum_g s_b[n,g] * sum_{k in g} a_code[m,k] *
b_code[n,k]` with `g = k / 128`. The weight is frozen, so the adjoint in the dequantised activation
basis is

```text
dx[m,k] += sum_n dy[m,n] * b_code[n,k] * s_b[n, k/128]        dy: [M,N]  b: [N,K]  dx: [M,K]
```

The scale varies along the reduction as well as along `k`, so the sum is not one product: a 128-wide
block of `dx` is the product of the gradient with that block's codes, each column of it scaled by its
own `s_b[:, block]`. One CTA owns one block, so `k`'s block index is the CTA's `blockIdx.y` and every
lane holds the eight scales its own fragment columns need.

`gemm_nt` is the same kernel with the weight already bf16 and no scale to apply.

## Precision

Precision is the one this issue decides: int8 forward, bf16 backward. The codes are exact in bf16,
so the products run on the bf16 WMMA path (`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12`) and
a token's sum over the reduction accumulates in f32. `dy` arrives bf16 and is multiplied by its
scale in f32, then rounded once to bf16 before the product -- the one rounding the kernel adds to
the host adjoint's f64 sum, and the reason the two are compared by tolerance rather than for
equality: each term carries a relative error up to 2^-9, so a heavily cancelling sum can show a
larger relative error than any single term.

The test states the tolerance it holds the kernel to: the L2 norm of the difference over the norm of
the reference, at most `1e-2`, which is `bf16_tol` in the test file. At the fixture's shapes it is
the whole of `dx`; at the model's shapes, where a full f64 product would be hundreds of GMAC on one
thread, it is 512 spread `(m,k)` probes against the same f64 sum the host adjoint forms.

## The tile

One CTA holds a 256-row band of `dx` and one 128-wide block of `k`: eight waves, each wave two
16-row tiles, and all eight `k` steps of the block in flight as 16 accumulators of 8 f32 (128 VGPRs).
The reduction runs in 16-channel steps; each step's 16 x 128 weight slice is staged in shared memory
in its own `[n][k]` order (`16 * 136` bf16, a 136-element pitch so a thread's eight-element chunk
lands 8-byte aligned), and read as the transposed fragment: WMMA wants the B operand's rows on `dx`'s
columns, which is the transpose of the natural weight order, and eight scalar LDS reads are cheaper
than transposing the tile during staging (which would turn each contiguous eight-element store into
eight scattered two-byte ones).

`dy` is `[M][N]` and K-contiguous in the order the A fragment wants, so it is read straight from
global with the same fragment helper the forward's tiled kernel uses; the scaled fragment is then
multiplied into every `k` step of the block. Blocks are dispatched with `M` fastest (`grid.x`), so
the slice for one block of `k` -- `N * 128` bytes, 0.65 MB at the model's 5120 -- is re-read by
consecutive CTAs and stays in the cache. `dy` is read once per block of `K`, `K/128` times per call;
that read is the dominant traffic of the kernel, which is why the split of `k` into CTAs is over
scale blocks and not over something narrower.

The store goes out as scalars: lane `l` holds column `l & 15` of a tile and rows `8 (l >> 4) + e`, so
sixteen lanes write 32 contiguous bytes of one row of `dx` per step, and the rows and columns past
`M` and `K` are dropped rather than written.

## Shapes

| shape | `M` | `N` | `K` |
| --- | --- | --- | --- |
| fixture, `gemm_nt_q` | 4 | 256 | 128 |
| fixture, `gemm_nt_q` | 1 | 128 | 256 |
| fixture, `gemm_nt_q` | 7 | 320 | 384 |
| fixture, gated-delta-net projection | 3 | 8 | 128 |
| fixture, `gemm_nt` | 3 | 8 | 128 |
| fixture, `ssm_out` | 2 | 128 | 64 |
| `attn_k/v` | 4096 | 1024 | 5120 |
| `attn_output` | 4096 | 5120 | 6144 |
| `attn_qg` | 1024, 4096 | 12288 | 5120 |
| `ssm_inz` | 4096 | 16384 | 5120 |
| `ffn_down` | 4096 | 5120 | 17408 |
| `ffn_gate_up` | 1024, 4096 | 34816 | 5120 |
| `ffn_gate_up`, a packed batch of three unequal sequences | 4023 | 34816 | 5120 |
| `ssm_out`, bf16 | 4096 | 5120 | 6144 |

The two 4096-token bands and the packed batch of 317, 1009 and 2697 rows are what a step's token
count looks like; the widths are the ones [`docs/ADAPTER.md`](../ADAPTER.md) writes down for
`qwen3.8-27b-iqr-v4`. No `model`-labelled test is needed for them: the rows take their shapes from
the record, and the container is what the forward measured beside them was run from.

Limits, each checked rather than assumed, and each leaving the record to the host row:

- on the int8 grid `group` must be 128, the width of the block a CTA owns (another grid is
  another kernel); the bf16 form carries no scale grid;
- `K` must be a positive multiple of 8, so that an eight-element chunk of codes is 4-byte aligned and
  the last chunk can be clamped to end on `K`;
- `M * N` and `M * K` must fit a 32-bit element offset -- 1.4e8 elements at the largest shape above;
- `N` is padded up to a multiple of 16 with zero weight rows and zero `dy` columns, because the
  reduction tile is 16 channels wide. The fixture's eight- and twelve-channel projections take that
  path, and the padded rows cannot add to the sum.

## Scratch memory

One call allocates and frees, at `Np` the padded `N` and `G = K/128`: `2*M*Np` bytes of bf16 `dy`,
`2*Np*K` of weights (one byte per code, or two for the bf16 form), `2*Np*G` of bf16 scales, and
`2*M*K` of bf16 `dx` results, which the host adds into the cotangent slot. That is 53.1 MiB for
`attn_k/v` at 4096 rows, 250.7 MiB for `ffn_gate_up` at 1024 rows and 484.7 MiB for `ffn_gate_up` at
4096 -- arithmetic on the formula, not a measurement, and the number the test prints is the sum of
the allocations the call actually made.

## Measurement

**Not measured here.** The run that wrote this file had no card it could open: `HIP` reported no
ROCm-capable device (`/dev/kfd` is present and answers `EACCES` even to root, so the container the
work was done in filters the device), so `tests/gpu/adjoint_linear.cpp` compiled and linked but its
two card tests could not execute and no timing below is filled in. The rows themselves are checked
without a card: the test's first test reads the registry and the predicates, and passes.

Fill the table with

```console
$ cmake -B build -G Ninja && cmake --build build -j
$ build/r97_gpu_tests                     # R97_DEVICE=<n> for a card other than 0
```

which prints one line per shape, in this order, as `staging`, `kernel`, TFLOP/s and `scratch`:

| shape | staging (host conversion + transfer) | kernel | TFLOP/s | scratch |
| --- | --- | --- | --- | --- |
| `attn_k/v` 4096 x 1024 x 5120 | | | | |
| `attn_output` 4096 x 5120 x 6144 | | | | |
| `attn_qg` 1024 x 12288 x 5120 | | | | |
| `attn_qg` 4096 x 12288 x 5120 | | | | |
| `ssm_inz` 4096 x 16384 x 5120 | | | | |
| `ffn_down` 4096 x 5120 x 17408 | | | | |
| `ffn_gate_up` 1024 x 34816 x 5120 | | | | |
| `ffn_gate_up` 4096 x 34816 x 5120 | | | | |
| `ffn_gate_up` batch, 4023 rows | | | | |
| `ssm_out` bf16 4096 x 5120 x 6144 | | | | |

`staging` is the host's f64 -> bf16 and f64 -> int8 conversion of `dy` and the weight, the scale
grid, and the uploads, and `kernel` runs from the launch to `hipDeviceSynchronize`, so the second
includes the download of `dx`. Each shape is called twice and the second call is the one reported,
so the code object's first load is not in the number.

The forward op's time per call, for the same shapes on the same run, is not in this file either: it
is a radiance launch, and the harness that times a forward op beside its adjoint is the one the
device-kernel plugin issue brings. Until it exists the comparison has to be made by hand, from the
forward's own timing on the same card and the same shapes, and the number it should be compared with
is `staging + kernel`, since a backward pass pays both.
