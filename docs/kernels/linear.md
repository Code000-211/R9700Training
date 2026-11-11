# Device adjoints: `linear`

The frozen linears on the card. The rows are registered by `src/adjoint/device/linear.hip`, which
is one of the kernels the device plugin builds into `libr97.so`
(`src/adjoint/device/README.md`); the checks are `tests/gpu/adjoint_linear.cpp`, built on the
harness every device adjoint uses (`tests/support/device_check.h`) for its case, card guard and
lookup, with this family's own comparison in the test file beside them, and the arithmetic they
implement is stated in [`docs/adjoints/linear.md`](../adjoints/linear.md), whose host rows they
stand beside.

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

The test holds the kernel to a tolerance with one term added to the harness's: every element it
compares is within 2% of the host row's value plus `1e-3` plus a tenth of the RMS of the elements
compared, and the relative L2 over those elements is under `bf16_tol` = `1e-2` in the test file. The
RMS term is there because a sum over hundreds of terms carries the same absolute rounding error
whatever it cancels to, so a rule relative to the element alone would fail a kernel that is right
for an element that came out near zero.

That comparison is the family's own, `Sample` and `compare` in `tests/gpu/adjoint_linear.cpp`,
rather than something the shared harness grew: `CONTRIBUTING.md` asks a family to add files and
change no shared one, and `tests/support/device_check.h` is the header every device family compares
through. What this family needs beyond the shared rule -- the RMS term, the rows a case samples, the
case that skips the twin, and the relative L2 in the report -- therefore lives in the test file,
which keeps the harness's case, card guard, plugin load and constants as they are. The rest of this
section and the next say what the family's comparison adds.

At the fixture's shapes, and at the model's hidden width with sixteen rows, the whole of `dx` is
compared. At the model's real shapes no test can pay for the host row of the whole tensor -- 4096
rows of the widest linear is about seven hundred GMAC of f64 on one thread, and the host row's inner
loop strides the weight, which makes it slower still -- so the case names rows instead
(`Sample::rows`) and the reference is the host adjoint, the registry's own host row, on a record of
just those rows, with the twin skipped because the adjoint reads none of the slots the forward wrote.
The sample is the first row, two spread through the tensor, the two rows a 256-row band turns over
at, and the last two, so a partial band and a batch's tail are in it; the rows are compared whole,
so every block of `K` is, and the seven rows are held to the same tolerance as the whole tensor.

A comparison has to have teeth, so the test runs three deliberately wrong kernels through the same
comparison and requires it to reject them: `dx` from the codes without the weight's scale grid, and
`dx` over half the reduction, in both weight forms. The card run put the relative L2 of the whole-tensor
cases at 0.0021-0.0031 and of the sampled rows at 0.0027-0.0028.

## The tile

One CTA holds a 256-row band of `dx` and one 128-wide block of `k`, in eight waves of 32 lanes; each
wave holds two 16-row tiles and all eight `k` steps of the block at once, as 16 accumulators of 8 f32
-- 128 VGPRs a lane.
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

The fixture's shapes of both ops, and the model's hidden width with sixteen rows, are compared over
the whole of `dx`; the model's other shapes are compared on the sampled rows above, where a shape
with two token counts is checked at both. The bench, when the binary is asked for it, calls every
one of them.

| shape | `M` | `N` | `K` |
| --- | --- | --- | --- |
| fixture, `gemm_nt_q` | 4 | 256 | 128 |
| fixture, `gemm_nt_q` | 1 | 128 | 256 |
| fixture, `gemm_nt_q` | 7 | 320 | 384 |
| fixture, the gated-delta-net projection's width, int8 form | 3 | 8 | 128 |
| fixture, `gemm_nt` | 4 | 256 | 128 |
| fixture, `gemm_nt` | 1 | 128 | 256 |
| fixture, `gemm_nt` | 7 | 320 | 384 |
| fixture, the gated-delta-net projection, bf16 | 3 | 8 | 128 |
| fixture, `ssm_out` | 2 | 128 | 64 |
| the hidden width whole, both ops | 16 | 5120 | 5120 |
| `attn_k/v` | 1024, 4096 | 1024 | 5120 |
| `attn_output` | 4096 | 5120 | 6144 |
| `attn_qg` | 1024, 4096 | 12288 | 5120 |
| `ssm_inz` | 4096 | 16384 | 5120 |
| `ffn_down` | 4096 | 5120 | 17408 |
| `ffn_gate_up` | 1024, 4096 | 34816 | 5120 |
| `ffn_gate_up`, a packed batch of three unequal sequences | 4023 | 34816 | 5120 |
| `ssm_out`, bf16 | 4096 | 5120 | 6144 |

The two 4096-token bands and the packed batch of 317, 1009 and 2697 rows are what a step's token
count looks like; the widths are the ones [`docs/ADAPTER.md`](../ADAPTER.md) writes down for
`qwen3.8-27b-iqr-v4`. No `model`-labelled test is needed for them: a device row takes its shape from
the record, not from the container, so the numbers below are the only thing a card run needs.

Limits, each checked rather than assumed. A record outside them -- one the predicate above did not
already decline -- is declined without writing to `dc`, the way a host row declines a shape it does
not serve; an adjoint must not throw (`src/adjoint/device/README.md`).

- on the int8 grid `group` must be 128, the width of the block a CTA owns (another grid is
  another kernel); the bf16 form carries no scale grid;
- `K` must be a positive multiple of 8, so that an eight-element chunk of codes is 4-byte aligned and
  the last chunk can be clamped to end on `K`;
- `M * N` and `M * K` must fit a 32-bit element offset -- 1.4e8 elements at the largest shape above;
- `N` is padded up to a multiple of 16 with zero weight rows and zero `dy` columns, because the
  reduction tile is 16 channels wide. The fixture's eight-channel projections take that path,
  and the padded rows cannot add to the sum.

## Scratch memory

One call allocates and frees, at `Np` the padded `N` and `G = K/128`: `2*M*Np` bytes of bf16 `dy`,
`Np*K` of weights (one byte per code, two per weight for the bf16 form), `2*Np*G` of bf16 scales, and
`2*M*K` of bf16 `dx` results, which the host adds into the cotangent slot. That is 53.1 MiB for
`attn_k/v` at 4096 rows, 250.7 MiB for `ffn_gate_up` at 1024 rows and 484.7 MiB for `ffn_gate_up` at
4096 -- arithmetic on the formula, and the peaks the card runs reported are 53.1, 250.7 and 484.6
MiB, so the formula is what the call asks for. The buffers come from the plugin's scratch
ledger (`src/adjoint/device/scratch.h`), so they are charged to radiance's VRAM pool, and the
test resets the ledger around the call, measures the peak the call asked for, and fails a kernel
that leaves any of it behind (`live_bytes` back to zero).

## Measurement

**Measured.** One call per shape by `tests/gpu/adjoint_linear.cpp`. With `--bench` the gpu tier's
binary prints one line per shape in the form `docs/kernels/README.md` asks a kernel doc to carry --
the adjoint's whole call, the forward twin, how they compare, the scratch peak -- and the card is
selected with `R97_DEVICE`:

```console
$ cmake -B build -G Ninja && cmake --build build -j
$ R97_DEVICE=0 build/r97_gpu_tests --bench
```

The adjoint's whole call is what the contract makes it pay: the host's f64 -> bf16 and f64 -> int8
conversion of `dy`, the weight and the scale grid, the uploads, the kernel and the download of `dx`.
The table splits it into its two measured halves, from the card run that this file carried before
the plugin landed (the kernel and the staging are unchanged; the buffers now come from the plugin's
scratch ledger instead of a private allocation). The card is one R9700 (gfx1201, 32 GB), one of this
machine's two; `R97_DEVICE` picks which of them the run holds, and the sibling kernels' numbers in
`docs/kernels/elementwise.md` and `docs/kernels/qk.md` come from a card of the same model, so they
are the ones to hold this table against -- numbers from another card are not comparable. The
forward column is the f64 host twin -- the op's own forward, one thread -- timed on the host by the
same `twin_find`/`TwinFn` path the bench uses, in seconds:

| shape | staging (host conversion + transfer) | kernel | whole call | forward twin (s) | TFLOP/s | scratch |
| --- | --- | --- | --- | --- | --- | --- |
| `attn_k/v` 4096 x 1024 x 5120 | 50.5 ms | 1.06 ms | 51.6 ms | 20.1 | 40.6 | 53 MiB |
| `attn_output` 4096 x 5120 x 6144 | 230.3 ms | 5.73 ms | 236.0 ms | 121.7 | 45.0 | 118 MiB |
| `attn_qg` 1024 x 12288 x 5120 | 341.6 ms | 3.39 ms | 345.0 ms | 61.2 | 38.0 | 94 MiB |
| `attn_qg` 4096 x 12288 x 5120 | 439.6 ms | 21.34 ms | 460.9 ms | 243.1 | 24.1 | 196 MiB |
| `ssm_inz` 4096 x 16384 x 5120 | 582.3 ms | 35.00 ms | 617.3 ms | 324.1 | 19.6 | 249 MiB |
| `ffn_down` 4096 x 5120 x 17408 | 765.1 ms | 18.47 ms | 783.6 ms | 344.4 | 39.5 | 262 MiB |
| `ffn_gate_up` 1024 x 34816 x 5120 | 1014.3 ms | 11.38 ms | 1025.7 ms | 172.2 | 32.1 | 250 MiB |
| `ffn_gate_up` 4096 x 34816 x 5120 | 1231.2 ms | 53.91 ms | 1285.1 ms | 688.8 | 27.1 | 484 MiB |
| `ffn_gate_up` batch, 4023 rows | 1222.1 ms | 51.39 ms | 1273.5 ms | 676.5 | 27.9 | 479 MiB |
| `ssm_out` bf16 4096 x 5120 x 6144 | 138.0 ms | 5.39 ms | 143.4 ms | 121.6 | 47.8 | 148 MiB |

`staging` is the host's conversion and the uploads, and `kernel` runs from the launch to
`hipDeviceSynchronize`, so the second includes the download of `dx`; `whole call` is their sum, the
number the bench prints. TFLOP/s counts `2*M*N*K` over `kernel` only. In the card run each shape was
called twice and the second call is the one reported, so the code object's first load is not in the
number; the bench warms up once and averages three calls instead. The scratch column is the peak the
call actually asked the ledger for, and it matches the formula above.

The card numbers are the last of three runs of this kernel on one R9700 (gfx1201, 32 GB) of this
machine's two, while contributors' builds shared it, and the columns move between them: the kernel
column repeated within about 5% of itself for every shape but `attn_k/v` at 4096 rows, which the
three runs put at 3.98, 1.68 and 1.06 ms, and the staging column, which is host work on a shared
CPU, varied by up to three times (`ffn_gate_up` at 4096 rows: 1.23 s to 4.02 s). Read a factor of
two into a single kernel number for this reason, and the scratch column, which is arithmetic, as
exact.

A run of this revision's bench on `card1` of the two -- the card `R97_DEVICE=0` reserves -- put
`attn_k/v` at 4096 rows at 98.6 ms and `attn_output` at 4096 rows at 313.4 ms against the table's
51.6 and 236.0 ms, which is the spread the paragraph above describes. Its line for `attn_k/v` at
1024 rows, which the table has no row for, is 41.2 ms. That run was stopped after the third shape to
hand the card back to the builds queued behind it, so the widest shapes were not repeated.

The staging column is not what a backward pays. It is dominated by converting the weight from the
harness's f64 values to the int8 codes the kernel wants -- 178 million elements at `ffn_gate_up`,
which is why 1024 rows and 4096 rows cost nearly the same -- and a real call is handed codes that a
`w8a8` container already holds, so only `dy` and the scale grid would be converted. `kernel` is
therefore the column to hold a real forward against, and a forward on the card pays no staging at
all; the twin pays nothing either, being the f64 arithmetic itself.

Why the forward is a host number here: the op has no device forward, so the harness's forward is the
f64 host twin, and `docs/kernels/README.md` calls that the honest reference until one exists. It is
a scalar f64 loop -- the twin of a quantised linear is not even a bf16 GEMM -- at 1.05-1.07 GMAC/s
on this machine while it is quiet: 20.1 s at `attn_k/v`, 121.7 s at `attn_output` and 61.2 s at
`attn_qg` 1024 are the three shapes that were timed then. Another session has since put the machine
under a load of 91 on 48 cores, where the same loop measured 0.29-0.85 GMAC/s, so the other rows
state the twin's product at 1.06 GMAC/s rather than a time off a contended machine. At the smallest
shape that arithmetic gives 20.3 s against the 20.1 s timed, and at the widest one a call is about
eleven and a half minutes. The bench times the twin once per shape rather than as often as the
adjoint for the same reason: the whole column is three quarters of an hour of one core when the
machine is quiet. A card's own forward, when an op has one, is what a kernel's `kernel` column
should be held against; the twin beside the adjoint says how much of a backward is the staging.

The card tier runs its tests from `main` (`tests/gpu/main.cpp`), not from the exit-time runner the
other tiers use: the same launch from the exit hook faults inside `hipLaunchKernel` once radiance's
device layer has initialised, so a device test cannot wait until the binary exits. `--bench` is the
same main's flag.
