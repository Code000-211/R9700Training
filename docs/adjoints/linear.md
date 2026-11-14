# Host adjoints: `linear`

The frozen linears of the training graph. Every weight here is frozen, so no adjoint in this file
writes a weight gradient; each returns only the cotangent of its input. The rows are registered by
`src/adjoint/host/linear.cpp` and checked by `tests/host/adjoint_linear.cpp`.

| op | reads | writes | frozen |
| --- | --- | --- | --- |
| `gemm_nt` | `a`, `b` | `y` | `b` |
| `gemm_nt_q` | `a`, `a_scale`, `b`, `b_scale` | `y` | `a_scale`, `b`, `b_scale` |
| `logits_gemm` | `x`, `lm_head` | `logits` | `lm_head` |
| `gather_rows` | `x`, `idx` | `y` | `idx` |
| `embed_lookup` | `tokens`, `wte` | `x` | `tokens`, `wte` |

Operands are named as the radiance schema names them. radiance marks `a_scale` optional and puts
`a_sum`/`b_ref` after `y`; the int8 grid always supplies `a_scale`, and it has neither tail, so
these rows read exactly the operands the table lists. The frozen positions travel with the row as
schema positions, and the column names them: `b` for `gemm_nt`, `a_scale`, `b`, `b_scale` for
`gemm_nt_q`, `lm_head` for `logits_gemm`, `idx` for `gather_rows`, and both operands of
`embed_lookup`.

## `gemm_nt`

The forward is the transposed product of a bf16 weight:

```text
y[m,n] = sum_k a[m,k] * b[n,k]                 a: [M,K]   b: [N,K]   y: [M,N]
```

The adjoint, with the weight frozen, is the product with the weight itself:

```text
dx[m,k] += sum_n dy[m,n] * b[n,k]              dy: [M,N]   dx: [M,K]
```

## `gemm_nt_q` on the int8 grid

`qwen35_w8a8` uses the `i8a8h` encoding: both operands are int8 codes, an activation scale is f32
or bf16 per row and 128-wide block of K, and the weight scale is bf16 per row and block of K. A
scale is written `sa[m,g]` and `sb[n,g]`, with `g = floor(k / 128)`. The twin is the host kernel's
grouped chain:

```text
s[m,n] = sum_{k in group g} a_code[m,k] * b_code[n,k]
y[m,n] = sum_g s[m,n] * sa[m,g] * sb[n,g]
```

The activation `a` is the dequantised, rotated activation of the forward: its code times
`sa[m,g]`. Its scale belongs to the quantiser that produced it, not to this GEMM, so the row marks
`a_scale` frozen beside `b` and `b_scale`: the walk asks for the cotangent of `a` alone. The weight
in the adjoint is the dequantised weight:

```text
W[n,k] = b_code[n,k] * sb[n,g]
dx_hat[m,k] += sum_n dy[m,n] * W[n,k]
```

`dx_hat` is therefore a cotangent in the rotated, dequantised basis. The scale `sa` does not appear
in it: the quantiser's straight-through adjoint passes `dx_hat` to its own bf16 input unchanged.
There is no cotangent for the codes, and no weight gradient.

The host issues `gemm_nt_q` (not the fp8 `gemm_nt_q_gated` form) for `qwen35_w8a8`'s gated MLP:
`gate_up.i8` disables that arch's gated fold and the two halves are two ordinary int8 linears. No
gated form is registered here.

### The plugin-private operands: `b` and `b_scale` on the card

A `w8a8_fwht128` container does not hand the tape f64 values for the weight or its scale. Both
operands arrive with the record's shape and a pointer into the engine's two stored planes
(`AdValue::raw`, `include/r97/adjoint.h`) and no values at all: the engine decodes the layout in its
kernel, so there is no f64 form to snapshot. The walk accepts such an operand -- one with neither
values nor a pointer is still refused -- and the device row decodes it.

The stored planes are the pin's (`third_party/radiance/libr4d/r4d_fp8_layout.cpp`,
`libr4d/r4d_fp8_frag.h`):

```text
codes  int8 [N][K]   a (16 row, 64 k) tile of 1024 B as 32 lanes x 32 B; lane l's bytes 8t cover
                     k = t*16 + (l>>4)*8 .. +7 of row l&15      ("r4d.i8h.w.frag64")
scale  bf16           tile-major [N/16][K/128][16]: d[((r/16)*G + g)*16 + r%16] = s[r*G + g]
                                                                 ("r4d.i8.s.t16")
```

`src/adjoint/device/linear.hip` reads them in place (`frag_go`, `t16_go`) and keeps the f64 path for
a record whose operands do carry values, so a fixture and the container take the same row by the
same `i8a8h` predicate. The decode is checked two ways: `tests/host/adjoint_private_weight.cpp`
writes the planes the pin's way, reads them back with the kernel's inverse, and holds the f64 row's
`dx` on the canonical weight to a plain f64 product of the decoded one (`1e-9` relative L2);
`tests/gpu/adjoint_private_weight.cpp` hands the device row the same planes on the card and holds
its `dx` to that product (bf16 tolerance).

The stored pair is behind `R97_STORED_WEIGHTS=1` while the trunk's real encoding is still being
identified. With the variable unset, `snapshot()` (`src/join_common.h`) returns the operand's shape
and no pointer, and the walk refuses it by name
(`tests/host/adjoint_private_weight.cpp:234`), which is what it did before the pointer was
threaded. The gate is there because the container's plane is **not** the int8 pair above: its
weights pool is 15.13 GiB over 27e9 parameters (4.81 bpw), and the first real run faulted 65.08 MiB
into an 85 MiB `N*K` read, so the codes are narrower than a byte.

### The trunk's real plane: an IQR form, decoded by libr4d's own kernel

`build/r97train info` reports the container's `quant` as `w8a8_fwht128`, but that is the header's
metadata string, not the trunk's per-weight encoding: the weights are IQR forms
(`libr4d/r4d_iqr.h`) -- a tail-biting trellis or a table at B bits a value, a state carried across a
128-K unit, and a scale form of its own (`kBf16`, `kE4m3Tensor`, `kE4m3Row`, `kRowF32`). The engine
selected `gemm_iqr_nt_m64` for these GEMMs (`libr4d/r4d_rows.cpp:8808`, layout
`r4d_rad_layout_iqr` at `:8819`), whose contract is "i8 A with an f32 scale per (row, 128 K), W an
IQR form's codes, decoding to int8 values".

The form is a function of the weight's `RadEncoding`, and patch
`patches/training/0124-encoding-on-record.patch` puts that encoding on the tape record
(`RadIssueOperand::enc`) for the same operand that already carries the view -- a dtype with no
element size. It is not a stored form word: libr4d *derives* the form from the encoding's planes
(`libr4d/r4d_iqr_layout.cpp:83-158`), so the core cannot compute one and does not try. `AdValue::enc`
carries it beside `raw`, and `R97_DUMP_SNAP=1` prints the planes, so a form the row derives is
checkable against what the layout hook saw.

`src/adjoint/device/iqr_plane.hip` decodes the plane **with the forward's own arithmetic**:
`r4d_iqr_dec.h` is a header of `__device__` functions that the forward's kernel includes
(`libr4d/r4d_gemm_iqr.hip:24`), and the decode is its `Dec::fetch`/`rows`/`unzero` then
`frag(w, tr, lane, s)` -- the sequence `r4d_gemm_iqr.hip:728-732` runs. r97 therefore carries no
second trellis: a wrong `dx` disagrees with the forward rather than agreeing with itself. What r97
does state is the *classification* (which form an encoding is), ported rule for rule from
`r4d_iqr_layout.cpp:31-158` with the lines cited, and the *addressing* (`walk_tile`,
`r4d_iqr_layout.cpp:261-282`), whose constants are header-exposed. One wave decodes one unit into
the int8 fragment order `frag_go` already reads (`plane_index.h`), so the linear row's `dx` runs
unchanged over a decoded plane.

The scale plane is emitted per form. The `dx` kernel reads one bf16 per (row, 128-K block) in
tile-major order (`t16_go`), and `iqr_plane_scale` writes exactly that from the form's own planes in
the layout hook's order (`r4d_iqr.h`'s `own_scale`, `fold_scale`, `cf_off`, `blk_off`): the bf16
level as stored, an E4M3 level times its f32 second level rounded to bf16, one f32 a row times its
block's f32 factor, and on the `tcq_rb` grid the block's own bf16 -- the row's factor is already in
the codes, which `Dec::rows` puts there with the same ratio. The record declares the scale operand
with the *weight's* shape, so its geometry comes from the form, never from `bs->shape`. The decoded
codes and the emitted scale are scratch for the call, and the row then runs the same
`linear_dx_kernel<Q, STORED>` the int8 pair does: an IQR form is a different *plane*, not a
different product.

`tests/gpu/adjoint_iqr_plane.cpp` holds that path to the f64 product on two forms -- a table trellis
on the `tcq_rb` grid under an e4m3 level, and a computed trellis (`kTrellisMad`) under one f32 a row
times its blocks -- reading the weight out of the planes' own bit streams by hand.


## `logits_gemm`

The vocabulary projection is the same product with the head weight:

```text
logits[m,v] = sum_k x[m,k] * lm_head[v,k]      x: [M,K]   lm_head: [V,K]   logits: [M,V]
dx[m,k] += sum_v dlogits[m,v] * lm_head[v,k]
```

The head is frozen, so no head gradient is produced. A quantised head belongs to the family that
owns its layout, not to this bf16 row.

## `gather_rows`

`gather_rows` selects rows of `x`:

```text
y[i,j] = x[idx[i],j]     for 0 <= idx[i] < rows(x)
y[i,j] = 0               for idx[i] < 0
```

Its adjoint is a scatter-add into the source rows:

```text
dx[idx[i],j] += dy[i,j]  for 0 <= idx[i] < rows(x)
dx[r,j]      += 0        otherwise
```

`idx` is frozen: it is discrete and has no cotangent. A negative index selects a zero row, as in the
host kernel.

## `embed_lookup` is a stop

`embed_lookup` gathers rows of the embedding table:

```text
x[i,j] = mult * wte[tokens[i] - vocab_offset, j]
```

with a zero row for an id outside the table. The table is frozen, and nothing in the graph lies
before the lookup: `tokens` are discrete too. Its host adjoint is therefore a stop that writes no
cotangent at all; both operands are marked frozen so the walk asks for neither, and the row is
registered only to end the walk.

## Checks

Each op's twin, `embed_lookup`'s included, is compared with radiance's host kernel at bf16
tolerance on three shapes, one of them the tiny fixture's (hidden 128, MLP 256, vocabulary 320).
Each adjoint that answers is compared with central finite differences of the twin to `1e-6`
relative; `embed_lookup`'s stop is checked to add nothing to either of its operands. For
`gemm_nt_q` the finite-difference case holds the activation scale at one, because the adjoint
describes the dequantised basis; a separate check perturbs the codes by `eps / sa` and confirms
`dx_hat` against the dequantised directional derivative with a real scale. One more check pins the
frozen positions of every row against the table above, so this page and the registration cannot
drift apart.
