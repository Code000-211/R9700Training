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
W[n,k]  = b_code[n,k] * sb[n,g]
dx_code[m,k] += sa[m,g] * sum_n dy[m,n] * W[n,k]
```

`dx_code` is therefore a cotangent in the codes' basis, which is what the walk hands an op that
writes codes (`docs/adjoints/quant.md` section 3): the quantiser's straight-through adjoint
divides it by `s` on the way to its bf16 input, and the two cancel on the path where nothing else
reads the codes. The dequantised `dx_hat = dx_code / sa` is what a consumer of the dequantised
product would be handed; it is not what this row returns, because the walk's contract is the
codes. There is no weight gradient.

The host issues `gemm_nt_q` (not the fp8 `gemm_nt_q_gated` form) for `qwen35_w8a8`'s gated MLP:
`gate_up.i8` disables that arch's gated fold and the two halves are two ordinary int8 linears. No
gated form is registered here.

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
`gemm_nt_q` the finite-difference case holds the activation scale at one, where the codes' basis
and the dequantised one are the same; a separate check keeps a real scale and perturbs the codes
by `eps * dir`, confirming `dx_code` against the code-basis directional derivative. One more check
pins the frozen positions of every row against the table above, so this page and the registration
cannot drift apart.
