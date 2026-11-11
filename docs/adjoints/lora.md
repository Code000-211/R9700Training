# The adapter path: `lora_shrink` and `lora_expand`, and the gradients of A and B

The forward is the adapter path of `docs/ADAPTER.md`: a low-rank correction added to the trunk's
output, `y += (alpha / r) * B * (A * x)`, served as two ops by radiance
(`third_party/radiance/libref/ref_lora.cpp`). The twin, the adjoints and the registrations are in
`src/adjoint/host/lora.cpp`; the tests are `tests/host/adjoint_lora.cpp`. These are the first
trainable leaves the walk carries: `A` and `B` are the parameters the optimiser moves, and the
cotangents of the activations are what the rest of the walk adds to.

```
        x [M, K]                                        y [M, N]
           |                                               ^
           |  lora_shrink                    lora_expand   |  y += scale . (t . B)
           v                                               |
        t [M, r]  ---------------------------------------- +
```

## 1. The forward, as radiance issues it

`x` is a rank-2 tensor of the batch's rows: `M` rows of the site, `K` in, `N` out. `slot[m]` names
the adapter row the module's row belongs to, `0 <= slot[m] < S`, with `S` the adapter pool's size at
this site (an op reads one record per targeted module, `include/r97/delta.h`); a row whose slot is
outside `[0, S)` is a **base row** — a row of the batch no adapter touches — and neither op reads or
writes it.

`lora_shrink` reads `a`, optionally `a_scale`, `slot`, `lora_a [S, r, K]`, and writes
`t [M, r]`:

    t[m, j] = sum_k x(m, k) * A[s, j, k]           s = slot[m],  0 <= m < M,  0 <= j < r

where `x(m, k)` is the value the trunk's activation holds for that element, which depends on the
trunk (`docs/ADAPTER.md` 1):

| trunk | `a` | `a_scale` | `x(m, k)` |
|---|---|---|---|
| bf16 | bf16 `[M, K]` | absent | `a[m, k]` |
| rotated int8 | i8 `[M, K]` codes | f32 `[M, ceil(K / group)]` | `a[m, k] * a_scale[m, k / group]` |

so the dequantisation is one product, `x = code * scale`, and `group` is the scale's group width
(128 in the checkpoint this repository stores, `docs/ADAPTER.md` 1). The scale's grid is
`ceil(K / group)` columns per row, which is what `make_scales` in the reference states; the twin
takes that per-row stream and lays out no other (radiance also reads a block grid over rows for
weights, which the adapter path does not use).

`lora_expand` reads `t [M, r]`, `slot`, `lora_b [S, N, r]`, `scale [S, N]`, and adds to
`y [M, N]`:

    y[m, n] += scale[s, n] * sum_j t[m, j] * B[s, n, j]      s = slot[m]

`scale` is the adapter's `alpha / r` (2.0 by default, `docs/ADAPTER.md` 4) laid out per adapter row
and output row. `lora_expand_mm` is the same op with `t` rounded to bf16 before the sum, the
formulation a GEMM that reads bf16 operands uses.

Both ops are in radiance's op inventory at this pin, `lora_shrink` including the coded activation
(`ref_lora.cpp` reads the scale stream for it), so the int8 form is checked against the kernel and
not only against the formula. An op of this family that turns up later in the inventory belongs to
this file.

## 2. The twin

The twin of each op is its arithmetic in f64, over the read operands in schema order:

- `lora_shrink_twin` writes `t` as above, `x` at full precision (a code times its scale, one
  product in f64).
- `lora_expand_twin` writes the **increment** of `y`: y's base value is not among the read
  operands, and every caller hands the write a zeroed buffer (`check_kernel` zeroes the written
  operands it fills). For `lora_expand_mm` the twin rounds `t` to bf16 first, ties to the even
  significand, a value below bf16's normal range flushing — the same rounding `rad_f32_to_bf16`
  does.

## 3. The adjoints

Let `cot t[m, j]`, `cot y[m, n]` be the cotangents of the written operands, which is what
`call.cot` carries. The shrink's cotangent in the **value space** `x` lives in is

    g[m, k] = sum_j cot t[m, j] * A[s, j, k]                    s = slot[m]

and the three routes out of it are:

| route | formula |
|---|---|
| the float activation | `dc a[m, k] += g[m, k]` |
| the codes | `dc a[m, k] += g[m, k] * a_scale[m, k / group]` |
| the group scale | `dc a_scale[m, ke] += sum_{k in group ke} g[m, k] * a[m, k]` |
| the weight | `dA[j, k] += sum_{m: slot[m] = s} cot t[m, j] * x(m, k)` |

i.e. the coded form splits the value-space cotangent over the two operands its dequantisation is
made of (`d/d code` is the scale, `d/d scale` is the code); a base row contributes to none of them.
The expand's:

    dt[m, j]  += sum_n cot y[m, n] * scale[s, n] * B[s, n, j]
    dB[n, j]  += sum_{m: slot[m] = s} cot y[m, n] * scale[s, n] * t[m, j]

`lora_expand_mm` has the same adjoint: rounding to bf16 is a step function, zero derivative almost
everywhere, so the walk takes the rounding as the identity (the straight-through convention every
bf16 GEMM backward takes).

### 3.1 The gradients in the ParamSet

The two leaves are added to `call.grads`, f32, added to and never overwritten, so a module reached
twice in one step sums both contributions:

| op | name | shape | what it is |
|---|---|---|---|
| `lora_shrink` | `<prefix>.lora_A` | `[r, K]` | the adapter's down factor, summed over the pool rows the record's slots name |
| `lora_expand` | `<prefix>.lora_B` | `[N, r]` | the adapter's up factor, likewise |

`<prefix>` is `call.prefix`, which the walk takes from the `Delta` the site belongs to
(`include/r97/delta.h`: `grad_prefix()`); `A` is stored `[r, K]` and `B` `[N, r]`, i.e. the layout
the adapter file holds them in, so the optimiser's update is a flat `ParamSet` update.

### 3.2 Frozen positions, and the two schema widths of the shrink

`slot` is an index list and `lora_a`/`lora_b` are trainable weights: neither is something the walk
asks a cotangent of, so both are frozen (`R97_REGISTER_ADJOINT_F`). The expand's `scale` is frozen
too: `alpha / r` is a hyperparameter read from the adapter file, not a leaf. The shrink's `a_scale`
is **not** frozen — the quantiser that produced the codes reads the cotangents of both of its
outputs, so the adjoint writes `dc` for the codes and for the scales and the int8 chain stays
complete.

The shrink's record carries `a_scale` only beside a coded activation (`ref_shapes.cpp` skips the
operand for a float `a`), so the schema is one of two widths and the frozen positions differ; two
rows are registered, told apart by the record's `dtype`, which is also what says which layout the
bytes are in:

| activation | schema | frozen | row |
|---|---|---|---|
| `i8` | `a, a_scale, slot, lora_a, t` | 2, 3 | `R97_REGISTER_ADJOINT_IF_F(..., coded_params, 2, 3)` |
| bf16 | `a, slot, lora_a, t` | 1, 2 | `R97_REGISTER_ADJOINT_IF_F(..., float_params, 1, 2)` |

### 3.3 The `B = 0` start

A fresh adapter starts with `B = 0` (`docs/ADAPTER.md` 4). Then the expand adds exactly nothing to
`y`, its cotangent of `t` is zero, and the shrink that follows gets `dA = 0` while `dB` is not:
the first step moves `B`, not `A`. `lora_path_with_b_zero` states both halves — the twin's `y` is
exactly zero, the kernel agrees, the expand's `dt` and `dB` are checked against the twin (dB not
zero), and the shrink handed that zero cotangent leaves `dA` and every `dc` at zero, also checked
against the twin.

## 4. What the tests cover

Five sites, one of them the tiny fixture's (`tests/support/fixture.h`: hidden 128, ffn 256), at
ranks 16 and 4:

| site | M | K | N | r | S | trunk |
|---|---|---|---|---|---|---|
| `tiny_q_proj` | 6 | 128 | 128 | 16 | 2 | bf16 |
| `tiny_gate_proj` | 5 | 128 | 256 | 16 | 1 | bf16 |
| `tiny_down_proj` | 4 | 256 | 128 | 4 | 1 | bf16 |
| `coded_q_proj` | 3 | 128 | 128 | 4 | 2 | int8, group 128 |
| `coded_down_proj` | 4 | 256 | 128 | 4 | 1 | int8, group 128 |

Every site has a base row spelled `-1` and one spelled past the pool (`S`), the two spellings a
record uses.

The two coded sites are widths apart on purpose. At `K = 128` a row has one scale group, so
every `k` reads `a_scale[m, 0]` and any group index at all agrees; at `K = 256` a row has two
groups of 128, so an index that is wrong — `k % groups`, say, or a group width derived from
something other than the stream's grid — reads a different scale over half the row and
`check_kernel` fails on radiance's `lora_shrink`. That is the check that has to catch it: the
adjoint and gradient checks read the same twin, so a twin wrong about the group index agrees with
its own differences.

| check | what it proves | tolerance |
|---|---|---|
| `check_kernel` | the twin against radiance's host row (`lora_expand`, `lora_expand_mm`, and `lora_shrink` on a coded activation) | 0.02 relative, bf16's own width |
| `float_shrink_kernel` (in the test) | the same for a bf16 activation, whose record has no `a_scale` | 0.02 relative |
| `check_adjoint` | the adjoint against central differences of the twin, both forms of the shrink | 1e-6 relative |
| `check_grad` (in the test) | `dA`/`dB` in the ParamSet against central differences of the twin, per entry and along a direction | 1e-5 of the block's largest entry |

The bf16 trunk's kernel check is launched in the test rather than through `check_kernel`, because
radiance's row wants `a_scale` in its place with no bytes (`abi/rad_plugin.h`) while `check_kernel`
fills every position and refuses a dtype radiance does not know; the test's launch is the same
recipe with position 1 absent (`RAD_DT_INVALID`, null data).

The gradient difference is measured against the block's largest entry and not against the entry
itself: a gradient in the `ParamSet` is f32 (`include/r97/params.h`), and an entry is a sum of f32
products that can cancel, so it carries about 1e-7 of the largest product it summed, which may be
far more than 1e-7 of the entry. The observed worst is 6e-7 of the block's largest, and a missing,
doubled or transposed term is orders of magnitude past that — it was checked by mutating the
adjoint and watching the tests fail.

`lora_expand_mm` is not in the finite-difference test: rounding `t` makes its twin piecewise
constant, so a central difference of it is the derivative of nothing. Its twin is checked against
the kernel and its adjoint is the `lora_expand` one, which is checkable.

## 5. What this file does not do

- **`y`'s base cotangent.** `lora_expand` writes `y` (an INOUT operand), so the cotangent `y`'s
  base value already holds is not among the read operands and not this op's to add: a write's slot
  in `call.dc` is never filled by an adjoint (`tests/support/adjoint_check.h`). The walk passes it
  on to whatever wrote that base value.
- **The trainer's binding.** Turning a `Delta` plus a step into these records — which rows of the
  batch a site serves, the pool's `S`, the `scale` buffer, the quantiser that writes the codes and
  reads their cotangents — is the step's business, not the family's.
