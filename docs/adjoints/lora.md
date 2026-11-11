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

`lora_shrink` reads `a`, `a_scale`, `slot`, `lora_a [S, r, K]`, and writes `t [M, r]`:

    t[m, j] = sum_k x(m, k) * A[s, j, k]           s = slot[m],  0 <= m < M,  0 <= j < r

where `x(m, k)` is the value the trunk's activation holds for that element, which depends on the
trunk (`docs/ADAPTER.md` 1):

| trunk | `a` | `a_scale` | `x(m, k)` |
|---|---|---|---|
| bf16 | bf16 `[M, K]` | absent | `a[m, k]` |
| rotated int8 | i8 `[M, K]` codes | f32 `[M, ceil(K / group)]` | `a[m, k] * a_scale[m, k / group]` |

so the dequantisation is one product, `x = code * scale`, and `group` is the scale's group width
(128 in the checkpoint this repository stores, `docs/ADAPTER.md` 1). Radiance's shape function
reads the `group` parameter to give the scale stream `ceil(K / group)` columns per row
(`ref_shapes.cpp`), and its kernel does not divide K by that column count: `make_scales` derives
the width over K from the stream's grid with `grid_block` (`ref_gemm.h`), the power of two whose
ceiling division gives those columns, so a partial last group keeps the width its neighbours have.
`K = 300` over three columns is groups of 128, 128 and 44 wide — not three groups of 100, which
would read the scale of 100..127 for the codes of 100..127. The twin reads the stream back the
same way (`group_width` in `src/adjoint/host/lora.cpp`) and uses the record's `group` only as the
consistency statement radiance's row treats it as: if the count `ceil(K / group)` differs from the
stream's column count, the record is one the kernel would refuse with `RAD_E_SHAPE`, and the twin
lays it out for no one.

`a_scale` is an optional operand of the schema (`ref_registry.cpp`: `OPD_O("a_scale")`), so it
keeps its place even when the call does not pass it: radiance issues `RAD_OPK_NONE` there
(`issue.cpp`), and `Tape::record` names the position `none:1` with dtype `invalid`
(`rad_dtype_name(RAD_DT_INVALID)`). The bf16 form's record is therefore five operands wide too,
with its `a_scale` an absent position rather than a missing one.

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

The twin of each op is its arithmetic in f64, over the read operands in schema order. The shrink's
reads are `a`, `a_scale` (real for a coded trunk, the absent position for a bf16 one), `slot` and
`lora_a`; the expand's are `t`, `slot`, `lora_b` and `scale`.

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

where `group` is the width section 1 derives from the stream's grid, so the last group of a row
whose K is not a multiple of it is narrower and holds `ke = groups - 1` only up to `K`.

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

### 3.2 Frozen positions

`slot` is an index list and `lora_a`/`lora_b` are trainable weights: neither is something the walk
asks a cotangent of, so both are frozen (`R97_REGISTER_ADJOINT_F`). The expand's `scale` is frozen
too: `alpha / r` is a hyperparameter read from the adapter file, not a leaf. The shrink's `a_scale`
is **not** frozen when it is there: the quantiser that produced the codes reads the cotangents of
both of its outputs, so the adjoint writes `dc` for the codes and for the scales and the int8 chain
stays complete. A bf16 trunk leaves that position absent instead: it has no tensor, so no
cotangent either, and position 1 is frozen with `slot` and `lora_a`.

So two rows serve the op, and the coded one is registered first: the registry applies the first row
whose predicate accepts the record, and a coded record and a float one share the op name and the
schema width — only the `dtype` parameter tells them apart (`ref_registry.cpp`):

| activation | schema | frozen | row |
|---|---|---|---|
| `i8` | `a, a_scale, slot, lora_a, t` | 2, 3 | `R97_REGISTER_ADJOINT_IF_F("lora_shrink", host, ..., coded_act, 2, 3)` |
| bf16 | `a, a_scale(absent), slot, lora_a, t` | 1, 2, 3 | `R97_REGISTER_ADJOINT_F("lora_shrink", host, ..., 1, 2, 3)` |

The `_IF_F` form is the one that takes a predicate, and the float row's null predicate accepts
every record, which is why it comes second: a single row freezing 1 as well would declare the
scale's cotangent away for the coded form, and the adjoint's write of it would be dead. The tests
state the coded route on its own (`lora_shrink_coded_scale_gets_a_cotangent`): the shared FD check
skips frozen operands, so it cannot notice a position frozen by mistake.

### 3.3 The `B = 0` start

A fresh adapter starts with `B = 0` (`docs/ADAPTER.md` 4). Then the expand adds exactly nothing to
`y`, its cotangent of `t` is zero, and the shrink that follows gets `dA = 0` while `dB` is not:
the first step moves `B`, not `A`. `lora_path_with_b_zero` states both halves — the twin's `y` is
exactly zero, the kernel agrees, the expand's `dt` and `dB` are checked against the twin (dB not
zero), and the shrink handed that zero cotangent leaves `dA` and every `dc` at zero, also checked
against the twin.

## 4. What the tests cover

Six sites, one of them the tiny fixture's (`tests/support/fixture.h`: hidden 128, ffn 256), at
ranks 16 and 4:

| site | M | K | N | r | S | trunk |
|---|---|---|---|---|---|---|
| `tiny_q_proj` | 6 | 128 | 128 | 16 | 2 | bf16 |
| `tiny_gate_proj` | 5 | 128 | 256 | 16 | 1 | bf16 |
| `tiny_down_proj` | 4 | 256 | 128 | 4 | 1 | bf16 |
| `coded_q_proj` | 3 | 128 | 128 | 4 | 2 | int8, group 128 |
| `coded_down_proj` | 4 | 256 | 128 | 4 | 1 | int8, group 128 |
| `coded_up_proj` | 5 | 300 | 96 | 4 | 2 | int8, group 128 |

Every site has a base row spelled `-1` and one spelled past the pool (`S`), the two spellings a
record uses.

The three coded sites are widths apart on purpose. At `K = 128` a row has one scale group, so
every `k` reads `a_scale[m, 0]` and any group index at all agrees; at `K = 256` a row has two
groups of 128, so an index that is wrong — `k % groups`, say — reads a different scale over half
the row; and at `K = 300` a row has three groups, the last 44 wide, which is the shape that
catches a width that is not the group's: `ceil(K / groups)` is 100 there and misreads 100..127 and
200..255, while `grid_block` gives the 128 the record states. `check_kernel` fails on radiance's
`lora_shrink` for either mistake (300 over 128 is exactly the case `grid_block`'s own comment
names). That is the check that has to catch it: the adjoint and gradient checks read the same
twin, so a twin wrong about the group index agrees with its own differences.

| check | what it proves | tolerance |
|---|---|---|
| `check_kernel` | the twin against radiance's host row (`lora_expand`, `lora_expand_mm`, and `lora_shrink` on a coded activation) | 0.02 relative, bf16's own width |
| `float_shrink_kernel` (in the test) | the same for a bf16 activation, whose `a_scale` is the absent position 1 | 0.02 relative |
| `check_adjoint` | the adjoint against central differences of the twin, both forms of the shrink; for the coded form the scale stream is among the perturbed operands, because position 1 is not frozen | 1e-6 relative |
| `lora_shrink_coded_scale_gets_a_cotangent` (in the test) | the coded shrink's `a_scale` cotangent against the formula and against a derivative of the twin along a direction in the scale stream, without consulting the frozen list | 1e-12 of the block's largest for the formula, 1e-6 relative for the derivative |
| `check_grad` (in the test) | `dA`/`dB` in the ParamSet against central differences of the twin, per entry and along a direction | 1e-5 of the block's largest entry |

The bf16 trunk's kernel check is launched in the test rather than through `check_kernel`, because
radiance's row wants the absent `a_scale` in its place with no bytes (`abi/rad_plugin.h`) while
`check_kernel` fills every position and refuses a dtype radiance does not know; the test's launch
is the same recipe with position 1 absent (`RAD_DT_INVALID`, null data).

The gradient difference is measured against the block's largest entry and not against the entry
itself: a gradient in the `ParamSet` is f32 (`include/r97/params.h`), and an entry is a sum of f32
products that can cancel, so it carries about 1e-7 of the largest product it summed, which may be
far more than 1e-7 of the entry. The observed worst is 6e-7 of the block's largest, and a missing,
doubled or transposed term is orders of magnitude past that — it was checked by mutating the
adjoint and watching the tests fail.

`lora_expand_mm` is not in the finite-difference test: rounding `t` makes its twin piecewise
constant, so a central difference of it is the derivative of nothing. Its twin is checked against
the kernel and its adjoint is the `lora_expand` one, which is checkable.

## 5. The write's cotangent, and what this file does not do

- **`y`'s base cotangent.** `lora_expand` writes `y` (an INOUT operand), so its adjoint puts
  `cot y` into the write's slot in `call.dc`, and the walk passes that on to the version of the
  buffer the write replaced. Both halves are needed together: a block's residual is written by
  these rows on top of what the step above left there, so a cotangent that stopped at the write
  would never reach the writer. The record's `in_place` mark is what tells the walk to carry it
  (`include/r97/op.h`; patch `0121-issue-inout-mark`), and the family's own check
  (`tests/support/adjoint_check.h`) leaves a write's slot alone, so the row fills it itself.
- **The trainer's binding.** Turning a `Delta` plus a step into these records — which rows of the
  batch a site serves, the pool's `S`, the `scale` buffer, the quantiser that writes the codes and
  reads their cotangents — is the step's business, not the family's.
