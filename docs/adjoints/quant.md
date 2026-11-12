# The activation quantisers: their f64 twins and their host adjoints

Reference: radiance commit `451b136874084650992668bed9e8662deb58de88` —
`libref/ref_quant.cpp` (`ref_quant_act_i8g`, `ref_had_quant_act_i8g` → `had_quant_act_q8`,
`ref_rmsnorm_quant_fp8`, `ref_gated_quant_fp8`) and `libref/ref_common.h` (`fwht`, `q8_group`).
The family is `src/adjoint/host/quant.cpp` and `tests/host/adjoint_quant.cpp` holds the int8
quantiser to the numbers quoted below, through the contract of `include/r97/adjoint.h` and the
checks of `tests/support/adjoint_check.h`; section 6 documents the fused E4M3 producers.

The int8 pair, one quantiser: `quant_act_i8g` quantises groups of an activation,
`had_quant_act_i8g` rotates each group first. Both write int8 codes and one f32 scale per group;
the network then consumes `code * scale`, so that product — not the code — is what the adjoint is
a function of. `rmsnorm_quant_fp8` and `gated_quant_fp8` apply the same rule to their E4M3 codes.

## 1. The forward

Notation, for a row `r` and a group `i` of `t` contiguous columns starting at column `i·t`
(`t` = `group`):

| symbol | meaning |
|---|---|
| `x` | the activation, bf16, `[M, n]`, `M` = rows, `n` = columns |
| `t` | the group width, `group`; `N = ceil(n / t)` groups per row |
| `W` | `W[k][j] = (-1)^popcount(k & j)`, the unnormalised Walsh–Hadamard matrix of order `t` |
| `v` | the rotated group, `v = W x_i` for `had_quant_act_i8g`, `v = x_i` for `quant_act_i8g` |
| `a` | `amax_j |v_j|`, the group's largest magnitude |
| `s` | the scale, `s = a / 127` (`s = 1` when `a = 0`) |
| `u` | `v / s`, the group in units of the scale |
| `c` | the codes, `c_j = rint(clamp(u_j, −127, 127))`, nearest even |
| `q` | the dequantised activation, `q_j = c_j · s` |

`rint` is round-half-to-even, which is what the kernel's `rad_round_even_sat` and the int8 store
do.

Two shape rules, both the kernel's:

- `had_quant_act_i8g` requires `t` a power of two and `n % t == 0`: a group must never straddle the
  end of a row, and `W` only exists for a power of two. `group ≤ 0`, a non-power of two, or a group
  that does not divide `n` is `std::invalid_argument` in the twin, as it is an error in the kernel.
- `quant_act_i8g` reads `group ≤ 0` or `group > n` as `n` (one scale per row), and its last group
  may be short: `N = ceil(n / t)`, the final group holding `n − (N−1)t` columns.

There is **no normalisation on the activation side of the rotation**: `W` has `W W = t I`, and `t`
is a per-group constant, which the weight side of the `i8a8h` product folds into its own scale,
exactly as it folds the group's `s` in. In exact arithmetic the quantiser is positively homogeneous,
which is what makes the scale carry no gradient (section 5):

```
x → λx   ⟹   v → λv,   a → λa,   s → λs,   u unchanged,   c unchanged,   q → λq.
```

### The saturation the mask is about

`|u_j| = 127 |v_j| / a ≤ 127` in exact arithmetic, so the clamp is a no-op for the group's values:
the largest value maps to exactly `±127`, every other value to something smaller. `|u_j| > 127` can
only happen through the rounding of the reciprocal — the kernel computes `inv = 1/s` and
`u = v * inv`, and for the value at the amax that product is `a·(1/(a/127))`, one rounding either
side of 127. Which side is arithmetic, not a property of the model: of the 64 candidate amax values
the test's scan tries, 14 round outward, and on the finite-difference data below 4 of 2224 codes
are clamped, each of them its group's amax.

Consequence: a saturated code is not reachable by choosing a large activation — scaling the group
scales `a` with it — but only by choosing the last bits of `a`. `tests/host/adjoint_quant.cpp`
scans for an `a` that rounds outward (`scanned_amax`) and builds a group around it, and it never
assumes the outward case elsewhere.

## 2. The record and the f64 twin

No forward driver appends these records yet, so this is where the schema is written down. The op
reads `x`; it writes the codes and the scale. The record's parameters are the four the kernel takes,
`M`, `n`, `group`, `dtype`; the twin reads `n` and `group` and the operand shapes, and derives the
rows from the activation's size, as the kernel does.

```
operand      dtype   shape        written
x            bf16    [M, n]       no
codes        i8      [M, n]       yes       R97_REGISTER_TWIN("quant_act_i8g", ...)
scale        f32     [M, N]       yes       R97_REGISTER_ADJOINT("quant_act_i8g", host, ...)
```

`had_quant_act_i8g` is the same schema under its own name. The twin computes section 1 in f64 and
writes the two operands the op writes, in the operands' order, so its output is what the walk's
cotangents are taken of — the codes, not the dequantised product. It refuses a record the kernel
would refuse, by name (`std::invalid_argument`).

Checked against libref's host kernel through `check_kernel` at nine shapes: `M ∈ {1,2,3,4,5,8}`,
`n ∈ {48,64,96,128,256}` and `t ∈ {4,16,32,64,128}` under both ops, including a short last group
(`n = 48, t = 32`), a single rotation block of the tiny fixture's hidden size
(`n = 128, t = 128`), `group = 0` and `group < n`. The checker launches the op by the record's
name, operand order and parameter names, so a wrong operand or a wrong parameter name fails the
test rather than passing it. The worst error over the nine shapes is **7.0e-09 relative** on the
rotated op and **2.0e-10** on the plain one, against a tolerance of `2/256` — two bf16 ulps.

**The codes agree exactly, except at a rounding tie.** `127 v / a` lands on `k + 1/2` exactly
whenever `v` is a small binary fraction of the group's amax (`v = −a/2` gives exactly `−63.5`), and
there the f64 twin and the f32 kernel round the last bit of the reciprocal to opposite sides of the
tie: one code apart, which is one scale step of the group. A tie is arithmetic, not a claim about
the op, so `check_kernel` runs on activations the test nudges one bf16 ulp at a time until no value
of any group sits within 32 f32 ulps of a half-integer — outside that window the two round the same
way, and inside it a tie of the test's own data is common enough that most of the nine shapes need
a nudge. The tie itself is what section 6 leaves open.

### The linearised twin

Finite differences cannot differentiate a step function, so each op also registers `<op>_ste`: the
same record with the base point `x0` appended as a second read operand, whose twin is the forward
linearised there (section 4). The row exists for the check; no walk appends a `<op>_ste` record, and
`ops --registered` listing it is the price of the contract's one-row-per-name registry.

## 3. The adjoint: the straight-through estimator

Let `L` be the loss. The walk hands this op the cotangent of what it wrote — of the codes, `ḡ_j =
∂L/∂c_j`, and of the scale, `∂L/∂s`, which is dropped — and asks for `∂L/∂x`. Rounding has zero
derivative almost everywhere, so this adjoint is a definition rather than a derivative of the
forward, the standard straight-through estimator:

```
m_j        =  1 if |u_j| ≤ 127,  0 if |u_j| > 127        (the clamp's own derivative)
∂L/∂v_j    =  m_j · ḡ_j / s                               (the rounding's, taken as 1)
∂L/∂s      =  0
∂L/∂x      =  Wᵀ (m ⊙ ḡ / s)  =  W (m ⊙ ḡ / s)            quant_act_i8g: the identity
```

The estimator the issue states it as, in the dequantised domain `q = s · c`, is the same function.
With `g_j = ∂L/∂q_j` the cotangent a walk over the dequantised product hands back, `q = s · c`
gives `ḡ_j = ∂L/∂c_j = s · g_j` (`∂q/∂c = s`), that is `g_j = ḡ_j / s`, and

```
∂L/∂x  =  Wᵀ (m ⊙ g)          g_j = ∂L/∂q_j  =  ḡ_j / s.
```

The division by `s` in the first form is not a factor of the adjoint; it is the change of units
between the operand the op wrote and the product the network reads. Both forms are the same
numbers, and the frozen twin below is stated in the first, because the check multiplies by the
cotangent of the codes.

Four parts, and their reasons:

- **The scale cancels, in the dequantised form and only there.** `q = s · rint(v/s)` has slope
  `dq/dv = s · (1/s) = 1` where the rounding is straight-through, so in `g` the cotangent passes
  unchanged and no `1/group` and no `1/sqrt(group)` appears. The unnormalised rotation's
  `Wᵀ W = t I` is not undone either: the forward never divided by `t`, so its adjoint must not
  multiply by `1/t`.
- **`Wᵀ` is `W`**, because `W` is symmetric and, for the Walsh order, `W[k][j] = W[j][k]`. The test
  checks it exactly, and without holding a second copy of `W`: with the scale 1 and nothing
  clamped, a cotangent at the first code of a `t`-wide group comes back as `W e0` — all ones — and
  a cotangent of one everywhere comes back as `(t, 0, …, 0)`, so `W` really is the transform the
  adjoint applies, not its transpose times `1/t`.
- **The mask is the clamp having engaged**, `m_j = 0 ⟺ |u_j| > 127`. That is the composition of the
  two piecewise derivatives the forward itself has: `round` is straight-through (derivative 1
  everywhere it is defined) and `clamp`'s derivative is 1 strictly inside `[−127, 127]` and 0
  outside. It is the rule a fake-quant pass uses, and it keeps the amax's cotangent on every group
  where the reciprocal rounded inward — the top of the grid is inside the grid: a code of `±127`
  that was not clamped passes its cotangent.
- **The all-zero group is the rotation alone.** `a = 0 ⟹ s = 1, c = 0, u = 0`, no clamp engages,
  and the adjoint is `W (1 ⊙ ḡ) = W ḡ`; the test checks the pass-through of the plain op and the
  two transforms of the rotated one there, exactly.
- **The scale's own cotangent is dropped.** The scale is a constant of the map from `x` to `c`:
  section 5 has the alternative and why it is not taken. The mask and zero-group cases hand a
  nonzero scale cotangent to the adjoint, and `check_adjoint` hands a random one, so an adjoint
  that let it through would fail at once.

## 4. The frozen twin and the finite-difference check

Finite differences cannot check a step function: the forward's codes are piecewise constant, so its
difference quotient is zero almost everywhere, and nothing at all near a boundary. The twin that is
differentiated is therefore the forward with the rounding residual frozen at the base point `x0`:
the codes `c0`, the scale `s0`, the rotated value `v0` and the mask `m0` are held, and

```
T_code(x)  =  c0  +  m0 ⊙ (W x − v0) / s0                (per group; W is the identity for
T_scale(x) =  s0                                          quant_act_i8g)
```

```
∂T_code/∂x  =  Wᵀ diag(m0) / s0  =  the adjoint above,     T_code(x0) = c0,  T_scale(x0) = s0.
```

In the dequantised domain that is the issue's twin `x_hat = W x + const`: `s0 · T_code(x)` is
`q0 + m0 ⊙ (W x − v0)`. The code domain is the one the check multiplies, the cotangent the walk
hands back being the codes'; `T_scale` is constant, so the finite differences drop the scale's
cotangent the way the adjoint does. Without the mask (`m0 ≡ 1`) it is the issue's `const`; with it,
the clamp's corner is where the estimator puts a zero.

The row is `R97_REGISTER_ADJOINT_F("<op>_ste", host, <the same adjoint>, 1)`: schema position 1 —
`x0` — is frozen, so the checker perturbs the activation alone and asks for the activation's
cotangent alone. `check_adjoint` then compares `∂L/∂x` against central differences of
`f(x) = <ḡ, T_code(x)> + <∂L/∂s, T_scale(x)>`, one coordinate at a time, step `1e-6` of the
coordinate, tolerance `1e-6` relative.

Measured over six shapes (`3×64 g32`, `1×48 g16`, `5×128 g128` plain; `2×96 g32`, `2×256 g64`,
`5×128 g128` rotated), 2224 coordinates in all, 4 of them clamped: **the worst relative error is
1.3e-09**, and the linearised twin at its own base point is the forward, byte for byte, which the
test checks before it differentiates anything.

The mask is checked on its own as well, on the constructed group of section 1: a cotangent at the
clamped code gives a gradient of zero everywhere, a cotangent at a code inside the grid comes back
as `±1/s0` in the positions of `W e0`, and the same saturated base point goes through the finite
differences like any other.

## 5. The rejected alternatives

### A gradient through the scale

`s` is a function of the same values, `s = amax_j |v_j| / 127`, so the adjoint could carry a second
path `∂L/∂v_j += Σ_k g_k c_k · ∂s/∂v_j`, with `∂s/∂v_j = sign(v_j)/127` at the argmax and 0
elsewhere, plus a subgradient convention where the argmax ties. It is not taken:

- **It double-counts the amax.** The argmax's code is `127` and `c·∂s/∂v = 127/127 = 1`, so the
  element that sets the scale would receive `1 + 1 = 2` in its own direction, while every other
  element of the group receives `c_k/127` of it. The estimator's whole content is `dq/dv = 1` — the
  rounding error is at most half a scale step, and `q` is the nearest grid point to `v` — and a
  correction that makes the largest element's slope 2 is not that estimator anywhere.
- **The argmax is not differentiable**, so the addition has to choose a subgradient, and which
  element is the argmax is a property of the representation, not of the model: two equal values in
  a group would each get `1/2` or one of them `1`, for no reason a training step should see.
- **It is unobservable in the forward.** `x → λx` leaves every code unchanged and scales `q` by
  `λ`; the scale path reproduces only what the code path already carries.

What the adjoint does keep is the scale's *value*: it is stored per group and used by the weight
side of the `i8a8h` product, so a change of `s` changes `q` in the forward without the codes
changing. That is not a path through this op's input, which is why it is not in this adjoint.

### Saturation at the top of the grid

The mask could instead be `m_j = 0 ⟺ |c_j| = 127`, which treats "the code sits at the end of the
grid" as saturated. It is not taken, for the reason in section 1: with a max-derived scale the code
at the amax is *always* 127 — it is at the top of the grid because that is where the scale put it,
not because anything was clamped — so this reading would drop the largest element's cotangent in
every group of every layer, and would still leave a value just below the top of the grid unmasked
at the next one. `clamp`'s own derivative does not zero at its boundary, and the estimator is the
composition of the forward's two piecewise pieces, not a rule about the code.

## 6. The fused producers

The int8 trunk does not run `add`/`rmsnorm` and then `quant_act_fp8`: `rmsnorm_quant_fp8` writes
the bf16 norm and its E4M3 codes in one op, and `gated_quant_fp8` writes the bf16 product and its
codes. Their code cotangents follow section 3 -- divide by the group scale, keep only where the
clamp did not engage, drop the scale's cotangent -- and the differentiable part around the round
uses the same straight-through rule.

`rmsnorm_quant_fp8` reads `x`, an in-place `residual`, and the gain `w`, and writes the new
residual, `q`, the scale and `out_bf16`. With `r = bf16(x + residual)` and

    y_mj = bf16( r_mj * rsqrt(mean_j(r_m)^2 + eps) * (w_j + wadd) ),

the output cotangent `d = dout_bf16 + live ⊙ dq / s` gives

    dw_j   = sum_m d_mj r_mj rs_m
    dx_mj  = d_mj rs_m (w_j + wadd) - (sum_k d_mk r_mk (w_k + wadd)) rs_m^3 r_mj / n,
    dres_mj = dx_mj + dresidual_out_mj,

and the new residual's cotangent goes to both the old residual and `x`, because the op wrote it in
place. The norm's `rs_m` is differentiated, not held constant: the `-drs rs^3 r/n` term is the
difference between the norm's real derivative and the elementwise scaling.

`gated_quant_fp8` reads a packed `[M, 2n]` `gate_up` plane, gate half first, and writes `q`, the
scale and `out_bf16` (`p = bf16(act(g) * u)`). For `silu`, `act(g) = g sigmoid(g)` and
`act'(g) = sigmoid(g) (1 + g (1 - sigmoid(g)))`; for `sigmoid`, `act(g) = sigmoid(g)` and
`act'(g) = sigmoid(g) (1 - sigmoid(g))`. With `dp` the cotangent of `p`, the packed plane gets

    dgate += dp * u * act'(g),      dup += dp * act(g).

Both ops are registered in the `quant` family beside the int8 quantiser, and the host step walks
them on the int8 trunk.

## 7. What this does not cover

- **Only the host kernel is the reference**, as the issue says. The device kernels of these ops are
  not compared against the twin here; the GPU tier's own checks own that.
- **A rounding tie is not settled.** At `u = k + 1/2` exactly, the twin and the kernel may write
  codes one apart, so the kernel check runs where no value is at a tie and this document does not
  claim which of the two is right there. Neither the adjoint nor its mask is affected: a tie is
  inside the grid, so the cotangent passes on both sides of it, and the dequantised values are one
  scale step apart, the smallest difference the quantiser can express.
- **The scale the adjoint divides by is the one in the record.** The mask is evaluated from the
  tape's activation and the tape's scale in f64, not re-derived from the bf16 the container stores,
  so a run that quantises against `±127·bf16(s)` can have the amax's value on the other side of
  the clamp from the one the adjoint assumes. That is the last-bit arithmetic of section 1, and the
  adjoint does not chase it: making the mask depend on the storage precision of the scale would
  make it differ between a run that keeps the scale in f32 and one that rounds it.
- **The fused producers have no `<op>_ste` twin yet.** Section 6's two rows are checked through
  the int8 trunk's descent test (`tests/host/step_host_int8.cpp`), not through the
  finite-difference harness that holds the int8 quantiser.
- **The weight side.** `i8a8h`'s weight quantisation and the group factor of the rotation live on
  the other operand of the product and are another family's adjoint.
- **Any op of this family that turns up later** in the op inventory belongs here, against the same
  standard, with the formulas added to this document.
