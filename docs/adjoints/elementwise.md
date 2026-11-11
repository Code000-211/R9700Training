# The elementwise host adjoints: `mul`, `scale`, `sigmoid`, `silu`, `silu_mul`, `act_mul`, `cast`, `rmsnorm`

Reference: radiance commit `451b136874084650992668bed9e8662deb58de88` —
`libref/ref_elementwise.cpp` (`ref_mul`, `ref_scale`, `ref_sigmoid`, `ref_silu`, `ref_silu_mul`,
`ref_act_mul`, `ref_cast`, `ref_rmsnorm`), the schemas in `libref/ref_registry.cpp`, and the
plugins that issue the ops on the host: `arch/common/rad_fp8.h`, `rad_block_mlp_fp8.h`,
`rad_block_attn_gated_fp8.h`, and the bf16 blocks `rad_block_mlp.h` and
`rad_block_attn_gated.h`.

The twins and their adjoints live in `src/adjoint/host/elementwise.cpp`. Each op registers both by
name — `R97_REGISTER_TWIN` and `R97_REGISTER_ADJOINT`, or `R97_REGISTER_ADJOINT_F` for the two
whose weight is frozen — so the backward walk finds it through `twin_find` / `adjoint_find` like
every other op, and `r97train ops --registered` lists it. `tests/host/adjoint_elementwise.cpp`
checks each of them twice, through `tests/support/adjoint_check.h`: the twin against radiance's
host kernel for the op, loaded from `libref.so`, and the adjoint against central finite
differences of the twin to 1e-6 relative.

## 1. Conventions

**The twin is the real-valued function.** radiance's host kernels read the operand bytes of the
tensor's dtype, work in f32 and store the result rounded through the output's dtype (`act_mul`
additionally rounds the activation to the output's dtype before the product, and `quick_gelu`
rounds 1.702 x). The twin is `void(const OpRecord&, const std::vector<AdValue>&, std::vector<AdValue>*)`
and works in f64 on the read operands' values, writing no rounding, so it differs from the kernel
by at most that store rounding: 2⁻⁹ relative at bf16. The test holds the twin to 0.02 (about four
bf16 ulps) against the kernel, and the adjoint to 1e-6 relative against finite differences of the
same twin. The adjoint is the derivative of the smooth twin, so the kernel's rounding is not in it
— which is what a gradient of the op is.

**The dtypes are storage, not arithmetic.** No twin takes a dtype argument: these ops compute the
same function whatever tensor holds their result. The dtypes appear in the record and in the
test's cases. Both tiny fixtures keep bf16 activations and f32 norm gains, and the int8 fixture is
converted from the same checkpoint, so every activation here is checked at bf16; `cast` and the
f32-side cases carry the pairs the plugins use.

**Real parameters arrive as decimals.** A record's parameter value is an int or a string, so
`eps`, `wadd` and `mult` reach the twin as their decimal spelling and the twin parses it
(`std::strtod`); an absent or unparsable one leaves the kernel's default (`1e-6`, `0`, `1`).
Zero is a value: `wadd=0` is the plain norm, not a missing parameter.

**Frozen weights.** `rmsnorm`'s gain (operand 1) and `scale`'s `[1]` factor (operand 1) are
checkpoint weights; the trainer adjusts LoRA adapters and no base weight. Both adjoints are
registered with `R97_REGISTER_ADJOINT_F(..., 1)`, so the walk never asks them for a cotangent and
they write the input's only. The kernel refuses `wadd` without a gain, and so does the twin.

**Broadcasting.** `mul` shares `add`'s one broadcast rule: an operand carrying exactly `n`
elements against a taller output broadcasts along rows, and nothing else broadcasts. Every other
op of the family is elementwise or row-local.

**Shapes.** `[rows, n]` row-major unless a section says otherwise; `n` is the record's `n`
parameter, which is the row width the schema declares.

## 2. `mul`

The gated attention applies the sigmoid gate to its output: radiance declares it as the unfused
pair `sigmoid` into a gate buffer and then `mul` over two same-shape `[rows, q_dim]` operands
(`rad_block_attn_gated.h`), and the fp8 block folds the same pair into one op. `mul` also carries
`add`'s one broadcast rule, an operand of `[1, n]` against a taller output.

```
y[r, i] = a[r, i] · b[r, i]              b of [rows, n]
y[r, i] = a[r, i] · b[0, i]              b of [1, n]
```

Adjoint, both operands differentiable (both are activations on the tape):

```
da[r, i] = dy[r, i] · b[r, i]        (b[0, i] when broadcast)
db[0, i] = Σ_r dy[r, i] · a[r, i]    (a row again; the row sum is the broadcast's contraction)
db[r, i] = dy[r, i] · a[r, i]        (same-shape b)
```

## 3. `scale`

`y = x · s`, `s` a `[1]` weight broadcast over every element — the per-tensor scalar a checkpoint
carries. `s` is frozen; the twin refuses a record with no `s`.

```
y[r, i] = s · x[r, i]
dx[r, i] = s · dy[r, i]
```

## 4. `sigmoid`

The gated attention's gate, and `act_mul`'s `"sigmoid"` mode, are the same function:

```
y = σ(x) = 1 / (1 + exp(-x))
dx = dy · y · (1 - y)
```

The adjoint recomputes `σ(x)` from the forward's input (`fwd[0]`), not from the op's output: the
tape snaps a written operand before the write, so `fwd[1]` is not the value the write left. When
the gate is the second column slice of a fused `[tokens, heads, 2·head_dim]` operand, the twin and
the adjoint both read that slice — `column_view` takes `n` from the output's width and the stride
from the input-to-output element ratio — and write the `n` columns out.

## 5. `silu`

```
y = x · σ(x)
dx = dy · σ(x) · (1 + x · (1 - σ(x)))
```

With `s = σ(x)` the factor is `s + x·s·(1 - s)`. The forward keeps nothing but `x`, so the adjoint
recomputes `σ(x)` — one `exp` per element.

## 6. `silu_mul`

The gated feed-forward, gate and up fused into one `[rows, 2n]` operand. The gate is the FIRST
half — radiance's `oGateUp` and rad-convert's weight order — and `n` is the output's width:

```
g[r, i] = x[r, i]          u[r, i] = x[r, n + i]
y[r, i] = silu(g[r, i]) · u[r, i]
```

Adjoint (`dx` has `x`'s shape):

```
dg[r, i] = dy[r, i] · u[r, i] · (σ(g) + g·σ(g)·(1 - σ(g)))
du[r, i] = dy[r, i] · silu(g[r, i])
```

## 7. `act_mul`

`y = act(a) · b` with both operands `[rows, n]`. The family implements all six modes the kernel's
schema enumerates (`silu`, `gelu`, `gelu_tanh`, `sigmoid`, `quick_gelu`, `relu`) so that a record
with any of them resolves. The two the Qwen3.5 plugins name on this path are `"sigmoid"` for the
gated attention's gate (`rad_block_attn_gated_fp8.h`, `rad_fp8.h`) and `"silu"` for a
feed-forward's activation (`rad_fp8.h`); the test checks those two and the other three smooth
modes against the kernel and against finite differences, and `relu` against the kernel alone (its
kink sits inside the samples' range, where a central difference is meaningless). A record with no
recognisable `act` gets no twin value and no cotangent.

`quick_gelu` is the sigmoid alone: `ref_act_mul` applies `σ(round(1.702 · a))` and leaves the
`× x` to the multiplier operand, so a caller that wants `x · σ(1.702x)` passes the same tensor as
`a` and `b` — the unfused `mul`-then-`act_mul` pair the kernel's comment says the mode replaces.
The twin therefore keeps `act(x) = σ(1.702x)`; folding the `x` into the activation would put it a
factor of `b` away from the kernel.

```
y[r, i] = act(a[r, i]) · b[r, i]
da[r, i] = dy[r, i] · b[r, i] · act'(a[r, i])
db[r, i] = dy[r, i] · act(a[r, i])

act = silu:      act(x) = x·σ(x)   act'(x) = σ(x) + x·σ(x)·(1 - σ(x))
act = sigmoid:   act(x) = σ(x)     act'(x) = σ(x)·(1 - σ(x))
act = gelu:      act(x) = ½x(1 + erf(x/√2))
                 act'(x) = ½(1 + erf(x/√2)) + x·e^(-x²/2)/√(2π)
act = gelu_tanh: u = k₀(x + k₁x³), t = tanh(u), k₀ = √(2/π), k₁ = 0.044715
                 act(x) = ½x(1 + t)
                 act'(x) = ½(1 + t) + ½x(1 - t²)·k₀(1 + 3k₁x²)
act = quick_gelu: act(x) = σ(1.702x)
                 act'(x) = 1.702·σ(1.702x)·(1 - σ(1.702x))
act = relu:      act(x) = max(x, 0)   act'(x) = 1 for x > 0, else 0
```

The kernel rounds `act(a)` to the output's dtype before the product and rounds `1.702x` at
`quick_gelu`; the twin keeps both exact.

## 8. `cast`

`y = x` with the values converted between `from` and `to`. The dtype pair selects the kernel; the
bytes are described by the tensors' own dtypes. Between float dtypes a cast only rounds, so on
values this is the identity and so is the adjoint:

```
y  = x
dx = dy
```

The plugin's path uses it as a row copy (`bf16 → bf16`: `rad_block_attn_fp8.h`'s q lift); the
widening and narrowing pairs are checked beside it.

## 9. `rmsnorm`

```
s_r = 1 / sqrt( (1/n) Σ_j x[r, j]² + eps )
y[r, i] = x[r, i] · s_r · (w[i] + wadd) · mult
```

`s_r` is computed per row from the sums of squares; `eps` is added before the reciprocal square
root, `wadd` shifts the gain (GemmaRMSNorm's `w + 1`, which Qwen3.5 uses with a zero-centred
gain), and `mult` scales the whole row before the store's single rounding (Gemma 4's audio
feed-forward uses 0.5). radiance accumulates `ss` in f32 and forms `w + wadd` in f32; the twin
accumulates and forms it in f64.

Input cotangent, with `u_i = w[i] + wadd` and `t_r = Σ_i dy[r, i] · x[r, i] · u_i` the row's
contraction against the gained input:

```
dx[r, j] = mult · s_r · ( u_j · dy[r, j] - (s_r² · x[r, j] / n) · t_r )
```

The derivation: differentiating `y_i = mult · s · x_i · u_i` gives
`∂y_i/∂x_j = mult(s·u_i·δ_ij + x_i·u_i·∂s/∂x_j)` with `∂s/∂x_j = -s³·x_j/n`, so
`dx_j = Σ_i dy_i ∂y_i/∂x_j = mult·s·(u_j·dy_j - (s²·x_j/n)·t)`. The second term is the row's
shared rms: every element of the row moves it, so every `dy_i` weights `x_j`. The gain is frozen
(`R97_REGISTER_ADJOINT_F("rmsnorm", host, …, 1)`), so the adjoint returns `dx` only.

The operand may be a column slice of a fused buffer rather than a `[rows, n]` tensor: the tiny
fixture's attention writes q and gate into `[tokens, heads, 2·head_dim]` and normalizes the first
slice. `column_view` takes `n` from the written operand's width and infers the group stride from
the input-to-output element ratio (the first slice has offset 0); the twin reads and the adjoint
writes that slice, leaving the other columns alone.

## 10. What the test covers

Per op: three shapes, one of them the tiny fixture's. Each case runs the twin against radiance's
host kernel and the adjoint against finite differences of the twin.

| op         | shapes                                | dtypes and modes |
| ---------- | ------------------------------------- | ---------------- |
| `mul`      | `[5,128]`, `[3,64]`, `[7,17]`         | bf16 with b whole and with b one row, f32 |
| `scale`    | `[5,128]`, `[3,64]`, `[7,17]`         | bf16, f32        |
| `sigmoid`  | `[5,128]`, `[3,64]`, `[7,17]`         | bf16             |
| `silu`     | `[5,128]`, `[3,64]`, `[7,17]`         | bf16             |
| `silu_mul` | `[5,512]→[5,256]`, `[2,128]→[2,64]`, `[3,34]→[3,17]` | bf16 |
| `act_mul`  | `[5,256]`, `[2,64]`, `[3,17]`         | bf16, `act` = `silu`, `gelu`, `gelu_tanh`, `sigmoid`, `quick_gelu` (both checks) and `relu` (kernel only) |
| `cast`     | `[5,128]`, `[3,64]`, `[7,17]`         | bf16→bf16, bf16→f32, f32→bf16, f32→f32; `[5,128]` bf16→f16 |
| `rmsnorm`  | `[5,128]`, `[20,32]` (5 tokens × 4 heads), `[7,17]` | bf16 with an f32 gain at `wadd` 1 and `mult` 1; bf16 with a bf16 gain at `wadd` 0 and `mult` 0.5 |

The tiny fixtures are `docs/FIXTURES.md`'s bf16 fixture and its int8 conversion: 128 hidden, 4
heads of 32, feed-forward 256. Both store bf16 activations and f32 gains, and both set `wadd`,
`wadd_qk` and the final norm's `wadd` to 1.0 (`arch/qwen35_bf16/qwen35_bf16.cpp`), so
`rmsnorm`'s `[5,128]` and `[20,32]` cases are the fixtures' own calls: the block/final norm at
`n` 128 and the per-head q/k norm at `n` 32, both with the gain shifted by 1. The int8 fixture's
declared host graph also issues `sigmoid` and `mul` for the attention output gate and the bf16
fixture issues `silu_mul` for its feed-forward, so those are checked at the same bf16.

Neither tiny fixture issues `act_mul`, `scale`, `silu` or `cast`: the int8 graph folds its gate
pair into `sigmoid` + `mul` (docs/FIXTURES.md) and its activation quantiser is fused, and the
plugins that name the other four are the fp8 gated blocks (`act_mul`), Gemma 4 (`scale`) and the
3-D towers (`silu`). They are in this family because the op inventory lists them, so they are
checked too, at the fixtures' bf16 activations, and `cast` at the pairs a float conversion can
take.

A last check reads the registry: every op of the family resolves to a twin and a host adjoint, and
`adjoint_frozen` names operand 1 for `rmsnorm` and `scale` and nothing for the rest.

Beside the values, the test pins the refusals §1, §3 and §7 name, so a guard that stops refusing
fails here instead of passing quietly. Each of the three is exercised on both sides — no twin value
is written, and no cotangent reaches any read operand — and they are `scale` with an empty `s`,
`act_mul` with the `act` parameter absent and with one the schema does not name, and `rmsnorm` with
`wadd` and no gain (`wadd` 0 and no gain is the plain norm, and the test checks it is still
computed). `mul`'s broadcast rule is pinned on the twin's side alone: a `b` that is neither all of
`a` nor one row of it writes nothing, so the walk never reaches the adjoint.

## 11. What is not in this family

The same kernel file carries `rmsnorm_p` (no gain), `rmsnorm_add` (a fused residual),
`mul_weight`, `scale_rows`, `add`, and the fp8 fusions in `ref_quant.cpp`; the rest of the unary
activations live beside them. `add` has its own file (`src/adjoint/host/add.cpp`); the others
belong to whoever owns that issue. This family is the eight ops above, and `twin_find` /
`adjoint_find` return nothing for a name no file registers.
