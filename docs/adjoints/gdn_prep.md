# The gdn_prep family: `gdn_conv_prep` and `gdn_gated_rmsnorm`

The gated delta net's two ops outside the recurrence, as host adjoints. Source:
`src/adjoint/host/gdn_prep.cpp`. Tests: `tests/host/adjoint_gdn_prep.cpp`. The
recurrence around them, and the derivation of its adjoints, is
`docs/adjoints/gdn-math.md`; the op schemas are radiance's `docs/OPS.md` (`gdn`
section) with the kernels in `libref/ref_gdn.cpp`.

Radiance commit: `451b136874084650992668bed9e8662deb58de88`.

## 0. What the records are

Both ops are recorded with every operand the schema names. The optional ones
(`b`, `a`, `b_gate`, `cache_idx`, `has_init`) are present in each record, so the
twin implements the form those operands describe: `x` carries the convolution's
channels only, and the gates arrive in their own operands — the path every
current caller takes. The fused-row form, where `x` also carries `a` and
`b_gate` in its last `2Hv` columns, is not modelled here, and neither is an
absent conv bias.

The conv op is recorded at a **prefill from position 0**: the carried convolution
state is zero. Two consequences throughout:

- `conv_state` is a **written** operand of these records. At a cold start the op
  reads no history, so there is nothing for it to be as a read; what it writes is
  the tail of `x`, and its cotangent is a route back into `x`.
- `has_init` is frozen at 0. At a cold start it does not vary, and it is an
  integer besides. The warm path, which would read the state the record has no
  read for, is outside what this family's twin computes.

`l2_eps`, `eps` and `softplus_thr` are f64 parameters the schema allows and the
tape's parameter list cannot carry (it holds integers and strings). The twin uses
the op's defaults — `l2_eps` 1e-6, `eps` 1e-6, `softplus_thr` 20 — and so does
the kernel, which is why the twin and the kernel agree on a record without them.

Frozen operands, per op, by schema position:

| op | frozen | why |
|----|--------|-----|
| `gdn_conv_prep` | `w` (1), `b` (2), `A_log` (3), `dt_bias` (4) | convolution and model weights |
| `gdn_conv_prep` | `cu` (6), `cache_idx` (9), `has_init` (10) | integers: no cotangent of an index means anything |
| `gdn_gated_rmsnorm` | `w` (2) | the norm weight |

Nothing frozen is ever written to, and nothing frozen ever contributes to a
cotangent.

## 1. `gdn_conv_prep`: the forward

Let `N` be the sequences of the batch, `cu[n]` their boundaries (`cu[0] = 0`,
`cu[N] = T` tokens), sequence `n` start at `bos = cu[n]` and run `slen =
cu[n+1] - cu[n]` tokens. `K` and `V` are the per-head widths, `Hg` and `H` the
head counts, so the conv row has

```
conv_dim = 2*Hg*K + H*V
```

channels: `q`'s, then `k`'s, then `v`'s. `W` is `conv_width`, `C` is `chunk`, and
`x` is `[M, xw]` with `xw >= conv_dim` (`x` may be the fused projection row, of
which only the first `conv_dim` columns are the convolution's).

### 1.1 The convolution and its activation

For token `t` of sequence `n` (global row `r = bos + t`) and channel `c`:

```
a[r,c] = b[c] + sum_{j=0}^{W-1} [ src >= 0 ] * x[bos+src, c] * w[c,j],
         src = t - (W - 1) + j
u[r,c] = silu(a[r,c]) = a[r,c] * sigmoid(a[r,c])
```

`src < 0` is the convolution's window reaching before the sequence start, where
the carried state is zero: the term is dropped. `u` lands in the outputs by
channel:

```
c <   Hg*K        -> q[r, c/K, c%K]
c < 2*Hg*K        -> k[r, (c - Hg*K)/K, (c - Hg*K)%K]
otherwise         -> v[r, (c - 2*Hg*K)/V, (c - 2*Hg*K)%V]
```

The kernel stores `u` in the output's bf16 dtype and reads *that* back before the
norm, so its `q` and `k` are normalised from the rounded values. The twin does
not round: the difference is about `2^-9` relative and the bf16 check tolerates
it (§4).

### 1.2 The l2 norm of q and k

Per token and head, over the `K` channels of that head:

```
ss[r,h]  = sum_d u[r,h,d]^2
inv[r,h] = 1 / sqrt(ss[r,h] + l2_eps)
q[r,h,d] = u[r,h,d] * inv[r,h]          (and the same for k)
```

`v` is not normalised.

### 1.3 The gate and beta

Per sequence and value head `h`:

```
run = 0
for t in 0..slen-1:
    if t % C == 0: run = 0
    run += -exp(A_log[h]) * softplus(a[cu[n]+t, h] + dt_bias[h])
    g[cu[n]+t, h]    = run
    beta[cu[n]+t, h] = sigmoid(b_gate[cu[n]+t, h])
```

with the reference's softplus and its threshold:

```
softplus(x) = x                       if x > 20
            = x + log1p(exp(-x))      if 0 < x <= 20
            = log1p(exp(x))           otherwise
softplus'(x) = 1                       if x > 20
             = sigmoid(x)              otherwise
```

### 1.4 The carried state

With `keep = W - 1` and `slot = cache_idx[n]`:

```
for c, for i in 0..keep-1:
    src = slen - (W - 1) + i
    conv_state[slot, c, i] = (src >= 0) ? x[bos + src, c] : 0
```

At a cold start the window's history is zero, so the state is the last `keep`
tokens of `x` — and zero for the positions before the sequence begins.

## 2. `gdn_conv_prep`: the adjoint

Cotangents arrive for the six written operands, in schema order:
`d_state`, `dq`, `dk`, `dv`, `dg`, `dbeta`. Each route below adds into the
cotangent of the operand it came from.

### 2.1 The l2 norm

With `u` the pre-norm value (`u = q * sqrt(ss + eps)`, recomputed from the
convolution) and `inv = (ss + eps)^{-1/2}`, the outward value is `u*inv`, so

```
du_d = inv * dq_d - inv^3 * u_d * sum_e (u_e * dq_e)
```

`inv^3` is `inv*inv*inv`, and the sum is over the `K` channels of the token and
head. The same holds for `k`, with its own `ss` and `inv`.

### 2.2 The activation

For every channel, `q`, `k` and `v` alike:

```
da[r,c] = du[r,c] * silu'(a[r,c]),      silu'(a) = sigmoid(a) * (1 + a * (1 - sigmoid(a)))
```

### 2.3 The convolution

```
for each output token and channel with da != 0:
    for j in 0..W-1:
        src = t - (W - 1) + j
        if src >= 0:
            dx[bos+src, c] += da[r,c] * w[c,j]
```

The weight's and the bias's cotangents — `dw[c,j] = da[r,c] * x[bos+src,c]` and
`db[c] = sum da` — are not taken: they are frozen.

### 2.4 The carried state

The state is the tail of `x` itself, so its cotangent goes straight back:

```
for c, for i in 0..keep-1:
    src = slen - (W - 1) + i
    if src >= 0:
        dx[bos+src, c] += d_state[slot, c, i]
```

### 2.5 The gate and beta

`g` is a within-chunk running sum of increments `inc_t = -exp(A_log[h]) *
softplus(a_t + dt_bias[h])`, so the cotangent of an increment is the sum of its
own `g`'s cotangent and every later `g`'s in its chunk:

```
for t from slen-1 down to 0:
    if t == slen-1 or (t+1) % C == 0: tail = 0
    tail += dg[cu[n]+t, h]
    da[cu[n]+t, h]     += tail * (-exp(A_log[h])) * softplus'(a + dt_bias[h])
    db_gate[cu[n]+t,h] += dbeta[cu[n]+t, h] * sigmoid(b) * (1 - sigmoid(b))
```

`A_log`'s and `dt_bias`'s cotangents are not taken: frozen. `cu`, `cache_idx` and
`has_init` are frozen too, and no route reads them for a value.

## 3. `gdn_gated_rmsnorm`

One row per (token, head): `x`, `z`, `w` and `o` are `[rows, channels]`, `w` is
`[channels]`, and `channels` is the value head's width.

### 3.1 The forward

```
ss[r] = sum_i x[r,i]^2
sc[r] = 1 / sqrt(ss[r]/channels + eps)
o[r,i] = x[r,i] * sc[r] * w[i] * act(z[r,i])
```

`act` is `silu` by default or `sigmoid`, as the op's parameter says.

### 3.2 The adjoint

With `do` the cotangent of `o` and, per row,
`A = sum_e (x[r,e] * w[e] * act(z[r,e]) * do[r,e])`:

```
dx[r,i] = w[i] * act(z[r,i]) * sc * do[r,i] - x[r,i] * sc^3 / channels * A
dz[r,i] = do[r,i] * x[r,i] * sc * w[i] * act'(z[r,i])
```

`A` already carries the row's weights and gates: the second term of `dx` takes
them from `A` alone, and multiplying it by `w[i] * act(z[r,i])` again would count
them twice.

where `act'` is `sigmoid(z) * (1 + z * (1 - sigmoid(z)))` for `silu` and
`sigmoid(z) * (1 - sigmoid(z))` for `sigmoid`. `dw` is frozen.

## 4. The checks

Both checks run at three shapes per op — the tiny fixture's gdn geometry and two
others:

| op | shapes (`name`: what varies) |
|----|------------------------------|
| `gdn_conv_prep` | `tiny-fixture`: `Hg=2, H=4, K=V=16, W=4, C=4`, 8 tokens, one sequence, so `conv_dim = 128` is the fixture's hidden width. `ragged-short`: two sequences (5 and 1 tokens), `W=3`, `C=2`, one shorter than the window. `one-head`: `Hg=H=1`, `K=4, V=8`, `W=2`, `C=7`. |
| `gdn_gated_rmsnorm` | `tiny-fixture`: 8 tokens of 4 heads, `channels=16` (the fixture's value head width). `wide`: 37 rows of 192. `narrow-sigmoid`: 3 rows of 8, `act=sigmoid`. |

- **Twin against the kernel.** Each record is launched through radiance's host
  kernel for the op (`libref`), and every written operand is compared with the
  twin's value at 0.02 absolute-plus-relative — bf16's rounding and the kernel's
  f32 accumulation fit inside it.
- **Adjoint against finite differences.** A random cotangent on every written
  operand, a random direction on every read operand the adjoint answers for, and
  a central difference of `<twin, cotangent>` at ±1e-6 against the adjoint's
  directional derivative, to 1e-6 relative.

## 5. The decode ops are stops

`gdn_conv_update`, `gdn_recurrent_update` and `gdn_conv_recurrent_update` (and
the `_had` form) write decode rows' `q, k, v, g, beta` and the caches, and no
prefill row's loss reads any of it: `docs/adjoints/gdn-math.md`, section 5, says
so and names each op. They are registered as stops. The registry has no kind of
its own for a stop, so a stop is a host adjoint row whose function adds nothing:
the lookup succeeds, which tells a deliberate stop from a family nobody has
written yet, and every cotangent stays as it was. No twin is registered — no
forward in the loss path runs them. The device side is left to the device adjoint
family, which will know better than a catch-all row here what it wants.
