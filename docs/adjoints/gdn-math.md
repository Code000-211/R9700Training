# The gated delta net: forward and adjoints as radiance computes them

Reference: radiance commit `451b136874084650992668bed9e8662deb58de88` —
`arch/common/rad_block_gdn_fp8.h`, `libref/ref_gdn.cpp`, and the `gdn_*` tables in
`docs/OPS.md`. Everything below is the reference's arithmetic, written in the state
conventions it uses. `tools/gdn_math_check.py` implements the per-token form, the
chunked form and both adjoints for one head at toy size and checks (plain Python,
no dependencies):

```
$ python3 tools/gdn_math_check.py
chunked == per-token        : 1.110e-16 (max abs), PASS
token-reverse == chunk-rev  : 0.000e+00 (worst ), PASS
kkt adjoint == finite diff  : 9.7e-10  (worst dg), PASS
scan adjoint == finite diff : 9.7e-10  (worst dg), PASS
ALL PASS
```

Every formula in this document (sections 2–4) is exactly what that script
validates: chunked equals per-token to 1.1e-16, the two reverse recursions agree
exactly, and every adjoint cotangent agrees with central finite differences to
~1e-9, comfortably inside the 1e-8 acceptance.

## 1. The layer as the code computes it

Qwen3.8-27B: `n_embd = 5120`, `Hg = 16` key heads, `H = 48` value heads, head
width `K = V = 128`, so `conv_dim = 2·Hg·K + H·V = 10240` and `v_dim = 6144`,
`chunk = 64`. 48 of the 64 layers are gated delta net layers
(`rad_block_gdn_fp8.h`). A value head `h` reads its key head `hk = h/3` (`grp =
H/Hg = 3`): q and k are shared by each group of three value heads, v, β and g are
per value head.

The block's `step()` (rad_block_gdn_fp8.h, `GdnFP8::step`) issues, in order:

| # | op (`ref_gdn.cpp`) | reads | writes |
|---|--------------------|-------|--------|
| 0 | the norm+quantiser `nq_h` and the in-projections | `w.h.x` `[T, 5120]` | `w.in.x` `[T, 10240+6144]` — columns `[0:10240)` are `[q\|k\|v]`, columns `[10240:16384)` are the output gate `z` |
| 0b| the a\|b projection (`op_ab`) | `w.h.x` | `w.ab` `[T, 96]` — 48 decay rows `a`, then 48 β rows `b` |
| 1 | `gdn_conv_prep` | `w.in.x` columns `[0:10240)`, conv weight `w [10240, 4]`, bias, `A_log [48]`, `dt_bias [48]`, `conv_state [10240, 3]` | `q, k` `[T, Hg, K]`, `v [T, H, V]`, `g, beta [T, H]`, rewritten `conv_state` |
| 2 | `gdn_kkt_solve` | `k, beta, g` | `A [T, H, chunk]` |
| 3 | `gdn_chunk_scan` | `q, k, v, A, g, beta`, the state cache (`h0`) | `o [T, H, V]` (`w.o.x`), the state cache (`ht`) |
| 4 | `gdn_gated_rmsnorm` | `o`, `z` (columns `[10240:16384)` of `w.in.x`), `w_out_norm [6144]` | `o` in place (the gated, normed value) |
| 5 | the out projection (`out.step`) | the quantised `o` | `w.h.x`-shaped output, into the residual stream |

`gdn_conv_prep` in detail (four phases in the reference):

1. **Convolution.** Every one of the `conv_dim` channels `c` runs its own causal
   conv of width `conv_width = 4` with bias, then `silu`. With history `hist`
   from the conv cache (zeros when the slot holds no history):

   `xconv[t][c] = silu(bias[c] + Σ_{j<4} x[t−3+j][c]·w[c][j])`, reading row history
   from the cache for `t < 3`. A [T, Hg, K] and the first `2·Hg·K` channels, k the
   next, v the last `H·V`.

2. **q and k head norm.** Per token, per key head: `x ← x / sqrt(Σ_d x_d² +
   l2eps)`, an L2 norm over the whole 128-wide head (no division by the width,
   `l2eps = 1e-6`). **The conv output is rounded to bf16 before the norm reads it
   back** — the backprop must reproduce that rounding boundary.

3. **Decay and write strength.** Per token, per value head:

   ```
   g_t   = g_{t-1} - exp(A_log[h]) · softplus(a_t + dt_bias[h])   per-step term
   beta_t = sigmoid(b_t)
   ```

   where `a_t, b_t` are `w.ab`'s columns. **The per-token log decay `g_t =
   -exp(A_log)·softplus(a_t + dt_bias)` is accumulated into a per-chunk,
   per-sequence cumsum in prefill**: `run` resets to 0 at every `t % chunk == 0`,
   so `g[t]` is the decay from the start of t's chunk **through** t. Decode
   instead uses the per-step exponent directly (section 5).

4. **Conv state rewrite.** The cache slot keeps the last `width−1 = 3` x rows
   (the previous sequence's tail when the sequence is shorter than 3), read first
   and written after, in place.

`gdn_gated_rmsnorm` (`channels = 6144`, rows = `[T·H] tokens·heads`): per row of
`o`, `o ← o · w[i] · act(z[i]) / sqrt(Σ (o_i²)/6144 + eps)`, `act` = `silu` for
this architecture (the block's default; a `sigmoid` variant exists for
Qwen4-Exp and Qwen3.8 does not use it).

## 2. The recurrence

Throughout: `S [K, V]` — the recurrent state, per value head, `k [K]`, `q [K]`,
`v [V]`, `β`, a log decay `γ ≥` decay is applied as `e^γ`; `scale` (the query
scale, per arch config) multiplies only the read.

### 2.1 Per token (the decode op's arithmetic)

Decode's `gdn_recurrent_update` runs, per token, with the **per-step** log decay
`γ = -exp(A_log)·softplus(a + dt_bias)`, and an L2 norm of q and k fused in its
prologue:

```
Sk = e^γ · S                    (decay first, over every (d, e))
u  = β · (v − Skᵀk)             (sk2[e] = Σ_d Sk[d][e]·k[d])
S  = Sk + k ⊗ u                 (rank-one write)
o  = scale · Sᵀq                (orow[e] = Σ_d S[d][e]·q[d])
```

and `o` then goes to the gated norm of the same op's epilogue (or that of the
next prefill stage). The op also updates the convolution cache (`gdn_conv_update`
or the fused `gdn_conv_recurrent_update`).

The loss side sees the same math in prefill, token by token, with the **chunk
cumsum** g: the decay of token `t` inside a chunk is the per-step difference

```
γ_t = g_t − g_{t−1}   (same chunk),    γ_t = g_t   (t the chunk's first token)
```

so the per-token and chunked forms compute the same layer (validated to 1.1e-16
by the script's first check).

### 2.2 The chunked form radiance evaluates in prefill

Per chunk of `L = chunk = 64` tokens, per value head, with the entering state
`S0` (from the cache, or zero):

**`gdn_kkt_solve`** computes, from `k, beta, g` only:

```
M[i][j] = β_i · (k_i·k_j) · e^{g_i − g_j}          (j < i, strictly lower)
A = (I + M)^{-1}, unit lower triangular, fp32
```

row by row by forward substitution:

```
A[i][j] = δ_ij − Σ_{l<i} M[i][l] · A[l][j]         (j ≤ i), upper triangle written 0
```

The exponents are never split about a chunk reference: only `j < i` is kept and
`g` decreases along the chunk, so `g_i − g_j ≤ 0` cannot overflow. `A` is stored
per token row `[T, H, chunk]` (the row of token `t`, offset `i`, holds row i of
the i-th chunk's A).

**`gdn_chunk_scan`** then does, per chunk:

```
W[j]  = β_j · (v_j − e^{g_j} · S0ᵀ k_j)                       [L, V]
V'[i] = Σ_{j≤i} A[i][j] · W[j]                                [L, V]   (A W)
o[i]  = scale · ( e^{g_i} · S0ᵀ q_i
                + Σ_{j≤i} e^{g_i − g_j} · (q_i·k_j) · V'[j] ) [V]
S1    = e^{g_L−1} · S0 + Σ_j e^{g_L−1 − g_j} · k_j ⊗ V'[j]    [K, V]
```

`S1` is written back to the state cache (`ht`) and is the next chunk's `S0`; the
`h0` operand is an optional external initial state (decode's cache slot, read
from `state_idx` column 0). The scan's state is held **transposed** in kernels
(`St[d][e]`), which the backprop must respect for memory layout but not for math.

Correctness of the two forms is the script's checks 1 and 2: the chunked
formulas above reproduce the per-token recursion exactly, and the reverse
recursion of section 4.4 agrees with the chunked adjoint exactly.

## 3. Adjoint of `gdn_kkt_solve` — an op on its own operands

Inputs `k [L, K], β [L], g [L]`; output `A [L, L]` (per value head; q, v, cu are
shape operands). Given `Ac [L, L]` (the output cotangent; only the stored
(`j ≤ i`) entries are read), start from the substitution row
`A[i][j] = δ_ij − Σ_{l<i} M[i][l] A[l][j]`, whose diagonal contributes nothing
(`A[i][i] ≡ 1`) and whose upper triangle the op never reads:

```
dM[i][l] = Σ_{j ≤ l} Ac[i][j] · A[l][j]                     (j ≤ l < i)
```

and then, from `M[i][l] = β_i (k_i·k_l) e^{g_i − g_l}`:

```
dβ_i    = Σ_{l<i} dM[i][l] · (k_i·k_l) · e^{g_i − g_l}
dk[i][d] += Σ_{l<i} dM[i][l] · e^{g_i − g_l} · k_l[d]      (i's role, the k_i row)
dk[l][d] += Σ_{i>l} dM[i][l] · e^{g_i − g_l} · k_i[d]      (l's role, the k_l row)
dg_i    =  Σ_{l<i} dM[i][l] · e^{g_i − g_l} · (k_i·k_l)    (i is the minuend)
dg_l    −= Σ_{i>l} dM[i][l] · e^{g_i − g_l} · (k_i·k_l)    (l is the subtrahend)
```

Both `dg_i` roles land on the same accumulator (i may be both minuend and
subtrahend for different pairs). `cu` needs no cotangent. `β`'s cotangent is
accumulated per token (one scalar per (t, h)); note **`β` is also a scan
operand**, and the two contributions add in the block's backward. Likewise for
`g` and `k`.

## 4. Adjoint of `gdn_chunk_scan` and the reverse per-token recursion

Inputs `q, k [T, Hg, K], v [T, H, V], A [T, H, chunk], g, β [T, H], h0`; outputs
`o [T, H, V]` and the carried state `ht` (= next chunk's `S0`). Per chunk, given
`oc [L, V]` (o's cotangent) and `St1 [K, V]` (the carried state's cotangent, i.e.
o's cotangent is `oc` and S1's is `St1`, both already accumulated from the
surrounding graph):

**Stage 1 — from `S1 = e^{g_L−1} S0 + Σ_j e^{g_L−1−g_j} k_j ⊗ V'[j]`:**

```
dh0[d][e]  += e^{g_L−1} · St1[d][e]
dg[L−1]    += Σ_{d,e} St1[d][e] · (e^{g_L−1}·S0[d][e])            (minuend term, see below)
dk[j][d]   += e^{g_L−1−g_j} · Σ_e St1[d][e] · V'[j][e]            (each j)
dg[j]      −= Σ_d dk1-term[j][d] · k_j[d]                          (subtrahend term)
dVp[j][e]  += e^{g_L−1−g_j} · Σ_d St1[d][e] · k_j[d]
```

Concretely, with `gs = e^{g_L−1−g_j}`, `w_j[d] = gs·Σ_e St1[d][e]·V'[j][e]`:
`dk[j][d] += w_j[d]`, `dg[j] −= Σ_d w_j[d]·k_j[d]`, and the `g_L−1` minuend part
of the `S0` term: `dg[L−1] += Σ_{d,e} gs_all·St1·(e^{g_L−1}S0[d][e])` summed over
j's subtrahend part is folded in the implementation as `dg[j] -= gw` plus
`dg[L-1] += gw` per j — the validated code's form:
`gw_j = Σ_d w_j[d]·k_j[d]`, then `dg[j] -= gw_j` and `dg[L−1] += gw_j`.
**For a middle chunk, neither `S0`'s nor `S1`'s `g` exponents mention the
previous chunk's `g`:** `dh0` is just `e^{g_L−1}`-scaled plus the o-path and
W-path terms below (the FD check pins this: the only g-index touching `dh0` is
`g[L−1]`).

**Stage 2 — from `o[i] = scale·(e^{g_i} S0ᵀq_i + Σ_{j≤i} e^{g_i−g_j}(q_i·k_j)V'[j])`:** with
`S0oi[d] = Σ_e S0[d][e]·oc[i][e]` and `phi = e^{g_i−g_j}`, `qk = q_i·k_j`,
`vo = Σ_e V'[j][e]·oc[i][e]`:

```
dq[i][d]   += scale·e^{g_i}·S0oi[d]                                (state read)
dh0[d][e]  += scale·e^{g_i}·q[i][d]·oc[i][e]                       (state read)
dg_i       += scale·e^{g_i}·(q_i·S0ᵀoc_i)
dq[i][d]   += scale·phi·vo·k_j[d]                                  (q_i·k_j)
dk[j][d]   += scale·phi·vo·q_i[d]                                  (q_i·k_j)
dVp[j][e]  += scale·phi·qk·oc[i][e]                                (V'[j])
dg_j       −= scale·phi·qk·vo
dg_i       += scale·phi·qk·vo
```

**Stage 3 — from `V' = A W`,** first collecting `dVp` (stages 1+2 accumulated it):

```
dAc[i][j] = Σ_e dVp[i][e] · W[j][e]                    (j ≤ i; upper untouched)
dW[j][e]  = Σ_{i ≥ j} A[i][j] · dVp[i][e]              (A's transpose, restricted)
```

**Stage 4 — from `W[j] = β_j(v_j − e^{g_j} S0ᵀ k_j)`,** with `r = dW[j][e]`:

```
dv[j][e]   += β_j · r
dβ_j       += Σ_e r · W[j][e] / β_j                    (= r·(v_j − e^{g_j}S0ᵀk_j))
dk[j][d]   −= β_j·e^{g_j}·S0[d][e]·r
dh0[d][e]  −= β_j·e^{g_j}·k_j[d]·r
dg_j       −= β_j·e^{g_j}·(S0ᵀk_j)[e]·r                (summed over e with r)
```

All six accumulators (`dq, dk, dv, dβ, dg, dAc, dh0`) are the FD-validated
adjoint: every one agrees with central differences on the full chunk output
(cotangents on both `o` and `ht`) to ≤1e-9 relative. The `A` cotangent of the
scan feeds the `Ac` cotangent of the kkt adjoint, section 3.

### 4.4 The reverse per-token recursion (the adjoints must equal it)

Per chunk, running tokens `t = L−1 … 0`, `Sst = St1` initially (state cotangent
arriving from the future). Per token, `D = e^{γ_t}` (`γ_t` in-chunk per-step,
section 2.1), `X = D·S_prev` the forward decayed state, `u = β(v − Xᵀk)`:

```
St    = Sst + scale·q_t ⊗ oc_t                    (S~_t, adds the read o_t = scale·S_tᵀq_t)
dq_t   = scale·S_t^T oc_t
ũ      = Stᵀk_t     (u's cotangent, summed over the write)
dv_t   = β·ũ
dβ_t   = Σ_e ũ[e]·(v_t − Xᵀk_t)[e]                (= ũ·(v − sk2))
dk_t   = St·u_fwd − β·X·ũ        with u_fwd = β(v − Xᵀk)  (FORWARD u, not ũ —
                  the write's δ_{dd'}u[e] term reads the forward u)
X~     = St − β·k_t ⊗ ũ
γ~_t   = Σ_{d,e} St[d][e]·X[d][e] − β·Σ_e (Xᵀk_t)[e]·ũ[e]     (X~ is NOT used here:
                  the −βk⊗ũ part belongs to the S_prev cotangent, not to γ)
Sst    = D·X~                                     (S~_{t−1})
```

then the fold through the in-chunk cumsum (`γ_t = g_t − g_{t−1}`, reset at the
chunk start):

```
g~_t = γ~_t − γ~_{t+1}   (t and t+1 in the same chunk)
g~_t = γ~_t              (t the last token of its chunk — nothing after it
                          subtrahends g_t within the chunk)
```

The script's second check requires this recursion's output to equal the chunked
adjoint of section 4 exactly — they do, to 0.0. The two easy-to-fuse-wrong terms
are called out above and are what the FD run found: the k-cotangent's first term
reads the **forward** `u`, and the γ-cotangent reads `S~_t` (**not** `X~`) against
`X`; the `−β k⊗ũ` correction is only for the state going backwards.

For a full sequence the reverse recursion is run chunk by chunk: each chunk's
adjoint consumes `St1` = the next chunk's `dh0` and produces the current chunk's
`dh0`, so the state cotangent flows right-to-left across chunk boundaries
(`chunk_reverse` in the script does exactly this and equals `token_reverse`
exactly).

## 5. Ops that only prepare state for a later decode step

`GdnFP8::step()` (rad_block_gdn_fp8.h) splits the batch with `batch_split`:
decode rows are sequences `[0, D)` — the decode path is issued **only inside
`if (D > 0 ...)` blocks**:

```
} else if (D > 0) {
    RAD_ISSUE_N(c, op_conv_update, ...);   /* gdn_conv_update            */
    RAD_ISSUE_N(c, op_recur, ...);         /* gdn_recurrent_update       */
}
if (D > 0 && op_cr && qd <= cr_rows) {     /* gdn_conv_recurrent_update (or _had) */
```

A pure prefill step has `D = 0` (`batch_split` gives decode count 0 when
`phase == RAD_PHASE_PREFILL`): none of the four ops is even launched, and the
prefill half issues only `op_conv_prep`, `op_kkt`, `op_scan`, `op_gnorm` plus
the projections. In a mixed step, the decode half works on rows `[0, DT)`, the
prefill loss reads rows of the prefill chunks `[DT, T)`: `gdn_conv_update`
writes only `q, k, v` rows `[0, DT)` and the conv cache; `gdn_recurrent_update`
writes only the state cache and `o` rows `[0, DT)`; the fused
`gdn_conv_recurrent_update` (and its `_had` form — the same op with a
Walsh–Hadamard rotation of the out projection's quantised codes, chosen when
`w.o.q8_rot`) writes the same things. `q, k, v` of the decode rows are consumed
by the recurrent update in the same step; `o` rows `[0, DT)` feed only the
decode logits. No prefill row's loss touches any of these writes: the four ops
only prepare state for a later decode step. The prefill loss reads, besides the
prefill projections and conv, exactly `gdn_kkt_solve`'s `A` and
`gdn_chunk_scan`'s `o` and the state cache — the two ops sections 3 and 4
derive adjoints for.

## 6. What the backward must keep, per chunk and per token

Geometry: `T = 4096`, `chunk = 64` (64 chunks), `Hg = 16`, `H = 48`,
`K = V = 128`. Element counts per value head and per token/chunk:

| item | shape | elements | bytes/elt | total |
|------|-------|----------|-----------|-------|
| q, k (bf16, per key head) | `[4096, 16, 128]` ×2 | 16,777,216 | 2 | 32 MiB |
| v (bf16) | `[4096, 48, 128]` | 25,165,824 | 2 | 48 MiB |
| g, β (f32) | `[4096, 48]` ×2 | 393,216 | 4 | 1.5 MiB |
| A (f32) | `[4096, 48, 64]` | 12,582,912 | 4 | 48 MiB |
| chunk-boundary states (f32) | `64 × [48, 128, 128]` | 50,331,648 | 4 | 192 MiB |
| o, pre-norm (bf16; norm is in place) | `[4096, 48, 128]` | 25,165,824 | 2 | 48 MiB |
| normed/quantised o for the out projection (E4M3) | `[4096, 6144]` | 25,165,824 | 1 | 24 MiB |
| `w.in.x` (bf16; the projection's output: conv input + gate z) | `[4096, 16384]` | 67,108,864 | 2 | 128 MiB |
| normed input `w.h.x` for the projections (E4M3 + scale planes) | `[4096, 5120]` | 20,971,520 | 1 | 20 MiB |
| `w.ab` (bf16) | `[4096, 96]` | 393,216 | 2 | 0.75 MiB |

State-boundary notes. The scan adjoint needs every chunk's entering state `S0`.
These are the states the scan wrote into the cache (one `[48, 128, 128]` state
per chunk boundary = 786,432 elements = 3 MiB f32 each; 64 of them for a 4096
sequence, or 63 if the step's `h0` — the state entering the first chunk — is
already pinned in the KV state cache and need not be copied). The alternative is
to not save the interior states and recompute them in the backward by running
the state-carry formula of section 2.2 chunk by chunk from `h0`: costs one extra
state-carry pass (the cheapest part of the scan), saves ~189 MiB, and only the
final cached state (`ht`) must live anyway.

Per-token items the kkt/scan adjoints consume again: `q, k, v, g, β, A` — all of
them read directly from the forward's saved tensors; no per-token workspace is
needed beyond the cotangent buffers, whose sizes are the same as the tensors
they mirror (`dq, dk` 32 MiB, `dv` 48 MiB, `dA` 48 MiB, `dg, dβ` 1.5 MiB, one
`[48, 128, 128]` state-cotangent slot at a time = 3 MiB).

Delta-net core subtotal (q, k, v, g, β, A, states): 321.5 MiB saved for the
backward per layer. The largest single item is the state-cotangent footprint
(states: 192 MiB), which the recompute variant trades for one cheap chunk pass.
The projection-side rows (`w.in.x`, `w.h.x`, `o`) are already kept by the
projection/norm backward in any case and are shared, not additional state of the
delta net's own bookkeeping.
