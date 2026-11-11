# `gdn_chunk_scan`: the chunked gated-delta-net scan and its adjoint

The scan carries the gated delta net's `[K, V]` state from chunk to chunk and
produces the layer's output rows. This is the family `gdn_scan`; the twin and
host adjoint live in `src/adjoint/host/gdn_scan.cpp`, the test in
`tests/host/adjoint_gdn_scan.cpp`. The derivation and the checks against the
per-token recursion are in `docs/adjoints/gdn-math.md`, sections 2.2, 4 and 4.4.

## The op

Schema positions and shapes, one sequence set with `T = cu[S]` tokens and `S`
sequences:

| pos | operand | shape | dtype | role |
|-----|---------|-------|-------|------|
| 0 | `q` | `[T, Hg, K]` | bf16 | read |
| 1 | `k` | `[T, Hg, K]` | bf16 | read |
| 2 | `v` | `[T, H, V]` | bf16 | read |
| 3 | `A` | `[T, H, chunk]` | bf16 | read, unit lower triangular within a chunk |
| 4 | `g` | `[T, H]` | f32 | read, within-chunk cumsum |
| 5 | `beta` | `[T, H]` | f32 | read |
| 6 | `h0` | `[Nslots, H, V, K]` | f32 | read, initial state; zero here |
| 7 | `cu` | `[S+1]` | i32 | read, sequence boundaries |
| 8 | `o` | `[T, H, V]` | bf16 | written |
| 9 | `ht` | `[Nslots, H, V, K]` | f32 | written, final state |
| 10 | `state_idx` | `[S]` | i32 | optional read, state-cache slot |

`H = Hg * grp`, `grp` value heads per key head. `h0` and `ht` are
`[Nslots, H, V, K]`, one block per state-cache slot; `state_idx` names the slot
row `n` owns, and without it row `n` owns slot `n`. `h0`, `cu` and `state_idx`
are structure, not data: the host adjoint registers them frozen. A prefill
starts from the zero state, so `h0` has no cotangent; when it is nonzero, its
cotangent is still discarded. The state is held transposed as `St[d][e]` in the
kernels and in this code, while `ht`/`h0` are stored `[slot, head, e, d]`.

`scale` is the op's optional real parameter. The kernel's own default is `1`,
but the architecture issues it as `1/sqrt(head_k)` at every delta-net layer, and
the tape carries parameters as integers and strings, so the f64 does not survive
a recorded step: a record off the tape carries no `scale` at all. The twin reads
`scale` when the record names it (an integer or a decimal string) and otherwise
uses `1/sqrt(head_k)`, the value the model ran with. Falling back to the
kernel's `1` instead would make every scan's `o` `sqrt(head_k)` too large (11.3x
at `K = 128`, 4x at the tiny fixture's `K = 16`). Each kernel check states a
`scale` of its own, and `adjoint_gdn_chunk_scan_defaults_scale_to_the_architectures`
pins the fallback.

Only the lower triangle `j <= i` of each `A` row is read. The upper triangle is
written as zero by `gdn_kkt_solve` and is not part of the scan; its cotangent is
zero.

## The forward twin

Per value head `h`, sequence `n` with token range `[cu[n], cu[n+1])`, and each
chunk of `L` tokens (`L = chunk` except for a short last chunk), with entering
state `S0[d][e]`:

```
W[j][e]  = beta_j * ( v[j][e] - e^{g_j} * sum_d S0[d][e] k[j][d] )
V'[i][e] = sum_{j<=i} A[i][j] * W[j][e]
o[i][e]  = scale * ( e^{g_i} sum_d S0[d][e] q[i][d]
                   + sum_{j<=i} e^{g_i-g_j} (sum_d q[i][d] k[j][d]) V'[j][e] )
S1[d][e] = e^{g_{L-1}} S0[d][e]
           + sum_j e^{g_{L-1}-g_j} k[j][d] V'[j][e]
```

`S1` is `ht` for the sequence after the last chunk and `S0` for the next chunk.
The twin evaluates these in f64 and is checked against radiance's reference
kernel within bf16 tolerance (`2^-9` relative, 0.02 in the test).

## The adjoint

Given cotangents `oc[i][e]` on the chunk's `o` rows and `St1[d][e]` on its output
state (`ht` for the last chunk, otherwise the cotangent produced by the next
chunk), a reverse scan over the chunk produces `dq, dk, dv, dA, dg, dbeta` and
the state cotangent `dS0` of the previous chunk. A sequence's scan runs from its
last chunk to its first and drops the `dS0` leaving the first chunk, because
`h0` is zero and frozen.

**Stage 1, from `S1`.** With `gl = g[L-1]` and `gs_j = e^{gl-g_j}`:

```
dS0[d][e]  += e^{gl} St1[d][e]
dg[L-1]    += sum_{d,e} St1[d][e] e^{gl} S0[d][e]
w_j[d]      = gs_j sum_e St1[d][e] V'[j][e]
dk[j][d]   += w_j[d]
gw_j        = sum_d w_j[d] k_j[d]
dg[j]      -= gw_j
dg[L-1]    += gw_j
dVp[j][e]  += gs_j sum_d St1[d][e] k_j[d]
```

**Stage 2, from `o`.** With `S0oi[d] = sum_e S0[d][e] oc[i][e]`,
`phi = e^{g_i-g_j}`, `qk = q_i.k_j` and `vo = sum_e V'[j][e] oc[i][e]`:

```
dq[i][d]   += scale e^{g_i} S0oi[d]
dS0[d][e]  += scale e^{g_i} q[i][d] oc[i][e]
dg[i]      += scale e^{g_i} sum_d q[i][d] S0oi[d]
dq[i][d]   += scale phi vo k_j[d]
dk[j][d]   += scale phi vo q_i[d]
dVp[j][e]  += scale phi qk oc[i][e]
dg[j]      -= scale phi qk vo
dg[i]      += scale phi qk vo                  (j <= i)
```

**Stage 3, from `V' = A W`.**

```
dA[i][j]   = sum_e dVp[i][e] W[j][e]           (j <= i; upper stays zero)
dW[j][e]   = sum_{i>=j} A[i][j] dVp[i][e]      (A transposed, restricted)
```

**Stage 4, from `W`.** With `sk2_j[e] = sum_d S0[d][e] k_j[d]` and
`r = dW[j][e]`:

```
dv[j][e]   += beta_j r
dbeta[j]   += sum_e r (v[j][e] - e^{g_j} sk2_j[e])     (= sum_e r W[j][e]/beta_j)
dg[j]      -= beta_j e^{g_j} sum_e r sk2_j[e]
dk[j][d]   -= sum_e beta_j e^{g_j} S0[d][e] r
dS0[d][e]  -= beta_j e^{g_j} k_j[d] r            (r is already r[j][e]: no sum)
```

The twin's own `W` and `V'` are recomputed from the forward operands and the
entering state, so the reverse scan never reads the forward's `o`. Every
accumulator above is checked by central differences of the twin with the two
outputs `o` and `ht` live, to `1e-8` relative (the test's `fd_tol`, tighter than
the family's `1e-6`); `A`'s upper triangle must come back zero because the
forward never reads it.

The state cotangent is per head: `ht` and `h0` have one `[K, V]` block per
`(sequence, value head)`, and the chunk loop advances `St1` through the chunks
of that block only.

## What the adjoint needs from the forward

The reverse scan needs, per chunk: the `q, k, v, A, g, beta` slices and the
chunk's entering state `S0`; it recomputes `W` and `V'` from them and never needs
the forward `o` or the forward `ht` values. This implementation starts each
sequence from `h0`'s zero state and replays the state-carry formula once per
head to obtain every `S0`, so the caller's forward need not save interior
states.

At the real prefill geometry `T = 4096`, `chunk = 64` (64 chunks), `Hg = 16`,
`H = 48`, `K = V = 128`:

| per chunk | elements | bytes |
|-----------|----------|-------|
| `q` (`[64,16,128]` bf16) | 131,072 | 262,144 |
| `k` (`[64,16,128]` bf16) | 131,072 | 262,144 |
| `v` (`[64,48,128]` bf16) | 393,216 | 786,432 |
| `A` (`[64,48,64]` bf16) | 196,608 | 393,216 |
| `g` (`[64,48]` f32) | 3,072 | 12,288 |
| `beta` (`[64,48]` f32) | 3,072 | 12,288 |
| **operands from the forward** | | **1,728,512 (1.65 MiB)** |
| entering state `S0` (`[48,128,128]` f32), if saved | 786,432 | 3,145,728 (3 MiB) |
| **with saved states** | | **4,874,240 (4.65 MiB)** |

Over the 64 chunks of a 4096-token sequence the operands are 110,624,768 bytes
(105.5 MiB); saving every boundary state adds 201,326,592 bytes (192 MiB), for
311,951,360 bytes (297.5 MiB) in total. The recompute variant used here keeps no
boundary states in the forward. During the backward it holds one head's
`St1`/`dS0` (`128*128*8 = 131,072` bytes each in f64) and, for the head being
processed, that head's 64 entering states (`64*128*128*8 = 8,388,608` bytes,
8 MiB); they are released before the next head.
