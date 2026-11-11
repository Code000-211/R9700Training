# `gdn_kkt_solve`: the adjoint

Reference: radiance `libref/ref_gdn.cpp` (`ref_gdn_kkt_solve`) and the `gdn_kkt_solve` row of
`libref/ref_registry.cpp`. The host adjoint is `src/adjoint/host/gdn_kkt.cpp`, its test
`tests/host/adjoint_gdn_kkt.cpp`. The chunked delta-rule pass it belongs to is written out in
`docs/adjoints/gdn-math.md`; this file is the op on its own operands.

## 1. The op, as the reference computes it

`gdn_kkt_solve` is the delta rule's in-chunk solve: inside one chunk of `chunk` tokens it inverts
the unit lower triangular system the delta rule would otherwise solve step by step, so that the
whole chunk can be scanned in one pass.

| operand | dtype | shape | read/write |
|---|---|---|---|
| `k` | bf16 | `[rows, key_heads, head_k]` | read |
| `beta` | f32 | `[rows, heads]` | read |
| `g` | f32 | `[rows, heads]` | read |
| `cu_seqlens` | i32 | `[n_seq + 1]` | read |
| `A` | bf16 | `[rows, heads, chunk]` | written |

Parameters: `M` (rows), `head_k` (derived from `k`'s last dimension), `chunk` (derived from `A`'s
last dimension; `RAD_DERIVED` lets a record leave it out and the shape then carries it).
`grp = heads / key_heads` value heads share each key head, and `hk = h / grp` is value head `h`'s
key head.

`rows = cu_seqlens[n_seq]` is every token of every sequence in the step. A dense token `tok` sits
at `i = tok − cu_seqlens[s]` inside sequence `s`, and its chunk is `c = i / chunk`, with `c0 = c ·
chunk` the chunk's first token and `L = min(chunk, len_s − c0)` its length. `L < chunk` only in a
sequence's last chunk. A zero-length sequence has no chunks.

For every (sequence `s`, value head `h`, chunk `c`), with `i, l < L`, load

    kc[i, d] = k[cu_seqlens[s] + c0 + i, hk, d]
    bc[i]    = beta[cu_seqlens[s] + c0 + i, h]
    gc[i]    = g[cu_seqlens[s] + c0 + i, h]

and form the strict lower block and its unit lower triangular inverse

    phi[i, l] = exp(gc[i] − gc[l])                                   (the chunk's decay)
    M[i, l]   = bc[i] · <kc[i], kc[l]> · phi[i, l]                    l < i, and 0 elsewhere
    A         = (I + M)^{-1}       i.e.   A[i, j] = δ_ij − Σ_{l<i} M[i, l] · A[l, j]

`gc` is the log decay, cumulative across the chunk (the forward driver feeds it as `GATE_CUMSUM`),
so `phi[i, l] ≤ 1`: the decay only ever attenuates what an earlier token contributes. Each token's
row of the inverse is written where that token lives:

    A[cu_seqlens[s] + c0 + i, h, j] = A_chunk[i, j]      j ≤ i
    A[cu_seqlens[s] + c0 + i, h, j] = 0                  j > i

so the column `j` is the position inside the token's own chunk, not a column of one matrix shared
by the whole sequence. Two consequences the adjoint has to keep:

- **Rows past the last chunk's length are left alone** by the kernel (they are not written at all,
  and the buffer starts zeroed). They are not tokens: token `cu_seqlens[s] + c0 + i` for `i ≥ L`
  belongs to the next sequence, and this op writes nothing there.
- `A` is unit lower triangular per token row, `A[i, i] = 1`, and nothing above the diagonal is
  read or written.

Cost per (sequence, head, chunk): `L²·head_k/2` for the block `M` and `L³/6` for the substitution,
so `O(chunk²·(head_k + chunk))` a chunk.

## 2. The f64 twin

`gdn_kkt_twin` runs the same arithmetic in f64 over the values the record carries, and writes `A`
as `[rows, heads, chunk]` with the rows past a last chunk's length and the entries above the
diagonal zero. It reads the operands exactly as recorded: `cu_seqlens` addresses rows and is
`llround`ed, `chunk` comes from the parameter when it is there and from `A`'s last dimension when
it is not. A record the op cannot compute -- `head_k` differing from `k`'s last dimension, a shape
that does not hold the record's row count, `cu_seqlens` that does not start at 0, end at `rows` or
ever go back -- is declined: the twin writes no output and the adjoint adds no cotangent.

The twin is checked against the reference kernel to bf16 tolerance (`0.02`, relative to the
kernel's value), i.e. the kernel's f32 arithmetic and its bf16 store of `A`. The test fills the
keys on bf16's own grid so that the twin and the kernel read the same keys and the tolerance is
spent on the arithmetic rather than on an accidental quantisation of the input.

## 3. The adjoint: a transposed triangular solve

Write `E = I + M`, so `A = E^{-1}` and, differentiating `A·E = I`,

    dA = −A · dE · A

With `Ac = ∂loss/∂A` (zero above the diagonal and on the rows the forward left alone), and only the
strict lower part of `dE` free:

    ∂loss/∂A = Ac  ⇒   ∂loss/∂E = −(Aᵀ · Ac · Aᵀ) =: Et

`Et` is the adjoint of a triangular solve written as a transposed triangular solve against
another, and it is evaluated in the tensors the forward wrote: the `A` it needs is the op's own
output, read back, not formed again. Two triangular products, each over the half-triangle it needs:

    Y[p, l]  = Σ_{j≤l} Ac[p, j] · A[l, j]                 l ≤ p          (cost L³/6)
    Et[i, l] = −Σ_{p≥i} A[p, i] · Y[p, l]                 l < i          (cost L³/6)

which is `L³/3` a chunk against the forward's `L³/6`: the same order, a factor of two. Written as
the double sum it stands for, this would be `O(L⁴)`, and recomputing `A` instead of reading it
would cost another `L³/6` on top of a solve the forward has already done.

Both products only ever need the lower triangles: `A[l, j]` is zero for `j > l` and `Ac[p, j]` for
`j > p`, and `p ≥ i > l` in the second, so no entry above the diagonal is touched.

## 4. From `Et` to the operands

With the block entry's two factors separated,

    c[i, l]   = <kc[i], kc[l]>           (the keys' cosine, before beta and the decay)
    M[i, l]   = bc[i] · c[i, l] · phi[i, l]

the cotangents are, for every `l < i` with `e = Et[i, l]`:

    ∂loss/∂beta[i]  += e · c[i, l] · phi[i, l]
    ∂loss/∂g[i]     += e · M[i, l]                     (g[i] enters phi[i, l] as exp(gc[i] − gc[l]))
    ∂loss/∂g[l]     −= e · M[i, l]
    ∂loss/∂k[i, d]  += e · bc[i] · phi[i, l] · kc[l, d]
    ∂loss/∂k[l, d]  += e · bc[i] · phi[i, l] · kc[i, d]

`beta[i]` needs no division by `beta[i]`: `M` is linear in it. `g[l]`'s column role carries the
opposite sign because the decay is `gc[i] − gc[l]`, and the same token is both a row (`i`) of its
own chunk's block and a column (`l`) of the rows below it, so both roles add into the one `g` row.
`k[i]` and `k[l]` likewise take both roles, and a token's two contributions land in the key head
`hk = h / grp` that its value head reads: `head_k` is the head that carries the cosine, so `∂loss/∂k`
accumulates over every value head of a group into the one key head.

Cost: `L³/3` for `Et`, plus `L²·head_k` to re-form the cosines and the two rank-1 updates of the
`k` cotangent, plus `O(L²)` for the block -- the forward's `M` is recomputed rather than stored,
because it is `O(L²·head_k)` and cheaper than a tape. The whole backward is the same order of work
as the forward.

## 5. Frozen operands

`cu_seqlens` is frozen (operand 3). It addresses rows -- it says which token belongs to which
sequence and where that sequence's chunk grid starts -- so a different integer in it is a different
op and not a small step, and no cotangent is written for it. Zero-length sequences, and the rows a
last chunk leaves past its length, are therefore not a case for a cotangent either: nothing reads
them.

## 6. What the test covers

`tests/host/adjoint_gdn_kkt.cpp` asks both questions of `tests/support/adjoint_check.h` on five
shapes:

| shape | key heads × value heads | head_k | chunk | sequence lengths |
|---|---|---|---|---|
| fixture, below one chunk | 2 × 4 | 16 | 64 | 5 |
| fixture, exactly one chunk | 2 × 4 | 16 | 64 | 64 |
| fixture, two chunks and a short last | 2 × 4 | 16 | 64 | 101 (64 + 37) |
| fixture, three sequences | 2 × 4 | 16 | 64 | 5, 64, 150 (64 + 64 + 22) |
| no grouping, two chunks | 2 × 2 | 32 | 64 | 90 (64 + 26) |

The first is the tiny fixture's delta-net geometry (`docs/FIXTURES.md`); the lengths below, at and
above the op's chunk size, and the last two carry a short last chunk -- 37, 22 and 26 tokens of 64
-- and a batch of sequences of different lengths.

Then two things the checks alone cannot say:

- **A chunk is solved on its own operands.** The batch of three sequences, run with one cotangent,
  answers each token exactly as the same sequence run alone with its slice of that cotangent does,
  for `k`, `beta` and `g` -- no cotangent crosses a chunk or a sequence boundary, which is what the
  `cu_seqlens` addressing has to get right.
- **`cu_seqlens` is frozen** -- the registered frozen list is `{3}`, and the adjoint writes nothing
  into its slot -- and a record the op cannot compute (`head_k` that is not the key's last
  dimension) is declined by both checks with a note rather than compared against nothing.

The finite-difference check is this family's 1e-8 relative in f64, not the standard 1e-6: the solve
is a 64-deep recurrence and the rounding of one row is carried into the next, so a looser check
cannot tell the transposed solve from a near miss. The shared checker takes its step at 1e-6, and at
that step this op's second-order term alone is already 1.4e-8 of the answer, so the test refines the
step: a symmetric difference at `5e-5` and at `2.5e-5`, combined as `(4·D(h/2) − D(h))/3`, whose
second-order terms cancel. What is left is the twin's f64 roundoff, and the check answers the same
to about 1e-9 for every step from `3e-5` to `1.5e-4`.

## 7. One correction to `gdn-math.md`

Section 3 of `docs/adjoints/gdn-math.md` states the block cotangent as

    dM[i][l] = Σ_{j ≤ l} Ac[i][j] · A[l][j]

which is one factor of the transposed solve with the sign dropped and the outer sum over `p`
missing; on the two-token block it gives `+Ac[1][0]` where the answer is `−Ac[1][0]`. The form
above, `Et = −(Aᵀ Ac Aᵀ)` restricted to the strict lower triangle, is the one this op implements, the
one the finite differences here confirm, and the one `tools/gdn_math_check.py` already implements
(it prints `kkt adjoint == finite diff` at about 1e-9), so only the prose of that section is off.
