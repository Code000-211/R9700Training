# `attn_paged`: the adjoint

Reference: radiance `libref/ref_attn.cpp` (`ref_attn_paged`, sharing `attn_paged_impl` with the
sink and tree variants) and the `attn_paged` row of `libref/ref_registry.cpp`. The host adjoint is
`src/adjoint/host/attention.cpp`, its test `tests/host/adjoint_attention.cpp`.

## 1. The op, as the reference computes it

`attn_paged` is causal scaled-dot-product attention at prefill, over a paged key/value pool:

| operand | dtype | shape | read/write |
|---|---|---|---|
| `q` | bf16 | `[rows, q_heads, head_dim]` | read |
| `kv_cache` | bf16 | `[num_blocks, kv_heads, block_size, 2·head_dim]` | read |
| `block_table` | i32 | `[n_seq, max_blocks]` | read |
| `seqused` | i32 | `[n_seq]` | read |
| `k_descale` | f32 | `[n_seq · kv_heads]` (optional) | read |
| `v_descale` | f32 | `[n_seq · kv_heads]` (optional) | read |
| `cu_seqlens` | i32 | `[n_seq + 1]` (optional) | read |
| `out` | bf16 | `q`'s shape | written |

Parameters: `head_dim` (derived from `q`'s last dimension), `gqa` = `q_heads / kv_heads`,
`block_size` (derived from `kv_cache`), `causal` (default 1), `window` (default 0: no sliding
mask), `scale` (default `1/sqrt(head_dim)`).

A dense query row `tok` belongs to sequence

    s     = the sequence whose [cu_seqlens[s], cu_seqlens[s+1]) holds tok
            (cu_seqlens absent: s = tok / q_len, with q_len = rows / n_seq)
    q_len = cu_seqlens[s+1] − cu_seqlens[s]
    i     = tok − cu_seqlens[s]

and, with `ctx = seqused[s]`, sits at key position

    pos = ctx − q_len + i        (base = ctx − q_len, pos = base + i)

Training runs each sequence as one prefill step from position 0, so `ctx = q_len`, `base = 0`,
`pos = i`. Key `j` of that sequence lives in the pool at

    blk  = block_table[s][j / block_size]
    slot = blk · block_size + j mod block_size
    K_j[d] = kv_cache[blk, kv_head, j mod block_size, d]          d < head_dim
    V_j[d] = kv_cache[blk, kv_head, j mod block_size, head_dim+d]

which, flattened, is the offset

    base(j) = ((blk · kv_heads + kh) · block_size + j mod block_size) · 2 · head_dim

so the key's `head_dim` doubles are at `base(j) + d` and the value's at
`base(j) + head_dim + d`. `kh = h / gqa` is query head `h`'s key/value head.

**Visibly** read keys are those with `0 ≤ j < min(ctx, max_blocks · block_size)` whose block is
mapped (`blk ≥ 0`, `blk < num_blocks`) and that the mask keeps:

    causal = 1:      j ≤ pos, and (window > 0: pos − j < window)
    causal = 0:      window > 0: |pos − j| < window;   window = 0: no mask

For each (row `tok`, head `h`), with `kds = k_descale[s·kv_heads + kh]` (1 if absent) and
`vds = v_descale[s·kv_heads + kh]` (1 if absent), over the visible keys `j` in ascending order:

    dot_j = Σ_{d < head_dim} q[tok, h, d] · K_j[d] · kds
    score_j = dot_j · scale
    mx    = max_j score_j
    w_j   = exp(score_j − mx)
    Z     = Σ_j w_j
    p_j   = w_j / Z
    out[tok, h, d] = Σ_j p_j · V_j[d] · vds

A row with no visible key is zero, not NaN. The value is `vds`-scaled inside the sum, so the
descale is a per-(sequence, key/value head) temperature on the values, not on the output.

## 2. The f64 twin

`attention_twin` computes exactly §1 in f64: the same visible-key walk through the same
`Call::key_slot`, the same ascending key order, the same `max`, `exp`, sum and division. Its output
is `q`'s dtype and shape. A record the op cannot compute -- `q`'s head count not `gqa·kv_heads`, a
`kv_cache` that is not rank 4, `cu_seqlens` whose last entry is not `rows` -- yields no output at
all rather than a wrong one.

## 3. The adjoint

Write `y = out[tok, h, ·]`, `ȳ = ∂L/∂y` (the cotangent `call.cot[0]`), and let `G_j` be the row's
cotangent contracted with key `j`'s value:

    G_j = Σ_d ȳ[d] · V_j[d] · vds

`y = Σ_j p_j u_j` with `u_j[d] = V_j[d]·vds`, so `∂L/∂p_j = G_j`: the output is linear in the
weights. The softmax's Jacobian then gives the score's cotangent without reference to `Z`:

    ∂p_j/∂score_k = p_j (δ_jk − p_k)
    score_cot_j  = ∂L/∂score_j = p_j (G_j − Σ_k p_k G_k)  =: p_j (G_j − Ḡ)

(equivalently `p_j (G_j − ȳ·y)`, since `ȳ·y = Σ_k p_k G_k`). With that, the three cotangents:

    ∂L/∂q[tok, h, d]      += Σ_j score_cot_j · scale · K_j[d] · kds
    ∂L/∂K_j[d]            += score_cot_j · scale · q[tok, h, d] · kds
    ∂L/∂V_j[d]            += p_j · ȳ[d] · vds

The first is `dq`, added to the cotangent of `q` at `[tok, h, d]`. The second and third are the
pool's: both land on the slot of key `j`, the key's at `base(j) + d` and the value's at
`base(j) + head_dim + d`, added to the cotangent of `kv_cache` laid out like the operand. A key
read by many (row, head) pairs accumulates its share from each; a pool row no query read -- a block
the step did not fill, a row past a sequence's last position, a block a table entry leaves at −1 --
keeps the zero it started with. That is the addressing the issue asks for: the cotangent of the
pool's key and value rows of the step, one per sequence and key position, at the same slots
`kv_store` (the `qk` family) writes, which is how the two chain.

The adjoint reads `w_j` and `p_j` from the forward values in `call.fwd` (`q`, `kv_cache`, the
integer operands), never from the output, and accumulates into `call.dc` -- add, never overwrite.
It computes in f64 throughout.

## 4. Frozen operands

`R97_REGISTER_ADJOINT_F("attn_paged", host, attention_host_adjoint, 2, 3, 4, 5, 6)` freezes, in
schema order, `block_table`, `seqused`, `k_descale`, `v_descale`, `cu_seqlens`.

- `block_table`, `seqused` and `cu_seqlens` address rows: a different integer is a different op,
  not a small step, so there is no derivative to take. Frozen, the finite-difference check
  perturbs only `q` and `kv_cache`.
- `k_descale` and `v_descale` are the fp8 pool's dequantisation scales, and the plan makes a scale
  a constant. Their cotangents would be computable
  (`∂L/∂kds = Σ_j score_cot_j · scale · Σ_d q[d]·K_j[d]`, per sequence and key/value head), but
  the training path treats them as given, so the adjoint returns none.

A record that leaves the optional operands out keeps the schema positions of the ones it has: the
record without `k_descale`, `v_descale` and `cu_seqlens` carries five operands, and a cotangent at
position 4 is then the written operand's, not a descale. `resolve` counts the record's reads and
treats an optional operand past them as absent.

## 5. What the test covers

`tests/host/adjoint_attention.cpp` runs both checks of `tests/support/adjoint_check.h`:

| case | head geometry | sequences | block size | window |
|---|---|---|---|---|
| fixture, one sequence | 4 q over 2 kv, dim 32 | 12 | 8 | 0 |
| fixture, three sequences | 4 over 2, dim 32 | 5, 9, 14 | 4 | 0 |
| grouped query, one sequence | 24 over 4, dim 256 | 7 | 16 | 0 |
| grouped query, three sequences | 24 over 4, dim 256 | 3, 6, 10 | 5 | 0 |
| fixture, sliding window | 4 over 2, dim 32 | 6, 6 | 3 | 4 |
| fixture, uniform lengths | 4 over 2, dim 32 | 6, 6 | 4 | 0 |

The first two shapes are the tiny fixture's (`tests/support/fixture.h`: 4 query heads over 2
key/value heads of dimension 32) and the grouped-query geometry of a Qwen3.8-27B block (24 over 4
of dimension 256). Each sequence's pool rows are its own blocks plus one block the step did not
fill, whose table entry stays −1.

`check_kernel` holds the twin against radiance's host kernel within 0.02 (bf16); the measured worst
case is below 0.007, the output's own bf16 store. `check_adjoint` holds the adjoint against central
finite differences of the twin to 1e-6 relative; the measured worst case is below 1e-8. The
uniform-length case carries no optional operands, so libref cannot read it positionally and only the
finite-difference check runs on it -- its forward is the same arithmetic the ragged cases check
against the kernel.

`adjoint_attention_pool_address` runs the adjoint directly with a random output cotangent and reads
the pool's cotangent back through the block table: every (sequence, key position) the step wrote has
moved, and every row past a sequence's last position stayed exactly zero.
