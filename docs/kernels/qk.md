# The qk family on the card

The gated attention block prepares its queries, keys and gate on the card with one fused op,
`qk_norm_rope_gate`, and with `qk_norm_rope_gate_kv_store` when the step also writes its keys and
values into the paged pool. `src/adjoint/device/qk.hip` in `libr97.so` is their device adjoint,
`tests/gpu/adjoint_qk.cpp` holds it to the host, and this file carries the record it reads, what
the kernel does, the numbers and the gaps. The host formulas are in `docs/adjoints/qk.md`; this
note does not repeat them and says instead what the card's fused form adds.

The op is one of the block's prologue steps (`arch/common/rad_block_attn_gated_fp8.h`):

```text
q lanes    rmsnorm(q_norm, eps, wadd) -> rope          -> q_out
k lanes    rmsnorm(k_norm, eps, wadd) -> rope          -> k_out -> kv_store -> kv_cache
gate lanes copied                                     -> gate_out
```

so the adjoint is the composition of the host adjoints of `rmsnorm`, `rope`, `kv_store` and the
gate's `sigmoid` then `mul`, in reverse. The reference of a device kernel is the host's own rows
for those parts -- the rmsnorm row through the registry, `rope_adjoint_f64` and
`kv_store_adjoint_f64` from `src/adjoint/host/qk.h` -- and not a second copy of the formulas.

## The record

Operands in the order the kernel reads them; the cotangents arrive in the order of the written
ones.

| # | buffer         | dtype | shape                                          | |
|---|----------------|-------|------------------------------------------------|---|
| 0 | `attn_qg`      | bf16  | `[tokens, n_head*2*head_dim]`                  | the projection, read |
| 1 | `attn_k`       | bf16  | `[tokens, n_head_kv*head_dim]`                 | read |
| 2 | `cos_sin`      | f32   | `[table rows, rot]`                            | the rope plane, constant |
| 3 | `positions`    | i32   | `[tokens]`                                     | constant |
| 4 | `q_norm`       | f32   | `[head_dim]`                                   | frozen |
| 5 | `k_norm`       | f32   | `[head_dim]`                                   | frozen |
| 6 | `attn_v`       | bf16  | `[tokens, n_head_kv*head_dim]`                 | store form only, read |
| 7 | `slot_mapping` | i32   | `[tokens]`                                     | store form only, frozen |
| 8 | `kv_cache`     | bf16  | `[blocks, n_head_kv, block_size, 2*head_dim]`  | store form only, inout |
| 9 | `q_out`        | bf16  | `[tokens, n_head*head_dim]`                    | written |
| 10| `k_out`        | bf16  | `[tokens, n_head_kv*head_dim]`                 | written |
| 11| `gate_out`     | bf16  | `[tokens, n_head*head_dim]`                    | written |

`qk_norm_rope_gate` is the same record without 6, 7 and 8, so its writes are 6, 7 and 8. A token's
head owns `2 * head_dim` lanes of the projection: its query lanes, then its gate lanes. The
parameters are radiance's -- `M`, `head_dim`, `n_head`, `n_head_kv`, `rot`, `q_dtype`, `eps`,
`wadd`, `cs_f32`, `pos_i64`, and `block_size`, `kv_dtype`, `k_scale`, `v_scale` for the store --
and the kernel reads them the way a device adjoint must: a parameter it cannot make sense of only
means the record is not served.

The kernel adds to `dc` and never overwrites it, because a cotangent can be shared by several ops.
The rows the store does not write still pass through the pool unchanged, and that identity is not
differentiated: `dc[kv_cache]` stays as the caller left it. A store consumes the pool's cotangent,
it does not produce one -- the pool is seeded per step by the attention that reads it.

## What the kernel does

One block of 128 threads per (token, head) row of the prologue, `head_dim` lanes at a time. The
row is the unit the two reductions the norm needs work inside -- the sum of squares of the norm's
input, and the sum the adjoint's shared term weights -- and the rope adjoint between them is a
pairing of lanes in the same row, so nothing crosses a block.

1. Sum the row's squares in f32, take `inv = rsqrt(sum/n + eps)`.
2. Load the output cotangent into the row's shared lanes. For a key row of the store form, add the
   pool's cotangent of that token's row, scaled by `k_scale`, first: the store's cotangent joins
   the key's before the rope turns it back.
3. Turn the rotated lanes with the rope adjoint, `dx_0 = c g_0 + s g_1`, `dx_1 = -s g_0 + c g_1`.
   The thread that owns a NeoX pair writes both its lanes, so the pass is in place.
4. Reduce `sum_i g_i x_i (w_i + wadd)` over the row -- the sum the norm's shared term weights,
   taken over the cotangent the rope has already turned back -- and write
   `dx_j = inv (w_j + wadd) g_j - inv^3 x_j / n * sum`.
5. A query row writes its `dx` into the projection's query lanes and the gate's cotangent into its
   gate lanes; a key row writes `dc` of the key projection, and, in the store form, the pool's
   value lanes scaled by `v_scale` into `dc` of the value projection.

The gate is a pass-through: the op copies those lanes, so their cotangent reaches `attn_qg`
unchanged. It arrives already through the chain above the op -- `sigmoid` then `mul` -- and the
test builds it that way, so the fused op is handed the cotangent a real composition produced, not
a copy of a random tensor.

The ends of the ladder need nothing special: `rot == head_dim` turns every lane, and the lanes
past `rot` pass their cotangent through. A token's pool row, in the store form, is the row its
`slot_mapping` entry names -- block `slot / block_size`, row `slot % block_size` -- so the pool
cotangent is reached slot by slot, not sequence by sequence.

A record whose extents do not agree with each other -- a projection width that is not
`n_head * 2 * head_dim`, a pool that is not `2 * head_dim` wide, a position the table does not
hold -- is not served: the adjoint returns with `dc` untouched, because an adjoint must not throw.

## bf16 and f32

The contract hands a device adjoint f64 host values. The kernel reads them at f32, accumulates
every reduction in f32, and rounds each lane once to bf16 on the way out -- gradients are bf16 on
the card, and a sum over many tokens accumulates in f32. A bf16 operand of the tape is rounded on
the way in, so the kernel reads what the card's op would read.

The rounding matters more here than for an elementwise kernel: the norm's adjoint subtracts two
terms that can nearly cancel, so a case that fed the device bf16 values but the reference f64
ones would sit outside the bf16 tolerance on the lanes where they cancel. The test gives both
sides the values a bf16 tape holds.

## The streamed form

`qk_norm_rope_gate_kv_store_streamed` is registered and refused: it throws

```text
qk_norm_rope_gate_kv_store_streamed is refused in training: the streamed store keeps the pool in
fp8 with a per-block residue row (`res`, `host`) that the trainer's bf16 pool does not carry, so
its pool cotangent has no bf16 form; issue qk_norm_rope_gate_kv_store against the bf16 pool
instead
```

Training keeps the pool in bf16, so there is no cotangent to hand the quantised store and no host
chain to compare it against. r97's registry has no way to say "this op is not served"; a row that
refuses with the reason is the way to deliver one that names it, and the test asserts the throw
and the message.

## The test

`tests/gpu/adjoint_qk.cpp` (gpu tier, one card, any contributor's) compares the kernel against the
composition of the host adjoints within `1e-3 + 0.02 * |host|` -- the bf16 tolerance
`docs/kernels/README.md` states -- at these shapes:

- the tiny fixture's: 4 query heads, 2 key-value heads, 32 lanes, 16 rotated, 6 tokens, both forms
  and a bf16 rope table;
- the real model's: 24 query heads, 4 key-value heads, 256 lanes, 64 rotated, 1024 and 4096 tokens
  in one sequence, a batch of three ragged sequences (448, 320 and 256 tokens), both forms.

It also

- checks the reference itself: a centred step of 1e-3 along a random direction of the reads must
  move the writes by as much as the reference's cotangent says, at the fixture and at a real
  geometry. Without it the comparison would be two opinions rather than a check, and this is the
  step that found the kernel's first bug;
- checks that the comparison rejects a reference that drops the gate's cotangent, so it has teeth;
- checks the frozen sets (the table, the positions and the two gains; the slot map too for the
  store form), the registration of the streamed row, and that the kernel leaves no scratch live.

## Measurement

Measured with `./build/r97_gpu_tests --bench` on an AMD Radeon AI PRO R9700 (gfx1201, 32 GB), 20
reps at the fixture and 6 at the real shapes. The card's line, verbatim:

```text
qk_norm_rope_gate fixture 6 tok, q and k    1.261 ms adjoint     0.010 ms forward twin  126.89x forward  scratch peak 19 KiB
qk_norm_rope_gate_kv_store fixture 6 tok, with the pool    1.535 ms adjoint     0.010 ms forward twin  148.30x forward  scratch peak 24 KiB
qk_norm_rope_gate 1024 tok, 1 seq, q and k  142.443 ms adjoint   345.627 ms forward twin    0.41x forward  scratch peak 133382 KiB
qk_norm_rope_gate_kv_store 1024 tok, 1 seq, with the pool  290.973 ms adjoint   560.327 ms forward twin    0.52x forward  scratch peak 143626 KiB
qk_norm_rope_gate_kv_store 1024 tok, 3 ragged seqs, pool  237.888 ms adjoint   663.335 ms forward twin    0.36x forward  scratch peak 143482 KiB
qk_norm_rope_gate_kv_store 4096 tok, 1 seq, with the pool 1112.168 ms adjoint  2454.585 ms forward twin    0.45x forward  scratch peak 574498 KiB
qk_norm_rope_gate 4096 tok, 1 seq, q and k 1166.192 ms adjoint  2533.004 ms forward twin    0.46x forward  scratch peak 533522 KiB
```

The adjoint's time is the whole call -- upload, launch, download -- because that is what the
contract makes a device adjoint pay, and the forward is the op's f64 host twin: until an op has a
device forward, the twin is the honest reference for "the forward op it belongs to".

The reading: at these widths the call is bounded by moving the tape's values, not by the kernel.
A 4096-token store-form call reads and writes about 180 million f64 tape values (1.4 GB) through
a scalar conversion into the f32 buffers the kernel reads, which is the whole of the second it
takes; the kernel's own work is one pass over the same lanes. That is why the ratio to the
forward twin is below one: the twin is the same f64 arithmetic without a card, and it is slower
than the conversion the call pays for. The fixture's 1.3 ms is launch and sync, not arithmetic.

The scratch peak is the process-wide high-water mark of the device allocator the kernel takes its
buffers from, per case: 130 MiB for 1024 tokens and 561 MiB for 4096. Nearly all of it is the
projection: its f32 copy, the f32 copy of its query cotangent, the f32 copy of the gate cotangent,
and the bf16 cotangent written back are four of the five largest buffers.

No speed target is set for this kernel; the measurement is the deliverable.

## What it does not serve

- **The k-only spelling.** The host issues rope separately for queries and keys (`n_head_kv = 0`
  with `n_head = n_head_kv`). The card's op is the gated form, and the kernel serves records with
  both `n_head` and `n_head_kv` positive; a record with no query heads is not served and is left
  untouched rather than half-written. The trainer always issues the gated form.
- **64-bit positions.** `pos_i64` is read as a parameter, but a position reaches the kernel as the
  index operand's values at i32, which is what the tape holds.
- **The pool's own cotangent.** As above, a store does not produce one.
- **A device forward.** The twin that gives a case its forward values lives in the test file, not
  in the plugin: radiance processes load `libr97.so` without r97's registry (`plugin.hip`), and a
  twin added there would be registered into a table those processes do not have.
