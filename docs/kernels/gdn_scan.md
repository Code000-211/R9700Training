# The gdn scan on the card

The delta net carries a `[K, V]` state through a sequence's chunks, and its adjoint carries the
state's cotangent back through them: at 4096 tokens a value head makes 64 state updates, and the
cotangent of each one depends on the cotangent of the next. `src/adjoint/device/gdn_scan.hip` in
`libr97.so` is the card's `gdn_chunk_scan` adjoint -- one kernel, one block per (sequence, value
head) -- `tests/gpu/adjoint_gdn_scan.cpp` holds it to the host, and this file carries the record it
reads, what the kernel does, the numbers and the gaps. The formulas are in
`docs/adjoints/gdn_scan.md`; this note does not repeat them and says instead what the card adds.

## The record

| # | buffer | dtype | shape | |
|---|--------|-------|-------|---|
| 0 | `q` | bf16 | `[T, Hg, K]` | read |
| 1 | `k` | bf16 | `[T, Hg, K]` | read |
| 2 | `v` | bf16 | `[T, H, V]` | read |
| 3 | `A` | bf16 | `[T, H, chunk]` | read, lower triangle |
| 4 | `g` | f32 | `[T, H]` | read, within-chunk cumsum |
| 5 | `beta` | f32 | `[T, H]` | read |
| 6 | `h0` | f32 | `[Nslots, H, V, K]` | frozen |
| 7 | `cu` | i32 | `[S+1]` | frozen |
| 8 | `o` | bf16 | `[T, H, V]` | written |
| 9 | `ht` | f32 | `[Nslots, H, V, K]` | written |
| 10 | `state_idx` | i32 | `[S]` | optional read, frozen |

The parameters are the host row's -- `M`, `head_k`, `head_v`, `chunk` and the f64 `scale` -- and
the kernel reads them as the host row does, including the fallback to `1/sqrt(head_k)` for a record
off the tape that carries no `scale`. The frozen set the row registers is `{h0, cu, state_idx}`.

`state_idx` sits at position 10, *after* the two writes, because that is where the record puts it:
`call.fwd` is every operand in schema order, so the kernel reads the slot indices there and not
among the reads that precede the writes. A record without the operand owns slot `n` for sequence
`n`, and then `h0`/`ht` must hold at least `S` slots. The kernel adds to `dc` and never overwrites
it, and it leaves `dc` untouched for a record whose operands do not agree with each other: an
adjoint must not throw, so a shape the kernel cannot serve is simply not served.

## What the kernel does

One block of 128 threads per (sequence, value head) -- `N * H` blocks, which at a 4096-token
prefill of one sequence is 48 -- and two passes over that block's chunks.

1. **Replay.** The block re-runs the forward's state carry from `h0`, one chunk at a time, and
   writes each chunk's *entering* state into its own f32 scratch at `[chunk, K, V]`. The reverse
   pass reads those instead of recomputing a state per chunk, and the caller's forward saves
   nothing: no interior state crosses the op's boundary. The replay also keeps each chunk's `W`
   and `V'` in scratch, the two intermediates stages 3 and 4 differentiate.
2. **The reverse walk.** For each chunk from the last to the first the block runs the host's four
   stages -- from the state cotangent (stage 1), from `o`'s cotangent (stage 2), from `V' = A W`
   (stage 3), from `W` (stage 4) -- and carries the state cotangent into the previous chunk.
   Nothing crosses a chunk or a sequence boundary except that `[K, V]` block, which lives in f32
   global scratch and is ping-ponged between two buffers.

The block's work per chunk is one `[L, L]` triangle (`qk[i][j]`, `phi[i][j]`, `vo[i][j]`), and the
reductions those need over `K` or `V` lanes are the block's, so `dq`, `dk`, `dv`, `dA`, `dg` and
`dbeta` are all resolved inside the block that owns the head. The one exception is a key head: `grp
= H / Hg` value heads share it, so `dq` and `dk` are accumulated with f32 `atomicAdd` into the
record's `[T, Hg, K]` gradients and added into the cotangent at that width -- finer than the bf16
the operands hold, which costs nothing since the cotangent is f64.

`g` is read as the within-chunk cumsum the tape holds, so the kernel takes differences of it,
`exp(g_i - g_j)` and `exp(g_last - g_j)`, exactly as the host does; `exp` is `expf` here. A chunk
shorter than `chunk` -- the last chunk of a sequence -- uses its own length: the kernel writes no
row past it, and the `A` entries above the triangle are written as explicit zeros, which is what
`gdn_kkt_solve` wrote and what the host row's `dc` keeps.

## bf16 and f32

The state and its cotangent are f32 in the card's scratch: the cotangent is carried over every
chunk of the sequence, and at 4096 tokens that is 64 updates of a `[128, 128]` block per value
head, so an f32 error introduced in the first chunk is still there in the last. Every reduction --
`qk`, `vo`, `sk2`, the sums over `V` or `K` a stage needs -- accumulates in f32, and each gradient
is rounded once, to its operand's dtype, on the way out: the gradients are bf16 on the card and a
sum over many tokens accumulates in f32. The host row is f64 throughout, so the comparison below is
exactly "the card's f32 carry against the host's f64 carry" on the same values.

The one thing the kernel cannot fix by itself is a case whose data is not the delta rule's:
`A` with random entries of order one, or keys that are not normalised, makes the state grow by tens
of times a chunk, and then an f32 carry and an f64 one diverge by orders of magnitude no matter how
the kernel is written. The test therefore builds its cases the way the pipeline does --
`gdn_prep`'s L2-normalised `q`/`k`, `gdn_kkt_solve`'s `A = (I + M)^{-1}`, and a gate that decays a
few per cent a token -- and `tests/gpu/adjoint_gdn_scan.cpp` says why at its head.

## What it needs from the forward, and the memory

The reverse walk needs, per chunk, the `q, k, v, A, g, beta` slices and the chunk's entering state.
The card takes nothing from the forward but the operands: like the host's recompute variant it
replays the state carry itself, so **the forward saves no interior state**, and the op's own
scratch holds the replay.

At `T = 4096`, `chunk = 64` (64 chunks), `Hg = 16`, `H = 48`, `K = V = 128`, one sequence:

| what | per value head | over the 48 heads |
|------|----------------|-------------------|
| the replay's entering states (`[64, 128, 128]` f32) | 4,194,304 B (4 MiB) | 201,326,592 B (192 MiB) |
| the block's working set (`St1`, `dS0`, `W`, `V'`, the `[L, L]` triangles) | 576,000 B (562.5 KiB) | 27,648,000 B (26.4 MiB) |
| **scratch in all** | **4,770,304 B (4.55 MiB)** | **228,974,592 B (218.4 MiB)** |

The host issue's figures for the same sequence (`docs/adjoints/gdn_scan.md`) are: 8,388,608 bytes
(8 MiB) per head for the 64 entering states in f64, held for one head at a time and released
before the next one starts; 105.5 MiB of operands the forward's tape keeps either way; and 192 MiB
of boundary states if the forward saved them all, which with the operands is 297.5 MiB in all.

So the card's states are half the host's per head, because they are f32, but all 48 heads' blocks
run at once, so the card holds every head's replay at the same time -- the same 192 MiB the host's
save-all variant would put on the forward's tape, plus a 26.4 MiB working set. It adds nothing to
the tape, though: the 218.4 MiB is the call's own scratch, freed with it, where the host's 297.5
MiB is tape that outlives the call. On a 32 GB card that is affordable and it is the simplest thing
that keeps the reverse walk to a single pass over the chunks; a replay that kept only every eighth
state and replayed eight chunks at a time would cut it to about 27 MiB, at the cost of a third
pass over the sequence.

## The test

`tests/gpu/adjoint_gdn_scan.cpp` (gpu tier, one card, any contributor's) compares the kernel
against the op's host row -- the card issues the scan as one op, so there is no fused op for the
reference to compose -- within `1e-3 + 0.02 * |host|`, the bf16 tolerance
`docs/kernels/README.md` states, on the same random inputs. The shapes are

- the tiny fixture's gdn geometry (`docs/FIXTURES.md`): two key heads and four value heads of 16,
  4-token chunks, one short sequence, a batch of three ragged ones (2, 4 and 13 tokens, the last
  a chunk of one), and two state pools deeper than their batches with `state_idx` naming slots out
  of order -- 3 slots for 2 sequences (slots 2 and 0) and 4 slots for 3 (slots 3, 0 and 2);
- the real model's (`docs/DESIGN.md`, hidden 5120): `K = V = 128`, 16 key heads, 48 value heads,
  64-token chunks, at 1024 tokens in one sequence, at 1024 tokens in a batch of three ragged
  sequences (448, 322 and 254, so one ends on a chunk boundary and two end short), and at 4096
  tokens in one sequence.

It also checks the frozen set, that the kernel leaves no scratch live, and that the comparison has
teeth: a device row that drops the gate's cotangent -- the term of the reverse pass a kernel is
most likely to leave out -- must be rejected in `dc[4]`.

## Measurement

Measured with the gpu tier's `--bench` pass on an AMD Radeon AI PRO R9700 (gfx1201, 32 GB): the
comparison runs first over each case, then a timed pass over the same values. The adjoint's time is
the whole call -- upload, launch, download -- because that is what the contract makes a device
adjoint pay, and the forward's is the op's f64 host twin: until an op has a device forward, the
twin is the honest reference for "the forward op it belongs to".

The lines below are from a run of this family's test alone, linked against the same objects the
test binary uses, so the card was held for minutes and not for the hour the whole tier's bench
pass takes; `./build/r97_gpu_tests --bench` prints them among every other family's. The three real
shapes, verbatim:

```text
gdn_chunk_scan 1024-one-sequence  322.846 ms adjoint  4263.266 ms forward twin    0.08x forward  scratch peak 156792 KiB
gdn_chunk_scan 1024-ragged    156.249 ms adjoint  3534.080 ms forward twin    0.04x forward  scratch peak 238440 KiB
gdn_chunk_scan 4096-one-sequence 1033.655 ms adjoint  14117.081 ms forward twin    0.07x forward  scratch peak 527736 KiB
```

The reading:

- **The error at 4096 tokens.** Max |device - host| 0.0309 and max relative deviation 3.9e-3, both
  on `dA`, inside the 2 % the bf16 tolerance allows. The same comparison at 1024 tokens gives 0.0292
  and 4.3e-3, so the deviation does not grow with the sequence: four times the chunks, and four
  times as many carries of the state cotangent through the [K, V] block, leave it where it was. That
  is the f32 carry doing its job -- a bf16 one would have compounded through 64 updates -- and what
  is left is the bf16 rounding of the operands the case is built from and of the gradients the call
  hands back, which does not depend on the length of the sequence.
- **The time.** 1.03 s a call at 4096 tokens against the twin's 14.1 s -- 0.07x, and the two
  1024-token cases say where the call's time goes. Both carry about 14 million operand values up
  and 15 million cotangent values down, yet the ragged batch takes 156 ms and the single sequence
  323 ms: 5.4 ns a value is the cheapest the whole call gets, which is what it costs to narrow the
  tape's f64 operands to bf16 and widen the cotangents back, and the rest of the single sequence's
  call is its kernel -- 48 blocks is fewer than the card's 64 compute units, where the ragged
  batch fills them with 144. At 4096 tokens the call is 9 ns a value: if 5.4 ns of that is the
  staging, the kernel has grown to about 0.4 s of the 1.03 s, 64 chunks a block against 16. Other
  work on the card moves these numbers -- a second run of the same test gave 311 ms, 222 ms and
  1.05 s for the same three cases -- so read them as a shared card's, not a quiet one's. The twin
  is the f64 host row, and the call beats it by 13.7 times at 4096 tokens, 13 times at 1024 in one
  sequence and 23 in the ragged batch.
- **The scratch.** 156,792 KiB (153 MiB) at 1024 in one sequence, 238,440 KiB (233 MiB) at 1024 in
  three ragged ones, 527,736 KiB (515 MiB) at 4096. Of the 4096 figure 218.4 MiB is the replay's
  entering states and the block's working set, the memory the section above derives; the rest is
  the call's staging, one buffer per operand as it goes up and one per cotangent as it comes down.
  The ragged batch takes 80 MiB more than the single sequence at the same token count because each
  of its 3 x 48 blocks owns a 576 KiB working set, where one sequence's 48 blocks own one each.

## What it does not serve

- **A device forward.** The twin that gives a case its forward values stays in the test file and
  in the plugin's host rows: a radiance process loads `libr97.so` without r97's registry, and a
  twin added there would be registered into a table those processes do not have.
- **A state cotangent the caller wants for `h0`.** `h0` is frozen: `dS0` leaving a sequence's first
  chunk is dropped, which is the host row's rule and the reason a prefill starts from the zero
  state.
- **64-bit token counts.** `cu` reaches the kernel as the i32 values the tape holds, so a sequence
  set wider than `2^31` tokens is not served.
- **A chunk width that does not divide the tape's `A` row.** `chunk` comes from the parameter or
  from `A`'s last extent; a record where the two disagree is refused rather than half-written.
