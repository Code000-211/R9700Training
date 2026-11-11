# The loss on the card: the head's cross-entropy in chunks

The device form of the reference loss (`docs/losses/ce-objective.md`): the negative
log-likelihood of the head's logits and their gradient, computed on the card a chunk of positions
at a time. The contract is `include/r97/head.h`, the walk that fills and drains the chunks is
`src/head.cpp`, and the chunk's arithmetic is `src/adjoint/device/head.hip` in `libr97.so`, with
the row's maths in `head_row.h` next to it.

A logit row is 248320 bf16 values -- 496 KiB (`docs/DESIGN.md`) -- so a step's worth of logits is
never held, on the card or on the host. The head product writes one chunk of rows, this loss turns
that chunk into its rows' losses and their cotangent, and the transposed head product (the
`linear` family's kernel on a chunk of the head weight, a later issue) consumes the cotangent
before the next chunk is produced. This issue produces the gradient and defines the loop's
interface; nothing here multiplies by the head weight.

## The chunk

`r97::HeadChunk` is one chunk, in device buffers, and nothing else:

| field | meaning |
| --- | --- |
| `logits` | in, `positions * vocab` bf16: the rows the head product wrote |
| `grad` | out, `positions * vocab` bf16: `(softmax(row) - onehot(ids[i])) * grad_scale` |
| `loss` | out, `positions` f32: each row's loss |
| `ids` | in, `positions` i32: each row's target, `-1` where the row is not scored |
| `positions`, `vocab` | rows in the chunk, values per row |
| `grad_scale` | `1 / the step's scored tokens`: the gradient of the mean |
| `stream` | the `RadStream` the kernel queues on |

The kernel is `r97::HeadCeKernel`: it queues the chunk and returns, or returns `false` with a
message. It does not wait, so a caller that reads `loss` synchronises, and a caller that hands
`grad` to another stream queues there behind this one. Kernels are not ops' adjoints -- a loss
answers with a scalar, and its buffers live across chunks -- so a chunk kernel registers through
`head_ce_register` instead of `R97_REGISTER_ADJOINT`; `libr97.so` calls it from a static
initializer as the plugin loads, and `src/adjoint/device/plugin.hip` carries a fallback so
radiance's converter, which loads the same plugin without linking r97, still loads it. A process
that does not load the plugin finds no kernel, and `head_ce_loop` says so rather than silently
computing nothing.

## The walk

`r97::head_ce_loop` is the loop the device join calls: a `HeadCeWindow` (rows, vocab, rows per
chunk, scale) and two callbacks.

- `fill` produces a chunk: it writes the rows into `chunk.logits` -- queueing whatever it needs on
  `chunk.stream` -- and answers with the rows it wrote and their targets on the host, `-1` where
  a row is not scored. 0 rows ends the window.
- `sink` takes the chunk once its gradient is complete on the card, before the next `fill`. This
  is where the transposed head product queues its work, so only one chunk's gradient is ever
  alive.

Between them the loop uploads the ids, runs the kernel, waits, and adds the chunk's f32 losses
into a f64 sum over the scored rows. A fill's host memory must live until the chunk's `sink`: the
loop synchronises the stream before it calls it. The chunk's buffers themselves are the loop's,
from the device scratch (`src/adjoint/device/scratch.h`), freed however the walk ends -- by the
end of the window, by a kernel failure or by a producer that throws -- and charged against
`scratch_stats()`, which is where the memory below can be read back.

## The gradient, and where it goes

`HeadCeObjective` states the same loss as an `Objective` (`include/r97/objective.h`): one call per
window of a `ForwardTape`, reading the window's `"logits"`, ids from `fwd.batch()`, scaled by
`1 / fwd.total_scored()`, and answering with that window's summed loss, its scored count and one
`Seed` on `"logits"` spanning its scored rows exactly as `CeObjective` does. On the host it is
indistinguishable from the reference: same sum, same cotangent rows, same zeros on the unscored
rows a window spans.

The seed's cotangent is f32 on the host. That vector is the contract's, and it is what the tests
use to hold this loss to the reference, but it is not what a device join spends: the join fills and
sinks the chunk loop itself -- the head product writing `logits`, this kernel writing `grad`, the
transposed product reading `grad` on the same stream -- so the gradient crosses no bus at all, and
a 248320-wide row's cotangent is never a host vector. `HeadCeObjective` remains the checked form
and the one a host-side driver can call.

## One chunk's memory

Buffers a chunk of `R` rows of vocabulary `V` holds: `2 * R * V` bytes of logits and as much
again of gradient, plus `4 * R` for the ids and `4 * R` for the losses:

```
2 * R * V * 2 bytes + 8 * R bytes
```

At the default `R = 64` and `V = 248320`: 31,784,960 bytes each way, 63,570,432 bytes in all --
63.6 MB (60.6 MiB) of the card's 32 GB, and the same wherever the window's row count is. Chunk
size is the whole memory-versus-parallelism trade: `R = 1` is just under a megabyte and one
block's work, `R = 512` is 508 MB and 512 blocks of it. `kHeadChunkRows` is 64.

## Cost per 1000 positions

One row costs three reads and one write of its 496 KiB, because the stable form makes two passes
over the row before the gradient pass: the maximum, the sum of `exp(x - max)`, then the gradient,
which recomputes `exp` rather than keeping a row of exponentials on the card. So 1000 positions
are about 1.99 GB of traffic and 497 million `exp` evaluations (two per element), and a card's
memory system, not its transcendentals, is what the loop waits on.

**Measured:** not in this run. The container this change was written in has no card it can reach
(`/dev/kfd` refuses to open, `hipGetDeviceCount` fails with 100), so the number below is the
traffic above and not a stopwatch. On a card, the same binary prints the line this section
quotes:

```sh
cmake -B build -G Ninja && cmake --build build
R97_DEVICE=0 ./build/r97_gpu_tests --bench   # tests/gpu/head_ce.cpp, the card's HIP ordinal
```

```
head ce: <t> ms per 1000 positions (256 rows of 248320, chunk 64, one upload); loss <l>; ...
```

The bench walks 256 positions of the real width through the loop and counts the host upload of
their logits once, so the line is the join's side of the cost -- the kernel, the losses coming
back and nothing else -- rather than the head product's. It is the same line as
`docs/kernels/README.md` describes, in this file's own form.

## Checked by

`tests/gpu/head_ce.cpp`:

- the row's arithmetic (`head_row.h`) against `ce_position` on rows of 320 and 248320 values, and
  on logits at 1e30 -- the same expressions the kernel compiles, so this part needs no card;
- the loop against `ce_sum` on the real vocabulary, chunk sizes 1, 64 and 512: the summed loss
  and every gradient value;
- the loop against `ce_loss` and `ce_grad` on the tiny fixture's recorded logits
  (`docs/FIXTURES.md`, three sequences of 140, 73 and 31 tokens, vocab 320, unscored rows
  included), the same chunk sizes;
- `HeadCeObjective` against `CeObjective` on a `VectorTape`, one window per sequence --
  the loss, the scored count and every cotangent value;
- logits at 1e30 and 3e38 staying finite, against the reference's f64 answer.

The tier's tolerance is `1e-3 + 0.02 * |reference|` (`tests/support/device_check.h`): the room a
bf16 write and an f32 reduction need, and not licence to compute another function -- the row check
above is held to 1e-3 relative, which the wrong formula fails.
