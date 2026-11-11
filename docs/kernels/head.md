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
are about 1.99 GB of traffic and 497 million `exp` evaluations, two per element.

**Measured** on an AMD Radeon AI PRO R9700 (gfx1201, 32 CUs, 32624 MiB; `R97_DEVICE=1`, sysfs
`card2`), `./build/r97_gpu_tests --bench`:

```
head ce: 96.575 ms per 1000 positions, the best of 7 passes of 256 rows of 248320 at chunk 64,
  one upload a pass (the slowest read 106.094); loss 3215.06; one chunk's scratch peak 62080 KiB
```

The bench walks 256 positions of the real width through the loop -- four chunks of 64 -- with one
upload of the chunk's logits a pass and the pass that warms the scratch not counted, so the line
is this side of the join: the kernel and the losses coming back, and not the head product. The
machine is shared, and the seven passes are one card's own state: 96.6 to 106.1 ms. With another
run on this card at the same time the same passes read 280 to 340 ms, which is why the bench takes
the best of them.

96.6 ms per 1000 positions is 21 GB/s of the 1.99 GB above, where this card streams 514 to 562 GB/s
of a 256 MiB device-to-device copy. The loop is therefore not waiting on the card's memory system
or on its transcendentals: a block a row is 64 blocks on 32 CUs, and a thread walks its 970 values
of the row one 2-byte load at a time with the accumulator in a chain, so each pass over a row pays
a memory latency per element instead of streaming. The traffic above is a floor, not a forecast:
reaching it wants more rows in flight or wider loads per lane, which is a change to this kernel's
launch and to `head_row.h`'s walk, not to the interface this issue fixes. As it stands a window of
4096 positions costs the step about 0.40 s.

## Checked by

`tests/gpu/head_ce.cpp`:

- the row's arithmetic (`head_row.h`) against `ce_position` on rows of 320 and 248320 values, and
  on logits at 1e30 -- the same expressions the kernel compiles, so this part needs no card;
- the loop against `ce_sum` on the real vocabulary, chunk sizes 1, 64 and 512: the summed loss
  and every gradient entry, each entry held relatively;
- the loop against `ce_loss` and `ce_grad` on the tiny fixture's recorded logits
  (`docs/FIXTURES.md`, three sequences of 140, 73 and 31 tokens, vocab 320, unscored rows
  included), the same chunk sizes;
- `HeadCeObjective` against `CeObjective` on a `VectorTape`, one window per sequence --
  the loss, the scored count and every cotangent entry;
- logits at 1e30 and 3e38 staying finite, against the reference's f64 answer;
- the comparison itself: a deliberately wrong kernel -- the reference's loss and its target
  entries and zeros everywhere else -- is refused, after its loss and its target entries have
  been shown to agree, so what the comparison catches is the gradient.

The tier's tolerance is `1e-3 + 0.02 * |reference|` (`tests/support/device_check.h`): the room a
bf16 write and an f32 reduction need, and not licence to compute another function. The gradient
and cotangent comparisons here take the relative half of it and no absolute floor, because bf16
rounds relatively and a floor does not: an off-target entry of the logit gradient is
`softmax * grad_scale`, 1.2e-6 on a typical one of these fixture rows and 1.3e-6 on a 248320-wide
one, and the tier's 1e-3 floor is three orders of magnitude above either -- it would accept a zero
where the softmax should be. The summed loss is held to 1e-3 relative instead.

The teeth are checked rather than claimed. `the_comparison_has_teeth` registers the wrong kernel
above, shows its loss and its target entries agree with the reference, and then shows that the
comparison refuses its walk: on the fixture it zeroes 76,879 off-target entries, 76,830 of them
under the tier's floor, so a comparison carrying that floor would have taken the kernel whole.
The row check is held to 1e-3 relative on every entry for the same reason, at 248320 wide too.
