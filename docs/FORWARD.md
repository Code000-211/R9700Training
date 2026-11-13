# The training forward

The driver that replaces radiance's HTTP server for training: a `Batch` in, the loss its `Model`'s
objective computes out. `include/r97/forward.h` is the API; this file says what one step is, what
it gives the backward, and why it refuses what it refuses.

## A step is one prefill

`ForwardDriver::step` runs every sequence of the batch as **one prefill step from position 0**,
the sequences together in that one step when they fit. Two things follow from that, and both are
the reason it is shaped this way:

- attention reads only the keys that step wrote, because the pool starts empty;
- the gated-delta-net state starts from zero, so a sequence's state is a function of its own
  tokens and nothing else.

So speculation, the drafter, the multi-token-prediction head and the vision tower stay off, and a
batch the step cannot hold is refused rather than cut: a batch over `max_tok` tokens (or a
container that would need a second step for it) throws with a message saying that one step holds
the batch. A prompt fills `max_ctx` exactly if it must: the step draws no token, so the engine
reserves no generation slot for it, and the last position of a prompt of `max_ctx` tokens is still
in the pool. The key-value pool is **bf16**, and only bf16: the backward differentiates through the
pool, and bf16 is the one dtype with an adjoint here, so `ForwardConfig::kv_dtype` set to anything
else is refused when the driver opens, not when a step runs.

`ForwardConfig` is the container, the plugin home, the host kernel tier (`libavx`, `libref`), and
the sizes: tokens and ctx one step may hold, sequences one step may hold, and how many rows of the
`logits` boundary one objective call reads.

## The one boundary, in windows

The plain forward has a single boundary, `"logits"`: the rows that predict a scored token, `L-1`
per sequence, in the batch's own row space. A batch longer than `window_rows` rows is read in
chunks of positions, **one objective call per window**, and the driver sums each `Result`'s `loss`
and `scored` into the step's. A `Result` whose window holds no scored row is not an error: the
objective returns nothing for it, and nothing is added.

The logits rows are read **through** to the engine's per-prompt buffers — the step keeps no copy
of them — and the driver checks each sequence's rows against the engine's own order rather than
trusting it.

Two row spaces meet here. An objective reads the batch's, where a sequence takes `L-1` rows
because the last position predicts nothing. The head's rows are the step's, where a sequence
takes all `L` positions and the step writes one row more a sequence before its first token (a
sequence's first row is at `logits rows - tokens + <its positions before it>`). The driver keeps
both and maps between them; a batch that does not sit in the batch's order in the step is what
the check above refuses, because the hidden rows and the block boundaries would land on the wrong
sequence.

`loss` is a **sum**, not a mean: what the objectives return, summed over the windows. So the loss
of a batch is the sum of the losses of its sequences run alone, and the token-weighted mean is
`loss / scored` — which is also the `scored` the objective normalises its gradients by. A test on
this is `forward_batch_of_three_is_the_weighted_mean`.

## What a step gives the backward

- `seeds()` — every window's seeds in window order, with `check_seeds` run on each `Result`
  **before** anything walks it. A differentiated read the objective made must be answered by a
  seed on that buffer; a missing one is a bug in the loss and throws here, not in the walk.
- `hidden()`, `hidden_rows()`, `hidden_cols()` — the final hidden state before the head, at the
  scored rows only, widened to f32. Row `i` of `hidden()` is row `hidden_rows()[i]` of the head's
  own input, which is the row a walk attaches a cotangent to; the batch's row space is not that
  space, so the two numbers differ by what the step wrote before the sequence.
- a block hook — `set_block_hook` is called at every block bracket of the step, with a
  `BlockView` of the residual at that block's entry or exit. The view is **live only for the
  length of the call**: the architecture writes that stream in place, so a caller that needs it
  later copies it. This is what a replay needs, and it cannot be reconstructed from `tape()` after
  the step.
- `tape()` — the step's graph, recorded from the architecture's own issues, and the engine that
  owns the operand buffers, which the driver keeps alive as long as it holds the step.

The `StepFn` that `step()` returns ignores `grads`: this is the forward half of the join. The step
stays in `last()` until the next one replaces it; a `StepFn` keeps the driver, and so the engine,
alive for exactly as long as the tape it recorded.

## One engine at a time

Radiance keeps **one plugin registry for the process** and refuses a second live engine with
`duplicate registration`. So a step releases the engine of the step before it before it starts its
own, and `last()`'s rows are gone once the next step begins; two `ForwardDriver`s cannot be open
at once, and neither can a driver and `run_prefill`. A training loop is one driver, one step after
another, which is what this shape asks for anyway.

The block hook is the one place where the fixture's architectures differ: `arch/qwen35_fp8` and
its block-quantised rebuilds (`qwen35_w8a8`) call `rad_block_begin`/`rad_block_end`, and
`arch/qwen35_bf16` does not. On the bf16 fixture the hook is therefore never called, and a
boundary test has to use the int8 one until the bf16 architecture brackets its blocks too. A
block-quantised build may also ping-pong two residual buffers between blocks, so the values carry
from one block's exit to the next block's entry but the pointer need not.

## The driver holds no loss and no adapter kind

The driver calls the `Model`'s `Objective` and nothing else — no loss lives in `src/forward.cpp`,
which is why a test can hand it its own objective (the sum of one logit column over the scored
rows, say) and get that objective's own value out. It never looks at the `Delta`: what the backward
trains is the model's business.

`src/objective_ce_register.cpp` is the one line that joins the cross-entropy loss to the name
`ce`; see `docs/OBJECTIVES.md`.

## What the driver needs from radiance

The training patch set, and only there: prefill mode without a sampled token
(`patches/serving/0202`, `0204`) and the issue observer with the rank's context
(`patches/training/0101`, `0119`). A serving build applies none of them, so `R97_FORWARD` is 0
there and the driver does not exist — the same rule `include/r97/tape.h` follows.
