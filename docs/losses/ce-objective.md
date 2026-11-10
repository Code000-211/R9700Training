# The reference loss: `CeObjective`

The loss every later issue is checked against, the first `Objective`
(`include/r97/objective.h`), in `include/r97/loss.h` and `src/loss.cpp`. Nothing
else in the engine defines it.

## The loss

One sequence of `L` tokens has `L - 1` logits rows: row `p` predicts token
`p + 1`. A batch's rows are its sequences concatenated, in order. Token `i`
with `score[i] == 1` is a target, predicted at position `i - 1`, so by row
`i - 1` of its sequence (with `score[0]` always 0, since nothing predicts it).

The loss is, over the scored tokens `i` of every sequence of the batch,

```
loss = sum_i  -(logits[i-1][ids[i]] - logsumexp(logits[i-1]))
```

accumulated in **f64**, with a stable log-sum-exp (`max + log(sum(exp(x - max)))`).
A batch returns the sum (`CeSum::loss`) and the count (`CeSum::scored`); the
reported loss is their quotient. Logits may be bf16 or f32: `bf16_to_f32`
converts a bf16 row, and `ce_loss` has a bf16 overload; a `ForwardTape`'s read
already returns f32.

## The gradient

With respect to a scored row: `softmax(row) - one_hot(ids[i])`; with respect to
an unscored row: 0. Scaled by `1 / scored` this is the gradient of the mean
loss (`ce_grad`); `CeObjective`'s seed scales by `1 / total_scored` instead, so
windowed calls sum to the single call's cotangent. The arithmetic — softmax and
the subtraction — happens in f64 and casts to f32 once, at the end. Rows are
independent, so a chunked computation is bit-identical to the whole one.

## Piecewise use

The per-position core (`logsumexp`, `ce_position`) and `ce_sum` over a list of
`ScoredRow`s (`(row, id)` pairs local to the buffer being read) let a caller
feed chunks of rows without holding every logit row: `scored_rows` in
`src/loss.cpp` walks the batch and yields the scored tokens whose prediction
row falls in a window, with row offsets that accumulate `L - 1` per sequence.

## CeObjective on the tape

`CeObjective(fwd, deltas)`:

- reads the boundary `"logits"` in the window `fwd.window` gives, as a
  **differentiated** read (a read the loss depends on, so the backward gets a
  Seed on it; a detached read would not carry one); a window with no scored row
  is not read at all, since there would be nothing to answer the read with,
- takes ids and the score mask from `fwd.batch()`,
- returns `ObjectiveResult`: the window's summed loss (f64), the window's
  scored count, and exactly one `Seed` on `"logits"` spanning the window's
  scored rows (first through last, in the buffer's own row numbers — the ones
  `window` and `read` count in — with zeros on unscored rows inside), scaled by
  `1 / fwd.total_scored()` — of the whole step, not of the window.

Called window by window it gives the same sum and the same cotangent rows as
called once; windows with no scored rows are not read and return an empty seed
list.

## Checked by

`tests/host/loss_ce.cpp`: hand-computed cases, the gradient against central
finite differences, a row with 1e30 logits staying finite, chunked equals
whole, bf16 equals f32. `tests/host/loss_ce_tape.cpp`: on a `VectorTape`
(`tests/support/vector_tape.h`) exactly one seed, on `logits`, bit-identical to
the gradient function's output, its read recorded as differentiated, and the
chunked tape equal to the whole call; and the seed's row range is the window's
span of scored rows — scored rows 2 and 3 of a five-row window give rows
`[2, 4)`, and a window's unscored edge rows are outside it — with a zero row
between two scored ones it spans, and a window with no scored row neither read
nor seeded.

Both run in the `host` tier, whose one binary takes every `tests/host/*.cpp`
and whose entry point belongs to `tests/host/vector_tape.cpp` (issue #1), so a
loss test file brings no `main()` of its own: `R97_SELF_CHECK`
(`tests/support/self_check.h`) runs the function from a static initializer,
before that main, and a failed `CHECK` — live even where `-DNDEBUG` compiles
`assert` out — aborts, which is what makes the binary exit non-zero.
