# OPTIM: AdamW on a `ParamSet`

`r97::Optimiser` (`include/r97/optim.h`, `src/optim.cpp`) updates a `ParamSet` in place:
its f32 values are the master weights. No model involved.

## Update

Per element, with `t` the 1-based update index and `g` the (clipped, accumulated) gradient:

```
m  = b1*m + (1-b1)*g
v  = b2*v + (1-b2)*g*g
w -= lr(t) * ( (m/(1-b1^t)) / (sqrt(v/(1-b2^t)) + eps) + wd * w )
```

Weight decay is decoupled (AdamW) and applies to every parameter by default; a parameter
whose name contains a substring in `AdamWConfig::no_decay` skips it.

## Schedule

`scheduled_lr(cfg, step)`: linear warm-up from 0 to `lr` over `warmup_steps`, then cosine
down to `lr * lr_floor_frac` at `total_steps`, flat at the floor beyond it. With
`total_steps <= 0` the rate stays at `lr` after warm-up.

## Clipping and accumulation

`clip_global_norm(grads, max_norm)` rescales the whole gradient set so its global norm
(the square root of the sum of squares over all parameters, computed in double) is at most
`max_norm`; it is deterministic — the cross-thread partial sums are combined in fixed chunk
order, so a restart reproduces the same bits.

With `accum_steps = k` each `step()` call sums its gradients into a buffer; the k-th call
applies the mean (`buffer/k`), clipped, as one AdamW step and clears the buffer. `step()`
returns whether it applied an update; `pending()` reports buffered gradients so far.

## State

`serialise()` writes the weights, both moments, the accumulation buffer, the update step
and the pending count into one byte buffer (format documented in `optim.h`);
`restore(bytes)` stages everything and only then swaps it in, so bad bytes leave the
optimiser unchanged. Restoring and continuing is bit-identical to not stopping — checked
by a host test, both mid-window and between updates.

## Threading

Elements across all parameters are cut into chunks and dispatched to worker threads over
both moments and weights in the same pass. `threads = 0` uses `std::thread::hardware_concurrency`.

## Measured: one update at 10^8 parameters

The adapter's scale: one 10^8-parameter tensor on 48 threads (`threads = 0`, every core
here), f32 weights + moments + accumulation buffer + gradients (about 2 GB of state),
clipping on -- the step a training run takes, not a bare loop over weights.

`tests/host/optim.cpp` measures it, but only when asked: the `host` label runs anywhere and
2 GB is not anywhere, so the measurement is opt-in and asserts nothing.

```
R97_OPTIM_BENCH=100000000 ./build/r97_host_tests
```

It prints the time of one warm update. On this run's host:

| machine                      | one AdamW update (warm) |
| ---------------------------- | ----------------------- |
| this run's host (48 threads) | 0.24 s                  |

Five runs after warm-up gave 0.232 s to 0.239 s, peak RSS 1.9 GB. Well under the
one-second bar.
