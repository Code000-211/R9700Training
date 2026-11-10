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

Single host test `optim` (`tests/host/optim.cpp`), threads=48, one 10^8-parameter tensor,
f32 weights + moments + accumulation buffer (about 2 GB of state):

| machine                         | one AdamW update (warm) |
| ------------------------------- | ----------------------- |
| this run's host (48 threads)    | 0.23 s                  |

Well under the one-second bar. The test asserts under 1 s and prints its own measurement.
