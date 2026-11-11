# Training

`include/r97/trainer.h` and `src/trainer.cpp` are the loop. It schedules one `StepFn`
(`include/r97/step.h`) over the caller's `ParamSet`, `Optimiser` and `Batcher`, logs every step,
evaluates the held-out set, and writes checkpoints that resume bit-identically. It never reads the
objective's or the delta's name: both are inside the `StepFn` it is given.

The caller owns everything the loop references. It builds the batcher from the training
sequences, the optimiser from `optimiser_config(config)`, the held-out batches, and the adapter
writer (the adapter-files code). The loop only schedules them.

## Configuration

One `key = value` per line. A line whose first non-space character is `#` is a comment, blank
lines are ignored, and a value is everything after the first `=` with the ends trimmed (so a path
may contain spaces; `#` is a comment only at the start of a line). Targets are a comma-separated
list. `parse_config()` reads text, `read_config()` reads a file, and `load_config(path, flags)`
reads the file, applies the flags over it and validates the result.

A flag is `--key=value` or `--key value`, with the same key names. `--config` is the command
line's own flag, not a configuration key.

An unknown key, a malformed value and a value outside its range are all `ConfigError`s that name
the key (`error.key()` and `error.what()`).

| key | type | default (`config.h`) | valid range and meaning |
| --- | --- | --- | --- |
| `model` | string | empty | the container to train against; the loop does not read it |
| `data` | string | empty | the training sequences; the loop does not read it |
| `heldout` | string | empty | the held-out sequences; the caller turns them into `TrainerOptions::heldout` |
| `out` | string | empty | **required**: the output directory for the log and checkpoints |
| `objective` | string | `ce` | **non-empty**; the name is the `StepFn`'s business, the loop only carries it |
| `base_id` | string | empty | opaque caller identifier stored in every checkpoint; `resume` refuses a different one |
| `delta.kind` | string | `lora` | **non-empty**; the delta's name, not read by the loop |
| `delta.rank` | integer | 16 | **> 0** |
| `delta.alpha` | number | 32 | **finite, > 0** |
| `delta.targets` | list | empty | comma-separated non-empty module names; empty means the delta's default |
| `delta.gate` | boolean | false | `true`/`false` (also `1`/`0`, `yes`/`no`, `on`/`off`) |
| `lr` | number | 0 | **finite, > 0**: the peak learning rate; must be set |
| `weight_decay` | number | 0 | finite, >= 0 |
| `warmup_steps` | integer | 0 | >= 0, counted in optimiser updates |
| `token_budget` | integer | 0 | **> 0**: the batcher's tokens per sub-batch; must be set |
| `accumulation` | integer | 0 | **>= 1**: micro-steps per optimiser update; must be set |
| `steps` | integer | 0 | >= 0: the loop-step horizon |
| `epochs` | integer | 0 | >= 0: the epoch horizon |
| `seed` | integer | 0 | >= 0: the batcher shuffle and the loop's random state |
| `eval_every` | integer | 0 | >= 0: evaluate after every N steps; 0 disables evaluation |
| `checkpoint_every` | integer | 0 | >= 0: checkpoint after every N steps; 0 means only when a run stops |

Exactly one of `steps` and `epochs` is positive. `out`, and the values marked **must be set**,
are required because their `Config` default is outside the valid range.

```
model = /srv/models/qwen3.8-27b-iq4r-s-v4.rad
data = /data/chat/train.jsonl
heldout = /data/chat/heldout.jsonl
out = runs/lora-v1
objective = ce
delta.kind = lora
delta.rank = 16
delta.alpha = 32
delta.targets = q_proj, v_proj
lr = 1e-4
weight_decay = 0.0
warmup_steps = 20
token_budget = 4096
accumulation = 1
steps = 1000
seed = 7
eval_every = 100
checkpoint_every = 100
base_id = sha256:8f3a...
```

## The run

`run()` runs to the horizon: `steps` loop steps, or until the batcher's epoch counter reaches
`epochs`. `run_steps(n)` runs at most `n` more steps and returns how many it took; it stops at the
horizon too. Both write a checkpoint when they stop (see below).

One loop step takes one step from the batcher: one sub-batch per `devices`, and the `StepFn` is
called once per sub-batch. The gradients are averaged over the sub-batches, the losses and scored
tokens are summed, and the average gradient is given to the optimiser. `accumulation` is passed to
the optimiser unchanged: it sums that many micro-steps and applies one update.

`optimiser_config()` is what the caller must build the `Optimiser` with, so the learning rate the
loop logs is the one that is applied:

- `lr` and `weight_decay` are passed through;
- `warmup_steps` is passed through and counted in updates;
- `accum_steps = accumulation`;
- `total_steps = ceil(steps / accumulation)` for a step horizon. A run measured in epochs has no
  known horizon, so `total_steps` is 0 and the schedule holds its value after warm-up.

After every completed step the loop advances its random state with one `splitmix64` draw. The loop
does not draw a random number for anything else today; the state is saved so a step that shares
the state (or a future sampling decision) resumes on the same stream. The batcher's own shuffle is
covered by its position.

Evaluation runs after the steps where `step % eval_every == 0` (`eval_every > 0`) and once more at
the end of a run if that step was not already evaluated. It calls the evaluation `StepFn` with
`grads == nullptr` on every held-out batch and logs

```
held-out loss = sum(loss) / sum(scored)
```

over the whole held-out set. If no token was scored the loss is logged as 0 and `scored` is 0.
Evaluation does not change the weights.

## Logging

Every train step writes one line to the terminal (when `TrainerOptions::terminal` is not null) and
one JSON object to `<out>/train.jsonl` (always). The file is opened for append, so resuming
continues the same log. Evaluation writes a second kind of line.

```
step 12/1000 epoch 0 loss 2.70318 lr 0.0001 |g| 1.4312 4096 tok 512345.6 tok/s step 0.0079s update 0.0004s
eval step 12 loss 2.6114 3980 scored tokens 410002.1 tok/s
```

One JSON line per step, one flat object:

```json
{"kind":"train","step":12,"epoch":0,"loss":2.70318,"scored":4096,"lr":0.0001,"grad_norm":1.4312,"tokens":4096,"tokens_per_s":512345.6,"step_s":0.0079,"update_s":0.0004,"updated":true}
{"kind":"eval","step":12,"epoch":0,"loss":2.6114,"scored":3980,"tokens":3900,"tokens_per_s":410002.1,"eval_s":0.0095,"batches":4}
```

| field | train | eval | meaning |
| --- | --- | --- | --- |
| `kind` | yes | yes | `"train"` or `"eval"` |
| `step` | yes | yes | completed loop steps; the step the line is about |
| `epoch` | yes | yes | the batcher epoch the step came from |
| `loss` | yes | yes | train: the summed loss over the step's sub-batches; eval: `sum(loss) / sum(scored)` |
| `scored` | yes | yes | scored tokens the loss covered |
| `lr` | yes | no | the learning rate of the update this step made, or of the next update when accumulation held it back |
| `grad_norm` | yes | no | the global gradient norm before clipping |
| `tokens` | yes | yes | token ids in the step's batches (all sub-batches) |
| `tokens_per_s` | yes | yes | `tokens / (step_s + update_s)`, or `tokens / eval_s` for eval |
| `step_s` | yes | no | wall time from the first `StepFn` call to the last one |
| `update_s` | yes | no | wall time of the gradient check, the norm and `Optimiser::step` |
| `updated` | yes | no | false when accumulation held the update back |
| `eval_s` | no | yes | wall time of the whole held-out pass |
| `batches` | no | yes | held-out batches evaluated |

`StepResult` reports no timing split today, so the loop reports the split it can measure: the
`StepFn` call(s) (`step_s`) and everything the loop does around them (`update_s`). When a
`StepResult` grows its own breakdown, this document and these two fields are the place to carry
it.

## Checkpoints

A checkpoint is a directory `<out>/checkpoint-<step>` with the step zero-padded to nine digits,
holding:

```
checkpoint-000000100/
  adapter/          the adapter directory, written by TrainerOptions::write_adapter
  trainer.state     the trainer state (below)
```

The loop writes one after every `checkpoint_every` completed steps, and once when `run()` /
`run_steps()` stops if the step it stopped on has not just been written. `checkpoint_every = 0`
therefore still leaves the checkpoint a resumed run needs. A checkpoint is assembled in
`checkpoint-<step>.tmp` and renamed into place, so a failure while writing leaves no half
checkpoint; an interrupted run leaves the last complete one.

The loop does not know the adapter format. It creates `<checkpoint>/adapter` and calls
`TrainerOptions::write_adapter(params, dir)`; the adapter-files code writes the files the serving
radiance reads. The callback is required.

`trainer.state` is this trainer's private encoding (magic `R97TRN01`, a version, then lengths and
values in the machine's byte order; only `read_trainer_state()` reads it):

- `base_id`, the caller's opaque string;
- `step`, the completed loop steps;
- the batcher position (`epoch`, `step`) of the next batch to read;
- `rng_state` after that step;
- the master weights: for each parameter its name, shape and `float` values;
- the optimiser's `serialise()` bytes (its moments, update count and pending accumulation).

### Resume

`Trainer::resume(checkpoint_dir)` reads `trainer.state`, compares its `base_id` with the run's
`config.base_id`, and refuses before touching anything when they differ, with both ids in the
message. Otherwise it restores the master weights, the optimiser bytes, the batcher position, the
step and the random state. With a deterministic `StepFn`, a run of `2N` steps and a run of `N`
steps resumed for `N` more give bit-identical parameters: the optimiser update count and its
pending accumulation are part of the state, so the schedule continues rather than restarting.

The checkpoint to resume is the one at `N`; both `run_steps()` stopping there and
`checkpoint_every = N` produce it.

## When a step is not finite

A non-finite training loss, gradient, gradient norm, or held-out loss throws `NonFiniteError` and
ends the run. The error names the failing step (the step that was being computed) and every
sequence of the batch, by sub-batch and index, with its length and first token ids, for example:

```
r97: non-finite gradient of 'model.layers.0.mlp.down_proj.lora_A' at step 5; sequences: device 0 sequence 0 (len 4, ids 151644 872 198 17), device 0 sequence 1 (len 3, ids 9906 1917 13)
```

The loop checks the loss and the gradients before the optimiser sees them, so the failing step
does not change the weights, moments or step count, and no checkpoint for it is written: the last
good checkpoint stays on disk and is what `resume` loads.

A held-out failure is noticed after the training step completed. The step's checkpoint, when one
is due, is written before the held-out pass starts, so the checkpoint of the step the failing
evaluation was measuring is kept.
