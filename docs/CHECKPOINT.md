# CHECKPOINT — what a checkpoint is, and resuming one

A checkpoint is the thing a run leaves behind: everything needed to continue the run where it
stopped, and the adapter the serving side loads. This file owns the artifact — its layout on disk,
the bytes of its state file, the rules for keeping, moving and resuming one. `docs/TRAINING.md`
owns the loop that writes it (when, and what happens when a step is not finite);
`docs/ADAPTER.md` owns the adapter directory inside it.

## On disk

`<out>/checkpoint-<step>/`, with the step zero-padded to nine digits:

```
runs/lora-v1/
  train.jsonl                     the log, one JSON object per line (docs/TRAINING.md)
  run.json                        the resolved configuration, once before the first step (docs/TRAINING.md)
  checkpoint-000000100/
    adapter/                      the adapter directory (docs/ADAPTER.md)
    trainer.state                 the trainer state (below)
  checkpoint-000000200/
    adapter/
    trainer.state
```

The step in the name is the number of completed loop steps, and it is also the `step` inside the
state file. Checkpoints of one run sort by name in the order the loop wrote them, and a directory
whose name has fewer than nine digits was not written by this loop.

`trainer.state` names the container the run trained against in `base_id`: `r97train train` writes
`sha256:` and the container file's SHA-256 there, over whatever the configuration names, and a
resume compares it before restoring anything (`docs/TRAINING.md`). Beside it the state holds the
optimiser's state, the batcher position and the loop's random-number state; the section below is
the exact encoding.

`run.json` sits beside the log rather than inside a checkpoint: the loop writes it once, before its
first step, and a refused resume throws before that, so the directory keeps the configuration it
already had. `docs/TRAINING.md` owns its fields.

A checkpoint is assembled in `checkpoint-<step>.tmp` and renamed into place; `trainer.state` itself
is written to `trainer.state.tmp` and renamed over. A failure while writing therefore leaves no
half checkpoint and no half state file, and an interrupted run leaves the last complete checkpoint.
The loop never deletes a checkpoint, so a long run keeps every `checkpoint_every` step's worth of
adapter weights: size the interval against the disk, not against what the run needs at the end.

## `trainer.state`

The state file is this trainer's private encoding. It is not a container format, not radiance's,
and nothing else in the tree reads it: `read_trainer_state()` does, and `write_trainer_state()`
writes it. Values are in the machine's byte order, with no padding and no checksum.

| field | encoding |
| --- | --- |
| magic | 8 bytes, `R97TRN01` |
| version | `u32`, currently `1` |
| reserved | `u32`, `0` |
| `base_id` | `u64` byte count, then that many bytes: the container's `sha256:<hex>` |
| `step` | `i64`, completed loop steps |
| `position.epoch` | `i64`, the batcher position of the next batch: epoch |
| `position.step` | `i64`, the batcher position of the next batch: step |
| `rng_state` | `u64`, the loop's random state after `step` |
| parameter count | `i64` |
| each parameter | name (`u64` count + bytes), rank `i64`, rank × dimension `i64`, value count `i64`, then that many `f32` |
| optimiser size | `u64` byte count |
| optimiser bytes | `Optimiser::serialise()` at `step` |

A `float` is 4 bytes of the machine's float; a parameter's value count must equal the product of
its dimensions, and its rank must be 0 to 8. The reader refuses, by name of the file, a magic that
is not this one, a version it does not know, a negative step or position, a parameter count larger
than the bytes left can hold, a rank outside 0 to 8, a dimension that is negative, a shape whose
elements overflow or do not agree with the value count, and an optimiser block longer than the rest
of the file. It does not checksum: a file truncated by a full disk fails one of those
checks, and a damaged byte that survives them is not detectable here.

Two rules follow for anyone touching the encoding:

- A new field is a new version. Version `1` readers refuse anything else, so a run in progress
  keeps resuming with the build that wrote it.
- The state file is not portable across machines of different byte order or `float` layout. A run
  moves by copying its `<out>` directory and resuming on the same architecture.

## What a checkpoint restores

`TrainerState` is exactly what `Trainer::state()` returns and what `resume()` puts back:

- the master weights (`f32`, updated in place by the optimiser), by name and shape;
- the optimiser's serialised moments, its update count and its pending accumulation, so the
  learning-rate schedule continues rather than restarting;
- `step`, so `steps` and `eval_every` are counted from where the run is;
- the batcher position, so the next batch is the one the stopped run would have read;
- `rng_state`, so shuffling and any sampled choice continue;
- `base_id`, the container's `sha256:<hex>` the run trained against, so a resume can compare it.

The adapter directory is written by `TrainerOptions::write_adapter`, which the caller supplies;
the loop creates the directory and knows nothing about the format.

## Resuming

```sh
build/r97train train --reference --config runs/lora-v1.conf \
    --resume runs/lora-v1/checkpoint-000000100
```

`--resume` is the `train` command's: it reads `<dir>/trainer.state` and calls
`Trainer::resume(dir)`. A run of `2N` steps that is stopped and continued at `N` resumes from
`checkpoint-<N>`: `checkpoint_every = N` writes it, and so does a `--steps N` run that stops there.
The log is opened for append, so the resumed steps are added to the same `train.jsonl`.

The rules a resume holds to:

- **`base_id` must match.** `r97train train` sets `config.base_id` to `sha256:` plus the SHA-256
  of the container it trains against, so the id in a checkpoint names that container and nothing
  else; `resume()` refuses with both ids in the message before it changes anything. It is how a
  checkpoint of one container is kept from being resumed into another. A caller that drives
  `Trainer` itself supplies its own string, which the trainer treats as opaque
  (`docs/TRAINING.md`).
- **The parameters must be the same adapter.** `resume()` restores the state's parameters into the
  `ParamSet` the run's `Delta` built, in order: a different count, or a parameter whose name, shape
  or size differs, is refused naming it. That comparison is all a resume learns about the delta.
  `delta.rank` and `delta.targets` change the set the `lora` kind builds — `<module>.lora_A` and
  `<module>.lora_B` for each targeted module, `rank` rows deep — so a checkpoint belongs to the
  rank and the targets that wrote it: any other, or a `delta.kind` that builds a different set, is
  refused on a count, a name or a shape. `delta.alpha` and `delta.gate` are not parameters, are not
  in `trainer.state`, and are not compared. `alpha` enters the run as the sites' scale,
  `alpha / rank` (`docs/ADAPTER.md`), so a resume with another one restores the same weights and
  optimiser and then trains a different `StepFn`: accepted, and not bit-identical to the run it
  continues. `delta.gate` asks for a non-LoRA slot that no delta builds yet, so it changes nothing;
  the `lora` kind never reads it.
  The delta configuration a run was trained under is `<out>/run.json`, written before the first
  step; the file the resume uses should repeat it.
- **The step function must be the same.** The bit-identical guarantee below is a guarantee about
  one `StepFn`, and `alpha` is part of it: the sites' scale is what the step applies. Resuming a
  `--reference` run as an eventual device run continues the weights and the schedule, but not bit
  for bit.
- **The horizon is the run's, not the checkpoint's.** `steps` and `epochs` in the resuming
  configuration are the whole run's; a resumed run stops at that horizon, not `horizon + step`.

With a deterministic `StepFn`, `2N` steps in one run and `N` steps resumed for `N` more give
bit-identical parameters: the optimiser's update count and pending accumulation are in the state,
so an update pending at the checkpoint is completed by the continuation, not dropped or repeated.
`tests/host/trainer.cpp` checks each rule — `trainer_state_round_trip`,
`trainer_resume_is_bit_identical`, `trainer_resume_keeps_a_pending_accumulation`,
`trainer_resume_refuses_a_different_base_id` — and `train_reference_cli_overfits_and_resumes`
checks the command line's path through `--resume`.

## Keeping, moving, inspecting

- **Keeping.** `checkpoint_every = 0` means the loop writes only when a run stops, which is the one
  checkpoint a resumed run needs; a long run that wants to survive a crash sets an interval. There
  is no pruning: remove `<out>/checkpoint-*` directories the run no longer needs, keeping the last
  and the one the run will resume from.
- **Moving.** Copy the `<out>` directory (log and checkpoints together) to the new machine and
  resume there, same architecture. A checkpoint directory can be copied on its own: reading it needs
  neither the container nor the data file, only a build that can resume it.
- **Inspecting.** `r97train loss --model <container> --data <jsonl> --lora
  <checkpoint>/adapter` scores the adapter a checkpoint holds against the held-out data without
  resuming the run (`docs/EVAL.md`), which is how checkpoints of one run are compared. The state
  file has no reader on the command line; it is read by `read_trainer_state()`.
- **Serving.** `<checkpoint>/adapter` is what the serving radiance loads, through the patches in
  `patches/serving/`: `docs/ADAPTER.md` says what the directory holds and what radiance reads from
  it.

## What the tests check

`ctest --test-dir build -L host` runs them. The loop's own tests are in `tests/host/trainer.cpp`
and named above; `trainer_log_and_checkpoint_files_match_the_document` is the one that holds this
file's layout — the directory name, `adapter/`, `trainer.state` — against what a real run writes.
