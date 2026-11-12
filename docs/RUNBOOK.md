# RUNBOOK — operating a run

What an operator does, in order: check the inputs, launch, watch, stop and resume, and finish. Each
step names the command and what a healthy and an unhealthy result look like; the last section lists
the host tests that run every command here, so the procedure is checked with the tree.

Today the step function that exists is `--reference` (the engine's op graph replayed in `f64`, no
card); `train` without it stops with `the host step is not wired in this build; use --reference`.
Where this file says "card run", the procedure is what the run will do once the device step lands —
`docs/PLAN.md` says which issue — and the preflight commands in it already answer a card's questions
(`ops --device`, `loss --device`).

## 0. What a run needs

- `build/r97train` from a finished build (`docs/USAGE.md`).
- A container `.rad`, and its id — the name you will put in `base_id`.
- A training JSONL and a held-out JSONL, in the container's chat template.
- An output directory with room for the adapters: one checkpoint every `checkpoint_every` steps,
  each holding a full copy of the adapter's weights plus the optimiser's moments. `du -sh` the
  first one before deciding the interval.
- For a card run: the card to itself. One card runs one job; `ctest -L gpu` takes `gpu_guard` for
  the same reason, and only the issues marked for the two-card machine use both cards.

## 1. Preflight

Do these in order. Each is cheap, and each one that fails saves the run.

```sh
build/r97train                                   # the commands this build has
build/r97train info /data/models/qwen3.8-27b-iq4r-s-v4.rad
```

`info` prints `arch`, `quant`, `layers`, `hidden`. A container that does not open exits `3` with the
engine's message: check the path, and whether it was converted for this engine.

```sh
build/r97train data --model /data/models/qwen3.8-27b-iq4r-s-v4.rad \
    --data /data/chat/train.jsonl --show 2
```

`conversations` is the file's count, `kept` the ones the batcher will use, `shortened` the ones
that hit the token limit, `skipped` the ones the template could not render, `tokens` what the run
will read, `scored` the tokens the loss is taken over. `skipped` equal to `conversations` is a
wrong template or a wrong format — `docs/CHATDATA.md` — not a data problem to train through. `--show
2` prints the first conversations with the scored runs in `[[…]]`: read them once before a long run.

```sh
build/r97train config --config runs/lora-v1.conf
```

This is the run's configuration resolved and validated: one `key = value` per line, in schema order,
and a non-zero exit naming the first key that is unknown or out of range. `build/r97train config`
with no arguments prints the schema the file has to fit. Check in what it prints: `out` names the
run's directory, `base_id` names the container, `steps` (or `epochs`) is the horizon the operator
wants, and `checkpoint_every` is not `0` for a run long enough to lose.

```sh
build/r97train loss --model /data/models/qwen3.8-27b-iq4r-s-v4.rad --data /data/chat/heldout.jsonl
```

The baseline held-out loss, before training. `loss: <mean per scored token>` and `scored: <count>`;
write both down — the run's checkpoints are judged against them, and a resumed run is checked the
same way. `docs/EVAL.md` says how the number is cross-checked.

For a card run, one more:

```sh
build/r97train ops /data/models/qwen3.8-27b-iq4r-s-v4.rad --device 0
```

The inventory of the container's forward pass on that card: every op the build has. An op that is
missing is a kernel this build does not have, and the run will stop on it; `--registered` prints
what the build has at all. `docs/OPS-qwen35-device.md` is the reference inventory.

## 2. Launch

```sh
mkdir -p runs/lora-v1
build/r97train train --reference --config runs/lora-v1.conf \
    2>&1 | tee -a runs/lora-v1/console.log
```

The loop owns `runs/lora-v1/train.jsonl`; the console copy keeps the terminal lines, which is what
an operator reads after the fact. It writes its first checkpoint at the first step the interval
falls on, or when it stops.

Before the first step the engine prints its own lines: with a container it can serve, the prefill
count, and — in a host-only build — a `host-backend` warning that the engine is on the CPU, which is
what `-DR97_WITH_HIP=OFF` means and not an error. Then one line per step:

```
step 2/3 epoch 0 loss 47.243417315629088 lr 0.0003250000299885869 |g| 31.531862258911133 53 tok 250.74407476257244 tok/s step 0.19507576099999999s update 0.016295136000000002s
```

(That line is the fixture's second step: the numbers and the horizon are the run's, and they print
at full double precision, so `train.jsonl` and `jq` are how they are read in bulk.) `loss` is the
step's mean per scored token, `|g|` the gradient norm before the update, `tok/s` the throughput, and
`update` the optimiser's share of the step — a step time that is all `update` is a card that is not
the bottleneck. A loss that is not finite ends the run at once (section 5).

## 3. Watch

```sh
tail -f runs/lora-v1/train.jsonl
jq -r 'select(.kind=="train") | [.step,.loss,.grad_norm,.tokens_per_s] | @tsv' runs/lora-v1/train.jsonl | tail -5
jq -r 'select(.kind=="eval") | [.step,.loss] | @tsv' runs/lora-v1/train.jsonl
```

Every line is one flat JSON object: `train` records carry `step`, `epoch`, `loss`, `scored`, `lr`,
`grad_norm`, `tokens`, `tokens_per_s`, `step_s`, `update_s`, `updated` (`updated` is `false` on the
micro-steps an accumulation step has not completed); `eval` records carry `step`, `epoch`, `loss`,
`scored`, `tokens`, `tokens_per_s`, `eval_s`, `batches`. `docs/TRAINING.md` owns the line.

What to look for: a loss that falls and then plateaus, a `grad_norm` that grows by an order of
magnitude (lower `lr` or look at the data), `updated` `true` once per `accumulation` records, and an
evaluation that is worse than the previous by more than noise (the run is overfitting; stop it and
keep the earlier checkpoint). The held-out loss of a checkpoint can be measured at any time without
touching the run — section 6.

## 4. Interrupt, and resume

`train` installs no signal handler: `Ctrl-C` ends the process, and what survives is the last
complete checkpoint — a checkpoint is renamed into place, so a half-written one is never left. If
`checkpoint_every` is `0`, an interrupted run has only the checkpoint of a run that stopped by
itself, so a run that may be interrupted sets an interval, and a run that is interrupted on purpose
gets a short `steps` horizon rather than a kill.

```sh
ls runs/lora-v1                                  # the checkpoints that survived, newest last
build/r97train train --reference --config runs/lora-v1.conf \
    --resume runs/lora-v1/checkpoint-000000400
```

The log is opened for append and the step numbering continues from the checkpoint. Before resuming,
check three things: `base_id` in the file still names the same container, the delta keys are the
ones the checkpoint was written with, and `steps` is the whole run's horizon (the resumed run stops
there, not at `400 + steps`). `docs/CHECKPOINT.md` has the rules the resume holds to and the
bit-identical guarantee for a deterministic step.

## 5. When something fails

The exit status is non-zero and the message on stderr names the thing at fault. The frequent ones:

| what you see | what it means | what to do |
| --- | --- | --- |
| `r97: config: 'lr': must be > 0 (got 0)` | a key outside its valid range, or a required key left empty (`'out': must not be empty`) | `build/r97train config --config <file>`; the message names the key, `docs/USAGE.md` its range |
| `r97: config: 'lr.rate': unknown key` | a typo, or a key this build does not have | run `build/r97train config` for the schema |
| `train: the host step is not wired in this build; use --reference` | `train` without the reference step | add `--reference`; the device step is a later issue's |
| `loss`/`data` exit `3` | the container did not open | the path, or the container was not converted for this engine |
| `data` prints `skipped: <conversations>` | every conversation failed the template | `docs/CHATDATA.md`; check `info`'s `arch` against the template, and `--show 1` |
| `r97: cannot resume checkpoint '<dir>': it was trained on base_id '<a>', this run's base_id is '<b>'` | the checkpoint belongs to another container | resume with the `base_id` it was written with, or pick another checkpoint |
| `r97: non-finite gradient of '<param>' at step N; sequences: ...` | the step's gradient was not finite; the step changed nothing | the message names the sequences, by sub-batch and index, with lengths and first token ids: find them in the JSONL, fix or drop them, resume from the last checkpoint |
| a write error on `checkpoint-<step>` | the disk filled, or the directory is not writable | free space, then resume; the last complete checkpoint is intact |
| the run stops at a step it should have passed | the horizon: `steps` is the whole run's | set the horizon the operator means and resume |
| `r97train loss: no radiance home: set RADIANCE_HOME to the directory holding architectures/, kernels/ and quantizers/` (`train --reference` fails with `reference step: live trace failed: …` instead) | the build's plugin directory was not found | `RADIANCE_HOME`, or build where the binary looks (`docs/FIXTURES.md`) |
| the card is in use | one run per card | wait for it, or use another card; the read-only commands take `--device N` to pick one |

The non-finite stop is the one failure that is not a mistake: the failing step changed no weight,
no moment and no step count, and no checkpoint was written for it, so the last good checkpoint is
what the resume loads (`docs/TRAINING.md`, "When a step is not finite"). Keep the message: its
sequences are the whole diagnosis.

## 6. Finish, evaluate, keep, serve

When the run reaches its horizon it writes its last checkpoint and prints
`train: wrote runs/lora-v1/checkpoint-000001000`.

```sh
build/r97train loss --model /data/models/qwen3.8-27b-iq4r-s-v4.rad --data /data/chat/heldout.jsonl \
    --lora runs/lora-v1/checkpoint-000001000/adapter
```

Compare with the baseline from the preflight: the same container, the same held-out file, the same
`--limit` if one was used. That number decides whether the run is worth keeping, and comparing two
checkpoints decides which one to serve:

```sh
for step in 400 700 1000; do
    printf '%s ' "$step"
    build/r97train loss --model /data/models/qwen3.8-27b-iq4r-s-v4.rad --data /data/chat/heldout.jsonl \
        --lora "runs/lora-v1/checkpoint-$(printf %09d "$step")/adapter" | head -1
done
```

Then:

- **Keep** the checkpoint you serve and the one the run would resume from; delete the rest only
  after the numbers are written down, and keep `train.jsonl` with them: it is the run's record.
- **Serve** it: the checkpoint's `adapter/` is what the serving radiance loads, through
  `patches/serving/`. `docs/ADAPTER.md` says what the directory holds and what radiance reads.
- **Report** the configuration, the baseline and the checkpoint's held-out loss: `config`'s output
  is the run's configuration, so paste it rather than retyping it.

## 7. Every command here is a host test

| command in this file | test (`ctest --test-dir build -L host`) |
| --- | --- |
| `build/r97train` and its command table | `usage_cli` |
| `config` with no argument, and with a file and flags | `config_schema` |
| `train --reference`, its log and its checkpoints, then `--resume` | `train_reference_cli_overfits_and_resumes` |
| the terminal line and the JSON records | `trainer_log_and_checkpoint_files_match_the_document` |
| the resume rules and the interrupted run | `trainer_resume_is_bit_identical`, `trainer_resume_keeps_a_pending_accumulation`, `trainer_resume_refuses_a_different_base_id` |
| the non-finite stop that keeps the last good checkpoint | `trainer_non_finite_loss_stops_and_keeps_the_last_good_checkpoint` |
| `loss`, with and without an adapter, and its refusal | `loss_cli_matches_radiance_kld`, `loss_cli_lora_matches_radiance_kld`, `loss_cli_reports_an_unloadable_lora` |
| `info` and `data` on a container | `info_cli`, `data_cli` |
| `ops` on a container, and the registry | `ops_inventory_bf16_fixture`, `ops_inventory_w8a8_fixture` |
| the range and unknown-key refusals | `trainer_config_rejects_a_value_out_of_range_naming_the_key`, `trainer_config_rejects_an_unknown_key` |
