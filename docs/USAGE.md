# USAGE — running `r97train`

`r97train` is the trainer's one binary: a subcommand per job, each registered in
`tools/commands.h`. This file is the user's walk through it — the build, what a run needs, every
subcommand, the configuration schema — and where each part is documented in depth:

| what | where |
| --- | --- |
| what this project is, and the pinned engine | `README.md`, `docs/UPSTREAM.md` |
| the training loop, its log and its horizons | `docs/TRAINING.md` |
| a checkpoint on disk, and resuming one | `docs/CHECKPOINT.md` |
| operating a real run: preflight, launch, failures | `docs/RUNBOOK.md` |
| the loss and the held-out measurement | `docs/EVAL.md` |
| the training sequences | `docs/DATA.md`, `docs/CHATDATA.md` |
| what an adapter is, and how radiance serves it | `docs/ADAPTER.md` |
| the tiny containers the tests run on | `docs/FIXTURES.md` |
| the three test labels and what each may assume | `docs/TESTING.md` |

## Build, and where the binary is

```sh
git submodule update --init          # fresh clone only
cmake -B build -G Ninja              # HIP from /opt/rocm, target gfx1201
cmake --build build                  # writes build/r97train
```

A host-only build — no card, no ROCm — is the same tree with `-DR97_WITH_HIP=OFF`. The engine is a
pinned submodule; the build copies it, applies `patches/` onto the copy and builds that, so nothing
under `third_party/radiance/` is ever edited. Which patches a build gets is `docs/UPSTREAM.md`'s
subject, not this file's.

The binary finds radiance's plugins (`libref`, `libavx`, …) through `RADIANCE_HOME`, else beside
itself in `<build>/radiance-build/radiance_home`; `docs/FIXTURES.md` says how the build puts them
there. A command that needs them says so, with the directory it looked in.

## What a run needs

- **A container**: a `.rad` file the engine serves. `r97train info` reads one without a card. The
  tests make theirs from the tiny checkpoint (`docs/FIXTURES.md`); a real run uses the 27B container
  under `data/`.
- **Training sequences**: one chat-format JSONL file, tokenised with the container's own chat
  template. `docs/CHATDATA.md` describes the format, what is scored and what is skipped.
- **An output directory**: the run's log and its checkpoints go here, and it must be writable
  before the first step.

## The commands

| command | what it does |
| --- | --- |
| `info <file.rad>` | architecture, quantisation, layers, hidden size |
| `data --model <file.rad> --data <file.jsonl> [--show N]` | the tokenisation statistics, and the first N conversations with their scored runs marked |
| `loss --model <file.rad> --data <file.jsonl> [--lora <dir>] [--limit N] [--device N]` | the held-out loss, cross-checked against radiance's own KL record |
| `grad --model <file.rad> --data <file.jsonl> [--delta lora] …` | one loss and its per-module gradient norms through the host adjoints |
| `train --model <file.rad> --out <dir> [--reference] [flags]` | the training loop, with the host step or, under `--reference`, the reference step |
| `ops <file.rad>\|--registered [--device N]` | the op inventory of a container's forward pass, or the registry |
| `deltas` | the registered trainable kinds, one per line |
| `objectives` | the registered losses, one per line |
| `config [--config <file>] [flags]` | the configuration schema, or the configuration those arguments resolve to |

A command returns `0` on success, `2` for a usage error, and something other than `0` for a check
it could not complete; the message names the file, the key or the parameters at fault. `r97train`
with no argument prints the command table with each command's argument hint.

### `train`

```sh
build/r97train train --model <file.rad> --data <train.jsonl> --heldout <heldout.jsonl> \
    --out runs/lora-v1 --steps 1000 --lr 1e-4 --token_budget 4096 --eval_every 100
```

The step is the ordinary host step: the engine's forward on the host kernels and the host adjoints
(`docs/STEPS-HOST.md`). It needs no card, and it is the one a real run uses. `--reference` replaces
it with the container's op graph replayed in `f64` from the engine's twins and the host adjoints,
with the `libavx` and `libref` kernel libraries: no card, slower, and the same loop, checkpoint and
resume. A build without radiance's forward tier has only the reference step, and `train` without
`--reference` stops and says so.

The command's own flags are `--reference`, `--resume <dir>`, `--program <file>` and `--config
<file>`; every configuration key is accepted too, as `--key value`. The file is read first and a
flag overrides it. Without `--data` the sequences are `tests/fixtures/chat_small.jsonl`, relative to
the working directory — give a real run its own file. `heldout` (or `--heldout <file>`) names the
file `eval_every` evaluates on; `eval_every = 0` turns the evaluation off. The loop logs to
`<out>/train.jsonl` and to the terminal, writes checkpoints under `<out>/`, and its last line is
`train: wrote <checkpoint>`.

`docs/TRAINING.md` says what the loop does with the log, the horizons and the evaluation;
`docs/CHECKPOINT.md` says what the directory it wrote contains and what `--resume` needs.

### `config`

```sh
build/r97train config                              # the schema, one line per key
build/r97train config --config runs/lora-v1.conf --steps 500   # resolved, validated
```

With no argument the command prints the configuration schema — the table below, from the binary, so
a reader can check the document against what the build accepts. With `--config` or any
configuration flag it loads the file, applies the flags over it, validates the result and prints
one `key = value` per line: what a `train` with those arguments will use. A key the file leaves
alone keeps its `config.h` default, and an unknown key or an out-of-range value stops it with the
key's name. It is the preflight `docs/RUNBOOK.md` runs before a long run.

`config` applies no fallbacks of its own. `train` has four — `lr = 1e-3`, `steps = 1`,
`token_budget = 2048`, `accumulation = 1` — so a run may leave those out of its file and still
start, while `r97train config --config` on the same file refuses it: their `config.h` default is
outside the valid range. Write them in the file and the two agree.

### The rest

- `info` and `data` are the two cheap reads of a run's inputs. `data` prints the counts the batcher
  will see (kept, shortened, skipped, tokens, scored) and marks the scored runs in the first
  `--show` conversations with `[[…]]`.
- `loss` and `grad` are the held-out and the gradient check: `docs/EVAL.md` and
  `docs/STEPS-HOST.md` own what they print and how the numbers are cross-checked.
- `ops` answers "can this build run this container, on the host and on the card": the inventory
  the host tier prints is `docs/OPS-qwen35-host.md`, the card's is `docs/OPS-qwen35-device.md`.
- `deltas` and `objectives` print what is registered — `docs/OBJECTIVES.md` says what a new one is
  and where it registers.

## Configuration

One `key = value` per line. A line whose first non-space character is `#` is a comment, blank
lines are ignored, and a value is everything after the first `=` with the ends trimmed (so a path
may contain spaces; `#` is a comment only at the start of a line). Targets are a comma-separated
list.

A flag is `--key=value` or `--key value`, with the same key names. `--config` is the command
line's own flag, not a configuration key.

| key | type | default (`config.h`) | valid range and meaning |
| --- | --- | --- | --- |
| `model` | string | empty | the container to train against; the loop does not read it |
| `data` | string | empty | the training sequences; the loop does not read it |
| `heldout` | string | empty | the held-out sequences; the caller turns them into `TrainerOptions::heldout` |
| `out` | string | empty | **required**: the output directory for the log and checkpoints |
| `objective` | string | `ce` | **non-empty**; the name is the `StepFn`'s business, the loop only carries it |
| `base_id` | string | empty | opaque caller identifier stored in every checkpoint; `resume` refuses a different one |
| `delta.kind` | string | `lora` | **non-empty**; the delta's name, not read by the loop |
| `delta.rank` | integer | 16 | **> 0**: the rank of the trainable factors |
| `delta.alpha` | number | 32 | **finite, > 0**: the LoRA scale, applied as `alpha / rank` |
| `delta.targets` | list | empty | comma-separated non-empty module names; empty means the delta's default |
| `delta.gate` | boolean | false | `true`/`false` (also `1`/`0`, `yes`/`no`, `on`/`off`): train a gate beside the sites |
| `lr` | number | 0 | **finite, > 0**: the peak learning rate; must be set |
| `weight_decay` | number | 0 | finite, >= 0: the optimiser's weight decay |
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

The default column is `include/r97/config.h`: a default lives there and nowhere else, and
`r97train config` prints it from a default-constructed `Config`. `tests/cmake/config_schema_test.sh`
keeps this table and the binary's schema equal — a key, a type, a default or a range edited here is
edited in `src/config_schema.cpp` too, or the test says which row differs.

A complete file:

```
model = /srv/models/qwen3.8-27b-iq4r-s-v4.rad
data = /data/chat/train.jsonl
heldout = /data/chat/heldout.jsonl
out = runs/lora-v1
objective = ce
base_id = qwen3.8-27b-iq4r-s-v4
delta.kind = lora
delta.rank = 16
delta.alpha = 32
delta.targets = q_proj, v_proj
delta.gate = false
lr = 1e-4
weight_decay = 0.0
warmup_steps = 20
token_budget = 4096
accumulation = 1
steps = 1000
seed = 7
eval_every = 100
checkpoint_every = 100
```

`r97train config --config that-file` prints the same keys in schema order and validates them; the
run is then `build/r97train train --config that-file`. `train` writes the container's SHA-256 into
`base_id`, over whatever the file names; the key is for a caller that drives `Trainer` itself
(`docs/TRAINING.md`).

## A first run, end to end

Over the tiny fixture container — no card and no data of your own. `docs/FIXTURES.md` owns the
fixture: the host tests make it, and it lands in their temporary directory as
`<tmpdir>/tiny_qwen35_bf16/tiny_qwen35_bf16.rad`. These commands find it and keep the path in
`$tiny`:

```sh
ctest --test-dir build -L host          # once: the host tests make the tiny container
tiny=$(find "${RAD_TEST_TMPDIR:-${TMPDIR:-/tmp}}" \
            -path '*tiny_qwen35_bf16/tiny_qwen35_bf16.rad' -printf '%T@ %p\n' \
       | sort -rn | head -1 | cut -d' ' -f2-)
```

`$tiny` is the tiny run's `model`; the file below is its configuration, not the real one above:

```sh
mkdir -p runs
cat > runs/tiny.conf <<CONF
model = $tiny
data = tests/fixtures/chat_small.jsonl
out = runs/tiny
objective = ce
delta.kind = lora
delta.rank = 4
delta.alpha = 8
delta.targets = q_proj, v_proj
lr = 1e-3
token_budget = 2048
accumulation = 1
steps = 20
checkpoint_every = 10
seed = 7
CONF

build/r97train info "$tiny"                                  # arch, quant, layers, hidden
build/r97train data --model "$tiny" --data tests/fixtures/chat_small.jsonl --show 2
build/r97train config --config runs/tiny.conf                # the schema, resolved
build/r97train train --reference --config runs/tiny.conf     # 20 steps; checkpoint-000000010, -000000020
build/r97train train --reference --config runs/tiny.conf --resume runs/tiny/checkpoint-000000010
build/r97train loss --model "$tiny" --data tests/fixtures/chat_small.jsonl \
    --lora runs/tiny/checkpoint-000000020/adapter
```

The first `train` writes `checkpoint-000000010` and the one it stops on,
`checkpoint-000000020`, with `run.json` beside them. The second resumes from step 10 and stops at
the file's horizon, 20 — `steps` is the whole run's, so the resumed run does steps 11 to 20 again
and the log gets a second copy of them (`docs/CHECKPOINT.md`). `loss` prints `loss: <mean per
scored token>` and the counts behind it.

The terminal line is one per step and `train.jsonl` holds the same numbers as fields;
`docs/TRAINING.md`'s Logging owns the line and what each field means.

A checkpoint directory holds `trainer.state` and `adapter/`; the adapter directory is what `--lora`
reads and what the serving side loads. `docs/CHECKPOINT.md` says what the state file holds and what
`--resume` requires. `docs/RUNBOOK.md` takes the commands through a real run's checks and failures.

## What the host build can and cannot do

- No card: everything in this file except `--device`, the `gpu` tests and the card's own inventory.
  `train` (the host step, or `--reference`), `loss`, `grad`, `data`, `info`, `ops`, `config` are
  host commands.
- A build without radiance's forward tier has no ordinary host step; `train` without `--reference`
  stops with `train: this build has no forward driver; use --reference with the host step`. The
  device step is a later issue's; `docs/PLAN.md` says which.
- `loss` and `grad` need a build with radiance's forward tier, and a plugin home; a build without it
  does not register them. `train` and `loss` find the plugin home from `RADIANCE_HOME` or the one
  the build keeps beside the binary.
- `--device N` selects the card through `ROCR_VISIBLE_DEVICES`; a `gpu` test takes `gpu_guard` so
  two runs never share the one card.

## The checks behind this file

`ctest --test-dir build -L host` runs them all. The ones a reader of this file will want by name:

| what | test |
| --- | --- |
| every command's listing and registration | `listings`, `tests/host/test_registrations.cpp` |
| `info` and `data` on a container | `info_cli`, `data_cli` |
| the schema, against `Config` and against this file | `tests/host/config_schema.cpp`, `config_schema` |
| the file/flags/defaults order | `config_file_overrides_command_fallbacks` |
| `train` (the host step), the held-out evaluation and the resume | `train_host_cli_overfits_and_resumes` |
| `train --reference`, overfit and resume | `train_reference_cli_overfits_and_resumes` |
| the log and the checkpoint directory | `trainer_log_and_checkpoint_files_match_the_document` |
| resume identity and the non-finite stops | `trainer_resume_is_bit_identical`, `trainer_resume_refuses_a_different_base_id`, `trainer_non_finite_loss_stops_and_keeps_the_last_good_checkpoint` |
| the held-out loss | `loss_cli_matches_radiance_kld` |
