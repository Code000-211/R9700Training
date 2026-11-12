# RUNBOOK — a run, from a bare tree to a served loss

A run, in order: the inputs it needs, the dry run, ten steps, the resume, the serve and the check
that the served loss is the trainer's. Read it once from the top; after that the sections are the
ones you come back to. `docs/USAGE.md` has each command's flags, `docs/TRAINING.md` the loop and its
log, `docs/CHECKPOINT.md` the checkpoint and the resume, and `docs/ADAPTER.md` what a served
adapter holds.

**Every `sh` block in this file is one fragment of a single script.** In order the blocks are the
whole procedure, and `tests/host/runbook.cpp` extracts them verbatim and runs them against the tiny
fixture: a command here that stops working is a failing host test rather than a stale document. The
blocks read `R97_BUILD`, `R97_STEPS` and the four inputs `R97_MODEL`, `R97_TRAIN`, `R97_HELDOUT`,
`R97_RUN`; the inputs default to the fixture, so the file as it stands is the smoke run, and section
10 sets them for a real one. A command in `backticks` in the prose is a one-time input (a build,
the fetch) rather than a step of the script: it costs minutes or the network, and the test does not
run it.

Save the blocks as one script and run it from the tree's top:

```text
mkdir -p runs
sed -n '/^```sh$/,/^```$/p' docs/RUNBOOK.md | grep -v '^```' > runs/runbook.sh
bash runs/runbook.sh
```

## 0. What a run needs

- **The build.** `cmake -B build -G Ninja` and `cmake --build build` — `docs/USAGE.md` has the
  one-off setup and the host-only variant. `build/r97train` is the binary every block below calls.
- **A container** `.rad`. `train` names it by its SHA-256 in `base_id`, so a checkpoint cannot be
  resumed against another container (`docs/CHECKPOINT.md` has the id and what a run writes).
- **The data: two JSONL files in the container's chat template, fetched, not shipped.** The first
  real run uses UltraChat 200k, which is not gated and needs no login:

  ```text
  python3 tools/fetch_ultrachat.py
  ```

  That writes `data/ultrachat/train.jsonl` (the `train_sft` split) and
  `data/ultrachat/heldout.jsonl` (the `test_sft` split); `docs/DATA.md` owns the format, the
  licence, and what the script refuses. The fetch needs the `hf` CLI and the network, it is
  idempotent (it skips a split whose file exists), and `docs/CHATDATA.md` says what a file has to
  look like if you bring your own. The smoke run needs none of it: it trains on
  `tests/fixtures/chat_small.jsonl`, which is checked in.
- **The tiny int8 fixture** (for the smoke run): `ctest --test-dir build -L host` once. It converts
  the fixture checkpoint to `tiny_qwen35_w8a8.rad` in the test binary's temporary directory, which
  is what the settings below find; `docs/FIXTURES.md` owns it.
- **An output directory that does not exist yet.** A run makes it; one checkpoint every
  `checkpoint_every` steps holds a full copy of the adapter plus the optimiser's moments, so `du
  -sh` the first one before deciding the interval. The smoke run writes under `runs/`.
- **The serving build** (for section 7): `cmake -B build-serving -G Ninja -DR97_PATCH_SET=serving`
  and `cmake --build build-serving`. Once, before the first serve.
- **Card discipline.** One training run at a time: the card to itself, one process, and `ctest -L
  gpu` takes `gpu_guard` for the same reason (only the issues marked for the two-card machine use
  both cards). **Never stop a process you did not start** — not another run, not someone else's
  `loss` or `ops`, not a `ctest`. If the card is busy, wait for it to end or take another card;
  killing a job that is not yours costs whoever started it the hours it had run. This is the one
  place the rule is written; the other documents point here.

## 1. The run's settings

One block sets the four inputs, so the rest of the file is the same whether it runs on the fixture
or on a real container. The default container is the newest `tiny_qwen35_w8a8.rad` under the test
temporary directory — the one `ctest -L host` just converted.

```sh
set -eu
set -o pipefail

: "${R97_BUILD:=build}"                          # where the build is
: "${R97_STEPS:=10}"                             # the horizon, and the run's whole length
: "${R97_TRAIN:=tests/fixtures/chat_small.jsonl}" # the training JSONL
: "${R97_HELDOUT:=$R97_TRAIN}"                   # the held-out JSONL
: "${R97_RUN:=runs/runbook}"                     # where the run writes; must not exist yet
r97="$R97_BUILD/r97train"

if [ -z "${R97_MODEL:-}" ]; then
    containers=$(find "${RAD_TEST_TMPDIR:-${TMPDIR:-/tmp}}" -name tiny_qwen35_w8a8.rad \
                      -printf '%T@ %p\n' | sort -rn)
    newest=${containers%%$'\n'*}
    R97_MODEL=${newest#* }
fi
[ -n "$R97_MODEL" ] || {
    echo "runbook: no tiny container under ${RAD_TEST_TMPDIR:-${TMPDIR:-/tmp}};" \
         "run 'ctest --test-dir $R97_BUILD -L host' once, or set R97_MODEL" >&2
    exit 1
}
export R97_MODEL R97_STEPS R97_TRAIN R97_HELDOUT R97_RUN
echo "runbook: $R97_MODEL, $R97_STEPS steps, $R97_TRAIN -> $R97_RUN"
```

## 2. The container and the data, read before anything runs

`info` says what the container is (`arch`, `quant`, `layers`, `hidden`) and exits `3` when it does
not open: check the path, and that it was converted for this engine. `data` says what the file will
do: `conversations` is its count, `kept` what the batcher takes, `shortened` the ones that hit the
token limit, `skipped` the ones the template could not render, `tokens` what the run will read,
`scored` the tokens the loss is taken over. `--show 2` prints the first conversations with the
scored runs in `[[…]]`: read them once. A `skipped` count above zero is a wrong template or a wrong
format (`docs/CHATDATA.md`), not something to train through.

```sh
"$r97" info "$R97_MODEL"
"$r97" data --model "$R97_MODEL" --data "$R97_TRAIN" --show 2

skipped=$("$r97" data --model "$R97_MODEL" --data "$R97_TRAIN" | sed -n 's/^skipped: *//p')
[ "$skipped" = 0 ] || {
    echo "runbook: $skipped conversation(s) skipped; docs/CHATDATA.md says why" >&2
    exit 1
}
```

## 3. The configuration

One file, written once, is what the dry run, the run and the resume all read. Every key, its type
and its valid range are `docs/USAGE.md`'s table; `r97train config` with no argument prints the same
schema. `model`, `data`, `heldout` and `out` come from the settings above; `delta.targets` is every
module the int8 trunk can carry an adapter on — `linear_attn.out_proj` is stored unrotated and is
refused, which is why the list is written out rather than left to the default (`docs/ADAPTER.md`).

```sh
mkdir -p "$(dirname "$R97_RUN")"
cat > "$R97_RUN.conf" <<CONF
model = $R97_MODEL
data = $R97_TRAIN
heldout = $R97_HELDOUT
out = $R97_RUN
objective = ce
delta.kind = lora
delta.rank = 16
delta.alpha = 32
delta.targets = q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,gate_proj,up_proj,down_proj
lr = 1e-2
token_budget = 2048
accumulation = 1
steps = $R97_STEPS
eval_every = $R97_STEPS
checkpoint_every = $((R97_STEPS / 2))
seed = 7
CONF
```

## 4. The dry run

Two commands, and the second is the one that matters. `config --config` reads the file, applies no
fallback and validates every key, then prints the configuration resolved: a `key = value` line per
key in schema order, and a non-zero exit naming the first key that is unknown or out of range. That
is the file's spelling checked against the schema.

`train --dry-run` goes further: it opens the container's tokenizer, reads both files, builds the
delta, starts the step's engine, and stops before the first step. It prints the same resolved
configuration and one line that says what the first step would read. It writes nothing — the output
directory is made by the run, not by the dry run — which is the point: a mistyped path or a key
out of range costs seconds here and hours into a run.

```sh
"$r97" config --config "$R97_RUN.conf" > "$R97_RUN.resolved"
grep -qx "steps = $R97_STEPS" "$R97_RUN.resolved" || {
    echo "runbook: the resolved configuration does not read $R97_STEPS steps:" >&2
    cat "$R97_RUN.resolved" >&2
    exit 1
}

"$r97" train --dry-run --config "$R97_RUN.conf"
[ ! -e "$R97_RUN" ] || {
    echo "runbook: the dry run wrote $R97_RUN;" \
         "a run's directory must not exist before it starts" >&2
    exit 1
}
```

## 5. Ten steps

The loop owns `$R97_RUN/train.jsonl`; the console copy keeps the terminal lines, which is what an
operator reads after the fact. The first checkpoint falls on the first step the interval divides,
and the run writes its last one at the horizon. Before the first step the engine prints its own
lines — in a host-only build a `host-backend` warning that the engine is on the CPU, which is what
`-DR97_WITH_HIP=OFF` means and not an error. Then one line per step and an `eval` line at the
horizon, and `train.jsonl` the same numbers as fields.

```sh
"$r97" train --config "$R97_RUN.conf" 2>&1 | tee "$R97_RUN.console.log" || {
    echo "runbook: the run failed; section 9 has the frequent ones" >&2
    exit 1
}

for step in $((R97_STEPS / 2)) "$R97_STEPS"; do
    checkpoint="$R97_RUN/checkpoint-$(printf %09d "$step")"
    [ -d "$checkpoint/adapter" ] || {
        echo "runbook: no checkpoint at $checkpoint; docs/CHECKPOINT.md has the layout" >&2
        exit 1
    }
done
```

## 6. Resume, and the check that it is the same run

`train` installs no signal handler, so `Ctrl-C` ends it where it is; what survives is the last
complete checkpoint — one is assembled as `checkpoint-<step>.tmp` and renamed into place, so a
half-written one is never left (`docs/CHECKPOINT.md`). A
resume names the checkpoint and an output directory, and continues the step numbering from there.
The horizon is the run's, not the remainder's: `steps` is where the resumed run stops.

The check is the whole point of the section. The smoke run trains ten steps straight and then
resumes from its half-way checkpoint into a second directory; the two runs' last held-out losses
must be the same number digit for digit, because the step is deterministic and the checkpoint
carries every input of it. A resumed run that drifts is a checkpoint that lost something —
`docs/CHECKPOINT.md` lists what the resume refuses and what it holds to.

```sh
half=$((R97_STEPS / 2))
"$r97" train --config "$R97_RUN.conf" \
    --resume "$R97_RUN/checkpoint-$(printf %09d "$half")" \
    --out "$R97_RUN-resumed" 2>&1 | tee "$R97_RUN-resumed.console.log" || {
    echo "runbook: the resume failed; docs/CHECKPOINT.md has the refusals" >&2
    exit 1
}

# The last held-out loss the log holds for the run's horizon.
horizon_loss() {
    sed -n 's/.*"kind":"eval","step":'"$R97_STEPS"',.*"loss":\([^,]*\),.*/\1/p' "$1" | tail -n 1
}
straight=$(horizon_loss "$R97_RUN/train.jsonl")
resumed=$(horizon_loss "$R97_RUN-resumed/train.jsonl")
[ -n "$straight" ] && [ -n "$resumed" ] || {
    echo "runbook: no eval line for step $R97_STEPS; set eval_every below steps and resume" >&2
    exit 1
}
[ "$straight" = "$resumed" ] || {
    echo "runbook: the resumed run's step-$R97_STEPS loss is $resumed," \
         "the straight run's $straight" >&2
    exit 1
}
```

## 7. Serve, and the served loss against the trainer's

The radiance a server runs is the serving patch set alone: `patches/serving/` applied to the pinned
submodule, no training hook and no trainer anywhere in it. `-DR97_PATCH_SET=serving` is that build
(`docs/UPSTREAM.md` says what the set must stand alone from), and section 0 built it once. Our own
build lists the radiance targets it needs and the server is not one of them; the serving tree's own
CMake has it, beside the architecture plugin the tiny int8 container needs, so the one-time step
that turns the serving build into a server is:

```text
export RADIANCE_HOME=$PWD/build-serving/radiance-build/radiance_home
cmake --build build-serving/radiance-build --target radiance qwen35_w8a8
```

The checkpoint's `adapter/` is the directory that build loads (`docs/ADAPTER.md` has what it holds
and what radiance reads). The server holds the terminal it starts in; `--lora` gives the adapter the
name a request asks for, and the run serves the checkpoint the resumed run wrote:

```text
adapter="$R97_RUN-resumed/checkpoint-$(printf %09d "$R97_STEPS")/adapter"
build-serving/radiance-build/bin/radiance --model "$R97_MODEL" --radiance-home "$RADIANCE_HOME" \
    --lora trained="$adapter" --host 127.0.0.1 --port 8080
```

A second terminal asks it for a completion, and an answer that is not empty is the proof that the
serving-only build serves this adapter:

```text
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
    -d '{"model":"trained","messages":[{"role":"user","content":"Hi"}],"max_tokens":16}' \
    | jq -r '.choices[0].message.content'
```

A server answers requests; it does not score one. The served loss is radiance's own measure of the
same adapter over the same scored tokens: its KL mode records the adapter's next-token NLL at every
scored position, and `tools/kld_loss.py` prints the mean and the count in the format `r97train
loss` uses (`docs/EVAL.md`). The corpus is the held-out file in radiance's shape — one document per
contiguous scored run — which `r97train data --kld-corpus` writes, so its rows are exactly the rows
the trainer scores:

```text
mkdir -p "$R97_RUN-kld"
"$r97" data --model "$R97_MODEL" --data "$R97_HELDOUT" --kld-corpus "$R97_RUN-kld/heldout.jsonl"
build-serving/radiance-build/bin/radiance --model "$R97_MODEL" --radiance-home "$RADIANCE_HOME" \
    --lora trained="$adapter" --kld-lora trained \
    --kld-corpus "$R97_RUN-kld/heldout.jsonl" --kld-record "$R97_RUN-kld/served"
python3 tools/kld_loss.py "$R97_RUN-kld/served"
```

`kld_loss.py` prints the loss of what was served; it must match the held-out loss the resumed run
reported in section 6 (`$resumed`) to 1e-4 relative. 1e-4 is the tolerance
`tests/host/train_host_cli.cpp` and `tests/host/loss_cli.cpp` hold these two numbers to, and the
same one applies here. The number also comes from the trainer's `loss --lora`, which loads the
adapter with radiance's own loader (`lora/preload.h`) on the same host kernels the serving build
compiles, and the script below measures it and compares it with `$resumed`, so a host-only pass over
this runbook still checks the loss against the trainer's:

```sh
adapter="$R97_RUN-resumed/checkpoint-$(printf %09d "$R97_STEPS")/adapter"
served=$("$r97" loss --model "$R97_MODEL" --data "$R97_HELDOUT" --lora "$adapter" |
         sed -n 's/^loss: //p')
[ -n "$served" ] || {
    echo "runbook: loss printed no loss line for $adapter" >&2
    exit 1
}
echo "served loss $served, the trainer's held-out loss $resumed"

awk -v trainer="$resumed" -v served="$served" 'BEGIN {
    d = served - trainer
    if (d < 0) d = -d
    t = trainer < 0 ? -trainer : trainer
    if (t == 0) t = 1
    if (d > 1e-4 * t) {
        printf "runbook: served %s, the trainer reported %s\n", served, trainer
        exit 1
    }
}'
```

A checkpoint can be measured at any time without touching the run: `loss` on another `adapter/`
answers whether it is worth keeping. Compare with the baseline of the same container and held-out
file before the run — `"$r97" loss --model "$R97_MODEL" --data "$R97_HELDOUT"`, no `--lora` — and
`docs/EVAL.md` says how the number is cross-checked.

## 8. Watch

Every line of `train.jsonl` is one flat JSON object; `docs/TRAINING.md`'s Logging names each
record's fields. While the run is going:

```text
tail -f "$R97_RUN/train.jsonl"
jq -r 'select(.kind=="train") | [.step,.loss,.grad_norm,.tokens_per_s] | @tsv' "$R97_RUN/train.jsonl" | tail -5
jq -r 'select(.kind=="eval") | [.step,.loss] | @tsv' "$R97_RUN/train.jsonl"
```

What to look for: a loss that falls and then plateaus; a `grad_norm` that grows by an order of
magnitude (lower `lr`, or look at the data); `updated` `true` once per `accumulation` records; an
evaluation worse than the previous one by more than noise (the run is overfitting — stop it and keep
the earlier checkpoint); a step time that is all `update` (the card is not the bottleneck). A loss
that is not finite ends the run at once, with the message section 9 describes.

## 9. When something fails

The exit status is non-zero and the message on stderr names the thing at fault. The frequent ones:

| what you see | what it means | what to do |
| --- | --- | --- |
| `r97: config: 'lr': must be a finite value > 0 (got 0)` | a key outside its valid range, or a required key left empty (`'out': must not be empty`) | `"$r97" config --config <file>`; the message names the key, `docs/USAGE.md` its range |
| `r97: config: 'lr.rate': unknown key in <file>:<line>` | a typo, or a key this build does not have | `"$r97" config` for the schema |
| `train: this build has no forward driver; use --reference with the host step` | `train` without `--reference` in a build without radiance's forward tier | add `--reference` (slow, no card), or use a build with the forward tier |
| `loss`/`data` exit `3` | the container did not open | the path, or the container was not converted for this engine |
| `data` prints `skipped: <conversations>` | every conversation failed the template | `docs/CHATDATA.md`; check `info`'s `arch` against the template, and `--show 1` |
| `r97: cannot resume checkpoint '<dir>': it was trained on base_id '<a>', this run's base_id is '<b>'` | the checkpoint belongs to another container | resume it against the container it was trained on; `train` computes `base_id` from that container's SHA-256 |
| `r97: non-finite loss at step N; sequences: ...` (kind `gradient` for the backwards pass) | the step's loss or gradient was not finite; the step changed nothing | the message names the sequences, with lengths and first token ids: find them in the JSONL, fix or drop them, resume from the last checkpoint |
| a write error on `checkpoint-<step>` | the disk filled, or the directory is not writable | free space, then resume; the last complete checkpoint is intact |
| the run stops before the step you meant | `steps` is the run's whole horizon, not the remainder's | set `steps` to where the run should end and resume |
| `train: no radiance home: set RADIANCE_HOME to the directory holding architectures/, kernels/ and quantizers/` (`r97train loss: …` for `loss`; `--reference` fails with `reference step: live trace failed: …` instead) | the build's plugin directory was not found | `RADIANCE_HOME`, or build where the binary looks (`docs/FIXTURES.md`) |
| a run fails at its first step with a device error while another job is on the card | one run per card | wait for the other job to end, or take another card with `--device N` on the read-only commands — never stop a process you did not start (section 0) |

The non-finite stop is the one failure that is not a mistake: the failing step changed no weight, no
moment and no step count, no checkpoint was written for it, so the last good checkpoint is what the
resume loads (`docs/TRAINING.md`, "When a step is not finite"). Keep the message: its sequences are
the whole diagnosis.

## 10. The real run

The same script with four variables set, and the fetched data in place:

```text
R97_MODEL=/srv/models/qwen3.8-27b-iq4r-s-v4.rad
R97_TRAIN=data/ultrachat/train.jsonl
R97_HELDOUT=data/ultrachat/heldout.jsonl
R97_RUN=runs/lora-v1
R97_STEPS=1000
```

Then the blocks of sections 1 to 7 run as they stand — the dry run on the container and the two
files, the run to its horizon, the resume from whatever checkpoint survived, and the served loss
against the trainer's. What changes with the size is the configuration, not the procedure: a longer
`eval_every` and `checkpoint_every` (each checkpoint is a full adapter and its moments: `du -sh` the
first), a `token_budget` and `accumulation` that fit the card, and an `lr` for the run's length.
`docs/TRAINING.md` has the loop's knobs, `docs/PLAN.md` the runs that are planned.

For a card run, two things are added: the card is the run's alone (section 0's discipline — one
run at a time, and never a process you did not start), and the pre-step inventory answers whether
the run can start at all:

```text
"$r97" ops "$R97_MODEL" --device 0
```

An op listed as missing is a kernel this build does not have and the run will stop on it;
`--registered` prints what the build has at all, and `docs/OPS-qwen35-host.md` and
`docs/OPS-qwen35-device.md` are the reference inventories. Picking the cards a run uses
(`--devices`, tracker #57) and splitting a container across them (`--placement`, tracker #59) are
the two-card issues' flags, listed as planned in `docs/USAGE.md`; until they land, one card is one
run on one device.

## 11. What checks this file

`tests/host/runbook.cpp` takes every `sh` block above, in order, and runs it on the tiny int8
fixture with `build/r97train`: the extraction is the file's own procedure, so a flag that changes
without the runbook, or a runbook command that stops working, fails `ctest --test-dir build -L
host`. The one-time inputs in section 0 are not run — a radiance build, the network fetch, and
`ctest` itself — and neither is section 7's serve, which needs the serving build and the server it
builds; the host tier has neither. What the test does check is that section 7 keeps naming them: the
target that builds the server out of the serving tree, the server started on the adapter, the corpus
`--kld-corpus` writes, and the record `tools/kld_loss.py` reads as the served loss. Run those once
by hand before a serve; the serving build has no test of its own beyond `patch_sets` applying the
patch set. The flag tables in `docs/USAGE.md` are held to `r97train <command> --help` by
`tests/cmake/config_schema_test.sh`, and the configuration table there is held to that same document
by the same test and to `include/r97/config.h` by `tests/host/config_schema.cpp`, which counts the
header's members against the schema's rows: a key added on either side of that pair fails.
