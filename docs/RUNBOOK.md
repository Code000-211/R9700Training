# RUNBOOK — a run, from a bare tree to a served loss

A run, in order: the inputs it needs, the dry run, ten steps, the resume, the serve and the check
that the served loss is the trainer's. Read it once from the top; after that the sections are the
ones you come back to. `docs/USAGE.md` has each command's flags, `docs/TRAINING.md` the loop and its
log, `docs/CHECKPOINT.md` the checkpoint and the resume, and `docs/ADAPTER.md` what a served
adapter holds.

**Every fenced block in this file is one fragment of a single script.** In order the blocks are the
whole procedure, and `tests/host/runbook.cpp` extracts every block, requires its fence to be `sh`,
and runs the fragments against the tiny fixture: a command here that stops working is a failing
host test rather than a stale document. The blocks read `R97_BUILD`, `R97_STEPS` and the four
inputs `R97_MODEL`, `R97_TRAIN`, `R97_HELDOUT`, `R97_RUN`, plus `R97_SERVING_BUILD` for section 7;
the inputs default to the fixture, so the file as it stands is the smoke run, and section 10 sets
them for a real one. A command in backticks in the prose is a one-time input (a build, the fetch)
rather than a step of the script: it costs minutes or the network, and the test does not run it. A
block whose step needs one of those inputs guards it and says so when it is absent (section 7's
serve needs the serving build); the test runs the block's guard branch, so the document still says
what it would have run.

Save the fenced blocks as one script under `runs/` and run it from the tree's top.

## 0. What a run needs

- **The build.** `cmake -B build -G Ninja` and `cmake --build build` — `docs/USAGE.md` has the
  one-off setup and the host-only variant. `build/r97train` is the binary every block below calls.
- **A container** `.rad`. `train` names it by its SHA-256 in `base_id`, so a checkpoint cannot be
  resumed against another container (`docs/CHECKPOINT.md` has the id and what a run writes).
- **The data: two JSONL files in the container's chat template, fetched, not shipped.** The first
  real run uses UltraChat 200k, which is not gated and needs no login: `python3
  tools/fetch_ultrachat.py` writes `data/ultrachat/train.jsonl` (the `train_sft` split) and
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
- **A serving-only build of radiance, for section 7.** It is a one-time input, like the fetch,
  and it takes three steps in this order. `cmake -B build-serving -G Ninja -DR97_PATCH_SET=serving`
  applies the serving patches in `patches/serving/`. `cmake --build build-serving --target
  radiance_core` then configures radiance's own build tree, `build-serving/radiance-build`, and
  builds the engine libraries in it; the parent's configure only makes that directory, so the
  sub-build has no cache before this step and nothing can be built there. `cmake --build
  build-serving/radiance-build --target radiance` builds the server from that cache
  (`build-serving/radiance-build/bin/radiance`; the serving tree has no `r97train`).
  Section 7 serves the trained adapter with it and measures the loss radiance computes for that
  adapter, through the same `--lora` / `--kld-lora` the deployment serves with (issue #61); that
  step also needs `curl` for the completion and `python3` for `tools/kld_loss.py`. A run without
  that build, such as the host-only smoke run, keeps the trainer's `loss --lora` measurement and
  says in section 7 that the serve was skipped; `docs/EVAL.md` has both.
- **Card discipline.** One training run at a time: the card to itself, one process, and `ctest -L
  gpu` takes `gpu_guard` for the same reason (only the issues marked for the two-card machine use
  both cards). **Never stop a process you did not start** — not another run, not someone else's
  `loss` or `ops`, not a `ctest`. If the card is busy, wait for it to end or take another card;
  killing a job that is not yours costs whoever started it the hours it had run. This is the one
  place the rule is written; the other documents point here.

## 1. The run's settings

One block sets the four inputs, so the rest of the file is the same whether it runs on the fixture
or on a real container. The default container is the newest `tiny_qwen35_w8a8.rad` under the test
temporary directory — the one `ctest -L host` just converted. `R97_DEVICE` is the card selector
the `gpu` tests read (`docs/TESTING.md`), and it is empty here because the smoke run has no card:
section 2's block then says what it skipped.

```sh
set -eu
set -o pipefail

: "${R97_BUILD:=build}"                          # where the build is
: "${R97_STEPS:=10}"                             # the horizon, and the run's whole length
: "${R97_TRAIN:=tests/fixtures/chat_small.jsonl}" # the training JSONL
: "${R97_HELDOUT:=$R97_TRAIN}"                   # the held-out JSONL
: "${R97_RUN:=runs/runbook}"                     # where the run writes; must not exist yet
: "${R97_SERVING_BUILD:=$PWD/build-serving}"     # the serving-only build, section 7
: "${R97_DEVICE:=}"                              # the card's ordinal; empty = no card
: "${RADIANCE_HOME:=$R97_BUILD/radiance-build/radiance_home}" # the build's plugins
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
export R97_MODEL R97_STEPS R97_TRAIN R97_HELDOUT R97_RUN R97_SERVING_BUILD R97_DEVICE RADIANCE_HOME
echo "runbook: $R97_MODEL, $R97_STEPS steps, $R97_TRAIN -> $R97_RUN" \
     "(serving build $R97_SERVING_BUILD, card ${R97_DEVICE:-none})"
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

The op inventory says whether this build can run the container at all. The host kernels are the
same check every run makes, and the block below always runs that line; on the card, the same
command with `--device N` asks the device kernels the same question, and the block runs that line
too when `R97_DEVICE` names a card. `docs/OPS-qwen35-host.md` and `docs/OPS-qwen35-device.md` are
the two inventories the output is read against.

```sh
"$r97" ops "$R97_MODEL" --home "$RADIANCE_HOME" --kernels libavx,libref

# The same inventory on the card: --device names the card, and libr4d is tried first. A host tier
# (the smoke run) has no card, so R97_DEVICE stays empty and the block says what it skipped.
if [ -n "$R97_DEVICE" ]; then
    "$r97" ops "$R97_MODEL" --home "$RADIANCE_HOME" --kernels libr4d,libavx,libref \
        --device "$R97_DEVICE"
else
    echo "runbook: R97_DEVICE is empty; the card's inventory (ops --device) was not taken," \
         "docs/OPS-qwen35-device.md has it"
fi
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
`docs/CHECKPOINT.md` lists what the resume refuses and what it holds to. The resume reads the same
`$R97_RUN.conf`, which is what keeps the delta configuration with the checkpoint: `delta.rank` and
`delta.targets` change the parameter set, so a change to either is refused, as is a `delta.kind`
that builds a different one; `delta.alpha` and `delta.gate` change nothing the resume compares, so
a resume that changes either is accepted and continues a run the checkpoint does not describe.

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

## 7. The adapter, and the served loss against the trainer's

`tests/gpu/serving_container.cpp` serves the container on the card's tier and
`tests/model/serving_container.cpp` checks its serving handshake -- basis and targetable linears
-- on the host tier; `ctest -L model` runs both when the container is present (the gpu case also
takes the card). The blocks below are the hand-run form of the same serve, with the numbers
printed.

The checkpoint's `adapter/` is the directory radiance loads (`docs/ADAPTER.md` has what it holds
and what radiance reads), and the run serves the checkpoint the resumed run wrote. The adapter's
held-out loss is measured twice. First the trainer's, with `loss --lora`: it loads the adapter
through radiance's own loader (`lora/preload.h`) and works in any build with the forward tier, so
the smoke run checks it. It must be the held-out loss the resumed run reported in section 6
(`$resumed`):

```sh
adapter="$R97_RUN-resumed/checkpoint-$(printf %09d "$R97_STEPS")/adapter"
trainer_loss=$("$r97" loss --model "$R97_MODEL" --data "$R97_HELDOUT" --lora "$adapter" |
               sed -n 's/^loss: //p')
[ -n "$trainer_loss" ] || {
    echo "runbook: loss printed no loss line for $adapter" >&2
    exit 1
}
echo "the trainer's loss for the adapter $trainer_loss, the run's held-out loss $resumed"

awk -v trainer="$resumed" -v measured="$trainer_loss" 'BEGIN {
    d = measured - trainer
    if (d < 0) d = -d
    t = trainer < 0 ? -trainer : trainer
    if (t == 0) t = 1
    if (d > 1e-4 * t) {
        printf "runbook: loss --lora %s, the trainer reported %s\n", measured, trainer
        exit 1
    }
}'
```

Then the serving build's, from the server that serves the adapter. Section 0's serving-only build
is a one-time input; when it is at hand (`$R97_SERVING_BUILD`), this block starts
`build-serving/radiance-build/bin/radiance` on `127.0.0.1:8099` with the adapter loaded, asks for
one chat completion, and stops it again. The smoke run of a host-only build has no serving tree and
takes the `else` branch, which says so; the trainer's measurement above is what that run keeps.

```sh
serving="$R97_SERVING_BUILD"
server="$serving/radiance-build/bin/radiance"
serving_home="$serving/radiance-build/radiance_home"
served=

if [ -x "$server" ]; then
    "$server" --model "$R97_MODEL" --radiance-home "$serving_home" \
              --lora trained="$adapter" --host 127.0.0.1 --port 8099 \
              > "$R97_RUN-resumed/serve.log" 2>&1 &
    server_pid=$!
    reply=
    for attempt in $(seq 1 60); do
        if curl -fsS http://127.0.0.1:8099/v1/chat/completions \
                 -H 'Content-Type: application/json' \
                 -d '{"model":"trained","messages":[{"role":"user","content":"Hello"}],"max_tokens":8}' \
                 > "$R97_RUN-resumed/serve.json" 2>/dev/null; then
            reply=1
            break
        fi
        sleep 1
    done
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    [ -n "$reply" ] || {
        echo "runbook: the serving radiance did not answer;" \
             "$R97_RUN-resumed/serve.log has its log" >&2
        exit 1
    }
    grep -q '"content"[[:space:]]*:' "$R97_RUN-resumed/serve.json" || {
        echo "runbook: the server answered no completion; see" \
             "$R97_RUN-resumed/serve.json" >&2
        exit 1
    }
    echo "the serving build served $adapter and answered one completion"
else
    echo "runbook: no $server; section 0's serving-only build is a one-time input," \
         "so this run does not serve"
fi
```

The served loss is the number the serving build computes for that same adapter, not the number the
server prints: `data --kld-corpus` writes the held-out file in radiance's KL shape, the server run
with `--kld-lora trained` records its per-position NLL over it, and `tools/kld_loss.py` prints the
loss. It is held to the same `$resumed` at the same 1e-4 relative tolerance — the loss radiance
serves is the loss the trainer reported.

```sh
kld="$R97_RUN-resumed/kld"
mkdir -p "$kld"
"$r97" data --model "$R97_MODEL" --data "$R97_HELDOUT" --kld-corpus "$kld/heldout.jsonl"

if [ -x "$server" ]; then
    "$server" --model "$R97_MODEL" --radiance-home "$serving_home" \
              --lora trained="$adapter" --kld-lora trained \
              --kld-corpus "$kld/heldout.jsonl" --kld-record "$kld/served"
    served=$(python3 tools/kld_loss.py "$kld/served" | sed -n 's/^loss: //p')
    [ -n "$served" ] || {
        echo "runbook: tools/kld_loss.py printed no loss for $kld/served" >&2
        exit 1
    }
    echo "the serving build's loss for the adapter $served, the trainer's held-out loss $resumed"

    awk -v trainer="$resumed" -v measured="$served" 'BEGIN {
        d = measured - trainer
        if (d < 0) d = -d
        t = trainer < 0 ? -trainer : trainer
        if (t == 0) t = 1
        if (d > 1e-4 * t) {
            printf "runbook: the serving build %s, the trainer reported %s\n", measured, trainer
            exit 1
        }
    }'
else
    echo "runbook: no $server; the serving build's loss is unchecked, the trainer's is above"
fi
```

The same adapter on a card is the second half of the same check, and it is a card's commands, not
the host's. The serving build has to be a HIP build (section 0's default); the server has no
`--device` of its own, so `ROCR_VISIBLE_DEVICES` names the card exactly as it does for the trainer;
and the card's own `loss --device` is the number its served loss is held to, since the serve runs
the same three libraries `--device` runs (`--kernels libr4d,libavx,libref`) and a HIP serving home
also holds the trainer's `libr97`, which that list keeps out. `$R97_DEVICE` is the selector section
2 already uses for the card, so the two sections cannot disagree about which one it is. The card's
serve log must say `AQL backend`. A host-only serving build answers the block above, and so does a
HIP build whose command names no kernel library: it finds no card, takes the host backend, and
serves out of `serving_home`'s `libavx`, which is not what is measured here. The line above names
`libr4d`, and a process that can see no card has no device code for it to declare, so it stops at
startup with `invalid argument` rather than serving on the CPU and calling it the card's number.

Two of the block's numbers need a run behind them. `$resumed` is the host's, and a model the host
tier cannot declare -- the 27B container is one (`docs/reports/2026-11-12-device-forward.md`) -- has
none, so there the served number is held to `loss --device` alone; the block below asks for the
host's number only when there is one. And a model whose 256-sequence worst case does not fit the
card needs its limits named on the serve (`--max-num-seqs 8`, `--max-num-batched-tokens 8192`,
`--max-model-len 4096` for the corpus `docs/reports/2026-11-13-serving-card.md` measured) or the
engine stops at startup instead of serving; the block below names those three. Card 0 has taken
that pair for the 27B container. The tiny int8 fixture is not a card's container: its geometry has
no `libr4d` kernel (`docs/DEPLOYMENT.md` section 4), so a card serves section 10's container, and
the smoke run leaves `R97_DEVICE` empty and takes the else branch below.

```sh
if [ -x "$server" ] && [ -n "$R97_DEVICE" ]; then
    trainer_card=$("$r97" loss --model "$R97_MODEL" --data "$R97_HELDOUT" \
                          --lora "$adapter" --device "$R97_DEVICE" | sed -n 's/^loss: //p')
    [ -n "$trainer_card" ] || {
        echo "runbook: the trainer printed no loss for card $R97_DEVICE" >&2
        exit 1
    }
    ROCR_VISIBLE_DEVICES="$R97_DEVICE" "$server" --model "$R97_MODEL" \
              --radiance-home "$serving_home" --max-num-seqs 8 \
              --max-num-batched-tokens 8192 --max-model-len 4096 \
              --lora trained="$adapter" --kernels libr4d,libavx,libref --kld-lora trained \
              --kld-corpus "$kld/heldout.jsonl" --kld-record "$kld/served-card" \
              > "$R97_RUN-resumed/serve-card.log" 2>&1
    grep -q 'AQL backend' "$R97_RUN-resumed/serve-card.log" || {
        echo "runbook: the serving build did not run on card $R97_DEVICE; see" \
             "$R97_RUN-resumed/serve-card.log" >&2
        exit 1
    }
    served_card=$(python3 tools/kld_loss.py "$kld/served-card" | sed -n 's/^loss: //p')
    [ -n "$served_card" ] || {
        echo "runbook: tools/kld_loss.py printed no loss for $kld/served-card" >&2
        exit 1
    }
    echo "the serving build's loss on card $R97_DEVICE $served_card, the trainer's card loss" \
         "$trainer_card, the host's ${resumed:-none}"

    awk -v card="$trainer_card" -v host="${resumed:-}" -v measured="$served_card" \
        -v dev="$R97_DEVICE" '
        function agree(a, b) {
            d = a - b
            if (d < 0) d = -d
            t = b < 0 ? -b : b
            if (t == 0) t = 1
            return d <= 1e-4 * t
        }
        BEGIN {
            if (!agree(measured, card) || (host != "" && !agree(measured, host))) {
                printf "runbook: the serving build on card %s reports %s, the trainer %s there", \
                       dev, measured, card
                if (host == "")
                    printf "; the host tier has no number for this container\n"
                else
                    printf ", the host %s\n", host
                exit 1
            }
        }'
else
    echo "runbook: no serving build or no R97_DEVICE; the card's served loss is unchecked"
fi
```

1e-4 is the tolerance `tests/host/loss_cli.cpp` holds the trainer's loss and radiance's KL record
to, and the same one `tests/host/train_host_cli.cpp` holds the loss across a resume to; the same
one applies to every measurement here, on the host and on the card. The card's serve names the
three kernel libraries the trainer's `loss --device` runs on (`libr4d,libavx,libref`), so the
served number and the number held to it come from the same kernels: a HIP serving home also holds
`libr97`, the trainer's plugin, which radiance would otherwise load beside them. `docs/EVAL.md`
says how the trainer's loss and radiance's record agree in the host tier, and `docs/DEPLOYMENT.md`
says what the serving-only build carries and what the two measurements mean for a deployment.

A checkpoint can be measured at any time without touching the run: `loss` on another `adapter/`
answers whether it is worth keeping. Compare with the baseline of the same container and held-out
file before the run — `"$r97" loss --model "$R97_MODEL" --data "$R97_HELDOUT"`, no `--lora` — and
`docs/EVAL.md` says how the number is cross-checked.

## 8. Watch

Every line of `train.jsonl` is one flat JSON object; `docs/TRAINING.md`'s Logging names each
record's fields. While the run is going, follow the file with any JSONL reader; after it stops,
this block prints the last five train records and every evaluation record, which is what to read
when the run is over:

```sh
grep '"kind":"train"' "$R97_RUN/train.jsonl" | tail -5
grep '"kind":"eval"' "$R97_RUN/train.jsonl"
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
| a run fails at its first step with a device error while another job is on the card | one run per card | wait for the other job to end, or take another card (`R97_DEVICE=N`, which section 2's block passes as `--device`) — never stop a process you did not start (section 0) |

The non-finite stop is the one failure that is not a mistake: the failing step changed no weight, no
moment and no step count, no checkpoint was written for it, so the last good checkpoint is what the
resume loads (`docs/TRAINING.md`, "When a step is not finite"). Keep the message: its sequences are
the whole diagnosis.

## 10. The real run

The same script with seven variables set, and the fetched data in place:

| variable | value |
| --- | --- |
| `R97_MODEL` | `/srv/models/qwen3.8-27b-iq4r-s-v4.rad` |
| `R97_TRAIN` | `data/ultrachat/train.jsonl` |
| `R97_HELDOUT` | `data/ultrachat/heldout.jsonl` |
| `R97_RUN` | `runs/lora-v1` |
| `R97_STEPS` | `1000` |
| `R97_SERVING_BUILD` | `build-serving` (section 0's serving-only build, for section 7) |
| `R97_DEVICE` | `0`: the card section 2's inventory is taken on (`--device "$R97_DEVICE"`); empty keeps the host branch and the card stays unchecked |

Then the blocks of sections 1 to 8 run as they stand — the dry run on the container and the two
files, the run to its horizon, the resume from whatever checkpoint survived, the serve and the
served loss against the trainer's, and the log the run wrote. What changes with the size is the
configuration, not the procedure: a longer `eval_every` and `checkpoint_every` (each checkpoint is a
full adapter and its moments: `du -sh` the first), a `token_budget` and `accumulation` that fit the
card, and an `lr` for the run's length.
`docs/TRAINING.md` has the loop's knobs, `docs/PLAN.md` the runs that are planned.

For a card run, two things are added: the card is the run's alone (section 0's discipline — one
run at a time, and never a process you did not start), and the pre-step inventory of section 2
answers whether the run can start at all; with `R97_DEVICE` naming a card, the block takes it a
second time, `--device N`, and the device kernels answer the same question. An op listed as
missing is a kernel this build does not have and the run will stop on it; `--registered` prints
what the build has at all, and `docs/OPS-qwen35-host.md` and `docs/OPS-qwen35-device.md` are the
reference inventories. Picking the cards a run uses (`--devices`, tracker #57) and splitting a
container across them (`--placement`, tracker #59) are the two-card issues' flags, listed as
planned in `docs/USAGE.md`; until they land, one card is one run on one device.

## 11. What checks this file

`tests/host/runbook.cpp` takes every fenced block above, requires its fence to be `sh`, and runs
the blocks in order on the tiny int8 fixture with `build/r97train`: the extraction is the file's
own procedure, so a flag that changes without the runbook, a runbook command that stops working, or
a non-`sh` block added to the file fails `ctest --test-dir build -L host`. The one-time inputs in
section 0 are inline prose, not blocks, and are not run — the serving-only radiance build, the
network fetch, and `ctest` itself. Section 7's serve and served-loss blocks are in the script and
the test runs them whenever the environment names a serving tree in `R97_SERVING_BUILD`: the server
serves the adapter and answers one completion, and the test prints the pair of losses it took. With
no tree they take their `else` branch and say the serve was skipped, and either way the test holds
the commands to the document, so removing them fails even when the branch did not execute. Building
the serving tree and the server still costs minutes, so the test does not do that; name a host-only
one (section 0's commands with `-DR97_WITH_HIP=OFF`) and it serves the fixture. That run took
`4.26155194807` from the serving build's KL record against the trainer's `4.2615519630392633` —
`1.5e-8` apart, `3.5e-9` relative, well inside the `1e-4`. Section 7's card blocks are untouched by
that run: with no `R97_DEVICE` they take their skip branch and say so, and the card's serve and its
number are taken on the card, where the command names `libr4d` and the log has to show the AQL
backend. Section 2's card inventory is held to this file
the same way: the script always carries the `--device` line, and a run with no card takes the
branch that says it was skipped, so removing the line fails here even though that branch ran. The
flag tables in `docs/USAGE.md` are held to `r97train <command> --help` by
`tests/cmake/config_schema_test.sh`, and the configuration table there is held to that same
document by the same test and to `include/r97/config.h` by `tests/host/config_schema.cpp`, which
counts the header's members against the schema's rows: a key added on either side of that pair
fails.
