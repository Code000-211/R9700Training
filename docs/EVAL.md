# EVAL

The held-out loss of a container is measured twice: by the trainer, and by radiance itself. The
two must agree, or what the trainer reports is not what radiance serves.

## The trainer's loss

`r97train loss` tokenises a chat JSONL file with the container's own chat-data code, runs the
training forward driver over it, and prints the mean loss of the scored (assistant) tokens.

```sh
build/r97train loss --model tiny_qwen35_bf16.rad --data tests/fixtures/chat_small.jsonl
```

It prints two lines:

```
loss: 5.12345678901
scored: 87
```

`loss` is the sum of the scored tokens' cross-entropy divided by `scored`, the number of tokens
the loss covered. The tokenisation is the one training uses: the chat template, the assistant
mask and the container's tokenizer. The forward is the one training uses: the container's
architecture on the host kernels, a bf16 key-value pool, speculation off.

`--lora <dir>` loads an adapter directory through radiance's own loader and scores through it.
`--limit N` scores only the first N conversations of the file, for a quick look.

The plugin home is `$RADIANCE_HOME`; beside the build it defaults to
`<build>/radiance-build/radiance_home`.

## The same loss from radiance's KL record

Radiance's KL mode scores a corpus and records the reference's next-token NLL per position
(`core/engine_kld.cpp`). `tools/kld_loss.py` reads that record and prints the same two numbers.

```sh
radiance --model tiny_qwen35_bf16.rad \
         --radiance-home "$RADIANCE_HOME" \
         --kld-corpus /tmp/chat_small.kld.jsonl \
         --kld-record /tmp/chat_small.kld

python3 tools/kld_loss.py /tmp/chat_small.kld
```

The corpus is one JSONL document per contiguous scored run of a conversation: `{"prompt":
"<the conversation prefix through that run>", "score_from": <the row predicting the run's first
token>}`. Its rows are then exactly the rows the trainer scores. `tests/host/loss_cli.cpp` builds
that corpus from `ChatData` for `tests/fixtures/chat_small.jsonl` and checks the two losses agree
to 1e-4 relative.

`tools/kld_loss.py` reads the NLL column of `rows.f32` and prints it in the same format as
`r97train loss`; it accepts full and `--kld-top` records.
