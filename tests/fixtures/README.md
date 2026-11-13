# Test fixtures: scored sequences, small containers, hand-built ModelMeta sources.

- `adapter_ref/` — a hand-written bf16 down_proj adapter, the reference the adapter writer is
  compared to (`tests/host/adapter_io.cpp`).
- `serving_adapter/` — one adapter `r97train` wrote on the tiny int8 fixture, with the reference
  rows of one served corpus through it. `tests/host/serving_prefill.cpp` scores that corpus in both
  builds and holds the bytes equal; the directory's `README.md` prints the run that made it.
- `serving_rows.f32`, `serving_logits.f32` — the reference serve of that corpus, written by the
  same test in the default build: the next-token NLL and entropy of every scored position, and the
  f32 logits they were scored from.
- `w8a8_fwht128.recipe` — the tiny fixture's int8 conversion recipe (`docs/FIXTURES.md`).
- `chat_small.jsonl` — the scored conversations `r97train` trains and evaluates on.
