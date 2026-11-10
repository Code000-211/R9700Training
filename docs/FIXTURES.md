# Fixtures

The tiny models host tests run on. A fixture is a checkpoint written on the fly, converted to a
`.rad` container, and loaded in-process through `rad_core` — no HTTP server, no real weights, no
card. `tests/support/fixture.h` is the API; this file says what the fixture is and how a test
obtains it.

## The tiny bf16 Qwen3.5

`r97::fixture()` writes radiance's tiny Qwen3.5 checkpoint (`tests/tiny_model.h`, `write_qwen35`)
once per test binary, writes the ChatML chat template into a `tokenizer_config.json` beside it, and
converts the result with `rad-convert` to a bf16 container that the `qwen35_bf16` plugin serves.

| property    | value                                                                     |
| ----------- | ------------------------------------------------------------------------- |
| arch        | `qwen3_5`, served by the `qwen35_bf16` plugin; the checkpoint's BF16 tensors are kept (no quantisation recipe) |
| hidden      | 128 — a multiple of 128, since the int8 fixture is converted from the same checkpoint |
| layers      | 4, with `full_attention_interval = 4`: layers 0–2 are gated delta net (`linear_attention`), layer 3 is full attention |
| attention   | 4 query heads of 32 over 2 key-value heads (grouped-query attention)      |
| ffn         | 256                                                                       |
| vocabulary  | 320; the tokenizer is byte-level, so ids 0–255 are the bytes and 256–258 the specials (`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`) |
| context     | 4096 trained, 512 in the engine the helper builds                         |
| tokenizer   | ChatML `chat_template` in `tokenizer_config.json`; `write_qwen35` writes none, and radiance reads the template only from that file (or a sibling `chat_template.jinja`), never from `tokenizer.json` |
| weights     | seeded by `write_qwen35`, so the same checkpoint every run                |

The checkpoint lands in the test binary's temporary directory (`radtest::tmpdir()`, which honours
`RAD_TEST_TMPDIR` and otherwise makes one under `$TMPDIR`), in a directory `tiny_qwen35_bf16/`
that holds the checkpoint and, beside it, the container `tiny_qwen35_bf16.rad`.

## Obtaining it

```c++
#include "support/fixture.h"

const r97::Fixture& f = r97::fixture();   // written and converted on first use
```

`Fixture` holds the checkpoint directory, the container path, and the shape (`n_vocab`, `n_layers`,
`hidden`). The conversion runs `rad-convert` as a child process; a failure throws
`std::runtime_error` with the command that failed, so a broken build says what is missing.

The fixture finds the pieces beside the test binary instead of being told where they are: the build
puts `r97_host_tests` in `<build>/` and radiance's own build in `<build>/radiance-build/`, so
`<build>/radiance-build/bin/rad-convert` is the converter and `<build>/radiance-build/radiance_home`
(holding `kernels/` and `architectures/`) is the plugin home. The environment wins over that guess —
`R97_RAD_CONVERT` and `RADIANCE_HOME` — and then the converter falls back to `rad-convert` on
`PATH`, so the host test binary runs outside the build tree too when the plugins are installed.
Building the host tier builds the converter, the plugins (`libref`, `libavx`) and the architecture
plugin, so the guess always finds something; when it does not, the exception names the directory it
looked in.

## Running a prefill

```c++
r97::Prefill p = r97::run_prefill(f, token_ids);          // one prompt
std::vector<r97::Prefill> all = r97::run_prefill(f, prompts);  // a batch
```

`run_prefill` starts an `Engine` on the container with the host kernel libraries (`libavx`, then
`libref` for the rows `libavx` does not cover), speculation off, and submits the prompts as
scoring requests with `max_tokens = 1`: one prefill step each, from position 0, with no decode.
Every position's logits are kept, so

- `Prefill::n_tokens` is the prompt length,
- `Prefill::n_vocab` is the vocabulary of the container (320),
- `Prefill::logits` is row-major, `n_tokens × n_vocab`, the row for position `pos` holding the
  distribution for the token after it,
- `Prefill::at(pos)` and `Prefill::at(pos, token)` read it.

Batches: all prompts are submitted under one hold of the scheduler's queue, so the batching
depends only on the prompts and the flags. A sequence's rows never depend on which others share
its step — prefills are recomputed, not read from a prefix cache (off in this mode) — so a batch
gives each sequence the logits that sequence gives alone, and two runs of the same input are
bit-identical.

The mode is radiance's `core/train_prefill.cpp`, added by
`patches/training/0103-prefill-logits.patch`: `Config::prefill_prompts` names the prompts,
`Config::prefill_logits` receives the rows, and `Engine::serve()` runs the prefill instead of the
server, as the KL mode (`core/engine_kld.cpp`) does. The host test binary must export the plugin
ABI (`ENABLE_EXPORTS`), because the plugins are `dlopen`ed and call back into `rad_core`.

`run_prefill` exists only in a build whose copy of radiance carries that patch, which the header
asks with `__has_include("train_prefill.h")` — the patch's marker header — and reports as
`R97_FIXTURE_PREFILL`. A test file that uses the helper wraps its contents in that macro, so a
serving build compiles it to nothing without a build rule naming it.

## Notes

- GCC 13 cannot compile radiance's architecture plugins: it refuses to take the address of a
  compound literal holding run-time values, `-fpermissive` does not downgrade that error, and GCC
  14 takes it. The configure builds the radiance subproject with `gcc-14`/`g++-14` (or newer) when
  the compiler it was configured with is older, and stops with a message naming the requirement
  when none is installed; the rest of the project keeps the configured compiler.
- A serving build (`-DR97_PATCH_SET=serving`) applies no training patch, so it has no prefill mode
  and `tests/host/fixture_prefill.cpp` compiles to nothing: the mode the helper drives is a training
  hook, and the fixture's prefill exists only in the `all` build.
- A host-only build (`-DR97_WITH_HIP=OFF`) serves on the CPU and prints a startup warning when the
  machine has a card; that is expected, and the fixture runs the same either way.
- The fixture is deterministic but not small in time: each `run_prefill` starts and stops an
  engine (a few hundred milliseconds for this shape). Compare against a reference captured in the
  same test rather than a stored one.
