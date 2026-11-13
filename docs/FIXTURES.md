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
Building the host tier builds the converter, the plugins (`libref`, `libavx`, `libquant`) and the
architecture plugins (`qwen35_bf16`, `qwen35_w8a8`), so the guess always finds something; when it
does not, the exception names the directory it looked in.

```c++
#include "support/fixture_w8a8.h"

const r97::Fixture& q = r97::w8a8_fixture();     // the same checkpoint, w8a8_fwht128
const r97::Fixture& s = r97::shuffled_fixture();  // the control: rows rotated, converted bf16
```

## The tiny int8 Qwen3.5

`r97::w8a8_fixture()` is the same checkpoint, converted a second time with
`tests/fixtures/w8a8_fwht128.recipe`: radiance's
`data/recipes/q38-27b-w8a8-df2.recipe` at this checkpoint's shapes, with one line left out. The
container lands beside the bf16 one as `tiny_qwen35_w8a8.rad`, states the quantisation
`w8a8_fwht128`, and the `qwen35_w8a8` plugin serves it on the same host kernel libraries (`libavx`,
then `libref`).

The recipe, as `tests/fixtures/w8a8_fwht128.recipe` holds it:

```
blk.*.attn_qg.weight                 rtn  codes=i8 group=128 scale=bf16 transform=fwht128 clamp=sym rule=search
blk.*.attn_k.weight                  rtn  codes=i8 group=128 scale=bf16 transform=fwht128 clamp=sym rule=search
blk.*.attn_v.weight                  rtn  codes=i8 group=128 scale=bf16 transform=fwht128 clamp=sym rule=search
blk.*.attn_output.weight             rtn  codes=i8 group=128 scale=bf16 transform=fwht128 clamp=sym rule=search
blk.*.ssm_inz.weight                 rtn  codes=i8 group=128 scale=bf16 transform=fwht128 clamp=sym rule=search
blk.*.ffn_gate_up.weight             rtn  codes=i8 group=128 scale=bf16 transform=fwht128 clamp=sym rule=search
blk.*.ffn_down.weight                rtn  codes=i8 group=128 scale=bf16 transform=fwht128 clamp=sym rule=search

output.weight                        rtn  codes=i8 group=128 scale=bf16 clamp=sym rule=search
```

So every trunk linear is an i8 code, a bf16 scale per 128 of K, the 128-wide Hadamard and
symmetric clipping, with the scale's rounding searched per group; the `lm_head` is the reference's
own unrotated int8 line. One of the reference recipe's lines is left out, because this
checkpoint's shape cannot take it:

- `blk.*.ssm_out.weight`: the delta-net value width of this checkpoint is 64 (4 value heads of
  16), and the served `i8a8h` form wants a scale per 128 of K; 64 does not divide, so it stays
  bf16. It is the only trunk linear that does.

The reference's dflash drafter lines have no counterpart here: this checkpoint has no drafter.

The conversion, as the helper runs it:

```
rad-convert <checkpoint> -o <container> --kernels libavx,libref --home <radiance_home> \
    --recipe tests/fixtures/w8a8_fwht128.recipe \
    --set quantization_config.quant_method=w8a8 --set quantization_config.fmt=fwht128
```

The two `--set`s are what selects the plugin: `qwen35_w8a8`'s `check_format` refuses a container
that does not state `w8a8` / `fwht128`, as the header of every int8 conversion states them.

### What the host issues for it

The plugin asks `rad_op_declared` for each fused producer and falls back to the unfused chain
when the answer is no. On the host every fused producer that would write int8 codes is absent
(`gate_had_quant_i8g`, `gate_quant_fp8`, `attn_paged_gate_quant`, `rmsnorm_had_quant_i8g`,
`gated_had_quant_i8g`, `qk_norm_rope_gate`), so the chain below is the declared graph for this
fixture, in declaration order, deduplicated over the four layers; counts are for the whole 96-token
prefill graph. There is no sampling tail: the request is prefill-only (`run_prefill` below), so no
sampler op is issued.

| op | n | what it does here |
| --- | --- | --- |
| `embed_lookup` | 1 | the token ids' rows of `embed_tokens` |
| `rope_table` | 1 | the cos/sin table, `partial_rotary_factor` 0.5 |
| `rmsnorm_quant_fp8` | 8 | the pre-norm of each linear input: 2 per delta-net layer (in projection, ffn) and 2 in the attention layer |
| `had_quant_act_i8g` | 15 | the int8 code and scale of each int8 linear's input, rotated 128 wide: one per `gemm_nt_q` |
| `gemm_nt_q` | 15 | an `i8a8h` GEMM: `ssm_inz` (N=192 K=128), `ffn_gate_up` (N=512 K=128) and `ffn_down` (N=128 K=256) in each delta-net layer, `attn_qg`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate_up` and `ffn_down` in the attention layer |
| `gemm_nt` | 6 | the bf16 GEMMs: 3 delta-net small projections (N=8 K=128) and 3 `ssm_out` (N=128 K=64) |
| `quant_act_fp8` | 4 | the fp8 activation the bf16 `ssm_out` GEMMs read (3), and the attention gate's own quantiser (1) |
| `gated_quant_fp8` | 4 | the silu-gated activation before each `ffn_down`: 3 delta-net layers and the attention layer |
| `gdn_conv_prep`, `gdn_conv_update`, `gdn_conv_recurrent_update` | 3 each | the short convolution over the delta-net's q/k/v |
| `gdn_kkt_solve`, `gdn_chunk_scan`, `gdn_gated_rmsnorm`, `gdn_recurrent_update` | 3 each | the delta-rule chunk and recurrent passes |
| `rmsnorm` | 3 | q norm, k norm and the final norm |
| `rope` | 2 | q and k |
| `kv_store` | 1 | the attention layer's keys and values |
| `attn_paged` | 1 | the one full-attention layer's prefill |
| `sigmoid`, `mul` | 1 each | the attention output gate, unfused |
| `add` | 1 | the attention layer's residual |
| `gather_rows` | 1 | the last position's row for the head |
| `logits_gemm` | 1 | the int8 `lm_head`: i8 codes, a bf16 scale per 128 of K, unrotated |
| `sample_*` | 14 | the sampling chain every engine declares; a prefill returns the logits, so a caller never reads a sample |

No kernel plugin of this repository was needed: each name in the table has a row in `libref`
(the `i8a8h` operands of `gemm_nt_q` and the int8 head's operands included), so `libr97ref` does
not exist.

### How this graph differs from the bf16 one

The two containers are the same checkpoint, but not the same graph, and a reader comparing them
should know where the difference is:

- **The delta net's projections are fused differently.** This container has a `ssm_inz` record
  writing the input projection and the gate as one packed 192-wide buffer, and the AB projection
  reaches `gdn_conv_prep` as an operand of its own; the bf16 graph hands that op a single 136-wide
  `gdn_in` and reads the AB operand as 4-wide slices of the same buffer. From `gdn_chunk_scan` on
  the two graphs record different operands.
- **The head's int8 weight is a joined buffer.** `output.weight` reaches the tape as a
  `RadPlanesHeader`, the codes and the scales, and the ABI's byte count covers the codes only; a
  reader that wants the dequantised weight has to read the planes (see `docs/STEPS-HOST.md`).
- Everything else in the table has a bf16 counterpart of the same shape and the same operand
  order, so per-boundary values and cotangents are comparable between the two.

These are layout differences, not differences of the function being differentiated: the tape
gathers each operand by its own view, the walk sums a buffer's cotangent over every reader, and
the fused site's pooled gradient is split back into the two modules by the row and rank windows of
its slices (`split_adapter_grads`). The adapter's parameter gradients of the two containers are
therefore comparable element for element, and `tests/host/step_host_int8.cpp` compares them; what
it measures is recorded in `docs/STEPS-HOST.md`.

### The attention gate on a narrow head

The fused gate producers write one scale per (token, head) row. That is the out projection's one
scale per 128 columns only when a head is whole 128-column blocks — the condition the
`q_dim % 128 == 0` check in the block already relies on for the unfused pair. This checkpoint's
head is 32 wide (4 heads of 32), so `head_dim % 128 != 0` but `q_dim` is 128: the block used to
declare neither the fused gate nor the unfused pair and then issued a null op. The unfused pair
is declared whenever `q_dim` is a whole number of 128-column groups, which is the condition its
quantiser actually needs; the fused producers keep the head-wide condition. That is
`patches/serving/0201-narrow-head-gate.patch`: a serving concern, self-contained, and it changes
no full-width model.

### The KL against the bf16 fixture

`fixture_w8a8_quantises_the_bf16_fixture` prefills the same 96-token prompt through the bf16
fixture, the int8 fixture, and a control, and reports the per-position mean and worst
KL(bf16 || other) of the next-token distributions:

- int8 fixture: mean 1.9e-2, worst 7.2e-1 (the worst is one position where the two models
disagree about a near-tie);
- control: mean 3.7e+0, worst 7.5e+0 — 190 times the int8 mean.

The control is `r97::shuffled_fixture()`: the same checkpoint with every matrix's rows rotated by
half, so the values are all there and each is in the wrong row, converted bf16. Vectors (norms,
decays) keep their values, so its logits are finite. The test holds the int8 mean below 0.05 and
the control above twenty times it.

## Running a prefill

```c++
r97::Prefill p = r97::run_prefill(f, token_ids);          // one prompt
std::vector<r97::Prefill> all = r97::run_prefill(f, prompts);  // a batch
```

`run_prefill` starts an `Engine` on the container with the host kernel libraries (`libavx`, then
`libref` for the rows `libavx` does not cover), speculation off, and submits the prompts as
scoring requests that are prefill-only (`Request::prefill_only`, patch
`patches/serving/0204-prefill-not-sampled.patch`): one prefill step each, from position 0, no
decode, and no sampler -- the step draws no token, so it issues no sampler op, and the request
finishes once its prompt is placed. Every position's logits are kept, so

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
`patches/serving/0202-prefill-logits.patch`: `Config::prefill_prompts` names the prompts,
`Config::prefill_logits` receives the rows, and `Engine::serve()` runs the prefill instead of the
server, as the KL mode (`core/engine_kld.cpp`) does. The host test binary must export the plugin
ABI (`ENABLE_EXPORTS`), because the plugins are `dlopen`ed and call back into `rad_core`.

`run_prefill` exists only in a build whose copy of radiance carries that patch, which the header
asks with `__has_include("train_prefill.h")` — the patch's marker header — and reports as
`R97_FIXTURE_PREFILL`. A test file that uses the helper wraps its contents in that macro. The
prefill is a serving patch, because a serving-only build has to answer a prefill with an adapter to
be checked; a build without the serving set compiles the helper to nothing.

## Notes

- What the fixture needs from the build is in `CMakeLists.txt`, the one shared file this topic
  touches, in four places, each of which the build skeleton's radiance block does not do for
  `rad_core` alone: the radiance subproject's compiler (`gcc-14`, below), the subproject's
  `BUILD_COMMAND` (it builds `libref`, `libavx`, `libquant`, `rad-convert`, `qwen35_bf16` and
  `qwen35_w8a8` beside `rad_core`, and the host tier depends on them), the vendor archives `rad_core` needs once the
  engine's chat and tokenizer code is linked in, and `r97_host_tests`'s `ENABLE_EXPORTS` plus the
  radiance copy's `tests/` directory on its include path (for `write_qwen35`). A host test on its
  own needs no build rule: the tiers are collected by directory glob.
- GCC 13 cannot compile radiance's architecture plugins: it refuses to take the address of a
  compound literal holding run-time values, `-fpermissive` does not downgrade that error, and GCC
  14 takes it. The configure builds the radiance subproject with `gcc-14`/`g++-14` (or newer) when
  the compiler it was configured with is older, and stops with a message naming the requirement
  when none is installed; the rest of the project keeps the configured compiler.
- A serving build (`-DR97_PATCH_SET=serving`) applies no training patch, so it has no issue observer
  and no replay hook; the prefill is serving's (`0202`..`0205`), so `tests/host/fixture_prefill.cpp`
  and the serving prefill parity test run in either build.
- A host-only build (`-DR97_WITH_HIP=OFF`) serves on the CPU and prints a startup warning when the
  machine has a card; that is expected, and the fixture runs the same either way.
- The fixture is deterministic but not small in time: each `run_prefill` starts and stops an
  engine (a few hundred milliseconds for this shape). A test that compares two prefills compares
  them in the same process; `tests/host/serving_prefill.cpp` is the one that compares across builds
  and so reads the reference logits in `tests/fixtures/` (see `docs/DEPLOYMENT.md`).
