# The PyTorch reference

The adjoints elsewhere in this repository are checked against finite differences of radiance's
forward. This is a second opinion that shares no code and no derivation with them: Hugging Face's
own Qwen3.5 implementation, `Qwen3_5ForCausalLM`, differentiated by `torch.autograd`.

PyTorch is a tool, not a dependency. `tools/oracle/make_reference.py` runs once and writes
`tests/fixtures/oracle/`; the build and every test read those files and never import Python. Only
whoever regenerates the files needs the environment. No card is used.

## Regenerating

From a clean checkout, with `build/` configured (`cmake -B build -G Ninja -DR97_WITH_HIP=OFF`):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r tools/oracle/requirements.txt

cmake --build build --target r97_write_fixture
build/r97_write_fixture /tmp/tiny_qwen35_bf16

.venv/bin/python tools/oracle/make_reference.py --checkpoint /tmp/tiny_qwen35_bf16
```

The virtual environment lives under the checkout and is ignored by git. Nothing is installed
system-wide and nothing goes into `/opt/hf`. The pinned versions are `torch==2.15.0+cpu`,
`transformers==5.22.2`, `safetensors==0.8.0` and `numpy==2.5.4`; `tools/oracle/requirements.txt`
says where the CPU wheel comes from. The writer is the fixture's own helper
(`tests/support/fixture.h`), so the script and the tests see the same checkpoint bytes.

On this machine, with the environment and the writer already built, the generator takes about 15
seconds of its own (16.9 s, 21.7 s, 39.5 s, 14.8 s and 14.7 s across the runs that produced the
reference), 17 s wall including the interpreter start. Creating the virtual environment and
installing the pins took 4 m 10 s from a warm pip cache, and building `r97_write_fixture` took
the better part of ten minutes once the radiance submodule had been fetched on a loaded machine.
Two runs from the same environment produce byte-identical files (checked with `sha256sum`).

`--check-only` runs every check and writes nothing. `--out DIR` writes elsewhere.

## The model, and the three things reading it found

The checkpoint is the multimodal release's text tower: `config.json` says
`Qwen3_5ForConditionalGeneration`, the weights are `model.language_model.…`, and there is no vision
configuration. The script builds a `Qwen3_5TextConfig` from the same `config.json` keys and changes
no value: `architectures` and `model_type` are container-level and are dropped, every text key is
passed through. The names are moved from `model.language_model.X` to `model.X`; the values are cast
to f64 and otherwise untouched. `Qwen3_5ForCausalLM` is then constructed under
`torch.set_default_dtype(torch.float64)` and loaded with `strict=True` in effect (the script fails
on both missing and unexpected tensors).

The gated delta net and the causal convolution have a pure-PyTorch path and a fast path taken when
`fla`, `causal_conv1d` or a hub kernel is installed. Before it patches anything, the script asserts
that `fla`, `causal_conv1d` and `kernels` are absent and that the decorated functions still wrap
the pure-PyTorch bodies (`torch.linalg.solve_triangular` in the chunked body, `last_recurrent_state`
in the recurrent one, `F.conv1d` in the convolution). If a fast path were in use the script stops.

`Qwen3_5RMSNormGated` and the rotary embedding cast to f32 internally. For an f64 reference those
casts are overridden. Every override is proved to change the dtype only: the script runs the stock
module and the override side by side in f32, where the stock casts are no-ops, and requires the
largest absolute difference to be exactly zero. The manifest records those zeros under
`dtype_overrides`.

## Every f32 cast in the text path, and what happens to it

Reading `modeling_qwen3_5.py` found more than the two named casts. All of them are in the text
path; the vision file's casts are left alone because the fixture has no vision tower.

| Where | What it does | Handling |
| --- | --- | --- |
| `Qwen3_5TextRotaryEmbedding.forward` | position ids and `inv_freq` to f32, cos/sin back to x's dtype | overridden; the same body keeps x's dtype |
| `Qwen3_5TextRotaryEmbedding.__init__` / `compute_default_rope_parameters` | `inv_freq` built with `dtype=torch.float` | the model's `inv_freq` is recomputed in f64 from the same theta in `prepare_f64_constants` |
| `Qwen3_5RMSNormGated.forward` | hidden state and gate to f32, result to input dtype | overridden |
| `Qwen3_5RMSNorm.forward` | norms and gain in f32, result to input dtype | overridden |
| `eager_attention_forward` | softmax in f32, result to query dtype | overridden; this is the attention block's softmax |
| `torch_chunk_gated_delta_rule` | query, key, value, beta and decay to f32, result to input dtype | overridden; without this the chunked form cannot agree with the recurrent one to 1e-10 |
| `torch_recurrent_gated_delta_rule` | the same casts | overridden |
| `Qwen3_5GatedDeltaNet.forward` | `A_log.float()` and `a.float()` before exp/softplus for the decay `g` | overridden |

The script does not rewrite the module file. Each override is the stock source with the dtype
expressions replaced, executed in the modeling module's own namespace; a replacement that does not
match exactly once stops the script, so a transformers update cannot silently produce a different
reference. The `@torch.no_grad()` and `@dynamic_rope_update` decorators on the rotary forward stay
on the override.

The chunked and the recurrent delta rule, both in f64, agree to 8.81e-16 relative on the reference's
own inputs, and to 1.63e-11 in the worst element. The check does not invent its inputs: it runs the
reference batch through the trunk once (without the adapter, and drawing nothing from the
generator, so the capture cannot move a reference tensor) and records the query, key, value, g and
beta that every gated delta net layer hands `torch_chunk_gated_delta_rule`, then runs the recurrent
form on those same tensors. The batch is padded to 140 tokens, so the chunked form really does
stitch three 64-token chunks together; a shorter input would pass the same check and say nothing
about the multi-chunk path, which is the reason the 140-token sequence is in the reference. The
check fails if the captured sequence would cover two chunks or fewer. The relative number is over
the whole tensor, `||chunk - recurrent|| / ||recurrent||`, for the same reason the logits
comparison is not elementwise: one output near zero would otherwise decide it. The elementwise
maximum is held to 1e-10 as well and recorded beside it. The manifest records all of this under
`delta_rule`: the shape, the chunk size, the number of chunks, the layers checked, and which gated
delta net had the worst difference (`gdn_layer_of_worst`, 0-based, 2 of 3).
`torch.autograd.gradcheck` passes in f64 on all three block cases (`eps=1e-6`, `atol=1e-6`,
`rtol=1e-5`, one sequence each).

## What the reference holds

`reference.safetensors` is f64 throughout; token ids and masks are i64. The manifest lists every
tensor with its shape and dtype.

- `seq{b}/input_ids`, `seq{b}/score` — a batch of three sequences of lengths 140, 73 and 31. The
  longest covers three chunks of the delta rule's scan (`chunk_size = 64`), so it is longer than
  two chunks. Scores are 1 for every token after the first.
- `seq{b}/logits_without_adapter` — f64 logits for every valid position, shape `[length, vocab]`.
- `with_adapter/loss` — the scalar loss of the padded batch with the adapter, and
  `with_adapter/lengths`.
- `seq{b}/loss_with_adapter` — the same loss restricted to sequence `b`, without the padding. The
  batch loss is their token-count-weighted mean; the script checks that identity before writing.
- `seq{b}/residual.{i}` and `seq{b}/residual_grad.{i}` — for `i = 0 … 4`, the residual stream at
  every layer boundary and the cotangent of the loss there. Boundary 0 is the embedding output,
  boundary `i` the output of layer `i - 1`, and boundary 4 the last decoder layer's output before
  the final norm. They are stored per sequence, without the batch's padding.
- `lora_grad/<module>.lora_A`, `lora_grad/<module>.lora_B` — the gradient of every LoRA `A` and
  `B`, for the ten target modules `docs/ADAPTER.md` lists, in every layer where the module exists.
- `gdn_mixer.{single,batch}.{input,output,cotangent,input_cotangent,lengths,mask}` — the module
  `linear_attn` from the normed hidden state to its output, before the residual add. `input` is
  `input_layernorm`'s output for a random residual of the same shape, computed and then detached:
  the norm is not part of this case, so `input_cotangent` is the mixer's own cotangent rather than
  the composite's through the norm. The other two cases do include their norm, because the issue
  defines them from residual to residual.
- `attention_sublayer.…` — residual in, residual after the add; input norm, attention and add are
  inside.
- `mlp_sublayer.…` — residual in, residual after the add; post-attention norm, MLP and add are
  inside.

All three are computed on the trunk as the checkpoint holds it, before the LoRA adapter is wrapped
in: no `A` and no `B` is inside a stored case, so a test differentiates the bf16 trunk's block
backwards with the stored cotangents as they are. The layer each case comes from is in the
manifest's `cases.blocks` — `gdn_mixer` and `mlp_sublayer` use layer 0, `attention_sublayer` the
only full-attention layer, 3 — together with `with_adapter: false`.

The single cases are one sequence of length 23; the batch cases are three sequences of lengths
17, 11 and 6, padded to 17 with an attention mask. Each case has a fixed random output cotangent and
the input cotangent autograd returns for it, so a test can differentiate a block on its own, without
a whole model. The masks travel with the case; the padding is at the end, so it cannot change an
earlier position.

The adapter is also written as a PEFT directory, `adapter/`, with `adapter_config.json` and
`adapter_model.safetensors`. `A` and `B` are both non-zero (PEFT's default `B` is zero, which would
leave every gradient zero), random from the seed, rank 4, alpha 8, so the scale is `alpha / r = 2.0`
as `docs/ADAPTER.md` describes. The LoRA formula is plain `W x + (alpha / r) B A x`, applied to the
same linears `docs/ADAPTER.md` lists for the bf16 trunk; there is no Hadamard basis and no
`radiance_input_basis` key, because this is the untransformed bf16 trunk. The PEFT file holds `A`
and `B` as f32, not the reference's f64: radiance's adapter loader accepts BF16, F16 and F32 only,
and f32 loses nothing that its bf16 trunk can use. `base_model_name_or_path` is the stable label
`r97-tiny-qwen35-bf16`, not the temporary checkpoint path, so the adapter's bytes do not depend on
where the writer put the checkpoint.

## The loss

`include/r97/batch.h`'s definition: the mean, over tokens whose `score` is 1, of the negative
log-probability of token `i` under the logits at position `i - 1`. The script computes
`log_softmax(logits[:, :-1])`, gathers the next token and divides the weighted sum by the number of
scored tokens; the first token of each sequence is never scored. Each sequence's loss is the same
mean over its own scored tokens, and the batch loss is their weighted mean.

## The host tests

- `tests/host/oracle_fixture_sha.cpp` recomputes the SHA-256 of today's fixture checkpoint and
  compares it with `fixture.sha256` in the manifest. A changed fixture fails here, by name, before
  any gradient is compared.
- `tests/host/oracle_prefill.cpp` runs `run_prefill` on the three sequences and compares the
  logits without the adapter in relative L2 norm, per sequence, allowing 5% (`docs/ORACLE.md` below
  says why the comparison is not elementwise).

These two tests are the contract with the reference. If the logits disagree, that is a finding
about the fixture, radiance or transformers: stop and report it, and write the cause into this file
before changing the reference. Do not tune the reference to radiance.

### What the logits comparison found

The first version of the test allowed 2% on every logit, `max(1, |reference|)` in the denominator.
It failed, and the finding is real: radiance's bf16 logits and transformers' f64 logits differ by
more than a bf16 rounding in a few places.

| sequence | relative L2 | worst absolute difference | where |
| --- | --- | --- | --- |
| 0 (140 tokens) | 1.94% | 0.595 | position 74, token 66 |
| 1 (73 tokens) | 2.40% | 0.783 | position 23, token 233 |
| 2 (31 tokens) | 1.17% | 0.166 | position 26, token 188 |

The batch was regenerated once, when the delta rule agreement check moved onto the reference's own
inputs and so had to draw the sequences first; that changed the generator's draw order and with it
the token ids, and the table is from the checked-in batch. The earlier batch's ids gave the same
picture (3.06%, 1.23%, 1.89% in relative L2); the emulation figures below were measured on it and
have not been re-run, because they are about the shape of the disagreement, not about this batch.

The cause is the precision of the two bf16 delta nets, not the checkpoint or the reference:

- transformers' stock bf16 model does the delta rule's arithmetic in f32 (`torch.float32` casts at
  entry, and f32 for the within-chunk solve), then casts the result back. The f64 reference removes
  those casts, so it is the same mathematics in f64.
- radiance keeps the recurrent state, decay `g` and `beta` in f32, but its chunk kernel
  `gdn_kkt_solve` writes the within-chunk inverse as **bf16** (`arch/qwen35_bf16/qwen35_bf16.cpp`,
  the `b_kkt` declaration says bf16 deliberately, and `gdn_chunk_scan` reads it as bf16). The
  activations are bf16 throughout.
- The elementwise spikes are therefore expected. A pure-bf16 emulation of transformers' delta rule
  on this fixture reaches 0.67 absolute error and can flip the argmax at position 66; stock
  transformers bf16, which is f32 inside the delta rule, reaches 0.31–0.43 absolute error and
  already fails the old 2% elementwise check on 28% of the logits. None of the three models is
  wrong; they are three precisions.

The test therefore compares the whole logits matrix in relative L2 norm: `||prefill - reference||
/ ||reference|| <= 0.05` per sequence. That is bf16 tolerance for this model, and a wrong weight or
a wrong layer moves whole rows far above it. The test still prints the worst element and its
position, so a regression that changes one place is visible in the log even when it passes. The
elementwise disagreement, including the argmax that can flip at a spike, is the finding about
radiance's bf16 GDN; changing it would be a change to radiance, outside this issue.

## What the block cases are computed on

The first version computed the three block cases after `add_lora()`, so every stored output and
cotangent was the adapter-wrapped `W x + (alpha / r) B A x` throughout, not the block the issue
defines and not what this file said. A test differentiating the trunk's block backwards could not
use them without applying the rank-4 adapter itself. Measured on the first checked-in file:
`mlp_sublayer.single.output` differed from the plain layer 0 block
(`inp + layers[0].mlp(post_attention_layernorm(inp))`) by 40.8, and matched the adapter-wrapped
layer 1 block to 5.8e-6.

`block_cases()` now runs before the adapter is wrapped in, and three things keep it that way:

- `assert_no_lora()` stops the script if any LoRA target is wrapped when a block case is computed.
- `check_blocks_exclude_adapter()` recomputes all six stored cases on the adapter-wrapped model and
  requires each output to differ; the smallest of those differences is in the manifest under
  `block_adapter_control`. It is a positive control: numbers taken with the adapter in would come
  back unchanged and the script would stop instead of checking them in under the block's name.
- The manifest's `cases.blocks` names the layer each case comes from and says `with_adapter: false`.
  This file used to claim layer 0 for `mlp_sublayer` while the script used layer 1, and neither the
  manifest nor the tensors recorded it. `block_layers()` is now the one place the script, the
  manifest and this file take the layers from.

Because the cases' random inputs now come from the seeded generator before the adapter's `A` and
`B`, the regenerated files hold a different adapter, loss, residual cotangents and adapter
gradients than the first version did. The token ids and the logits without the adapter are
unchanged, so the table above still applies.

## What this reference does not cover

The int8 trunk. transformers has no `w8a8_fwht128` model, and the straight-through gradient is this
project's own definition. That stays with the descent check of the int8 backward.

## Decisions taken without a person to ask

- Rank 4, alpha 8 rather than PEFT's default rank 16, alpha 32: it keeps `reference.safetensors`,
  the adapter and the manifest together under the issue's 5 MB. The scale, `alpha / r = 2.0`, is
  the default contract's.
- Sequence lengths 140, 73 and 31: 140 is more than two 64-token chunks of the delta rule's scan,
  and the total keeps the file under 5 MB.
- Residuals are stored per sequence without padding, so "per sequence" is literal and the file does
  not pay for padding.
- The block cases use layer 0 for `gdn_mixer` and `mlp_sublayer`, and the only full-attention
  layer (layer 3) for `attention_sublayer`. `block_layers()` computes them once and the manifest
  records them, so this file and the script cannot drift apart again.
- `gdn_mixer` starts at the normed hidden state, as the issue defines it, and the other two blocks
  start at the residual, as it defines them. The first version of the script started `gdn_mixer` at
  the residual too, which made its `input_cotangent` the composite's through `input_layernorm` and
  made the three `.input` tensors byte-identical; the case was wrong, not the model.
