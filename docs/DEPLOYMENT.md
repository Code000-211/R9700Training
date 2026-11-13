# Deployment: serving a trained adapter

A deployment here is a radiance that serves the adapter `r97train train` wrote, from a build that
carries `patches/serving/` and nothing else, on the host kernels or on a card's. It is not the
trainer: no `r97train`, no checkpoint, no optimiser state, and no patch that only training needs.
`docs/RUNBOOK.md` section 0 builds it step by step and section 7 serves with it; this document says
what such a build is, what the adapter gives it, and what makes a serve a checked one.

## 1. What a deployment carries

A deployment is the server, its plugin home, the adapter and a model container:

| Item | Made by | What it is |
| --- | --- | --- |
| `bin/radiance` | `cmake --build build-serving/radiance-build --target radiance` | the server, built from the patched copy of the engine |
| `radiance_home/` | the `radiance_core` step | `kernels/libref.so`, `kernels/libavx.so`, `kernels/libr4d.so` (a HIP build only), `quantizers/libquant.so`, `architectures/qwen35_bf16.so` and `qwen35_w8a8.so`, and the engine's own data beside them |
| the adapter directory | `r97train train` | `adapter_model.safetensors` and `adapter_config.json` |
| the container `.rad` | `rad-convert` on the checkpoint | the trunk the adapter rides on, `docs/FIXTURES.md` for the fixture's |

What is not in a deployment: `r97train` (a serving build builds no tool but `rad-convert`, and no
patch of `patches/training/` is applied). One thing such a build does make and a deployment leaves
out is `kernels/libr97.so`, the trainer's own device plugin: the build makes it in any HIP tree,
and section 4 is where a serve says not to load it.
`tests/cmake/patch_sets_test.cmake` holds the mechanism to the set: with `-DR97_PATCH_SET=serving`
only `patches/serving/` is applied, and the three patches that are the whole of what a deployment
adds to the pinned engine are:

| Patch | `Concern:` | What it serves |
| --- | --- | --- |
| `0201-lora-set-weights.patch` | put an adapter slot in service from factors already in memory, not only from a directory | the trainer's live adapter during training |
| `0201-narrow-head-gate.patch` | an int8 attention gate on a head narrower than the 128-column quantiser group | the `w8a8_fwht128` container's gated attention |
| `0210-lora-int8-trunk.patch` | a LoRA adapter on the rotated int8 trunk, through the host and card kernels | the adapter this deployment serves, on `libavx`/`libref` and on `libr4d` |

`0201-lora-set-weights.patch` is the trainer's half of the same hook, and it stays in the serving
set because radiance's own loader and the trainer's live slot go through one code path: the
serving set is what a radiance needs to serve an adapter trained here, and it stands alone.

## 2. The build

```sh
cmake -B build-serving -G Ninja -DR97_PATCH_SET=serving                    # HIP: the card
cmake -B build-serving-host -G Ninja -DR97_PATCH_SET=serving -DR97_WITH_HIP=OFF   # the host only
cmake --build build-serving --target radiance_core
cmake --build build-serving/radiance-build --target radiance
```

The patch set is applied to a copy of `third_party/radiance` under the build directory, never to
the submodule's working tree (`cmake/ApplyPatches.cmake`), so a deployment build and a training
build of different sets can live side by side. The parent build's own configure only makes
`build-serving/radiance-build`; the `radiance_core` step configures it and builds the engine
libraries and the plugins, and the server is the third step, from that sub-build's cache. The
first two flavours differ only in `R97_WITH_HIP` (which sets radiance's `RAD_WITH_HIP`): a HIP build
produces `libr4d.so`, which is the card's kernels, and a `-DR97_WITH_HIP=OFF` build does not.

## 3. The adapter it serves

The adapter directory is the one `r97train train --out <run>` leaves at
`<run>/checkpoint-<n>/adapter`: `adapter_config.json` — PEFT's `base_model_name_or_path`, `r`,
`lora_alpha`, `target_modules`, `bias: "none"` and `use_rslora: false` — and
`adapter_model.safetensors` with the factors under PEFT's names, each tensor's `dtype` in the
safetensors header. `r97train loss --lora` reads back only what the trainer writes: it refuses a
bias adapter, rsLoRA and a per-module `rank_pattern` or `alpha_pattern`, because its scale is
`alpha/r` and it has one rank per site (`src/adapter_io.cpp`). A deployment has no `r97train` in it
and reads more than the trainer wrote: radiance's loader honours what a PEFT config asks for —
`use_rslora: true` scales each site `alpha/sqrt(r)`, and `rank_pattern`/`alpha_pattern` give a
module its own rank and its own scale by PEFT's first-matching-key rule, up to a per-module rank of
`--max-lora-rank` — and refuses only what it could not serve: a bias adapter, DoRA,
`modules_to_save`, `fan_in_fan_out`, `target_parameters`, `trainable_token_indices`,
`layer_replication`, `use_qalora`, `alora_invocation_tokens`, a per-module rank above
`--max-lora-rank` (16 by default), a pickle `adapter_model.bin`, and a tensor that is neither
`lora_A` nor `lora_B` (`core/lora/lora.cpp`, `docs/ADAPTER.md` section 4). An adapter trained
elsewhere therefore serves here even though the trainer could not read it back, at the scales its
config asks for. On a transformed trunk — the `w8a8_fwht128` container's rotated int8 trunk is one
— the config also carries `radiance_input_basis`, the basis the factors are written in; radiance
refuses that key on a bf16 trunk, and without it refuses an adapter on a transformed trunk
(`docs/ADAPTER.md` section 4). A deployment serves the directory by name:

```sh
bin/radiance --model tiny_qwen35_w8a8.rad \
             --radiance-home radiance_home \
             --lora trained=/runs/runbook/checkpoint-000000010/adapter
```

`--radiance-home` is the `radiance_home/` from section 1: without it radiance looks for its
plugins beside the binary and finds none of this build's. The adapter is named, not merged: the
same directory serves beside an unmodified trunk, and two names (`trained=`, and any second
adapter the server is given) are how radiance's requests select one.

## 4. Serving on the card

The same commands, with the HIP build's `radiance_home`, and the card named the way the trainer
names it. The server has no `--device` of its own; `ROCR_VISIBLE_DEVICES` is what the runtime
reads, before anything opens the engine:

```sh
ROCR_VISIBLE_DEVICES=0 bin/radiance --model tiny_qwen35_w8a8.rad \
             --radiance-home radiance_home \
             --lora trained=/runs/runbook/checkpoint-000000010/adapter \
             --kernels libr4d,libavx,libref
```

`--kernels` names the libraries the serve runs on, and on the card that list matters: the trainer's
`loss --device` runs on `libr4d,libavx,libref` (`tools/loss.cpp`), a HIP serving home also holds
`kernels/libr97.so`, the trainer's device plugin, and radiance loads every library under `kernels/`
when `--kernels` does not name them. A card serve that leaves the list out can answer from the
trainer's kernels, and then the served number is not the number it is held to.

The server's log also says which backend it took: `device: AQL backend, 1 device(s)` is the card,
`device: host backend` (or `built with HIP but no device is visible`) is not, and a serve that
claims the card has to show the first of them. What runs under it is libr4d's `lora_shrink` and
`lora_expand`, which `0210-lora-int8-trunk.patch` teaches the int8 trunk's step order —
`lora_shrink_scalar` and `lora_shrink_wmma_q` among the rows — and
`tests/gpu/lora_shrink_step_order_r4d.cpp` holds both of those rows on a card.

## 5. What makes a serve a checked one

Serving is checked the way every number in this repository is: against the trainer's own, at
`1e-4` relative (`docs/EVAL.md`). Radiance's KL mode scores the run's held-out corpus for the
adapter it was given and records the reference's next-token NLL per position; `tools/kld_loss.py`
turns that record into the same loss `r97train loss` prints.

```sh
radiance --model tiny_qwen35_w8a8.rad --radiance-home radiance_home \
         --lora trained=/runs/runbook/checkpoint-000000010/adapter --kld-lora trained \
         --kld-corpus /runs/runbook-resumed/kld/heldout.jsonl \
         --kld-record /runs/runbook-resumed/kld/served
python3 tools/kld_loss.py /runs/runbook-resumed/kld/served
```

On the fixture, with the runbook's ten-step run and a host-only serving build, the trainer's
`loss --lora` is `4.66220409013` and the serving build's record is `4.66220409289`: `6e-10`
relative, so what the deployment computes is the trainer's number to the digits it prints. That
measurement is `tests/host/runbook.cpp`'s when it is given the serving tree (`R97_SERVING_BUILD`,
`docs/TESTING.md`): the test serves the adapter, answers one completion, and prints the two losses
it compared, so both of the host's numbers are taken by `ctest` rather than by hand. The card's
measurement is the same one in `docs/RUNBOOK.md` section 7's second block, held to the trainer's
card number (`loss --device`) and to the host's, both at `1e-4`, with the serve log required to show
the AQL backend; those numbers belong to the card, so the script is where they are, not this
document, and no card has taken them yet.

A serve that was not measured is not a checked deployment: the host's is measured, the card's is
not, so the deployment is checked on the host alone. Section 7's blocks say in their output which
measurement they took and which they skipped, and a run without a serving build keeps the
trainer's `loss --lora` number alone.

## 6. Upstreaming

Each of the three patches carries `Concern:` and `Upstream: not-submitted`, one concern each, and
is written self-contained so it can go upstream alone. A fresh radiance plus `patches/serving/` is
a server that loads an adapter trained here; `patches/training/` is the trainer's and never enters
a deployment. `docs/UPSTREAM.md` has the rules for the patch sets and what upstream has answered.
