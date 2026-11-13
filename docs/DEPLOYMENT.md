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
only `patches/serving/` is applied. Of what a deployment adds to the pinned engine, the adapter's
half is:

| Patch | `Concern:` | What it serves |
| --- | --- | --- |
| `0201-lora-set-weights.patch` | put an adapter slot in service from factors already in memory, not only from a directory | an adapter handed over in memory, the trainer's live slot or a serve's |
| `0201-narrow-head-gate.patch` | an int8 attention gate on a head narrower than the 128-column quantiser group | the `w8a8_fwht128` container's gated attention |
| `0210-lora-int8-trunk.patch` | a LoRA adapter on the rotated int8 trunk, through the host and card kernels | the adapter this deployment serves, on `libavx`/`libref` and on `libr4d` |

`0201-lora-set-weights.patch` is in the serving set because radiance's own loader puts an adapter
in service through it, and the trainer's live slot goes through the same code path: the serving
set is what a radiance needs to serve an adapter trained here, and it stands alone. The
trainer's prefill mode is not in it: `run_prefill`, its adapter slot (`0104`) and its adapter
lookup (`0106`) exist for `r97train loss`'s pass and no serving process calls them. What a
deployment answers a prefill with is radiance's own server and KL mode, which stand in the pinned
engine, and section 5 is the measured serve.

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

CI builds this serving tree beside the default one and runs the serving tests against it
(`ci/serving_build.sh`): both trees run the host tests, and a machine with the card runs the `gpu`
and `model` ones in the serving tree too. The default build is what runs `r97train` and writes the
adapter; the serving tree only reads it.

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
ROCR_VISIBLE_DEVICES=0 bin/radiance --model qwen3.8-27b-iq4r-s-v4.rad \
             --radiance-home radiance_home \
             --lora trained=/runs/run-real/checkpoint-000000010/adapter \
             --kernels libr4d,libavx,libref
```

The container matters on this side: the int8 fixture's geometry (`head_dim 32`, `head_k 16`)
has no `libr4d` kernel (`tests/gpu/device_forward.cpp` records why), so the card's serve is the
real container with the adapter a real run trained, not the fixture the host's checks use. A
serving build still has to accept an adapter shaped for that container, and that is what
`tests/gpu/serving_container.cpp` takes on the card. Until a real run has handed one over,
the card test writes a rank-4 `down_proj` adapter for the container's own targetable linears
through `write_adapter` (the writer `tools/train.cpp` uses) and holds the serve to the base
model's own loss: the adapter must be accepted and must move the number.

`--kernels` names the libraries the serve runs on, and on the card that list matters: the trainer's
`loss --device` runs on `libr4d,libavx,libref` (`tools/loss.cpp`), a HIP serving home also holds
`kernels/libr97.so`, the trainer's device plugin, and radiance loads every library under `kernels/`
when `--kernels` does not name them. A card serve that leaves the list out can answer from the
trainer's kernels, and then the served number is not the number it is held to.

The trainer and the server share the card and are never run at once: a run takes the card for the
whole of its training, and the serve is started after it has exited, on the same
`ROCR_VISIBLE_DEVICES`.
Starting a serve beside a training run would put two engines on one allocation and neither number
would be the one the deployment is held to. `tests/gpu/serving_container.cpp` holds the container's
serve to the card and is a model case: it skips when the container is absent and takes the card
when it is present. `tests/model/serving_container.cpp` reads the same container on the host
tier: its arch, its basis and its targetable linears, and the adapter's targets against them.

The server's log also says which backend it took: `device: AQL backend, 1 device(s)` is the card,
`device: host backend` (or `built with HIP but no device is visible`) is not, and a serve that
claims the card has to show the first of them. What runs under it is libr4d's `lora_shrink` and
`lora_expand`, which `0210-lora-int8-trunk.patch` teaches the int8 trunk's step order —
`lora_shrink_scalar` and `lora_shrink_wmma_q` among the rows — and
`tests/gpu/lora_shrink_step_order_r4d.cpp` holds both of those rows on a card.

## 5. What makes a serve a checked one

The first check is exact: the serving build's answer for a corpus with a trained adapter is the
default build's, bit for bit. `tests/host/serving_prefill.cpp` serves the int8 fixture through
`tests/fixtures/serving_adapter` — an adapter `r97train` wrote — with radiance's own KL-scored
prefill (`--kld-record`, `--kld-lora trained`), and compares every recorded row (the next-token
NLL and the row's entropy) to `tests/fixtures/serving_rows.f32`, which the same test wrote in the
default build. Both trees run it: the prefill is the server's and the KL mode is in the pinned
engine, so the test needs neither the observer nor the replay patch nor anything else in
`patches/training/`. It also serves the same corpus with no adapter and requires the loss to
differ, so an adapter a build silently dropped cannot pass on the bytes. That exact check is what
holds the adapter path to the number a training run measured.

The second check is the trainer's own number, at `1e-4` relative (`docs/EVAL.md`). Radiance's KL
mode scores the run's held-out corpus for the
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
`loss --lora` is `4.2615519630392633` and the serving build's record is `4.26155194807`: `3.5e-9`
relative, so what the deployment computes is the trainer's number to the digits it prints. That
measurement is `tests/host/runbook.cpp`'s when it is given the serving tree (`R97_SERVING_BUILD`,
`docs/TESTING.md`): the test serves the adapter, answers one completion, and prints the two losses
it compared, so both of the host's numbers are taken by `ctest` rather than by hand. The card's
measurement is the same one in `docs/RUNBOOK.md` section 7's second block, held to the trainer's
card number (`loss --device`) and, where a host run of the model exists, to the host's, both at
`1e-4`, with the serve log required to show the AQL backend. Card 0 took those numbers on
2026-11-13 for the 27B container: `loss --device 0` printed `1.37629250328` and the serving-only
build's record `1.37629250298`, `2.2e-10` relative over the 1840 held-out positions, on `libr4d`.
The record needs the card's limits named (`--max-num-seqs 8`,
`--max-num-batched-tokens 8192`, `--max-model-len 4096`): the engine's 256-sequence default worst
case asks for `36.70 GiB` of linear and conv state against the card's `16.17 GiB` of cache, and it
stops at startup instead of serving. Both commands, and what the card did not take, are in
`docs/reports/2026-11-13-serving-card.md`.

A serve that was not measured is not a checked deployment: the host's exact served rows and its
loss are measured here, and the card's adapter path is `tests/gpu/serving_container.cpp`'s under
the model label (the base container's prefill, then the same prefill through the adapter, and the
two must differ). `tests/model/serving_container.cpp` checks the real container's serving
handshake on the host: its arch, its basis, and that the adapter's targets are among its
targetable linears. The card's *trained* number is the real run's (`docs/RUNBOOK.md` section 7's
second block, `loss --device` against the served record), and no card has taken it: it needs the
adapter a short real run writes, and `r97train train` gives a run the host kernels (`opts.kernels`
in `tools/train.cpp`) while the host tier cannot declare this container
(`docs/reports/2026-11-12-device-forward.md`), so the card's serve above was held to the trainer's
card number with the adapter the trainer's own writer made for the container it serves (section
4), which is what a run writes. Section 7's blocks say in their output which measurement they
took and which they skipped, and a run without a serving build keeps the trainer's `loss --lora`
number alone.

## 6. Upstreaming

Each patch of the serving set carries `Concern:` and `Upstream: not-submitted`, one concern each,
and is written self-contained so it can go upstream alone. A fresh radiance plus
`patches/serving/` is a server that loads an adapter trained here; `patches/training/` is the
trainer's and never enters a deployment — its prefill mode and adapter lookup included. A serving
patch that will not apply or will not build without one of them is a defect to cut in that patch,
not a reason to move the training patch into this set. `docs/UPSTREAM.md` has the rules for the
patch sets and what upstream has answered.
