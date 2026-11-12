# Plan: a purpose-built LLM training engine for the R9700

Repository: `Code000-211/R9700Training`. This plan is the outcome of the grilling session that cut
the work into 59 issues, and of the additions of 2026-11-10, which added nine (59 to 67) and amended
nineteen of the first 59 (0, 1, 3, 7, 8, 11, 12, 15, 17, 18, 33 to 37, 44, 53, 54, 56), and of
one issue (68), added when the plan was merged with the radtrain amendment: the milestone-2 base
container. The numbers below are positions in that planning list, not issue-tracker numbers; the
titles identify the issues.

## 1. Goal

Build `r97train`, a training engine for the AMD Radeon AI PRO R9700 (gfx1201, 32 GB), for one or two
cards. The long-term scope is every kind of training. The first milestone is a LoRA fine-tune of a
frozen Qwen3.8-27B with plain assistant-turn cross-entropy, whose adapter radiance then serves. The
second milestone is the ALoDLM objective on the same base.

**The first milestone is reached when:**

- the held-out loss falls over a run on both cards, and
- a radiance built with the serving patch set only, serving the trained adapter, reproduces the
  trainer's held-out loss to 1e-3 relative.

**The second milestone** is separate and later. It starts when the first is closed, is not folded
into it and does not replace it. It is a masked denoising rollout with an adaptive-depth exit gate:
mask, rollout, objective and gate (issues 62 to 65). Its base is decided when the first milestone
closes; the default is the 0.8B bf16 base first (the same architecture family as the 27B, loop range
`[8,16)`, no quantised backward needed), then the 27B. Its report adds the depth distribution, the
gate's hazard per depth, and the held-out loss on the `gen.u8` positions only (a mean over all rows
is about 165 times larger and means nothing for this objective). Issue 68 picks that base container
and fetches it. Its run and its report have no issue yet; they are cut when the base is decided.

**Not planned yet:** full fine-tuning, pre-training, distillation, int8 gradients, a PyTorch speed
baseline.
No speed target is set; the issues ask for measurements next to the forward's time.

## 2. Settled decisions

| Topic | Decision |
|---|---|
| Relation to radiance | Own repository and binary. It reuses radiance's ABI, `.rad` format, forward kernels (`libr4d`) and host kernels (`libref`, `libavx`). |
| radiance version | Submodule `third_party/radiance` pinned to published `origin/main` (`451b136874084650992668bed9e8662deb58de88`). Changes are numbered patches in `patches/serving/` and `patches/training/`, applied by the build and upstreamed later. |
| Forward graph | The unmodified `qwen35` plugin runs the forward. The engine records its `rad_issue` calls on a tape and runs one adjoint per op in reverse. |
| Precision | int8 forward (int8 weights, Hadamard-rotated int8 activation codes). Every gradient and trainable weight in bf16 or f32. |
| Gradient through rounding | Straight-through: the gradient passes unchanged where the code is inside the grid and is zero where it saturated. Scales are constants. |
| Base model | Containers of `StillDeadcode/qwen3.8-27b-iqr-v4`. First target `qwen3.8-27b-iq4r-s-v4.rad` (15.4 GiB, fits one card). |
| Adapter | LoRA whose `A` reads the rotated int8 activation the base GEMM reads. PEFT file layout plus one key, `"radiance_input_basis": "fwht128"`. Default rank 16, alpha 32. |
| Data | Chat JSONL, one `{"messages": [...]}` a line, rendered with the container's chat template. Loss on assistant turns only, up to 4096 tokens. Placeholder dataset: UltraChat 200k (MIT). |
| Memory | The forward keeps only the residual at the 65 layer boundaries (about 2.5 GiB at 4096 tokens). The backward replays one layer at a time. |
| Two cards | Data-parallel first (whole container on each card), then a layer split for containers that do not fit one card, then a measurement against tensor-parallel. |
| Verification | Finite differences on the host in f64; functional checks where rounding is involved; radiance itself as the final check. In addition, one independent reference for the bf16 trunk: transformers' own Qwen3.5 with PyTorch autograd on the tiny fixture. |
| PyTorch | A reference, not a dependency. A script generates the reference files once; they are checked in under `tests/fixtures/oracle/`. The build and the tests never import Python. |
| Checkpoints | A checkpoint records the SHA-256 of the container it was trained on. Resuming on a different container is refused. |
| Memory budget | Before the first step the memory plan is evaluated for the largest batch the configuration allows. A run that does not fit refuses to start, with the numbers. |
| Loss and trainable kind | Not built into the engine. An `Objective` owns the loss and names, as `Seed`s, the forward buffers where its gradient enters the tape. A `Delta` owns the trainable parameters, their initialisation and how they cross to radiance. A `Model` is one of each. First pair: `CeObjective` with `LoraDelta`. |
| Contracts from the start (Q1) | `objective.h`, `delta.h` and `config.h` are contract headers of issue 0. The reference loss, the adapter, the forward driver, the backward core and the joins are written against them; nothing is hardcoded and migrated later. Issue 59 lands beside issue 0. |
| Two milestones (Q2) | The first stays plain assistant-turn cross-entropy (issues 54 and 56). The ALoDLM objective is the second (issues 62 to 65), separate, on the same base. |
| Patch sets | `patches/serving/` (`0201`, `0210`, `0211`) is what a radiance needs to serve an adapter trained here; it must stand alone and is what goes upstream. `patches/training/` (`0101`, `0102`) is what only the trainer needs. `-DR97_PATCH_SET=serving` builds the serving radiance. |
| Training and serving | They share the card and never run at once. The trainer exits, then the serving-only radiance loads the adapter directory. |
| A step without a card | `ReferenceStepFn`: the tiny fixture's op graph in f64 from the twins and the host adjoints. No value comes from a radiance kernel, no kernel library is loaded on the `--program` path, no Python, no card; the default path runs the host kernel chain once per batch and keeps only its `OpRecord`s, and both paths read the weights through `rad_core`'s reader. The loop, the checkpoint and the resume are tested on it. |
| Seam to radiance | As small as possible: kernels and adjoints stay behind radiance's plugin ABI; every patch has one concern, a size budget and an upstream status in `docs/UPSTREAM.md`. |

**Three decisions made while cutting the work, still to be confirmed by the owner:**

1. The adapter's forward is radiance's own LoRA path in training too; the trainer writes weights from
   memory into radiance's slot. Training and serving then run the same code.
2. The optimiser runs on the host. Issue 54 measures its share of the step time; above 5 %, a device
   optimiser becomes a follow-up.
3. Host adjoints are written per elementary op; device adjoints cover the fused ops the card issues
   and are checked against the composition of the host adjoints.

## 3. How one training step works

Once, before the first step: the trainer hashes the container (the hash goes into every checkpoint),
and evaluates the memory plan for the largest batch the configuration allows. A run that does not fit
the card's free VRAM does not start.

1. **Tokenise.** A conversation becomes token ids and a mask that is 1 on the tokens the model would
   have generated.
2. **Forward.** Each sequence runs as one prefill step from position 0 through radiance's plugin, with
   the adapter in radiance's LoRA slot. The residual at each layer boundary is stored and the ops are
   recorded on the tape.
3. **Loss.** The run's `Objective` reads the boundaries the forward exposes and returns the loss and
   the seeds: the buffers its gradient enters at, with their cotangents. In the first milestone that
   is cross-entropy over the scored tokens, computed in chunks of positions (the vocabulary is 248320
   wide), with one seed on the logits. Every boundary the loss read must be seeded, or the step is
   refused.
4. **Backward.** Start from the seeds. From the last layer to the first: decode that layer's IQR weights to int8, replay the
   layer from its stored input, snapshot what the adjoints need, run the adjoints in reverse, free the
   scratch.
5. **Update.** AdamW on the f32 master weights, then the `Delta` packs the new weights and they go
   back into radiance's slot.

## 4. Rules that make the work parallel

- **Fixed contracts first.** Issue 0 defines seven small headers (`Batch`, `ParamSet`, `StepFn` with
  `Model`, `OpRecord`, `Objective`, `Delta`, `Config`). Optimiser, batcher, training loop and backward
  core are built against them and tested on toy problems, without a model.
- **A loss or an adapter kind is one new file.** Objectives and deltas register by name (issue 59).
  The forward driver, the backward core and the joins never name one.
- **No shared files.** Sources and tests are collected by directory glob; ops, tests and subcommands
  register themselves; every topic has its own doc file; patches are separate numbered files.
- **One family, one owner.** Each adjoint family has one source file, one test file and one doc, on
  the host and again on the card.
- **Paper before kernels.** The gated delta net's adjoint is derived and checked in a plain Python
  script before six kernel issues start from it.
- **A second opinion, generated once.** The PyTorch reference (issue 29) is a set of checked-in
  files. The two block checks and the whole-model backward read them; no contributor needs PyTorch
  to run the tests.
- **A live step, generated from our own twins.** The `ReferenceStepFn` (issue 60) runs the real op
  graph of the tiny fixture in f64 with no card, so the loop is not only tested on a toy. No value
  comes from a radiance kernel; only the `--program` path also loads no kernel library at all.
- **Three test labels.** `host` needs no card, `gpu` needs one R9700, `model` needs the real container
  and is skipped without it.
- **Two plans, two namespaces.** Cite this plan as `R97 #<n>` and the amendment it was merged with as
  `radtrain §<n>`, never by a bare number: `radtrain §66` is the ALoDLM objective, `R97 #66` the
  patch-surface audit.

## 5. The work: 69 issues in 14 rounds

A round is the earliest point an issue can start, assuming unlimited people. "Card" says what hardware
the issue needs: none, one R9700 of any contributor, or the machine with both.

| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Issues that can start | 5 | 8 | 17 | 18 | 5 | 3 | 2 | 2 | 2 | 2 | 2 | 1 | 1 | 1 |

**Totals:** 46 issues need no card, 19 need one R9700, 4 need both.
Rounds 1 to 10 are the first milestone, rounds 11 to 14 the second.

### Round 1: five issues, no dependencies

| # | Issue | Card |
|---|---|---|
| 0 | Build skeleton: submodule, the two patch sets, `r97train`, test tiers, seven contract headers | none |
| 1 | `docs/DESIGN.md`, `CONTRIBUTING.md` and the plan as `docs/PLAN.md` | none |
| 2 | Fetch the placeholder dataset | none |
| 3 | Adapter contract `docs/ADAPTER.md` | none |
| 4 | Derivation of the gated delta net's forward and adjoints | none |

Issue 0 is the bottleneck: almost everything waits for it. One person should finish it fast. It now
also freezes the `Objective` and `Delta` contracts, so it is a little larger than before.

### Round 2: eight issues, all after 0

| # | Issue | Card |
|---|---|---|
| 5 | GPU test guard: card selection, lock per card, free-VRAM check | one |
| 6 | Tiny bf16 fixture with a prefill helper | none |
| 7 | Adjoint contract: registry, f64 twins, finite-difference checker | none |
| 8 | Tape: radiance patch `0101` and the recorder | none |
| 9 | Optimiser: AdamW, schedule, clipping, accumulation | none |
| 10 | Batcher: token-budget batches, deterministic by seed | none |
| 11 | Reference loss `CeObjective` in f64 with its seed on the logits | none |
| 59 | Model registry and the seed check: objectives and deltas by name | none |

Issue 59 is the only added issue on the critical path: the adapter files, the forward driver and the
backward core wait for it. It is small and needs no model.

### Round 3: seventeen issues

| # | Issue | After | Card |
|---|---|---|---|
| 12 | Adapter files and `LoraDelta`: read, write, initialise | 0, 3, 6, 59 | none |
| 13 | Tiny int8 fixture (`w8a8_fwht128`) | 6 | none |
| 14 | Chat data with the assistant loss mask | 6 | none |
| 15 | Live adapter: a delta's `WireBuffers` in radiance's LoRA slot (bf16 trunk) | 3, 6 | none |
| 18 | Training forward on the host, for a `Model` | 6, 11, 59 | none |
| 20 | Host adjoints `elementwise` | 7 | none |
| 21 | Host adjoints `linear` (frozen GEMMs, int8 grid) | 7 | none |
| 22 | Host adjoints `quant` (straight-through) | 7 | none |
| 23 | Host adjoints `lora` (gradients of `A` and `B`) | 3, 7 | none |
| 24 | Host adjoints `qk` (RoPE, `kv_store`, gate) | 7 | none |
| 25 | Host adjoints `attention` | 7 | none |
| 26 | Host adjoints `gdn_prep` | 4, 7 | none |
| 27 | Host adjoints `gdn_kkt` | 4, 7 | none |
| 28 | Host adjoints `gdn_scan` | 4, 7 | none |
| 29 | PyTorch reference: transformers' Qwen3.5 with autograd, as checked-in files | 3, 6 | none |
| 33 | Backward core: reverse walk from a list of seeds, cotangents per buffer version | 7, 8, 59 | none |
| 38 | Device kernel plugin `libr97` and the device-versus-host harness | 5, 7 | one |

### Round 4: eighteen issues

| # | Issue | After | Card |
|---|---|---|---|
| 16 | radiance patch `0210`: LoRA on the int8 trunk, host kernels | 3, 13 | none |
| 17 | Op inventory `r97train ops`, with the seed boundaries | 8, 13 | none |
| 19 | `r97train loss`, cross-checked against radiance's KL mode | 14, 18 | none |
| 30 | Whole gated-delta-net layer: gradient check, also against the PyTorch reference | 6, 21, 26 to 29 | none |
| 31 | Attention and MLP blocks: gradient check, also against the PyTorch reference | 6, 20, 21, 24, 25, 29 | none |
| 32 | Layer replay on the host, bit-identical | 8, 18 | none |
| 34 | Training loop: `Config`, logging, checkpoints, resume | 9, 10, 12 | none |
| 40 | Device adjoints `elementwise` | 20, 38 | one |
| 41 | Device adjoints `quant` and the fused producers | 20, 22, 38 | one |
| 42 | Device adjoints `linear` (transposed int8 linear) | 21, 38 | one |
| 43 | Device kernel: IQR decode to int8 | 38 | one |
| 44 | Device kernel: loss on the card | 11, 38 | one |
| 45 | Device adjoints `qk` (`qk_norm_rope_gate`) | 20, 24, 38 | one |
| 46 | Device adjoints `attention` | 22, 25, 38 | one |
| 47 | Device adjoints `gdn_prep` | 26, 38 | one |
| 48 | Device adjoints `gdn_kkt` | 27, 38 | one |
| 49 | Device adjoints `gdn_scan` | 28, 38 | one |
| 50 | Device adjoints `lora` | 23, 38 | one |

### Rounds 5 to 8: the joins

| # | Round | Issue | After | Card |
|---|---|---|---|---|
| 35 | 5 | Host backward on the bf16 trunk, checked end to end by finite differences and against the PyTorch reference | 15, 17, 18, 23, 29 to 33 | none |
| 39 | 5 | Device forward on the real container at 4096 tokens | 5, 17 to 19 | one |
| 51 | 5 | radiance patch `0211`: int8-code LoRA on the R9700 kernels | 5, 16 | one |
| 60 | 5 | `ReferenceStepFn`: the tiny fixture in f64 from the twins, `train --reference` | 6, 11, 12, 17, 20 to 29, 33, 34 | none |
| 66 | 5 | Patch-surface audit, size budgets, `docs/UPSTREAM.md` | 0, 8, 15, 32 | none |
| 36 | 6 | Host backward on the int8 trunk, checked by descent | 13, 16, 22, 35 | none |
| 52 | 6 | Layer replay on the card, with a memory plan | 32, 39 | one |
| 61 | 6 | Serving-only radiance serves a trained adapter, host and card; `docs/DEPLOYMENT.md` | 12, 16, 51 | one |
| 37 | 7 | `r97train train` on the host: overfit, resume, serve with radiance; the reference run beside it | 14, 19, 34, 36, 60 | none |
| 53 | 7 | Device backward: one 4096-token step on one card | 36, 40 to 52 | one |
| 54 | 8 | Train on one card: a short real run, adapter served by the serving-only radiance | 2, 37, 53, 61 | one |
| 67 | 8 | `docs/USAGE.md`, `docs/CHECKPOINT.md`, `docs/RUNBOOK.md`, configuration schema | 34, 37, 60 | none |

After issue 37 the whole trainer exists without a card. This tail is narrow because everything
converges here. Issue 60 gives a model-shaped training step without a card two rounds earlier, and
issue 37 uses it as its proof beside the host backward.

### Rounds 9 and 10: two cards

| # | Round | Issue | After | Card |
|---|---|---|---|---|
| 55 | 9 | Data-parallel on two cards | 54 | both |
| 57 | 9 | Layer split for containers that do not fit one card | 54 | both |
| 56 | 10 | First milestone run with a written report | 55 | both |
| 58 | 10 | Measurement: tensor-parallel against the layer split | 57 | both |

These four need the machine with both cards and run one at a time.

### Rounds 11 to 14: the second milestone (ALoDLM)

| # | Round | Issue | After | Card |
|---|---|---|---|---|
| 62 | 11 | Masked batch for the denoising rollout | 6, 14, 56 | none |
| 68 | 11 | Milestone 2 base container: pick it and fetch it | 56 | none |
| 63 | 12 | Rollout driver: adaptive depth with gold forcing, on the trace | 17, 18, 32, 62, 68 | none |
| 64 | 13 | `ALoDLMObjective`: loss and the seeds on `ce` and `gate_logits` | 59, 63 | none |
| 65 | 14 | `GateDelta` and its adjoint | 15, 35, 59, 64 | none |

These five need no card. Issue 62 and issue 68 wait for issue 56, so that nothing of the second
milestone starts before the first is closed. Issues 62 to 65 form a chain and issue 68 feeds it. The
run of the second milestone and its report are not cut into issues yet (section 1).

### Where the additions of 2026-11-10 sit

The additions gave a round for each new issue. The rounds above follow from what each issue needs,
and differ in five places:

| Issue | Additions | Here | Why |
|---|---|---|---|
| 59 | round 1, after 0 | round 2, after 0 | It lands beside issue 0, as decided (Q1): the headers themselves moved into issue 0, and what is left needs them. |
| 66 | round 2, after 0, 8 | round 5, after 0, 8, 15, 32 | It reviews `0201` and `0102` and builds without `0102`; those patches come from issues 15 and 32. |
| 67 | round 4, after 34 | round 8, after 34, 37, 60 | Every command of the runbook is a `host` test, and `train`, `--resume`, `--reference` and serving exist after issues 37 and 60. |
| 61 | round 5, after 51 | round 6, after 12, 16, 51 | Issue 51 is itself in round 5. |
| 62 to 65 | round 8 | rounds 11 to 14 | They are a chain, and the second milestone starts after issue 56 (Q2). |

Issue 60 also waits for issues 6, 11, 12, 17, 29 and 34 (the fixture, `CeObjective`, `LoraDelta`, the
op list, the reference files and the loop), and stays in round 5.

Issue 68 is not one of the 2026-11-10 additions: it was added when the plan was merged with the
radtrain amendment, which named a milestone-2 base and gave no way to obtain it. It sits in round 11.

## 6. Risks and open points

- **The gated delta net's adjoint.** 48 of the 64 layers are of this kind, and the backward through
  the chunked recurrence has to be derived. Seven issues cover it (4, 26 to 28, 30, 47 to 49).
  Issue 29 adds a check that does not rest on our own derivation: autograd through transformers'
  implementation.
- **The straight-through gradient is a definition.** It cannot be checked by finite differences.
  Issue 36 checks that a descent step lowers the loss (a control step raises it) and that the int8
  gradient agrees with the bf16 one where the quantisation allows. The plan set "cosine above 0.9"
  before any gradient was measured; the measurements since do not reach it and set the acceptance
  (`docs/STEPS-HOST.md`): the bf16 walk against the exact reference agrees to 0.8716 whole-gradient
  cosine (floor 0.85), 0.76 to 0.99 per module; the int8 walk against the bf16 one to 0.67
  whole-gradient (floor 0.6), per site from -0.63 to 0.96. The acceptance is the measured floors.
- **Host kernels for the int8 chain.** The host has no rows for the fused int8 ops, and the plugin
  falls back to an unfused chain. Whether every op of that chain has a host row is not confirmed;
  issue 13 finds out and adds reference rows in this repository if needed.
- **IQR exists on the card only.** Host tests run on a plain `w8a8` fixture; the IQR decode is first
  checked on the real container in issue 43.
- **The PyTorch reference covers the bf16 trunk only.** transformers has no int8 model, and the
  straight-through gradient is this project's definition. The reference may also differ from radiance
  in details (an epsilon, a cast); issue 29 stops and reports such a difference instead of adjusting
  the files.
- **Speed is unproven.** Nothing shows yet that this engine trains faster than a general framework.
  The milestone report states where the step time goes.
- **Nothing has been built or run.** All statements about radiance come from reading its source at
  the pin.
- **The two cards on the two-card machine are occupied.** The radiance server holds about 32.5 GiB on
  each. Rounds 9 and 10 need it off.
- **The seed contract is fixed before any loss uses a second boundary.** `Seed`, `ForwardTape` and
  `WireBuffers` are frozen in issue 0, and the first objective seeds only the logits. The ALoDLM
  objective is the first to seed two boundaries; if it needs something the contract lacks, the
  headers change then, with a note. Issue 35 tests a second objective to catch this early.
- **1e-8 against the PyTorch reference.** Issue 60 asks the f64 twins to reproduce the reference
  gradient to 1e-8 relative. The twins are checked against radiance's kernels only within bf16
  tolerance, and the reference may differ from radiance in an epsilon or a cast (issue 29). If it is
  not reached, the issue names the op; the tolerance is not widened silently.
- **Whether the observer can leave the patches is open.** radiance's plugin documentation at the pin
  lists optional kernel hooks and new ops, but no hook on `rad_issue`. Issue 66 finds out; the
  fallback is one small generic hook that can go upstream.
- **The second milestone rests on things not seen from here.** The ALoDLM sources
  (`aijuus/looping/ALoDLM/alodlm/`) are in another repository and were not read for this plan; their
  arithmetic, their random-number use and their licence are taken on trust. The pinned radiance has
  no adaptive driver (`core/rad_adaptive_driver.cpp` does not exist at the pin), and neither the
  `qwen35` plugin nor the first-milestone container shows a loop range or a gate. No 0.8B
  container is on the machine this plan was written on; issue 68 picks and fetches the base, and issue
  63 establishes the rest of these facts before it writes code.
- **The second milestone has no run yet.** Issues 62 to 65 give the pieces, on the host. A device
  path for the rollout, the run and the report are cut when the first milestone closes.
