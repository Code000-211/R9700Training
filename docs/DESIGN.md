# Design: a purpose-built training engine for the R9700

This document states the settled design decisions. The plan that cuts them into
work is `docs/PLAN.md`; the rules for parallel work are `CONTRIBUTING.md`.

## 1. Goal

A purpose-built training engine for the AMD Radeon AI PRO R9700 (gfx1201,
32 GB), one or two cards. The long-term scope is every kind of training; the
first milestone is LoRA on a frozen Qwen3.8-27B.

## 2. Relation to radiance

Own repository and binary. It reuses radiance's ABI, `.rad` format, forward
kernels (`libr4d`) and host kernels (`libref`, `libavx`). Backward kernels,
optimiser and data pipeline live here.

## 3. Forward graph

The unmodified radiance architecture plugin runs the forward; the engine
records its `rad_issue` calls on a tape and runs one adjoint per op backwards.

## 4. Precision

Forward on radiance's int8 kernels (int8 weights, Hadamard-rotated int8
activation codes); every gradient and trainable weight in bf16 or f32.

## 5. Base model

The containers of `StillDeadcode/qwen3.8-27b-iqr-v4` (architecture `qwen35`,
source quant `w8a8_fwht128`; 64 layers, 16 of them full attention and 48 gated
delta net; hidden 5120; vocabulary 248320). First target
`qwen3.8-27b-iq4r-s-v4.rad`, 15.4 GiB.

## 6. Adapter

LoRA whose `A` reads the rotated int8 activation the base GEMM reads. Its
forward is radiance's own LoRA path in training and in serving, so the two
cannot differ; radiance is patched to accept it.

## 7. radiance patches

radiance changes are patches, applied by the build and upstreamed later by the
owner. They come in two sets: `patches/serving/` is what a radiance needs to
serve an adapter trained here and must stand alone; `patches/training/` is what
only the trainer needs. A radiance built with the serving set alone serves a
trained adapter. Pin `451b136874084650992668bed9e8662deb58de88`.

## 8. Data

Chat JSONL, one `{"messages": [...]}` a line, rendered with the container's
chat template; loss on assistant turns only; up to 4096 tokens.

## 9. Memory

The forward keeps only the residual stream at layer boundaries; the backward
replays one layer at a time.

## 10. Two cards

Data-parallel first (the whole container on each card), then a layer split for
containers that do not fit one card, then a measurement against
tensor-parallel.

## 11. Host first

On the host kernels the plugin issues an unfused chain where the card gets
fused ops (it asks `rad_op_declared`, `arch/qwen35_fp8/qwen35_fp8.cpp` near
line 872). Host adjoints are written per elementary op; a device adjoint of a
fused op is checked against their composition.

## 12. PyTorch as a reference, not a dependency

`tools/oracle/` generates reference gradients for the tiny bf16 fixture from
transformers' own Qwen3.5 with autograd; the files are checked in under
`tests/fixtures/oracle/`, and the build and the tests never import Python.

## 13. Objective and Delta

The loss and the trainable kind are not built into the engine. An `Objective`
(`include/r97/objective.h`) owns the loss and names, as `Seed`s, the forward
buffers at which its gradient enters the tape; a `Delta`
(`include/r97/delta.h`) owns the trainable parameters, their initialisation and
how they cross to radiance; a `Model` is one of each. The forward driver, the
backward core and the joins take a `Model`.

First implementations: `CeObjective` with `LoraDelta` (the first milestone),
then `ALoDLMObjective` with `GateDelta` (the second).

## 14. Two milestones

The first is plain assistant-turn cross-entropy with LoRA on Qwen3.8-27B. The
second is the ALoDLM objective (a masked denoising rollout with an
adaptive-depth exit gate) on the same base; it starts when the first is closed
and does not change it.

## Not planned yet

Full fine-tuning, pre-training, distillation, int8 gradients, a PyTorch speed
baseline.
