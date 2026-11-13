# The patches

r97 builds radiance from the pinned submodule by copying it into the build tree and applying a
patch set to the copy. There are two sets, told apart by the number that starts a patch's file
name:

- `patches/serving/` (`02xx`): what a radiance needs to serve an adapter trained here. A patch in
  this set stands alone on the pin and is what goes upstream. `-DR97_PATCH_SET=serving` applies
  only these, and that build is radiance and its kernels, not `r97train`.
- `patches/training/` (`01xx`): runtime hooks only the trainer needs. A serving-only build applies
  none of them. `-DR97_PATCH_SET=all`, the default, applies both sets in file-name order.

Each patch carries one concern and is written so it can be submitted upstream alone. Every patch
begins with two header lines, `Concern:` and `Upstream:` (`not-submitted`, `open` or `merged`);
each set's `README.md` has that rule. `docs/UPSTREAM.md` is the ledger that adds each patch's size
and line budget. `ci/patch_audit.sh` holds the patches, that ledger and this table together, and
`tests/cmake/patch_audit_test.sh` drives its failures.

## For each patch

| Patch | Set | What it is for | Upstream |
| --- | --- | --- | --- |
| `patches/serving/0201-lora-set-weights.patch` | serving | an adapter slot put in service from factors already in memory, not only from an adapter directory; the trainer's live slot and a serve's handover both use it | not-submitted |
| `patches/serving/0201-narrow-head-gate.patch` | serving | the gated attention of a head narrower than the quantiser's group, which is the int8 fixture's 32-by-16 head, so that fixture's own attention runs | not-submitted |
| `patches/serving/0210-lora-int8-trunk.patch` | serving | an adapter on the rotated int8 trunk through the host and card kernels, `libavx`, `libref` and `libr4d`: the adapter this deployment serves | not-submitted |
| `patches/training/0101-issue-observer.patch` | training | one optional observer called per issue after its op has run, plus a pair of block brackets: the hook the tape, the boundary recorder and the replay all read | not-submitted |
| `patches/training/0102-replay-block.patch` | training | running one recorded block alone from a restored residual, and each trunk architecture's agreement to honour it | not-submitted |
| `patches/training/0103-prefill-logits.patch` | training | a fixture prefill driven through the engine that keeps every position's logits, which is where the trainer takes its loss | not-submitted |
| `patches/training/0104-prefill-lora.patch` | training | letting that prefill run one named adapter slot, so an in-memory update is one step away | not-submitted |
| `patches/training/0105-prefill-not-sampled.patch` | training | a prefill that draws no token: no sampler op is issued, and a prompt of exactly the context still fits | not-submitted |
| `patches/training/0106-preload-adapter.patch` | training | a run whose prefill steps learn their adapter at preload, before an engine exists | not-submitted |
| `patches/training/0119-issue-observer-hook.patch` | training | the training tape attaching to a runtime context, with the engine remaining its only owner | not-submitted |
| `patches/training/0120-view-operands.patch` | training | the issue record naming a narrowing view and where its values are, so the tape can read them | not-submitted |
| `patches/training/0121-issue-inout-mark.patch` | training | marking an operand whose kernel also reads the value it replaces | not-submitted |
| `patches/training/0122-issue-stream.patch` | training | the stream a step's ops issue on, so an observer can order its own copy behind the op it saw | not-submitted |
