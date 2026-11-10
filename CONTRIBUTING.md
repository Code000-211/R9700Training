# Contributing

Work on this repository is driven through issues. These rules exist so that
several people can work at the same time without touching each other's files.

## Files, not shared edits

Add files instead of editing shared ones.

- **Docs:** one doc per topic. `docs/adjoints/<family>.md` for each adjoint
  family, `docs/kernels/<family>.md` for each kernel family,
  `docs/reports/<date>-<topic>.md` for reports.
- **Patches:**
  - `patches/serving/NNNN-<topic>.patch`, numbered in the `02xx` range (LoRA
    lives there); this is what a radiance needs to serve an adapter trained
    here, and it must stand alone.
  - `patches/training/NNNN-<topic>.patch`, numbered in the `01xx` range
    (runtime hooks); this is what only the trainer needs.
  - Each patch carries one concern and two header lines, `Concern:` and
    `Upstream:`, and is written self-contained so it can be upstreamed alone.
  - A patch that only the trainer needs never goes into the serving set.

## Contract headers

The contract headers in `include/r97/` — `batch.h`, `params.h`, `step.h`,
`op.h`, `objective.h`, `delta.h`, `config.h` — are fixed interfaces between
contributors. They change only with a note at the top of the pull request.

## Losses and trainable kinds

A new loss is a new `Objective` and a new trainable kind is a new `Delta`,
each in its own file and registered by name. Never add a branch for one in the
forward driver or the backward core.

## Test labels

What the labels may assume:

- `host`: no card. Runs anywhere.
- `gpu`: one R9700. Runs on any contributor's card.
- `model`: the real container. Skipped when the container is absent.

## The two-card machine

The machine with both cards is shared, and only issues marked for it use both
cards.
