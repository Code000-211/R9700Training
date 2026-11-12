# AGENTS.md

Notes for coding agents working in this repository. Work is driven through
issues; read `CONTRIBUTING.md` for the contributor rules this page sums up,
and the `docs/` files for design detail.

## What this is

`r97train`, a trainer for the `radiance` engine on the AMD Radeon AI PRO
R9700 (`gfx1201`). The engine is a pinned submodule at
`third_party/radiance`. **Never touch the submodule's working tree**, and
never edit anything under it: the build copies it into the build tree,
applies the patch sets onto the copy and builds that copy.

## Build and test

```sh
git submodule update --init          # fresh clone only
cmake -B build -G Ninja              # HIP from /opt/rocm, target gfx1201
cmake --build build

cmake -B build -G Ninja -DR97_WITH_HIP=OFF   # host-only build, no card needed

ctest --test-dir build -L host       # no card, always runnable
ctest --test-dir build -L gpu        # one R9700, runs serially, takes gpu_guard
ctest --test-dir build -L model      # needs the real container, exits 77 when absent
```

Default build is `RelWithDebInfo` (`NDEBUG`): `assert` compiles out. Checks
in tests are `R97_CHECK`, never `assert`.

Test tiers are globs of their directories, so a new `.cpp` anywhere under
`src/` or `tests/<tier>/` needs no edit to an existing file. Tests register
with `R97_TEST(name)` (`tests/support/test.h`); a case needing container
*and* card uses `R97_MODEL_TEST` instead. No test other than a `model`
one may skip.

## Hard rules

- **Add files, don't edit shared ones.** One doc per topic
  (`docs/adjoints/`, `docs/kernels/`, `docs/reports/<date>-<topic>.md`).
- **Contract headers are fixed interfaces** between parallel issues
  (`include/r97/`): `batch.h`, `params.h`, `step.h`, `op.h`, `objective.h`,
  `delta.h`, `config.h`. Change them only with a note in the PR.
- **A new loss is a new `Objective`, a new trainable kind a new `Delta`**,
  each in its own file and registered by name. No branch for one in the
  forward driver or the backward core.
- **Patch sets**, not submodule edits:
  - `patches/training/01NN-<topic>.patch` — hooks only the trainer needs.
  - `patches/serving/02NN-<topic>.patch` — what serving needs; stands alone.
  - Unified diff (`git apply -p1`), header lines `Concern:` and `Upstream:`,
    one concern per patch, named so file order applies it correctly.
- **Test inputs live in `tests/fixtures/`**, never in any `data/` directory
  (every `data/` is gitignored).
- **New subcommands** register in the table of `tools/commands.h`; no edit to
  existing files needed.

## Repo layout

- `include/r97/` — contract headers; `src/` — trainer code
  (`src/adjoint/host/`, `src/adjoint/device/`); `tools/` — `r97train`.
- `tests/host/`, `tests/gpu/`, `tests/model/` — the three tiers;
  `tests/headers/` standalone-compile checks; `tests/cmake/` script tests;
  `tests/support/` header-only helpers; `tests/fixtures/` inputs.
- `docs/` — design docs, one topic per file.
