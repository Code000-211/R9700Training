# R9700Training

Training on the AMD Radeon AI PRO R9700.

The engine, [radiance](https://codeberg.org/StillDeadcode/radiance), is a pinned git submodule at
`third_party/radiance`. The build never touches the submodule's working tree: cmake copies it into
the build tree, applies the selected patch sets onto the copy (name order, across both
directories), and builds its static `rad_core` from there. `r97train` links that archive.

## Build

Ninja, HIP from `/opt/rocm`, target `gfx1201`:

```sh
cmake -B build -G Ninja
cmake --build build
```

A host-only configuration without HIP builds too:

```sh
cmake -B build -G Ninja -DR97_WITH_HIP=OFF
cmake --build build
```

Options:

- `R97_PATCH_SET` — `all` (default) applies `patches/serving/` and `patches/training/` to the
  radiance copy and builds `r97train`; `serving` applies only `patches/serving/` and builds the
  patched radiance without `r97train` (the build that serves an adapter trained here).
- `R97_WITH_HIP` — `AUTO` (default, radiance detects the toolchain), `ON`, `OFF` (host-only).
- `RAD_GPU_TARGETS` — AMDGPU architectures radiance's HIP kernels target, default `gfx1201`.
- `R97_BUILD_TESTS` — build the tests, default `ON`.
- `R97_MODEL_RAD` — path to the `.rad` container the model tests read; skipped when absent.

Initialise the submodule first on a fresh clone:

```sh
git submodule update --init
```

## Test

```sh
ctest --test-dir build -L host    # needs no card
ctest --test-dir build -L gpu     # needs the card, runs serially
ctest --test-dir build -L model   # needs the real container under data/, skipped when absent
```

## Layout

- `include/r97/` — the contract headers the parallel issues code against.
- `src/` — the trainer's own code; `src/adjoint/host/`, `src/adjoint/device/`.
- `tools/` — `r97train` and one file per subcommand; each registers itself in the table of
  `tools/commands.h`, so a new subcommand or a new source under `src/` needs no edit to existing
  files.
- `tests/host/` — tests that need no card; `tests/gpu/` — tests that need the card (serial);
  `tests/model/` — tests that need the real container, skipped when absent. `tests/headers/` —
  the standalone-compile checks of the contract headers; `tests/cmake/` — script-mode tests of
  the build machinery; `tests/support/` — header-only test support (`vector_tape.h`); test
  inputs live in `tests/fixtures/`, never in a `data/` directory (every `data/` is ignored).
- `patches/serving/`, `patches/training/` — the patch sets; see the READMEs there. The sets are
  empty for now: the issues that own the patches fill them, and the mechanism stands
  (`tests/cmake/patch_sets_test.cmake`).
- `docs/` — design documents.

## Layout conventions

Sources and tests are collected by glob (`CONFIGURE_DEPENDS`), so a new `.cpp` under `src/` or a
new test under `tests/host/` needs no edit to any existing file.

Work on this repository is driven through issues.
