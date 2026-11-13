#!/usr/bin/env bash
# ci/serving_build.sh -- the two radiance builds, and the tests each one runs.
#
# The default build applies both patch sets and carries r97train; the serving build applies only
# patches/serving/ and carries no trainer. They live side by side under one root and are built from
# the same working tree, which is the arrangement docs/DEPLOYMENT.md section 2 describes. Any CI
# that can run ctest can run this; it needs no card for the host labels.
#
#   ci/serving_build.sh                     # host-only, the default labels
#   R97_CI_HIP=1 ci/serving_build.sh        # the card's kernels too (needs /opt/rocm)
#   R97_CI_CARD=1 R97_CI_HIP=1 ci/serving_build.sh   # also gpu and model, on the card machine
#
# Build trees are ${R97_CI_ROOT:-build-ci}/{training,serving}; a re-run reuses them, and a patch
# set change rebuilds radiance's copy (CMakeLists.txt's patch stamp does that). The default tree is
# built first because the reference rows in tests/fixtures/ are the default build's, and the serving
# tree's serve must reproduce them byte for byte. Compiles go through ccache when it is on PATH, so
# a round reuses what the last one compiled and the serving tree, which is nearly the default
# tree's sources, is nearly free once the default tree is warm; R97_NO_CCACHE=1 turns that off.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${R97_CI_ROOT:-$root/build-ci}"
generator="${R97_CI_GENERATOR:-Ninja}"
jobs="${R97_CI_JOBS:-}"

if [[ "${R97_CI_CARD:-0}" == "1" && "${R97_CI_HIP:-0}" != "1" ]]; then
  echo "ci: R97_CI_CARD=1 needs R97_CI_HIP=1: the card's tests need the device build" >&2
  exit 2
fi

configure_args=(-G "$generator")
if [[ "${R97_CI_HIP:-0}" != "1" ]]; then
  configure_args+=(-DR97_WITH_HIP=OFF)
fi

# Every compile of both trees -- the project's own and radiance's sub-build, host and device alike
# -- goes through ccache when it is on PATH. Its cache lives in the default place under $HOME and
# outlives this worktree, so a round compiles only what changed since the last one.
# R97_NO_CCACHE=1 turns it off, and so does a machine without ccache. The launchers are named
# either way, empty when off, so a tree configured with ccache earlier does not go on using it.
launcher=""
if [[ "${R97_NO_CCACHE:-0}" != "1" ]] && command -v ccache >/dev/null 2>&1; then
  launcher=ccache
  # ccache hashes the absolute paths it is given, and with -g (this build) also the directory it
  # runs in: a new round's worktree is a new path, so without these two every round would miss.
  # Relative to the worktree root, both are the same in every worktree; the debug info of a cached
  # object then names the round that compiled it, which is what a cache for CI accepts.
  export CCACHE_BASEDIR="${CCACHE_BASEDIR:-$root}"
  export CCACHE_NOHASHDIR=1
  echo "ci: compiling through ccache, with $CCACHE_BASEDIR as the base directory"
fi
configure_args+=(-DCMAKE_C_COMPILER_LAUNCHER="$launcher"
                 -DCMAKE_CXX_COMPILER_LAUNCHER="$launcher"
                 -DCMAKE_HIP_COMPILER_LAUNCHER="$launcher")

build_and_test() {
  local tree="$1"
  shift
  echo "== configure $tree"
  cmake -S "$root" -B "$tree" "${configure_args[@]}" "$@"
  echo "== build $tree"
  if [[ -n "$jobs" ]]; then
    cmake --build "$tree" --parallel "$jobs"
  else
    cmake --build "$tree"
  fi
  echo "== host tests in $tree"
  ctest --test-dir "$tree" -L host --output-on-failure
  if [[ "${R97_CI_CARD:-0}" == "1" ]]; then
    echo "== gpu tests in $tree"
    ctest --test-dir "$tree" -L gpu --output-on-failure
    echo "== model tests in $tree (skipped when the container is absent)"
    ctest --test-dir "$tree" -L model --output-on-failure
  fi
}

# Why the patch sets are what they are (one concern per patch, serving independent of training).
"$root/ci/patch_audit.sh"

build_and_test "$out/training"
build_and_test "$out/serving" -DR97_PATCH_SET=serving

# The serve itself, not only the rows beside it: docs/RUNBOOK.md section 7's block starts the
# serving build's server with a trained adapter and holds its loss to the trainer's. The server is
# not a target of this project's build (docs/RUNBOOK.md section 0's third step); section 7's block
# skips unless R97_SERVING_BUILD names a serving tree that has it, so build it there first and name
# the tree for the default tree's host tests. A HIP tree is not the tree to name here: its home
# offers libr4d, which declares nothing without a device, so the serve is the host-only tree's.
if [[ "${R97_CI_HIP:-0}" != "1" ]]; then
  echo "== the server in $out/serving"
  cmake --build "$out/serving/radiance-build" --target radiance
  echo "== host tests with the serving tree named (section 7's serve)"
  R97_SERVING_BUILD="$out/serving" ctest --test-dir "$out/training" -L host --output-on-failure
fi

echo "ci: serving build passed the same labels as the default build"
