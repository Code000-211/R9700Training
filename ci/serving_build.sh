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
# built first because the reference logits in tests/fixtures/ are the default build's, and the
# serving tree's prefill must reproduce them byte for byte.
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

echo "ci: serving build passed the same labels as the default build"
