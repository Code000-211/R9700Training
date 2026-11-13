#!/usr/bin/env bash
# tests/cmake/serving_build_ccache_test.sh <source root> -- ci/serving_build.sh and ccache.
#
# The script is run with stand-in cmake, ctest and patch_audit, so nothing is configured or built:
# what is checked is the configure calls it makes. With ccache on PATH, both trees' configure calls
# name ccache as the C, C++ and HIP compiler launcher (HIP too, or the card's kernels would miss
# the cache), and hand the compiles the worktree root as ccache's base directory, so a round's
# worktree path does not enter the hash and the next round's worktree hits the cache.
# R97_NO_CCACHE=1 turns that off, and so does a machine that has no ccache; neither may leave a
# launcher or a base directory behind.
set -euo pipefail

src=${1:?usage: serving_build_ccache_test.sh <source root>}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# The script under test, in a tree of its own, with a patch audit that passes: the audit is
# tests/cmake/patch_audit_test.sh's business, not this test's.
tree=$tmp/tree
mkdir -p "$tree/ci"
cp "$src/ci/serving_build.sh" "$tree/ci/"
printf '#!/bin/sh\nexit 0\n' > "$tree/ci/patch_audit.sh"
chmod +x "$tree/ci/patch_audit.sh"

# Stand-ins that record the arguments of every call -- and the ccache settings the compiles would
# inherit -- and succeed. The log has one line per call.
mkdir -p "$tmp/bin" "$tmp/bin-noccache"
for dir in "$tmp/bin" "$tmp/bin-noccache"; do
    cat > "$dir/cmake" <<'EOF'
#!/bin/sh
{
    printf 'cmake'
    printf ' %s' "$@"
    printf ' CCACHE_BASEDIR=%s' "${CCACHE_BASEDIR:-none}"
    printf ' CCACHE_NOHASHDIR=%s\n' "${CCACHE_NOHASHDIR:-none}"
} >> "$R97_TEST_CMAKE_LOG"
EOF
    cat > "$dir/ctest" <<'EOF'
#!/bin/sh
{ printf 'ctest'; printf ' %s' "$@"; printf '\n'; } >> "$R97_TEST_CMAKE_LOG"
EOF
    chmod +x "$dir/cmake" "$dir/ctest"
    # The script reads its own location with dirname; nothing else outside the shell's builtins is
    # needed, so the test's PATH is the stand-ins only and no host tool can leak in.
    ln -s "$(command -v dirname)" "$dir/dirname"
done
# A ccache of this test's own: whether the host has one installed must not change the answer.
printf '#!/bin/sh\nexit 0\n' > "$tmp/bin/ccache"
chmod +x "$tmp/bin/ccache"

run_script() { # run_script <PATH value> <log> [VAR=value ...]
    local path=$1 log=$2
    shift 2
    # Bash by absolute path: the PATH the script gets is the stand-ins only. The ccache settings this
    # test is about are cleared, whichever shell started it -- ci/serving_build.sh sets them itself.
    if ! env -u CCACHE_BASEDIR -u CCACHE_NOHASHDIR PATH="$path" \
        R97_TEST_CMAKE_LOG="$log" R97_CI_ROOT="$tmp/build-ci" "$@" \
        "${BASH:-/bin/bash}" "$tree/ci/serving_build.sh" > "$tmp/out" 2>&1; then
        cat "$tmp/out"
        printf 'FAIL: ci/serving_build.sh failed: %s\n' "$*" >&2
        exit 1
    fi
}

configure_lines() { grep -E '^cmake (-S|.* -S) ' "$1" || true; }

fail() { # fail <log> <message>
    cat "$1"
    printf 'FAIL: %s\n' "$2" >&2
    exit 1
}

check() { # check <pass|fail> <log> <what>
    local want=$1 log=$2 what=$3 n
    n=$(configure_lines "$log" | wc -l)
    if (( n != 2 )); then
        fail "$log" "expected two configure calls, saw $n: $what"
    fi
    local line flag
    while IFS= read -r line; do
        if [[ $want == pass ]]; then
            for flag in CMAKE_C_COMPILER_LAUNCHER CMAKE_CXX_COMPILER_LAUNCHER \
                        CMAKE_HIP_COMPILER_LAUNCHER; do
                if [[ " $line " != *" -D$flag=ccache "* ]]; then
                    fail "$log" "a configure call did not name ccache for $flag: $what"
                fi
            done
            if [[ " $line " != *" CCACHE_BASEDIR=$tree "* ]] ||
               [[ $line != *" CCACHE_NOHASHDIR=1" ]]; then
                fail "$log" "a configure call left ccache hashing the worktree path: $what"
            fi
        else
            if [[ $line == *ccache* ]]; then
                fail "$log" "a configure call named ccache when it was disabled: $what"
            fi
            if [[ $line != *" CCACHE_BASEDIR=none"* ]] ||
               [[ $line != *" CCACHE_NOHASHDIR=none"* ]]; then
                fail "$log" "a configure call set a ccache base directory when disabled: $what"
            fi
        fi
    done < <(configure_lines "$log")
}

run_script "$tmp/bin" "$tmp/present"
check pass "$tmp/present" "ccache on PATH"

run_script "$tmp/bin" "$tmp/off" R97_NO_CCACHE=1
check fail "$tmp/off" "R97_NO_CCACHE=1"

run_script "$tmp/bin-noccache" "$tmp/missing"
check fail "$tmp/missing" "no ccache on PATH"

printf 'serving_build_ccache_test: ok\n'
