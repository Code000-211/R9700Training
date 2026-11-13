#!/usr/bin/env bash
# tests/cmake/patch_audit_test.sh <source root> -- exercise ci/patch_audit.sh on throwaway copies.
#
# The audit runs against the tree (it must pass) and against copies that are made to break each of
# its rules: a patch grown past its budget (whose diff stat must be printed), a patch with no
# Concern line, a serving patch that touches a file a training patch introduces, and a stale row
# in docs/UPSTREAM.md.
set -euo pipefail

src=${1:?usage: patch_audit_test.sh <source root>}
audit=$src/ci/patch_audit.sh
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
tree=$tmp/tree

prepare() {
    rm -rf "$tree"
    mkdir -p "$tree/docs"
    cp -a "$src/patches" "$tree/"
    cp "$src/docs/UPSTREAM.md" "$tree/docs/"
}

check() { # check <pass|fail> <needle> <what>
    local want=$1 needle=$2 what=$3 rc
    set +e
    out=$("$audit" "$tree" 2>&1)
    rc=$?
    set -e
    if [[ $want == pass ]]; then
        if (( rc != 0 )); then
            printf '%s\n' "$out"
            printf 'FAIL: audit rejected a clean tree: %s\n' "$what" >&2
            exit 1
        fi
    else
        if (( rc == 0 )); then
            printf '%s\n' "$out"
            printf 'FAIL: audit accepted a broken tree: %s\n' "$what" >&2
            exit 1
        fi
        if ! grep -qF "$needle" <<< "$out"; then
            printf '%s\n' "$out"
            printf 'FAIL: audit did not say %s: %s\n' "$needle" "$what" >&2
            exit 1
        fi
    fi
}

prepare
check pass "" "the tree as it is"

# A patch grown past its budget fails, and its diff stat is printed: 40 added lines of filler take
# the smallest patch well past the budget its row names.
prepare
small=$tree/patches/serving/0201-narrow-head-gate.patch
{
    head -2 "$small"
    printf 'diff --git a/ci-filler.txt b/ci-filler.txt\nnew file mode 100644\n'
    printf -- '--- /dev/null\n+++ b/ci-filler.txt\n@@ -0,0 +1,40 @@\n'
    for i in $(seq 1 40); do printf '+filler %s\n' "$i"; done
    tail -n +3 "$small"
} > "$tmp/grown"
mv "$tmp/grown" "$small"
check fail "0201-narrow-head-gate.patch costs" "a patch over its budget"
grep -qF 'ci-filler.txt' <<< "$out" ||
    { printf '%s\n' "$out"; printf 'FAIL: no diff stat for the grown patch\n' >&2; exit 1; }

# A patch without its two header lines fails.
prepare
small=$tree/patches/serving/0201-narrow-head-gate.patch
tail -n +2 "$small" > "$tmp/headerless"
mv "$tmp/headerless" "$small"
check fail "line 1 is not 'Concern:" "a patch with no Concern line"

# A serving patch that touches a file a training patch introduces fails: the serving set stands
# alone on a radiance that carries no training patch. The tree's training set introduces no file
# of its own (the prefill that used to be training's is serving's, 0202), so the copy gets a
# training patch that does, and its row, before the serving patch is made to touch it.
prepare
cat > "$tree/patches/training/0199-ci-fixture.patch" <<'PATCH'
Concern: a training-side file for the audit's overlap rule
Upstream: not-submitted
diff --git a/core/ci-fixture.txt b/core/ci-fixture.txt
new file mode 100644
--- /dev/null
+++ b/core/ci-fixture.txt
@@ -0,0 +1,1 @@
+// the file the overlap check pairs with
PATCH
cat >> "$tree/docs/UPSTREAM.md" <<'ROW'
| `patches/training/0199-ci-fixture.patch` | training | 1 | 0 | 4 | not-submitted | a training-side file for the audit's overlap rule |
ROW
small=$tree/patches/serving/0201-narrow-head-gate.patch
{
    head -2 "$small"
    tail -n +3 "$small"
    printf 'diff --git a/core/ci-fixture.txt b/core/ci-fixture.txt\n'
    printf -- '--- a/core/ci-fixture.txt\n+++ b/core/ci-fixture.txt\n'
    printf '@@ -1,1 +1,1 @@\n-// old\n+// new\n'
} > "$tmp/overlap"
mv "$tmp/overlap" "$small"
check fail "ci-fixture.txt" "a serving patch touching a training patch's file"

# A row in the document without a patch, or a patch without a row, fails.
prepare
grep -v '`patches/training/0102-replay-block.patch`' "$tree/docs/UPSTREAM.md" > "$tmp/doc"
mv "$tmp/doc" "$tree/docs/UPSTREAM.md"
check fail "0102-replay-block.patch has no row" "a patch missing from docs/UPSTREAM.md"

printf 'patch_audit_test: ok\n'
