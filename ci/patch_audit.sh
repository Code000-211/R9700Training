#!/usr/bin/env bash
# ci/patch_audit.sh [root] -- audit the patch surface against its two documents.
#
# Every patch under patches/ is checked for
#   * its two header lines, `Concern:` and `Upstream:`,
#   * a row in docs/UPSTREAM.md that agrees with those lines and with the patch's size,
#   * a row in patches/README.md that agrees with its set and upstream status,
#   * a size no larger than the budget of that row (added + removed lines),
# and every serving patch is checked not to touch a file a training patch introduced: the serving
# set stands alone.
#
# A patch that exceeds its budget prints its diff stat and fails. A patch that must grow raises
# its budget, with the reason, in the same pull request; that is the only way this audit moves.
#
# <root> defaults to the tree this script lives in. Pass another root to audit a copy, which is
# how tests/cmake/patch_audit_test.sh checks the failures below.
set -euo pipefail

root=${1:-$(cd "$(dirname "$0")/.." && pwd)}
root=$(cd "$root" && pwd)
doc=$root/docs/UPSTREAM.md
readme=$root/patches/README.md
patches=$root/patches

die() { printf 'patch_audit: %s\n' "$*" >&2; exit 2; }
fail() { printf 'patch_audit: %s\n' "$*" >&2; bad=1; }

[[ -d $patches ]] || die "no patches/ under $root"
[[ -f $doc ]] || die "missing $doc"
[[ -f $readme ]] || die "missing $readme"
command -v git >/dev/null || die "git is needed to count a patch's lines"

bad=0

# One row per patch in docs/UPSTREAM.md:
#   | `patches/<set>/<name>.patch` | <set> | <added> | <removed> | <budget> | <upstream> | <concern> |
# The backticks around the path are how a row is found; the concern is the patch's Concern line
# verbatim (no '|' in it).
doc_field() { # doc_field <file> <path> <field number>
    awk -F'|' -v want="$2" -v field="$3" '
        !found && $0 ~ /^[ \t]*\|/ && index($0, "`" want "`") {
            v = $field
            gsub(/^[ \t]+|[ \t]+$/, "", v)
            gsub(/`/, "", v)
            print v
            found = 1
        }
        END { if (!found) exit 1 }
    ' "$1"
}

row_paths() { # row_paths <file> -- the backticked patch path of every table row
    awk -F'|' '/^[ \t]*\|[ \t]*`patches\// {
                   p = $2; gsub(/^[ \t]+|[ \t]+$/, "", p); gsub(/`/, "", p); print p
               }' "$1"
}

doc_paths=$(row_paths "$doc")
readme_paths=$(row_paths "$readme")

all_patches=$(cd "$root" && find patches -name '*.patch' | LC_ALL=C sort)

[[ -n $all_patches ]] || die "no patches under $patches"

# The files the training set introduces from nothing. A serving patch may not touch one: it has to
# stand alone, on a radiance that carries no training patch.
training_new=$(mktemp)
trap 'rm -f "$training_new"' EXIT
while IFS= read -r rel; do
    [[ $rel == patches/training/* ]] || continue
    new=0
    while IFS= read -r line; do
        case $line in
            "new file mode"*) new=1 ;;
            "diff --git"*)    new=0 ;;
            "+++ /dev/null")  new=0 ;;
            "+++ b/"*)
                if (( new )); then printf '%s\n' "${line#+++ b/}" >> "$training_new"; fi
                new=0 ;;
        esac
    done < "$root/$rel"
done <<< "$all_patches"

# Every patch of the tree must be in the document, and every document row must be a patch.
while IFS= read -r rel; do
    name=${rel##*/}
    if ! grep -qxF "$rel" <<< "$doc_paths"; then
        fail "$rel has no row in docs/UPSTREAM.md (its budget and upstream status)"
        continue
    fi

    concern=$(sed -n '1p' "$root/$rel")
    upstream=$(sed -n '2p' "$root/$rel")
    if [[ ! $concern =~ ^Concern:[[:space:]]+. ]]; then
        fail "$name: line 1 is not 'Concern: <what it changes>'"
        continue
    fi
    if [[ ! $upstream =~ ^Upstream:[[:space:]]+(not-submitted|open|merged)[[:space:]]*$ ]]; then
        fail "$name: line 2 is not 'Upstream: not-submitted|open|merged'"
        continue
    fi
    concern=${concern#Concern:}
    concern=${concern#"${concern%%[![:space:]]*}"}
    upstream=${upstream#Upstream:}
    upstream=${upstream#"${upstream%%[![:space:]]*}"}
    upstream=${upstream%"${upstream##*[![:space:]]}"}

    set=${rel#patches/}
    set=${set%%/*}
    d_set=$(doc_field "$doc" "$rel" 3 || true)
    d_added=$(doc_field "$doc" "$rel" 4 || true)
    d_removed=$(doc_field "$doc" "$rel" 5 || true)
    d_budget=$(doc_field "$doc" "$rel" 6 || true)
    d_upstream=$(doc_field "$doc" "$rel" 7 || true)
    d_concern=$(doc_field "$doc" "$rel" 8 || true)

    # patches/README.md is the index: for each patch its set, what it is for, its upstream status.
    r_set=$(doc_field "$readme" "$rel" 3 || true)
    r_what=$(doc_field "$readme" "$rel" 4 || true)
    r_upstream=$(doc_field "$readme" "$rel" 5 || true)
    if [[ -z $r_what ]]; then
        fail "$rel has no row in patches/README.md (its set, what it is for, its upstream status)"
    else
        [[ $r_set == "$set" ]] ||
            fail "$name: patches/README.md says set '$r_set', the file is in '$set'"
        [[ $r_upstream == "$upstream" ]] ||
            fail "$name: patches/README.md says upstream '$r_upstream', the patch says '$upstream'"
    fi

    for v in "$d_set" "$d_added" "$d_removed" "$d_budget" "$d_upstream" "$d_concern"; do
        [[ -n $v ]] || { fail "$rel: row in docs/UPSTREAM.md is missing a field"; break; }
    done
    [[ -n $d_budget ]] || continue

    [[ $d_set == "$set" ]] ||
        fail "$name: docs/UPSTREAM.md says set '$d_set', the file is in '$set'"
    [[ $d_concern == "$concern" ]] ||
        fail "$name: docs/UPSTREAM.md concern differs from the patch's Concern line"
    [[ $d_upstream == "$upstream" ]] ||
        fail "$name: docs/UPSTREAM.md says upstream '$d_upstream', the patch says '$upstream'"
    [[ $d_budget =~ ^[0-9]+$ ]] ||
        { fail "$name: budget '$d_budget' in docs/UPSTREAM.md is not a number"; continue; }

    read -r added removed < <(git -C "$root" apply --numstat "$rel" |
        awk '{ a += $1; d += $2 } END { printf "%d %d\n", a, d }')
    size=$((added + removed))
    [[ $d_added == "$added" && $d_removed == "$removed" ]] ||
        fail "$name: docs/UPSTREAM.md says +$d_added/-$d_removed, the patch is +$added/-$removed"

    if (( size > d_budget )); then
        printf 'patch_audit: %s costs %d lines (+%d/-%d), budget %d\n' \
            "$name" "$size" "$added" "$removed" "$d_budget" >&2
        git -C "$root" apply --stat "$rel" | sed 's/^/    /' >&2
        bad=1
    fi
done <<< "$all_patches"

# A serving patch that touches a file a training patch introduced cannot stand alone.
while IFS= read -r rel; do
    [[ $rel == patches/serving/* ]] || continue
    while IFS= read -r touched; do
        if grep -qxF "$touched" "$training_new"; then
            fail "${rel##*/} touches $touched, which a training patch introduces; \
the serving set must stand alone"
        fi
    done < <(sed -n 's/^+++ b\///p' "$root/$rel")
done <<< "$all_patches"

# A document row without a patch is stale too.
while IFS= read -r rel; do
    [[ -n $rel ]] || continue
    grep -qxF "$rel" <<< "$all_patches" || fail "docs/UPSTREAM.md lists $rel, which is not a patch"
done <<< "$doc_paths"
while IFS= read -r rel; do
    [[ -n $rel ]] || continue
    grep -qxF "$rel" <<< "$all_patches" || fail "patches/README.md lists $rel, which is not a patch"
done <<< "$readme_paths"

if (( bad )); then
    printf 'patch_audit: FAILED\n' >&2
    exit 1
fi
printf 'patch_audit: %d patches, all within budget and in docs/UPSTREAM.md and patches/README.md\n' \
    "$(wc -l <<< "$all_patches" | tr -d ' ')"
