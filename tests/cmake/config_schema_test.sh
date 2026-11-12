#!/usr/bin/env bash
# tests/cmake/config_schema_test.sh <source root> <r97train> -- the schema the binary prints and the
# configuration table docs/USAGE.md prints are one schema.
#
# The tables are compared row for row, after the markdown the document needs (`|`, backticks, bold)
# is taken off: key, type, default, and the sentence that carries the valid range and the meaning.
# The comparison runs against the tree, where it must pass, and against copies of the document made
# to break it — a row removed, a type, a default and a range changed — each of which must fail and
# name the row. A schema that quietly disagrees with the document is the one thing this test exists
# to prevent, so its reasons to fail are checked with it.
set -euo pipefail

src=${1:?usage: config_schema_test.sh <source root> <r97train>}
bin=${2:?usage: config_schema_test.sh <source root> <r97train>}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# R97_USAGE_DOC lets the drills below point the comparison at a broken copy of the document; the
# document the tree ships is the default.
usage_doc() { printf '%s\n' "${R97_USAGE_DOC:-$src/docs/USAGE.md}"; }

# The binary's rows: the `config` command prints one line per key, two spaces or more between the
# columns, the sentence last.
schema_rows() {
    "$bin" config | tail -n +2 | awk -F'[ ]{2,}' -v OFS='\t' '{
        for (i = 1; i <= 4; ++i) {
            gsub(/[`*]/, "", $i)
            gsub(/[ \t]+/, " ", $i)
            sub(/^ /, "", $i)
            sub(/ $/, "", $i)
        }
        print $1, $2, $3, $4
    }'
}

# The document's rows: the table whose header starts `| key |`, up to the first line that is not a
# row. Cells 2..5 are key, type, default and the range-and-meaning sentence.
doc_rows() {
    awk -F'|' '
        /^\|[[:space:]]*`?key`?[[:space:]]*\|/ { inside = 1; next }
        inside && /^\|[[:space:]]*-/ { next }
        inside && /^\|/ {
            for (i = 2; i <= 5; ++i) {
                gsub(/[`*]/, "", $i)
                gsub(/[ \t]+/, " ", $i)
                sub(/^ /, "", $i)
                sub(/ $/, "", $i)
            }
            print $2 "\t" $3 "\t" $4 "\t" $5
            next
        }
        inside { inside = 0 }
    ' "$(usage_doc)"
}

compare() { # compare <pass|fail> <row the message must name, when it fails> <what>
    local want=$1 needle=$2 what=$3 status=0
    schema_rows > "$tmp/binary"
    doc_rows > "$tmp/doc"
    if ! diff -u "$tmp/doc" "$tmp/binary" > "$tmp/diff"; then status=1; fi
    if [[ $want == pass ]]; then
        if (( status != 0 )); then
            cat "$tmp/diff"
            printf 'FAIL: the schema and docs/USAGE.md differ: %s\n' "$what" >&2
            exit 1
        fi
        return
    fi
    if (( status == 0 )); then
        printf 'FAIL: the comparison accepted a broken document: %s\n' "$what" >&2
        exit 1
    fi
    if ! grep -qF -- "$needle" "$tmp/diff"; then
        cat "$tmp/diff"
        printf 'FAIL: the comparison did not name %s: %s\n' "$needle" "$what" >&2
        exit 1
    fi
}

# The schema must not be empty, and every row must have all four columns: a table that lost its
# rows would otherwise compare equal to an empty binary output.
compare pass "" "the tree as it is"
if [[ $(wc -l < "$tmp/binary") -lt 10 ]]; then
    printf 'FAIL: the binary printed %s schema rows\n' "$(wc -l < "$tmp/binary")" >&2
    exit 1
fi

# A row removed from the document: the key is named in the diff.
sed '/^| `delta.rank` /d' "$(usage_doc)" > "$tmp/removed-doc"
R97_USAGE_DOC=$tmp/removed-doc compare fail "delta.rank" "a key missing from the document"

# A type changed.
sed 's/^| `delta.rank` | integer /| `delta.rank` | number /' "$(usage_doc)" > "$tmp/type-doc"
R97_USAGE_DOC=$tmp/type-doc compare fail "delta.rank" "a key with the wrong type"

# A default changed.
sed 's/^| `delta.rank` | integer | 16 /| `delta.rank` | integer | 8 /' "$(usage_doc)" > "$tmp/default-doc"
R97_USAGE_DOC=$tmp/default-doc compare fail "delta.rank" "a key with the wrong default"

# A range or meaning changed: the sentence is part of the schema.
sed 's/\*\*finite, > 0\*\*: the peak/**finite, >= 0**: the peak/' "$(usage_doc)" > "$tmp/range-doc"
R97_USAGE_DOC=$tmp/range-doc compare fail "lr" "a key with the wrong range"

# The resolved configuration: the file the command is given, the flags over it, validated. A key the
# flags set is printed with the flag's value; an unknown key stops the command with its name.
cat > "$tmp/run.conf" <<'EOF'
model = /srv/models/tiny.rad
out = /tmp/r97-config-schema-test
lr = 1e-4
token_budget = 4096
accumulation = 2
steps = 100
EOF
resolved=$("$bin" config --config "$tmp/run.conf" --steps 7 2>&1) || {
    printf '%s\n' "$resolved"
    printf 'FAIL: the command rejected a valid configuration\n' >&2
    exit 1
}
grep -qx 'steps = 7' <<< "$resolved" || {
    printf '%s\n' "$resolved"
    printf 'FAIL: the flag did not reach the resolved configuration\n' >&2
    exit 1
}
grep -qx 'model = /srv/models/tiny.rad' <<< "$resolved" || {
    printf '%s\n' "$resolved"
    printf 'FAIL: the file did not reach the resolved configuration\n' >&2
    exit 1
}

set +e
refused=$("$bin" config --config "$tmp/run.conf" --no_such_key 1 2>&1)
rc=$?
set -e
if (( rc == 0 )); then
    printf '%s\n' "$refused"
    printf 'FAIL: the command accepted an unknown key\n' >&2
    exit 1
fi
grep -qF 'no_such_key' <<< "$refused" || {
    printf '%s\n' "$refused"
    printf 'FAIL: the refusal did not name the key\n' >&2
    exit 1
}

# A value outside its range is refused, naming the key too.
set +e
refused=$("$bin" config --config "$tmp/run.conf" --delta.rank 0 2>&1)
rc=$?
set -e
if (( rc == 0 )); then
    printf '%s\n' "$refused"
    printf 'FAIL: the command accepted a rank of 0\n' >&2
    exit 1
fi
grep -qF 'delta.rank' <<< "$refused" || {
    printf '%s\n' "$refused"
    printf 'FAIL: the refusal did not name delta.rank\n' >&2
    exit 1
}
