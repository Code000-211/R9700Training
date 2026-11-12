#!/usr/bin/env bash
# tests/cmake/config_schema_test.sh <source root> <r97train> -- the commands and the schema the
# binary prints and docs/USAGE.md prints are one thing.
#
# Two comparisons, both run against the tree and then against copies of the document made to break
# them, each of which must fail and name what broke:
#
#   - the configuration table, row for row, after the markdown the document needs (`|`, backticks,
#     bold) is taken off: key, type, default, and the sentence that carries the valid range and the
#     meaning. Drills: a row removed; a type, a default and a range changed.
#   - the commands: every command the binary registers has a section in the document whose usage
#     line is the one `r97train <command> --help` prints and whose `now` rows are exactly the flags
#     that line carries; a `planned` row is not in `--help` and names the issue it waits on. Drills:
#     a flag row removed; a planned flag turned into a now one; a usage line that lost a flag; a
#     command's section removed; a planned row with no issue.
#
# A document that quietly disagrees with the binary is the one thing this test exists to prevent,
# so its reasons to fail are checked with it. The command table's other side — that the table the
# binary prints is the table the subcommands registered — is tests/host/usage_cli.cpp's.
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

# What the command prints is itself a config file, and `empty` is the table's word for an empty
# value, not a placeholder the printer strips: a directory or an opaque `base_id` named `empty`
# survives the round trip.
cat > "$tmp/empty.conf" <<'EOF'
model = /srv/models/tiny.rad
out = empty
base_id = empty
delta.targets = empty
lr = 1e-3
token_budget = 8
accumulation = 1
steps = 3
EOF
empty_run=$("$bin" config --config "$tmp/empty.conf") || {
    printf '%s\n' "$empty_run"
    printf 'FAIL: the command rejected a configuration that names the word empty\n' >&2
    exit 1
}
for line in 'out = empty' 'base_id = empty' 'delta.targets = empty'; do
    grep -qx "$line" <<< "$empty_run" || {
        printf '%s\n' "$empty_run"
        printf 'FAIL: the resolved configuration lost `%s`\n' "$line" >&2
        exit 1
    }
done
printf '%s\n' "$empty_run" > "$tmp/printed.conf"
round_trip=$("$bin" config --config "$tmp/printed.conf") || {
    printf '%s\n' "$round_trip"
    printf 'FAIL: the configuration the command printed does not read back\n' >&2
    exit 1
}
if [[ $round_trip != "$empty_run" ]]; then
    diff -u <(printf '%s\n' "$empty_run") <(printf '%s\n' "$round_trip") || true
    printf 'FAIL: the configuration the command printed reads back changed\n' >&2
    exit 1
fi

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

# ---------------------------------------------------------------- the commands and their flags

# The table the binary prints for `r97train --help`: one `  <name> <hint>` line per command.
command_table() { "$bin" --help; }

# The hint the table carries for <name>, empty when the command registers none.
hint_of() { # hint_of <table> <name>
    awk -v n="$2" '$1 == n { $1 = ""; sub(/^ /, ""); print; exit }' <<< "$1"
}

# The flags a usage hint carries: its `--flag` words, in brackets, parentheses or bare, sorted.
hint_flags() {
    printf '%s\n' "$1" | tr ' ' '\n' | sed 's/^[][(]*//' |
        sed -n 's/^\(--[A-Za-z0-9_-]*\).*/\1/p' | sort -u
}

# The facts a command's section of the document states, one per line:
#   usage<TAB><the line after `usage: r97train `>
#   row<TAB><flag><TAB><state>
section() { # section <document> <name>
    awk -v name="$2" '
        /^### / { inside = ($0 == "### `" name "`"); next }
        /^## / { inside = 0; next }
        !inside { next }
        /^usage: / {
            line = $0
            sub(/^usage: r97train /, "", line)
            print "usage\t" line
            next
        }
        /^\|/ {
            split($0, cell, "|")
            token = cell[2]
            state = cell[3]
            gsub(/^[ \t]+|[ \t]+$/, "", token)
            gsub(/^[ \t]+|[ \t]+$/, "", state)
            gsub(/`/, "", token)
            sub(/[ \t].*/, "", token)
            # the header row and its separator
            if (token == "flag" || token == "" || token ~ /^-+$/) next
            print "row\t" token "\t" state
            next
        }
    ' "$1"
}

# Every command the binary has, documented as the binary prints it. Prints one line per
# disagreement and returns 1 if there is one; the document may hold a section for a command this
# build does not have (a serving build has no `grad`), which is not a disagreement.
check_commands() { # check_commands <document> [<table>]
    local doc=$1 table=${2:-$(command_table)} failed=0
    local name hint want facts first usage now want_flags flag state help
    for name in $(awk 'NF > 1 && $1 != "usage:" && $1 != "commands:" { print $1 }' <<< "$table"); do
        hint=$(hint_of "$table" "$name")
        want="usage: r97train $name"
        [[ -n $hint ]] && want+=" $hint"
        if ! help=$("$bin" "$name" --help 2>&1); then
            printf '%s: `--help` exits non-zero: %s\n' "$name" "$(tr '\n' ' ' <<< "$help")"
            failed=1
            continue
        fi
        first=$(sed -n 1p <<< "$help")
        if [[ $first != "$want" ]]; then
            printf '%s: --help prints `%s`, the table says `%s`\n' "$name" "$first" "$want"
            failed=1
        fi

        facts=$(section "$doc" "$name")
        if [[ -z $facts ]]; then
            printf '%s: docs/USAGE.md has no `### `%s`` section for it\n' "$name" "$name"
            failed=1
            continue
        fi

        usage=$(awk -F'\t' '$1 == "usage" { print $2 }' <<< "$facts")
        want="$name"
        [[ -n $hint ]] && want+=" $hint"
        if [[ $usage != "$want" ]]; then
            printf '%s: the document says `usage: r97train %s`, the binary `%s`\n' \
                "$name" "$usage" "$want"
            failed=1
        fi

        now=$(awk -F'\t' '$1 == "row" && $2 ~ /^--/ && $3 == "now" { print $2 }' <<< "$facts" |
                  sort -u)
        want_flags=$(hint_flags "$hint")
        if [[ $now != "$want_flags" ]]; then
            printf '%s: the document lists the flags [%s], the binary takes [%s]\n' \
                "$name" "$(tr '\n' ' ' <<< "$now")" "$(tr '\n' ' ' <<< "$want_flags")"
            failed=1
        fi

        while IFS=$'\t' read -r _ flag state; do
            [[ -z $flag ]] && continue
            [[ $state == now ]] && continue
            if ! [[ $state =~ ^planned:\ .+\ \(#[0-9]+\)$ ]]; then
                printf '%s: `%s` is `%s`, which is neither now nor planned with its issue\n' \
                    "$name" "$flag" "$state"
                failed=1
                continue
            fi
            if grep -qx -- "$flag" <<< "$want_flags"; then
                printf '%s: `%s` is marked planned and the binary already takes it\n' \
                    "$name" "$flag"
                failed=1
            fi
        done < <(awk -F'\t' '$1 == "row" && $2 ~ /^--/ { print $1 "\t" $2 "\t" $3 }' <<< "$facts")
    done
    return $failed
}

# The tree as it is: the document and the binary agree about every command.
if problems=$(check_commands "$(usage_doc)"); then
    :
else
    printf '%s\n' "$problems"
    printf 'FAIL: the commands and docs/USAGE.md differ\n' >&2
    exit 1
fi
if [[ -z $(command_table | sed -n 's/^  \([a-z][a-z0-9_-]*\) .*/\1/p') ]]; then
    printf 'FAIL: r97train --help printed no command table\n' >&2
    exit 1
fi

# The drills: a copy of the document broken one way each must be refused, and the reason must name
# what broke.
drill() { # drill <copy> <the text the reason must name> <what>
    local copy=$1 needle=$2 what=$3 problems status=0
    problems=$(check_commands "$copy") || status=$?
    if (( status == 0 )); then
        printf 'FAIL: the comparison accepted a broken document: %s\n' "$what" >&2
        exit 1
    fi
    if ! grep -qF -- "$needle" <<< "$problems"; then
        printf '%s\n' "$problems"
        printf 'FAIL: the comparison did not name %s: %s\n' "$needle" "$what" >&2
        exit 1
    fi
}

# A flag the document lists and the binary does not take.
grep -v '^| `--dry-run` |' "$(usage_doc)" > "$tmp/no-dry-run"
drill "$tmp/no-dry-run" "--dry-run" "a flag missing from the document"

# A planned flag turned into one the build has.
sed 's/^| `--placement <spec>` | planned:/| `--placement <spec>` | now:/' "$(usage_doc)" \
    > "$tmp/planned-as-now"
drill "$tmp/planned-as-now" "--placement" "a planned flag marked as now"

# A usage line that lost a flag, which is the failure the document had before this test existed.
sed 's/ \[--dry-run\] \[--resume/ [--resume/' "$(usage_doc)" > "$tmp/stale-usage"
drill "$tmp/stale-usage" "train: the document says" "a usage line the binary no longer prints"

# A command with no section.
sed '/^### `info`$/,/^### `loss`$/{/^### `loss`$/!d}' "$(usage_doc)" > "$tmp/no-info"
drill "$tmp/no-info" "info: docs/USAGE.md has no" "a command missing from the document"

# A planned flag with no issue: the reader cannot tell which issue it waits on. A row carries the
# issue's title and its tracker number; docs/PLAN.md cites its own positions as `R97 #<n>`, so the
# two are not the same number and a row that drops them is what sends a reader to the wrong issue.
sed 's/^| `--devices <N>` | planned: two cards, data-parallel (#57) /| `--devices <N>` | planned: two cards /' \
    "$(usage_doc)" > "$tmp/planned-without-issue"
drill "$tmp/planned-without-issue" "--devices" "a planned flag with no issue"
