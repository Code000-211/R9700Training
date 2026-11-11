# Patch-surface audit, 2026-11-11

The patches under `patches/` were re-read against radiance's plugin ABI at the pin
`451b136874084650992668bed9e8662deb58de88`: make the seam as small as the ABI allows, review
`0101` (issue observer), `0102` (replay block) and `0201` (set LoRA weights), give every patch a
budget in `docs/UPSTREAM.md`, and add `ci/patch_audit.sh`, which enforces the budget and the
formatting rules.

## What the ABI offers, and what it does not

- `docs/PLUGIN.md` §4 ("Optional kernel hooks") lists `init`/`fini`, `scratch`, `layout`,
  `relayout`, `unrelayout`, `opd_shape`, `replaces`, `tunables`/`tune_valid`. Every one is a
  property of a kernel, resolved by the loader; none is a callback on `rad_issue`.
- §7 ("Defining a new op") is the only way for a plugin to add behaviour: declare a schema and its
  kernel. It describes an op, not an observer of ops the *architecture* issues.
- §13 is about building, installing and the plugin search path (`RADIANCE_HOME`, `kernels/`,
  `architectures/`), so it moves where a plugin is found, not what the runtime offers.
- `abi/rad_plugin.h` says in its header that what it holds is "pure, total, header-only functions
  over the ABI's own types. Nothing allocates, holds state or calls back into the core." A plugin
  cannot subscribe to the issue path through this header.
- `abi/rad_runtime.h` exports `rad_issue` and the `RAD_ISSUE*` macros, but `RadCtx` is opaque and
  there is no registration function for an issue observer, a block bracket or a replay. Nothing in
  `abi/` lets a caller read the ops an architecture issued.

The replay also needs the architecture to honour a block boundary, and an architecture plugin may
do that (§11); `0102`'s `qwen35_bf16` hunk is architecture code, not an ABI addition. What the ABI
does not allow is the runtime telling an outside consumer that an issue happened. Consequently the
observer and replay hooks cannot move behind the plugin ABI at this pin; the patches stay, reduced
to the smallest generic hook. `0201` is different: it is a service entry point that reuses the
engine's existing slot swap, and there is no ABI extension point for "set the weights of a loaded
slot" either, so it stays as it is.

## `0101` — the issue observer

**Before:** 152 added, 0 removed. The observer was called *before* the op ran and carried a
`RadIssueOperand.raw` field ("the pointer as issued") that no consumer used. The ABI comment did
not say when the observer runs, so a consumer reading operand bytes could read the op's output
before it was written.

**After:** 152 added, 0 removed. The same size, with a different composition:

- The call moved to the last thing an issue does, after the op ran and after `horacle` /
  `oracle_after`; the comment now says so. This is what lets the tape and the boundary recorder
  read what an op wrote instead of what was there before it.
- `RadIssueOperand.raw` and its assignment were removed (the field was unused by this tree and by
  the patch's own callers).
- The block brackets are unchanged.

The patch names nothing of this repository: its types are radiance's `Rad*` (`RadIssueObserver`,
`RadIssueOp`, `RadIssueOperand`, `RAD_OPK_*`) and its prose is written as a candidate for
upstream, so it can be submitted as it is.

## `0102` — the replay block

**Before:** 156 added, 27 removed (183 lines of diff). It added a second observer
(`RadReplayObserver`, `rad_set_replay_observer`), a second slot on the context, an `issued` flag
threaded through `observe_issue` so the two observers could be told apart, and a fan-out that
called both, plus `rad_set_replay_block`, which nothing called.

**After:** 114 added, 19 removed (133 lines). The second observer, its setter, its context slot,
the `issued` fan-out and the unused `rad_set_replay_block` are gone. What remains is only what the
single `0101` observer cannot give:

- `Engine::replay_block` / `Ctx::set_replay_block` and the public `RAD_REPLAY_BLOCK`, a way to run
  one block alone from a restored residual;
- the `qwen35_bf16` block split that makes a block boundary something the architecture honours.

The tape subscribes to the `0101` observer, the boundary recorder to the tape's hooks, and replay
does not add an observer of its own. The reduction is 50 diff lines (42 fewer added, 8 fewer removed).

## `0201` — set LoRA weights from memory

**Before and after:** 75 added, 0 removed, unchanged. The review found no ABI extension point to
use: `docs/PLUGIN.md` §4 and §7 cover kernels and ops, and the service API only loads an adapter
from a directory. The patch adds one service entry point that reuses the slot and the swap the
engine already has, stays inside the serving set, and touches no file a training patch introduces.
Its size and budget are unchanged.

## The whole surface

Eight patches, all `not-submitted` at this pin (none is in radiance's `origin/main`). A patch's
size is added + removed lines from `git apply --numstat`; its budget is that size plus about a
tenth of headroom (a little more on the small patches), and is a hard cap. The full table, with the concern of each patch, is in
`docs/UPSTREAM.md`.

| Patch | Set | Size (added/removed) | Budget |
| --- | --- | --- | ---: |
| `0101-issue-observer` | training | 152 / 0 | 170 |
| `0102-replay-block` | training | 114 / 19 | 150 |
| `0103-prefill-logits` | training | 201 / 20 | 240 |
| `0104-prefill-lora` | training | 16 / 0 | 25 |
| `0105-prefill-not-sampled` | training | 38 / 8 | 55 |
| `0119-issue-observer-hook` | training | 12 / 0 | 20 |
| `0201-lora-set-weights` | serving | 75 / 0 | 85 |
| `0201-narrow-head-gate` | serving | 6 / 3 | 15 |

## The audit and its self-test

`ci/patch_audit.sh [root]` reads the table in `docs/UPSTREAM.md` and every patch, and fails (exit
1) when:

- a patch is missing the first two lines, or they are not `Concern:` and `Upstream:`;
- the patch, set, concern, status or size disagrees with its row, or the row is missing;
- the size exceeds the budget — it then prints the patch's `git apply --stat` and the numbers;
- a serving patch touches a file a training patch introduced (`new file mode` / `+++ b/`), because
  `-DR97_PATCH_SET=serving` must stand alone.

`tests/cmake/patch_audit_test.sh` is registered as the `patch_audit` host test. It copies
`patches/` and `docs/UPSTREAM.md` into a temporary tree and checks that the clean copy passes, then
that the audit fails on a copy where one patch has grown past its budget (printing that patch's
diff stat), on a patch with no `Concern:` line, on a serving patch touching a training file, and on
a missing row. It prints `patch_audit_test: ok` when every case behaves.

## Skipping a patch

`-DR97_PATCH_SKIP=<number>` leaves out every patch whose file name starts with that number, and
configuring fails if no patch matches. The stamp that decides whether the patched radiance must be
re-copied includes the skip and each patch's SHA-1, so editing a patch or changing the skip
re-applies it. A build with `-DR97_PATCH_SKIP=0102` has the observer and no replay: the two
`layer_replay_*` tests print, through `tests/support/self_check.h`'s `skip()`,

```
[host] layer_replay_keeps_every_boundary_and_is_bit_identical ... skipped (the replay patch 0102 is not in this build: -DR97_PATCH_SKIP took it out (RAD_REPLAY_BLOCK is undefined))
[host] layer_replay_runs_in_reverse_order ... skipped (the replay patch 0102 is not in this build: -DR97_PATCH_SKIP took it out (RAD_REPLAY_BLOCK is undefined))
```

and do not fail. In the full build the same two names run and pass, with no `threw` line.

## Checks run

- `./ci/patch_audit.sh` and `./tests/cmake/patch_audit_test.sh "$PWD"`: pass on this tree.
- `cmake -B build -G Ninja -DR97_WITH_HIP=OFF && cmake --build build`: builds clean.
- `cmake -B build-skip -G Ninja -DR97_WITH_HIP=OFF -DR97_PATCH_SKIP=0102`, then
  `./build-skip/r97_host_tests`: the two replay tests are skipped by name (see below); the rest of
  the host tier behaves as it does on the unmodified tree (see "Unrelated failures").
- `ctest -L host` on both builds runs `patch_audit`, `patch_sets` and the header tests green; it
  reports `host_tests` failed for the 23 pre-existing names, not for either replay test, which run
  and pass in the full build and skip in the `0102`-less build. The tier is therefore not green in
  this sandbox, on this tree or on the baseline.
- A fresh copy of `radiance` with `0101` and `0102` applied in order with `patch -p1 -F0`: both
  apply cleanly.

## Unrelated failures, seen and left alone

Two things outside this issue showed up while running the tier; both are reported here rather than
fixed, except the first, which had to be fixed for the tier to run at all.

1. **`tests/host/loss_ce_objective_loss.cpp` read and wrote past its buffers.** The sequence's ids
   run to 5 but the test allocated a 4-wide vocabulary, so the windowed `ce_sum` wrote
   `grad[4*4+5]` into a 20-float vector. It corrupted the heap and aborted the tier at the second
   test. The widths are now 7, which covers every id the batch uses; no expectation is weakened.
   This is a test-only fix, unrelated to the patches, and is in its own commit.
2. **The host tier has 23 named failing tests in this sandbox** (24 failure lines; one test hits
   two checks), in the fixtures and quant-kernel area: `fixture_container_carries_its_chat_template`
   and the other fixture tests fail to open a container whose path has been overwritten by a heap
   corruption that starts after `adjoint_quant`'s `had_quant_act_i8g` rows. A baseline build of
   `origin/main` with only the loss-test fix above (no `0101`/`0102`/tape changes) fails the **same
   23 names**, so this is not caused by this audit. The corruption is layout-dependent: the
   `0102`-less build loses one of the 23 (`fixture_container_carries_its_chat_template` passes
   there) and reports 24 failure lines, the same family. The environment lacks a device build
   (`R97_WITH_HIP=OFF`) and `RADIANCE_HOME` is unset; setting it to the build's `radiance_home`
   fixes the `libref` load failures but not the fixture corruption. Left alone: the corruption is
   in another card's test area.
3. The two serving patches both start with the number `0201`, so `-DR97_PATCH_SKIP=0201` leaves
   out both. The skip matches the leading digits by design (the number names a topic, not a file);
   it is noted in `docs/UPSTREAM.md`.
