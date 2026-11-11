# Layer replay

The memory scheme of the backward: the forward keeps only the residual stream at the layer
boundaries, and the backward re-runs one layer at a time from its stored input. This is
`include/r97/replay.h` and `src/replay.cpp`; the radiance hook it needs is
`patches/training/0102-replay-block.patch`.

## The store

A trunk of `L` blocks has `L + 1` boundaries. Boundary 0 is what ran before the first block, the
embeddings' output; boundary `i` is block `i - 1`'s output, so boundary `L` is the last layer's
residual. One boundary is `tokens * hidden * 2` bytes (bf16 on the host): the step's sequences'
tokens together, one row each.

`ResidualStore` copies each boundary out of the step's own buffer, in order from zero; a gap or a
repeat throws, and so does a null buffer. It also keeps the address each boundary was copied from,
because a replay has to write its input back into that same buffer before it runs the block.
`bytes()` is `boundaries() * tokens * hidden * 2`, and the host test reports it:
for the bf16 fixture (4 layers, 3 sequences of 8, 5 and 13 tokens, hidden 128) that is
`5 * 26 * 128 * 2 = 33280` bytes.

The bytes are the ones the model actually computed: the checkpoint as loaded, with any adapter
that was wrapped in at the time, not a re-computation from weights. The test compares a replay to
these bytes only, so what it proves is that the replay is bit-identical to the forward on the same
step, not that either matches an oracle.

## Recording

`BoundaryRecorder` subscribes to the tape's issue hook and block-bracket hook, so the tape remains
the one holder of the context's observer; it is constructed with the attached tape. The observer
runs after each op, so an operand is the residual when it is written, names a declared buffer,
and is exactly `tokens * hidden * 2` bytes. The first one seen names the buffer, and later
operands must carry the same name, so a norm, a calibration tap of the same size or a KV write
(whose tape identity is synthetic) is not mistaken for the residual. At a block begin the
recorder remembers the last residual write for boundary 0, and at the block end it keeps that
write as boundary `block + 1`. Detach before a replay: the replay replaces those two hooks.

## Replaying one layer

`LayerReplay` copies the tape's op records and block ranges at construction and detaches the tape.
`replay(layer)` attaches that same tape again for the replayed block, with the replay's own hooks:

1. copies stored boundary `layer` into the buffer the forward wrote boundary `layer` from, so the
   block starts from its stored input;
2. attaches the tape, whose issue hook is now the replay's, and calls `Engine::replay_block(layer)`,
   which clears the stateful groups' slots for the step's sequences (the batch's own
   `state_index`, the same slots the step used) and runs the step with the replay block set;
3. `qwen35_bf16`'s step, with the replay block set, skips the embeddings and the head and calls the
   recorded layer's `run_layer()` alone, calibration taps included;
4. the tape's issue hook calls the caller's `after` once per op, in the forward's order, before the
   next op can reuse a transient buffer. Nothing written by an earlier op may be assumed to survive
   that call.

The gated-delta-net state must start at zero, not where the forward left it; clearing the slots
does that. The paged key-value pages need no reset: the batch's used counts and slot map are the
step's start, and the block's own `kv_store` overwrites every slot its attention then reads. What
the first forward left in those pages is the same block's own output for the same tokens, and a
correct replay writes the same bytes before reading them.

The engine must outlive the replay, and the recorded step must be the engine's last live prefill.

## Building it

The code is guarded by `RAD_ISSUE_OBSERVER` (patch 0101) and `RAD_REPLAY_BLOCK` (patch 0102), so a
serving build compiles it away and has no replay rather than one that would run the whole model.
Both patches are training-set patches and apply in name order; 0102 adds the replay block and the
architecture's agreement to honour it, and no observer of its own. A build with
`-DR97_PATCH_SKIP=0102` has the observer and the brackets but no replay: the two host tests skip by
name rather than fail. `docs/UPSTREAM.md` carries each patch's budget and `ci/patch_audit.sh`
enforces it.

## The host test

`tests/host/replay_layers.cpp`, with `tests/support/replay_step.h`: one prefill step of three
sequences of 8, 5 and 13 tokens, the boundary store, then every layer replayed forward and in
reverse. For each layer the tape's op count must be the block's count, the snapshot after every op
must be the stored next boundary, and the final residual buffer must hold the next boundary's
bytes.

`ReplayStep` writes its own fixture and finds the plugin home and `rad-convert` beside the
executable instead of calling `r97::fixture()`, because that cache and `radtest::tmpdir()` are
function-local statics destroyed before the exit-time `R97_TEST` runner (docs/BLOCKS.md records
that defect). For the same reason this test registers its own `atexit` handler: it is registered
after those statics' own, so it runs before them and before the shared runner, while the engine's
globals are alive. The handler prints the same `[host]` lines and aborts on a failed check.

The shared host binary currently aborts in `test_loss_ce_objective_loss` during static
initialisation (docs/BLOCKS.md), before any handler runs; that failure predates this test. Run the
binary with that one self-check skipped, or run this test alone: build the same compile and link
the shared host target uses, with a trivial `int main() { return 0; }` in place of the test
binary's own main, and let the test's handler do the work. `R97_RAD_CONVERT` and `RADIANCE_HOME`
may name the converter and the plugin home, or the test finds them under `<build>/radiance-build/`.
