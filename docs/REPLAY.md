# Layer replay

The memory scheme of the backward: the forward keeps only the residual stream at the layer
boundaries, and the backward re-runs one layer at a time from its stored input. This is
`include/r97/replay.h` and `src/replay.cpp`; the radiance hook it needs is
`patches/training/0102-replay-block.patch`.

## The store

A trunk of `L` blocks has `L + 1` boundaries. Boundary 0 is what ran before the first block, the
embeddings' output; boundary `i` is block `i - 1`'s output, so boundary `L` is the last layer's
residual. One boundary is `tokens * hidden * 2` bytes (bf16, on the host or on the card): the
step's sequences' tokens together, one row each.

Not every block reads its input as the residual alone. In the qwen35 int8 trunk (the `qwen35_fp8`
arch the 27B container serves) every block after the first does: its first norm is a fused add,
which reads the previous block's output and the residual and writes their sum, so such a block's
input state is two boundary-sized buffers. The plan says which boundaries carry that second copy
(`ResidualPlan::extra`, one entry per boundary), and the store keeps it beside the residual; a
replay restores both before it runs the block, so the replayed layer reads exactly what the
forward's layer read, in either order. The first block and the whole bf16 trunk need no second
copy.

`ResidualStore` copies each boundary out of the step's own buffer, in order from zero; a gap or a
repeat throws, and so does a null buffer. It also keeps the address each boundary was copied from,
because a replay has to write its input back into that same buffer before it runs the block.
`bytes()` is the copies kept, each `tokens * hidden * 2`: for the bf16 fixture (4 layers, 3
sequences of 8, 5 and 13 tokens, hidden 128) that is `5 * 26 * 128 * 2 = 33280` bytes.

The same store lives on the card: `ResidualStore(stream, plan, budget)` makes one device
allocation, sized by a `ResidualPlan`, keeps each boundary as a stream-ordered device copy on
the step's lane, and restores it the same way, so a keep orders behind the op the observer just
saw and a restore ahead of the replayed block's reads. Copies of the store share the one
allocation, and the last copy frees it.

## Memory plan

`ResidualPlan` prices the store before the step: `layers + 1` boundaries of `tokens * hidden * 2`
bytes each, plus one copy per boundary that carries a second one, so the store is
`(layers + 1 + extras) * tokens * hidden * 2` bytes. For the 27B model at 4096 tokens per step,
with a second copy at every boundary a block after the first starts at (65 boundaries and 63
second copies, the last boundary having no block after it): `128 x 4096 x 5120 x 2 = 5368709120`
bytes, exactly 5 GiB. A run that does not fit its budget is refused before it starts: the
constructor throws `std::runtime_error` with those numbers, before the allocation.

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
write as boundary `block + 1`. The second copy, when the plan holds one at the boundary, is the
buffer the block *at* that boundary reads as its second input: a block whose first boundary-sized
operand is a read names that buffer, and the recorder then keeps the block's last write of the
same name -- the value that buffer holds when the block after it takes it. The last write overall
is not enough, because a block's last op can land in a boundary-sized scratch buffer the next block
never reads; the fixture that first proved the walk hid that, every one of its blocks ending in the
write the next block reads. Only the first boundary has no read behind it; there the recorder falls
back to the last such write, and the block that reads it checks the copy came from the same
buffer. The recorder cross-checks the plan against the tape in both directions: it throws when the
plan expects a copy the block before the boundary never wrote, when a block reads a second
boundary-sized buffer from a boundary the plan did not keep, and when the copy kept there is not
the buffer the block reads. Detach before a replay: the replay replaces those two hooks.

## Replaying one layer

`LayerReplay` copies the tape's op records and block ranges at construction and detaches the tape.
`replay(layer)` attaches that same tape again for the replayed block, with the replay's own hooks:

1. copies stored boundary `layer` into the buffer the forward wrote boundary `layer` from -- and,
   when the plan holds a second copy at the boundary, that copy into the buffer it came out of,
   so the block starts from its full stored input;
2. attaches the tape, whose issue hook is now the replay's, and calls `Engine::replay_block(layer)`,
   which clears the stateful groups' slots for the step's sequences (the batch's own
   `state_index`, the same slots the step used) and runs the step with the replay block set;
3. `qwen35_bf16`'s step, with the replay block set, skips the embeddings and the head and calls the
   recorded layer's `run_layer()` alone, calibration taps included;
4. the tape's issue hook calls the caller's `after` once per op, in the forward's order, before the
   next op can reuse a transient buffer. Nothing written by an earlier op may be assumed to survive
   that call.

The gated-delta-net state must start at zero, not where the forward left it; clearing the slots
does that. The slots are the group's own: `state_ptr` takes a layer index within the group, and a
group holds one cache per bound layer, not per model layer, so the engine maps the absolute block
through the group's binding (`KVPoolBinding::layer_slot`) before it clears anything. Without that
map a group whose layers are not the model's first *n* hands back a null pointer and never has its
state cleared: the layer's first replay starts from the zero the recording left, passes, and writes
its own final state into the slot -- which the layer's second replay then starts from. The paged
key-value pages need no reset: the batch's used counts and slot map are the step's start, and the
block's own `kv_store` overwrites every slot its attention then reads. What the first forward left
in those pages is the same block's own output for the same tokens, and a correct replay writes the
same bytes before reading them.

The engine must outlive the replay, and the recorded step must be the engine's last live prefill.

## Building it

The code is guarded by `RAD_ISSUE_OBSERVER` (patch 0101) and `RAD_REPLAY_BLOCK` (patch 0102), and
the card store needs the step's stream, `RAD_CTX_STREAM` (patch 0122), so a serving build
compiles it away and has no replay rather than one that would run the whole model. All three
patches are training-set patches and apply in name order; 0102 adds the replay block and the
architecture's agreement to honour it, and no observer of its own. A build with
`-DR97_PATCH_SKIP=0102` has the observer and the brackets but no replay: the host replay tests
skip by name rather than fail. A build with `-DR97_PATCH_SKIP=0122` has the replay but no stream
accessor: the vram-store and card-replay tests skip by name, and the plan's arithmetic and
refusal still run. `docs/UPSTREAM.md` carries each patch's budget and `ci/patch_audit.sh`
enforces it.

## The host test

`tests/host/replay_layers.cpp`, with `tests/support/replay_step.h`: one prefill step of three
sequences of 8, 5 and 13 tokens, the boundary store, then every layer replayed forward and in
reverse. For each layer the tape's op count must be the block's count, the snapshot after every op
must be the stored next boundary, and the final residual buffer must hold the next boundary's
bytes.

`ReplayStep` writes its own fixture and finds the plugin home and `rad-convert` beside the
executable, rather than calling `r97::fixture()`: a step that keeps its engine and its tape alive
owns its directory and removes it with the step. The two checks are plain `R97_TEST`s, which
`tests/host/main.cpp` runs with every static alive. `R97_RAD_CONVERT` and `RADIANCE_HOME` may name
the converter and the plugin home, or the test finds them under `<build>/radiance-build/`.

The host proof of the card path is `tests/host/replay_vram_store.cpp`: the host backend is a full
backend, so the VRAM store's keep, restore and replay, the plan's arithmetic and the
pre-allocation refusal run in the host tier without a card. The card proof is the
`R97_MODEL_TEST` case in `tests/gpu/replay_layers.cpp`, on the real 27B container in the device
tier: every layer replayed from the VRAM store, forward and in reverse, bit-identical to the
stored next boundary, with the plan checked against the card's free VRAM and the numbers
printed.

The int8 trunk gets its own host test, `tests/host/replay_w8a8.cpp`: the 27B container is the
w8a8 recipe served by `qwen35_w8a8` (arch `qwen35_fp8`), and the replay branch 0102 adds to that
architecture is the one the card runs. The same tiny checkpoint converted to the w8a8 descriptor
(four blocks, `tests/support/fixture_w8a8.h`) is prefilled through the int8 path and every layer
replayed, forward and in reverse: bit-identical, with the second copies the plan holds at every
boundary after the first doing their work in the reverse order, where no preceding replay
recomputes the values.

## Where the production walk builds the plan

The host join's reverse walk (`src/step_host.cpp`) replays each block from this store, so it
builds the plan from the container's metadata: `meta.layers + 1` boundaries, and a second copy at
every boundary after the first when the container's `input_basis` is `fwht128` (the rotated int8
trunk). The recorder cross-checks the plan against the tape both ways, so a trunk whose blocks
read a second boundary input is not walked with a plan that keeps no copy for it: before that
check, the walk fed those blocks the buffer's stale contents, and the int8 agreement it scored is
what the restored copy recovered (`docs/STEPS-HOST.md`).
