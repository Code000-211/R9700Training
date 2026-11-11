// tests/host/replay_layers.cpp -- the residual at every layer boundary, and one layer re-run from
// it, bit for bit.
//
// The fixture is the tiny bf16 Qwen3.5: four blocks, so five boundaries. The step holds three
// sequences of different lengths, as one prefill batch. The forward records the tape and the
// boundary store; a LayerReplay then restores a boundary, clears the block's recurrent state and
// re-issues only that block. The residual the block leaves must be the bytes the store took from
// the forward's same buffer, both in the forward order and in reverse.
//
// The block brackets and the issue observer are training patches on radiance
// (patches/training/0101 and 0102); a serving build applies neither, so this test compiles to
// nothing there. A build with 0102 left out (-DR97_PATCH_SKIP=0102) has the observer but no replay:
// the two tests are skipped by name, not failed.
#include "support/replay_step.h"

#if defined(R97_REPLAY) && R97_FIXTURE_PREFILL

// radiance's tests/rad_test.h, which the fixture header brings in, defines CHECK for its own
// TEST() registry; this file checks with R97_CHECK only, so the alias is dropped rather than
// warned about.
#undef CHECK
#include "support/self_check.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

std::vector<int32_t> prompt(size_t n, int32_t first) {
    std::vector<int32_t> ids(n);
    for (size_t i = 0; i < n; ++i) ids[i] = (first + 7 * (int32_t)i) % 256;
    return ids;
}

// Three sequences of different lengths, packed into one prefill step.
std::vector<std::vector<int32_t>> three_sequences() {
    return {prompt(8, 3), prompt(5, 41), prompt(13, 97)};
}

// Copy the residual out of `content` when it is the buffer at `residual`: the snapshot the caller
// keeps, taken after the op issued and before the next op can reuse it.
void snap_residual(const std::vector<r97::OperandContent>& content, const void* residual,
                   int64_t elements, std::vector<uint16_t>& out) {
    for (const r97::OperandContent& c : content) {
        if (c.data == residual && c.bytes == (size_t)elements * 2) {
            const uint16_t* p = static_cast<const uint16_t*>(c.data);
            out.assign(p, p + elements);
        }
    }
}

}  // namespace

static void layer_replay_keeps_every_boundary_and_is_bit_identical() {
    r97::ReplayStep step;
    step.run(three_sequences());

    const int64_t layers = step.layers;
    const int64_t hidden = step.hidden;
    const int64_t elements = step.tokens * hidden;

    R97_CHECK(step.tape.blocks().size() == (size_t)layers);
    R97_CHECK(step.store.boundaries() == layers + 1);
    R97_CHECK(step.store.boundary_bytes() == elements * 2);
    R97_CHECK(step.store.bytes() == (layers + 1) * elements * 2);
    std::printf(
        "replay store: %lld boundaries of %lld tokens x %lld hidden x 2 bytes = %lld bytes\n",
        (long long)step.store.boundaries(), (long long)step.tokens, (long long)hidden,
        (long long)step.store.bytes());

    std::vector<size_t> block_ops((size_t)layers);
    for (const r97::Tape::Block& b : step.tape.blocks())
        block_ops[(size_t)b.index] = b.end - b.begin;

    r97::LayerReplay replay(*step.engine, step.tape, step.store);
    R97_CHECK(replay.layers() == layers);
    R97_CHECK(replay.store().bytes() == step.store.bytes());

    for (int64_t layer = 0; layer < layers; ++layer) {
        const void* residual = step.store.source(layer + 1);
        R97_CHECK(residual != nullptr);

        int64_t seen = 0;
        std::vector<uint16_t> snapshot;
        replay.replay(layer, [&](const r97::OpRecord& op,
                                 const std::vector<r97::OperandContent>& content) {
            ++seen;
            R97_CHECK(op.block == layer);
            snap_residual(content, residual, elements, snapshot);
        });

        R97_CHECK(seen == (int64_t)block_ops[(size_t)layer]);
        R97_CHECK(snapshot.size() == (size_t)elements);
        R97_CHECK(std::memcmp(snapshot.data(), step.store.at(layer + 1),
                              (size_t)step.store.boundary_bytes()) == 0);
        // The replayed block wrote its output into the same step buffer the forward used: the
        // boundary's source, not a copy of it.
        R97_CHECK(std::memcmp(residual, step.store.at(layer + 1),
                              (size_t)step.store.boundary_bytes()) == 0);
        R97_CHECK(replay.residual(layer + 1) == residual);
    }
}

static void layer_replay_runs_in_reverse_order() {
    r97::ReplayStep step;
    step.run(three_sequences());

    const int64_t layers = step.layers;
    const int64_t elements = step.tokens * step.hidden;
    r97::LayerReplay replay(*step.engine, step.tape, step.store);

    for (int64_t layer = layers - 1; layer >= 0; --layer) {
        const void* residual = step.store.source(layer + 1);
        R97_CHECK(residual != nullptr);

        std::vector<uint16_t> snapshot;
        replay.replay(layer, [&](const r97::OpRecord&,
                                 const std::vector<r97::OperandContent>& content) {
            snap_residual(content, residual, elements, snapshot);
        });

        R97_CHECK(snapshot.size() == (size_t)elements);
        R97_CHECK(std::memcmp(snapshot.data(), step.store.at(layer + 1),
                              (size_t)step.store.boundary_bytes()) == 0);
        R97_CHECK(std::memcmp(residual, step.store.at(layer + 1),
                              (size_t)step.store.boundary_bytes()) == 0);
    }
}

// Run before tests/support/test.h's runner and before the static destructors that the runner
// comes after: this test drives a live engine, and the fixture that chatdata's self-check creates
// during static initialisation is a function-local static whose contents are destroyed before an
// exit-time R97_TEST can use them (docs/BLOCKS.md records the same failure). Registering this
// handler after that static's own means it runs before the static goes away.
namespace {

void run_replay_tests() {
    ::r97test::run("layer_replay_keeps_every_boundary_and_is_bit_identical",
                   layer_replay_keeps_every_boundary_and_is_bit_identical);
    ::r97test::run("layer_replay_runs_in_reverse_order", layer_replay_runs_in_reverse_order);
}

struct ReplayAtExit {
    ReplayAtExit() {
        if (std::atexit(&run_replay_tests) != 0)
            std::fprintf(stderr, "replay: could not register the runner\n");
    }
} replay_at_exit;

}  // namespace

#else

// A build without the patch (or without the tiny prefill fixture) cannot replay a layer. Say so by
// name so the host tier's output distinguishes a skip from a test that passed or never existed.
#undef CHECK
#include "support/self_check.h"

namespace {

struct ReplaySkips {
    ReplaySkips() {
#if !defined(RAD_ISSUE_OBSERVER)
        constexpr const char* why = "the issue observer patch 0101 is not in this build";
#elif !defined(RAD_REPLAY_BLOCK)
        constexpr const char* why = "the replay patch 0102 is not in this build: "
                                    "-DR97_PATCH_SKIP took it out (RAD_REPLAY_BLOCK is undefined)";
#else
        constexpr const char* why = "the tiny prefill fixture is not in this build";
#endif
        ::r97test::skip("layer_replay_keeps_every_boundary_and_is_bit_identical", why);
        ::r97test::skip("layer_replay_runs_in_reverse_order", why);
    }
} replay_skips;

}  // namespace

#endif  // R97_REPLAY && R97_FIXTURE_PREFILL
