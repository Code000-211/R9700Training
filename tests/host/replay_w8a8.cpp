// tests/host/replay_w8a8.cpp -- layer replay on the w8a8_fwht128 int8 trunk, bit for bit.
//
// replay_layers.cpp proves the replay on the bf16 container, served by qwen35_bf16. The model
// that runs on the card is the w8a8 trunk: the same tiny checkpoint converted to the
// w8a8_fwht128 descriptor, served by qwen35_w8a8 (arch qwen35_fp8). patches/training/0102 teaches
// that architecture to honour the replay block the way qwen35_bf16 already does; this test proves
// the new branch without a card, on the int8 fixture, in both the forward and the reverse order.
//
// The fixture is the tiny int8 container of tests/support/fixture_w8a8.h: four blocks, so five
// boundaries, converted once per process. The step holds three sequences of different lengths, as
// one prefill batch. The forward records the tape and the boundary store; a LayerReplay then
// restores a boundary, clears the block's recurrent state and re-issues only that block. The
// residual the block leaves must be the bytes the store took from the forward's same buffer, both
// in the forward order and in reverse. The fixture's full-attention layer reads the previous
// block's output beside the residual (its first norm is a fused add), so the store keeps a second
// copy at the boundary that layer starts at; the reverse order is the case that needs it, since
// no preceding replay recomputes the value.
//
// The replay is a training path: an architecture must honour the block, and a serving build
// applies no training patch, so this test compiles to nothing there. A build with 0102 left out
// (-DR97_PATCH_SKIP=0102) has the observer but no replay: the two tests are skipped by name, not
// failed.
#include "support/replay_step.h"
#include "support/fixture_w8a8.h"

#if defined(R97_REPLAY) && R97_FIXTURE_PREFILL

#include "support/test.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
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

// One prefill step of the int8 fixture, engine and tape kept alive: the same start as
// tests/support/replay_step.h's ReplayStep::run, on the w8a8 container.
struct W8a8Step {
    std::unique_ptr<rad::Engine> engine;
    r97::Tape tape;
    std::unique_ptr<r97::BoundaryRecorder> recorder;
    r97::ResidualStore store;
    r97::ResidualPlan plan;
    int64_t tokens = 0;
    int64_t layers = 0;
    int64_t hidden = 0;

    W8a8Step() = default;
    ~W8a8Step() {
        if (recorder) recorder->detach();
        recorder.reset();
        tape.detach();
        engine.reset();
    }
    W8a8Step(const W8a8Step&) = delete;
    W8a8Step& operator=(const W8a8Step&) = delete;

    void run(std::vector<std::vector<int32_t>> sequences) {
        const std::string& container = r97::w8a8_fixture().container;
        tokens = 0;
        for (const std::vector<int32_t>& s : sequences) tokens += (int64_t)s.size();

        const auto spec = r97::detail::qwen35_spec();
        layers = spec.layers;
        hidden = spec.hidden;

        rad::Config cfg;
        cfg.model            = container;
        cfg.radiance_home    = r97::detail::radiance_home();
        cfg.kernel_hierarchy = {"libavx", "libref"};
        cfg.n_spec           = 0;
        cfg.max_tok          = 512;
        cfg.max_seqs         = 8;
        cfg.max_ctx          = 512;
        cfg.prefill_prompts  = std::move(sequences);

        engine.reset(new rad::Engine(std::move(cfg)));
        int s = engine->start();
        if (s < 0)
            throw std::runtime_error(std::string("w8a8 replay step: the engine could not start: ") +
                                     rad_strerror(s));
        RadCtx* ctx = engine->issue_ctx();
        if (!ctx) throw std::runtime_error("w8a8 replay step: the engine has no issue context");
        tape.attach(ctx);
        // The store: one residual copy per boundary, plus a second copy at every boundary after
        // the first: in the int8 trunk every block after the first reads the previous block's
        // output beside the residual (its first norm is a fused add); only the first block does
        // not.
        plan.boundaries = layers + 1;
        plan.tokens     = tokens;
        plan.hidden     = hidden;
        for (int64_t b = 0; b < plan.boundaries; ++b)
            plan.extra.push_back(b >= 1 && b < layers ? 1 : 0);
        recorder.reset(new r97::BoundaryRecorder(tape, r97::ResidualStore(plan)));

        s = engine->serve(std::make_unique<rad::StallWatch>("w8a8 replay", 60000));
        if (s < 0)
            throw std::runtime_error(std::string("w8a8 replay step: the prefill failed: ") +
                                     rad_strerror(s));
        store = recorder->store();
        recorder->detach();
        tape.detach();
    }
};

}  // namespace

// The w8a8 trunk honours the replay block: the residual the replayed block leaves, read out of
// the same step buffer the forward used, is the bytes the store took from the forward, for every
// layer in the forward order.
R97_TEST(layer_replay_on_the_int8_trunk_is_bit_identical) {
    W8a8Step step;
    step.run(three_sequences());

    const int64_t layers = step.layers;
    const int64_t hidden = step.hidden;
    const int64_t elements = step.tokens * hidden;

    R97_CHECK(step.tape.blocks().size() == (size_t)layers);
    R97_CHECK(!step.store.on_device());
    R97_CHECK(step.store.boundaries() == layers + 1);
    R97_CHECK(step.store.boundary_bytes() == elements * 2);
    R97_CHECK(step.store.bytes() == step.plan.store_bytes());
    std::printf("w8a8 replay store: %lld boundaries, %lld second copies, %lld tokens x %lld "
                "hidden x 2 bytes = %lld bytes\n",
                (long long)step.store.boundaries(), (long long)step.plan.extras(),
                (long long)step.tokens, (long long)hidden, (long long)step.store.bytes());

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

// The same replay, blocks in the reverse order: the int8 trunk's replayed layer is bit-identical
// when the layers run last-first, not only first-last.
R97_TEST(layer_replay_on_the_int8_trunk_runs_in_reverse) {
    W8a8Step step;
    step.run(three_sequences());

    const int64_t layers = step.layers;
    const int64_t elements = step.tokens * step.hidden;
    r97::LayerReplay replay(*step.engine, step.tape, step.store);
    R97_CHECK(replay.layers() == layers);
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

#else

// A build without the patch (or without the tiny prefill fixture) cannot replay a layer. Say so by
// name so the host tier's output distinguishes a skip from a test that passed or never existed.
#undef CHECK
#include "support/self_check.h"

namespace {

struct W8a8ReplaySkips {
    W8a8ReplaySkips() {
#if !defined(RAD_ISSUE_OBSERVER)
        constexpr const char* why = "the issue observer patch 0101 is not in this build";
#elif !defined(RAD_REPLAY_BLOCK)
        constexpr const char* why = "the replay patch 0102 is not in this build: "
                                    "-DR97_PATCH_SKIP took it out (RAD_REPLAY_BLOCK is undefined)";
#else
        constexpr const char* why = "the tiny prefill fixture is not in this build";
#endif
        ::r97test::skip("layer_replay_on_the_int8_trunk_is_bit_identical", why);
        ::r97test::skip("layer_replay_on_the_int8_trunk_runs_in_reverse", why);
    }
} w8a8_replay_skips;

}  // namespace

#endif  // R97_REPLAY && R97_FIXTURE_PREFILL
