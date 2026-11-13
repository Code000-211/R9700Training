// tests/host/replay_vram_store.cpp -- the boundary store on the card: the plan that prices it
// before the step, the refusal with the numbers when it does not fit, and a layer re-run that
// reads it back, bit for bit.
//
// replay_layers.cpp keeps the residual store in host memory. This one keeps it on the card: one
// device allocation for the whole store, a keep() per boundary that is a stream-ordered copy
// behind the op the observer just saw, and a restore() that is a device-to-device copy the
// replayed block reads from. The step's stream is the lane 0 stream the engine issues on;
// patches/training/0122 exposes it as rad_ctx_cur_stream. The host backend is a full backend, so
// the whole path runs in this host-tier test without a card; the card tier drives it on the 27B
// container, where the store is 65 x 4096 x 5120 x 2 bytes, and the memory plan is what keeps a
// run that does not fit from failing in the middle of a step.
//
// ResidualPlan and the card-backed ResidualStore need no patch, so the plan's arithmetic and the
// refusal run in every build. The replay test needs the issue observer and the replay block
// (patches/training/0101 and 0102), the context stream (0122) and the tiny prefill fixture; a
// build without any of them skips it by name, not fails it.
#include "support/replay_step.h"
#include "support/test.h"

#include "rad_device.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// The plan's arithmetic needs no patch: ResidualPlan is part of r97/replay.h in every build.
R97_TEST(residual_plan_counts_the_store_and_refuses_with_the_numbers) {
    r97::ResidualPlan empty;
    R97_CHECK(empty.boundary_bytes() == 0);
    R97_CHECK(empty.store_bytes() == 0);
    R97_CHECK(empty.fits(0));

    // The 27B shape: a trunk of 64 layers, so 65 boundaries, 4096 tokens per step, 5120 wide.
    r97::ResidualPlan plan;
    plan.boundaries = 65;
    plan.tokens     = 4096;
    plan.hidden     = 5120;
    R97_CHECK(plan.boundary_bytes() == 41943040);
    R97_CHECK(plan.store_bytes() == 2726297600);
    R97_CHECK(plan.fits(2726297600));
    R97_CHECK(!plan.fits(2726297599));

    const std::string refusal = plan.refusal(1ll << 30);
    // The refusal carries the numbers: what the store costs and what the budget is.
    R97_CHECK(refusal.find("2726297600") != std::string::npos);
    R97_CHECK(refusal.find("1073741824") != std::string::npos);

    // A plan with a second copy at one boundary: the block that starts there reads another
    // boundary-sized buffer, and the store pays for one more copy there.
    r97::ResidualPlan two;
    two.boundaries = 2;
    two.tokens     = 8;
    two.hidden     = 128;
    two.extra      = {0, 1};
    R97_CHECK(two.extras() == 1);
    R97_CHECK(two.store_bytes() == 3 * 2048);
    R97_CHECK(two.fits(6144));
    R97_CHECK(!two.fits(6143));
}

// The refusal happens before the allocation: a store that does not fit the budget throws with
// the numbers and leaves nothing to free. The stream is the host backend's, created directly,
// since no engine is needed for what is being checked.
R97_TEST(vram_store_refuses_before_it_allocates) {
    RadStream stream = nullptr;
    R97_CHECK(rad_stream_create(&stream, 0) >= 0);

    r97::ResidualPlan plan;
    plan.boundaries = 2;
    plan.tokens     = 8;
    plan.hidden     = 128;
    const int64_t want = plan.store_bytes();
    R97_CHECK(want == 4096);

    bool refused = false;
    try {
        r97::ResidualStore store(stream, plan, want - 1);
    } catch (const std::runtime_error& e) {
        refused = true;
        const std::string why = e.what();
        R97_CHECK(why.find("4096") != std::string::npos);
        R97_CHECK(why.find("4095") != std::string::npos);
    }
    R97_CHECK(refused);

    // A budget that fits constructs, and the last copy takes the allocation with it.
    {
        r97::ResidualStore store(stream, plan, want);
        R97_CHECK(store.on_device());
        R97_CHECK(store.boundary_bytes() == 2048);
        R97_CHECK(store.bytes() == want);
        R97_CHECK(store.tokens() == 8);
        R97_CHECK(store.hidden() == 128);
    }
    rad_stream_destroy(stream);
}

#if defined(R97_REPLAY) && R97_FIXTURE_PREFILL && defined(RAD_CTX_STREAM)

namespace {

// One directory of the test's own, removed with the test: a step that keeps its engine alive
// owns the checkpoint and container it writes (tests/support/replay_step.h does the same).
struct TempDir {
    std::string path;

    explicit TempDir(const char* template_) {
        char name[64];
        std::snprintf(name, sizeof(name), "%s", template_);
        if (::mkdtemp(name) == nullptr)
            throw std::runtime_error("vram store test: cannot make a temporary directory");
        path = name;
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
};

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

R97_TEST(vram_store_replay_is_bit_identical) {
    const auto spec = r97::detail::qwen35_spec();
    const int64_t layers = spec.layers;
    const int64_t hidden = spec.hidden;
    std::vector<std::vector<int32_t>> sequences = three_sequences();
    int64_t tokens = 0;
    for (const std::vector<int32_t>& s : sequences) tokens += (int64_t)s.size();
    const int64_t elements = tokens * hidden;

    TempDir dir("/tmp/r97-replay-vram-XXXXXX");
    r97::detail::write_checkpoint(dir.path);
    const std::string container = dir.path + "/tiny_qwen35_bf16.rad";
    const std::string home = r97::detail::radiance_home();
    R97_CHECK(!home.empty());
    const std::string cmd = r97::detail::shell_quote(r97::detail::rad_convert()) + " " +
                            r97::detail::shell_quote(dir.path) + " -o " +
                            r97::detail::shell_quote(container) + " --kernels libavx,libref" +
                            " --home " + r97::detail::shell_quote(home);
    R97_CHECK(std::system(cmd.c_str()) == 0);

    rad::Config cfg;
    cfg.model            = container;
    cfg.radiance_home    = home;
    cfg.kernel_hierarchy = {"libavx", "libref"};
    cfg.n_spec           = 0;
    cfg.max_tok          = 512;
    cfg.max_seqs         = 8;
    cfg.max_ctx          = 512;
    cfg.prefill_prompts  = std::move(sequences);

    std::unique_ptr<rad::Engine> engine;
    r97::Tape tape;
    std::unique_ptr<r97::BoundaryRecorder> recorder;

    engine.reset(new rad::Engine(std::move(cfg)));
    R97_CHECK(engine->start() >= 0);
    RadCtx* ctx = engine->issue_ctx();
    R97_CHECK(ctx != nullptr);
    tape.attach(ctx);

    // The step's stream: the lane 0 stream the engine issues on (patches/training/0122 exposes
    // it as rad_ctx_cur_stream), so the store's copies order behind the ops the observer sees.
    RadStream stream = rad_ctx_cur_stream(ctx);
    R97_CHECK(stream != nullptr);

    // A fixed 1 GiB budget: the host backend's reported free VRAM is not a meaningful budget, and
    // the refusal has its own test.
    const int64_t budget = 1ll << 30;
    r97::ResidualPlan plan;
    plan.boundaries = layers + 1;
    plan.tokens     = tokens;
    plan.hidden     = hidden;
    R97_CHECK(plan.boundary_bytes() == elements * 2);
    R97_CHECK(plan.store_bytes() == (layers + 1) * elements * 2);
    R97_CHECK(plan.fits(budget));

    r97::ResidualStore store(stream, plan, budget);
    recorder.reset(new r97::BoundaryRecorder(tape, std::move(store)));

    R97_CHECK(engine->serve(std::make_unique<rad::StallWatch>("vram store replay", 60000)) >= 0);

    const r97::ResidualStore& kept = recorder->store();
    R97_CHECK(kept.on_device());
    R97_CHECK(kept.boundaries() == layers + 1);
    R97_CHECK(kept.tokens() == tokens);
    R97_CHECK(kept.hidden() == hidden);
    R97_CHECK(kept.bytes() == plan.store_bytes());
    std::printf("vram store: %lld boundaries of %lld tokens x %lld hidden x 2 bytes = %lld "
                "bytes on the card\n",
                (long long)kept.boundaries(), (long long)tokens, (long long)hidden,
                (long long)kept.bytes());

    recorder->detach();

    std::vector<size_t> block_ops((size_t)layers);
    for (const r97::Tape::Block& b : tape.blocks())
        block_ops[(size_t)b.index] = b.end - b.begin;

    // A copy of the store shares the one allocation on the card; the replay takes its own copy
    // through its by-value parameter.
    const r97::ResidualStore shared = kept;
    R97_CHECK(shared.on_device());
    R97_CHECK(shared.bytes() == plan.store_bytes());
    r97::LayerReplay replay(*engine, tape, shared);
    R97_CHECK(replay.layers() == layers);
    R97_CHECK(replay.store().on_device());
    R97_CHECK(replay.store().bytes() == plan.store_bytes());

    for (int64_t layer = 0; layer < layers; ++layer) {
        const void* residual = replay.residual(layer + 1);
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

        // Pull the store's copy of the boundary back over the wire and compare it with what the
        // replayed block just wrote.
        std::vector<uint8_t> host((size_t)plan.boundary_bytes());
        replay.store().read(layer + 1, host.data());
        R97_CHECK(std::memcmp(host.data(), snapshot.data(), (size_t)plan.boundary_bytes()) == 0);
        // The replayed block wrote its output into the same step buffer the forward used: the
        // boundary's source, not a copy of it.
        R97_CHECK(std::memcmp(host.data(), residual, (size_t)plan.boundary_bytes()) == 0);
        R97_CHECK(replay.residual(layer + 1) == residual);
    }
}

#else

// A build without the patch (or without the tiny prefill fixture) cannot replay a layer from the
// card's store. Say so by name so the host tier's output distinguishes a skip from a test that
// passed or never existed. test.h is already included here, so take back R97_CHECK before
// self_check.h redefines it; this branch needs only skip().
#undef CHECK
#undef R97_CHECK
#include "support/self_check.h"

namespace {

struct VramStoreSkips {
    VramStoreSkips() {
#if !defined(RAD_ISSUE_OBSERVER)
        constexpr const char* why = "the issue observer patch 0101 is not in this build";
#elif !defined(RAD_REPLAY_BLOCK)
        constexpr const char* why = "the replay patch 0102 is not in this build: "
                                    "-DR97_PATCH_SKIP took it out (RAD_REPLAY_BLOCK is undefined)";
#elif !defined(RAD_CTX_STREAM)
        constexpr const char* why = "the context-stream patch 0122 is not in this build: "
                                    "-DR97_PATCH_SKIP took it out (RAD_CTX_STREAM is undefined)";
#else
        constexpr const char* why = "the tiny prefill fixture is not in this build";
#endif
        ::r97test::skip("vram_store_replay_is_bit_identical", why);
    }
} vram_store_skips;

}  // namespace

#endif  // R97_REPLAY && R97_FIXTURE_PREFILL && RAD_CTX_STREAM
