// tests/gpu/replay_layers.cpp -- the layer replay on the card: the residual store lives in VRAM,
// and every layer re-runs from it bit-identically, in the forward order and in reverse.
//
// The case is replay_layers_card, a model case: the real container (the qwen3.8-27b-iq5r-s-v4
// one on disk, resolved through support/model_dir.h) on the device kernel tier, one prefill
// step of four short sequences. A BoundaryRecorder keeps the step's residual at every layer
// boundary in the card's VRAM (ResidualStore's device-backed construction, on the stream the
// step's ops issue on -- patches/training/0122), and a LayerReplay then re-runs every block of
// the step from the store, forward and reverse. For each replay, the residual the block leaves
// in the step's own buffer must be bit-identical to the boundary the forward stored at the same
// place, and the replay must issue exactly the ops the forward's block issued.
//
// The memory plan. One boundary is tokens * hidden * 2 bytes, and a trunk of L layers has L+1 of
// them. In this container's int8 trunk every block after the first reads the previous block's
// output beside the residual (its first norm is a fused add), so the plan keeps a second copy at
// every boundary a block after the first starts at: at the ticket's 4096 tokens the 27B (64
// layers, hidden 5120) holds 65 + 63 = 128 copies, 128 x 4096 x 5120 x 2 = 5368709120 bytes,
// exactly 5 GiB, in one VRAM
// allocation. The child prints the plan with its own numbers -- boundaries, the second copies,
// tokens x hidden x 2, the store's bytes, the card's free VRAM -- before allocating, and the
// store's device constructor refuses with the same numbers when the step does not fit, before
// the step runs.
//
// The child/parent split. The engine picks the card only through ROCR_VISIBLE_DEVICES, which the
// ROCR runtime reads once, at init, and the shared r97_gpu_tests binary runs every test in one
// process where the other gpu tests init ROCR before this one does. So no engine work runs in
// this process at all: the parent forks, and the child re-execs this same binary with
// R97_TEST_ONLY (only this test) and ROCR/HIP_VISIBLE_DEVICES already set -- a fresh process
// whose runtimes point at the card from the first device call. Every check runs in the child;
// the parent reads its exit code and re-skips (exit 77) when the child reports the container
// absent.
//
// The replay is a training path: it needs the block brackets and the issue observer
// (patches/training/0101), the replay block (0102), the context stream (0122) and the prefill
// mode (0103); a serving build applies none of the set, so the case compiles to nothing there
// and says so by name instead.
#include "support/gpu_guard.h"
#include "support/model_case.h"
#include "support/model_dir.h"
#include "support/test.h"

#include "r97/replay.h"

// R97_REPLAY is RAD_ISSUE_OBSERVER and RAD_REPLAY_BLOCK together (r97/replay.h); RAD_CTX_STREAM
// is patch 0122's marker; the prefill mode (0103) is asked for over the same include path the
// compiler reads, the way tests/support/fixture.h does.
#if defined(R97_REPLAY) && defined(RAD_CTX_STREAM) && __has_include("train_prefill.h")

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "engine.h"           // radiance's core/; Config, Engine, rad_strerror
#include "rad_device.h"       // the device layer: the card's props, the copies the test reads
#include "rad_runtime.h"      // RadCtx, RadStream; rad_ctx_cur_stream (patch 0122)
#include "radfile.h"          // the .rad reader: the container's layers and hidden, no card
#include "util/stallwatch.h"  // radiance's core/; the watch serve() takes

namespace {

namespace fs = std::filesystem;

const char* kTest    = "replay_layers_card";
const char* kRoleVar = "R97_RLC_ROLE";  // set in the re-exec'd child; the role that does the work

// ------------------------------------------------------------------ the small helpers

const std::string& self_exe() {
    static const std::string exe = [] {
        std::error_code ec;
        const fs::path p = fs::read_symlink("/proc/self/exe", ec);
        if (ec) throw std::runtime_error(std::string(kTest) + ": cannot read /proc/self/exe");
        return p.string();
    }();
    return exe;
}

// The plugin home: RADIANCE_HOME, else the build's radiance_home, which is beside the binary.
std::string radiance_home() {
    if (const char* e = std::getenv("RADIANCE_HOME"))
        if (*e) return e;
    std::error_code ec;
    const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    if (ec) return std::string();
    const std::string h = exe.parent_path().string() + "/radiance-build/radiance_home";
    return fs::exists(h + "/architectures", ec) ? h : std::string();
}

// Wait until the selected card reports the guard's need free. A previous test's child may have
// just exited and the driver can take a moment to return its VRAM to mem_info_vram_used; the
// guard would otherwise refuse on a stale reading of a card that is in fact free.
void wait_for_card_free(const r97test::GuardOptions& o) {
    const std::string sysfs = "/sys/class/drm";
    const std::string card  = r97test::card_for_bdf(sysfs, o.bdf);
    for (int i = 0; i < 120; ++i) {  // up to ~60 s
        try {
            const r97test::CardVram v = r97test::read_card_vram(sysfs, card);
            if (v.total - v.used >= o.need_vram) return;
        } catch (const std::exception&) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

// The tiny fixture's id scheme (tests/host/replay_layers.cpp): ids in the container's
// vocabulary, different per sequence.
std::vector<int32_t> prompt(size_t n, int32_t first) {
    std::vector<int32_t> ids(n);
    for (size_t i = 0; i < n; ++i) ids[i] = (first + 7 * (int32_t)i) % 256;
    return ids;
}

// ------------------------------------------------------------------ the parent path

// Fork, and have the child re-exec this binary running only this test, with the ROCR/HIP
// runtimes pointed at the card before their first init. The parent never opens the card or inits
// an engine: its ROCR was already init'd by the other gpu tests, so only a fresh child process
// can be pointed at the card. Every check runs in the child; here the exit code is what counts,
// and 77 re-skips the way ctest records it.
void run_in_child() {
    const char* dev = std::getenv("R97_DEVICE");
    const std::string device_num = dev && *dev ? dev : "0";

    const pid_t pid = ::fork();
    if (pid < 0)
        throw std::runtime_error(std::string(kTest) + ": fork: " + std::strerror(errno));
    if (pid == 0) {
        // The child: point both runtimes at the card before any init, mark this re-exec'd copy
        // as the one that does the work, and select this test so the fresh process runs it
        // alone. The re-exec'd process is fresh, so the runtimes read these for the first time.
        if (::setenv("HIP_VISIBLE_DEVICES", device_num.c_str(), 1) != 0 ||
            ::setenv("ROCR_VISIBLE_DEVICES", device_num.c_str(), 1) != 0 ||
            ::setenv("R97_TEST_ONLY", kTest, 1) != 0 || ::setenv(kRoleVar, "card", 1) != 0)
            std::_Exit(1);
        const std::string exe = self_exe();
        std::string arg0 = exe;  // execv's argv must be mutable
        char* const argv[2] = {arg0.data(), nullptr};
        ::execv(arg0.c_str(), argv);
        std::_Exit(1);  // execv failed
    }

    int status = 0;
    if (::waitpid(pid, &status, 0) != pid)
        throw std::runtime_error(std::string(kTest) + ": waitpid: " + std::strerror(errno));
    if (!WIFEXITED(status))
        throw std::runtime_error(std::string(kTest) + "'s child was killed by a signal; the card "
                                 "may be held by another run");
    const int code = WEXITSTATUS(status);
    if (code != 0) {
        // 77 (the container absent) re-skips; anything else is a failure.
        if (code == r97test::kSkip)
            r97test::skip("the model container is absent (see the child run of " +
                          std::string(kTest) + " above)");
        throw std::runtime_error(std::string(kTest) + "'s child exited " + std::to_string(code) +
                                 ": its stderr above says why; the card is R97_DEVICE, and the "
                                 "guard names the card's holders when the VRAM is short");
    }
}

// ------------------------------------------------------------------ the child path

// Guard the card, start the engine on the device tier, record the step's boundaries in the
// card's VRAM, and replay every layer from the store, forward and reverse.
void card_case(const std::string& container) {
    r97test::GuardOptions o;
    const char* de = std::getenv("R97_DEVICE");
    o.device = r97test::parse_device(de ? de : "0");
    // The number device_forward's card cases declare for this container (20 GiB: ~15.1 GiB of
    // weights, plus KV and the engine's arena), plus 1 GiB for the residual store, which is
    // plan.store_bytes() below (~306 MiB for this step's 245 tokens) with a little headroom.
    o.need_vram = int64_t(21) << 30;
    o.bdf       = r97test::hip_bdf(o.device);  // HIP is inited only now: ROCR_VISIBLE_DEVICES is set
    wait_for_card_free(o);
    r97test::DeviceGuard guard(o);

    const std::string home = radiance_home();
    if (home.empty())
        throw std::runtime_error(std::string(kTest) + ": no radiance home beside the build");

    // The step: four short sequences of different lengths, one prefill step. The ids are the
    // tiny fixture's scheme, all in the container's vocabulary.
    const std::vector<std::vector<int32_t>> sequences = {prompt(64, 3),  prompt(51, 41),
                                                         prompt(90, 97), prompt(40, 131)};
    int64_t tokens = 0;
    for (const std::vector<int32_t>& s : sequences) tokens += int64_t(s.size());
    R97_CHECK(tokens == 245);

    rad::Config cfg;
    cfg.model            = container;
    cfg.radiance_home    = home;
    cfg.kernel_hierarchy = {"libr4d", "libref"};  // the device tier, the fallback library after
    cfg.n_spec           = 0;                     // the prefill mode drafts nothing; state it
    cfg.mm_max_patches   = 0;                     // the vision tower is not built
    // The step is one prefill piece, and radiance's linear-state checkpoint interval also caps a
    // prefill chunk (src/forward.cpp turns it off for the same reason): training never resumes
    // from a checkpoint.
    cfg.checkpoint_interval = 0;
    cfg.max_tok             = 512;
    cfg.max_ctx             = 512;
    cfg.max_seqs            = 8;
    cfg.kv_cache_dtype      = "bf16";
    cfg.prefill_prompts     = sequences;

    rad::Engine engine(std::move(cfg));
    int s = engine.start();
    if (s < 0)
        throw std::runtime_error(std::string(kTest) + ": the engine could not start: " +
                                 rad_strerror(s));
    RadCtx* ctx = engine.issue_ctx();
    if (!ctx) throw std::runtime_error(std::string(kTest) + ": the engine has no issue context");

    // The container's shape, read the way `info` does (rad_core's .rad reader; no card
    // needed): the plan must know it before the step runs.
    rad::RadFile meta;
    if (meta.open(container.c_str()) != RAD_OK)
        throw std::runtime_error(std::string(kTest) + ": the container's metadata did not open");
    int64_t layers = 0, hidden = 0;
    if (!meta.get_i("n_layers", &layers) || !meta.get_i("n_embd", &hidden) || layers <= 0 ||
        hidden <= 0)
        throw std::runtime_error(std::string(kTest) +
                                 ": the container names no layers or hidden size");

    // The stream the step's ops issue on (patch 0122): a copy enqueued on it orders behind the
    // op the recorder saw, which is what makes a boundary's keep land after its block.
    RadStream stream = rad_ctx_cur_stream(ctx);
    R97_CHECK(stream != nullptr);

    // The card's free VRAM. This child points both runtimes at its one card before the first
    // init, so that card is device 0 in this process, whatever R97_DEVICE is on the machine.
    R97_CHECK(rad_dev_count() >= 1);
    RadDeviceProps props{};
    R97_CHECK(rad_dev_props(0, &props) == RAD_OK);
    R97_CHECK(!props.is_host_backend);  // a card, not the host backend standing in for one
    const int64_t free_bytes = props.vram_free;

    // The memory plan, printed with its numbers before anything is allocated.
    r97::ResidualPlan plan;
    plan.boundaries = layers + 1;
    plan.tokens     = tokens;
    plan.hidden     = hidden;
    // The second copies: in this container's int8 trunk every block after the first reads the
    // previous block's output beside the residual (its first norm is a fused add), so one at
    // every boundary after the first. The recorder cross-checks the plan against the tape in
    // both directions and throws when they disagree.
    for (int64_t b = 0; b < plan.boundaries; ++b)
        plan.extra.push_back(b >= 1 && b < layers ? 1 : 0);
    R97_CHECK(plan.fits(free_bytes));
    std::printf(
        "%s: the plan: %lld boundaries plus %lld second copies, %lld tokens x %lld hidden x 2 "
        "bytes = %lld bytes; the card holds %lld bytes free\n",
        kTest, (long long)plan.boundaries, (long long)plan.extras(), (long long)plan.tokens,
        (long long)plan.hidden, (long long)plan.store_bytes(), (long long)free_bytes);

    // The store on the card: one VRAM allocation of plan.store_bytes(); the constructor refuses
    // with the plan's numbers when it does not fit, before the step runs.
    r97::ResidualStore store(stream, plan, free_bytes);
    R97_CHECK(store.on_device());
    R97_CHECK(store.boundary_bytes() == plan.boundary_bytes());
    R97_CHECK(store.bytes() == plan.store_bytes());

    r97::Tape tape;
    tape.attach(ctx);
    R97_CHECK(tape.attached());
    r97::BoundaryRecorder rec(tape, std::move(store));

    s = engine.serve(std::make_unique<rad::StallWatch>(kTest, 60000));
    if (s < 0)
        throw std::runtime_error(std::string(kTest) + ": the prefill failed: " +
                                 rad_strerror(s));

    // What the forward left behind: one bracketed block per layer, each with ops of its own,
    // and the store holds every boundary.
    R97_CHECK(tape.blocks().size() == (size_t)layers);
    std::vector<size_t> block_ops((size_t)layers);
    for (const r97::Tape::Block& b : tape.blocks()) {
        R97_CHECK(b.index >= 0 && b.index < layers);
        R97_CHECK(b.end > b.begin);
        block_ops[(size_t)b.index] = b.end - b.begin;
    }
    for (size_t l = 0; l < block_ops.size(); ++l)
        R97_CHECK(block_ops[l] > 0);  // no layer was skipped or bracketed twice

    const r97::ResidualStore& kept = rec.store();
    R97_CHECK(kept.on_device());
    R97_CHECK(kept.boundaries() == layers + 1);
    R97_CHECK(kept.boundary_bytes() == plan.boundary_bytes());
    R97_CHECK(kept.bytes() == plan.store_bytes());

    rec.detach();
    tape.detach();

    // The replay over a copy of the store: the copy shares the VRAM with the recorder's.
    r97::LayerReplay replay(engine, tape, kept);
    R97_CHECK(replay.layers() == layers);
    R97_CHECK(replay.store().on_device());
    R97_CHECK(replay.store().bytes() == plan.store_bytes());

    const int64_t boundary_bytes = plan.boundary_bytes();
    std::vector<uint8_t> from_card((size_t)boundary_bytes);   // the step buffer, after a replay
    std::vector<uint8_t> from_store((size_t)boundary_bytes);  // the store's slot, from the forward

    // One replay of `layer`: it must issue exactly the forward's block's ops, and the residual
    // the block leaves in the step's own buffer must be bit-identical to the boundary the
    // forward stored at the same place.
    const auto replay_one = [&](int64_t layer, const char* pass) {
        const void* residual = replay.residual(layer + 1);
        R97_CHECK(residual != nullptr);
        // Before the block runs: the replay's first act is to put boundary `layer` back -- the
        // residual into the buffer the forward wrote it from, the second copy into the buffer the
        // block reads. Do that here, sync, and read both back, so a restore that missed its
        // buffer is caught here instead of being blamed on the block's arithmetic.
        {
            const void* dst = replay.residual(layer);
            R97_CHECK(dst != nullptr);
            replay.store().restore(layer, const_cast<void*>(dst));
            R97_CHECK(rad_stream_sync(stream) == RAD_OK);
            std::vector<uint8_t> back((size_t)boundary_bytes), want((size_t)boundary_bytes);
            replay.store().read(layer, want.data());
            R97_CHECK(rad_memcpy_async(back.data(), dst, boundary_bytes, stream) == RAD_OK);
            R97_CHECK(rad_stream_sync(stream) == RAD_OK);
            if (std::memcmp(back.data(), want.data(), (size_t)boundary_bytes) != 0)
                throw std::runtime_error(std::string(kTest) + ": layer " + std::to_string(layer) +
                                         ": the restored residual is not what the store holds");
            const void* extra = replay.store().source_extra(layer);
            if (extra) {
                replay.store().read_extra(layer, want.data());
                R97_CHECK(rad_memcpy_async(back.data(), extra, boundary_bytes, stream) == RAD_OK);
                R97_CHECK(rad_stream_sync(stream) == RAD_OK);
                if (std::memcmp(back.data(), want.data(), (size_t)boundary_bytes) != 0)
                    throw std::runtime_error(std::string(kTest) + ": layer " + std::to_string(layer) +
                                             ": the restored second copy is not what the store "
                                             "holds");
            }
        }
        int64_t seen = 0;
        replay.replay(layer, [&](const r97::OpRecord& op,
                                 const std::vector<r97::OperandContent>&) {
            ++seen;
            R97_CHECK(op.block == layer);
        });
        R97_CHECK(seen == (int64_t)block_ops[(size_t)layer]);
        // The replayed block has finished: wait on the lane, then read the block's output out
        // of the step's own buffer (device to host), and the forward's stored boundary out of
        // the store's slot (device to host, synced).
        R97_CHECK(rad_stream_sync(stream) == RAD_OK);
        R97_CHECK(rad_memcpy_async(from_card.data(), residual, boundary_bytes, stream) == RAD_OK);
        R97_CHECK(rad_stream_sync(stream) == RAD_OK);
        replay.store().read(layer + 1, from_store.data());
        if (std::memcmp(from_card.data(), from_store.data(), (size_t)boundary_bytes) != 0) {
            // Name the layer, how much differs and the first place it does, in bytes and in
            // values, and how many rows carry a difference: a whole boundary is a wrong buffer,
            // every value slightly off is an input the replay rounds differently, and a few rows
            // is an edge only those rows touch.
            size_t first = 0, differing = 0;
            int64_t values = 0, rows = 0;
            const size_t row_bytes = (size_t)plan.hidden * 2;
            for (size_t i = 0; i < from_card.size(); ++i) {
                if (from_card[i] == from_store[i]) continue;
                if (differing == 0) first = i;
                ++differing;
            }
            for (size_t i = 0; i + 1 < from_card.size(); i += 2) {
                const uint16_t a = (uint16_t)(from_card[i] | (from_card[i + 1] << 8));
                const uint16_t b = (uint16_t)(from_store[i] | (from_store[i + 1] << 8));
                if (a != b) ++values;
            }
            for (int64_t r = 0; r < tokens; ++r) {
                const size_t off = (size_t)r * row_bytes;
                if (std::memcmp(from_card.data() + off, from_store.data() + off, row_bytes) != 0)
                    ++rows;
            }
            const auto hex = [](uint8_t b) {
                char buf[8];
                std::snprintf(buf, sizeof buf, "%02x", b);
                return std::string(buf);
            };
            std::string why = std::string(kTest) + ": layer " + std::to_string(layer) + " (" +
                              pass + "): the replayed block's output differs from the stored "
                              "boundary: " + std::to_string(differing) + " of " +
                              std::to_string(boundary_bytes) + " bytes (" +
                              std::to_string(values) + " of " +
                              std::to_string(tokens * plan.hidden) + " values, " +
                              std::to_string(rows) + " of " + std::to_string(tokens) +
                              " rows), first at " + std::to_string(first) + " (card " +
                              hex(from_card[first]) + " " + hex(from_card[first + 1]) +
                              ", store " + hex(from_store[first]) + " " +
                              hex(from_store[first + 1]) + ")";
            std::printf("%s\n", why.c_str());
            std::fflush(stdout);
            throw std::runtime_error(why);
        }
        // The replayed block wrote its output into the same step buffer the forward used: the
        // boundary's source, not a copy of it.
        R97_CHECK(replay.residual(layer + 1) == residual);
    };

    for (int64_t layer = 0; layer < layers; ++layer)
        replay_one(layer, "forward");
    // A replay must not depend on which replay ran before it: the store holds every buffer the
    // block reads, so a layer replays the same in any order. Probe the layer the reverse walk
    // reaches second -- the first one there that needs a second copy restored -- before the walk,
    // and twice, to tell "depends on what ran before it" apart from "not deterministic at all".
    if (layers >= 2) {
        replay_one(layers - 2, "out of order");
        replay_one(layers - 2, "out of order again");
    }
    for (int64_t layer = layers - 1; layer >= 0; --layer)
        replay_one(layer, "reverse");

    std::printf(
        "%s: %s, %lld layers, %lld tokens: every layer replayed forward and reverse from the "
        "%lld-byte store in VRAM, bit-identical\n",
        kTest, container.c_str(), (long long)layers, (long long)tokens,
        (long long)plan.store_bytes());
}

}  // namespace

R97_MODEL_TEST(replay_layers_card) {
    std::string container;
#ifdef R97_MODEL_RAD
    container = r97test::container_or_skip(R97_MODEL_RAD);
#else
    container = r97test::container_or_skip("");
#endif
    if (std::getenv(kRoleVar)) {
        card_case(container);
        return;  // the child did the work; run_all reports the pass
    }

    run_in_child();
}

#else

// A build without the patch set cannot replay a layer on the card. Say so by name so the tier's
// output distinguishes a skip from a test that passed or never existed; no model case is
// registered in such a build.
namespace {

struct ReplayCardSkips {
    ReplayCardSkips() {
#if !defined(RAD_ISSUE_OBSERVER)
        constexpr const char* why = "the issue observer patch 0101 is not in this build";
#elif !defined(RAD_REPLAY_BLOCK)
        constexpr const char* why = "the replay patch 0102 is not in this build: "
                                    "-DR97_PATCH_SKIP took it out (RAD_REPLAY_BLOCK is undefined)";
#elif !defined(RAD_CTX_STREAM)
        constexpr const char* why = "the context-stream patch 0122 is not in this build: the "
                                    "store keeps its boundaries on the stream the step's ops "
                                    "issue on";
#else
        constexpr const char* why = "the prefill mode patch 0103 is not in this build";
#endif
        std::printf("[gpu] replay_layers_card ... skipped (%s)\n", why);
        std::fflush(stdout);
    }
} replay_card_skips;

}  // namespace

#endif  // R97_REPLAY && RAD_CTX_STREAM && train_prefill.h
