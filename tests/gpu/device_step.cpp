// tests/gpu/device_step.cpp -- the device join (issue #55): one step of the real container on one
// card, as a StepFn that comes back with the adapter's gradient.
//
// The join runs the batch through the forward driver on the device kernel tier (libr4d, the host
// tier as the per-op fallback), keeps the residual at every layer boundary in one VRAM allocation
// (r97/step_device.h's ResidualStore, on the stream the step's ops issue on -- patches/training/
// 0122), walks each block back on the card with the device adjoints (Side::device, the rows
// libr97.so registers as the plugin loads), and flushes the adapter's gradients once, at the
// step's end, into the ParamSet the caller passes as grads. docs/STEPS-DEVICE.md owns the path
// and the memory it holds. A call with grads == nullptr is a true forward-only step: the forward
// runs with the adapter fed, the report is filled, and the head's gradient pass and the block
// walk are skipped -- so a forward-only call is the cheap reference the checks lean on: the
// pre-step loss, and the descent, which proves the gradient's direction, not just its shape (a
// sign or scale error in the walk would leave the loss up).
//
//   device_step_tape       512 tokens. The op-coverage probe plus a full step, with the adapter
//                          seeded. The probe runs the join forward-only (grads == nullptr) on
//                          the step's own params, and then reads the step's tape, which the
//                          driver keeps until the next step: every op the 64 blocks issued must
//                          have a device adjoint row -- the join walks exactly those blocks and,
//                          of the head's records, the rmsnorm and the gather_rows before the
//                          logits op. The ops the walk does not need a row for: logits_gemm_vocab
//                          (the join handles the logits op itself; no adjoint row exists for it,
//                          by design), rope_table (off the walk's path) and embed_lookup (below
//                          the first boundary, a frozen leaf). What has no row is reported with
//                          adjoint_find's own message. The adapter is seeded with a
//                          deterministic non-zero pattern over both factors -- init leaves
//                          lora_B exactly zero, and a zero B never reaches lora_A's factor, so an
//                          unseeded start only proves the chain half-way -- and the step's
//                          gradient must be finite and non-zero on every leaf of both factors:
//                          one AdamW step then moves both factors, and a forward-only call on the
//                          updated adapter sees the loss fall.
//   device_step_seq4096    the ticket: one 4096-token step on the card, at the seed-0 start where
//                          lora_B is exactly zero and lora_A is random at 1/sqrt(fan_in). The
//                          forward peaks at 20.2 GiB of VRAM (docs/reports/2026-11-12-device-
//                          forward.md, which records this exact batch's loss at 34682.089228),
//                          the residual store adds 5 GiB -- 128 copies x 4096 x 5120 x 2 bytes,
//                          the L+1 = 65 boundaries plus the L-1 = 63 second copies the int8 trunk
//                          reads beside the residual (replay_layers' plan, on the card) -- so the
//                          case declares 26 GiB to the guard. At the zero-B start the `.lora_A`
//                          gradients are exactly zero (no cotangent reaches A's factor) while the
//                          `.lora_B` gradients are not -- the pair that proves the chain reached
//                          the adapter -- and one AdamW step moves only the lora_B's, at weight
//                          decay 0.
//
// The gradients the walk returns are mean-loss gradients: the head's chunk kernel is already
// handed grad_scale = 1/scored (include/r97/head.h), so nothing here rescales, and no check
// compares a magnitude against a hand-normalised expectation.
//
// Both cases run with keep_boundaries off, its default: the report's f64 copies of the 65
// boundaries would be ~2.7 GiB of host RAM at 512 tokens and ~43 GiB at 4096.
//
// The child/parent split. The engine picks the card only through ROCR_VISIBLE_DEVICES, which the
// ROCR runtime reads once, at init, and the shared r97_gpu_tests binary runs every test in one
// process where the other gpu tests init ROCR before this one does. So no engine work runs in
// this process at all: the parent forks, and the child re-execs this same binary with
// R97_TEST_ONLY (only this test), R97_DSTEP_OUT (where to write its results file) and
// ROCR_VISIBLE_DEVICES already set -- a fresh process whose ROCR points at the card from the
// first device call. The child does the real work and writes a JSON file the parent reads and
// checks; the parent resolves the container's shape from its metadata (no card) so its checks
// carry their own expected values. Each child waits for the card to report its VRAM free
// (wait_for_card_free) before the guard checks it: a previous child's VRAM is returned to the
// driver asynchronously. The child prints each phase's elapsed time -- the step's wall, the
// join's own forward/head/backward split, and each forward-only call's wall -- so a run's
// numbers size the window the parent holds.
//
// The tier. The driver's kernel hierarchy names the card's library and libr97, the plugin the
// device adjoints load from: it serves no forward kernel, it is in the list so the engine
// dlopen()s it through the hierarchy (core/plugin/loader.cpp loads every named library), and its
// static initialisers register the device rows into this process's r97 registry as it loads. The
// build passes the exact file it built as R97_DEVICE_PLUGIN (CMakeLists.txt) and
// support/device_check.h resolves it: the probe dlopen()s it once, through load_plugin(), as a
// reference count on the file the engine already loaded, so its adjoint_find does not depend on
// the engine's load. libr97 is installed in radiance_home's kernels directory beside libr4d, so
// the name is all the hierarchy needs.
//
// The join is a training path: it needs the issue observer (patches/training/0101), the replay
// block (0102) and the context stream (0122) the store keeps its boundaries on; a serving build
// applies none of the set, so the cases compile to nothing there and say so by name instead.
#include "support/gpu_guard.h"
#include "support/model_case.h"
#include "support/model_dir.h"
#include "support/test.h"

#include "r97/forward.h"
#include "r97/step_device.h"

#if R97_FORWARD && R97_DEVICE_STEP

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>
#include <sys/wait.h>

#include "engine.h"             // radiance's core/; rad::Engine must be complete: the driver's
                                // own std::unique_ptr<rad::Engine> instantiates its destructor
                                // at the first use
#include "rad_device.h"         // rad_dev_count/props: the card, not the host backend
#include "support/device_check.h"  // load_plugin(): libr97.so, resolved and dlopen()ed once

#include "r97/adapter_io.h"
#include "r97/adjoint.h"
#include "r97/optim.h"

namespace {

namespace fs = std::filesystem;

using r97::Batch;
using r97::TokenSeq;

const char* kRoleVar = "R97_DSTEP_ROLE";  // set in the re-exec'd child; the role that does the work
const char* kOutVar  = "R97_DSTEP_OUT";   // the child's results directory

// ------------------------------------------------------------------ the small helpers

const std::string& self_exe() {
    static const std::string exe = [] {
        std::error_code ec;
        const fs::path p = fs::read_symlink("/proc/self/exe", ec);
        if (ec) throw std::runtime_error("device_step: cannot read /proc/self/exe");
        return p.string();
    }();
    return exe;
}

// The build directory this test binary sits in: radiance-build/ (the plugins, the home) is beside
// it.
std::string build_dir() {
    std::error_code ec;
    const fs::path p = fs::read_symlink("/proc/self/exe", ec);
    return ec ? std::string() : p.parent_path().string();
}

// The plugin home: RADIANCE_HOME, else the build's radiance_home.
std::string radiance_home() {
    if (const char* e = std::getenv("RADIANCE_HOME"))
        if (*e) return e;
    const std::string h = build_dir() + "/radiance-build/radiance_home";
    std::error_code ec;
    return fs::exists(h + "/architectures", ec) ? h : std::string();
}

// A fresh directory under TMPDIR (else /tmp) for one case's results. mkdtemp, so two copies of the
// test never collide. The test tier does not carry radiance's tests/ path, so this is the
// equivalent of radtest::tmp_path, which the gpu target cannot use.
std::string temp_dir(const std::string& name) {
    const char* base = std::getenv("TMPDIR");
    if (!base || !*base) base = "/tmp";
    std::string path = std::string(base) + "/r97-dstep-" + name + "-XXXXXX";
    char* d = ::mkdtemp(path.data());
    if (!d)
        throw std::runtime_error("device_step: cannot make a temp directory under " +
                                 std::string(base) + ": " + std::strerror(errno));
    return d;
}

// Fork, and have the child re-exec this binary running only `test_name`, with ROCR_VISIBLE_DEVICES
// pointing the ROCR runtime at the card before its first init. R97_TEST_ONLY selects this test so
// the re-exec'd copy runs it alone, and -- seeing R97_DSTEP_ROLE -- does the child's work instead
// of forking again. Wait, and end the run as skipped (exit 77) when the child skips (the container
// absent). The parent never opens the card or inits an engine: its ROCR was already init'd by the
// other gpu tests, so only a fresh child process can be pointed at the card.
void run_in_child(const std::string& dir, const char* test_name) {
    const char* dev = std::getenv("R97_DEVICE");
    const std::string device_num = dev && *dev ? dev : "0";

    const pid_t pid = ::fork();
    if (pid < 0)
        throw std::runtime_error(std::string("device_step: fork: ") + std::strerror(errno));
    if (pid == 0) {
        // The child: point both runtimes at the card before any init, mark this re-exec'd copy as
        // the one that does the work, and hand it the results directory. The re-exec'd process is
        // fresh, so the runtimes read these for the first time.
        if (::setenv("HIP_VISIBLE_DEVICES", device_num.c_str(), 1) != 0 ||
            ::setenv("ROCR_VISIBLE_DEVICES", device_num.c_str(), 1) != 0 ||
            ::setenv("R97_TEST_ONLY", test_name, 1) != 0 ||
            ::setenv(kRoleVar, "device", 1) != 0 || ::setenv(kOutVar, dir.c_str(), 1) != 0)
            std::_Exit(1);
        std::string exe = self_exe();
        char* const argv[2] = {exe.data(), nullptr};
        ::execv(exe.c_str(), argv);
        std::_Exit(1);  // execv failed
    }

    int status = 0;
    if (::waitpid(pid, &status, 0) != pid)
        throw std::runtime_error(std::string("device_step: ") + test_name + ": waitpid failed: " +
                                 std::strerror(errno));
    if (!WIFEXITED(status))
        throw std::runtime_error(std::string("device_step: ") + test_name +
                                 "'s child was killed by a signal; the card may be held by another run");
    const int code = WEXITSTATUS(status);
    if (code != 0) {
        // 77 (the container absent) re-skips the way ctest records it; anything else is a failure.
        if (code == r97test::kSkip)
            r97test::skip(std::string("the model container is absent (see the child run of ") +
                          test_name + " above");
        throw std::runtime_error(std::string("device_step: ") + test_name +
                                 "'s child exited " + std::to_string(code) +
                                 ": its stderr above says why; the card is R97_DEVICE, and the "
                                 "guard names the card's holders when the VRAM is short");
    }
}

// The results directory this (child) process was told to write into.
std::string out_dir() {
    const char* d = std::getenv(kOutVar);
    if (!d || !*d)
        throw std::runtime_error("device_step: " + std::string(kOutVar) +
                                 " is unset; the results have nowhere to go");
    return d;
}

// ------------------------------------------------------------------ the card-free wait

// Wait until the selected card reports `need` bytes free. A previous test's child may have just
// exited and the driver can take a moment to return its VRAM to mem_info_vram_used; the guard
// would otherwise refuse on a stale reading of a card that is in fact free.
void wait_for_card_free(const r97test::GuardOptions& o) {
    const std::string sysfs = "/sys/class/drm";
    const std::string card = r97test::card_for_bdf(sysfs, o.bdf);
    for (int i = 0; i < 120; ++i) {  // up to ~60 s
        try {
            const r97test::CardVram v = r97test::read_card_vram(sysfs, card);
            if (v.total - v.used >= o.need_vram) return;
        } catch (const std::exception&) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

int64_t read_vram_used(const std::string& card) {
    return r97test::read_sys_int("/sys/class/drm/" + card + "/device/mem_info_vram_used");
}

// ------------------------------------------------------------------ the adapter's start

// A deterministic, non-zero seed over both factors: init leaves `lora_B` exactly zero, and a zero
// B never reaches `lora_A`'s factor, so an unseeded start proves the chain half-way only. The
// seed is splitmix64 over each leaf's index, one draw per value, in [-1/sqrt(fan_in),
// 1/sqrt(fan_in)]: both factors take rank inputs per output row, and every leaf is a different
// pattern, so a step that trains the adapter moves every one of them.
void seed_adapter(r97::ParamSet& params) {
    for (size_t leaf = 0; leaf < params.size(); ++leaf) {
        r97::Param& p = params[leaf];
        const int64_t fan_in = p.shape.size() >= 2 ? p.shape[1] : 1;
        const double scale = 1.0 / std::sqrt((double)std::max<int64_t>(1, fan_in));
        uint64_t state = 0x9e3779b97f4a7c15ull * (uint64_t(leaf) + 1);
        for (float& v : p.value)
            v = float((2.0 * r97::device_check::urand(&state) - 1.0) * scale);
    }
}

// ------------------------------------------------------------------ the child path

// One joined step of one `n`-token sequence on the card, the op-coverage probe when asked, the
// gradient's checks, one AdamW step, and the forward-only descent check: the checks run where
// the data is, and the parent reads the numbers out of the results file.
void card_case(const char* test, int64_t n, bool probe) {
    r97test::GuardOptions o;
    const char* de = std::getenv("R97_DEVICE");
    o.device    = r97test::parse_device(de ? de : "0");
    // The container's device forward peaks at 20.2 GiB at 4096 tokens (weights, KV and the
    // engine's arena), the residual store adds 5 GiB there and ~0.7 at 512, and the join's
    // snapshots and the gradient staging sit beside them: the 4096 case declares 26 GiB, the
    // 512 one 21.
    o.need_vram = (n >= 4096) ? int64_t(26) << 30 : int64_t(21) << 30;
    o.bdf       = r97test::hip_bdf(o.device);  // HIP is inited only now: ROCR_VISIBLE_DEVICES is set
    wait_for_card_free(o);
    r97test::DeviceGuard guard(o);
    const std::string card = guard.card();  // the /sys/class/drm/cardN directory

    const std::string container = r97test::container_or_skip();
    const std::string home = radiance_home();
    if (home.empty())
        throw std::runtime_error(std::string(test) + ": no radiance home beside the build");
    const std::string dir = out_dir();

    // This child points both runtimes at its one card before the first init, so that card is
    // device 0 in this process, whatever R97_DEVICE is on the machine -- and it is a card. The
    // host backend standing in for one cannot run this container's trunk linears at all.
    R97_CHECK(rad_dev_count() >= 1);
    RadDeviceProps props{};
    if (rad_dev_props(0, &props) != 0)
        throw std::runtime_error(std::string(test) + ": rad_dev_props: " + rad_dev_last_error());
    R97_CHECK(!props.is_host_backend);
    std::printf("%s: %s, %lld tokens, %lld MiB free on the card\n", test, card.c_str(),
                (long long)n, (long long)(props.vram_free / r97test::kMiB));

    // The one-prompt length this child steps: the same id scheme as device_forward's seq cases,
    // every token but the first scored.
    TokenSeq seq;
    seq.ids.resize(size_t(n));
    for (size_t i = 0; i < seq.ids.size(); ++i) seq.ids[i] = 1 + int32_t((i * 7) % 299);
    seq.score.assign(size_t(n), 1);
    seq.score[0] = 0;
    Batch batch;
    batch.push_back(std::move(seq));

    // The model and its adapter, rank 16, alpha 32. The 4096 case starts where init leaves the
    // adapter -- lora_B exactly zero, lora_A random at 1/sqrt(fan_in) -- so the step's loss is
    // the base model's and the gradient's split is the seed-0 pair; the 512 case seeds both
    // factors non-zero so every leaf of the gradient has to be non-zero.
    const r97::ModelMeta meta = r97::read_model_meta(container);
    r97::Config mc;
    mc.model     = container;
    mc.objective = "ce";
    mc.delta.kind = "lora";
    mc.delta.rank = 16;
    mc.delta.alpha = 32.0;
    const r97::Model model = r97::make_model(mc);
    r97::ParamSet params = model.delta->init(meta, mc.delta, /*seed=*/0);
    R97_CHECK(!params.empty());
    if (probe) seed_adapter(params);

    // The driver for the tier: the card's library first, libr97 the plugin the device adjoints
    // load from (it serves no forward kernel; it is in the list so the engine dlopen()s it), the
    // host tier as the per-op fallback. One sequence, one piece: the driver turns radiance's
    // checkpoint interval off for the step itself (see src/forward.cpp).
    r97::ForwardConfig cfg;
    cfg.model         = container;
    cfg.radiance_home = home;
    cfg.kernels       = {"libr4d", "libr97", "libavx", "libref"};
    cfg.max_tok       = n;
    cfg.max_ctx       = n;
    cfg.max_seqs      = 1;
    cfg.window_rows   = 256;
    std::string err;
    std::unique_ptr<r97::ForwardDriver> driver = r97::ForwardDriver::open(cfg, &err);
    if (!driver) throw std::runtime_error(err);

    r97::DeviceStepOptions options;  // keep_boundaries stays off, its default: the report's f64
                                     // copies of the 65 boundaries are ~10 GiB a vector, ~21 GiB
                                     // for values and cotangents, at 4096 tokens
    r97::DeviceStepReport report;
    const r97::StepFn step = r97::device_step(*driver, model, options, &report);

    nlohmann::json j;
    j["name"] = test;

    // ---- the op-coverage probe: the join forward-only, then the step's tape

    double probe_seconds = 0.0;
    if (probe) {
        // grads == nullptr is a forward-only call on the step's own params: the forward runs
        // with the adapter fed, and the driver keeps this step's tape until the next step. The
        // tape is read here, before the gradient call replaces last().
        const auto tp0 = std::chrono::steady_clock::now();
        const r97::StepResult r0 = step(batch, params, nullptr);
        probe_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
        R97_CHECK(r0.scored == n - 1);
        R97_CHECK(std::isfinite(r0.loss));
        // The engine loaded libr97 through the kernel hierarchy when the step's engine started
        // (the join's head loop needs the plugin's head kernel, and the step would have thrown
        // without one). dlopen() the same file -- R97_DEVICE_PLUGIN is the build's exact path,
        // and radiance_home's kernels directory is where the build puts it -- as a reference
        // count, so the lookups below do not depend on that load.
        r97::device_check::load_plugin();
        const r97::Tape& tape = driver->last().tape();
        // The ops the walk does not need a row for: the logits op the join handles itself (no
        // adjoint row exists for it, by design), the rope table, which is off the walk's path,
        // and the embedding below the first boundary, a frozen leaf.
        static const std::set<std::string> kNotWalked = {"logits_gemm_vocab", "rope_table",
                                                         "embed_lookup"};
        std::set<std::string> block_ops, head_ops;
        std::set<int64_t> block_ids;
        std::vector<std::pair<std::string, std::string>> missing;  // (op, adjoint_find's message)
        int64_t records = 0;
        for (const r97::OpRecord& rec : tape.ops()) {
            ++records;
            if (rec.block >= 0) {
                R97_CHECK(rec.block < meta.layers);
                block_ops.insert(rec.op);
                block_ids.insert(rec.block);
            } else {
                head_ops.insert(rec.op);
            }
            if (rec.block < 0 && kNotWalked.count(rec.op)) continue;
            std::string why;
            if (!r97::adjoint_find(rec.op, r97::Side::device, rec.params, &why)) {
                if (std::find_if(missing.begin(), missing.end(),
                                 [&](const std::pair<std::string, std::string>& m) {
                                     return m.first == rec.op && m.second == why;
                                 }) == missing.end())
                    missing.push_back({rec.op, std::move(why)});
            }
        }
        R97_CHECK(records > 0);
        nlohmann::json block_list = nlohmann::json::array();
        for (const std::string& op : block_ops) block_list.push_back(op);
        nlohmann::json head_list = nlohmann::json::array();
        for (const std::string& op : head_ops) head_list.push_back(op);
        nlohmann::json missing_list = nlohmann::json::array();
        for (const auto& m : missing) missing_list.push_back({m.first, m.second});
        j["probe_ops"]       = records;
        j["probe_blocks"]    = int64_t(block_ids.size());
        j["probe_block_ops"] = block_list;
        j["probe_head_ops"]  = head_list;
        j["probe_missing"]   = missing_list;
        j["probe_loss"]      = r0.loss;
        j["probe_scored"]    = r0.scored;
    }

    // ---- the step: one prefill piece with the adapter's gradient, VRAM at the peak

    r97::ParamSet grads;
    // The step runs the batch as one engine run: start() (the model's load) and serve() (the
    // forward) inside it, then the head and the walk. Poll the card's VRAM at the peak while it
    // runs.
    std::atomic<bool> running{true};
    std::atomic<long long> peak{0};
    std::thread poller([&] {
        while (running.load(std::memory_order_relaxed)) {
            try {
                const int64_t v = read_vram_used(card);
                long long cur = peak.load(std::memory_order_relaxed);
                while (v > cur && !peak.compare_exchange_weak(cur, v)) {
                }
            } catch (const std::exception&) {
            }  // one unreadable poll is noise; the step's time is what matters
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    });
    // The poller must be joined before any exception leaves this function: a joinable std::thread
    // destroyed during unwinding calls std::terminate, which would hide the step's real error.
    double step_seconds = 0.0;
    r97::StepResult result{0.0, 0};
    try {
        const auto t0 = std::chrono::steady_clock::now();
        result = step(batch, params, &grads);
        step_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    } catch (...) {
        running.store(false);
        poller.join();
        throw;
    }
    running.store(false);
    poller.join();
    const int64_t vram_after_load = read_vram_used(card);

    // The report the gradient call left: the descent call below overwrites it (grad_stage_bytes
    // is 0 on a step that runs without grads), so the step's own numbers are copied out first.
    const r97::DeviceStepReport rep = report;

    // ---- the gradient: the same names and shapes as the params, and the start's invariants

    // The CE chunk's gradient is already scaled by grad_scale = 1/scored (include/r97/head.h),
    // so what the walk returns is the mean-loss gradient: no rescaling here, and no magnitude
    // compared against a hand-normalised expectation -- the checks are the shape, the zero /
    // non-zero split, and the descent.
    const auto ends = [](const std::string& s, const char* suffix) {
        const std::string tail(suffix);
        return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
    };
    bool names_match      = grads.size() == params.size();
    bool shapes_match     = names_match;
    bool all_finite       = true;
    bool a_all_zero       = true;   // the zero-B start: no cotangent reaches A's factor
    bool a_all_nonzero    = true;   // the seeded start: every A leaf sees the cotangent
    bool b_all_nonzero    = true;   // both starts: the cotangent reaches B's factor
    int64_t n_a = 0, n_b = 0, n_other = 0;
    double grad_norm_sq   = 0.0;
    for (const r97::Param& p : params) {
        const r97::Param* g = nullptr;
        for (const r97::Param& c : grads)
            if (c.name == p.name) { g = &c; break; }
        if (!g) {
            names_match = false;
            continue;
        }
        if (g->shape != p.shape) shapes_match = false;
        const bool is_a = ends(p.name, "lora_A");
        const bool is_b = ends(p.name, "lora_B");
        if (is_a) ++n_a;
        else if (is_b) ++n_b;
        else ++n_other;
        double part = 0.0;
        for (size_t i = 0; i < g->value.size(); ++i) {
            const float v = g->value[i];
            if (!std::isfinite(v)) all_finite = false;
            part += double(v) * double(v);
        }
        grad_norm_sq += part;
        if (is_a) {
            if (part == 0.0) a_all_nonzero = false;
            if (part != 0.0) a_all_zero = false;
        }
        if (is_b && part == 0.0) b_all_nonzero = false;
    }

    // One AdamW step off the gradient, at the config's default lr, weight decay 0: a zero
    // gradient leaves its parameter bit-identical, and a nonzero one moves it.
    const r97::ParamSet before = params;
    r97::AdamWConfig oc;
    oc.lr = 1e-3f;
    r97::Optimiser opt(params, oc);
    const bool applied = opt.step(grads);  // throws when grads do not name the optimiser's params
    R97_CHECK(applied);
    R97_CHECK(opt.updates_done() == 1);
    int64_t a_changed = 0, b_changed = 0;
    for (size_t i = 0; i < params.size(); ++i) {
        if (params[i].value == before[i].value) continue;
        if (ends(params[i].name, "lora_A")) ++a_changed;
        else if (ends(params[i].name, "lora_B")) ++b_changed;
    }

    // ---- the descent: a forward-only call on the updated adapter

    // The step's own loss is the pre-step reference -- the same params the forward-only probe
    // ran -- and the loss after one AdamW step must be lower: a sign or scale error in the walk
    // would leave it up.
    double after_seconds = 0.0;
    r97::StepResult after{0.0, 0};
    {
        const auto t0 = std::chrono::steady_clock::now();
        after = step(batch, params, nullptr);
        after_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    R97_CHECK(after.scored == result.scored);
    R97_CHECK(std::isfinite(after.loss));
    R97_CHECK(after.loss < result.loss);  // the step descended

    // ---- the results

    j["card"]                 = card;
    j["tokens"]               = n;
    j["scored"]               = result.scored;
    j["loss"]                 = result.loss;
    j["loss_after"]           = after.loss;
    j["loss_after_scored"]    = after.scored;
    j["report_loss"]          = rep.loss;
    j["report_scored"]        = rep.scored;
    j["hidden"]               = rep.hidden;
    j["layers"]               = rep.layers;
    j["residual_boundaries"]  = rep.residual_boundaries;
    j["residual_extras"]      = rep.residual_extras;
    j["residual_store_bytes"] = rep.residual_store_bytes;
    j["residual_budget_bytes"] = rep.residual_budget_bytes;
    j["peak_snapshot_bytes"]  = rep.peak_snapshot_bytes;
    j["peak_snapshot_store_bytes"] = rep.peak_snapshot_store_bytes;
    j["peak_snapshot_op_bytes"] = rep.peak_snapshot_op_bytes;
    j["peak_cotangent_bytes"] = rep.peak_cotangent_bytes;
    j["grad_stage_bytes"]     = rep.grad_stage_bytes;
    j["d_hout_bytes"]         = rep.d_hout_bytes;
    j["head_rows"]            = rep.head_rows;
    j["head_chunks"]          = rep.head_chunks;
    j["kept_boundary_bytes"]  = rep.kept_boundary_bytes;
    j["boundary_names"]       = int64_t(rep.boundary_name.size());
    j["forward_seconds"]      = rep.forward_seconds;
    j["head_seconds"]         = rep.head_seconds;
    j["backward_seconds"]     = rep.backward_seconds;
    j["probe_seconds"]        = probe_seconds;
    j["step_seconds"]         = step_seconds;
    j["after_seconds"]        = after_seconds;
    j["tok_s"]                = step_seconds > 0.0 ? double(n) / step_seconds : 0.0;
    j["vram_after_load"]      = vram_after_load;
    j["vram_peak"]            = peak.load();
    j["n_params"]             = int64_t(params.size());
    j["n_grads"]              = int64_t(grads.size());
    j["names_match"]          = names_match;
    j["shapes_match"]         = shapes_match;
    j["all_finite"]           = all_finite;
    j["n_lora_a"]             = n_a;
    j["n_lora_b"]             = n_b;
    j["n_other"]              = n_other;
    j["a_all_zero"]           = a_all_zero;
    j["a_all_nonzero"]        = a_all_nonzero;
    j["b_all_nonzero"]        = b_all_nonzero;
    j["grad_norm"]            = std::sqrt(grad_norm_sq);
    j["opt_applied"]          = applied;
    j["updates_done"]         = opt.updates_done();
    j["a_changed"]            = a_changed;
    j["b_changed"]            = b_changed;
    // The walk's split by op name: the row calls (with the bus each one owns) and the replay's
    // pull-backs. Written before any check below can throw, so a refused walk still leaves them.
    auto timing = [](const std::vector<r97::OpTiming>& v) {
        nlohmann::json a = nlohmann::json::array();
        for (const r97::OpTiming& t : v)
            a.push_back({{"op", t.op}, {"calls", t.calls}, {"seconds", t.seconds}});
        return a;
    };
    j["row_seconds"]          = timing(rep.row_seconds);
    j["pull_seconds"]         = timing(rep.pull_seconds);

    std::ofstream out(dir + "/" + test + ".json");
    if (!out) throw std::runtime_error(std::string(test) + ": cannot write the results");
    out << j.dump(2) << "\n";

    // Each phase's elapsed time, so a run's numbers size the window the parent holds: the
    // forward-only probe, the step's wall (and the join's own forward/head/backward split
    // inside it), and the descent call.
    std::printf("%s: %s: %lld scored, loss %.6f -> %.6f after one AdamW step, grad norm %.6g, "
                "store %lld MiB, probe %.1f s, step %.1f s (forward %.1f, head %.1f, backward "
                "%.1f), descent %.1f s\n",
                test, card.c_str(), (long long)result.scored, result.loss, after.loss,
                std::sqrt(grad_norm_sq), (long long)(rep.residual_store_bytes / r97test::kMiB),
                probe_seconds, step_seconds, rep.forward_seconds, rep.head_seconds,
                rep.backward_seconds, after_seconds);
    // The same split, the few ops that hold the walk: the number that says whether a 4096-token
    // step needs a longer window, and which op to look at when it does.
    auto top = [](const std::vector<r97::OpTiming>& v) {
        std::vector<const r97::OpTiming*> p;
        p.reserve(v.size());
        for (const r97::OpTiming& t : v) p.push_back(&t);
        std::sort(p.begin(), p.end(),
                  [](const r97::OpTiming* a, const r97::OpTiming* b) { return a->seconds > b->seconds; });
        std::string s;
        for (size_t i = 0; i < p.size() && i < 5; ++i) {
            char b[160];
            std::snprintf(b, sizeof b, "%s%.1f s/%lld", i ? ", " : " ", p[i]->seconds,
                          (long long)p[i]->calls);
            s += p[i]->op + b;
        }
        return s.empty() ? std::string(" (none)") : s;
    };
    std::printf("%s: walk by op:%s | pull by op:%s\n", test, top(rep.row_seconds).c_str(),
                top(rep.pull_seconds).c_str());
}

}  // namespace

// ------------------------------------------------------------------ the two cases

// One joined step of `n` tokens, the op-coverage probe when asked. The parent resolves the
// container's shape from its metadata (no card) so its expected values carry their own
// provenance: the store's plan, the head's rows and its chunks, the brackets' count.
static void device_step_case(const char* test, int64_t n, bool probe) {
#ifdef R97_MODEL_RAD
    const std::string container = r97test::container_or_skip(R97_MODEL_RAD);
#else
    const std::string container = r97test::container_or_skip("");
#endif
    if (std::getenv(kRoleVar)) {
        card_case(test, n, probe);
        return;  // the child did the work and wrote its file; run_all reports the pass
    }

    const std::string dir = temp_dir(test);
    run_in_child(dir, test);

    const std::string path = dir + "/" + test + ".json";
    std::ifstream in(path);
    if (!in) throw std::runtime_error(std::string(test) + ": no results at " + path);
    nlohmann::json j;
    in >> j;

    const int64_t tokens = j.at("tokens").get<int64_t>();
    const int64_t scored = j.at("scored").get<int64_t>();
    const double loss    = j.at("loss").get<double>();
    const int64_t hidden = j.at("hidden").get<int64_t>();
    const int64_t layers = j.at("layers").get<int64_t>();
    const double after   = j.at("loss_after").get<double>();

    const r97::ModelMeta meta = r97::read_model_meta(container);

    // The step's loss over every scored token: the sum the join's forward and the objective
    // agree on, covering the whole prompt, and the head's rows and its chunks.
    R97_CHECK(tokens == n);
    R97_CHECK(scored == n - 1);  // every token but the first is scored
    R97_CHECK(std::isfinite(loss));
    R97_CHECK(hidden == meta.hidden);
    R97_CHECK(layers == meta.layers);
    R97_CHECK(j.at("report_scored").get<int64_t>() == scored);
    R97_CHECK(std::fabs(j.at("report_loss").get<double>() - loss) <=
              1e-6 * std::max(1.0, std::fabs(loss)));
    R97_CHECK(j.at("head_rows").get<int64_t>() == n - 1);
    R97_CHECK(j.at("head_chunks").get<int64_t>() ==
              (n - 1 + r97::kHeadChunkRows - 1) / r97::kHeadChunkRows);
    R97_CHECK(j.at("d_hout_bytes").get<int64_t>() > 0);

    // The card's residual store: the L+1 boundaries plus the second copy at every boundary a
    // block after the first starts at -- this container's int8 trunk reads the previous block's
    // output beside the residual -- in one VRAM allocation, checked against the card's free VRAM
    // when the engine was up.
    const int64_t store = j.at("residual_store_bytes").get<int64_t>();
    R97_CHECK(j.at("residual_boundaries").get<int64_t>() == layers + 1);
    R97_CHECK(j.at("residual_extras").get<int64_t>() == layers - 1);
    R97_CHECK(store == 2 * layers * n * hidden * 2);  // (L+1) + (L-1) copies, tokens x hidden x 2
    R97_CHECK(j.at("residual_budget_bytes").get<int64_t>() > 0);
    R97_CHECK(store <= j.at("residual_budget_bytes").get<int64_t>());

    // The adapter's gradient: the same parameters and shapes as the params, finite on every
    // leaf, and staged on the card at the step's end. The start's split is checked per case
    // below. The report kept no boundary copies: keep_boundaries is off.
    const int64_t n_a = j.at("n_lora_a").get<int64_t>();
    const int64_t n_b = j.at("n_lora_b").get<int64_t>();
    R97_CHECK(j.at("n_grads").get<int64_t>() == j.at("n_params").get<int64_t>());
    R97_CHECK(j.at("names_match").get<bool>());
    R97_CHECK(j.at("shapes_match").get<bool>());
    R97_CHECK(j.at("n_other").get<int64_t>() == 0);
    R97_CHECK(n_a > 0);
    R97_CHECK(n_b > 0);
    R97_CHECK(j.at("all_finite").get<bool>());
    R97_CHECK(j.at("grad_stage_bytes").get<int64_t>() > 0);
    R97_CHECK(j.at("kept_boundary_bytes").get<int64_t>() == 0);

    // The descent: one AdamW update, and the forward-only call on the updated adapter sees the
    // loss fall -- the step's own loss is the pre-step reference, the same params.
    R97_CHECK(j.at("opt_applied").get<bool>());
    R97_CHECK(j.at("updates_done").get<int64_t>() == 1);
    R97_CHECK(std::isfinite(after));
    R97_CHECK(after < loss);  // the step descended: strict, the drop is far above f64 noise
    R97_CHECK(j.at("loss_after_scored").get<int64_t>() == scored);

    R97_CHECK(j.at("vram_peak").get<int64_t>() >= j.at("vram_after_load").get<int64_t>());
    R97_CHECK(j.at("tok_s").get<double>() > 0);

    if (probe) {
        // The forward-only probe and the gradient call agree on the loss: the same (seeded)
        // params, the same forward.
        const double probe_loss = j.at("probe_loss").get<double>();
        R97_CHECK(std::fabs(probe_loss - loss) <= 1e-6 * std::max(1.0, std::fabs(loss)));

        // Every op the step's 64 blocks issued has a device adjoint row, and the brackets
        // cover the trunk; what has no row is named with adjoint_find's own message.
        R97_CHECK(j.at("probe_blocks").get<int64_t>() == layers);
        R97_CHECK(j.at("probe_ops").get<int64_t>() > 0);
        const nlohmann::json missing = j.at("probe_missing");
        if (!missing.empty()) {
            std::string why;
            for (const auto& m : missing) {
                why += (why.empty() ? "" : "; ") + m.at(0).get<std::string>() + ": " +
                       m.at(1).get<std::string>();
            }
            throw std::runtime_error(std::string(test) + ": no device adjoint row for " + why);
        }
        // The head the join walks: the rmsnorm and the gather_rows need a row (the probe above
        // found one for each), and the logits op the join handles itself is on the tape.
        const nlohmann::json head_ops = j.at("probe_head_ops");
        const auto head_has = [&](const char* op) {
            for (const auto& o : head_ops)
                if (o.get<std::string>() == op) return true;
            return false;
        };
        R97_CHECK(head_has("rmsnorm"));
        R97_CHECK(head_has("gather_rows"));
        R97_CHECK(head_has("logits_gemm_vocab"));

        // The seeded start: the cotangent reaches every leaf of both factors, so the gradient
        // is non-zero on each of them -- and the step moves both factors, not one.
        R97_CHECK(j.at("a_all_nonzero").get<bool>());
        R97_CHECK(j.at("b_all_nonzero").get<bool>());
        R97_CHECK(j.at("a_changed").get<int64_t>() == n_a);
        R97_CHECK(j.at("b_changed").get<int64_t>() == n_b);
    } else {
        // The zero-B start: no cotangent reaches A's factor, so its gradients are exactly zero,
        // and the step moves only the lora_B's -- the pair that proves the chain reached the
        // adapter.
        R97_CHECK(j.at("a_all_zero").get<bool>());
        R97_CHECK(j.at("b_all_nonzero").get<bool>());
        R97_CHECK(j.at("a_changed").get<int64_t>() == 0);
        R97_CHECK(j.at("b_changed").get<int64_t>() == n_b);
    }

    std::printf("%s: %s: %lld scored, loss %.6f (mean NLL %.6f) -> %.6f (mean NLL %.6f) after "
                "one AdamW step, %.1f tok/s, grad norm %.6g, store %lld MiB, VRAM %lld MiB "
                "after load, %lld MiB at the step's peak\n",
                test, j.at("card").get<std::string>().c_str(), (long long)scored, loss,
                loss / double(scored), after, after / double(scored),
                j.at("tok_s").get<double>(), j.at("grad_norm").get<double>(),
                (long long)(store / r97test::kMiB),
                (long long)(j.at("vram_after_load").get<int64_t>() / r97test::kMiB),
                (long long)(j.at("vram_peak").get<int64_t>() / r97test::kMiB));
    if (!probe) {
        // No check against the recorded figure (docs/reports/2026-11-12-device-forward.md): the
        // report compares them. Print them next to each other so the comparison is easy.
        std::printf("%s: this step's loss %.6f against the device forward's recorded %.6f for "
                    "this batch and container\n",
                    test, loss, 34682.089228);
    }
}

// 512 tokens: the op-coverage probe plus a full step, with the adapter seeded over both factors.
R97_MODEL_TEST(device_step_tape) { device_step_case("device_step_tape", 512, /*probe=*/true); }

// The ticket: one 4096-token step on the card, with the adapter's gradient.
R97_MODEL_TEST(device_step_seq4096) {
    device_step_case("device_step_seq4096", 4096, /*probe=*/false);
}

#else

// A build without the patch set cannot join a step on the card. Say so by name so the tier's
// output distinguishes a skip from a test that passed or never existed; no model case is
// registered in such a build.
namespace {

const char* device_step_skip_reason() {
#if !defined(RAD_ISSUE_OBSERVER)
    return "the issue observer patch 0101 is not in this build";
#elif !__has_include("train_prefill.h")
    return "the prefill mode patch 0103 is not in this build";
#elif !defined(RAD_REPLAY_BLOCK)
    return "the replay patch 0102 is not in this build: "
           "-DR97_PATCH_SKIP took it out (RAD_REPLAY_BLOCK is undefined)";
#else
    return "the context-stream patch 0122 is not in this build: the store keeps its boundaries "
           "in the card's VRAM on the stream the step's ops issue on";
#endif
}

struct DeviceStepTapeSkips {
    DeviceStepTapeSkips() {
        std::printf("[gpu] device_step_tape ... skipped (%s)\n", device_step_skip_reason());
        std::fflush(stdout);
    }
} device_step_tape_skips;

struct DeviceStepSeq4096Skips {
    DeviceStepSeq4096Skips() {
        std::printf("[gpu] device_step_seq4096 ... skipped (%s)\n", device_step_skip_reason());
        std::fflush(stdout);
    }
} device_step_seq4096_skips;

}  // namespace

#endif  // R97_FORWARD && R97_DEVICE_STEP
