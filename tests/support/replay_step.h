// replay_step.h -- one live prefill step, kept for a layer replay.
//
// tests/support/fixture.h's run_prefill() destroys its engine, so a test that replays a layer of
// the step it recorded cannot use it. This keeps the engine and the tape alive and records the
// residual at every block boundary with a BoundaryRecorder.
//
// The fixture is written here, into a directory of the step's own, rather than through
// r97::fixture(): a step that keeps its engine and its tape alive owns that directory, and removes
// it with the step. The plugin home and the converter are found by asking the executable's
// directory, so neither has to be passed in.
#pragma once

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fixture.h"
#include "r97/replay.h"

#if defined(R97_REPLAY) && R97_FIXTURE_PREFILL

#include "engine.h"

namespace r97 {
namespace replay_detail {

inline std::string env_or_empty(const char* name) {
    const char* v = std::getenv(name);
    return v && *v ? std::string(v) : std::string();
}

inline std::string exe_dir() {
    std::error_code ec;
    const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
    return ec ? std::string() : exe.parent_path().string();
}

inline std::string plugin_home() {
    std::string h = env_or_empty("RADIANCE_HOME");
    if (!h.empty()) return h;
    const std::string d = exe_dir();
    if (d.empty()) return {};
    const std::filesystem::path p =
        std::filesystem::path(d) / "radiance-build" / "radiance_home";
    std::error_code ec;
    return std::filesystem::exists(p, ec) ? p.string() : std::string();
}

inline std::string converter() {
    std::string p = env_or_empty("R97_RAD_CONVERT");
    if (!p.empty()) return p;
    const std::string d = exe_dir();
    if (!d.empty()) {
        const std::filesystem::path c =
            std::filesystem::path(d) / "radiance-build" / "bin" / "rad-convert";
        std::error_code ec;
        if (std::filesystem::exists(c, ec)) return c.string();
    }
    return "rad-convert";
}

}  // namespace replay_detail

struct ReplayStep {
    std::string dir;
    std::string container;
    std::unique_ptr<rad::Engine> engine;
    Tape tape;
    std::unique_ptr<BoundaryRecorder> recorder;
    ResidualStore store;
    int64_t tokens = 0;
    int64_t layers = 0;
    int64_t hidden = 0;

    ReplayStep() = default;
    ~ReplayStep() {
        if (recorder) recorder->detach();
        recorder.reset();
        tape.detach();
        engine.reset();
        if (!dir.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(dir, ec);
        }
    }
    ReplayStep(const ReplayStep&) = delete;
    ReplayStep& operator=(const ReplayStep&) = delete;

    // Start an engine on `sequences` as one prefill step, record its tape and its boundary
    // residuals, and leave the engine on that step. Throws when anything fails.
    void run(std::vector<std::vector<int32_t>> sequences) {
        if (sequences.empty()) throw std::invalid_argument("replay step needs a sequence");
        tokens = 0;
        for (const std::vector<int32_t>& s : sequences) tokens += (int64_t)s.size();

        const auto spec = detail::qwen35_spec();
        layers = spec.layers;
        hidden = spec.hidden;
        char tmpl[] = "/tmp/r97-replay-XXXXXX";
        if (::mkdtemp(tmpl) == nullptr)
            throw std::runtime_error("replay step: cannot make a temporary directory");
        dir = tmpl;
        detail::write_checkpoint(dir);
        container = dir + "/tiny_qwen35_bf16.rad";

        const std::string home = replay_detail::plugin_home();
        if (home.empty())
            throw std::runtime_error(
                "replay step: no plugin home: the build makes one at "
                "<build>/radiance-build/radiance_home; set RADIANCE_HOME to say where it is");
        const std::string cmd = detail::shell_quote(replay_detail::converter()) + " " +
                                detail::shell_quote(dir) + " -o " +
                                detail::shell_quote(container) + " --kernels libavx,libref" +
                                " --home " + detail::shell_quote(home);
        if (std::system(cmd.c_str()) != 0)
            throw std::runtime_error("replay step: rad-convert failed: " + cmd);

        rad::Config cfg;
        cfg.model            = container;
        cfg.radiance_home    = home;
        cfg.kernel_hierarchy = {"libavx", "libref"};
        cfg.n_spec           = 0;
        cfg.max_tok          = 512;
        cfg.max_seqs         = 8;
        cfg.max_ctx          = 512;
        cfg.prefill_prompts  = std::move(sequences);

        engine.reset(new rad::Engine(std::move(cfg)));
        int s = engine->start();
        if (s < 0)
            throw std::runtime_error(std::string("replay step: the engine could not start: ") +
                                     rad_strerror(s));
        RadCtx* ctx = engine->issue_ctx();
        if (!ctx) throw std::runtime_error("replay step: the engine has no issue context");
        tape.attach(ctx);
        recorder.reset(new BoundaryRecorder(tape, tokens, hidden));

        s = engine->serve(std::make_unique<rad::StallWatch>("replay step", 60000));
        if (s < 0)
            throw std::runtime_error(std::string("replay step: the prefill failed: ") +
                                     rad_strerror(s));
        store = recorder->store();
        recorder->detach();
        tape.detach();
    }
};

}  // namespace r97

#endif  // R97_REPLAY && R97_FIXTURE_PREFILL
