// tests/support/serve_prefill.h -- one prompt served through the engine's prefill, no tape.
//
// tests/support/lora_w8a8.h does the same but carries the tape of the step, and the tape is the
// training set's (patches/training/0101, 0119). A serving build has the prefill
// (patches/serving/0202, 0203) and no observer, so a check that a serving build serves an adapter
// cannot include that header. This one names only what both sets carry: start the engine on a
// container, load an adapter from a directory, run the prefill, hand back the logits. It does not
// include fixture.h either, so the gpu tier can use it without the fixture's writers.
//
// Header-only, like the rest of tests/support/: one includer, no object file to collect.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "engine.h"           // radiance's core/; Config, Engine, rad_strerror
#include "util/stallwatch.h"  // radiance's core/; the watch serve() takes

// The prefill mode is patches/serving/0202-prefill-logits.patch: it adds the driver
// (core/train_prefill.cpp) and its marker header to the copy the build compiles. A build that
// skips the serving set has neither, and these tests compile to nothing rather than be named by a
// build rule.
#if __has_include("train_prefill.h")
#define R97_FIXTURE_PREFILL 1
#elif __has_include("train_prefill.cpp")
#error "r97: stale radiance copy: no train_prefill.h -- delete <build>/radiance and reconfigure"
#else
#define R97_FIXTURE_PREFILL 0
#endif

#if R97_FIXTURE_PREFILL

namespace r97 {
namespace serve_prefill {

// What run_on hands back: every position's whole vocabulary, row-major.
struct Logits {
    int32_t            n_tokens = 0;
    int32_t            n_vocab  = 0;
    std::vector<float> logits;
};

// The name radiance takes the adapter under (--lora NAME=PATH), and the slot the prefill runs.
inline constexpr const char* kAdapterName = "mine";

// The plugin home: the environment's RADIANCE_HOME, else the build's radiance_home beside the test
// binary (fixture.h finds it the same way for the tiny fixture).
inline std::string radiance_home() {
    if (const char* h = std::getenv("RADIANCE_HOME"); h && *h) return h;
    std::error_code ec;
    const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (ec) return {};
    const std::filesystem::path home = exe.parent_path() / "radiance-build" / "radiance_home";
    return std::filesystem::exists(home, ec) ? home.string() : std::string();
}

// `hierarchy` is radiance's ordered kernel tier: {"libavx", "libref"} on the host, and the card's
// test puts "libr4d" in front of it.
inline rad::Config config(const std::string& container,
                          const std::vector<std::vector<int32_t>>& prompts,
                          const std::string& adapter, int64_t rank,
                          const std::vector<std::string>& hierarchy) {
    rad::Config cfg;
    cfg.model            = container;
    cfg.radiance_home    = radiance_home();
    cfg.kernel_hierarchy = hierarchy;
    cfg.n_spec           = 0;
    cfg.max_tok          = 512;
    cfg.max_seqs         = 8;
    cfg.max_ctx          = 512;
    cfg.prefill_prompts  = prompts;
    if (!adapter.empty()) {
        cfg.prefill_lora = kAdapterName;
        cfg.lora.adapters.push_back({ kAdapterName, adapter });
        cfg.lora.max_rank = rank;
    }
    return cfg;
}

inline std::vector<std::string> host_hierarchy() { return { "libavx", "libref" }; }

// One prompt through the engine's prefill, with `adapter` loaded (empty for none). Throws when the
// engine cannot start or the prefill fails, so a caller that gets a value knows the adapter was
// accepted. `n_vocab` is the row width the engine returned over the prompt's positions.
inline Logits run_on(const std::string& container, const std::vector<int32_t>& ids,
                     const std::string& adapter, int64_t rank,
                     const std::vector<std::string>& hierarchy) {
    if (ids.empty()) throw std::invalid_argument("serve_prefill: an empty prompt");
    rad::Engine engine(config(container, { ids }, adapter, rank, hierarchy));
    const int   s = engine.start();
    if (s < 0)
        throw std::runtime_error(std::string("serve_prefill: the engine could not start on ") +
                                 container + ": " + rad_strerror(s));
    const int served = engine.serve(std::unique_ptr<rad::StallWatch>());
    if (served < 0)
        throw std::runtime_error(std::string("serve_prefill: the prefill failed: ") +
                                 rad_strerror(served));

    const std::vector<std::vector<float>>& rows = engine.config().prefill_logits;
    if (rows.empty() || rows[0].empty() || rows[0].size() % ids.size() != 0)
        throw std::runtime_error("serve_prefill: the engine returned no logits for the prompt");
    Logits out;
    out.n_tokens = (int32_t)ids.size();
    out.n_vocab  = (int32_t)(rows[0].size() / ids.size());
    out.logits   = rows[0];
    return out;
}

}  // namespace serve_prefill
}  // namespace r97

#endif  // R97_FIXTURE_PREFILL
