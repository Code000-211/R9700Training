// tests/support/served_prefill.h -- a corpus served through radiance's own KL-scored prefill.
//
// A serving build has to answer a prefill with the adapter r97train wrote using nothing but what
// stock radiance carries. `--kld-record` is that path: it prefills every corpus document, takes the
// next-token log-probability of each scored position and writes the NLL and the row's entropy to
// DIR/rows.f32 -- the same two numbers `r97train loss` prints for the same tokens, from the model a
// deployment actually runs. The adapter is named by `--kld-lora` (`cfg.kld_lora`), so the serve
// goes through a slot radiance acquired itself: none of the trainer's prefill mode, its driver or
// its adapter lookup is named here, and a build with only patches/serving/ has all of it.
//
// Header-only, like the rest of tests/support/: one includer, no object file to collect.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "engine.h"           // radiance's core/; Config, Engine, rad_strerror
#include "util/stallwatch.h"  // radiance's core/; the watch serve() takes

namespace r97 {
namespace served_prefill {

// The name radiance takes the adapter under (--lora NAME=PATH) and the slot the prefill scores.
inline constexpr const char* kAdapterName = "trained";

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
inline rad::Config config(const std::string& container, const std::string& corpus,
                          const std::string& record, const std::string& adapter, int64_t rank,
                          const std::vector<std::string>& hierarchy) {
    rad::Config cfg;
    cfg.model            = container;
    cfg.radiance_home    = radiance_home();
    cfg.kernel_hierarchy = hierarchy;
    cfg.n_spec           = 0;
    cfg.max_tok          = 512;
    cfg.max_seqs         = 8;
    cfg.max_ctx          = 512;
    cfg.kld_corpus       = corpus;
    cfg.kld_record       = record;
    if (!adapter.empty()) {
        cfg.kld_lora = kAdapterName;
        cfg.lora.adapters.push_back({ kAdapterName, adapter });
        cfg.lora.max_rank = rank;
    }
    return cfg;
}

inline std::vector<std::string> host_hierarchy() { return { "libavx", "libref" }; }

// One JSON line per prompt, the corpus format `--kld-corpus` reads.
inline void write_corpus(const std::string& path, const std::vector<std::string>& prompts) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("served_prefill: cannot write " + path);
    for (const std::string& p : prompts) {
        std::string quoted;
        for (char c : p) {
            if (c == '\\' || c == '"') quoted += '\\';
            quoted += c;
        }
        out << "{\"prompt\": \"" << quoted << "\"}\n";
    }
    if (!out) throw std::runtime_error("served_prefill: short write to " + path);
}

// The rows of one record: NLL and entropy per scored position, row-major as rows.f32 holds them.
struct Served {
    int64_t            n_rows = 0;
    std::vector<float> rows;

    // The number `r97train loss` prints: the mean next-token NLL over the scored positions.
    double loss() const {
        double sum = 0.0;
        for (int64_t i = 0; i < n_rows; ++i) sum += rows[(size_t)i * 2];
        return n_rows > 0 ? sum / double(n_rows) : 0.0;
    }

    bool finite() const {
        for (float v : rows)
            if (!std::isfinite(v)) return false;
        return n_rows > 0;
    }
};

// The value of `"key": <int>` in a kld.json, or -1.
inline int64_t json_int(const std::string& text, const std::string& key) {
    const size_t k = text.find("\"" + key + "\"");
    if (k == std::string::npos) return -1;
    const size_t colon = text.find(':', k + key.size() + 2);
    if (colon == std::string::npos) return -1;
    return std::strtoll(text.c_str() + colon + 1, nullptr, 10);
}

inline std::string json_str(const std::string& text, const std::string& key) {
    const size_t k = text.find("\"" + key + "\"");
    if (k == std::string::npos) return {};
    const size_t open = text.find('"', text.find(':', k + key.size() + 2));
    if (open == std::string::npos) return {};
    const size_t close = text.find('"', open + 1);
    return close == std::string::npos ? std::string() : text.substr(open + 1, close - open - 1);
}

// The record `--kld-record` wrote: its manifest must name the whole-vocabulary form, and rows.f32
// must be exactly the rows the manifest claims (two floats each).
inline Served read_record(const std::string& dir) {
    std::ifstream manifest(dir + "/kld.json");
    if (!manifest) throw std::runtime_error("served_prefill: " + dir + " has no kld.json");
    const std::string text((std::istreambuf_iterator<char>(manifest)),
                           std::istreambuf_iterator<char>());
    const std::string format = json_str(text, "format");
    if (format != "radiance-kld-1")
        throw std::runtime_error("served_prefill: " + dir + " is a '" + format +
                                 "' record; this test needs the whole-vocabulary radiance-kld-1");

    std::ifstream in(dir + "/rows.f32", std::ios::binary);
    if (!in) throw std::runtime_error("served_prefill: " + dir + " has no rows.f32");
    std::vector<char> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (raw.size() % (2 * sizeof(float)) != 0)
        throw std::runtime_error("served_prefill: " + dir + "/rows.f32 is not whole rows");

    Served out;
    out.n_rows = (int64_t)(raw.size() / (2 * sizeof(float)));
    if (out.n_rows != json_int(text, "rows"))
        throw std::runtime_error("served_prefill: " + dir + "/rows.f32 and its kld.json disagree");
    out.rows.resize(out.n_rows * 2);
    std::memcpy(out.rows.data(), raw.data(), raw.size());
    return out;
}

// Serve `corpus` with `adapter` loaded (empty for none) and hand back the rows radiance recorded.
// Throws when the engine cannot start or the serve fails, so a caller that gets a value knows the
// adapter was accepted and the prefill ran.
inline Served served(const std::string& container, const std::string& corpus,
                     const std::string& adapter, int64_t rank,
                     const std::vector<std::string>& hierarchy, const std::string& record) {
    if (corpus.empty()) throw std::invalid_argument("served_prefill: an empty corpus");
    rad::Engine engine(config(container, corpus, record, adapter, rank, hierarchy));
    const int   s = engine.start();
    if (s < 0)
        throw std::runtime_error(std::string("served_prefill: the engine could not start on ") +
                                 container + ": " + rad_strerror(s));
    const int rc = engine.serve(std::unique_ptr<rad::StallWatch>());
    if (rc < 0)
        throw std::runtime_error(std::string("served_prefill: the serve failed: ") + rad_strerror(rc));
    return read_record(record);
}

}  // namespace served_prefill
}  // namespace r97
