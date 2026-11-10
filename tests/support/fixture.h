// tests/support/fixture.h -- the tiny model the host tests run on.
//
// The fixture is a tiny Qwen3.5 checkpoint, written once per test binary by radiance's
// tests/tiny_model.h (write_qwen35) into the test's temporary directory, then converted with
// rad-convert to a bf16 .rad container that the qwen35_bf16 plugin serves. Its shape:
//
//   arch        qwen3_5, served by qwen35_bf16
//   hidden      128  (a multiple of 128, since the int8 fixture is converted from this checkpoint)
//   layers      4    ((l + 1) % 4 == 0 is full attention, so 3 gated delta net layers and 1 full
//                     attention layer)
//   attention   4 query heads of 32 over 2 key-value heads (grouped-query attention)
//   vocabulary  320, byte-level: ids 0..255 are the bytes, 256..258 the specials
//   tokenizer   a ChatML chat template in a tokenizer_config.json beside it
//
// run_prefill() drives that container through rad_core in-process -- no HTTP server -- with the
// host kernel libraries, one prefill step from position 0 and speculation off, and returns every
// position's logits. The build tells this header where the converter and the plugin home are
// (R97_RAD_CONVERT, R97_RADIANCE_HOME); docs/FIXTURES.md says how to build them.
//
// Header-only, like the rest of tests/support/: one includer, no object file to collect.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "engine.h"           // radiance's core/; Config, Engine, rad_strerror
#include "rad_test.h"         // radiance's tests/; radtest::tmpdir
#include "tiny_model.h"       // radiance's tests/; tiny::Qwen35, tiny::write_qwen35
#include "util/stallwatch.h"  // radiance's core/; the watch serve() takes

#ifndef R97_RAD_CONVERT
#define R97_RAD_CONVERT ""
#endif
#ifndef R97_RADIANCE_HOME
#define R97_RADIANCE_HOME ""
#endif

namespace r97 {

// The tiny checkpoint and the container converted from it.
struct Fixture {
    std::string checkpoint;   // the Hugging Face-style directory write_qwen35 wrote
    std::string container;    // the bf16 .rad the engine loads
    int32_t     n_vocab = 0;  // the vocabulary the tokenizer configuration declares
    int32_t     n_layers = 0;
    int32_t     hidden = 0;
};

// The logits of one prefill, row-major: row `pos` holds the distribution for the token after
// position `pos`, over `n_vocab` ids, in the order the plugin produced it.
struct Prefill {
    int32_t            n_tokens = 0;
    int32_t            n_vocab = 0;
    std::vector<float> logits;

    const float* at(int32_t pos) const { return logits.data() + (std::ptrdiff_t)pos * n_vocab; }
    float        at(int32_t pos, int32_t token) const { return at(pos)[token]; }
};

namespace detail {

inline std::string env_or(const char* name) {
    const char* v = std::getenv(name);
    return v && *v ? std::string(v) : std::string();
}

inline std::string shell_quote(const std::string& s) {
    std::string q = "'";
    for (char c : s) {
        if (c == '\'') q += "'\\''";
        else q += c;
    }
    return q + "'";
}

// The converter: the build's own rad-convert, else one the environment names, else PATH.
inline std::string rad_convert() {
    std::string p = R97_RAD_CONVERT;
    if (p.empty()) p = env_or("R97_RAD_CONVERT");
    if (p.empty()) p = "rad-convert";
    return p;
}

// The plugin home: the build's radiance_home, else the environment's RADIANCE_HOME.
inline std::string radiance_home() {
    return R97_RADIANCE_HOME[0] ? std::string(R97_RADIANCE_HOME) : env_or("RADIANCE_HOME");
}

// The chat template the tokenizer configuration carries: ChatML, which the tiny tokenizer's
// <|im_start|>/<|im_end|> specials fit. Nothing here applies it; it is part of the fixture's shape,
// so a test that formats a conversation finds one.
inline const char* chat_template() {
    return "{% for message in messages %}"
           "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + "
           "'\\n' }}"
           "{% endfor %}"
           "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";
}

// write_qwen35 writes a tokenizer.json and no tokenizer_config.json, and radiance reads the chat
// template only from the sibling tokenizer_config.json or chat_template.jinja (core/text/
// tokenizer_json.cpp) -- a key inside tokenizer.json is ignored. So the template goes into a
// tokenizer_config.json of its own. It names no bos/eos token: those stay what tokenizer.json's
// post-processor said, and this file adds the template and nothing else.
inline void write_chat_template(const std::string& dir) {
    nlohmann::ordered_json cfg = nlohmann::ordered_json::object();
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast";
    cfg["chat_template"]   = chat_template();
    const std::string path = dir + "/tokenizer_config.json";
    if (!tiny::write_text(path, cfg.dump(2) + "\n"))
        throw std::runtime_error("fixture: cannot write " + path);
}

inline Fixture make() {
    namespace fs = std::filesystem;

    const fs::path dir = fs::path(radtest::tmpdir()) / "tiny_qwen35_bf16";
    fs::remove_all(dir);
    fs::create_directories(dir);

    tiny::Qwen35 s;
    s.hidden   = 128;  // a multiple of 128: the int8 fixture converts this same checkpoint
    s.layers   = 4;    // (l + 1) % interval == 0 is full attention: 3 GDN layers, 1 attention
    s.interval = 4;
    s.heads    = 4;
    s.head_dim = 32;   // 4 x 32 = 128 = hidden
    s.kv_heads = 2;    // grouped-query attention
    s.ffn      = 256;
    s.vocab    = 320;
    s.ctx      = 4096;
    if (!tiny::write_qwen35(dir.string(), s))
        throw std::runtime_error("fixture: cannot write the tiny Qwen3.5 checkpoint at " +
                                 dir.string());
    write_chat_template(dir.string());

    Fixture f;
    f.checkpoint = dir.string();
    f.container  = (dir / "tiny_qwen35_bf16.rad").string();
    f.n_vocab    = (int32_t)s.vocab;
    f.n_layers   = (int32_t)s.layers;
    f.hidden     = (int32_t)s.hidden;

    // rad-convert declares the checkpoint with the plugins and writes the bf16 container. The host
    // kernel hierarchy is named explicitly so the conversion and the engine agree on it.
    std::string cmd = shell_quote(rad_convert()) + " " + shell_quote(f.checkpoint) + " -o " +
                      shell_quote(f.container) + " --kernels libavx,libref";
    const std::string home = radiance_home();
    if (!home.empty()) cmd += " --home " + shell_quote(home);
    if (std::system(cmd.c_str()) != 0)
        throw std::runtime_error("fixture: rad-convert failed: " + cmd);
    std::error_code ec;
    if (!fs::exists(f.container, ec))
        throw std::runtime_error("fixture: rad-convert wrote no container at " + f.container);
    return f;
}

}  // namespace detail

// The fixture for this test binary: written and converted on first use, then kept for its life.
inline const Fixture& fixture() {
    static const Fixture f = detail::make();
    return f;
}

// Run `sequences` as one prefill: all are submitted together, each from position 0, with
// speculation off, and each gets the logits of every position of its own prompt. The engine
// batches what the scheduler admits; a sequence's rows do not depend on which others share its
// step. Throws std::runtime_error when the fixture or the engine cannot do it.
inline std::vector<Prefill> run_prefill(const Fixture& f,
                                        const std::vector<std::vector<int32_t>>& sequences) {
    if (sequences.empty()) return {};

    rad::Config cfg;
    cfg.model            = f.container;
    cfg.radiance_home    = detail::radiance_home();
    cfg.kernel_hierarchy = {"libavx", "libref"};
    cfg.n_spec           = 0;    // speculation off: one prefill step, nothing drafted
    cfg.max_tok          = 512;  // room for the three-sequence batch as well as the 96-token case
    cfg.max_seqs         = 8;
    cfg.max_ctx          = 512;  // the KV pool covers what these prompts need, not the 4096 trained
    cfg.prefill_prompts  = sequences;

    rad::Engine engine(std::move(cfg));
    int s = engine.start();
    if (s < 0)
        throw std::runtime_error(std::string("fixture: the engine could not start on ") +
                                 f.container + ": " + rad_strerror(s));
    auto watch = std::make_unique<rad::StallWatch>("fixture prefill", 60000);
    s = engine.serve(std::move(watch));
    if (s < 0)
        throw std::runtime_error(std::string("fixture: the prefill failed: ") + rad_strerror(s));

    const std::vector<std::vector<float>>& rows = engine.config().prefill_logits;
    if (rows.size() != sequences.size())
        throw std::runtime_error("fixture: the engine returned " + std::to_string(rows.size()) +
                                 " prompts' logits for " + std::to_string(sequences.size()));

    std::vector<Prefill> out(sequences.size());
    for (size_t i = 0; i < sequences.size(); ++i) {
        const size_t want = sequences[i].size() * (size_t)f.n_vocab;
        if (rows[i].size() != want)
            throw std::runtime_error("fixture: prompt " + std::to_string(i) + " holds " +
                                     std::to_string(rows[i].size() / (size_t)f.n_vocab) +
                                     " logits rows, its prompt has " +
                                     std::to_string(sequences[i].size()) + " tokens");
        out[i].n_tokens = (int32_t)sequences[i].size();
        out[i].n_vocab  = f.n_vocab;
        out[i].logits   = rows[i];
    }
    return out;
}

// The one-sequence form of run_prefill.
inline Prefill run_prefill(const Fixture& f, const std::vector<int32_t>& token_ids) {
    std::vector<Prefill> one = run_prefill(f, std::vector<std::vector<int32_t>>{token_ids});
    return one[0];
}

}  // namespace r97
