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
// position's logits. The mode it drives is patches/serving/'s, so a serving build runs this
// helper as well; the observer it does not use is the training set's (R97_FIXTURE_PREFILL below).
//
// The converter and the plugin home are found beside this test binary, where the build keeps
// radiance-build/; R97_RAD_CONVERT and RADIANCE_HOME in the environment still win.
// docs/FIXTURES.md says how the build makes both.
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

// The prefill mode the fixture drives is patches/serving/0202-prefill-logits.patch: it adds the
// driver (core/train_prefill.cpp) and its marker header to the copy the build compiles. The mode
// is asked for over the same include path the compiler reads, so neither build rule has to know a
// test's name; a build that skips the serving set gets no run_prefill().
#if __has_include("train_prefill.h")
#define R97_FIXTURE_PREFILL 1
#elif __has_include("train_prefill.cpp")
// The tree keeps its patched copy of radiance until the patch set's file names change, so a build
// directory made before the marker header existed has the mode but not the header. Say so rather
// than compile these tests away and report success.
#error "r97: stale radiance copy: no train_prefill.h -- delete <build>/radiance and reconfigure"
#else
#define R97_FIXTURE_PREFILL 0
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

#if R97_FIXTURE_PREFILL
// The logits of one prefill, row-major: row `pos` holds the distribution for the token after
// position `pos`, over `n_vocab` ids, in the order the plugin produced it.
struct Prefill {
    int32_t            n_tokens = 0;
    int32_t            n_vocab = 0;
    std::vector<float> logits;

    const float* at(int32_t pos) const { return logits.data() + (std::ptrdiff_t)pos * n_vocab; }
    float        at(int32_t pos, int32_t token) const { return at(pos)[token]; }
};
#endif  // R97_FIXTURE_PREFILL

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

// The directory this test binary sits in: <build>/, where the build also keeps radiance's own
// build tree (<build>/radiance-build) and its plugins. Empty if the answer cannot be read.
inline const std::string& build_dir() {
    static const std::string dir = [] {
        std::error_code ec;
        const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
        return ec ? std::string() : exe.parent_path().string();
    }();
    return dir;
}

// A path in the build's radiance build tree, if it is there.
inline std::string in_radiance_build(const char* first, const char* second = nullptr) {
    if (build_dir().empty()) return {};
    std::filesystem::path p = std::filesystem::path(build_dir()) / "radiance-build" / first;
    if (second) p /= second;
    std::error_code ec;
    return std::filesystem::exists(p, ec) ? p.string() : std::string();
}

// The converter: one the environment names, else the build's own rad-convert, else PATH.
inline std::string rad_convert() {
    std::string p = env_or("R97_RAD_CONVERT");
    if (p.empty()) p = in_radiance_build("bin", "rad-convert");
    if (p.empty()) p = "rad-convert";
    return p;
}

// The plugin home: the environment's RADIANCE_HOME, else the build's radiance_home.
inline std::string radiance_home() {
    std::string h = env_or("RADIANCE_HOME");
    if (h.empty()) h = in_radiance_build("radiance_home");
    return h;
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

// The tiny Qwen3.5 the fixture, the oracle writer and every test share: one definition, so a
// change to the shape changes the checkpoint the reference was made from and the hash test says
// so by name.
inline tiny::Qwen35 qwen35_spec() {
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
    return s;
}

// The checkpoint alone, at `dir`: what tools/oracle/make_reference.py loads. make() converts it
// afterwards. The bytes are the fixture's own, so the reference and the tests see one checkpoint.
inline std::string write_checkpoint(const std::string& dir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec)
        throw std::runtime_error("fixture: cannot create " + dir + ": " + ec.message());
    if (!tiny::write_qwen35(dir, qwen35_spec()))
        throw std::runtime_error("fixture: cannot write the tiny Qwen3.5 checkpoint at " + dir);
    write_chat_template(dir);
    return dir;
}

inline Fixture make() {
    namespace fs = std::filesystem;

    const fs::path dir = fs::path(radtest::tmpdir()) / "tiny_qwen35_bf16";
    fs::remove_all(dir);
    write_checkpoint(dir.string());
    const tiny::Qwen35 s = qwen35_spec();

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
    if (home.empty())
        throw std::runtime_error(
            "fixture: no plugin home: the build makes one at "
            "<build>/radiance-build/radiance_home and this binary found none beside itself (" +
            (build_dir().empty() ? std::string("unknown directory") : build_dir()) +
            "); set RADIANCE_HOME to say where it is");
    cmd += " --home " + shell_quote(home);
    if (std::system(cmd.c_str()) != 0)
        throw std::runtime_error("fixture: rad-convert failed: " + cmd);
    std::error_code ec;
    if (!fs::exists(f.container, ec))
        throw std::runtime_error("fixture: rad-convert wrote no container at " + f.container);
    return f;
}

}  // namespace detail

// Public form for tools/oracle/write_fixture.cpp: the checkpoint alone, without a container.
inline std::string write_checkpoint(const std::string& dir) {
    return detail::write_checkpoint(dir);
}

// The fixture for this test binary: written and converted on first use, then kept for its life.
inline const Fixture& fixture() {
    static const Fixture f = detail::make();
    return f;
}

#if R97_FIXTURE_PREFILL
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
    cfg.max_tok          = 512;  // one step holds the full-length case, and the three-sequence batch
    cfg.max_seqs         = 8;
    cfg.max_ctx          = 512;  // the KV pool covers what these prompts need, not the 4096 trained
                                 // (a prefill-only request fills it exactly: no generation slot)
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
#endif  // R97_FIXTURE_PREFILL

}  // namespace r97
