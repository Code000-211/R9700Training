// tests/host/serving_prefill.cpp -- a serving build answers a prefill with an adapter written by
// r97train, and the served loss is the default build's to the bit.
//
// The two builds this repository makes differ only in the patch sets applied to radiance: the
// default ("all") build carries the trainer's issue observer and replay hook, a serving-only build
// carries neither. If the serving build computes anything other than what the default build's
// radiance computes, the adapter a deployment loads does not answer what the trainer measured.
// This test fixes the comparison to bytes: the served logits of one corpus with one adapter, and
// the NLL and entropy rows those logits score, against tests/fixtures/. The logits are the check --
// a permutation of two non-target logits, or a pair moved together, leaves both rows where they
// were -- and the rows are kept beside them because `r97train loss` prints those two numbers.
//
// The serve is radiance's own: `--kld-record` prefills the corpus and scores it through the adapter
// slot (`--kld-lora`), which stock radiance has, and `--kld-logits` keeps the f32 logits it scored
// (patches/serving/0202-kld-record-logits.patch), so this test runs in a serving build without any
// trainer-only patch (patches/serving/ has the LoRA on the int8 trunk, 0210, and nothing else it
// needs). The reference was written by this test in the default build
// (R97_WRITE_SERVING_GOLDEN=<path>); the same test runs in a serving build and must reproduce it
// exactly. The adapter in tests/fixtures/serving_adapter is one r97train wrote on the int8 fixture
// (tests/fixtures/serving_adapter/README.md prints the run). docs/DEPLOYMENT.md §5 has the place of
// this check in a deployment.
#include "support/fixture_w8a8.h"
#include "support/served_prefill.h"
#include "support/test.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

fs::path fixtures() {
    return fs::path(__FILE__).parent_path().parent_path() / "fixtures";
}

// This process's own directory for one run's corpus and record, beside the fixture the run reads
// (radtest::tmpdir(), the tests' own temp root), emptied first so a rerun cannot read a stale
// record.
std::string temp_dir(const char* name) {
    std::error_code ec;
    const fs::path dir = fs::path(radtest::tmpdir()) / ("served-" + std::string(name));
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    if (ec) throw std::runtime_error("serving_prefill: cannot make " + dir.string() + ": " + ec.message());
    return dir.string();
}

// Two short ASCII prompts: the fixture's vocabulary is byte-level (ids 0..319), so each is a
// handful of scored positions and the reference stays a few kilobytes.
const std::vector<std::string>& prompts() {
    static const std::vector<std::string> text = {
        "the quick brown fox jumps over the lazy dog",
        "pack my box with five dozen liquor jugs",
    };
    return text;
}

// On disk: "R97S" then n_rows as little-endian u32, then the rows (NLL, entropy) as f32; the logits
// file is "R97L" then n_rows and n_vocab as u32, then the rows' logits as f32. Written and read only
// here, so the layout can be this small.
constexpr char kMagic[4]      = { 'R', '9', '7', 'S' };
constexpr char kLogitMagic[4] = { 'R', '9', '7', 'L' };

void put_u32(std::ofstream& out, uint32_t v) {
    unsigned char b[4] = { (unsigned char)(v), (unsigned char)(v >> 8), (unsigned char)(v >> 16),
                           (unsigned char)(v >> 24) };
    out.write(reinterpret_cast<const char*>(b), 4);
}

uint32_t get_u32(std::ifstream& in) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

std::vector<float> read_reference(const fs::path& path, uint32_t* n_rows) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("serving_prefill: cannot read " + path.string());
    char magic[4];
    in.read(magic, 4);
    if (std::memcmp(magic, kMagic, 4) != 0)
        throw std::runtime_error("serving_prefill: " + path.string() + " is not a served record");
    *n_rows = get_u32(in);
    std::vector<float> rows((size_t)*n_rows * 2);
    in.read(reinterpret_cast<char*>(rows.data()), (std::streamsize)(rows.size() * 4));
    if (!in) throw std::runtime_error("serving_prefill: " + path.string() + " is short");
    return rows;
}

void write_reference(const fs::path& path, const r97::served_prefill::Served& s) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("serving_prefill: cannot write " + path.string());
    out.write(kMagic, 4);
    put_u32(out, (uint32_t)s.n_rows);
    out.write(reinterpret_cast<const char*>(s.rows.data()),
              (std::streamsize)(s.rows.size() * 4));
    if (!out) throw std::runtime_error("serving_prefill: short write to " + path.string());
}

std::vector<float> read_logits(const fs::path& path, uint32_t* n_rows, uint32_t* n_vocab) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("serving_prefill: cannot read " + path.string());
    char magic[4];
    in.read(magic, 4);
    if (std::memcmp(magic, kLogitMagic, 4) != 0)
        throw std::runtime_error("serving_prefill: " + path.string() + " is not a served logits record");
    *n_rows  = get_u32(in);
    *n_vocab = get_u32(in);
    std::vector<float> logits((size_t)*n_rows * (size_t)*n_vocab);
    in.read(reinterpret_cast<char*>(logits.data()), (std::streamsize)(logits.size() * 4));
    if (!in) throw std::runtime_error("serving_prefill: " + path.string() + " is short");
    return logits;
}

void write_logits(const fs::path& path, const r97::served_prefill::Served& s) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("serving_prefill: cannot write " + path.string());
    out.write(kLogitMagic, 4);
    put_u32(out, (uint32_t)s.n_rows);
    put_u32(out, (uint32_t)s.n_vocab);
    out.write(reinterpret_cast<const char*>(s.logits.data()),
              (std::streamsize)(s.logits.size() * 4));
    if (!out) throw std::runtime_error("serving_prefill: short write to " + path.string());
}

}  // namespace

R97_TEST(serving_prefill_with_the_fixture_adapter_matches_the_reference) {
    const r97::Fixture f = r97::w8a8_fixture();
    const fs::path      adapter = fixtures() / "serving_adapter";
    R97_CHECK(fs::exists(adapter / "adapter_config.json"));

    const std::string dir = temp_dir("serving_prefill");
    const std::string corpus = dir + "/corpus.jsonl";
    r97::served_prefill::write_corpus(corpus, prompts());

    const r97::served_prefill::Served served =
        r97::served_prefill::served(f.container, corpus, adapter.string(), 16,
                                    r97::served_prefill::host_hierarchy(), dir + "/served",
                                    /*logits=*/true);

    R97_CHECK(served.n_rows > 0);
    R97_CHECK(served.finite());
    R97_CHECK(served.n_vocab > 0);
    R97_CHECK((int64_t)served.logits.size() == served.n_rows * served.n_vocab);

    // Prove the adapter was loaded and used: without it the same serve must score something else,
    // or the comparison below would pass on a build that ignored the adapter.
    const r97::served_prefill::Served plain =
        r97::served_prefill::served(f.container, corpus, "", 16,
                                    r97::served_prefill::host_hierarchy(), dir + "/plain");
    R97_CHECK(plain.n_rows == served.n_rows);
    R97_CHECK(plain.finite());
    R97_CHECK(plain.loss() != served.loss());

    const fs::path rows_file   = fixtures() / "serving_rows.f32";
    const fs::path logits_file = fixtures() / "serving_logits.f32";
    if (const char* write = std::getenv("R97_WRITE_SERVING_GOLDEN")) {
        write_reference(write, served);
        // The logits go beside the rows the run named: one reference, two files of it.
        const fs::path logits = fs::path(write).parent_path() / "serving_logits.f32";
        write_logits(logits, served);
        std::printf("serving_prefill: wrote the served rows to %s and the logits to %s\n", write,
                    logits.string().c_str());
        return;
    }

    // The row itself first: every scored position's logits, byte for byte. These are the number a
    // serve answers with, so a build whose logits differ anywhere fails here even where the NLL and
    // the entropy of the scored token are the same.
    uint32_t           logit_rows = 0, n_vocab = 0;
    std::vector<float> logits = read_logits(logits_file, &logit_rows, &n_vocab);
    R97_CHECK((int64_t)logit_rows == served.n_rows);
    R97_CHECK((int64_t)n_vocab == served.n_vocab);
    R97_CHECK(logits.size() == served.logits.size());
    R97_CHECK(std::memcmp(logits.data(), served.logits.data(), logits.size() * 4) == 0);

    // Then the two numbers the logits score, which are what `r97train loss` prints.
    uint32_t           n_rows = 0;
    std::vector<float> reference = read_reference(rows_file, &n_rows);
    R97_CHECK((int64_t)n_rows == served.n_rows);
    R97_CHECK(reference.size() == served.rows.size());
    R97_CHECK(std::memcmp(reference.data(), served.rows.data(), reference.size() * 4) == 0);
}
