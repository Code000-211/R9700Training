// tests/host/serving_prefill.cpp -- a serving build answers a prefill with an adapter written by
// r97train, and the logits are the default build's to the bit.
//
// The two builds this repository makes differ only in the patch sets applied to radiance: the
// default ("all") build carries the trainer's issue observer and replay hook, a serving-only build
// carries neither. If serving computes anything other than what training's radiance computes, the
// adapter a deployment loads does not answer what the trainer measured. This test fixes the
// comparison to bytes: the logits of one prompt with one adapter, against tests/fixtures/.
//
// The reference logits were written by this test in the default build
// (R97_WRITE_SERVING_GOLDEN=<path>); the same test runs in a serving build and must reproduce them
// exactly. The adapter in tests/fixtures/serving_adapter is one r97train wrote on the int8 fixture
// (tests/fixtures/serving_adapter/README.md prints the run). The prefill mode is serving's
// (patches/serving/0202, 0203), so both builds have it and neither needs the observer or replay
// patch (docs/DEPLOYMENT.md §5).
#include "../support/fixture_w8a8.h"
#include "../support/serve_prefill.h"
#include "../support/test.h"

#if R97_FIXTURE_PREFILL

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

// The prompt is short and its ids are below the fixture's vocabulary (320): the reference is then
// a few kilobytes and the prefill a handful of positions.
const std::vector<int32_t>& prompt() {
    static const std::vector<int32_t> ids = { 1, 2, 3, 4, 5, 6, 7, 8 };
    return ids;
}

// On disk: "R97P" then n_tokens and n_vocab as little-endian u32, then the logits as f32. Written
// and read only here, so the layout can be this small.
constexpr char kMagic[4] = { 'R', '9', '7', 'P' };

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

std::vector<float> read_reference(const fs::path& path, int32_t* n_tokens, int32_t* n_vocab) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("serving_prefill: cannot read " + path.string());
    char magic[4];
    in.read(magic, 4);
    if (std::memcmp(magic, kMagic, 4) != 0)
        throw std::runtime_error("serving_prefill: " + path.string() + " is not a logits record");
    *n_tokens = (int32_t)get_u32(in);
    *n_vocab  = (int32_t)get_u32(in);
    std::vector<float> logits((size_t)*n_tokens * (size_t)*n_vocab);
    in.read(reinterpret_cast<char*>(logits.data()), (std::streamsize)(logits.size() * 4));
    if (!in) throw std::runtime_error("serving_prefill: " + path.string() + " is short");
    return logits;
}

void write_reference(const fs::path& path, const r97::serve_prefill::Logits& p) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("serving_prefill: cannot write " + path.string());
    out.write(kMagic, 4);
    put_u32(out, (uint32_t)p.n_tokens);
    put_u32(out, (uint32_t)p.n_vocab);
    out.write(reinterpret_cast<const char*>(p.logits.data()),
              (std::streamsize)(p.logits.size() * 4));
    if (!out) throw std::runtime_error("serving_prefill: short write to " + path.string());
}

}  // namespace

R97_TEST(serving_prefill_with_the_fixture_adapter_matches_the_reference) {
    const r97::Fixture f = r97::w8a8_fixture();
    const fs::path adapter = fixtures() / "serving_adapter";
    R97_CHECK(fs::exists(adapter / "adapter_config.json"));

    const r97::serve_prefill::Logits served =
        r97::serve_prefill::run_on(f.container, prompt(), adapter.string(), 16,
                                   r97::serve_prefill::host_hierarchy());

    // Prove the adapter was loaded and used: without it the same prefill must compute something
    // else, or the comparison below would pass on a build that ignored the adapter.
    const r97::serve_prefill::Logits plain =
        r97::serve_prefill::run_on(f.container, prompt(), "", 16,
                                   r97::serve_prefill::host_hierarchy());
    bool changed = false;
    for (size_t i = 0; i < served.logits.size() && !changed; ++i)
        changed = std::memcmp(&served.logits[i], &plain.logits[i], sizeof(float)) != 0;
    R97_CHECK(changed);

    const fs::path reference_file = fixtures() / "serving_prefill_logits.bin";
    if (const char* write = std::getenv("R97_WRITE_SERVING_GOLDEN")) {
        write_reference(write, served);
        std::printf("serving_prefill: wrote the reference logits to %s\n", write);
        return;
    }

    int32_t            n_tokens = 0;
    int32_t            n_vocab  = 0;
    std::vector<float> reference = read_reference(reference_file, &n_tokens, &n_vocab);
    R97_CHECK(n_tokens == served.n_tokens);
    R97_CHECK(n_vocab == served.n_vocab);
    R97_CHECK(reference.size() == served.logits.size());
    R97_CHECK(std::memcmp(reference.data(), served.logits.data(), reference.size() * 4) == 0);
}

#endif  // R97_FIXTURE_PREFILL
