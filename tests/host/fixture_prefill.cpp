// tests/host/fixture_prefill.cpp -- the tiny Qwen3.5 fixture prefills through rad_core, and a
// batch gives each sequence the rows that sequence gives alone. The fixture itself, and what it
// holds, is tests/support/fixture.h; docs/FIXTURES.md says how a test obtains it.
//
// The prefill mode is patches/serving/'s, so this test runs in either build; a build that skips
// the serving set has no run_prefill(), and this file then compiles to nothing rather than being
// named by a build rule of its own.
#include "format/radfile.h"
#include "support/fixture.h"
#include "support/test.h"

#if R97_FIXTURE_PREFILL

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// A prompt of byte ids: the tiny tokenizer is byte-level, so ids 0..255 are the bytes.
std::vector<int32_t> prompt(size_t n, int32_t first) {
    std::vector<int32_t> ids(n);
    for (size_t i = 0; i < n; ++i) ids[i] = (first + 7 * (int32_t)i) % 256;
    return ids;
}

// Bit-for-bit equality of two prefills, with `what` in the message: a batch case that compares
// three sequences at once has to say which one disagreed.
void same_logits(const r97::Prefill& a, const r97::Prefill& b, const std::string& what) {
    if (a.logits.size() != b.logits.size())
        throw std::runtime_error(what + ": " + std::to_string(a.logits.size()) + " logits against " +
                                 std::to_string(b.logits.size()));
    if (a.logits.empty()) return;
    for (size_t i = 0; i < a.logits.size(); ++i) {
        if (std::memcmp(&a.logits[i], &b.logits[i], sizeof(float)) != 0)
            throw std::runtime_error(what + ": the logits differ at " + std::to_string(i) + " (" +
                                     std::to_string(a.logits[i]) + " against " +
                                     std::to_string(b.logits[i]) + ")");
    }
}

}  // namespace

// The container is what the fixture promises: the tokenizer configuration carries a chat template,
// so a test that formats a conversation finds one. radiance reads the template from the sibling
// tokenizer_config.json only, never from a key inside tokenizer.json, so this is checked on the
// container the tests load rather than on the checkpoint directory.
R97_TEST(fixture_container_carries_its_chat_template) {
    const r97::Fixture& f = r97::fixture();
    rad::RadFile file;
    R97_CHECK(file.open(f.container.c_str()) == RAD_OK);
    const rad::VocabSection vocab = file.vocab();
    R97_CHECK(bool(vocab));
    R97_CHECK(file.str(vocab.h->chat_template)[0] != '\0');
    std::printf("  fixture: the container's tokenizer carries a chat template ok\n");
}

// A 96-token prefill runs as one step and every position's row comes back finite.
R97_TEST(fixture_prefill_logits_are_finite) {
    const r97::Fixture& f = r97::fixture();
    const std::vector<int32_t> ids = prompt(96, 3);
    const r97::Prefill p = r97::run_prefill(f, ids);

    R97_CHECK(p.n_tokens == 96);
    R97_CHECK(p.n_vocab == f.n_vocab);
    R97_CHECK(p.logits.size() == 96 * (size_t)f.n_vocab);
    R97_CHECK(p.logits.size() == (size_t)p.n_tokens * (size_t)p.n_vocab);
    for (size_t i = 0; i < p.logits.size(); ++i) R97_CHECK(std::isfinite(p.logits[i]));
    std::printf("  fixture: %d positions of one prefill are finite over %d ids ok\n", p.n_tokens,
                p.n_vocab);
}

// Two runs of the same prompt are bit-identical: the fixture depends on nothing but its input.
R97_TEST(fixture_prefill_is_repeatable) {
    const r97::Fixture& f = r97::fixture();
    const std::vector<int32_t> ids = prompt(96, 3);
    same_logits(r97::run_prefill(f, ids), r97::run_prefill(f, ids), "two runs of one prompt");
    std::printf("  fixture: two runs of one prompt are bit-identical ok\n");
}

// Three sequences of different lengths in one batch each get the rows they get alone.
R97_TEST(fixture_prefill_batches_like_solo) {
    const r97::Fixture& f = r97::fixture();
    const std::vector<std::vector<int32_t>> prompts = {prompt(8, 11), prompt(21, 5),
                                                       prompt(40, 29)};
    const std::vector<r97::Prefill> batch = r97::run_prefill(f, prompts);
    R97_CHECK(batch.size() == prompts.size());

    for (size_t i = 0; i < prompts.size(); ++i) {
        const r97::Prefill solo = r97::run_prefill(f, prompts[i]);
        R97_CHECK(batch[i].n_tokens == (int32_t)prompts[i].size());
        same_logits(batch[i], solo,
                    "prompt " + std::to_string(i) + " of " + std::to_string(prompts[i].size()) +
                        " tokens in the batch and alone");
    }
    std::printf("  fixture: three sequences of 8, 21 and 40 tokens keep their solo logits ok\n");
}

#endif  // R97_FIXTURE_PREFILL
