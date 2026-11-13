// tests/host/fixture_w8a8.cpp -- the int8 fixture: the bf16 fixture's checkpoint converted to a
// w8a8_fwht128 container, prefilled through rad_core on the host kernels, and measured against the
// bf16 fixture it came from. The fixture itself, and the shuffled-weight control, are
// tests/support/fixture_w8a8.h; docs/FIXTURES.md has the recipe and the ops the host issues.
//
// Like tests/host/fixture_prefill.cpp this is the serving prefill patch's mode, so it runs in a
// serving build too; a build that skips the serving set compiles the file to nothing rather than
// naming it by a build rule.
#include "format/radfile.h"
#include "support/fixture_w8a8.h"
#include "support/test.h"

#if R97_FIXTURE_PREFILL

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// A prompt of byte ids, the same shape and stride the bf16 fixture's tests use.
std::vector<int32_t> prompt(size_t n, int32_t first) {
    std::vector<int32_t> ids(n);
    for (size_t i = 0; i < n; ++i) ids[i] = (first + 7 * (int32_t)i) % 256;
    return ids;
}

// One row's next-token distribution as log-probabilities, subtracted from its maximum so a row
// whose logits are all near 1e3 cannot overflow exp().
std::vector<double> log_probs(const float* row, int32_t n) {
    double m = -std::numeric_limits<double>::infinity();
    for (int32_t i = 0; i < n; ++i) m = std::max(m, (double)row[i]);
    double sum = 0.0;
    for (int32_t i = 0; i < n; ++i) sum += std::exp((double)row[i] - m);
    const double log_sum = m + std::log(sum);
    std::vector<double> lp((size_t)n);
    for (int32_t i = 0; i < n; ++i) lp[(size_t)i] = (double)row[i] - log_sum;
    return lp;
}

// KL(p || q) of two log-softmaxed rows.
double kl(const std::vector<double>& lp, const std::vector<double>& lq) {
    double s = 0.0;
    for (size_t i = 0; i < lp.size(); ++i) s += std::exp(lp[i]) * (lp[i] - lq[i]);
    return s;
}

// The mean and largest per-position KL from `a`'s next-token distributions to `b`'s.
void kl_from(const r97::Prefill& a, const r97::Prefill& b, double* mean, double* worst) {
    if (a.n_tokens != b.n_tokens || a.n_vocab != b.n_vocab)
        throw std::runtime_error("fixture_w8a8: prefills differ in shape");
    *mean = 0.0;
    *worst = 0.0;
    for (int32_t t = 0; t < a.n_tokens; ++t) {
        const double d =
            kl(log_probs(&a.logits[(size_t)t * (size_t)a.n_vocab], a.n_vocab),
               log_probs(&b.logits[(size_t)t * (size_t)b.n_vocab], b.n_vocab));
        if (!std::isfinite(d)) throw std::runtime_error("fixture_w8a8: a KL is not finite");
        *mean += d;
        *worst = std::max(*worst, d);
    }
    *mean /= (double)a.n_tokens;
}

// Exact bit equality, as the bf16 fixture's tests check it.
void same_logits(const r97::Prefill& a, const r97::Prefill& b, const std::string& what) {
    if (a.logits.size() != b.logits.size())
        throw std::runtime_error(what + ": " + std::to_string(a.logits.size()) + " logits against " +
                                 std::to_string(b.logits.size()));
    for (size_t i = 0; i < a.logits.size(); ++i) {
        if (std::memcmp(&a.logits[i], &b.logits[i], sizeof(float)) != 0)
            throw std::runtime_error(what + ": the logits differ at " + std::to_string(i) + " (" +
                                     std::to_string(a.logits[i]) + " vs " +
                                     std::to_string(b.logits[i]) + ")");
    }
}

}  // namespace

// The container is the w8a8_fwht128 form, so the qwen35_w8a8 plugin is the one that served it.
R97_TEST(fixture_w8a8_container_names_its_quantisation) {
    const r97::Fixture& f = r97::w8a8_fixture();
    rad::RadFile file;
    R97_CHECK(file.open(f.container.c_str()) == RAD_OK);
    // rad-convert writes `quant` into the metadata and the header; read it as `info` does.
    const char* quant = file.get_s("quant");
    if (!quant) quant = file.str(file.header().quant);
    R97_CHECK(quant != nullptr && std::string(quant) == "w8a8_fwht128");
    std::printf("  fixture_w8a8: the container's quantisation is w8a8_fwht128 ok\n");
}

// A 96-token prefill through the int8 trunk returns every position finite, over the same vocab.
R97_TEST(fixture_w8a8_prefill_logits_are_finite) {
    const r97::Fixture& f = r97::w8a8_fixture();
    const std::vector<int32_t> ids = prompt(96, 3);
    const r97::Prefill p = r97::run_prefill(f, ids);

    R97_CHECK(p.n_tokens == 96);
    R97_CHECK(p.n_vocab == f.n_vocab);
    R97_CHECK(p.logits.size() == 96 * (size_t)f.n_vocab);
    for (size_t i = 0; i < p.logits.size(); ++i) R97_CHECK(std::isfinite(p.logits[i]));
    std::printf("  fixture_w8a8: %d positions of one prefill are finite over %d ids ok\n",
                p.n_tokens, p.n_vocab);
}

// Two runs of the same prompt are bit-identical: the host chain is deterministic.
R97_TEST(fixture_w8a8_prefill_is_repeatable) {
    const r97::Fixture& f = r97::w8a8_fixture();
    const std::vector<int32_t> ids = prompt(96, 3);
    same_logits(r97::run_prefill(f, ids), r97::run_prefill(f, ids), "two runs of one prompt");
    std::printf("  fixture_w8a8: two runs of one prompt are bit-identical ok\n");
}

// The int8 fixture is a quantisation of the bf16 one: the KL between their next-token
// distributions is small, and far below the KL to a checkpoint whose matrix rows are shuffled.
R97_TEST(fixture_w8a8_quantises_the_bf16_fixture) {
    const std::vector<int32_t> ids = prompt(96, 3);
    const r97::Prefill bf16 = r97::run_prefill(r97::fixture(), ids);
    const r97::Prefill i8 = r97::run_prefill(r97::w8a8_fixture(), ids);
    const r97::Prefill control = r97::run_prefill(r97::shuffled_fixture(), ids);

    double quant_mean = 0.0, quant_worst = 0.0, control_mean = 0.0, control_worst = 0.0;
    kl_from(bf16, i8, &quant_mean, &quant_worst);
    kl_from(bf16, control, &control_mean, &control_worst);

    std::printf("  fixture_w8a8: KL(bf16 || w8a8) mean %.3e max %.3e; shuffled control mean %.3e "
                "max %.3e (%.0fx) ok\n",
                quant_mean, quant_worst, control_mean, control_worst,
                control_mean / quant_mean);
    R97_CHECK(quant_mean < 0.05);
    // A quantisation keeps the model; a shuffled one does not. Twenty times is a wide margin
    // against the ~1e-2 a good int8 trunk costs and the ~1e1 an unrelated model costs.
    R97_CHECK(quant_mean * 20.0 < control_mean);
}

#endif  // R97_FIXTURE_PREFILL
