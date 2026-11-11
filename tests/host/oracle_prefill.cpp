// tests/host/oracle_prefill.cpp -- the PyTorch reference's logits are radiance's logits.
//
// Without the adapter, transformers' Qwen3_5ForCausalLM in f64 and run_prefill() on the bf16
// fixture are two implementations of the same checkpoint. They cannot agree to the last bit: one
// computes in f64, the other in bf16, and radiance's bf16 delta net stores the within-chunk KKT
// solve as bf16 while transformers does that arithmetic in f32. docs/ORACLE.md records the
// measurement. They must agree in relative L2 norm, per sequence, within bf16 tolerance. A failure
// here is a finding about the fixture, radiance or transformers, not a reference to adjust.
//
// The prefill mode is patches/training/'s, so a serving build compiles this file to nothing.
#include "support/fixture.h"
#include "support/oracle.h"
#include "support/test.h"

#if R97_FIXTURE_PREFILL

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

std::filesystem::path oracle_dir() {
    return std::filesystem::path(__FILE__).parent_path().parent_path() / "fixtures" / "oracle";
}

// Relative L2 norm over every valid logit of a sequence. 5% is bf16 tolerance for this model:
// radiance's delta net keeps the KKT solve in bf16 and its activations are bf16 throughout, while
// the reference is f64. The worst measured on this fixture is 3.1% (sequence 0); a wrong layer or
// a wrong weight moves whole rows and is far above 5%.
constexpr double kRelativeTolerance = 0.05;

std::vector<int32_t> ids_of(const r97::oracle::Tensor& t) {
    std::vector<int32_t> ids;
    ids.reserve(t.data.size());
    for (double v : t.data) ids.push_back(int32_t(v));
    return ids;
}

}  // namespace

R97_TEST(oracle_logits_without_adapter_match_prefill) {
    const r97::oracle::detail::File ref((oracle_dir() / "reference.safetensors").string());
    const r97::Fixture& f = r97::fixture();

    std::vector<std::vector<int32_t>> sequences;
    sequences.reserve(3);
    for (int s = 0; s < 3; ++s) {
        sequences.push_back(ids_of(ref.tensor("seq" + std::to_string(s) + "/input_ids")));
    }
    const std::vector<r97::Prefill> got = r97::run_prefill(f, sequences);

    for (int s = 0; s < 3; ++s) {
        const auto& want = ref.tensor("seq" + std::to_string(s) + "/logits_without_adapter");
        const int64_t n = int64_t(sequences[s].size());
        if (want.shape.size() != 2 || want.shape[0] != n || want.shape[1] != f.n_vocab) {
            std::fprintf(stderr, "oracle_prefill: sequence %d: reference has shape [%lld, %lld], "
                                 "expected [%lld, %d]\n",
                         s, want.shape.size() == 2 ? (long long)want.shape[0] : -1LL,
                         want.shape.size() == 2 ? (long long)want.shape[1] : -1LL, (long long)n,
                         f.n_vocab);
            R97_CHECK(false);
        }
        if (got[size_t(s)].n_tokens != n || got[size_t(s)].n_vocab != f.n_vocab) {
            std::fprintf(stderr, "oracle_prefill: sequence %d: prefill returned %d rows of %d, "
                                 "reference has %lld rows of %d\n",
                         s, got[size_t(s)].n_tokens, got[size_t(s)].n_vocab, (long long)n, f.n_vocab);
            R97_CHECK(false);
        }

        double diff_sq = 0.0;
        double ref_sq = 0.0;
        double worst = 0.0;
        int64_t worst_pos = 0;
        int64_t worst_tok = 0;
        for (int64_t pos = 0; pos < n; ++pos) {
            for (int64_t v = 0; v < f.n_vocab; ++v) {
                const double want_v = want.data[size_t(pos * f.n_vocab + v)];
                const double got_v = double(got[size_t(s)].at(int32_t(pos), int32_t(v)));
                const double diff = got_v - want_v;
                diff_sq += diff * diff;
                ref_sq += want_v * want_v;
                if (std::fabs(diff) > worst) {
                    worst = std::fabs(diff);
                    worst_pos = pos;
                    worst_tok = v;
                }
            }
        }
        const double rel = std::sqrt(diff_sq / ref_sq);
        std::printf("  oracle_prefill: sequence %d (%lld tokens), relative L2 %.4f, worst |diff| "
                    "%.4g at position %lld, token %lld\n",
                    s, (long long)n, rel, worst, (long long)worst_pos, (long long)worst_tok);
        if (!(rel <= kRelativeTolerance)) {
            std::fprintf(stderr,
                         "oracle_prefill: sequence %d: prefill and the f64 reference differ by "
                         "%.1f%% in relative L2, over %.0f%%; worst |diff| %.6g at position %lld, "
                         "token %lld\n",
                         s, 100.0 * rel, 100.0 * kRelativeTolerance, worst, (long long)worst_pos,
                         (long long)worst_tok);
            R97_CHECK(false);
        }
    }
}

#endif  // R97_FIXTURE_PREFILL
