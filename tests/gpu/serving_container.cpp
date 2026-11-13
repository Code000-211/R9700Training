// tests/gpu/serving_container.cpp -- the real container's prefill through the card kernels.
//
// A serving-only radiance has to bring the real container up on the R9700, not only the tiny
// fixture's host kernels. This is a model case (it needs the container; R97_MODEL_CASES registers
// it, and it skips 77 when the container is absent) run with the card held, and it takes the
// device tier: the ordered hierarchy puts "libr4d" first.
//
// The prompt is four ids below the container's vocabulary. Two runs of the same prompt must agree
// bit for bit; the serving set's prefill (patches/serving/0202) is what hands the rows back, and
// the model test (tests/model/serving_container.cpp) checks what the container exposes to a serve
// on the host, so a serving build covers both tiers.
//
// Label: model (via tests/support/model_case.h). docs/DEPLOYMENT.md §4 has the card's place in the
// deployment.
#include "../support/gpu_guard.h"
#include "../support/model_case.h"
#include "../support/serve_prefill.h"

#if R97_FIXTURE_PREFILL

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const std::vector<int32_t>& prompt() {
    static const std::vector<int32_t> ids = { 1, 2, 3, 4 };
    return ids;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

bool finite(const std::vector<float>& a) {
    for (float v : a)
        if (!std::isfinite(v)) return false;
    return true;
}

}  // namespace

R97_MODEL_TEST(serving_container_prefill_runs_on_the_card) {
    const std::string container = r97test::container_or_skip();

    r97test::GuardOptions o;
    const char* de = std::getenv("R97_DEVICE");
    o.device    = r97test::parse_device(de ? de : "0");
    o.need_vram = int64_t(20) << 30;  // the 27B container: ~15.1 GiB weights, plus KV and arena
    o.bdf       = r97test::hip_bdf(o.device);
    r97test::DeviceGuard guard(o);

    const std::vector<std::string> tier{ "libr4d", "libavx", "libref" };
    const r97::serve_prefill::Logits first =
        r97::serve_prefill::run_on(container, prompt(), "", 16, tier);
    const r97::serve_prefill::Logits second =
        r97::serve_prefill::run_on(container, prompt(), "", 16, tier);
    R97_CHECK(first.n_tokens == (int32_t)prompt().size());
    R97_CHECK(first.n_vocab > 0);
    R97_CHECK(finite(first.logits));
    R97_CHECK(same_bits(first.logits, second.logits));
}

#endif  // R97_FIXTURE_PREFILL
