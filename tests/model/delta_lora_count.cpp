// tests/model/delta_lora_count.cpp -- LoraDelta::init on the real container, at the document's
// rank, gives the parameter count docs/ADAPTER.md's worked example states.
//
// docs/ADAPTER.md, section 6: 108,789,760 parameters at rank 16 over every targetable linear --
// 16 * 1,736,704 on the sixteen full-attention layers, 48 * 1,687,552 on the forty-eight
// delta-net layers. The count only needs the container's metadata (read_model_meta) and the
// ParamSet's own sizes, so no card is involved.
//
// It names the kind's factory (../src/delta_lora.h) rather than looking it up: the model tier links
// r97 as it comes, so the kind's file is in this binary only because this test mentions a symbol in
// it, and an archive link keeps it only for that reason (docs/OBJECTIVES.md, "Where a registration
// runs").
//
// Label: model. Skipped when the container is absent.
#include "r97/adapter_io.h"
#include "r97/config.h"
#include "r97/delta.h"
#include "r97/params.h"

#include "../src/delta_lora.h"
#include "../support/model_dir.h"
#include "../support/self_check.h"

#include <cstdint>
#include <cstdio>
#include <string>

namespace {

constexpr int64_t kWant = 108789760;
constexpr int64_t kRank = 16;

void delta_lora_parameter_count() {
    const std::string rad = r97test::container_or_skip(R97_MODEL_RAD);

    const r97::ModelMeta meta = r97::read_model_meta(rad);
    CHECK(meta.layers == 64);
    CHECK(meta.hidden == 5120);
    CHECK(meta.targetable.size() == 16 * 7 + 48 * 6);

    r97::Config config;
    config.delta.kind = "lora";
    config.delta.rank = kRank;
    config.delta.alpha = 32.0;

    const r97::ParamSet params = r97::make_lora_delta(config)->init(meta, config.delta, 1);
    CHECK(params.size() == 2 * meta.targetable.size());

    int64_t total = 0;
    for (const r97::Param& p : params) total += int64_t(p.value.size());
    if (total != kWant)
        std::fprintf(stderr, "delta_lora_count: %lld parameters at rank %lld, docs/ADAPTER.md says %lld\n",
                     (long long)total, (long long)kRank, (long long)kWant);
    CHECK(total == kWant);
}

}  // namespace

R97_SELF_CHECK(delta_lora_parameter_count)
