// tests/model/delta_lora_count.cpp -- LoraDelta::init on the real container, at the document's
// rank, gives the parameter count docs/ADAPTER.md's worked example states.
//
// docs/ADAPTER.md, section 6: 108,789,760 parameters at rank 16 over every targetable linear --
// 16 * 1,736,704 on the sixteen full-attention layers, 48 * 1,687,552 on the forty-eight
// delta-net layers. The count only needs the container's metadata (read_model_meta) and the
// ParamSet's own sizes, so no card is involved.
//
// It runs from an atexit handler rather than an R97_SELF_CHECK initialiser: a delta kind registers
// itself from a static object in the library's own file, and a test object's static initialisers
// run before the library's, so a check that asked the registry there would find it empty.
//
// Label: model. Skipped when the container is absent.
#include "r97/adapter_io.h"
#include "r97/delta.h"
#include "r97/delta_registry.h"
#include "r97/params.h"

#include "../support/model_dir.h"
#include "../support/self_check.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

    r97::DeltaConfig cfg;
    cfg.kind = "lora";
    cfg.rank = kRank;
    cfg.alpha = 32.0;

    const r97::ParamSet params = r97::make_delta("lora")->init(meta, cfg, 1);
    CHECK(params.size() == 2 * meta.targetable.size());

    int64_t total = 0;
    for (const r97::Param& p : params) total += int64_t(p.value.size());
    if (total != kWant)
        std::fprintf(stderr, "delta_lora_count: %lld parameters at rank %lld, docs/ADAPTER.md says %lld\n",
                     (long long)total, (long long)kRank, (long long)kWant);
    CHECK(total == kWant);
}

void run_at_exit() { r97test::run("delta_lora_parameter_count", delta_lora_parameter_count); }

[[maybe_unused]] const int registered = std::atexit(run_at_exit);

}  // namespace
