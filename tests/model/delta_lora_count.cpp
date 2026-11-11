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

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace {

constexpr int64_t kWant = 108789760;
constexpr int64_t kRank = 16;

void delta_lora_parameter_count() {
    const std::string rad = r97test::container_or_skip(R97_MODEL_RAD);

    const r97::ModelMeta meta = r97::read_model_meta(rad);
    CHECK(meta.layers == 64);
    CHECK(meta.hidden == 5120);
    CHECK(meta.targetable.size() == 16 * 7 + 48 * 6);
    // IQR-V4 rotates every trunk linear by fwht128, and every targeted weight says so, so the
    // adapter written from this meta records the basis radiance's site filter needs
    // (docs/ADAPTER.md, section 4). The bf16 fixture is the natural-basis case, tested on the host.
    CHECK(meta.input_basis == "fwht128");

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

// The addresses the wire record carries on this container, which the tiny fixture only shows in
// part: the weight_name is a weight of the same layer the module path names, and the slots inside
// a shared weight are 0..n-1 in module order -- docs/ADAPTER.md 2.1's rule, whatever the container
// fuses (it keeps the 27b's `ssm_inz` and `ffn_gate_up` fused, the fixture does not).
void delta_lora_wire_addresses() {
    const std::string rad = r97test::container_or_skip(R97_MODEL_RAD);
    const r97::ModelMeta meta = r97::read_model_meta(rad);

    const std::string kPath = "model.layers.";
    std::map<std::string, std::vector<int64_t>> slots;  // weight_name -> the slots seen
    int64_t delta_in = 0;
    for (const r97::ModelMeta::Linear& lin : meta.targetable) {
        CHECK(lin.module.starts_with(kPath));
        const std::string rest = lin.module.substr(kPath.size());
        const std::string blk = "blk." + rest.substr(0, rest.find('.')) + ".";
        // The layer number in the weight names the same layer as the module path does.
        CHECK(lin.weight_name.starts_with(blk));
        slots[lin.weight_name].push_back(lin.slot);
        if (lin.module.ends_with(".linear_attn.in_proj_qkv") ||
            lin.module.ends_with(".linear_attn.in_proj_z"))
            ++delta_in;
    }

    int64_t shared = 0;
    for (auto& [weight, seen] : slots) {
        CHECK(!seen.empty());
        if (seen.size() == 1) {
            CHECK(seen[0] == 0);  // a weight with one targeted source needs no slot
            continue;
        }
        ++shared;
        std::vector<int64_t> want(seen.size());
        for (size_t i = 0; i < want.size(); ++i) want[i] = int64_t(i);
        std::sort(seen.begin(), seen.end());
        CHECK(seen == want);  // the shared weight's slots are 0..n-1, each once
    }
    CHECK(delta_in == 2 * 48);  // one qkv and one z module per delta-net layer, in 48 layers
    CHECK(shared > 0);          // this container fuses at least one weight pair
}

}  // namespace

R97_SELF_CHECK(delta_lora_parameter_count)
R97_SELF_CHECK(delta_lora_wire_addresses)
