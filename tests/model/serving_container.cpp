// tests/model/serving_container.cpp -- the real container's serving handshake, on the host tier.
//
// A serve has two halves that must agree before any card is involved: the container's trunk is
// stored under a transform the adapter is written for (`radiance_input_basis`), and every module
// the adapter names is a linear the container declares. Both are read from the container's own
// metadata through rad_core -- no engine, no plugin, no card (r97/adapter_io.h's
// read_model_meta). The serving prefill itself is on the card: tests/gpu/serving_container.cpp
// runs it under the model label, with the card guard.
//
// The adapter the *trained* run is served with is the fixture the trainer wrote
// (tests/fixtures/serving_adapter, its README has the run). Its factors are the tiny fixture's
// shapes, so this test checks names and basis, not dimensions. The card's serve has no trained
// 27B adapter to load in the tree, so tests/gpu/serving_container.cpp writes a rank-4 `down_proj`
// adapter for the linears read here and loads it: radiance accepts it only if its shapes are the
// container's, which is the dimension check, and the served loss differing from the base's is
// what says the factors reached the trunk.
//
// Label: model. Skipped (exit 77) when the container is absent; docs/DEPLOYMENT.md §4 says where
// the container and the card sit.
#include "../support/model_dir.h"
#include "../support/test.h"

#include <algorithm>

#include "r97/adapter_io.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

std::string serving_adapter() {
    return (std::filesystem::path(__FILE__).parent_path().parent_path() / "fixtures" /
            "serving_adapter")
        .string();
}

// "model.layers.5.mlp.gate_proj" targets "gate_proj"; a target is one module name and the
// container's linear is that name at the end of its own path.
bool declares(const r97::ModelMeta& meta, const std::string& target) {
    for (const r97::ModelMeta::Linear& linear : meta.targetable) {
        const std::string& module = linear.module;
        if (module == target) return true;
        if (module.size() > target.size() &&
            module.compare(module.size() - target.size(), target.size(), target) == 0 &&
            module[module.size() - target.size() - 1] == '.')
            return true;
    }
    return false;
}

}  // namespace

R97_TEST(serving_container_can_take_the_trainers_adapter) {
    const std::string container = r97test::container_or_skip(R97_MODEL_RAD);
    const r97::ModelMeta meta = r97::read_model_meta(container);
    R97_CHECK(meta.arch == "qwen35");
    R97_CHECK(meta.layers == 64);
    R97_CHECK(meta.hidden > 0);
    R97_CHECK(!meta.targetable.empty());
    R97_CHECK(meta.input_basis == "fwht128");

    const r97::AdapterFiles adapter = r97::read_adapter(serving_adapter());
    R97_CHECK(adapter.config.input_basis == meta.input_basis);
    R97_CHECK(!adapter.config.targets.empty());
    for (const std::string& target : adapter.config.targets)
        R97_CHECK(declares(meta, target));
    // The card test's adapter targets `down_proj`, and the container must give it a shape: a
    // linear with both dimensions set, and at most one per layer so the adapter's names are the
    // container's own modules.
    std::vector<std::string> layers;
    int64_t                  down_proj = 0;
    for (const r97::ModelMeta::Linear& linear : meta.targetable) {
        const std::string& module = linear.module;
        if (module.size() < 10 || module.compare(module.size() - 10, 10, ".down_proj") != 0)
            continue;
        R97_CHECK(linear.in > 0);
        R97_CHECK(linear.out > 0);
        layers.push_back(module.substr(0, module.size() - 10));
        ++down_proj;
    }
    R97_CHECK(down_proj > 0);
    std::sort(layers.begin(), layers.end());
    R97_CHECK(std::adjacent_find(layers.begin(), layers.end()) == layers.end());
    std::printf("  model: %lld layers, %zu targetable linears, basis %s;"
                " adapter targets %zu modules\n",
                (long long)meta.layers, meta.targetable.size(), meta.input_basis.c_str(),
                adapter.config.targets.size());
}
