// tests/host/delta_lora.cpp -- the `lora` trainable kind, against docs/ADAPTER.md.
//
// The container is the tiny fixture's (tests/support/late_fixture.h), whose metadata read_model_meta
// turns into 25 targetable linears: four layers, layer 3 full attention, so 3 * 6 + 7. The checks:
//
//   * init names and shapes: A [r, K] and B [N, r] per linear, A in PEFT's bound 1/sqrt(K), B zero;
//   * two calls with one seed are bit-identical, and another seed moves A;
//   * a target selects the modules it names, in three spellings, and an unknown one is an error
//     that lists the valid names;
//   * to_wire is one site per targeted linear, carrying that linear's factors, with scale alpha/r;
//   * the registry builds the kind by name and refuses one it does not know.
//
// Label: host (the fixture container needs rad-convert, no card).
#include "support/late_fixture.h"
#include "support/test.h"

#include "r97/adapter_io.h"
#include "r97/delta.h"
#include "r97/delta_registry.h"
#include "r97/params.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace r97;

// The container is converted here rather than at exit: see support/late_fixture.h. The reference
// is what brings that about.
[[maybe_unused]] const std::string& kTmpDir = r97test::late_paths().tmpdir;

const ModelMeta& fixture_meta() {
    static const ModelMeta meta = read_model_meta(r97test::late_container());
    return meta;
}

DeltaConfig config(int64_t rank, double alpha, std::vector<std::string> targets = {}) {
    DeltaConfig cfg;
    cfg.kind = "lora";
    cfg.rank = rank;
    cfg.alpha = alpha;
    cfg.targets = std::move(targets);
    return cfg;
}

bool same_values(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// The module part of a ParamSet name ("<module>.lora_A" -> "<module>").
std::string module_of(const std::string& name) {
    const size_t dot = name.rfind('.');
    return dot == std::string::npos ? name : name.substr(0, dot);
}

const Param* find(const ParamSet& params, const std::string& name) {
    for (const Param& p : params)
        if (p.name == name) return &p;
    return nullptr;
}

}  // namespace

// Names and shapes, for every module the container targets, and PEFT's starting values.
R97_TEST(delta_lora_init_shapes_and_starting_values) {
    const ModelMeta& meta = fixture_meta();
    R97_CHECK(meta.targetable.size() == 25);

    constexpr int64_t kRank = 4;
    ParamSet params = make_delta("lora")->init(meta, config(kRank, 8.0), 11);
    R97_CHECK(params.size() == 2 * meta.targetable.size());

    size_t at = 0;
    for (const ModelMeta::Linear& lin : meta.targetable) {
        const Param& a = params[at++];
        const Param& b = params[at++];

        R97_CHECK(a.name == lin.module + ".lora_A");
        R97_CHECK(a.shape == (std::vector<int64_t>{kRank, lin.in}));
        R97_CHECK(a.value.size() == size_t(kRank * lin.in));
        R97_CHECK(b.name == lin.module + ".lora_B");
        R97_CHECK(b.shape == (std::vector<int64_t>{lin.out, kRank}));
        R97_CHECK(b.value.size() == size_t(lin.out * kRank));

        // PEFT's reset: kaiming_uniform_ with a = sqrt(5) is uniform on (-1/sqrt(K), 1/sqrt(K)),
        // and B is zero so that the adapter starts as a no-op.
        const double bound = 1.0 / std::sqrt(double(lin.in));
        for (float v : a.value) R97_CHECK(std::abs(double(v)) <= bound);
        for (float v : b.value) R97_CHECK(v == 0.0f);
    }
}

// The seed: same seed, same bits; another seed, another A.
R97_TEST(delta_lora_init_is_seeded) {
    const ModelMeta& meta = fixture_meta();
    const DeltaConfig cfg = config(4, 8.0);
    const ParamSet first = make_delta("lora")->init(meta, cfg, 99);
    const ParamSet again = make_delta("lora")->init(meta, cfg, 99);
    const ParamSet other = make_delta("lora")->init(meta, cfg, 100);

    R97_CHECK(first.size() == again.size());
    for (size_t i = 0; i < first.size(); ++i) R97_CHECK(same_values(first[i].value, again[i].value));

    R97_CHECK(other.size() == first.size());
    bool moved = false;
    for (size_t i = 0; i < first.size() && !moved; ++i)
        moved = !same_values(first[i].value, other[i].value);
    R97_CHECK(moved);
}

// A target selects what it names: the leaf, the name inside a block, or the whole path.
R97_TEST(delta_lora_targets_select_modules) {
    const ModelMeta& meta = fixture_meta();

    const ParamSet leaf = make_delta("lora")->init(meta, config(4, 8.0, {"q_proj"}), 5);
    R97_CHECK(leaf.size() == 2);  // the one full-attention layer
    R97_CHECK(module_of(leaf[0].name) == "model.layers.3.self_attn.q_proj");

    const ParamSet inner = make_delta("lora")->init(meta, config(4, 8.0, {"self_attn.q_proj"}), 5);
    R97_CHECK(inner.size() == leaf.size());
    for (size_t i = 0; i < leaf.size(); ++i) R97_CHECK(inner[i].name == leaf[i].name);

    const ParamSet whole =
        make_delta("lora")->init(meta, config(4, 8.0, {"model.layers.0.mlp.down_proj"}), 5);
    R97_CHECK(whole.size() == 2);
    R97_CHECK(whole[0].name == "model.layers.0.mlp.down_proj.lora_A");

    // Overlapping targets name each linear once.
    const ParamSet both = make_delta("lora")->init(meta, config(4, 8.0, {"q_proj", "self_attn.q_proj"}),
                                                  5);
    R97_CHECK(both.size() == 2);

    // The delta net's modules on all three of its layers.
    const ParamSet z = make_delta("lora")->init(meta, config(4, 8.0, {"in_proj_z"}), 5);
    R97_CHECK(z.size() == 3 * 2);
}

// An unknown target is an error that says which names the container does have.
R97_TEST(delta_lora_refuses_an_unknown_target) {
    const ModelMeta& meta = fixture_meta();
    std::string message;
    bool threw = false;
    try {
        make_delta("lora")->init(meta, config(4, 8.0, {"q_proj", "no_such_proj"}), 5);
    } catch (const std::invalid_argument& e) {
        threw = true;
        message = e.what();
    }
    R97_CHECK(threw);
    R97_CHECK(message.find("no_such_proj") != std::string::npos);
    for (const char* valid : {"self_attn.q_proj", "linear_attn.in_proj_qkv", "mlp.gate_proj"})
        R97_CHECK(message.find(valid) != std::string::npos);
}

// to_wire: one site per targeted linear, that linear's own factors, and the document's scale
// alpha/r. A different alpha moves the scale and nothing else.
R97_TEST(delta_lora_to_wire_is_one_site_per_linear) {
    const ModelMeta& meta = fixture_meta();
    auto delta = make_delta("lora");
    const ParamSet params = delta->init(meta, config(16, 32.0), 3);

    WireBuffers wire;
    delta->to_wire(params, &wire);
    R97_CHECK(wire.sites.size() == meta.targetable.size());
    R97_CHECK(wire.extra.empty());

    for (size_t i = 0; i < wire.sites.size(); ++i) {
        const WireBuffers::Site& site = wire.sites[i];
        const ModelMeta::Linear& lin = meta.targetable[i];
        R97_CHECK(site.module == lin.module);
        R97_CHECK(site.rank == 16);
        R97_CHECK(site.scale == 32.0 / 16.0);  // 2, the document's worked example
        const Param* a = find(params, lin.module + ".lora_A");
        const Param* b = find(params, lin.module + ".lora_B");
        R97_CHECK(a != nullptr && b != nullptr);
        R97_CHECK(same_values(site.a, a->value));
        R97_CHECK(same_values(site.b, b->value));
        R97_CHECK(site.a.size() == size_t(16 * lin.in));
        R97_CHECK(site.b.size() == size_t(lin.out * 16));
    }

    auto scaled = make_delta("lora");
    const ParamSet small = scaled->init(meta, config(4, 16.0, {"q_proj"}), 3);
    WireBuffers wide;
    scaled->to_wire(small, &wide);
    R97_CHECK(wide.sites.size() == 1);
    R97_CHECK(wide.sites[0].scale == 4.0);
}

// The registry: the kind builds by name, and a name it does not know is refused with the kinds it
// does have.
R97_TEST(delta_lora_registers_its_kind) {
    const std::vector<std::string> kinds = delta_kinds();
    R97_CHECK(std::find(kinds.begin(), kinds.end(), "lora") != kinds.end());

    const std::shared_ptr<Delta> delta = make_delta("lora");
    R97_CHECK(delta != nullptr);
    R97_CHECK(std::string(delta->grad_prefix()).empty());  // the parameters carry their modules

    std::string message;
    bool threw = false;
    try {
        make_delta("no_such_kind");
    } catch (const std::invalid_argument& e) {
        threw = true;
        message = e.what();
    }
    R97_CHECK(threw);
    R97_CHECK(message.find("lora") != std::string::npos);
}

// A degenerate config is an error, not a bad ParamSet.
R97_TEST(delta_lora_refuses_a_bad_config) {
    const ModelMeta& meta = fixture_meta();
    for (const DeltaConfig& bad : {config(0, 8.0), config(-1, 8.0), config(4, std::nan(""))}) {
        bool threw = false;
        try {
            make_delta("lora")->init(meta, bad, 1);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        R97_CHECK(threw);
    }
}
