// tests/host/late_delta_lora.cpp -- the `lora` trainable kind, against docs/ADAPTER.md.
//
// `late_` for the reason tests/host/late_adapter_io.cpp's header gives: these checks run at exit,
// with the tier's older static initialisers ahead of them.
//
// The container is the tiny fixture's (tests/support/late_fixture.h), whose metadata
// read_model_meta turns into 25 targetable linears: four layers, layer 3 full attention, so
// 3 * 6 + 7. The checks:
//
//   * init names and shapes: A [r, K] and B [N, r] per linear, A in PEFT's bound 1/sqrt(K), B zero;
//   * two calls with one seed are bit-identical, and another seed moves A;
//   * a target selects the modules it names, in three spellings, and an unknown one is an error
//     that lists the valid names;
//   * to_wire is one site per targeted linear, carrying that linear's factors, its address (the
//     declared weight it lands in and its slot there) and the document's scale alpha/r;
//   * the registry's list has the kind, and building it by name gives the same Delta.
//
// Label: host (the fixture container needs rad-convert, no card).
#include "support/late_fixture.h"
#include "support/test.h"

#include "r97/adapter_io.h"
#include "r97/config.h"
#include "r97/delta.h"
#include "r97/model.h"
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

// The kind, built the way make_model builds it: the registry's row for the name (r97/model.h).
std::shared_ptr<Delta> lora() {
    Config cfg;
    cfg.delta.kind = "lora";
    return delta_registry().at(cfg.delta.kind)(cfg);
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
    ParamSet params = lora()->init(meta, config(kRank, 8.0), 11);
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
    const ParamSet first = lora()->init(meta, cfg, 99);
    const ParamSet again = lora()->init(meta, cfg, 99);
    const ParamSet other = lora()->init(meta, cfg, 100);

    R97_CHECK(first.size() == again.size());
    for (size_t i = 0; i < first.size(); ++i)
        R97_CHECK(same_values(first[i].value, again[i].value));

    R97_CHECK(other.size() == first.size());
    bool moved = false;
    for (size_t i = 0; i < first.size() && !moved; ++i)
        moved = !same_values(first[i].value, other[i].value);
    R97_CHECK(moved);
}

// A target selects what it names: the leaf, the name inside a block, or the whole path.
R97_TEST(delta_lora_targets_select_modules) {
    const ModelMeta& meta = fixture_meta();

    const ParamSet leaf = lora()->init(meta, config(4, 8.0, {"q_proj"}), 5);
    R97_CHECK(leaf.size() == 2);  // the one full-attention layer
    R97_CHECK(module_of(leaf[0].name) == "model.layers.3.self_attn.q_proj");

    const ParamSet inner = lora()->init(meta, config(4, 8.0, {"self_attn.q_proj"}), 5);
    R97_CHECK(inner.size() == leaf.size());
    for (size_t i = 0; i < leaf.size(); ++i) R97_CHECK(inner[i].name == leaf[i].name);

    const ParamSet whole =
        lora()->init(meta, config(4, 8.0, {"model.layers.0.mlp.down_proj"}), 5);
    R97_CHECK(whole.size() == 2);
    R97_CHECK(whole[0].name == "model.layers.0.mlp.down_proj.lora_A");

    // Overlapping targets name each linear once.
    const ParamSet both =
        lora()->init(meta, config(4, 8.0, {"q_proj", "self_attn.q_proj"}), 5);
    R97_CHECK(both.size() == 2);

    // The delta net's modules on all three of its layers.
    const ParamSet z = lora()->init(meta, config(4, 8.0, {"in_proj_z"}), 5);
    R97_CHECK(z.size() == 3 * 2);
}

// An unknown target is an error that says which names the container does have.
R97_TEST(delta_lora_refuses_an_unknown_target) {
    const ModelMeta& meta = fixture_meta();
    std::string message;
    bool threw = false;
    try {
        lora()->init(meta, config(4, 8.0, {"q_proj", "no_such_proj"}), 5);
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
    auto delta = lora();
    const ParamSet params = delta->init(meta, config(16, 32.0), 3);

    WireBuffers wire;
    delta->to_wire(params, &wire);
    R97_CHECK(wire.sites.size() == meta.targetable.size());
    R97_CHECK(wire.extra.empty());

    for (size_t i = 0; i < wire.sites.size(); ++i) {
        const WireBuffers::Site& site = wire.sites[i];
        const ModelMeta::Linear& lin = meta.targetable[i];
        R97_CHECK(site.module == lin.module);
        R97_CHECK(site.weight_name == lin.weight_name);
        R97_CHECK(site.slot == lin.slot);
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

    auto scaled = lora();
    const ParamSet small = scaled->init(meta, config(4, 16.0, {"q_proj"}), 3);
    WireBuffers wide;
    scaled->to_wire(small, &wide);
    R97_CHECK(wide.sites.size() == 1);
    R97_CHECK(wide.sites[0].scale == 4.0);
}

// The address a site carries is the document's, module by module: the declared weight the module
// lands in and its slot there, so a fused weight's two modules are told apart.
R97_TEST(delta_lora_to_wire_addresses_every_module) {
    const ModelMeta& meta = fixture_meta();
    auto delta = lora();
    WireBuffers wire;
    delta->to_wire(delta->init(meta, config(16, 32.0), 5), &wire);

    auto site_of = [&](const std::string& module) -> const WireBuffers::Site& {
        for (const WireBuffers::Site& site : wire.sites)
            if (site.module == module) return site;
        throw std::runtime_error("no site for " + module);
    };
    // Layers 0..2 are the delta net's, layer 3 the attention block's (docs/ADAPTER.md's table).
    // The tiny fixture keeps `ssm_in` and `ssm_z` apart, so each is a declared weight of its own
    // and both modules sit at slot 0; `ffn_gate_up` is fused, slots 0 and 1.
    R97_CHECK(site_of("model.layers.0.linear_attn.in_proj_qkv").weight_name == "blk.0.ssm_in");
    R97_CHECK(site_of("model.layers.0.linear_attn.in_proj_qkv").slot == 0);
    R97_CHECK(site_of("model.layers.0.linear_attn.in_proj_z").weight_name == "blk.0.ssm_z");
    R97_CHECK(site_of("model.layers.0.linear_attn.in_proj_z").slot == 0);
    R97_CHECK(site_of("model.layers.0.linear_attn.out_proj").weight_name == "blk.0.ssm_out");
    R97_CHECK(site_of("model.layers.0.mlp.gate_proj").weight_name == "blk.0.ffn_gate_up");
    R97_CHECK(site_of("model.layers.0.mlp.gate_proj").slot == 0);
    R97_CHECK(site_of("model.layers.0.mlp.up_proj").weight_name == "blk.0.ffn_gate_up");
    R97_CHECK(site_of("model.layers.0.mlp.up_proj").slot == 1);
    R97_CHECK(site_of("model.layers.3.self_attn.q_proj").weight_name == "blk.3.attn_qg");
    R97_CHECK(site_of("model.layers.3.self_attn.q_proj").slot == 0);
    R97_CHECK(site_of("model.layers.3.mlp.down_proj").weight_name == "blk.3.ffn_down");

    // The address comes from init; a module the delta never initialised has none, and that is an
    // error rather than a site radiance could not place.
    ParamSet stranger;
    Param a;
    a.name = "model.layers.0.mlp.gate_proj.lora_A";
    a.shape = {16, 128};
    a.value.assign(16 * 128, 0.0f);
    Param b;
    b.name = "model.layers.0.mlp.gate_proj.lora_B";
    b.shape = {256, 16};
    b.value.assign(256 * 16, 0.0f);
    stranger.push_back(std::move(a));
    stranger.push_back(std::move(b));

    auto fresh = lora();
    bool threw = false;
    std::string message;
    try {
        WireBuffers unused;
        fresh->to_wire(stranger, &unused);
    } catch (const std::invalid_argument& e) {
        threw = true;
        message = e.what();
    }
    R97_CHECK(threw);
    R97_CHECK(message.find("model.layers.0.mlp.gate_proj") != std::string::npos);

    // The fused weights the tiny fixture does not have: `ssm_inz` holds both delta-net input
    // modules, and the address tells them apart by slot (docs/ADAPTER.md 2.1, 5).
    ModelMeta fused;
    fused.arch = "qwen35";
    fused.hidden = 512;
    fused.layers = 1;
    fused.targetable.push_back(ModelMeta::Linear{"model.layers.0.linear_attn.in_proj_qkv",
                                                 "blk.0.ssm_inz", 0, 512, 288});
    fused.targetable.push_back(ModelMeta::Linear{"model.layers.0.linear_attn.in_proj_z",
                                                 "blk.0.ssm_inz", 1, 512, 96});
    fused.targetable.push_back(ModelMeta::Linear{"model.layers.0.mlp.gate_proj",
                                                 "blk.0.ffn_gate_up", 0, 512, 1024});
    fused.targetable.push_back(ModelMeta::Linear{"model.layers.0.mlp.up_proj",
                                                 "blk.0.ffn_gate_up", 1, 512, 1024});

    WireBuffers fused_wire;
    auto fused_delta = lora();
    fused_delta->to_wire(
        fused_delta->init(fused, config(4, 8.0, {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
                                                "mlp.gate_proj", "mlp.up_proj"}),
                          3),
        &fused_wire);
    R97_CHECK(fused_wire.sites.size() == 4);
    struct Slot {
        const char* module;
        const char* weight;
        int64_t slot;
    };
    const Slot want[] = {
        {"model.layers.0.linear_attn.in_proj_qkv", "blk.0.ssm_inz", 0},
        {"model.layers.0.linear_attn.in_proj_z", "blk.0.ssm_inz", 1},
        {"model.layers.0.mlp.gate_proj", "blk.0.ffn_gate_up", 0},
        {"model.layers.0.mlp.up_proj", "blk.0.ffn_gate_up", 1},
    };
    for (const Slot& s : want) {
        const WireBuffers::Site* site = nullptr;
        for (const WireBuffers::Site& candidate : fused_wire.sites)
            if (candidate.module == s.module) site = &candidate;
        R97_CHECK(site != nullptr);
        R97_CHECK(site->weight_name == s.weight);
        R97_CHECK(site->slot == s.slot);
    }
}

// The registration: the registry's list has the kind, and building it by name gives a Delta with no
// module prefix of its own (each parameter's name carries its module).
R97_TEST(delta_lora_registers_its_kind) {
    const std::vector<std::string> kinds = delta_names();
    R97_CHECK(std::find(kinds.begin(), kinds.end(), "lora") != kinds.end());

    const std::shared_ptr<Delta> delta = lora();
    R97_CHECK(delta != nullptr);
    R97_CHECK(std::string(delta->grad_prefix()).empty());
}

// A degenerate config is an error, not a bad ParamSet.
R97_TEST(delta_lora_refuses_a_bad_config) {
    const ModelMeta& meta = fixture_meta();
    for (const DeltaConfig& bad : {config(0, 8.0), config(-1, 8.0), config(4, std::nan(""))}) {
        bool threw = false;
        try {
            lora()->init(meta, bad, 1);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        R97_CHECK(threw);
    }
}
