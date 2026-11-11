// tests/host/live_adapter.cpp -- a delta's WireBuffers in radiance's LoRA slot on the bf16 fixture.
// Label: host; needs the fixture's prefill (patch 0103).
//
// A test registered with R97_TEST runs from an atexit handler, after namespace-scope objects have
// been destroyed, so this file keeps its targets in a function rather than in a static vector.
#include "support/fixture.h"
#include "support/test.h"

#include "r97/live_adapter.h"

#include "lora/adapter.h"
#include "tiny_model.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#if R97_FIXTURE_PREFILL
namespace {

std::vector<std::string> targets() { return { "q_proj", "v_proj" }; }
constexpr int    kRank  = 4;
constexpr double kAlpha = 8.0;
const char* const kName = "live";

rad::Config config(const r97::Fixture& f, const std::vector<std::vector<int32_t>>& sequences,
                   bool lora, const std::string& adapter_dir) {
    rad::Config cfg;
    cfg.model            = f.container;
    cfg.radiance_home    = r97::detail::radiance_home();
    cfg.kernel_hierarchy = { "libavx", "libref" };
    cfg.n_spec           = 0;
    cfg.max_tok          = 512;
    cfg.max_seqs         = 8;
    cfg.max_ctx          = 512;
    cfg.prefill_prompts  = sequences;
    if (lora) {
        cfg.lora.max_loras = 1;
        cfg.lora.max_rank  = kRank;
        cfg.lora.targets   = targets();
        if (!adapter_dir.empty()) cfg.lora.adapters.push_back({ kName, adapter_dir });
        cfg.prefill_lora = kName;
    }
    return cfg;
}

std::unique_ptr<rad::Engine> start(const r97::Fixture& f,
                                   const std::vector<std::vector<int32_t>>& sequences, bool lora,
                                   const std::string& adapter_dir) {
    std::unique_ptr<rad::Engine> engine(new rad::Engine(config(f, sequences, lora, adapter_dir)));
    const int s = engine->start();
    if (s < 0)
        throw std::runtime_error(std::string("live_adapter: engine start: ") + rad_strerror(s));
    return engine;
}

std::vector<std::vector<float>> prefill(rad::Engine& engine) {
    auto watch = std::make_unique<rad::StallWatch>("live adapter prefill", 60000);
    const int s = engine.serve(std::move(watch));
    if (s < 0)
        throw std::runtime_error(std::string("live_adapter: prefill: ") + rad_strerror(s));
    return engine.config().prefill_logits;
}

std::vector<std::vector<float>> prefill_once(
    const r97::Fixture& f, const std::vector<std::vector<int32_t>>& sequences, bool lora,
    const std::string& adapter_dir) {
    auto engine = start(f, sequences, lora, adapter_dir);
    return prefill(*engine);
}

r97::LiveAdapterConfig live_config() {
    r97::LiveAdapterConfig cfg;
    cfg.name    = kName;
    cfg.targets = targets();
    cfg.rank    = kRank;
    cfg.alpha   = kAlpha;
    return cfg;
}

// Read the adapter radiance would load back into the flat layout a delta writes. This is the
// "by hand" adapter-files side of the acceptance: the same factors, spelled as WireBuffers.
r97::WireBuffers wire_from(const std::string& dir) {
    rad::lora::AdapterFile file;
    std::string err;
    const int s = file.open(dir + "/adapter_model.safetensors", &err);
    if (s < 0) throw std::runtime_error("live_adapter: adapter file: " + err);

    r97::WireBuffers wire;
    for (const rad::lora::FileModule& module : file.modules()) {
        if (module.kind != rad::lora::FileModule::LORA) continue;
        std::string name = module.key;
        const std::string prefix = "base_model.model.";
        if (name.compare(0, prefix.size(), prefix) == 0) name.erase(0, prefix.size());

        r97::WireBuffers::Site site;
        site.module = name;
        site.rank   = module.t[0]->shape[0];
        site.scale  = (float)(kAlpha / (double)kRank);
        const int64_t a_n = tiny::numel(module.t[0]->shape);
        const int64_t b_n = tiny::numel(module.t[1]->shape);
        site.a.resize((size_t)a_n);
        site.b.resize((size_t)b_n);
        for (int64_t i = 0; i < a_n; ++i)
            site.a[(size_t)i] =
                rad_load_f32(module.t[0]->data, module.t[0]->dtype, (size_t)i);
        for (int64_t i = 0; i < b_n; ++i)
            site.b[(size_t)i] =
                rad_load_f32(module.t[1]->data, module.t[1]->dtype, (size_t)i);
        wire.sites.push_back(std::move(site));
    }
    if (wire.sites.empty()) throw std::runtime_error("live_adapter: adapter has no LoRA modules");
    return wire;
}

r97::WireBuffers zero_b(r97::WireBuffers wire) {
    for (r97::WireBuffers::Site& site : wire.sites)
        for (float& v : site.b) v = 0.0f;
    return wire;
}

void same_logits(const std::vector<std::vector<float>>& got,
                 const std::vector<std::vector<float>>& want, const char* what) {
    R97_CHECK(got.size() == want.size());
    for (size_t p = 0; p < got.size(); ++p) {
        R97_CHECK(got[p].size() == want[p].size());
        if (got[p] != want[p]) {
            size_t i = 0;
            while (i < got[p].size() && got[p][i] == want[p][i]) ++i;
            throw std::runtime_error(std::string("live_adapter: ") + what + " differs at prompt " +
                                     std::to_string(p) + ", logit " + std::to_string(i));
        }
    }
}

void test_memory_equals_directory_and_base() {
    const r97::Fixture& f = r97::fixture();
    const std::vector<std::vector<int32_t>> sequences = { { 1, 7, 11, 19 }, { 2, 3 } };

    const std::string dir =
        (std::filesystem::path(f.checkpoint).parent_path() / "live_adapter").string();
    std::filesystem::remove_all(dir);
    R97_CHECK(tiny::write_lora(dir, f.checkpoint, targets(), kRank, kAlpha, 29, 0.25f));

    const r97::WireBuffers wire = wire_from(dir);

    const auto base      = prefill_once(f, sequences, false, "");
    const auto directory = prefill_once(f, sequences, true, dir);

    std::string err;
    std::vector<std::vector<float>> memory;
    {
        auto engine = start(f, sequences, true, "");
        auto adapter = r97::LiveAdapter::open(*engine, live_config(), &err);
        R97_CHECK(adapter != nullptr);
        R97_CHECK(adapter->update(wire, &err) == 0);
        memory = prefill(*engine);
    }
    same_logits(memory, directory, "memory adapter and directory adapter");

    std::vector<std::vector<float>> zero_b_logits;
    {
        auto engine = start(f, sequences, true, "");
        auto zero = r97::LiveAdapter::open(*engine, live_config(), &err);
        R97_CHECK(zero != nullptr);
        R97_CHECK(zero->update(zero_b(wire), &err) == 0);
        zero_b_logits = prefill(*engine);
    }
    same_logits(zero_b_logits, base, "B = 0 adapter and the base model");
}

void test_update_replaces_the_slot() {
    const r97::Fixture& f = r97::fixture();
    const std::vector<std::vector<int32_t>> sequences = { { 3, 5, 8 } };

    const std::string a_dir =
        (std::filesystem::path(f.checkpoint).parent_path() / "live_adapter_a").string();
    const std::string b_dir =
        (std::filesystem::path(f.checkpoint).parent_path() / "live_adapter_b").string();
    std::filesystem::remove_all(a_dir);
    std::filesystem::remove_all(b_dir);
    R97_CHECK(tiny::write_lora(a_dir, f.checkpoint, targets(), kRank, kAlpha, 31, 0.25f));
    R97_CHECK(tiny::write_lora(b_dir, f.checkpoint, targets(), kRank, kAlpha, 32, 0.5f));

    std::string err;
    std::vector<std::vector<float>> updated;
    {
        auto engine = start(f, sequences, true, "");
        auto adapter = r97::LiveAdapter::open(*engine, live_config(), &err);
        R97_CHECK(adapter != nullptr);
        R97_CHECK(adapter->update(wire_from(a_dir), &err) == 0);
        R97_CHECK(adapter->update(wire_from(b_dir), &err) == 0);
        updated = prefill(*engine);
    }

    same_logits(updated, prefill_once(f, sequences, true, b_dir),
                "updated adapter and the same adapter loaded fresh");
}

void test_ops_and_refusal() {
    const r97::Fixture& f = r97::fixture();
    const std::vector<std::vector<int32_t>> sequences = { { 4, 6 } };
    auto engine = start(f, sequences, true, "");
    std::string err;
    auto adapter = r97::LiveAdapter::open(*engine, live_config(), &err);
    R97_CHECK(adapter != nullptr);
    R97_CHECK(adapter->slot() >= 0);
    R97_CHECK(adapter->rank() == kRank);
    R97_CHECK(adapter->alpha() == kAlpha);
    R97_CHECK(adapter->name() == kName);

    const rad::Program& program = engine->program();
    R97_CHECK(!program.lora_sites.empty());
    size_t shrink = 0, expand = 0;
    for (const rad::LoraIssueSite& site : program.lora_sites) {
        R97_CHECK(site.shrink < program.ops.size());
        R97_CHECK(site.expand < program.ops.size());
        if (program.ops[site.shrink].op == "lora_shrink") ++shrink;
        if (program.ops[site.expand].op == "lora_expand") ++expand;
    }
    R97_CHECK(shrink == program.lora_sites.size());
    R97_CHECK(expand == program.lora_sites.size());

    r97::WireBuffers wire;
    wire.extra.push_back(r97::Param{ "not-a-lora-weight", { 1 }, { 1.0f } });
    R97_CHECK(adapter->update(wire, &err) < 0);
    R97_CHECK(err.find("extra") != std::string::npos);
}

void live_adapter_tests() {
    test_memory_equals_directory_and_base();
    test_update_replaces_the_slot();
    test_ops_and_refusal();
    std::puts("live_adapter ok");
}

}  // namespace

R97_TEST(live_adapter) {
    live_adapter_tests();
}
#endif  // R97_FIXTURE_PREFILL
