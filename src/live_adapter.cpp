// src/live_adapter.cpp -- WireBuffers into radiance's runtime LoRA slot.
#include "r97/live_adapter.h"

#include "engine.h"
#include "lora/lora.h"
#include "rad_core.h"
#include "rad_plugin.h"

#include <cmath>
#include <map>
#include <utility>

namespace r97 {
namespace {

int fail(std::string* err, const std::string& message, int code = RAD_E_INVAL) {
    if (err) *err = message;
    return code;
}

std::string module_leaf(const std::string& module) {
    const size_t dot = module.rfind('.');
    return dot == std::string::npos ? module : module.substr(dot + 1);
}

bool is_target(const std::vector<std::string>& targets, const std::string& module) {
    const std::string leaf = module_leaf(module);
    for (const std::string& target : targets)
        if (target == leaf) return true;
    return false;
}

// Pack the wire's sites into radiance's in-memory adapter form. `as`/`bs` own the bf16 bytes the
// StTensors point at; both outlive the caller's lora_set_weights() call.
int pack(const WireBuffers& wire, const LiveAdapterConfig& config,
         const std::vector<std::vector<uint16_t>>& as, const std::vector<std::vector<uint16_t>>& bs,
         std::vector<rad::StTensor>& at, std::vector<rad::StTensor>& bt,
         std::map<std::string, rad::lora::AdModule>* mods, std::string* err) {
    for (size_t i = 0; i < wire.sites.size(); ++i) {
        const WireBuffers::Site& site = wire.sites[i];
        if (site.module.empty())
            return fail(err, "WireBuffers site " + std::to_string(i) + " has no module");
        if (site.rank <= 0)
            return fail(err, "WireBuffers site " + site.module + " has rank " +
                                 std::to_string(site.rank));
        if (site.rank > config.rank)
            return fail(err, "WireBuffers site " + site.module + " has rank " +
                                 std::to_string(site.rank) + ", above the slot's rank " +
                                 std::to_string(config.rank));
        if (site.a.size() % (size_t)site.rank != 0 || site.b.size() % (size_t)site.rank != 0 ||
            site.a.empty() || site.b.empty())
            return fail(err, "WireBuffers site " + site.module + " does not hold rank-" +
                                 std::to_string(site.rank) + " A and B factors");
        if (!std::isfinite(site.scale))
            return fail(err, "WireBuffers site " + site.module + " has a non-finite scale");
        if (!is_target(config.targets, site.module))
            return fail(err, "WireBuffers site " + site.module +
                                 " is not one of this adapter's targets");

        const int64_t in = (int64_t)(site.a.size() / (size_t)site.rank);
        const int64_t out = (int64_t)(site.b.size() / (size_t)site.rank);
        at[i].name = site.module + ".lora_A.weight";
        at[i].dtype = RAD_BF16;
        at[i].shape = {site.rank, in};
        at[i].data = reinterpret_cast<const uint8_t*>(as[i].data());
        bt[i].name = site.module + ".lora_B.weight";
        bt[i].dtype = RAD_BF16;
        bt[i].shape = {out, site.rank};
        bt[i].data = reinterpret_cast<const uint8_t*>(bs[i].data());

        rad::lora::AdModule am;
        am.name = site.module;
        am.A = &at[i];
        am.B = &bt[i];
        am.rank = site.rank;
        am.scale = site.scale;
        am.alpha = (double)site.scale * (double)site.rank;
        const auto inserted = mods->emplace(site.module, std::move(am));
        if (!inserted.second)
            return fail(err, "WireBuffers names module " + site.module + " more than once");
    }
    return RAD_OK;
}

}  // namespace

struct LiveAdapter::Impl {
    rad::Engine*     engine = nullptr;
    LiveAdapterConfig config;
    int32_t          slot = -1;

    int put(const WireBuffers& wire, std::string* err) {
        if (!wire.extra.empty())
            return fail(err, "WireBuffers::extra is not empty: no delta has a non-LoRA slot yet");
        if (!std::isfinite(config.alpha))
            return fail(err, "live adapter " + config.name + " has a non-finite alpha");

        std::map<std::string, rad::lora::AdModule> mods;
        std::vector<std::vector<uint16_t>> as(wire.sites.size()), bs(wire.sites.size());
        std::vector<rad::StTensor> at(wire.sites.size()), bt(wire.sites.size());
        for (size_t i = 0; i < wire.sites.size(); ++i) {
            as[i].resize(wire.sites[i].a.size());
            bs[i].resize(wire.sites[i].b.size());
            for (size_t j = 0; j < wire.sites[i].a.size(); ++j)
                as[i][j] = rad_f32_to_bf16(wire.sites[i].a[j]);
            for (size_t j = 0; j < wire.sites[i].b.size(); ++j)
                bs[i][j] = rad_f32_to_bf16(wire.sites[i].b[j]);
        }
        const int packed =
            pack(wire, config, as, bs, at, bt, &mods, err);
        if (packed < 0) return packed;
        return engine->lora_set_weights(config.name, mods, &slot, err);
    }
};

LiveAdapter::LiveAdapter(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
LiveAdapter::~LiveAdapter() = default;

std::unique_ptr<LiveAdapter> LiveAdapter::open(rad::Engine& engine, const LiveAdapterConfig& config,
                                              std::string* err) {
    if (config.name.empty()) {
        fail(err, "live adapter has no name");
        return nullptr;
    }
    if (config.rank <= 0) {
        fail(err, "live adapter " + config.name + " has no rank");
        return nullptr;
    }
    if (!std::isfinite(config.alpha)) {
        fail(err, "live adapter " + config.name + " has a non-finite alpha");
        return nullptr;
    }
    if (config.targets.empty()) {
        fail(err, "live adapter " + config.name + " names no targets");
        return nullptr;
    }
    const rad::Config& cfg = engine.config();
    if (!cfg.lora.enabled()) {
        fail(err, "live adapter " + config.name +
                      " needs runtime LoRA: start the engine with Config::lora targets, rank and "
                      "at least one slot");
        return nullptr;
    }
    if (cfg.lora.targets.empty()) {
        fail(err, "live adapter " + config.name +
                      " needs Config::lora.targets at engine start; the engine has no LoRA sites");
        return nullptr;
    }
    const int64_t rank = cfg.lora.rank(false);
    if (config.rank > rank) {
        fail(err, "live adapter " + config.name + " asks for rank " +
                      std::to_string(config.rank) + ", above the engine's slot rank " +
                      std::to_string(rank));
        return nullptr;
    }
    for (const std::string& target : config.targets) {
        bool found = false;
        for (const std::string& engine_target : cfg.lora.targets)
            found = found || engine_target == target;
        if (!found) {
            fail(err, "live adapter " + config.name + " targets " + target +
                          ", which the engine was not started for");
            return nullptr;
        }
    }

    auto impl = std::make_unique<Impl>();
    impl->engine = &engine;
    impl->config = config;
    std::map<std::string, rad::lora::AdModule> none;
    const int opened = engine.lora_set_weights(config.name, none, &impl->slot, err);
    if (opened < 0) return nullptr;
    return std::unique_ptr<LiveAdapter>(new LiveAdapter(std::move(impl)));
}

int LiveAdapter::update(const WireBuffers& wire, std::string* err) {
    return impl_->put(wire, err);
}

int32_t LiveAdapter::slot() const { return impl_->slot; }
int64_t LiveAdapter::rank() const { return impl_->config.rank; }
double LiveAdapter::alpha() const { return impl_->config.alpha; }
const std::string& LiveAdapter::name() const { return impl_->config.name; }

}  // namespace r97
