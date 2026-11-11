// src/delta_lora.cpp -- the `lora` trainable kind, registered under the kind "lora".
//
// docs/ADAPTER.md owns this shape: one A [r, K] and one B [N, r] per targeted linear, A from
// PEFT's own initialisation and B zero, and a wire site per linear carrying its factors, its
// address (the declared weight it lands in and its slot there) and the scale alpha/r. The
// ParamSet names are `<module>.lora_A` and `<module>.lora_B`, so the adjoint's `<prefix>.lora_A`
// needs no prefix of its own: grad_prefix() is empty and each site carries its module.
//
// The module names are the ones read_model_meta gives (r97/adapter_io.h): the container's
// targetable linears in PEFT's spelling. A target may name one by its whole path
// (`model.layers.3.self_attn.q_proj`), by the name inside a block (`self_attn.q_proj`, the way
// docs/ADAPTER.md's table lists them) or by its leaf (`q_proj`, the way PEFT writes
// target_modules); anything else is an error that lists the names the container does have.
#include "delta_lora.h"

#include "r97/config.h"
#include "r97/delta.h"
#include "r97/model.h"
#include "r97/params.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace r97 {
namespace {

// "model.layers.<n>.<family>.<leaf>" -> "<family>.<leaf>": the module's name inside its block.
std::string inner_name(const std::string& module) {
    size_t dots = 0;
    for (size_t i = 0; i < module.size(); ++i)
        if (module[i] == '.' && ++dots == 3) return module.substr(i + 1);
    return module;
}

std::string leaf_name(const std::string& module) {
    const size_t dot = module.rfind('.');
    return dot == std::string::npos ? module : module.substr(dot + 1);
}

bool target_names(const std::string& target, const std::string& module) {
    return target == module || target == inner_name(module) || target == leaf_name(module);
}

// The targetable linears the config selects, in the container's own order, each once. An empty
// target list is every targetable linear (docs/ADAPTER.md, section 2).
std::vector<size_t> select(const ModelMeta& meta, const std::vector<std::string>& targets) {
    std::vector<size_t> lines;
    if (targets.empty()) {
        lines.resize(meta.targetable.size());
        for (size_t i = 0; i < lines.size(); ++i) lines[i] = i;
        return lines;
    }

    std::vector<std::string> unknown;
    for (const std::string& target : targets) {
        bool matched = false;
        for (size_t i = 0; i < meta.targetable.size(); ++i) {
            if (!target_names(target, meta.targetable[i].module)) continue;
            lines.push_back(i);
            matched = true;
        }
        if (!matched) unknown.push_back(target);
    }
    if (!unknown.empty()) {
        std::vector<std::string> valid;
        for (const ModelMeta::Linear& lin : meta.targetable) valid.push_back(inner_name(lin.module));
        std::sort(valid.begin(), valid.end());
        valid.erase(std::unique(valid.begin(), valid.end()), valid.end());
        std::string listed;
        for (const std::string& name : valid) listed += (listed.empty() ? "" : ", ") + name;
        throw std::invalid_argument("r97: LoraDelta: unknown target '" + unknown.front() +
                                    "'; the targetable modules are: " +
                                    (listed.empty() ? std::string("none") : listed));
    }
    std::sort(lines.begin(), lines.end());
    lines.erase(std::unique(lines.begin(), lines.end()), lines.end());
    return lines;
}

int64_t product(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    return n;
}

// One trainable kind: PEFT's A/B factors for every targeted linear.
class LoraDelta final : public Delta {
public:
    explicit LoraDelta(double alpha) : alpha_(alpha) {}

    ParamSet init(const ModelMeta& meta, const DeltaConfig& cfg, uint64_t seed) override {
        if (cfg.rank < 1)
            throw std::invalid_argument("r97: LoraDelta: rank must be at least 1, not " +
                                        std::to_string(cfg.rank));
        if (!std::isfinite(cfg.alpha))
            throw std::invalid_argument("r97: LoraDelta: alpha must be finite");
        if (meta.targetable.empty())
            throw std::invalid_argument("r97: LoraDelta: the container has no targetable linears");

        const std::vector<size_t> lines = select(meta, cfg.targets);
        alpha_ = cfg.alpha;
        // The wire record's address comes from the same meta init() saw: which declared weight each
        // module lands in, and where in it (docs/ADAPTER.md 5).
        address_.clear();
        for (size_t index : lines)
            address_[meta.targetable[index].module] = Address{meta.targetable[index].weight_name,
                                                              meta.targetable[index].slot};

        // PEFT's reset_lora_parameters: A is kaiming_uniform_ with a=sqrt(5), whose bound is
        // 1/sqrt(fan_in), and B is zero so the adapter starts as a no-op. One stream for the whole
        // ParamSet, so two calls with the same seed are bit-identical.
        std::mt19937_64 rng(seed);
        ParamSet params;
        params.reserve(2 * lines.size());
        for (size_t index : lines) {
            const ModelMeta::Linear& lin = meta.targetable[index];
            if (lin.in <= 0 || lin.out <= 0)
                throw std::invalid_argument("r97: LoraDelta: '" + lin.module + "' has a degenerate shape");

            const double bound = 1.0 / std::sqrt(double(lin.in));
            std::uniform_real_distribution<float> dist(float(-bound), float(bound));

            Param a;
            a.name = lin.module + ".lora_A";
            a.shape = {cfg.rank, lin.in};
            a.value.resize(size_t(cfg.rank * lin.in));
            for (float& v : a.value) v = dist(rng);

            Param b;
            b.name = lin.module + ".lora_B";
            b.shape = {lin.out, cfg.rank};
            b.value.assign(size_t(lin.out * cfg.rank), 0.0f);

            params.push_back(std::move(a));
            params.push_back(std::move(b));
        }
        return params;
    }

    // The parameters are already named `<module>.lora_A`, so the adjoint adds no prefix of its
    // own; the walk takes each call's prefix from the site's module (docs/adjoints/lora.md).
    const char* grad_prefix() const override { return ""; }

    // One site per targeted linear, with the factor pair the ParamSet holds for it and the scale
    // alpha/r of docs/ADAPTER.md's section 5.
    void to_wire(const ParamSet& p, WireBuffers* out) override {
        if (!out) throw std::invalid_argument("r97: LoraDelta::to_wire: null WireBuffers");
        std::map<std::string, const Param*> by_name;
        for (const Param& param : p) by_name[param.name] = &param;

        out->sites.clear();
        for (const Param& a : p) {
            const std::string suffix = ".lora_A";
            if (a.name.size() <= suffix.size() ||
                a.name.compare(a.name.size() - suffix.size(), suffix.size(), suffix) != 0)
                continue;
            const std::string module = a.name.substr(0, a.name.size() - suffix.size());
            const auto b = by_name.find(module + ".lora_B");
            if (b == by_name.end())
                throw std::invalid_argument("r97: LoraDelta::to_wire: '" + a.name +
                                            "' has no lora_B beside it");
            const Param& pa = a;
            const Param& pb = *b->second;
            if (pa.shape.size() != 2 || pb.shape.size() != 2)
                throw std::invalid_argument("r97: LoraDelta::to_wire: '" + module +
                                            "' is not a rank-2 factor pair");
            if (pa.shape[0] != pb.shape[1] || pa.shape[0] < 1)
                throw std::invalid_argument("r97: LoraDelta::to_wire: '" + module + "' has ranks " +
                                            std::to_string(pa.shape[0]) + " and " +
                                            std::to_string(pb.shape[1]));
            if (int64_t(pa.value.size()) != product(pa.shape) ||
                int64_t(pb.value.size()) != product(pb.shape))
                throw std::invalid_argument("r97: LoraDelta::to_wire: '" + module +
                                            "' has a shape and value size that disagree");

            const auto at = address_.find(module);
            if (at == address_.end())
                throw std::invalid_argument("r97: LoraDelta::to_wire: '" + module +
                                            "' is not one of this delta's initialised targets");

            WireBuffers::Site site;
            site.weight_name = at->second.weight_name;
            site.slot = at->second.slot;
            site.module = module;
            site.rank = pa.shape[0];
            site.scale = alpha_ / double(site.rank);
            site.a = pa.value;
            site.b = pb.value;
            out->sites.push_back(std::move(site));
        }
    }

private:
    // Where a module's factors land on the engine's side: the declared weight and the slot in it.
    struct Address {
        std::string weight_name;
        int64_t slot = 0;
    };

    double alpha_;  // the config's alpha; the wire site's scale is alpha/r
    std::map<std::string, Address> address_;  // module -> its address, from the last init()
};

}  // namespace

// The factory the registry holds (r97/model.h). The DeltaConfig reaches init(), not the
// constructor, so one kind serves every rank and target set; only alpha is carried, for to_wire.
std::shared_ptr<Delta> make_lora_delta(const Config& cfg) {
    return std::make_shared<LoraDelta>(cfg.delta.alpha);
}

R97_REGISTER_DELTA(lora, make_lora_delta);

}  // namespace r97
