// tests/host/step_host_int8.cpp -- the int8 trunk's host gradient, checked by a descent step.
//
// The trunk's codes are a step function of the parameters, so a finite difference of the loss is
// not a usable reference. The check is the one the quantised trunk can give: the straight-through
// gradient is finite and nonzero, and a small step along it lowers the scored loss.
#include "r97/adapter_io.h"
#include "r97/batch.h"
#include "r97/config.h"
#include "r97/forward.h"
#include "r97/model.h"
#include "r97/step_host.h"
#include "../support/fixture_w8a8.h"
#include "../support/test.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if R97_FORWARD && defined(RAD_REPLAY_BLOCK)

namespace {

r97::Batch batch_of(const std::vector<std::vector<int32_t>>& sequences) {
    r97::Batch batch;
    for (const std::vector<int32_t>& ids : sequences) {
        r97::TokenSeq s;
        s.ids = ids;
        s.score.assign(ids.size(), 1);
        s.score[0] = 0;
        batch.push_back(std::move(s));
    }
    return batch;
}

std::vector<int32_t> prompt(int32_t seed, int32_t n) {
    std::vector<int32_t> ids(n);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = (seed + 7 * int32_t(i)) % 256;
    return ids;
}

std::unique_ptr<r97::ForwardDriver> open_forward(const std::string& container) {
    r97::ForwardConfig cfg;
    cfg.model = container;
    cfg.radiance_home = r97::detail::radiance_home();
    cfg.window_rows = 1 << 20;
    std::string err;
    std::unique_ptr<r97::ForwardDriver> driver = r97::ForwardDriver::open(cfg, &err);
    if (!driver) throw std::runtime_error(err);
    return driver;
}

r97::Config lora_config(const std::string& container) {
    r97::Config cfg;
    cfg.model = container;
    cfg.objective = "ce";
    cfg.delta.kind = "lora";
    cfg.delta.rank = 4;
    cfg.delta.alpha = 8.0;
    // The int8 trunk's linear_attn.out_proj stays bf16 in the natural basis (its K is 64), and
    // the engine refuses an adapter on it, so the case targets the nine modules both containers
    // can carry.
    cfg.delta.targets = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                         "self_attn.o_proj", "linear_attn.in_proj_qkv",
                         "linear_attn.in_proj_z", "mlp.gate_proj", "mlp.up_proj",
                         "mlp.down_proj"};
    return cfg;
}

const r97::Param* find_param(const r97::ParamSet& set, const std::string& name) {
    for (const r97::Param& p : set)
        if (p.name == name) return &p;
    return nullptr;
}

double norm(const r97::ParamSet& set) {
    double sum = 0.0;
    for (const r97::Param& p : set)
        for (float v : p.value) sum += double(v) * double(v);
    return std::sqrt(sum);
}

bool ends_with(const std::string& s, const char* suffix) {
    const std::string tail(suffix);
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

}  // namespace

R97_TEST(host_step_int8) {
    const r97::Fixture& f = r97::w8a8_fixture();
    const r97::Batch batch = batch_of({prompt(21, 5), prompt(13, 7)});
    const r97::Config cfg = lora_config(f.container);
    const r97::Model model = r97::make_model(cfg);
    const r97::ModelMeta meta = r97::read_model_meta(f.container);
    const r97::ParamSet params = model.delta->init(meta, cfg.delta, 1234);

    r97::ParamSet grads;
    r97::StepResult before;
    {
        std::unique_ptr<r97::ForwardDriver> driver = open_forward(f.container);
        before = r97::host_step(*driver, model)(batch, params, &grads);
    }
    R97_CHECK(before.scored > 0);
    R97_CHECK(std::isfinite(before.loss));
    R97_CHECK(grads.size() == params.size());

    double grad_norm = 0.0;
    bool b_nonzero = false;
    for (const r97::Param& g : grads) {
        double part = 0.0;
        for (float v : g.value) {
            R97_CHECK(std::isfinite(v));
            part += double(v) * double(v);
        }
        grad_norm += part;
        if (ends_with(g.name, "lora_B") && part > 0.0) b_nonzero = true;
    }
    grad_norm = std::sqrt(grad_norm);
    R97_CHECK(std::isfinite(grad_norm) && grad_norm > 0.0);
    R97_CHECK(b_nonzero);

    // A normalised step keeps the check independent of the int8 head's missing dequantisation
    // scale: the direction is what the straight-through walk must get right.
    const double eta = 0.01;
    const double lr = eta / grad_norm;
    r97::ParamSet stepped = params;
    for (r97::Param& p : stepped) {
        const r97::Param* g = find_param(grads, p.name);
        R97_CHECK(g != nullptr);
        for (size_t j = 0; j < p.value.size(); ++j)
            p.value[j] = float(double(p.value[j]) - lr * double(g->value[j]));
    }

    r97::StepResult after;
    {
        std::unique_ptr<r97::ForwardDriver> driver = open_forward(f.container);
        after = r97::host_step(*driver, model)(batch, stepped, nullptr);
    }
    R97_CHECK(std::isfinite(after.loss));
    R97_CHECK(after.loss < before.loss);
}

#endif
