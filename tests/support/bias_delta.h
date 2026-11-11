// tests/support/bias_delta.h -- a test delta: a single bias vector. Registered by name from the
// test binaries only. `rank` is the bias's length; it starts at zero and names "<prefix>b", so a
// test that trains it sees one vector of gradients under "test.bias.".
#pragma once

#include "r97/delta.h"

#include <memory>
#include <vector>

namespace r97test {

class BiasDelta : public r97::Delta {
public:
    explicit BiasDelta(int64_t rank) : rank_(rank > 0 ? rank : 1) {}

    r97::ParamSet init(const r97::ModelMeta&, const r97::DeltaConfig&, uint64_t) override {
        r97::Param bias;
        bias.name = "test.bias.b";
        bias.shape = {rank_};
        bias.value.assign(static_cast<size_t>(rank_), 0.0f);
        return {bias};
    }

    const char* grad_prefix() const override { return "test.bias."; }

    void to_wire(const r97::ParamSet& p, r97::WireBuffers* out) override {
        out->extra = p;   // a bias is not a LoRA site; it crosses as `extra`
    }

private:
    int64_t rank_;
};

inline std::shared_ptr<r97::Delta> make_bias_delta(const r97::Config& cfg) {
    return std::make_shared<BiasDelta>(cfg.delta.rank);
}

}  // namespace r97test
