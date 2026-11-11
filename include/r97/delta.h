// r97/delta.h -- the trainable kind.
//
// What is trained: LoRA today, the config's DeltaConfig names it. A Delta builds the adapter's
// ParamSet for a container, names the gradient prefix the adjoint writes under, and packs the
// weights into the slot radiance's live adapter reads. docs/ADAPTER.md owns the meaning of a
// WireBuffers site.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "r97/params.h"

namespace r97 {

struct DeltaConfig {
    std::string kind;                 // "lora" (config.h sets the default)
    int64_t rank = 16;
    double alpha = 32.0;
    std::vector<std::string> targets; // module names; empty means the default targeting
    bool gate = false;                // train a gate beside the sites
};

// What a delta may know about a container; the adapter-files issue fills it from rad_core, a
// test builds it by hand.
struct ModelMeta {
    std::string arch;
    std::string input_basis;
    int64_t hidden = 0, layers = 0;
    struct Linear {
        std::string module;       // the PEFT module path, "model.layers.5.mlp.gate_proj"
        std::string weight_name;  // the declared weight it lands in, "blk.5.ffn_gate_up"
        int64_t slot = 0;         // its index among that weight's targeted sources
        int64_t in = 0, out = 0;
    };
    std::vector<Linear> targetable;
};

// What the live adapter puts into radiance's slot: one Site per targeted PEFT module, addressed
// by weight_name and slot, so two modules of one fused weight are told apart (docs/ADAPTER.md 5).
// `extra` holds what is not a LoRA site (a gate).
struct WireBuffers {
    struct Site {
        std::string weight_name;  // "blk.5.ffn_gate_up": the declared weight it lands in
        int64_t slot = 0;         // its index among that weight's targeted sources
        std::string module;
        int64_t rank = 0;
        double scale = 0.0;
        std::vector<float> a, b;
    };
    std::vector<Site> sites;
    ParamSet extra;
};

struct Delta {
    virtual ~Delta() = default;

    // Build the ParamSet for a container (replaces init_adapter's LoRA-only signature).
    virtual ParamSet init(const ModelMeta& meta, const DeltaConfig& cfg, uint64_t seed) = 0;

    // The name prefix an adjoint adds to the gradient ParamSet (e.g. "<module>.lora_A").
    virtual const char* grad_prefix() const = 0;

    // What crosses to the engine: how the weights are packed into the LoRA slot (A/B/gate).
    virtual void to_wire(const ParamSet& p, WireBuffers* out) = 0;
};

}  // namespace r97
