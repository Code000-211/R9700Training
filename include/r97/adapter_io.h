// r97/adapter_io.h -- the adapter directory on disk and the container's targeted linears.
//
// An adapter is written where radiance serves it from (docs/ADAPTER.md, section 4): one directory
// holding PEFT's two files, `adapter_config.json` and `adapter_model.safetensors`. The config
// carries the rank, the alpha, the target modules, the base model's name and -- on a trunk whose
// weights are stored under a transform -- `radiance_input_basis`. The safetensors file carries the
// factors under PEFT's tensor names, `base_model.model.<module>.lora_A.weight` and its `lora_B`
// twin, in f32 or on request bf16, so the trainer's fp32 master and the file it hands radiance are
// one layout.
//
// `read_model_meta` is the other half: a container's targetable linears, named as PEFT spells them
// and shaped as the document's table says, read from the container's own metadata through
// `rad_core` -- no card, no engine, no plugin.
//
// The ParamSet name is the PEFT module path plus the factor, `<module>.lora_A`; the file's name is
// that with PEFT's `base_model.model.` prefix and a `.weight` suffix. `peft_tensor_name` and
// `param_name_from_peft` are the two directions, so a name never has two spellings in the trainer.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "r97/delta.h"
#include "r97/params.h"

namespace r97 {

// PEFT stores the factors as floats. f32 is the trainer's master copy; bf16 is what a radiance
// slot holds, so it is what a serving-only copy is written as.
enum class AdapterDtype { F32, BF16 };

// What adapter_config.json says. `alpha` is the config's `lora_alpha`; the rank is one number for
// the whole adapter, as PEFT writes it when no rank_pattern is involved.
struct AdapterConfig {
    std::string base_model;            // base_model_name_or_path
    int64_t rank = 16;                 // r
    double alpha = 32.0;               // lora_alpha
    std::vector<std::string> targets;  // target_modules, in the caller's spelling
    std::string input_basis;           // "fwht128" on a rotated trunk, empty on a natural one
    AdapterDtype dtype = AdapterDtype::F32;
};

// One adapter directory read back: its config and its factors.
struct AdapterFiles {
    AdapterConfig config;
    ParamSet params;
};

inline constexpr const char* kAdapterConfigFile = "adapter_config.json";
inline constexpr const char* kAdapterModelFile = "adapter_model.safetensors";
// PEFT saves the model under this prefix; radiance's reader strips it.
inline constexpr const char* kPeftTensorPrefix = "base_model.model.";

// The name the file carries for a ParamSet entry, and back. The suffix is `lora_A`/`lora_B` in
// both forms; a name that lacks PEFT's prefix or its `.weight` suffix is an error, not a guess.
std::string peft_tensor_name(const std::string& param_name);
std::string param_name_from_peft(const std::string& tensor_name);

// Write `params` into `dir` (created if missing), replacing the two files. The config's dtype
// says how the factors are stored; reading a bf16 file widens each value exactly.
void write_adapter(const std::string& dir, const AdapterConfig& config, const ParamSet& params);

// Read `dir` back. The entries come in the safetensors header's order (the file's own order on
// disk is the sorted tensor name, so a write-then-read round trip is bit-identical in f32).
AdapterFiles read_adapter(const std::string& dir);

// The container's targetable linears: every layer's blocks, each fused weight split into the
// modules the document's table names, with in/out as the adapter sees them. The basis is the
// transform the container's encodings carry ("fwht128" there), empty when the trunk is natural.
ModelMeta read_model_meta(const std::string& container);

}  // namespace r97
