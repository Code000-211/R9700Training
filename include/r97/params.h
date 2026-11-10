// r97/params.h -- the named tensors that flow between the trainer's parts.
//
// A ParamSet describes the trained parameters (and, with the same names and shapes, a
// step's gradients). Names are the container's weight names; the delta's names carry the
// prefix Delta::grad_prefix() gives.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace r97 {

struct Param {
    std::string name;              // e.g. "model.layers.13.mlp.down_proj.lora_A"
    std::vector<int64_t> shape;    // row-major, shape[0] slowest
    std::vector<float> value;      // row-major, shape.size() elements product long
};

using ParamSet = std::vector<Param>;

}  // namespace r97
