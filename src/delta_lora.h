// src/delta_lora.h -- the `lora` kind's factory.
//
// The kind registers itself from a static object in its own file (R97_REGISTER_DELTA, r97/model.h),
// which a link that keeps that file sees: r97train and the host and gpu test tiers link r97 whole.
// The model test tier links it as it comes, so it names this factory to pull the file in. init() is
// still the only thing that makes an adapter.
#pragma once

#include <memory>

#include "r97/config.h"
#include "r97/delta.h"

namespace r97 {

std::shared_ptr<Delta> make_lora_delta(const Config& cfg);

}  // namespace r97
