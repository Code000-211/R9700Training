// r97/delta_registry.h -- the trainable kinds by name.
//
// The model registry (issue #59) owns the whole registry: objectives and deltas by name, and the
// seed check. This is the delta half of it, added here because the adapter kind has to be built by
// name before that issue lands: a new trainable kind is one new file that registers itself, never
// a branch in the forward driver or the backward core (CONTRIBUTING.md).
//
// Registration is a static object in the kind's own file:
//
//     #include "r97/delta_registry.h"
//
//     R97_REGISTER_DELTA("lora", my_delta_factory)
//
// The factory takes no argument; the DeltaConfig reaches init(), not the constructor, so one
// registered kind serves every rank and target set.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "r97/delta.h"

namespace r97 {

// One kind's constructor: an empty Delta that init() fills from the container and the config.
using DeltaFactory = std::shared_ptr<Delta> (*)();

// Register `kind`. Registering the same kind twice is a programming error and aborts, as a
// duplicate op row does in the adjoint registry.
void delta_register(const char* kind, DeltaFactory factory);

// The kinds registered so far, sorted, so a listing is stable across translation units.
std::vector<std::string> delta_kinds();

// The delta `kind` names. An unknown kind throws std::invalid_argument listing the registered
// ones.
std::shared_ptr<Delta> make_delta(const std::string& kind);

struct DeltaRegistrar {
    DeltaRegistrar(const char* kind, DeltaFactory factory) { delta_register(kind, factory); }
};

#define R97_DELTA_REG_NAME2(a, b) a##b
#define R97_DELTA_REG_NAME(a, b) R97_DELTA_REG_NAME2(a, b)

// The registrar object's name comes from the line, so one file may register more than one kind.
#define R97_REGISTER_DELTA(kind, factory) \
    static ::r97::DeltaRegistrar R97_DELTA_REG_NAME(r97_reg_delta_, __LINE__) { kind, factory };

}  // namespace r97
