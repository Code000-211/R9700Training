// objective_ce_register.cpp -- `ce` by name: the reference loss in the objective registry.
//
// The registry (include/r97/model.h) is filled by the translation units that own an objective, and
// the reference loss is CeObjective's, declared in include/r97/loss.h and defined in src/loss.cpp.
// This file joins the two: one name, one factory, no knowledge of either side's internals.
// Docs: docs/losses/ce-objective.md.
#include "r97/config.h"
#include "r97/loss.h"
#include "r97/model.h"
#include "r97/objective.h"

namespace r97 {
namespace {

// The registry's factory shape. Cross-entropy is configured by nothing but the batch it is handed:
// the score mask says which tokens count, and the forward's rows say what predicts them.
Objective ce_factory(const Config&) { return CeObjective; }

}  // namespace
}  // namespace r97

R97_REGISTER_OBJECTIVE(ce, r97::ce_factory);
