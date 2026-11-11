// tests/host/objective_ce_register.cpp -- `ce` is a name in the objective registry, and it gives
// the reference loss (include/r97/loss.h). Label: host.
//
// The registry is filled by the translation units that own an objective, so what this checks is
// that src/objective_ce_register.cpp's line reaches the table a trainer looks a config's
// `objective = "ce"` up in -- and that the name resolves to a callable objective.
// docs/OBJECTIVES.md says where a registration runs.
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "r97/config.h"
#include "r97/model.h"
#include "r97/objective.h"
#include "support/test.h"

R97_TEST(objective_ce_is_registered_by_name) {
    auto& registry = r97::objective_registry();
    R97_CHECK(registry.count("ce") == 1);

    const r97::Config cfg;
    const r97::Objective objective = registry.at("ce")(cfg);
    R97_CHECK(bool(objective));

    // Listed with the rest, in name order, so `r97train objectives` shows it.
    const std::vector<std::string> names = r97::objective_names();
    R97_CHECK(std::find(names.begin(), names.end(), "ce") != names.end());
    std::printf("  registry: 'ce' resolves to an objective ok\n");
}
