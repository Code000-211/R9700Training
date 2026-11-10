// tests/host/model_registry.cpp -- objectives and deltas resolve by name, and an unknown name is
// refused with the list of the registered ones (label: host). The names used here are registered
// once for the whole binary, by tests/host/test_registrations.cpp.
#include "r97/config.h"
#include "r97/model.h"

#include "../support/recorded_vector_tape.h"
#include "../support/test.h"

#include <exception>
#include <string>
#include <vector>

namespace {

bool contains(const std::vector<std::string>& names, const std::string& name) {
    for (const std::string& n : names)
        if (n == name) return true;
    return false;
}

// The refusal message of a Config whose name is unknown. A make_model that does not refuse is a
// failed check; it is raised outside the catch so it cannot be mistaken for the refusal.
std::string refusal_of(const r97::Config& cfg) {
    try {
        (void)r97::make_model(cfg);
    } catch (const std::exception& e) {
        return e.what();
    }
    R97_CHECK(false && "make_model must refuse an unknown name");
    return "";
}

}  // namespace

R97_TEST(model_registry) {
    // The test registrations are in the lists.
    R97_CHECK(contains(r97::objective_names(), "test_sum"));
    R97_CHECK(contains(r97::delta_names(), "test_bias"));

    // A Model resolves by name and both halves work.
    r97::Config cfg;
    cfg.objective = "test_sum";
    cfg.delta.kind = "test_bias";
    cfg.delta.rank = 3;
    r97::Model model = r97::make_model(cfg);
    r97test::RecordedVectorTape tape;
    tape.set_buffer("logits", {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    r97::ObjectiveResult result = model.objective(tape, {});
    R97_CHECK(result.loss == 10.0);   // the sum of rows [0, 1): {1, 2, 3, 4}
    R97_CHECK(result.scored == 1);
    R97_CHECK(result.seeds.size() == 1);
    r97::ParamSet p = model.delta->init({}, cfg.delta, 0);
    R97_CHECK(p.size() == 1 && p[0].name == "test.bias.b" && p[0].value.size() == 3);

    // An unknown objective is refused with the list of the registered ones.
    r97::Config bad = cfg;
    bad.objective = "nonesuch";
    const std::string what = refusal_of(bad);
    R97_CHECK(what.find("nonesuch") != std::string::npos);
    R97_CHECK(what.find("test_sum") != std::string::npos);

    // So is an unknown delta kind, with the list of the registered kinds.
    r97::Config bad_kind = cfg;
    bad_kind.delta.kind = "nonesuch";
    const std::string what_kind = refusal_of(bad_kind);
    R97_CHECK(what_kind.find("nonesuch") != std::string::npos);
    R97_CHECK(what_kind.find("test_bias") != std::string::npos);
}
