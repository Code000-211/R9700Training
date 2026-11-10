// tests/host/model_registry.cpp -- objectives and deltas resolve by name, and an unknown name is
// refused with the list of the registered ones (label: host).
#include "r97/model.h"

#include "r97/config.h"

#include "../support/bias_delta.h"
#include "../support/sum_objective.h"
#include "../support/recorded_vector_tape.h"

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

R97_REGISTER_OBJECTIVE(test_sum, r97test::make_sum_objective)
R97_REGISTER_DELTA(test_bias, r97test::make_bias_delta)

static bool contains(const std::vector<std::string>& names, const std::string& name) {
    for (const std::string& n : names)
        if (n == name) return true;
    return false;
}

int main() {
    // The registrations are in the list.
    assert(contains(r97::objective_names(), "test_sum"));
    assert(contains(r97::delta_names(), "test_bias"));

    // A Model resolves by name and both halves work.
    r97::Config cfg;
    cfg.objective = "test_sum";
    cfg.delta.kind = "test_bias";
    cfg.delta.rank = 3;
    r97::Model model = r97::make_model(cfg);
    r97test::RecordedVectorTape tape;
    tape.set_buffer("logits", {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    r97::ObjectiveResult result = model.objective(tape, {});
    assert(result.loss == 10.0);   // the sum of rows [0, 1): {1, 2, 3, 4}
    assert(result.scored == 1);
    assert(result.seeds.size() == 1);
    r97::ParamSet p = model.delta->init({}, cfg.delta, 0);
    assert(p.size() == 1 && p[0].name == "test.bias.b" && p[0].value.size() == 3);

    // An unknown objective is refused with the list of the registered ones.
    r97::Config bad = cfg;
    bad.objective = "nonesuch";
    try {
        r97::make_model(bad);
        assert(false && "unknown objective must be refused");
    } catch (const std::exception& e) {
        const std::string what = e.what();
        assert(what.find("nonesuch") != std::string::npos);
        assert(what.find("test_sum") != std::string::npos);
    }

    // So is an unknown delta kind, with the list of the registered kinds.
    r97::Config bad_kind = cfg;
    bad_kind.delta.kind = "nonesuch";
    try {
        r97::make_model(bad_kind);
        assert(false && "unknown delta kind must be refused");
    } catch (const std::exception& e) {
        const std::string what = e.what();
        assert(what.find("nonesuch") != std::string::npos);
        assert(what.find("test_bias") != std::string::npos);
    }

    return 0;
}
