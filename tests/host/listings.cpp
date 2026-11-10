// tests/host/listings.cpp -- the listing the `r97train objectives` and `r97train deltas`
// subcommands print, with the test registrations (label: host). The registrations live in the
// test binary, so the same library listing is run here; the subcommands print it unchanged.
#include "r97/model.h"

#include "../support/bias_delta.h"
#include "../support/sum_objective.h"

#include <cassert>
#include <iostream>
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
    std::putc('\n', stdout);

    std::puts("objectives registered:");
    for (const std::string& name : r97::objective_names()) std::printf("  %s\n", name.c_str());
    std::puts("deltas registered:");
    for (const std::string& kind : r97::delta_names()) std::printf("  %s\n", kind.c_str());

    // The test registrations print; the subcommands run exactly this listing.
    assert(contains(r97::objective_names(), "test_sum"));
    assert(contains(r97::delta_names(), "test_bias"));
    return 0;
}
