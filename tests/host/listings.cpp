// tests/host/listings.cpp -- the `r97train objectives` and `r97train deltas` subcommands run and
// print the test registrations (label: host). The registrations live in this binary, so the
// subcommand implementations are linked in and called here; the same command table is what the
// r97train executable dispatches.
#undef NDEBUG   // the checks must survive the default RelWithDebInfo build's -DNDEBUG
#include "../../tools/commands.h"

#include "r97/model.h"

#include "../support/bias_delta.h"
#include "../support/sum_objective.h"

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

R97_REGISTER_OBJECTIVE(test_sum, r97test::make_sum_objective);
R97_REGISTER_DELTA(test_bias, r97test::make_bias_delta);

static std::string run(const char* name) {
    auto it = r97tool::commands().find(name);
    assert(it != r97tool::commands().end() && "the subcommand is registered");

    std::ostringstream captured;
    std::streambuf* old = std::cout.rdbuf(captured.rdbuf());
    char arg0[] = "r97train";
    char arg1[] = "subcommand";
    char* argv[] = {arg0, arg1, nullptr};
    const int rc = it->second.run(1, argv);
    std::cout.rdbuf(old);

    assert(rc == 0 && "the subcommand succeeds");
    return captured.str();
}

int main() {
    const std::string objectives = run("objectives");
    const std::string deltas = run("deltas");

    // The test registrations print from the subcommands, not from a copy of their loop.
    assert(objectives.find("test_sum") != std::string::npos);
    assert(deltas.find("test_bias") != std::string::npos);

    std::cout << objectives << deltas;
    return 0;
}
