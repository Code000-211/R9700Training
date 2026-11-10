// tests/host/listings.cpp -- the `r97train objectives` and `r97train deltas` subcommands run and
// print the test registrations (label: host).
//
// The command translation units are compiled into tests/host/test_registrations.cpp, because the
// host tier links r97 but not tools/ while the test registrations live in this binary. The same
// command table is what the r97train executable dispatches.
#include "../support/test.h"

#include "../../tools/commands.h"

#include <iostream>
#include <sstream>
#include <string>

namespace {

std::string run(const char* name) {
    auto it = r97tool::commands().find(name);
    R97_CHECK(it != r97tool::commands().end());

    std::ostringstream captured;
    std::streambuf* old = std::cout.rdbuf(captured.rdbuf());
    char arg0[] = "r97train";
    char arg1[] = "subcommand";
    char* argv[] = {arg0, arg1, nullptr};
    const int rc = it->second.run(1, argv);
    std::cout.rdbuf(old);

    R97_CHECK(rc == 0);
    return captured.str();
}

}  // namespace

R97_TEST(listings) {
    const std::string objectives = run("objectives");
    const std::string deltas = run("deltas");

    // The test registrations print from the subcommands, not from a copy of their loop.
    R97_CHECK(objectives.find("test_sum") != std::string::npos);
    R97_CHECK(deltas.find("test_bias") != std::string::npos);

    std::cout << objectives << deltas;
}
