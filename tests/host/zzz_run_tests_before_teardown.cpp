// The host suite runs its registered tests from an at-exit handler (support/test.h), but the
// binary's own main is only one of those exit handlers. This translation unit sorts last in the
// host glob, so its constructor registers after every other handler and every static object: at
// exit it runs first, while the fixtures and radiance's statics are still alive, and _Exit skips
// the teardown that would otherwise run between tests.
#include <cstdio>
#include <cstdlib>

#include "support/test.h"

namespace {

void run_tests() {
    const int failed = r97test::run_all();
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(failed ? 1 : 0);
}

struct RunBeforeTeardown {
    RunBeforeTeardown() {
        if (std::atexit(&run_tests) != 0)
            std::fprintf(stderr, "r97test: could not register the runner\n");
    }
} run_before_teardown;

}  // namespace
