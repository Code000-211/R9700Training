// tests/gpu/main.cpp -- the entry point of the collected gpu test binary.
//
// Like the host tier, the sources under tests/gpu/ build into one executable; each brings its
// checks as R97_TEST, and this file owns main so no test has to.
//
// The tests run here, from main, rather than from the at-exit runner tests/support/test.h arms
// from the first registration. That runner is armed during static initialisation -- in a gpu
// binary, before the r97 registry another translation unit's initialisers build -- and at exit
// statics are destroyed in the reverse of the order they were built, so the registry is gone
// before an armed handler runs. A device test dlopen()s libr97.so, whose rows register into that
// registry as the plugin loads, so running the tests at exit would register into a destroyed
// table. Running them inside main keeps every static alive for the whole run; the process then
// _Exits, which skips the second, at-exit run and the teardown behind it. A failure is the exit
// status either way, which is what ctest reads.
//
// `--bench` adds a timed pass to each device case (tests/support/bench.h); the correctness
// comparison runs either way. Anything else is a usage error.
#include "support/bench.h"
#include "support/test.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) {
            r97test::bench_flag() = true;
            continue;
        }
        std::fprintf(stderr, "usage: %s [--bench]\n", argv[0]);
        std::_Exit(2);  // a usage error runs no tests
    }

    const int failed = r97test::run_all();
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(failed ? 1 : 0);
}
