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
// comparison runs either way. `--model` runs the model cases -- the ones that also need the real
// container, registered with R97_MODEL_TEST -- and, like every model test, skips (exit 77) when
// the container is absent; without it the run is the card's own tests, which never skip. ctest
// starts this binary once per kind (CMakeLists.txt), because one process cannot be both a `gpu`
// run that never skips and a `model` run that skips. Anything else is a usage error.
#include "support/bench.h"
#include "support/test.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    r97test::Select sel = r97test::Select::card;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) {
            r97test::bench_flag() = true;
            continue;
        }
        if (std::strcmp(argv[i], "--model") == 0) {
            sel = r97test::Select::model;
            continue;
        }
        std::fprintf(stderr, "usage: %s [--bench] [--model]\n", argv[0]);
        std::_Exit(2);  // a usage error runs no tests
    }

    const int failed = r97test::run_all(sel);
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(failed ? 1 : 0);
}
