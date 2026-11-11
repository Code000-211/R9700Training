// tests/gpu/main.cpp -- the entry point of the collected gpu test binary.
//
// Like the host tier, the sources under tests/gpu/ build into one executable; each brings its
// checks as R97_TEST, and this file owns main so no test has to. The runner is test.h's, armed by
// the first registration and run at exit, so a failure is the process's exit status.
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
    return 0;
}
