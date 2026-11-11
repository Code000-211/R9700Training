// tests/support/self_check.h -- a host test file that brings no main() of its own.
//
// The host tests are one binary, collected from tests/host/*.cpp, and its entry point belongs to
// tests/host/vector_tape.cpp (issue #1); a second main() would not link, and that file is not this
// issue's to change. So a test file runs itself from a static initializer instead, which the binary
// reaches before that main: R97_SELF_CHECK names the function, it is run once with its name
// printed, and a check that fails prints its expression and location and aborts, which is what
// makes the binary exit non-zero and ctest report it. When the tier's own runner takes the entry
// point over, an R97_SELF_CHECK line becomes one registration and nothing else changes.
//
// CHECK is a real check, not assert(): the project builds RelWithDebInfo, where -DNDEBUG makes
// assert() compile out and a check that never runs passes.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <exception>

namespace r97test {

[[noreturn]] inline void check_failed(const char* expr, const char* file, int line) {
    std::fprintf(stderr, "%s:%d: check failed: %s\n", file, line, expr);
    std::fflush(stderr);
    std::abort();
}

// Announce a test this build leaves out, by name, the way run() announces one it runs. A tier
// built without a patch then says which tests it skipped instead of passing silently or failing.
inline void skip(const char* name, const char* why) {
    std::printf("[host] %s ... skipped (%s)\n", name, why);
    std::fflush(stdout);
}

// Run one test. A test that throws is a failure too, reported by name, not a terminate() in
// another file's main.
inline void run(const char* name, void (*fn)()) {
    std::printf("[host] %s\n", name);
    std::fflush(stdout);
    try {
        fn();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[host] %s: threw: %s\n", name, e.what());
        std::fflush(stderr);
        std::abort();
    }
}

}  // namespace r97test

#define R97_CHECK(expr) ((expr) ? (void)0 : ::r97test::check_failed(#expr, __FILE__, __LINE__))
#define CHECK(expr) R97_CHECK(expr)

#define R97_SELF_CHECK(fn)                     \
    namespace {                                \
    struct fn##_self_check {                   \
        fn##_self_check() { ::r97test::run(#fn, fn); } \
    } fn##_self_check_instance;                \
    }
