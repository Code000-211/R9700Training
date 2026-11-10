// tests/support/test_reg.h -- the self-registering host-test table.
//
// The host tests are one binary collected from tests/host/*.cpp, so a test is a registered
// function, not a main: one file defines run_all()'s main (test_runner.cpp), every other file
// registers itself with R97_REGISTER_TEST and owns its checks. Order follows the name; every
// test announces itself with [name] so a crash says which one.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

namespace r97test {

using TestFn = void (*)();

inline std::map<std::string, TestFn>& tests() {
    static std::map<std::string, TestFn> t;
    return t;
}

struct TestReg {
    TestReg(const char* name, TestFn fn) { tests()[name] = fn; }
};

// CHECK is a real check, not assert(): the project builds RelWithDebInfo (-DNDEBUG), where
// assert() compiles out. A failed check prints its expression and location and aborts, so the
// host-tests binary exits non-zero and ctest reports the failure.
[[noreturn]] inline void check_failed(const char* expr, const char* file, int line) {
    std::fprintf(stderr, "%s:%d: check failed: %s\n", file, line, expr);
    std::fflush(stderr);
    std::abort();
}

inline int run_all() {
    for (const auto& [name, fn] : tests()) {
        std::printf("[host] %s\n", name.c_str());
        std::fflush(stdout);
        fn();
    }
    std::printf("host tests: %zu ok\n", tests().size());
    return 0;
}

}  // namespace r97test

#define R97_CHECK(expr) \
    ((expr) ? (void)0 : ::r97test::check_failed(#expr, __FILE__, __LINE__))
#define CHECK(expr) R97_CHECK(expr)
#define R97_REGISTER_TEST(name, fn) static r97test::TestReg r97_testreg_##fn(name, fn);
