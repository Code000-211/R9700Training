// tests/support/test.h -- the self-registration every test file in a collected test
// binary uses. A test file defines its checks with R97_TEST(name) { ... } instead of
// main; the binary's single main (tests/host/test_main.cpp and its siblings) runs every
// registered test in name order through run_all().
#pragma once

#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <vector>

namespace r97test {

struct Test {
    const char* name;
    void (*fn)();
};

inline std::vector<Test>& tests() {
    static std::vector<Test> t;
    return t;
}

struct Reg {
    Reg(const char* name, void (*fn)()) { tests().push_back({name, fn}); }
};

inline int run_all() {
    int failed = 0;
    for (const Test& t : tests()) {
        try {
            t.fn();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s: threw: %s\n", t.name, e.what());
            ++failed;
        } catch (...) {
            std::fprintf(stderr, "%s: threw\n", t.name);
            ++failed;
        }
    }
    std::printf("%zu tests, %d failed\n", tests().size(), failed);
    return failed ? 1 : 0;
}

}  // namespace r97test

// The default build is RelWithDebInfo, where <cassert> is compiled out; use R97_CHECK,
// which throws on failure and is caught by run_all(), for every real check.
#define R97_CHECK(cond)                                     \
    do {                                                    \
        if (!(cond))                                        \
            throw std::runtime_error("check failed: " #cond); \
    } while (0)

// Defines static void test_##name_() from the brace body that follows.
#define R97_TEST(name_)                                  \
    static void test_##name_();                          \
    static r97test::Reg reg_##name_(#name_, test_##name_); \
    static void test_##name_()
