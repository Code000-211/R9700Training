// tests/support/test_main.h -- one `main` for every host test file.
//
// The host tests build as one binary (CMake globs tests/host/*.cpp), so test files must not
// define `main` themselves. Include this header and wrap the body in `R97_TEST(<name>) { ... }`;
// the registry below runs every registered test in link order and prints "<name> ok".
#pragma once

#include <cstdio>
#include <vector>

namespace r97test {

struct TestCase {
    const char* name;
    void (*fn)();
};

inline std::vector<TestCase>& tests() {
    static std::vector<TestCase> t;
    return t;
}

struct Registrar {
    Registrar(const char* name, void (*fn)()) { tests().push_back({name, fn}); }
};

}  // namespace r97test

#define R97_TEST_CONCAT2(a, b) a##b
#define R97_TEST_CONCAT(a, b) R97_TEST_CONCAT2(a, b)
#define R97_TEST(name)                                                            \
    static void R97_TEST_CONCAT(r97_test_fn_, name)();                            \
    static r97test::Registrar R97_TEST_CONCAT(r97_test_reg_, name)(               \
        #name, R97_TEST_CONCAT(r97_test_fn_, name));                              \
    static void R97_TEST_CONCAT(r97_test_fn_, name)()

// Weak so one strong `main` elsewhere (a test that wants its own driver) still wins; when all
// test files rely on the registry, this one runs everything. One per binary is enough.
__attribute__((weak)) int main() {
    for (const r97test::TestCase& t : r97test::tests()) {
        t.fn();
        std::printf("%s ok\n", t.name);
        std::fflush(stdout);
    }
    return 0;
}
