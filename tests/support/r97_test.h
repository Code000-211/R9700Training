// tests/support/r97_test.h -- the tier test binaries' registry.
//
// The build globs every source under tests/<tier>/ into one binary per tier (CMakeLists.txt), so
// a tier holds one main. Tests register themselves with R97_TEST(name) and
// tests/support/r97_test.cpp's main runs them in registration order, printing one line each.
//
// Note on asserts: the default build type is RelWithDebInfo, which defines NDEBUG and compiles
// assert() out. Test files that check with assert() must `#undef NDEBUG` before including
// <cassert>; see tests/host/vector_tape.cpp.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace r97test {

class Registry {
public:
    using Test = std::pair<std::string, void (*)()>;

    static std::vector<Test>& tests() {
        static std::vector<Test> t;
        return t;
    }

    static bool add(const char* name, void (*fn)()) {
        tests().emplace_back(name, fn);
        return true;
    }
};

// Defines a registration: the test body follows the macro, as in
//     R97_TEST(vram_guard) { ... }
#define R97_TEST(name)                                                              \
    static void r97test_##name##_body();                                            \
    static const bool r97test_##name##_reg = ::r97test::Registry::add(#name,        \
        r97test_##name##_body);                                                     \
    static void r97test_##name##_body()

}  // namespace r97test
