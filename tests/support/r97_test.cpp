// tests/support/r97_test.cpp -- the main the tier test binaries share; a test that throws fails.
#include "support/r97_test.h"

#include <cstdio>
#include <exception>

int main() {
    int failed = 0;
    const auto& all = r97test::Registry::tests();
    for (const auto& [name, fn] : all) {
        std::printf("%-20s ", name.c_str());
        std::fflush(stdout);
        try {
            fn();
            std::puts("ok");
        } catch (const std::exception& e) {
            std::printf("FAIL: %s\n", e.what());
            ++failed;
        }
    }
    if (failed)
        std::printf("%d of %zu failed\n", failed, all.size());
    return failed ? 1 : 0;
}
