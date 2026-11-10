// tests/support/r97_test.cpp -- the main the tier test binaries share; a test that throws fails,
// and one that throws Skip is skipped. See tests/support/r97_test.h.
#include "r97_test.h"

#include "model_dir.h"

#include <cstdio>
#include <exception>

int main() {
    int failed = 0;
    int skipped = 0;
    const auto& all = r97test::Registry::tests();
    for (const auto& [name, fn] : all) {
        std::printf("%-20s ", name.c_str());
        std::fflush(stdout);
        try {
            fn();
            std::puts("ok");
        } catch (const r97test::Skip& s) {
            std::printf("skip: %s\n", s.why.c_str());
            ++skipped;
        } catch (const std::exception& e) {
            std::printf("FAIL: %s\n", e.what());
            ++failed;
        }
    }
    if (failed) {
        std::printf("%d of %zu failed\n", failed, all.size());
        return 1;
    }
    if (skipped) {
        std::printf("%d of %zu skipped\n", skipped, all.size());
        return r97test::kSkip;
    }
    return 0;
}
