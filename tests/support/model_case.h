// tests/support/model_case.h -- registering a test that needs the real container as well as
// whatever its own tier needs.
//
// The three labels make different promises (docs/TESTING.md): a `gpu` test never skips, and a
// `model` test resolves the real container first and skips -- exit 77, ctest records it as skipped
// -- when the container is absent. A test that needs both cannot be one of either only: started as
// the `gpu` label's own tests it would have to skip on a machine without the container, and
// started as a `model` run it would have to take the card a `model` run does not promise.
//
// So the gpu binary is started twice, once per label, and the difference between the two runs is
// one environment variable: ctest sets R97_MODEL_CASES=1 for the model run (CMakeLists.txt, test
// `gpu_model_tests`), and a case registered with R97_MODEL_TEST below runs only in that run. It
// registers through the same registry as R97_TEST (tests/support/test.h), so it keeps its place in
// the name order and needs no flag in tests/gpu/main.cpp.
//
//   R97_MODEL_TEST(name) { ... }   // a gpu test that also reads the container
//
// The case itself must resolve the container first thing, with r97test::container_or_skip
// (tests/support/model_dir.h), so a machine without it skips before the card guard takes a card.
// tests/gpu/iqr_decode.cpp is the example.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "test.h"

namespace r97test {

// Whether this is the run that also starts the model cases: set by the build, not by a person.
inline bool model_cases() {
    const char* v = std::getenv("R97_MODEL_CASES");
    return v && *v && std::strcmp(v, "0") != 0;
}

}  // namespace r97test

// R97_TEST for a case that also needs the container. In a run without R97_MODEL_CASES it says so
// and returns, rather than passing quietly: the plain `gpu` run reports a case it did not start.
#define R97_MODEL_TEST(name_)                                                  \
    static void test_##name_##_body();                                         \
    static void test_##name_() {                                               \
        if (!r97test::model_cases()) {                                         \
            std::printf("not run: %s is a model case; it needs the real "      \
                        "container\n",                                         \
                        #name_);                                               \
            return;                                                            \
        }                                                                      \
        test_##name_##_body();                                                 \
    }                                                                          \
    static r97test::Reg reg_##name_(#name_, test_##name_);                     \
    static void test_##name_##_body()
