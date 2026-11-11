// tests/support/model_case.h -- registering a test that needs the real container as well as
// whatever its own tier needs.
//
// The labels make different promises (docs/TESTING.md): a `gpu` test never skips, and a `model`
// test resolves the real container first and skips -- exit 77, ctest records it as skipped -- when
// the container is absent. A test that needs both cannot be one of either only: started as the
// `gpu` label's own tests it would have to skip on a machine without the container, and started as
// a `model` run it would have to take the card a `model` run does not promise.
//
// So the gpu binary is started twice, and the difference between the two runs is one environment
// variable: ctest sets R97_MODEL_CASES=1 for the second start (CMakeLists.txt, test
// `gpu_model_tests`, which carries the `model` label and no `gpu` one), and a case registered with
// R97_MODEL_TEST below is registered only in that run. The plain `gpu` run does not hold it at
// all: that run promises never to skip, so a case that returned without doing anything would be a
// pass it did not earn. Registration goes through the same registry as R97_TEST
// (tests/support/test.h), so a model case keeps its place in the name order and needs no flag in
// tests/gpu/main.cpp.
//
//   R97_MODEL_TEST(name) { ... }   // a gpu test that also reads the container
//
// The case itself must resolve the container first thing, with r97test::container_or_skip
// (tests/support/model_dir.h), so a machine without it skips before the card guard takes a card.
// tests/gpu/iqr_decode.cpp is the example.
#pragma once

#include <cstdlib>
#include <cstring>

#include "test.h"

namespace r97test {

// Whether this is the run that also starts the model cases: set by the build, not by a person.
inline bool model_cases() {
    const char* v = std::getenv("R97_MODEL_CASES");
    return v && *v && std::strcmp(v, "0") != 0;
}

// Registers a model case, in the model run only: the case is not in the plain `gpu` run's list, so
// nothing there can report it as passed. The Reg lives for the rest of the process -- the registry
// keeps its name and its function, so the object itself has nothing to release.
inline bool register_model_case(const char* name, void (*fn)()) {
    if (!model_cases()) return false;
    new Reg(name, fn);
    return true;
}

}  // namespace r97test

// R97_TEST for a case that also needs the container: registered in the run that sets
// R97_MODEL_CASES, and in no other.
#define R97_MODEL_TEST(name_)                                                       \
    static void test_##name_();                                                     \
    [[maybe_unused]] static const bool reg_##name_ =                                \
        r97test::register_model_case(#name_, test_##name_);                         \
    static void test_##name_()
