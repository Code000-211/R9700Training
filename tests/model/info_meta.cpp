// tests/model/info_meta.cpp -- `info` prints the container's metadata through rad_core.
// Needs the real container (a model test); it skips, exit 77, with a clear message when the
// container is absent (support/model_dir.h resolves R97_MODEL_DIR and the build's macro).
#include "../src/info.h"

#include "support/model_dir.h"
#include "support/r97_test.h"

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <stdexcept>
#include <string>

#ifndef R97_MODEL_RAD
#define R97_MODEL_RAD "data/qwen3.8-27b-iq4r-s-v4.rad"
#endif

// Registered as a tier test (tests/support/r97_test.cpp runs R97_TEST bodies in turn):
R97_TEST(info_meta) {
    std::string why;
    if (auto rad = r97test::resolve_container(R97_MODEL_RAD, &why)) {
        std::ostringstream out;
        int rc = r97::cmd_info(*rad, out);
        if (rc != 0)
            throw std::runtime_error("info failed on " + *rad);
        std::string text = out.str();
        std::cout << text;
        // What the acceptance names for this container.
        for (const char* want : {"arch:     qwen35\n", "quant:    w8a8_fwht128\n",
                                 "layers:   64\n", "hidden:   5120\n"}) {
            if (text.find(want) == std::string::npos)
                throw std::runtime_error(std::string("info output lacks \"") + want + "\"");
        }
        exit(r97test::kSkip);  // a skip leaves the whole model binary with the skip status
    }
    std::cout << why << "\n";
    exit(r97test::kSkip);
}
