// tests/model/info_meta.cpp -- `info` prints the container's metadata through rad_core.
// Needs the real container (a model test); it skips, exit 77, when the container is absent.
#include "../src/info.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#ifndef R97_MODEL_RAD
#define R97_MODEL_RAD "data/qwen3.8-27b-iq4r-s-v4.rad"
#endif

static constexpr int kSkip = 77;

static bool exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

int main() {
    if (!exists(R97_MODEL_RAD)) {
        std::cout << "skip: no container at " << R97_MODEL_RAD << "\n";
        return kSkip;
    }

    std::ostringstream out;
    int rc = r97::cmd_info(R97_MODEL_RAD, out);
    if (rc != 0) {
        std::cerr << "info failed on " << R97_MODEL_RAD << "\n";
        return 1;
    }
    const std::string text = out.str();
    std::cout << text;
    // What the acceptance names for this container.
    for (const char* want : {"arch:     qwen35\n", "quant:    w8a8_fwht128\n",
                             "layers:   64\n", "hidden:   5120\n"}) {
        if (text.find(want) == std::string::npos) {
            std::cerr << "info output lacks \"" << want << "\"\n";
            return 1;
        }
    }
    return 0;
}
