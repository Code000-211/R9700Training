// tools/deltas.cpp -- `r97train deltas`: the registered trainable kinds, one per line.
#include "commands.h"

#include "r97/model.h"

#include <cstdio>
#include <iostream>

namespace {

int run_deltas(int /*argc*/, char** /*argv*/) {
    std::puts("deltas registered:");
    for (const std::string& kind : r97::delta_names()) std::printf("  %s\n", kind.c_str());
    return 0;
}

}  // namespace

R97_REGISTER_COMMAND("deltas", "", run_deltas);
