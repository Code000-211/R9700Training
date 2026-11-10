// tools/objectives.cpp -- `r97train objectives`: the registered objectives, one per line.
#include "commands.h"

#include "r97/model.h"

#include <cstdio>
#include <iostream>

namespace {

int run_objectives(int /*argc*/, char** /*argv*/) {
    std::puts("objectives registered:");
    for (const std::string& name : r97::objective_names()) std::printf("  %s\n", name.c_str());
    return 0;
}

}  // namespace

R97_REGISTER_COMMAND("objectives", "", run_objectives);
