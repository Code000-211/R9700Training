// tools/objectives.cpp -- `r97train objectives`: the registered objectives, one per line.
#include "commands.h"

#include "r97/model.h"

#include <iostream>

namespace {

int run_objectives(int /*argc*/, char** /*argv*/) {
    std::cout << "objectives registered:\n";
    for (const std::string& name : r97::objective_names()) std::cout << "  " << name << "\n";
    return 0;
}

}  // namespace

R97_REGISTER_COMMAND("objectives", "", run_objectives);
