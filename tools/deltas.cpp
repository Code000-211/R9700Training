// tools/deltas.cpp -- `r97train deltas`: the registered trainable kinds, one per line.
#include "commands.h"

#include "r97/model.h"

#include <iostream>

namespace {

int run_deltas(int /*argc*/, char** /*argv*/) {
    std::cout << "deltas registered:\n";
    for (const std::string& kind : r97::delta_names()) std::cout << "  " << kind << "\n";
    return 0;
}

}  // namespace

R97_REGISTER_COMMAND("deltas", "", run_deltas);
