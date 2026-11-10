// r97train -- the trainer's command line.
// Subcommands register themselves in tools/commands.h (one file each); this file only dispatches.
#include "commands.h"

#include <algorithm>
#include <cstdio>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: r97train <command> [...]\n\ncommands:\n";
        for (const auto& [name, cmd] : r97tool::commands())
            std::cerr << "  " << name << " " << cmd.usage << "\n";
        return 2;
    }

    auto& table = r97tool::commands();
    auto it = table.find(argv[1]);
    if (it == table.end()) {
        std::cerr << "r97train: unknown command '" << argv[1] << "'\n";
        return 2;
    }
    return it->second.run(argc - 1, argv + 1);
}
