// r97train -- the trainer's command line.
// Subcommands register themselves in tools/commands.h (one file each); this file only dispatches.
//
// A bare `r97train` prints the command table and is the error it has always been; `r97train
// --help` prints the same table as the answer it is, and `r97train <command> --help` prints the
// command's argument line -- every flag it takes -- and points at docs/USAGE.md, whose flag table
// says what each one does. tests/cmake/config_schema_test.sh holds that table and this output
// equal, so a flag added here without the document, or the other way round, fails a host test.
#include "commands.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <ostream>

namespace {

// The table: one line per command, the name and its argument hint.
void print_commands(std::ostream& out) {
    out << "usage: r97train <command> [...]\n\ncommands:\n";
    for (const auto& [name, cmd] : r97tool::commands())
        out << "  " << name << " " << cmd.usage << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_commands(std::cerr);
        return 2;
    }

    if (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
        print_commands(std::cout);
        return 0;
    }

    auto& table = r97tool::commands();
    auto it = table.find(argv[1]);
    if (it == table.end()) {
        std::cerr << "r97train: unknown command '" << argv[1] << "'\n";
        return 2;
    }

    // The command's help is asked for before its own parser sees the arguments: every subcommand
    // takes `--help` the same way, whether or not it has a flag by that name.
    if (argc > 2 && (std::strcmp(argv[2], "--help") == 0 || std::strcmp(argv[2], "-h") == 0)) {
        std::cout << "usage: r97train " << it->first;
        if (it->second.usage[0] != '\0') std::cout << " " << it->second.usage;
        std::cout << "\ndocs/USAGE.md lists every flag and what it does.\n";
        return 0;
    }

    return it->second.run(argc - 1, argv + 1);
}
