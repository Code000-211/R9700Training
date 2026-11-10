// tools/info.cpp -- the `info` subcommand: what r97train prints about a .rad container.
#include "commands.h"
#ifndef R97_TOOLS_NO_INFO_H
#include "../src/info.h"
#endif

#include <iostream>
#include <string>

namespace r97tool {

static int info_main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: r97train info <file.rad>\n";
        return 2;
    }
    return r97::cmd_info(argv[1], std::cout);
}

R97_REGISTER_COMMAND("info", "<file.rad>", info_main);

}  // namespace r97tool
