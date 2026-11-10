// tools/commands.cpp -- the table's storage. Registration happens at static-init time, in any
// order, from each subcommand's own file.
#include "commands.h"

namespace r97tool {

std::map<std::string, Command>& commands() {
    static std::map<std::string, Command> table;
    return table;
}

}  // namespace r97tool
