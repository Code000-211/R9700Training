// tools/commands.h -- the self-registering subcommand table.
//
// One file per subcommand; it registers itself with a file-static R97_REGISTER_COMMAND, so a new
// subcommand is a new .cpp and no edit to any existing file. The order of the registration is
// arbitrary; the usage output sorts by name.
#pragma once

#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace r97tool {

using CommandFn = int (*)(int argc, char** argv);   // argv[0] is the subcommand's name

struct Command {
    const char* name;
    const char* usage;    // the argument hint, e.g. "<file.rad>"
    CommandFn run;
};

// The table lives behind a function so registration order never matters.
std::map<std::string, Command>& commands();

struct Registration {
    Registration(const char* name, const char* usage, CommandFn run) {
        commands()[name] = Command{name, usage, run};
    }
};

}  // namespace r97tool

#define R97_REGISTER_COMMAND(name, usage, fn) \
    static r97tool::Registration r97_register_##fn(name, usage, fn)
