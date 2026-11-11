// tests/support/command_table.h -- the tool command table as the exit-time tests see it.
//
// tools/commands() keeps its map in a function-local static. The host tests run from an atexit
// handler, after this binary's static destructors have run, so reading the live map there would
// touch freed memory. tests/host/test_registrations.cpp takes the copy below once all command
// files have registered and keeps it on the heap, so it outlives the command table's own
// destructor. The test binary alone uses it; r97train keeps reading r97tool::commands().
#pragma once

#include <map>
#include <string>

#include "../../tools/commands.h"

namespace r97test {

const std::map<std::string, r97tool::Command>& command_table();

}  // namespace r97test
