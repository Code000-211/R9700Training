// tests/support/command_table.h -- the tool command table as a test sees it.
//
// tools/commands() keeps its map in a function-local static. tests/host/test_registrations.cpp
// takes the copy below once all command files have registered and keeps it on the heap, so the
// table a test reads stays readable whatever order this binary's statics are destroyed in. The
// test binary alone uses it; r97train keeps reading r97tool::commands().
#pragma once

#include <map>
#include <string>

#include "../../tools/commands.h"

namespace r97test {

const std::map<std::string, r97tool::Command>& command_table();

}  // namespace r97test
