// src/info.h -- internal: the `info` subcommand's implementation, shared by the CLI and the
// model test. Not a contract header; include/r97/ holds the frozen ones.
#pragma once

#include <iosfwd>
#include <string>

namespace r97 {

// Print a container's architecture, quantisation, layer count and hidden size.
// Returns 0, or 3 when the container cannot be opened (the model test's skip signal).
int cmd_info(const std::string& path, std::ostream& out);

}  // namespace r97
