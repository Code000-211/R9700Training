// tools/ops.cpp -- the `ops` subcommand: what the op registry holds.
//
//   r97train ops --registered
//
// lists every op with a twin, a host adjoint or a device adjoint, and the rows
// it carries.
#include "commands.h"

#include "../include/r97/adjoint.h"

#include <iostream>
#include <string>
#include <vector>

namespace r97tool {

static void print_rows(const r97::OpRows& rows) {
    if (rows.twin) std::cout << "  twin\n";
    if (rows.host_adjoint) {
        std::cout << "  adjoint host";
        if (!rows.host_frozen.empty()) {
            std::cout << " (frozen:";
            for (int f : rows.host_frozen) std::cout << " " << f;
            std::cout << ")";
        }
        std::cout << "\n";
    }
    if (rows.device_adjoint) {
        std::cout << "  adjoint device";
        if (!rows.device_frozen.empty()) {
            std::cout << " (frozen:";
            for (int f : rows.device_frozen) std::cout << " " << f;
            std::cout << ")";
        }
        std::cout << "\n";
    }
}

static int ops_main(int argc, char** argv) {
    bool registered = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--registered") registered = true;
    if (!registered || argc > 2) {
        std::cerr << "usage: r97train ops --registered\n";
        return 2;
    }
    std::vector<std::string> ops = r97::ops_registered();
    std::cout << ops.size() << " op" << (ops.size() == 1 ? "" : "s") << " registered:\n";
    for (const std::string& op : ops) {
        std::cout << op << "\n";
        print_rows(r97::op_rows(op));
    }
    return 0;
}

R97_REGISTER_COMMAND("ops", "--registered", ops_main);

}  // namespace r97tool
