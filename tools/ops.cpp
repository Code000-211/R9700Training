// tools/ops.cpp -- the `ops` subcommand: what one prefill step issues, and what the op registry
// already holds for it.
//
//   r97train ops <file.rad> [--home DIR] [--kernels libavx,libref]
//   r97train ops [file.rad] --device N
//   r97train ops --registered
//
// The first form runs one text-only prefill step of the container with speculation off (the
// training forward), records the tape and prints the inventory: every distinct op form, the block
// it runs in, how often, the adjoint rows registered for it, the seed boundary its write exposes
// and whether it lies on the walk from the loss back to a trunk linear. docs/OPS-qwen35-host.md
// holds the two tiny fixtures' inventories and says how the columns are derived.
//
// --device N is that same step on card N: libr4d is the kernel tier, the card is the only one HIP
// is shown, and the container defaults to $R97_MODEL_RAD / $R97_MODEL_DIR. The inventory then
// names the fused device rows the runtime selected in place of host chains;
// docs/OPS-qwen35-device.md holds the real container's and the chains they replace.
//
// The last form lists every op with a twin, a host adjoint or a device adjoint.
#include "commands.h"

#include "../include/r97/adjoint.h"
#include "../include/r97/device_tier.h"

#include <iostream>
#include <string>
#include <vector>

#include "../src/op_inventory.h"

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

static int registered_main() {
    std::vector<std::string> ops = r97::ops_registered();
    std::cout << ops.size() << " op" << (ops.size() == 1 ? "" : "s") << " registered:\n";
    for (const std::string& op : ops) {
        std::cout << op << "\n";
        print_rows(r97::op_rows(op));
    }
    return 0;
}

static std::vector<std::string> split_list(const std::string& list) {
    std::vector<std::string> out;
    size_t at = 0;
    while (at <= list.size()) {
        const size_t comma = list.find(',', at);
        const std::string one = list.substr(at, comma == std::string::npos ? std::string::npos
                                                                          : comma - at);
        if (!one.empty()) out.push_back(one);
        if (comma == std::string::npos) break;
        at = comma + 1;
    }
    return out;
}

static int inventory_main(const std::string& container, const std::string& home,
                          const std::vector<std::string>& kernels) {
#if defined(RAD_ISSUE_OBSERVER)
    // Byte ids: every container's vocabulary holds 1..8 (the tiny one's is 320).
    const std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    r97::Tape tape;
    std::string error;
    if (!r97::record_prefill_step(container, home, kernels, tokens, tape, &error)) {
        std::cerr << "r97train ops: " << error << "\n";
        return 1;
    }
    const r97::OpInventory inventory =
        r97::build_op_inventory(tape, container, {"logits", "ce", "gate_logits"});
    std::cout << "# " << container << ": one prefill step, " << inventory.records
              << " issues, " << inventory.rows.size() << " op forms\n\n";
    r97::print_op_inventory(inventory, std::cout);
    return 0;
#else
    (void)container;
    (void)home;
    (void)kernels;
    std::cerr << "r97train ops: the container inventory needs the training build (the tape's "
                 "issue observer is patches/training/0101)\n";
    return 2;
#endif
}

static int ops_main(int argc, char** argv) {
    bool registered = false;
    int device = -1;
    std::string container, home;
    std::vector<std::string> kernels;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--registered") {
            registered = true;
        } else if (arg == "--home" && i + 1 < argc) {
            home = argv[++i];
        } else if (arg == "--kernels" && i + 1 < argc) {
            kernels = split_list(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            try {
                device = std::stoi(argv[++i]);
            } catch (...) {
                device = -1;
            }
            if (device < 0) {
                std::cerr << "r97train ops: --device takes a card ordinal\n";
                return 2;
            }
        } else if (arg.rfind("--", 0) != 0 && container.empty()) {
            container = arg;
        } else {
            std::cerr << "usage: r97train ops <file.rad> [--home DIR] [--kernels LIST]\n"
                         "       r97train ops [file.rad] --device N\n"
                         "       r97train ops --registered\n";
            return 2;
        }
    }
    if (registered) {
        if (!container.empty() || !home.empty() || !kernels.empty() || device >= 0) {
            std::cerr << "r97train ops: --registered takes no container\n";
            return 2;
        }
        return registered_main();
    }
    if (device >= 0) {
        if (!kernels.empty()) {
            std::cerr << "r97train ops: --device and --kernels name the same choice; give one\n";
            return 2;
        }
        kernels = {"libr4d"};
        if (container.empty()) container = r97::default_model_rad();
        if (container.empty()) {
            std::cerr << "r97train ops: --device needs a container, and $R97_MODEL_RAD / "
                         "$R97_MODEL_DIR names none\n";
            return 2;
        }
        std::string err;
        if (!r97::select_device(device, &err)) {
            std::cerr << "r97train ops: " << err << "\n";
            return 1;
        }
    }
    if (container.empty()) {
        std::cerr << "usage: r97train ops <file.rad> [--home DIR] [--kernels LIST]\n"
                     "       r97train ops [file.rad] --device N\n"
                     "       r97train ops --registered\n";
        return 2;
    }
    if (home.empty()) home = r97::default_radiance_home();
    return inventory_main(container, home, kernels);
}

R97_REGISTER_COMMAND("ops", "<file.rad>|--registered", ops_main);

}  // namespace r97tool
