// tools/ops.cpp -- the `ops` subcommand: what one prefill step issues, and what the op registry
// already holds for it.
//
//   r97train ops <file.rad> [--home DIR] [--kernels libavx,libref] [--record FILE] [--batch FILE]
//   r97train ops --registered
//
// The first form runs one text-only prefill step of the container with speculation off (the
// training forward), records the tape and prints the inventory: every distinct op form, the block
// it runs in, how often, the adjoint rows registered for it, the seed boundary its write exposes
// and whether it lies on the walk from the loss back to a trunk linear. docs/OPS-qwen35-host.md
// holds the two tiny fixtures' inventories and says how the columns are derived.
//
// The second form lists every op with a twin, a host adjoint or a device adjoint.
#include "commands.h"

#include "../include/r97/adjoint.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

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

static nlohmann::json program_json(const r97::Tape& tape,
                                   const std::vector<std::vector<int32_t>>& prompts) {
    nlohmann::json doc;
    doc["format"] = "r97-program";
    doc["version"] = 1;
    doc["batch"] = nlohmann::json::array();
    for (const std::vector<int32_t>& p : prompts) doc["batch"].push_back(p);
    doc["ops"] = nlohmann::json::array();
    for (const r97::OpRecord& rec : tape.ops()) {
        nlohmann::json op;
        op["op"] = rec.op;
        op["block"] = rec.block;
        op["position"] = rec.position;
        op["n"] = rec.n;
        nlohmann::json params = nlohmann::json::object();
        for (const auto& [name, value] : rec.params) {
            if (const auto* i = std::get_if<int64_t>(&value)) params[name] = *i;
            else params[name] = std::get<std::string>(value);
        }
        op["params"] = params;
        op["operands"] = nlohmann::json::array();
        for (const r97::Operand& o : rec.operands) {
            op["operands"].push_back({{"buffer", o.buffer}, {"dtype", o.dtype},
                                      {"shape", o.shape}, {"row_lo", o.row_lo},
                                      {"rows", o.rows}, {"written", o.written}});
        }
        doc["ops"].push_back(op);
    }
    return doc;
}

static bool read_batch(const std::string& path, std::vector<std::vector<int32_t>>* prompts,
                       std::string* error) {
    try {
        std::ifstream in(path);
        if (!in) {
            *error = "cannot read " + path;
            return false;
        }
        nlohmann::json doc;
        in >> doc;
        if (doc.is_object() && doc.contains("batch")) doc = doc["batch"];
        if (!doc.is_array()) {
            *error = path + ": expected a JSON array of token-id arrays";
            return false;
        }
        for (const nlohmann::json& seq : doc) {
            if (!seq.is_array()) {
                *error = path + ": every sequence must be an array of token ids";
                return false;
            }
            std::vector<int32_t> ids;
            for (const nlohmann::json& id : seq) {
                if (!id.is_number_integer()) {
                    *error = path + ": every token id must be an integer";
                    return false;
                }
                ids.push_back(id.get<int32_t>());
            }
            prompts->push_back(std::move(ids));
        }
        if (prompts->empty()) {
            *error = path + ": the batch is empty";
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        *error = path + ": " + e.what();
        return false;
    }
}

static int inventory_main(const std::string& container, const std::string& home,
                          const std::vector<std::string>& kernels, const std::string& record,
                          const std::string& batch_path) {
#if defined(RAD_ISSUE_OBSERVER)
    // Byte ids: every container's vocabulary holds 1..8 (the tiny one's is 320).
    std::vector<std::vector<int32_t>> prompts = {{1, 2, 3, 4, 5, 6, 7, 8}};
    std::string error;
    if (!batch_path.empty()) {
        prompts.clear();
    }
    if (!batch_path.empty() && !read_batch(batch_path, &prompts, &error)) {
        std::cerr << "r97train ops: " << error << "\n";
        return 1;
    }
    r97::Tape tape;
    if (!r97::record_prefill_batch(container, home, kernels, prompts, tape, &error)) {
        std::cerr << "r97train ops: " << error << "\n";
        return 1;
    }
    if (!record.empty()) {
        std::ofstream out(record);
        if (!out) {
            std::cerr << "r97train ops: cannot write " << record << "\n";
            return 1;
        }
        out << program_json(tape, prompts).dump(2) << "\n";
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
    std::string container, home, record, batch_path;
    std::vector<std::string> kernels;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--registered") {
            registered = true;
        } else if (arg == "--home" && i + 1 < argc) {
            home = argv[++i];
        } else if (arg == "--kernels" && i + 1 < argc) {
            kernels = split_list(argv[++i]);
        } else if (arg == "--record" && i + 1 < argc) {
            record = argv[++i];
        } else if (arg == "--batch" && i + 1 < argc) {
            batch_path = argv[++i];
        } else if (arg.rfind("--", 0) != 0 && container.empty()) {
            container = arg;
        } else {
            std::cerr << "usage: r97train ops <file.rad> [--home DIR] [--kernels LIST]\n"
                         "       [--record FILE] [--batch FILE]\n"
                         "       r97train ops --registered\n";
            return 2;
        }
    }
    if (registered) {
        if (!container.empty() || !home.empty() || !kernels.empty()) {
            std::cerr << "r97train ops: --registered takes no container\n";
            return 2;
        }
        return registered_main();
    }
    if (container.empty()) {
        std::cerr << "usage: r97train ops <file.rad> [--home DIR] [--kernels LIST]\n"
                     "       r97train ops --registered\n";
        return 2;
    }
    return inventory_main(container, home, kernels, record, batch_path);
}

R97_REGISTER_COMMAND("ops", "<file.rad>|--registered", ops_main);

}  // namespace r97tool
