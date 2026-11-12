// tools/config.cpp -- `r97train config`: the configuration schema, or a configuration resolved.
//
//   r97train config                                   the schema, one line per key
//   r97train config --config run.conf --steps 500     that file and those flags, resolved
//
// With no argument nothing is loaded and nothing is validated: the schema is what a config file
// may say, and printing it must work from a bare build. With a file or a flag the command reads
// the file, applies the flags over it and validates the result exactly as `train` does — except
// that it supplies no fallbacks, so a required key missing from the file is reported rather than
// filled in. That is the preflight `docs/RUNBOOK.md` runs before a long run.
#include "commands.h"

#include "r97/config_schema.h"
#include "r97/trainer.h"

#include <iostream>
#include <string>
#include <vector>

namespace r97tool {
namespace {

int config_main(int argc, char** argv) {
    std::string config_path;
    std::vector<std::string> flags;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" || arg == "--config-file") {
            if (++i >= argc) {
                std::cerr << "usage: r97train config [--config <file>] [flags]\n";
                return 2;
            }
            config_path = argv[i];
        } else {
            flags.push_back(arg);
        }
    }

    if (config_path.empty() && flags.empty()) {
        std::cout << r97::config_schema_text();
        return 0;
    }

    try {
        const r97::Config config = r97::load_config(config_path, flags, r97::Config{});
        std::cout << r97::config_values_text(config);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "r97train config: " << e.what() << "\n";
        return 1;
    }
}

R97_REGISTER_COMMAND("config", "[--config <file>] [flags]", config_main);

}  // namespace
}  // namespace r97tool
