// tests/host/config_defaults.cpp -- a command's fallbacks sit under --config, and flags over both.
//
// `r97train train` has its own fallback lr/steps/token_budget/accumulation, but a key the config
// file sets must reach the loop unchanged. Pass the fallbacks to load_config() instead of writing
// them over the parsed file; this is the order documented for that entry point.
// Label: host.
#include "support/test.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "r97/trainer.h"

using namespace r97;

R97_TEST(config_file_overrides_command_fallbacks) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "r97_config_defaults_test.conf";
    {
        std::ofstream out(path);
        out << "lr = 0.5\nsteps = 7\naccumulation = 2\n";
    }
    Config fallback;
    fallback.lr = 1e-3;
    fallback.steps = 1;
    fallback.token_budget = 2048;
    fallback.accumulation = 1;
    fallback.out = "out";
    const Config file = load_config(path.string(), {}, fallback);
    R97_CHECK(file.lr == 0.5);
    R97_CHECK(file.steps == 7);
    R97_CHECK(file.accumulation == 2);
    R97_CHECK(file.token_budget == 2048);
    R97_CHECK(file.out == "out");

    /* A flag is still last. */
    const Config flagged = load_config(path.string(), {"--steps", "9"}, fallback);
    R97_CHECK(flagged.steps == 9);
    R97_CHECK(flagged.lr == 0.5);

    /* A bare command gets the fallbacks. */
    const Config bare = load_config("", {}, fallback);
    R97_CHECK(bare.lr == 1e-3);
    R97_CHECK(bare.steps == 1);
    std::remove(path.string().c_str());
}
