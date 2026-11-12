// tests/host/usage_cli.cpp -- `r97train` with no command: the usage table it prints.
//
// docs/RUNBOOK.md opens on `build/r97train` and its command table, and its last section promises
// that command is a host test. This runs the binary the build produced with no argument and reads
// stderr, the stream it prints to: the exit status is 2, the first line is the usage line, and
// under `commands:` one `  <name> <usage>` line per registered subcommand, sorted by name. The
// subcommands the host binary itself registers (commands, objectives, deltas) must appear with the
// very usage string they registered, so a table that drifts from the binary is caught here.
// Label: host.
#include "../support/test.h"

#if R97_HAS_R97TRAIN

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <vector>

#include "../support/command_table.h"
#include "../support/fixture.h"

namespace {

struct Run {
    int status = -1;
    std::string output;
};

Run run(const std::string& args) {
    const std::string build = r97::detail::build_dir();
    const std::string tool = build.empty() ? "r97train" : build + "/r97train";
    // The usage table goes to stderr, so both streams are kept; no card is needed to print it.
    const std::string command = "HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=99 " +
                                r97::detail::shell_quote(tool) + " " + args + " 2>&1";
    Run r;
    FILE* p = popen(command.c_str(), "r");
    if (!p) throw std::runtime_error("usage_cli: cannot run " + command);
    char buf[4096];
    while (size_t n = fread(buf, 1, sizeof buf, p)) r.output.append(buf, n);
    const int status = pclose(p);
    if (status >= 0 && WIFEXITED(status)) r.status = WEXITSTATUS(status);
    return r;
}

std::vector<std::string> lines(const std::string& text) {
    std::vector<std::string> out;
    for (size_t at = 0; at < text.size();) {
        size_t end = text.find('\n', at);
        if (end == std::string::npos) end = text.size();
        out.push_back(text.substr(at, end - at));
        at = end + 1;
    }
    return out;
}

}  // namespace

R97_TEST(usage_cli) {
    const Run plain = run("");
    R97_CHECK(plain.status == 2);

    const std::vector<std::string> out = lines(plain.output);
    // The header: the usage line, a blank line, then `commands:`.
    R97_CHECK(out.size() >= 3);
    R97_CHECK(out[0] == "usage: r97train <command> [...]");
    R97_CHECK(out[1].empty());
    R97_CHECK(out[2] == "commands:");

    // One line per command, `  <name> <usage>`, sorted by name and unique: the table the runner
    // dispatches. A command may register an empty hint (`deltas` does), which leaves the trailing
    // space and nothing after it.
    std::vector<std::string> names;
    for (size_t i = 3; i < out.size(); ++i) {
        const std::string& line = out[i];
        if (line.empty()) continue;   // the child's last newline leaves no trailing line
        R97_CHECK(line.rfind("  ", 0) == 0);
        const size_t name_end = line.find(' ', 2);
        R97_CHECK(name_end != std::string::npos);
        R97_CHECK(name_end > 2);
        names.push_back(line.substr(2, name_end - 2));
    }
    R97_CHECK(!names.empty());
    R97_CHECK(std::is_sorted(names.begin(), names.end()));
    R97_CHECK(std::adjacent_find(names.begin(), names.end()) == names.end());

    // What the host binary's own table registers is what the binary prints, with the same hint.
    for (const auto& [name, command] : r97test::command_table()) {
        const std::string expected = "  " + name + " " + command.usage;
        R97_CHECK(std::find(out.begin(), out.end(), expected) != out.end());
    }

    // A word that is not a command is refused by name, and is not a usage error.
    const Run unknown = run("no-such-command");
    R97_CHECK(unknown.status == 2);
    R97_CHECK(unknown.output.find("r97train: unknown command 'no-such-command'") !=
              std::string::npos);
}

#endif  // R97_HAS_R97TRAIN
