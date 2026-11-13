// tests/host/info_cli.cpp -- `r97train info`, the runbook's first preflight read of a container.
//
// The command prints arch, quant, layers and hidden for a container, and exits 3 for one that does
// not open, which is the exit `docs/RUNBOOK.md` tells an operator to read. Reading the metadata
// needs no card, so the check runs on the tiny fixture's container.
// Label: host.
#include "../support/fixture.h"
#include "../support/test.h"

#if R97_FIXTURE_PREFILL

#include <cstdio>
#include <stdexcept>
#include <string>
#include <sys/wait.h>

namespace {

struct Run {
    int status = -1;
    std::string output;
};

Run run(const std::string& args) {
    const std::string build = r97::detail::build_dir();
    const std::string tool = build.empty() ? "r97train" : build + "/r97train";
    Run r;
    const std::string command = r97::detail::shell_quote(tool) + " " + args + " 2>&1";
    FILE* p = popen(command.c_str(), "r");
    if (!p) throw std::runtime_error("info_cli: cannot run " + command);
    char buf[4096];
    while (size_t n = fread(buf, 1, sizeof buf, p)) r.output.append(buf, n);
    const int status = pclose(p);
    if (status >= 0 && WIFEXITED(status)) r.status = WEXITSTATUS(status);
    return r;
}

std::string quote(const std::string& text) { return r97::detail::shell_quote(text); }

// The integer after `name:` in the command's output.
long field(const std::string& output, const std::string& name) {
    const std::string prefix = name + ":";
    const size_t at = output.find(prefix);
    if (at == std::string::npos)
        throw std::runtime_error("info_cli: no " + prefix + " in " + output);
    const size_t start = output.find_first_of("-0123456789", at + prefix.size());
    if (start == std::string::npos) throw std::runtime_error("info_cli: no number after " + prefix);
    const size_t end = output.find_first_not_of("0123456789", start + 1);
    return std::stol(output.substr(start, end == std::string::npos ? end : end - start));
}

}  // namespace

R97_TEST(info_cli) {
    const r97::Fixture& fixture = r97::fixture();

    const Run ok = run("info " + quote(fixture.container));
    R97_CHECK(ok.status == 0);
    R97_CHECK(ok.output.find("arch:") != std::string::npos);
    R97_CHECK(ok.output.find("quant:") != std::string::npos);
    // The fixture's shape, from its own record: the four numbers are not prose.
    R97_CHECK(field(ok.output, "layers") == fixture.n_layers);
    R97_CHECK(field(ok.output, "hidden") == fixture.hidden);

    // A container that does not open: exit 3, and nothing printed as if it had.
    const Run missing = run("info /nonexistent/r97-info-cli.rad");
    R97_CHECK(missing.status == 3);
    R97_CHECK(missing.output.find("arch:") == std::string::npos);

    // A usage error: the command takes exactly one file.
    const Run bare = run("info");
    R97_CHECK(bare.status == 2);
}

#endif  // R97_FIXTURE_PREFILL
