// tests/host/data_cli.cpp -- `r97train data`, the runbook's read of the training sequences.
//
// The command prints the counts the batcher will see and, with --show, the first conversations with
// their scored runs marked. docs/RUNBOOK.md reads those counts before a run: skipped equal to the
// conversations is a template or a format problem, not something to train through, and `[[…]]` is
// what the loss is taken over. The tiny fixture's container and tests/fixtures/chat_small.jsonl are
// enough for both, and no card is available to the child.
// Label: host.
#include "../support/fixture.h"
#include "../support/test.h"

#if R97_FIXTURE_PREFILL

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <sys/wait.h>

namespace {

namespace fs = std::filesystem;

struct Run {
    int status = -1;
    std::string output;
};

std::string chat_small() {
    const fs::path here = fs::absolute(fs::path(__FILE__));
    return (here.parent_path().parent_path() / "fixtures" / "chat_small.jsonl").string();
}

Run run(const std::string& args) {
    const std::string build = r97::detail::build_dir();
    const std::string tool = build.empty() ? "r97train" : build + "/r97train";
    // The child must see no card: the fixture's shapes fit no device kernel, and a run that saw a
    // card would refuse the model instead of tokenising it.
    const std::string command =
        "HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=99 " + r97::detail::shell_quote(tool) + " " +
        args + " 2>/dev/null";
    Run r;
    FILE* p = popen(command.c_str(), "r");
    if (!p) throw std::runtime_error("data_cli: cannot run " + command);
    char buf[4096];
    while (size_t n = fread(buf, 1, sizeof buf, p)) r.output.append(buf, n);
    const int status = pclose(p);
    if (status >= 0 && WIFEXITED(status)) r.status = WEXITSTATUS(status);
    return r;
}

std::string quote(const std::string& text) { return r97::detail::shell_quote(text); }

long field(const std::string& output, const std::string& name) {
    const std::string prefix = name + ":";
    const size_t at = output.find(prefix);
    if (at == std::string::npos)
        throw std::runtime_error("data_cli: no " + prefix + " in " + output);
    const size_t start = output.find_first_of("0123456789", at + prefix.size());
    if (start == std::string::npos) throw std::runtime_error("data_cli: no number after " + prefix);
    const size_t end = output.find_first_not_of("0123456789", start);
    return std::stol(output.substr(start, end == std::string::npos ? end : end - start));
}

}  // namespace

R97_TEST(data_cli) {
    const std::string model = r97::fixture().container;

    const Run plain = run("data --model " + quote(model) + " --data " + quote(chat_small()));
    R97_CHECK(plain.status == 0);

    // Every conversation is accounted for by exactly one of the three outcomes, and the scored
    // tokens are a part of what is read.
    const long conversations = field(plain.output, "conversations");
    const long kept = field(plain.output, "kept");
    const long shortened = field(plain.output, "shortened");
    const long skipped = field(plain.output, "skipped");
    const long tokens = field(plain.output, "tokens");
    const long scored = field(plain.output, "scored");
    R97_CHECK(conversations > 0);
    R97_CHECK(kept > 0);
    R97_CHECK(kept + shortened + skipped == conversations);
    R97_CHECK(tokens > 0);
    R97_CHECK(scored > 0);
    R97_CHECK(scored <= tokens);

    // --show prints the scored runs, and only those, in [[…]].
    const Run shown = run("data --model " + quote(model) + " --data " + quote(chat_small()) +
                          " --show 1");
    R97_CHECK(shown.status == 0);
    R97_CHECK(shown.output.find("conversation 1:") != std::string::npos);
    R97_CHECK(shown.output.find("[[") != std::string::npos);
    R97_CHECK(shown.output.find("]]") != std::string::npos);
    R97_CHECK(shown.output.find("conversation 2:") == std::string::npos);

    // A container that does not open: exit 3, the same reading as `info`.
    const Run missing =
        run("data --model /nonexistent/r97-data-cli.rad --data " + quote(chat_small()));
    R97_CHECK(missing.status == 3);
}

#endif  // R97_FIXTURE_PREFILL
