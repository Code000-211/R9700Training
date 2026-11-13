// tests/host/runbook.cpp -- `docs/RUNBOOK.md`'s commands, extracted and run.
//
// The runbook says that every fenced block in it is one fragment of a single script, and that this
// test is what keeps the document honest. It is: every fenced block is taken in the order the file
// has them, its fence must be ```sh (a `text` block of commands would be a part of the procedure
// the test never runs), the blocks are written out as one script, and the script is run from the
// tree's top against the tiny int8 fixture, with the binary of this build. A flag that changes
// without the document, a check the document states that the tree no longer holds to, or a block
// that stops working fails here rather than in front of an operator.
//
// What is checked beyond "the script exits 0":
//   - the extracted script really is the procedure: it has the preflight, the dry run, the run, the
//     resume, the trainer's served-loss comparison, the serve from the serving-only build and that
//     build's own served-loss comparison, and the log check in it, so a fenced block that loses
//     its `sh` tag does not turn the script into a shorter one that still passes;
//   - the one-time inputs the script deliberately does not run are still named by the document: the
//     data fetch (`tools/fetch_ultrachat.py`, `docs/DATA.md`), the serving-only build
//     (`-DR97_PATCH_SET=serving`) whose tree section 7 serves from, and the rule that one run has
//     the card to itself;
//   - the card's command is in the script beside the tested one: section 2's inventory carries the
//     `--device` line next to the host line, and a run with no card takes the branch that says it
//     was skipped, so deleting the line fails even though that branch is what executed.
#include "../support/fixture.h"
#include "../support/fixture_w8a8.h"
#include "../support/test.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <vector>

#if R97_FIXTURE_PREFILL && defined(R97_HAS_R97TRAIN)

namespace {

namespace fs = std::filesystem;

// The tree's top, the directory holding docs/RUNBOOK.md. `__FILE__` is relative to the build
// directory, so the search starts from the paths the build wrote into this binary and from the
// binary itself, and takes the first ancestor that holds the document.
fs::path source_root() {
    std::vector<fs::path> starts;
    starts.push_back(fs::path(R97_FIXTURE_W8A8_RECIPE));
    starts.push_back(fs::absolute(fs::path(__FILE__)));
    starts.push_back(fs::path(r97::detail::build_dir()));
    for (fs::path at : starts) {
        for (int up = 0; up < 6 && !at.empty(); ++up) {
            if (fs::exists(at / "docs" / "RUNBOOK.md")) return at;
            at = at.parent_path();
        }
    }
    throw std::runtime_error("runbook: cannot find docs/RUNBOOK.md from " + std::string(__FILE__));
}

std::string read_file(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("runbook: cannot read " + path.string());
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// The body of every fenced block, in file order, as one script. Every fence must be ```sh: a
// `text` block of commands is a part of the procedure the test would never run. A fence that is
// never closed, a stray close, or a non-`sh` opening is an error too.
std::string extract_script(const std::string& document) {
    std::string script;
    bool open = false;
    std::size_t blocks = 0;
    std::size_t line = 0;
    for (std::size_t at = 0; at <= document.size();) {
        const std::size_t end = document.find('\n', at);
        const std::string text = document.substr(at, end == std::string::npos ? end : end - at);
        ++line;
        // A block may sit inside a list item, so the fence is the line without its indentation.
        const std::size_t first = text.find_first_not_of(" \t");
        const std::string fence = first == std::string::npos ? std::string() : text.substr(first);
        if (open) {
            if (fence == "```") {
                open = false;
            } else if (fence.rfind("```", 0) == 0) {
                throw std::runtime_error("runbook: a fenced block at line " +
                                         std::to_string(line) + " opens inside a ```sh block");
            } else {
                script += text;
                script += '\n';
            }
        } else if (fence == "```sh") {
            open = true;
            ++blocks;
        } else if (fence.rfind("```", 0) == 0) {
            throw std::runtime_error("runbook: line " + std::to_string(line) +
                                     " opens a fenced block that is not ```sh: " + fence);
        }
        if (end == std::string::npos) break;
        at = end + 1;
    }
    if (open) throw std::runtime_error("runbook: the last ```sh block is never closed");
    if (blocks == 0) throw std::runtime_error("runbook: no fenced block in the document");
    return script;
}

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

// The last `bytes` of a file, for a failure message that is worth reading.
std::string tail_of(const fs::path& path, std::size_t bytes) {
    std::string text;
    try {
        text = read_file(path);
    } catch (const std::exception&) {
        return {};
    }
    return text.size() > bytes ? text.substr(text.size() - bytes) : text;
}

R97_TEST(runbook_script_runs_as_the_document_orders_it) {
    const fs::path root = source_root();
    const fs::path document_path = root / "docs" / "RUNBOOK.md";
    const std::string document = read_file(document_path);

    // The one-time inputs the script does not run are still the document's to name: a reader who
    // has neither written nor read the code starts by fetching the data, and section 7 serves
    // from the serving-only build rather than from the trainer's build.
    R97_CHECK(contains(document, "tools/fetch_ultrachat.py"));
    R97_CHECK(contains(document, "docs/DATA.md"));
    R97_CHECK(contains(document, "-DR97_PATCH_SET=serving"));
    R97_CHECK(contains(document, "stop a process you did not start"));

    const std::string script = extract_script(document);
    // The script is the whole procedure, not a fragment of it that still exits 0.
    R97_CHECK(contains(script, "\"$r97\" ops"));
    R97_CHECK(contains(script, "train --dry-run"));
    R97_CHECK(contains(script, "train --config"));
    R97_CHECK(contains(script, "--resume"));
    R97_CHECK(contains(script, "loss --model"));
    R97_CHECK(contains(script, "\"kind\":\"train\""));
    // The serve and the served loss are commands of the script, not prose beside it: the serving
    // build's server, the completion it answers, and radiance's KL record of the same adapter.
    // The test runs them, but with no serving tree they take their documented `else` branch.
    R97_CHECK(contains(script, "radiance-build/bin/radiance"));
    R97_CHECK(contains(script, "--lora trained="));
    R97_CHECK(contains(script, "/v1/chat/completions"));
    R97_CHECK(contains(script, "--kld-corpus"));
    R97_CHECK(contains(script, "--kld-lora trained"));
    R97_CHECK(contains(script, "tools/kld_loss.py"));
    // The card's inventory is a command of the script, not prose beside it: the real `--device`
    // line sits next to the tested host one, and the no-card run says it took the skip branch.
    R97_CHECK(contains(script, "--device \"$R97_DEVICE\""));

    const fs::path dir = fs::path(radtest::tmpdir());
    const fs::path run = dir / "runbook";
    // The document's own rule: the run's directory must not exist before the run. A rerun into a
    // fixed temporary directory starts from nothing, as a fresh checkout does.
    for (const fs::path& stale : {run, fs::path(run.string() + "-resumed"),
                                  fs::path(dir / "runbook.conf"), fs::path(dir / "runbook.resolved"),
                                  fs::path(dir / "runbook.console.log"),
                                  fs::path(dir / "runbook-resumed.console.log")})
        fs::remove_all(stale);

    const r97::Fixture& fixture = r97::w8a8_fixture();
    R97_CHECK(!fixture.container.empty());
    R97_CHECK(!r97::detail::build_dir().empty());

    const fs::path script_path = dir / "runbook.sh";
    const fs::path log_path = dir / "runbook.log";
    {
        std::ofstream out(script_path);
        out << script;
        if (!out) throw std::runtime_error("runbook: cannot write " + script_path.string());
    }

    // Run it the way the document says to: from the tree's top, with bash. The container is named
    // the way the document finds it -- the newest tiny_qwen35_w8a8.rad under the temporary
    // directory -- so the settings block is exercised rather than bypassed. The serving build is
    // pointed at a directory that does not exist: building it costs minutes, so the host tier runs
    // section 7's trainer measurement and the serve blocks' documented skip branch, not a serve.
    // `R97_DEVICE` is cleared for the same reason: a card handed down from the environment would
    // make section 2's card branch run on a host tier that is meant never to touch one.
    const fs::path no_serving = dir / "no-serving-build";
    const std::string command =
        "cd " + r97::detail::shell_quote(root.string()) + " && " +
        "HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=99 R97_DEVICE= " + "RAD_TEST_TMPDIR=" +
        r97::detail::shell_quote(dir.string()) + " " + "R97_BUILD=" +
        r97::detail::shell_quote(r97::detail::build_dir()) + " " + "R97_RUN=" +
        r97::detail::shell_quote(run.string()) + " " + "R97_SERVING_BUILD=" +
        r97::detail::shell_quote(no_serving.string()) + " bash " +
        r97::detail::shell_quote(script_path.string()) + " >" +
        r97::detail::shell_quote(log_path.string()) + " 2>&1";
    const int status = std::system(command.c_str());
    const bool ok = status != -1 && WIFEXITED(status) && WEXITSTATUS(status) == 0;
    if (!ok) {
        const int code = status != -1 && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        throw std::runtime_error("runbook: the document's script failed (exit " +
                                 std::to_string(code) + ", " + script_path.string() + ", log " +
                                 log_path.string() + ")\n" + tail_of(log_path, 4000));
    }

    // And it ran the run: the two checkpoints, the resumed one, the trainer's measurement of the
    // adapter, and the documented skip of the serve that needs the serving tree.
    const std::string log = read_file(log_path);
    R97_CHECK(contains(log, "the trainer's loss for the adapter"));
    R97_CHECK(contains(log, "section 0's serving-only build is a one-time input"));
    R97_CHECK(contains(log, "the serving build's loss is unchecked"));
    R97_CHECK(contains(log, "the card's inventory (ops --device) was not taken"));
    R97_CHECK(contains(log, "# dry run:"));
    R97_CHECK(fs::exists(run / "checkpoint-000000005" / "adapter"));
    R97_CHECK(fs::exists(run / "checkpoint-000000010" / "adapter"));
    R97_CHECK(fs::exists(fs::path(run.string() + "-resumed") / "checkpoint-000000010" / "adapter"));
}

}  // namespace

#endif  // R97_FIXTURE_PREFILL && defined(R97_HAS_R97TRAIN)
