// tests/host/runbook.cpp -- `docs/RUNBOOK.md`'s commands, extracted and run.
//
// The runbook says that every `sh` block in it is one fragment of a single script, and that this
// test is what keeps the document honest. It is: the blocks between ```sh fences are taken in the
// order the file has them, written out as one script, and run from the tree's top against the tiny
// int8 fixture, with the binary of this build. A flag that changes without the document, a check
// the document states that the tree no longer holds to, or a block that stops working fails here
// rather than in front of an operator.
//
// What is checked beyond "the script exits 0":
//   - the extracted script really is the procedure: it has the dry run, the run, the resume and
//     the served-loss comparison in it, so a fenced block that loses its `sh` tag does not turn the
//     script into a shorter one that still passes;
//   - the one-time inputs the script deliberately does not run are still named by the document: the
//     data fetch (`tools/fetch_ultrachat.py`, `docs/DATA.md`) and the serving-only build
//     (`-DR97_PATCH_SET=serving`), which is what "fetch the data, ..., serve" asks the file to say.
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

// The body of every ```sh block, in file order, as one script. A fenced block that is never closed
// is an error: the file would otherwise lose its last commands silently.
std::string extract_script(const std::string& document) {
    std::string script;
    bool open = false;
    std::size_t blocks = 0;
    std::size_t line = 0;
    for (std::size_t at = 0; at <= document.size();) {
        const std::size_t end = document.find('\n', at);
        const std::string text = document.substr(at, end == std::string::npos ? end : end - at);
        ++line;
        if (text == "```sh") {
            if (open) throw std::runtime_error("runbook: a ```sh block at line " +
                                               std::to_string(line) + " opens inside another");
            open = true;
            ++blocks;
        } else if (text == "```") {
            if (open) open = false;
        } else if (open) {
            script += text;
            script += '\n';
        }
        if (end == std::string::npos) break;
        at = end + 1;
    }
    if (open) throw std::runtime_error("runbook: the last ```sh block is never closed");
    if (blocks == 0) throw std::runtime_error("runbook: no ```sh block in the document");
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
    // has neither written nor read the code starts by fetching the data and ends by serving from
    // the serving-only build.
    R97_CHECK(contains(document, "tools/fetch_ultrachat.py"));
    R97_CHECK(contains(document, "docs/DATA.md"));
    R97_CHECK(contains(document, "-DR97_PATCH_SET=serving"));
    R97_CHECK(contains(document, "stop a process you did not start"));

    const std::string script = extract_script(document);
    // The script is the whole procedure, not a fragment of it that still exits 0.
    R97_CHECK(contains(script, "train --dry-run"));
    R97_CHECK(contains(script, "train --config"));
    R97_CHECK(contains(script, "--resume"));
    R97_CHECK(contains(script, "loss --model"));

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
    // directory -- so the settings block is exercised rather than bypassed.
    const std::string command =
        "cd " + r97::detail::shell_quote(root.string()) + " && " +
        "HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=99 " + "RAD_TEST_TMPDIR=" +
        r97::detail::shell_quote(dir.string()) + " " + "R97_BUILD=" +
        r97::detail::shell_quote(r97::detail::build_dir()) + " " + "R97_RUN=" +
        r97::detail::shell_quote(run.string()) + " bash " +
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

    // And it ran the run: the two checkpoints, the resumed one, and the served loss's line.
    const std::string log = read_file(log_path);
    R97_CHECK(contains(log, "served loss"));
    R97_CHECK(contains(log, "# dry run:"));
    R97_CHECK(fs::exists(run / "checkpoint-000000005" / "adapter"));
    R97_CHECK(fs::exists(run / "checkpoint-000000010" / "adapter"));
    R97_CHECK(fs::exists(fs::path(run.string() + "-resumed") / "checkpoint-000000010" / "adapter"));
}

}  // namespace

#endif  // R97_FIXTURE_PREFILL && defined(R97_HAS_R97TRAIN)
