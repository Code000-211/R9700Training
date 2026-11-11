// tests/host/train_reference_cli.cpp -- `r97train train --reference` end to end.
//
// tests/host/reference_step.cpp checks one step's loss and gradient, and
// tests/host/loop_reference.cpp the loop with the step in hand. This file runs the command the
// issue's third acceptance names, on the fixed conversations: the real Trainer, the real batcher
// over tests/fixtures/chat_small.jsonl, the live trace, the checkpoint and the resume, in a child
// process with no card available (HIP_VISIBLE_DEVICES empty, ROCR_VISIBLE_DEVICES pointing at no
// card) and with the loader's own log (LD_DEBUG=libs) watching what it loads.
//
// What it checks:
//   - the run overfits the three conversations: grouped by their scored-token count, each
//     conversation's loss at its last step is at most half its loss at its first;
//   - a run of 2N steps and a run of the same 2N-step configuration resumed from its own N-step
//     checkpoint give the same losses, bit for bit, and the same checkpoint files byte for byte.
//     The learning-rate schedule is the run's horizon, so the N-step checkpoint has to come out of
//     a 2N-step run (`--checkpoint_every N`): a run told `--steps N` is not the first half of a
//     run told `--steps 2N`;
//   - no device library is loaded. LD_DEBUG names every object the loader initialised, so a
//     process that had loaded libr4d could not pass; libavx/libref appearing in the same list is
//     the control that the log is not simply empty (the live trace does run the host chain).
//     On a build without libr4d.so this half proves nothing on its own, which is why the control
//     is checked too.
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <vector>

#include "support/late_fixture.h"
#include "support/test.h"

namespace {

namespace fs = std::filesystem;
// The fixture's paths, copied before the host tier's `main` runs the checks.
const r97test::LatePaths& late = r97test::late_paths();

// 2N in the acceptance's "2N steps equals N resumed for N more": a little over six passes over the
// three fixed conversations, which is long enough for the loss to halve on each of them.
constexpr int64_t kSteps = 20;
constexpr int64_t kHalf = kSteps / 2;

fs::path chat_small() {
    return fs::path(__FILE__).parent_path().parent_path() / "fixtures" / "chat_small.jsonl";
}

// The Trainer names a checkpoint directory `checkpoint-%09d` of the step it holds.
std::string checkpoint_name(int64_t step) {
    char name[32];
    std::snprintf(name, sizeof name, "checkpoint-%09lld", static_cast<long long>(step));
    return name;
}

struct Outcome {
    int status = 0;
    std::string error;   // the child's stderr, which also carries LD_DEBUG=libs
};

// One `r97train train --reference` run, its stderr to a file of its own. The environment is the
// one a person gets, except that no card is visible and the loader is asked to say what it loads.
Outcome run_train(const std::string& args, const fs::path& out) {
    const std::string exe = fs::read_symlink("/proc/self/exe");
    const std::string tool = (fs::path(exe).parent_path() / "r97train").string();
    const fs::path stderr_path = out.parent_path() / (out.filename().string() + ".stderr");
    const std::string cmd = "HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=99 LD_DEBUG=libs " +
                            r97::detail::shell_quote(tool) + " train --reference --model " +
                            r97::detail::shell_quote(r97test::late_container()) + " --data " +
                            r97::detail::shell_quote(chat_small().string()) + " --out " +
                            r97::detail::shell_quote(out.string()) + " " + args + " 2>" +
                            r97::detail::shell_quote(stderr_path.string());
    FILE* pipe = ::popen(cmd.c_str(), "r");
    if (pipe == nullptr) throw std::runtime_error("train --reference: cannot start the tool");
    char buf[4096];
    while (std::fread(buf, 1, sizeof buf, pipe) > 0) {
    }
    const int status = ::pclose(pipe);

    Outcome result;
    result.status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    std::ifstream in(stderr_path);
    result.error.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return result;
}

void require_ok(const Outcome& run, const std::string& what) {
    if (run.status == 0) return;
    std::string tail = run.error;
    const std::size_t kept = 2000;
    if (tail.size() > kept) tail = tail.substr(tail.size() - kept);
    throw std::runtime_error("train --reference: " + what + ": exit " +
                             std::to_string(run.status) + "\n" + tail);
}

struct Step {
    int64_t step = 0;
    int64_t tokens = 0;
    double loss = 0.0;
};

// The step log the Trainer writes beside the checkpoints: one JSON object per step, the loss at
// 17 significant digits, so a value read back here is the value the step computed.
std::vector<Step> read_log(const fs::path& out) {
    std::ifstream in(out / "train.jsonl");
    if (!in) throw std::runtime_error("train --reference: no train.jsonl in " + out.string());
    std::vector<Step> steps;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const nlohmann::json row = nlohmann::json::parse(line);
        if (row.value("kind", std::string()) != "train") continue;
        Step s;
        s.step = row.at("step").get<int64_t>();
        s.tokens = row.at("tokens").get<int64_t>();
        s.loss = row.at("loss").get<double>();
        steps.push_back(s);
    }
    return steps;
}

// Every conversation's last loss is at most half its first. A conversation is identified by the
// scored tokens one step of it costs, which separates these three (29, 53 and 119).
void check_overfits(const std::vector<Step>& steps) {
    R97_CHECK(steps.size() == std::size_t(kSteps));
    std::map<int64_t, std::vector<double>> by_conversation;
    for (const Step& s : steps) by_conversation[s.tokens].push_back(s.loss);
    if (by_conversation.size() != 3)
        throw std::runtime_error("train --reference: expected the three fixed conversations, saw " +
                                 std::to_string(by_conversation.size()) + " token counts");
    for (const auto& [tokens, losses] : by_conversation) {
        if (losses.size() < 3)
            throw std::runtime_error("train --reference: the conversation of " +
                                     std::to_string(tokens) + " scored tokens ran " +
                                     std::to_string(losses.size()) + " time(s)");
        const double first = losses.front();
        const double last = losses.back();
        if (!(last <= 0.5 * first))
            throw std::runtime_error("train --reference: the conversation of " +
                                     std::to_string(tokens) + " scored tokens went from " +
                                     std::to_string(first) + " to " + std::to_string(last) +
                                     ", not to half of it");
    }
}

// The objects the loader mapped, from LD_DEBUG=libs's report: it names each library it initialises
// (`calling init:`) and each one it unloads or runs the exit of (`calling fini:`), which between
// them is every shared object the process held, dlopen'ed plugin or not.
std::vector<std::string> mapped_objects(const std::string& loader_log) {
    std::vector<std::string> objects;
    const std::string markers[] = {"calling init: ", "calling fini: "};
    for (const std::string& marker : markers) {
        for (std::size_t at = loader_log.find(marker); at != std::string::npos;
             at = loader_log.find(marker, at + 1)) {
            std::size_t begin = at + marker.size();
            std::size_t end = loader_log.find('\n', begin);
            if (end == std::string::npos) end = loader_log.size();
            objects.push_back(loader_log.substr(begin, end - begin));
        }
    }
    return objects;
}

void check_loads_no_device_library(const std::string& loader_log) {
    const std::vector<std::string> objects = mapped_objects(loader_log);
    bool host_kernel = false;
    for (const std::string& path : objects) {
        if (path.find("libr4d") != std::string::npos)
            throw std::runtime_error("train --reference loaded the device library: " + path);
        if (path.find("libavx") != std::string::npos || path.find("libref") != std::string::npos)
            host_kernel = true;
    }
    if (!host_kernel)
        throw std::runtime_error(
            "train --reference: the loader's log names no host kernel library, so it cannot be "
            "read as evidence about what the run loaded");
}

// Two checkpoint directories, file for file and byte for byte.
void check_same_checkpoint(const fs::path& a, const fs::path& b) {
    std::vector<fs::path> names;
    for (const fs::path& dir : {a, b}) {
        for (const fs::directory_entry& e : fs::recursive_directory_iterator(dir))
            if (e.is_regular_file()) names.push_back(fs::relative(e.path(), dir));
    }
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    if (names.empty())
        throw std::runtime_error("train --reference: empty checkpoint " + a.string());
    for (const fs::path& name : names) {
        std::ifstream left(a / name, std::ios::binary), right(b / name, std::ios::binary);
        if (!left || !right)
            throw std::runtime_error("train --reference: the two checkpoints disagree on " +
                                     name.string());
        const std::string l((std::istreambuf_iterator<char>(left)),
                            std::istreambuf_iterator<char>());
        const std::string r((std::istreambuf_iterator<char>(right)),
                            std::istreambuf_iterator<char>());
        if (l != r)
            throw std::runtime_error("train --reference: " + name.string() +
                                     " differs between the run and its resume");
    }
}

}  // namespace

R97_TEST(train_reference_cli_overfits_and_resumes) {
#if !R97_FIXTURE_PREFILL
    return;
#else
    const fs::path root = fs::path(late.tmpdir) / "train-reference-cli";
    fs::remove_all(root);
    fs::create_directories(root);

    // The run the two halves are compared over: 2N steps in one process, which also leaves the
    // N-step checkpoint the resumed half starts from.
    const fs::path straight = root / "straight";
    const Outcome one = run_train("--steps " + std::to_string(kSteps) + " --checkpoint_every " +
                                      std::to_string(kHalf),
                                  straight);
    require_ok(one, "the run of " + std::to_string(kSteps) + " steps");
    check_loads_no_device_library(one.error);
    const std::vector<Step> steps = read_log(straight);
    for (std::size_t i = 0; i < steps.size(); ++i)
        R97_CHECK(steps[i].step == int64_t(i) + 1);
    check_overfits(steps);

    const fs::path checkpoint_half = straight / checkpoint_name(kHalf);
    if (!fs::exists(checkpoint_half / "trainer.state"))
        throw std::runtime_error("train --reference: no " + checkpoint_half.string());

    const fs::path resumed = root / "resumed";
    const Outcome two = run_train("--steps " + std::to_string(kSteps) + " --resume " +
                                      r97::detail::shell_quote(checkpoint_half.string()),
                                  resumed);
    require_ok(two, "the resume from step " + std::to_string(kHalf));
    check_loads_no_device_library(two.error);

    // The resumed run's steps are the straight run's second half, loss for loss.
    const std::vector<Step> tail = read_log(resumed);
    R97_CHECK(tail.size() == std::size_t(kSteps - kHalf));
    for (const Step& s : tail) {
        const auto found = std::find_if(steps.begin(), steps.end(),
                                        [&](const Step& a) { return a.step == s.step; });
        if (found == steps.end())
            throw std::runtime_error("train --reference: the resume logged a step the run did not");
        // Both losses are the same double, printed at 17 significant digits and read back, so the
        // comparison is bit for bit and not a tolerance.
        if (found->loss != s.loss || found->tokens != s.tokens)
            throw std::runtime_error("train --reference: step " + std::to_string(s.step) +
                                     " differs between the run and its resume");
    }

    check_same_checkpoint(straight / checkpoint_name(kSteps), resumed / checkpoint_name(kSteps));
#endif
}
