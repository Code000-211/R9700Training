// tests/host/train_host_cli.cpp -- `r97train train` over the host step, end to end.
//
// tests/host/train_reference_cli.cpp runs the reference step beside this: the same command and
// loop, but a step built from the f64 twins instead of the engines. This file runs the ordinary
// host step (ForwardDriver, LiveAdapter, radiance's host kernels) on the int8 fixture, whose trunk
// is stored rotated (w8a8_fwht128): the live adapter must declare the basis its A factors are
// stored in, or radiance refuses the slot before a single step runs.
//
// What it checks:
//   - both runs overfit the three conversations: grouped by their scored-token count, each
//     conversation's loss at its last step is at most half its loss at its first;
//   - a run of 2N steps and a run of the same 2N-step configuration resumed from its own N-step
//     checkpoint give the same losses, bit for bit, and the same checkpoint files byte for byte;
//   - the run's base_id is the container's SHA-256: resuming its checkpoint against the other
//     fixture is refused, and the error names both hashes;
//   - the run's configuration in the output directory (`run.json`) names the objective and the
//     delta and carries that base_id;
//   - `train --dry-run` resolves the same configuration and stops before the first step: it prints
//     the settings and the first step's shape, and writes nothing -- not the output directory.
//     It refuses `--resume`, which would be a run;
//   - the adapter the run wrote is served by radiance itself: `r97train loss --lora <adapter>`,
//     which preloads the directory through lora/preload.h, agrees with the held-out loss the
//     trainer reported on the same file to 1e-4 relative. The adapter's adapter_config.json names
//     the trunk's basis, which is what lets radiance ride the rotated weights.
#include "../support/fixture_w8a8.h"
#include "../support/test.h"
#include "r97/sha256.h"

#include <algorithm>
#include <cmath>
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

#if R97_FIXTURE_PREFILL && defined(R97_HAS_R97TRAIN)

namespace {

namespace fs = std::filesystem;

// 2N in the acceptance's "2N steps equals N resumed for N more": the host step runs an engine per
// step, so this is long enough to overfit without making the check take minutes.
constexpr int64_t kSteps = 20;
constexpr int64_t kHalf = kSteps / 2;

// The int8 run's horizon. Its trunk's straight-through gradient is noisier than the bf16 run's, so
// it takes more steps to bring the longest conversation (the one that costs 119 tokens a step)
// below half; 48 steps at 1e-2 do it with room to spare.
constexpr int64_t kInt8Steps = 48;
constexpr const char* kInt8Lr = "1e-2";

// The modules this recipe leaves in the rotated trunk. `linear_attn.out_proj` (blk.<l>.ssm_out) is
// the one line the w8a8 recipe stores unrotated, so an adapter in the fwht128 basis cannot ride
// it; radiance refuses it at engine start and the target list leaves it out.
const char* const kTargets =
    "q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,gate_proj,up_proj,down_proj";

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
    int status = -1;
    std::string output;  // stdout
    std::string error;   // stderr, where radiance logs
};

// One `r97train` run in a child process, the environment a person gets except that no card is
// visible: the host step must not reach for one.
Outcome run(const std::string& args, const fs::path& stdout_path) {
    const fs::path exe = fs::read_symlink("/proc/self/exe");
    const std::string tool = (exe.parent_path() / "r97train").string();
    const fs::path stderr_path = stdout_path.string() + ".stderr";
    const std::string cmd = "HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=99 " +
                            r97::detail::shell_quote(tool) + " " + args + " >" +
                            r97::detail::shell_quote(stdout_path.string()) + " 2>" +
                            r97::detail::shell_quote(stderr_path.string());
    FILE* pipe = ::popen(cmd.c_str(), "r");
    if (pipe == nullptr) throw std::runtime_error("r97train: cannot start the tool");
    const int status = ::pclose(pipe);

    Outcome result;
    result.status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    for (const fs::path& path : {stdout_path, stderr_path}) {
        std::ifstream in(path);
        std::string& to = path == stdout_path ? result.output : result.error;
        to.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    return result;
}

void require_ok(const Outcome& out, const std::string& what) {
    if (out.status == 0) return;
    std::string tail = out.error;
    const std::size_t kept = 2000;
    if (tail.size() > kept) tail = tail.substr(tail.size() - kept);
    throw std::runtime_error(what + ": exit " + std::to_string(out.status) + "\n" + tail);
}

// One training run: the host step, the given container and the fixed conversations.
Outcome run_train(const fs::path& container, const fs::path& out, const std::string& args) {
    return run("train --model " + r97::detail::shell_quote(container.string()) + " --data " +
                   r97::detail::shell_quote(chat_small().string()) + " --out " +
                   r97::detail::shell_quote(out.string()) + " --delta.targets " + kTargets + " " +
                   args,
               out.parent_path() / (out.filename().string() + ".stdout"));
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
    if (!in) throw std::runtime_error("train: no train.jsonl in " + out.string());
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
void check_overfits(const std::vector<Step>& steps, int64_t horizon) {
    R97_CHECK(steps.size() == std::size_t(horizon));
    std::map<int64_t, std::vector<double>> by_conversation;
    for (const Step& s : steps) by_conversation[s.tokens].push_back(s.loss);
    if (by_conversation.size() != 3)
        throw std::runtime_error("train: expected the three fixed conversations, saw " +
                                 std::to_string(by_conversation.size()) + " token counts");
    for (const auto& [tokens, losses] : by_conversation) {
        if (losses.size() < 3)
            throw std::runtime_error("train: the conversation of " + std::to_string(tokens) +
                                     " scored tokens ran " + std::to_string(losses.size()) +
                                     " time(s)");
        const double first = losses.front();
        const double last = losses.back();
        if (!(last <= 0.5 * first))
            throw std::runtime_error("train: the conversation of " + std::to_string(tokens) +
                                     " scored tokens went from " + std::to_string(first) + " to " +
                                     std::to_string(last) + ", not to half of it");
    }
}

// Two checkpoint directories, file for file and byte for byte.
void check_same_checkpoint(const fs::path& a, const fs::path& b) {
    std::vector<fs::path> names;
    for (const fs::path& dir : {a, b})
        for (const fs::directory_entry& e : fs::recursive_directory_iterator(dir))
            if (e.is_regular_file()) names.push_back(fs::relative(e.path(), dir));
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    if (names.empty()) throw std::runtime_error("train: empty checkpoint " + a.string());
    for (const fs::path& name : names) {
        std::ifstream left(a / name, std::ios::binary), right(b / name, std::ios::binary);
        if (!left || !right)
            throw std::runtime_error("train: the two checkpoints disagree on " + name.string());
        const std::string l((std::istreambuf_iterator<char>(left)),
                            std::istreambuf_iterator<char>());
        const std::string r((std::istreambuf_iterator<char>(right)),
                            std::istreambuf_iterator<char>());
        if (l != r)
            throw std::runtime_error("train: " + name.string() +
                                     " differs between the run and its resume");
    }
}

double read_loss(const Outcome& out, const std::string& what) {
    require_ok(out, what);
    const std::string key = "loss:";
    const std::size_t at = out.output.find(key);
    if (at == std::string::npos) throw std::runtime_error(what + ": no loss line\n" + out.output);
    return std::stod(out.output.substr(at + key.size()));
}

// The last held-out line the loop logged: the loss it reports for the file `--heldout` named, and
// the tokens it covered.
struct Eval {
    int64_t step = 0;
    double  loss = 0.0;
    int64_t scored = 0;
};

Eval read_last_eval(const fs::path& out) {
    std::ifstream in(out / "train.jsonl");
    if (!in) throw std::runtime_error("train: no train.jsonl in " + out.string());
    bool found = false;
    Eval last;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const nlohmann::json row = nlohmann::json::parse(line);
        if (row.value("kind", std::string()) != "eval") continue;
        last.step = row.at("step").get<int64_t>();
        last.loss = row.at("loss").get<double>();
        last.scored = row.at("scored").get<int64_t>();
        found = true;
    }
    if (!found)
        throw std::runtime_error("train: the run logged no held-out loss in " + out.string());
    return last;
}

nlohmann::json read_run_config(const fs::path& out) {
    std::ifstream in(out / "run.json");
    if (!in) throw std::runtime_error("train: no run.json in " + out.string());
    nlohmann::json config;
    in >> config;
    return config;
}

std::string container_base_id(const std::string& path) {
    return "sha256:" + r97::sha256_file_hex(path);
}

// The adapter's own declaration of the basis its A factors are stored in. It is what lets
// radiance's loader ride the rotated weights, and the run is refused at engine start without it.
void check_adapter_names_the_basis(const fs::path& adapter) {
    std::ifstream in(adapter / "adapter_config.json");
    if (!in) throw std::runtime_error("train: no adapter_config.json in " + adapter.string());
    nlohmann::json j;
    in >> j;
    if (j.value("radiance_input_basis", std::string()) != "fwht128")
        throw std::runtime_error("train: the adapter does not declare the w8a8 trunk's fwht128 "
                                 "basis");
}

}  // namespace

R97_TEST(train_host_cli_overfits_and_resumes) {
    const r97::Fixture& fixture = r97::fixture();
    const fs::path    root      = fs::path(radtest::tmpdir()) / "train-host-cli";
    std::error_code   ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);

    // The run the two halves are compared over: 2N steps in one process, which also leaves the
    // N-step checkpoint the resumed half starts from.
    const fs::path straight = root / "straight";
    const Outcome one =
        run_train(fixture.container, straight,
                  "--steps " + std::to_string(kSteps) + " --checkpoint_every " +
                      std::to_string(kHalf) + " --lr 1e-2");
    require_ok(one, "the run of " + std::to_string(kSteps) + " steps");
    const std::vector<Step> steps = read_log(straight);
    for (std::size_t i = 0; i < steps.size(); ++i) R97_CHECK(steps[i].step == int64_t(i) + 1);
    check_overfits(steps, kSteps);

    // The configuration the run left in the output directory names what it trained: the
    // objective, the delta, and the container it rides, by hash.
    const nlohmann::json config = read_run_config(straight);
    R97_CHECK(config.at("objective").get<std::string>() == "ce");
    R97_CHECK(config.at("delta.kind").get<std::string>() == "lora");
    const std::string base_id = container_base_id(fixture.container);
    R97_CHECK(config.at("base_id").get<std::string>() == base_id);

    const fs::path checkpoint_half = straight / checkpoint_name(kHalf);
    if (!fs::exists(checkpoint_half / "trainer.state"))
        throw std::runtime_error("train: no " + checkpoint_half.string());

    // The resumed run is the straight run told to continue from its own half: same horizon (the
    // learning-rate schedule is the run's horizon), same batcher state from the checkpoint.
    const fs::path resumed = root / "resumed";
    const Outcome two =
        run_train(fixture.container, resumed,
                  "--steps " + std::to_string(kSteps) + " --lr 1e-2 --resume " +
                      r97::detail::shell_quote(checkpoint_half.string()));
    require_ok(two, "the resume from step " + std::to_string(kHalf));

    const std::vector<Step> tail = read_log(resumed);
    R97_CHECK(tail.size() == std::size_t(kSteps - kHalf));
    for (const Step& s : tail) {
        const auto found = std::find_if(steps.begin(), steps.end(),
                                        [&](const Step& a) { return a.step == s.step; });
        if (found == steps.end())
            throw std::runtime_error("train: the resume logged a step the run did not");
        // Both losses are the same double, printed at 17 significant digits and read back, so the
        // comparison is bit for bit and not a tolerance.
        if (found->loss != s.loss || found->tokens != s.tokens)
            throw std::runtime_error("train: step " + std::to_string(s.step) +
                                     " differs between the run and its resume");
    }
    check_same_checkpoint(straight / checkpoint_name(kSteps), resumed / checkpoint_name(kSteps));
    R97_CHECK(read_run_config(resumed).at("base_id").get<std::string>() == base_id);

    // Serve the adapter with radiance: `loss --lora` preloads the directory through
    // radiance's own loader, so the measurement runs on what the run wrote, not on the trainer's
    // live slot. The base model is the control: the adapter has to carry the overfit.
    const fs::path adapter = straight / checkpoint_name(kSteps) / "adapter";
    const fs::path loss_log = root / "loss";
    const double base = read_loss(run("loss --model " +
                                          r97::detail::shell_quote(fixture.container) +
                                          " --data " +
                                          r97::detail::shell_quote(chat_small().string()),
                                      loss_log),
                                  "the base model's loss");
    const double served = read_loss(run("loss --model " +
                                            r97::detail::shell_quote(fixture.container) +
                                            " --data " +
                                            r97::detail::shell_quote(chat_small().string()) +
                                            " --lora " +
                                            r97::detail::shell_quote(adapter.string()),
                                        loss_log),
                                    "the served adapter's loss");
    if (!(served <= 0.5 * base))
        throw std::runtime_error("train: radiance serves the adapter at " +
                                 std::to_string(served) + ", not at half the base model's " +
                                 std::to_string(base));
}

R97_TEST(train_host_cli_serves_the_int8_trunk) {
    const r97::Fixture& fixture = r97::w8a8_fixture();
    const fs::path    root      = fs::path(radtest::tmpdir()) / "train-host-cli-int8";
    std::error_code   ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);

    // The rotated trunk's run: 2N steps, a checkpoint at N, and the same horizon resumed from it.
    // The held-out file is scored at the horizon, so the last eval line and the last checkpoint
    // are the same weights -- the comparison below is against the adapter radiance will load.
    const fs::path straight = root / "straight";
    const Outcome one =
        run_train(fixture.container, straight,
                  "--steps " + std::to_string(kInt8Steps) + " --checkpoint_every " +
                      std::to_string(kInt8Steps / 2) + " --lr " + std::string(kInt8Lr) +
                      " --heldout " + r97::detail::shell_quote(chat_small().string()) +
                      " --eval_every " + std::to_string(kInt8Steps));
    require_ok(one, "the int8 run of " + std::to_string(kInt8Steps) + " steps");
    const std::vector<Step> steps = read_log(straight);
    R97_CHECK(steps.size() == std::size_t(kInt8Steps));
    check_overfits(steps, kInt8Steps);

    const fs::path checkpoint_half = straight / checkpoint_name(kInt8Steps / 2);
    if (!fs::exists(checkpoint_half / "trainer.state"))
        throw std::runtime_error("train: no " + checkpoint_half.string());
    const fs::path resumed = root / "resumed";
    const Outcome two =
        run_train(fixture.container, resumed,
                  "--steps " + std::to_string(kInt8Steps) + " --lr " + std::string(kInt8Lr) +
                      " --resume " +
                      r97::detail::shell_quote(checkpoint_half.string()));
    require_ok(two, "the int8 resume from step " + std::to_string(kInt8Steps / 2));

    const std::vector<Step> tail = read_log(resumed);
    R97_CHECK(tail.size() == std::size_t(kInt8Steps / 2));
    for (const Step& s : tail) {
        const auto found = std::find_if(steps.begin(), steps.end(),
                                        [&](const Step& a) { return a.step == s.step; });
        if (found == steps.end())
            throw std::runtime_error("train: the int8 resume logged a step the run did not");
        if (found->loss != s.loss || found->tokens != s.tokens)
            throw std::runtime_error("train: int8 step " + std::to_string(s.step) +
                                     " differs between the run and its resume");
    }
    check_same_checkpoint(straight / checkpoint_name(kInt8Steps),
                          resumed / checkpoint_name(kInt8Steps));

    // The adapter declares the trunk's basis -- radiance refuses the slot at engine start without
    // it -- and radiance's own loader then serves it: `loss --lora` scores the held-out file
    // through preload.h, on the written directory, and must report what the trainer's held-out
    // pass reported on the same weights. 1e-4 relative is the acceptance's tolerance.
    const fs::path adapter = straight / checkpoint_name(kInt8Steps) / "adapter";
    check_adapter_names_the_basis(adapter);
    const Eval held = read_last_eval(straight);
    R97_CHECK(held.step == kInt8Steps);
    if (!(held.loss > 0.0) || !std::isfinite(held.loss))
        throw std::runtime_error("train: the train reported a held-out loss of " +
                                 std::to_string(held.loss));
    const double served =
        read_loss(run("loss --model " + r97::detail::shell_quote(fixture.container) +
                          " --data " + r97::detail::shell_quote(chat_small().string()) +
                          " --lora " + r97::detail::shell_quote(adapter.string()),
                      root / "loss"),
                  "the int8 adapter's loss");
    const double relative = std::fabs(served - held.loss) / std::fabs(held.loss);
    if (!(relative <= 1e-4))
        throw std::runtime_error("train: the trainer reported a held-out loss of " +
                                 std::to_string(held.loss) + " and radiance serves it at " +
                                 std::to_string(served) + " (relative " +
                                 std::to_string(relative) + ")");

    // The checkpoint carries the container's hash as its base_id: resuming it against the other
    // fixture is refused, and the error names both ids. It is refused before the loop starts, so
    // no run.json is written into the refused run's directory.
    const r97::Fixture& bf16 = r97::fixture();
    const Outcome refused =
        run_train(bf16.container, root / "refused",
                  "--steps " + std::to_string(kInt8Steps) + " --lr " + std::string(kInt8Lr) +
                      " --resume " +
                      r97::detail::shell_quote((straight / checkpoint_name(kInt8Steps)).string()));
    if (refused.status == 0)
        throw std::runtime_error("train: resuming an int8 checkpoint against the bf16 container "
                                 "was allowed");
    const std::string int8_id = container_base_id(fixture.container);
    const std::string bf16_id = container_base_id(bf16.container);
    if (refused.error.find(int8_id) == std::string::npos ||
        refused.error.find(bf16_id) == std::string::npos)
        throw std::runtime_error("train: the refused resume does not name both base ids (" +
                                 int8_id + ", " + bf16_id + ")\n" + refused.error);
    if (fs::exists(root / "refused" / "run.json"))
        throw std::runtime_error("train: the refused resume wrote run.json");
}

// `train --dry-run`: the preflight docs/USAGE.md's flag table and docs/RUNBOOK.md section 4 promise.
// The same arguments as a run, one flag more, and nothing written: the configuration resolved and
// printed, and the output directory not made -- which is the difference the flag is for, since the
// Trainer makes it as its first act.
R97_TEST(train_cli_dry_run_resolves_and_writes_nothing) {
    const r97::Fixture& fixture = r97::fixture();
    const fs::path root = fs::path(radtest::tmpdir()) / "dry_run";
    fs::create_directories(root);

    const Outcome dry = run_train(fixture.container, root / "out",
                                  "--steps 10 --lr 1e-2 --heldout " +
                                      r97::detail::shell_quote(chat_small().string()) +
                                      " --dry-run");
    require_ok(dry, "train --dry-run");
    R97_CHECK(dry.output.find("steps = 10") != std::string::npos);
    R97_CHECK(dry.output.find("# dry run:") != std::string::npos);
    R97_CHECK(!fs::exists(root / "out"));
    R97_CHECK(!fs::exists(root / "out" / "train.jsonl"));

    // A dry run is not a resume: the flag refuses the pair rather than silently ignoring one.
    const Outcome refused =
        run_train(fixture.container, root / "out2",
                  "--steps 10 --lr 1e-2 --dry-run --resume " +
                      r97::detail::shell_quote((root / "nothing").string()));
    R97_CHECK(refused.status != 0);
    R97_CHECK(refused.error.find("--dry-run does not start a resume") != std::string::npos);
    R97_CHECK(!fs::exists(root / "out2"));
}

#endif  // R97_FIXTURE_PREFILL && R97_HAS_R97TRAIN
