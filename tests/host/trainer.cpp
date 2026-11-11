// tests/host/trainer.cpp -- the training loop over a toy StepFn (include/r97/trainer.h).
//
// The toy is a least-squares problem: `w` has two values, each sequence in a batch carries one
// example as the bit patterns of three floats (x0, x1, y), and the StepFn returns the summed
// squared error and its mean gradient. It needs no model and no container, and it is exactly
// deterministic, so the loop's scheduling, logging, checkpointing and resume can be tested on
// their own; the ReferenceStepFn repeats these properties on the real graph in its own file.
//
// The tests cover the acceptance of docs/TRAINING.md: the toy converges, a 2N-step run and an
// N-step run resumed for N more are bit-identical, a resume on another base_id is refused with
// both ids printed, a non-finite loss and a non-finite gradient stop the run and leave the last
// good checkpoint, and the log and checkpoint files are the ones the document describes.
#include "support/test.h"

#include "r97/trainer.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using r97::AdamWConfig;
using r97::AdapterWriter;
using r97::BaseIdMismatch;
using r97::Batch;
using r97::Batcher;
using r97::BatcherConfig;
using r97::Config;
using r97::ConfigError;
using r97::NonFiniteError;
using r97::Optimiser;
using r97::Param;
using r97::ParamSet;
using r97::StepFn;
using r97::StepResult;
using r97::TokenSeq;
using r97::Trainer;
using r97::TrainerOptions;
using r97::TrainerState;

namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------- temporary directories

struct TempDir {
    std::string path;

    TempDir() {
        char name[] = "/tmp/r97_trainer_XXXXXX";
        char* made = ::mkdtemp(name);
        if (made == nullptr) throw std::runtime_error("mkdtemp failed");
        path = made;
    }

    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);  // best effort: a leftover temp dir must not fail the test
    }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    fs::path operator/(const std::string& name) const { return fs::path(path) / name; }
};

std::vector<std::string> read_lines(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path.string());
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) lines.push_back(line);
    return lines;
}

std::string checkpoint_dir(const std::string& out, int64_t step) {
    std::ostringstream name;
    name << "checkpoint-" << std::setw(9) << std::setfill('0') << step;
    return (fs::path(out) / name.str()).string();
}

// ---------------------------------------------------------------- a very small JSON test reader
//
// Every line docs/TRAINING.md promises is one flat object whose values are numbers or strings;
// this finds a key and returns its raw value, so the checks below can assert the fields without
// pulling a JSON library into the test tier.

std::string json_raw(const std::string& line, const std::string& key) {
    const std::string pattern = "\"" + key + "\":";
    const size_t at = line.find(pattern);
    if (at == std::string::npos) throw std::runtime_error("no key '" + key + "' in: " + line);
    size_t begin = at + pattern.size();
    if (begin < line.size() && line[begin] == '"') {
        const size_t end = line.find('"', begin + 1);
        if (end == std::string::npos) throw std::runtime_error("unterminated string in: " + line);
        return line.substr(begin + 1, end - begin - 1);
    }
    size_t end = begin;
    while (end < line.size() && line[end] != ',' && line[end] != '}') ++end;
    return line.substr(begin, end - begin);
}

double json_number(const std::string& line, const std::string& key) {
    const std::string raw = json_raw(line, key);
    char* end = nullptr;
    const double value = std::strtod(raw.c_str(), &end);
    if (end == raw.c_str() || *end != '\0') throw std::runtime_error("not a number: " + raw);
    return value;
}

std::string json_string(const std::string& line, const std::string& key) {
    return json_raw(line, key);
}

// ---------------------------------------------------------------- the toy least-squares StepFn
//
// The target is y = x0 - 2*x1. Four examples are enough to pin both weights; the initial weights
// are zero, so a converging run ends near (1, -2).

constexpr float kExamples[4][3] = {
    {1.0f, 0.0f, 1.0f},
    {0.0f, 1.0f, -2.0f},
    {1.0f, 1.0f, -1.0f},
    {1.0f, -1.0f, 3.0f},
};

int32_t float_bits(float value) {
    int32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float bits_float(int32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

TokenSeq toy_sequence(int example) {
    TokenSeq seq;
    for (int i = 0; i < 3; ++i) seq.ids.push_back(float_bits(kExamples[example][i]));
    seq.score.assign(3, 0);
    return seq;
}

std::vector<TokenSeq> toy_sequences() {
    return {toy_sequence(0), toy_sequence(1), toy_sequence(2), toy_sequence(3)};
}

Batch toy_batch(std::initializer_list<int> examples) {
    Batch batch;
    for (const int example : examples) batch.push_back(toy_sequence(example));
    return batch;
}

ParamSet toy_initial_params() { return ParamSet{Param{"w", {2}, {0.0f, 0.0f}}}; }

StepResult toy_step(const Batch& batch, const ParamSet& params, ParamSet* grads) {
    if (params.size() != 1 || params[0].value.size() != 2)
        throw std::runtime_error("toy_step: unexpected parameters");
    if (batch.empty()) throw std::runtime_error("toy_step: empty batch");

    double loss = 0.0;
    double g0 = 0.0;
    double g1 = 0.0;
    for (const TokenSeq& seq : batch) {
        if (seq.ids.size() != 3) throw std::runtime_error("toy_step: not a toy sequence");
        const double x0 = bits_float(seq.ids[0]);
        const double x1 = bits_float(seq.ids[1]);
        const double y = bits_float(seq.ids[2]);
        const double residual = params[0].value[0] * x0 + params[0].value[1] * x1 - y;
        loss += residual * residual;
        g0 += 2.0 * residual * x0;
        g1 += 2.0 * residual * x1;
    }
    const double n = static_cast<double>(batch.size());
    if (grads != nullptr) {
        grads->at(0).value[0] = static_cast<float>(g0 / n);
        grads->at(0).value[1] = static_cast<float>(g1 / n);
    }
    return StepResult{loss, static_cast<int64_t>(batch.size())};
}

// The adapter-files code the checkpoint calls. The loop owns no adapter format, so this toy
// writer stands in for it and records the weights it was given.
void write_toy_adapter(const ParamSet& params, const std::string& dir) {
    std::ofstream out(fs::path(dir) / "toy_adapter.txt");
    if (!out) throw std::runtime_error("cannot write the toy adapter");
    out << params[0].name << ' ' << params[0].value[0] << ' ' << params[0].value[1] << '\n';
}

bool same_bits(const ParamSet& a, const ParamSet& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].name != b[i].name || a[i].shape != b[i].shape) return false;
        if (a[i].value.size() != b[i].value.size()) return false;
        if (!a[i].value.empty() &&
            std::memcmp(a[i].value.data(), b[i].value.data(),
                        a[i].value.size() * sizeof(float)) != 0)
            return false;
    }
    return true;
}

Config base_config(const std::string& out, int64_t steps) {
    Config config;
    config.objective = "toy-mse";
    config.delta.kind = "lora";
    config.delta.rank = 8;
    config.delta.alpha = 16.0;
    config.lr = 0.05;
    config.weight_decay = 0.0;
    config.warmup_steps = 0;
    config.token_budget = 3;
    config.accumulation = 1;
    config.steps = steps;
    config.seed = 7;
    config.eval_every = 0;
    config.checkpoint_every = 50;
    config.out = out;
    config.base_id = "base-A";
    return config;
}

// One run's live objects. Members are declared in dependency order: the optimiser holds `params`
// by reference and the trainer holds all of them, so the struct must not be moved or copied.
struct Harness {
    Config config;
    ParamSet params;
    AdamWConfig adamw;
    Optimiser optimiser;
    Batcher batcher;
    std::ostringstream terminal;
    Trainer trainer;

    Harness(Config cfg, StepFn step, StepFn eval, std::vector<Batch> heldout)
        : config(std::move(cfg)),
          params(toy_initial_params()),
          adamw(r97::optimiser_config(config)),
          optimiser(params, adamw),
          batcher(toy_sequences(),
                  BatcherConfig{3, 1, static_cast<uint64_t>(config.seed)}),
          trainer(step, eval, params, optimiser, batcher,
                  TrainerOptions{config, std::move(heldout), &terminal, write_toy_adapter}) {}
};

}  // namespace

// ---------------------------------------------------------------- configuration

R97_TEST(trainer_config_file_and_flags) {
    TempDir dir;
    const fs::path file = dir / "run.conf";
    {
        std::ofstream out(file);
        out << "# a run; comments and blank lines are ignored\n"
            << "\n"
            << "model = /models/base.rad\n"
            << "data = /data/train.jsonl\n"
            << "heldout = /data/heldout.jsonl\n"
            << "out = " << dir.path << "\n"
            << "objective = toy-mse\n"
            << "delta.kind = lora\n"
            << "delta.rank = 8\n"
            << "delta.alpha = 16.5\n"
            << "delta.targets = q_proj, v_proj\n"
            << "lr = 1e-3\n"
            << "weight_decay = 0.01\n"
            << "warmup_steps = 2\n"
            << "token_budget = 4096\n"
            << "accumulation = 2\n"
            << "steps = 10\n"
            << "seed = 3\n"
            << "eval_every = 5\n"
            << "checkpoint_every = 4\n"
            << "base_id = abc\n";
        if (!out) throw std::runtime_error("cannot write the config file");
    }

    const Config parsed = r97::read_config(file.string());
    R97_CHECK(parsed.model == "/models/base.rad");
    R97_CHECK(parsed.data == "/data/train.jsonl");
    R97_CHECK(parsed.heldout == "/data/heldout.jsonl");
    R97_CHECK(parsed.out == dir.path);
    R97_CHECK(parsed.objective == "toy-mse");
    R97_CHECK(parsed.delta.kind == "lora");
    R97_CHECK(parsed.delta.rank == 8);
    R97_CHECK(parsed.delta.alpha == 16.5);
    R97_CHECK(parsed.delta.targets.size() == 2);
    R97_CHECK(parsed.delta.targets[0] == "q_proj");
    R97_CHECK(parsed.delta.targets[1] == "v_proj");
    R97_CHECK(parsed.lr == 1e-3);
    R97_CHECK(parsed.weight_decay == 0.01);
    R97_CHECK(parsed.warmup_steps == 2);
    R97_CHECK(parsed.token_budget == 4096);
    R97_CHECK(parsed.accumulation == 2);
    R97_CHECK(parsed.steps == 10);
    R97_CHECK(parsed.epochs == 0);
    R97_CHECK(parsed.seed == 3);
    R97_CHECK(parsed.eval_every == 5);
    R97_CHECK(parsed.checkpoint_every == 4);
    R97_CHECK(parsed.base_id == "abc");

    // Flags come after the file, in both spellings, and the gate key parses as a boolean.
    const Config merged = r97::load_config(
        file.string(), {"--lr=2e-3", "--steps", "20", "--delta.gate", "true", "--accumulation=1"});
    R97_CHECK(merged.lr == 2e-3);
    R97_CHECK(merged.steps == 20);
    R97_CHECK(merged.delta.gate);
    R97_CHECK(merged.accumulation == 1);
    R97_CHECK(merged.token_budget == 4096);  // untouched by the flags

    // The optimiser settings the loop assumes follow the schedule keys.
    Config schedule = base_config(dir.path, 100);
    schedule.accumulation = 4;
    schedule.warmup_steps = 5;
    schedule.weight_decay = 0.1;
    const AdamWConfig adamw = r97::optimiser_config(schedule);
    R97_CHECK(adamw.total_steps == 25);  // 100 loop steps at accumulation 4
    R97_CHECK(adamw.accum_steps == 4);
    R97_CHECK(adamw.warmup_steps == 5);
    R97_CHECK(adamw.lr == 0.05f);
    R97_CHECK(adamw.weight_decay == 0.1f);
}

R97_TEST(trainer_config_rejects_an_unknown_key) {
    bool threw = false;
    try {
        r97::parse_config("objective = toy-mse\nbogus = 1\n", "test.conf:2");
    } catch (const ConfigError& error) {
        threw = true;
        R97_CHECK(error.key() == "bogus");
        R97_CHECK(std::string(error.what()).find("bogus") != std::string::npos);
    }
    R97_CHECK(threw);

    // A flag is a key too, and an unknown one names itself.
    Config config;
    threw = false;
    try {
        r97::apply_config_flags(config, {"--no_such_key=1"});
    } catch (const ConfigError& error) {
        threw = true;
        R97_CHECK(error.key() == "no_such_key");
        R97_CHECK(std::string(error.what()).find("no_such_key") != std::string::npos);
    }
    R97_CHECK(threw);
}

R97_TEST(trainer_config_rejects_a_value_out_of_range_naming_the_key) {
    TempDir dir;

    auto refused = [&](Config config, const std::string& key) {
        try {
            r97::validate_config(config);
        } catch (const ConfigError& error) {
            R97_CHECK(error.key() == key);
            R97_CHECK(std::string(error.what()).find(key) != std::string::npos);
            return true;
        }
        return false;
    };

    Config config = base_config(dir.path, 4);

    config.lr = 0.0;
    R97_CHECK(refused(config, "lr"));
    config.lr = 0.05;

    config.delta.rank = 0;
    R97_CHECK(refused(config, "delta.rank"));
    config.delta.rank = 8;

    config.accumulation = 0;
    R97_CHECK(refused(config, "accumulation"));
    config.accumulation = 1;

    config.token_budget = 0;
    R97_CHECK(refused(config, "token_budget"));
    config.token_budget = 3;

    config.out.clear();
    R97_CHECK(refused(config, "out"));
    config.out = dir.path;

    config.objective.clear();
    R97_CHECK(refused(config, "objective"));
    config.objective = "toy-mse";

    config.steps = 0;
    R97_CHECK(refused(config, "steps"));
    config.steps = 4;

    config.epochs = 2;
    R97_CHECK(refused(config, "steps"));  // set exactly one of steps/epochs
    config.epochs = 0;

    config.checkpoint_every = -1;
    R97_CHECK(refused(config, "checkpoint_every"));

    // The untouched configuration is valid.
    config.checkpoint_every = 50;
    r97::validate_config(config);
}

// ---------------------------------------------------------------- the toy problem

R97_TEST(trainer_toy_problem_converges) {
    TempDir dir;
    Config config = base_config(dir.path, 200);
    config.eval_every = 50;
    config.checkpoint_every = 50;

    Harness harness(config, toy_step, toy_step, {toy_batch({0, 1, 2, 3})});
    R97_CHECK(harness.trainer.run() == 200);
    R97_CHECK(harness.trainer.state().step == 200);

    const ParamSet& w = harness.params;
    R97_CHECK(std::fabs(w[0].value[0] - 1.0f) < 1e-2f);
    R97_CHECK(std::fabs(w[0].value[1] + 2.0f) < 1e-2f);

    // The held-out line is the last eval line and it is small: sum of squared errors over the
    // four examples divided by the four scored examples.
    const std::vector<std::string> lines = read_lines(dir / "train.jsonl");
    double heldout = -1.0;
    for (const std::string& line : lines) {
        if (json_string(line, "kind") == "eval") heldout = json_number(line, "loss");
    }
    R97_CHECK(heldout >= 0.0);
    R97_CHECK(heldout < 1e-4);
}

// ---------------------------------------------------------------- checkpoint files and the log

R97_TEST(trainer_log_and_checkpoint_files_match_the_document) {
    TempDir dir;
    Config config = base_config(dir.path, 4);
    config.eval_every = 2;
    config.checkpoint_every = 2;

    Harness harness(config, toy_step, toy_step, {toy_batch({0, 1, 2, 3})});
    R97_CHECK(harness.trainer.run() == 4);

    const std::vector<std::string> lines = read_lines(dir / "train.jsonl");
    R97_CHECK(lines.size() == 6);  // four train lines, and an eval at step 2 and step 4

    int train_lines = 0;
    int eval_lines = 0;
    for (const std::string& line : lines) {
        R97_CHECK(!line.empty());
        R97_CHECK(line.front() == '{');
        R97_CHECK(line.back() == '}');
        const std::string kind = json_string(line, "kind");
        const int64_t step = static_cast<int64_t>(json_number(line, "step"));
        R97_CHECK(step >= 1 && step <= 4);
        if (kind == "train") {
            ++train_lines;
            R97_CHECK(json_number(line, "loss") >= 0.0);
            R97_CHECK(std::isfinite(json_number(line, "loss")));
            R97_CHECK(json_number(line, "lr") > 0.0);
            R97_CHECK(std::isfinite(json_number(line, "grad_norm")));
            R97_CHECK(json_number(line, "grad_norm") >= 0.0);
            R97_CHECK(json_number(line, "tokens") == 3.0);
            R97_CHECK(json_number(line, "tokens_per_s") > 0.0);
            R97_CHECK(json_number(line, "step_s") >= 0.0);
            R97_CHECK(json_number(line, "update_s") >= 0.0);
            R97_CHECK(json_number(line, "epoch") >= 0.0);
        } else if (kind == "eval") {
            ++eval_lines;
            R97_CHECK(std::isfinite(json_number(line, "loss")));
            R97_CHECK(json_number(line, "loss") >= 0.0);
            R97_CHECK(json_number(line, "scored") == 4.0);
            R97_CHECK(json_number(line, "tokens") == 12.0);
            R97_CHECK(json_number(line, "tokens_per_s") > 0.0);
            R97_CHECK(json_number(line, "eval_s") >= 0.0);
        } else {
            throw std::runtime_error("unexpected log kind: " + kind);
        }
    }
    R97_CHECK(train_lines == 4);
    R97_CHECK(eval_lines == 2);

    for (const int64_t step : {2, 4}) {
        const fs::path checkpoint = checkpoint_dir(dir.path, step);
        R97_CHECK(fs::is_directory(checkpoint));
        R97_CHECK(fs::is_directory(checkpoint / "adapter"));
        R97_CHECK(fs::is_regular_file(checkpoint / "adapter" / "toy_adapter.txt"));
        R97_CHECK(fs::is_regular_file(checkpoint / "trainer.state"));

        const TrainerState state =
            r97::read_trainer_state((checkpoint / "trainer.state").string());
        R97_CHECK(state.base_id == "base-A");
        R97_CHECK(state.step == step);
        R97_CHECK(state.master.size() == 1);
        R97_CHECK(!state.optimiser.empty());
    }

    // A checkpoint is the adapter directory and the state file only; the temporary directory
    // the writer used is gone.
    R97_CHECK(!fs::exists(checkpoint_dir(dir.path, 2) + ".tmp"));

    const std::string terminal = harness.terminal.str();
    R97_CHECK(terminal.find("step 1") != std::string::npos);
    R97_CHECK(terminal.find("eval") != std::string::npos);
}

R97_TEST(trainer_epoch_horizon_and_final_checkpoint_when_interval_is_off) {
    TempDir dir;
    Config config = base_config(dir.path, 0);
    config.epochs = 1;
    config.eval_every = 0;
    config.checkpoint_every = 0;  // only the checkpoint a stopping run writes

    Harness harness(config, toy_step, {}, {});
    R97_CHECK(harness.trainer.run_steps(0) == 0);
    R97_CHECK(harness.trainer.state().step == 0);

    // The batcher packs one of the four sequences per step at a budget of three tokens.
    R97_CHECK(harness.trainer.run() == 4);
    R97_CHECK(harness.trainer.state().step == 4);
    R97_CHECK(fs::exists(checkpoint_dir(dir.path, 4)));
    R97_CHECK(read_lines(dir / "train.jsonl").size() == 4);
}

R97_TEST(trainer_state_round_trip) {
    TempDir dir;
    Config config = base_config(dir.path, 3);
    config.checkpoint_every = 1;
    config.eval_every = 1;

    Harness harness(config, toy_step, {}, {});
    R97_CHECK(harness.trainer.run_steps(2) == 2);

    const TrainerState live = harness.trainer.state();
    const std::string checkpoint = harness.trainer.save_checkpoint();
    const TrainerState read =
        r97::read_trainer_state((fs::path(checkpoint) / "trainer.state").string());

    R97_CHECK(read.base_id == live.base_id);
    R97_CHECK(read.step == live.step);
    R97_CHECK(read.position.epoch == live.position.epoch);
    R97_CHECK(read.position.step == live.position.step);
    R97_CHECK(read.rng_state == live.rng_state);
    R97_CHECK(same_bits(read.master, live.master));
    R97_CHECK(read.optimiser == live.optimiser);

    // The same encoding through write_trainer_state, outside a checkpoint directory.
    const fs::path path = dir / "trainer.state";
    r97::write_trainer_state(path.string(), live);
    const TrainerState again = r97::read_trainer_state(path.string());
    R97_CHECK(again.base_id == live.base_id);
    R97_CHECK(again.step == live.step);
    R97_CHECK(again.position.epoch == live.position.epoch);
    R97_CHECK(again.position.step == live.position.step);
    R97_CHECK(again.rng_state == live.rng_state);
    R97_CHECK(same_bits(again.master, live.master));
    R97_CHECK(again.optimiser == live.optimiser);
}

// ---------------------------------------------------------------- resume

R97_TEST(trainer_resume_is_bit_identical) {
    TempDir straight;
    TempDir resumed_in_place;
    TempDir resumed_fresh;
    const int64_t n = 200;

    Config straight_config = base_config(straight.path, 2 * n);
    Config in_place_config = base_config(resumed_in_place.path, 2 * n);
    Config fresh_config = base_config(resumed_fresh.path, 2 * n);

    Harness one(straight_config, toy_step, {}, {});
    Harness two(in_place_config, toy_step, {}, {});
    Harness three(fresh_config, toy_step, {}, {});

    R97_CHECK(one.trainer.run() == 2 * n);

    // Stop at N: the interval checkpoint and the final one land on the same step.
    R97_CHECK(two.trainer.run_steps(n) == n);
    const std::string checkpoint = checkpoint_dir(resumed_in_place.path, n);
    R97_CHECK(fs::exists(checkpoint));

    // Resume in place and run the other N steps.
    two.trainer.resume(checkpoint);
    R97_CHECK(two.trainer.state().step == n);
    R97_CHECK(two.trainer.run_steps(n) == n);
    R97_CHECK(two.trainer.state().step == 2 * n);
    R97_CHECK(same_bits(one.params, two.params));

    // Resume from a fresh process and run to the horizon.
    three.trainer.resume(checkpoint);
    R97_CHECK(three.trainer.state().step == n);
    R97_CHECK(three.trainer.run() == n);
    R97_CHECK(three.trainer.state().step == 2 * n);
    R97_CHECK(same_bits(one.params, three.params));

    // The saved bookkeeping is the same too, not only the weights.
    const TrainerState a = one.trainer.state();
    const TrainerState b = three.trainer.state();
    R97_CHECK(a.rng_state == b.rng_state);
    R97_CHECK(a.position.epoch == b.position.epoch);
    R97_CHECK(a.position.step == b.position.step);

    // The resumed run continued the log rather than replacing it.
    R97_CHECK(read_lines(resumed_in_place / "train.jsonl").size() == 2 * n);
}

R97_TEST(trainer_resume_refuses_a_different_base_id) {
    TempDir dir;
    Config first = base_config(dir.path, 5);
    first.checkpoint_every = 5;

    Harness original(first, toy_step, {}, {});
    R97_CHECK(original.trainer.run() == 5);
    const std::string checkpoint = checkpoint_dir(dir.path, 5);
    R97_CHECK(fs::exists(checkpoint));

    Config second = first;
    second.base_id = "base-B";
    second.out = (dir / "other").string();
    Harness other(second, toy_step, {}, {});
    const ParamSet before = other.params;

    bool refused = false;
    try {
        other.trainer.resume(checkpoint);
    } catch (const BaseIdMismatch& error) {
        refused = true;
        R97_CHECK(error.checkpoint_base_id() == "base-A");
        R97_CHECK(error.requested_base_id() == "base-B");
        R97_CHECK(error.checkpoint_dir() == checkpoint);
        const std::string what = error.what();
        R97_CHECK(what.find("base-A") != std::string::npos);
        R97_CHECK(what.find("base-B") != std::string::npos);
    }
    R97_CHECK(refused);
    R97_CHECK(same_bits(before, other.params));  // refused before touching the weights
}

// ---------------------------------------------------------------- non-finite stops

R97_TEST(trainer_non_finite_loss_stops_and_keeps_the_last_good_checkpoint) {
    TempDir dir;
    Config config = base_config(dir.path, 10);
    config.checkpoint_every = 2;

    auto calls = std::make_shared<int>(0);
    StepFn step = [calls](const Batch& batch, const ParamSet& params, ParamSet* grads) {
        ++*calls;
        StepResult result = toy_step(batch, params, grads);
        if (*calls == 5) result.loss = std::numeric_limits<double>::quiet_NaN();
        return result;
    };

    Harness harness(config, step, {}, {});
    bool stopped = false;
    try {
        harness.trainer.run();
    } catch (const NonFiniteError& error) {
        stopped = true;
        R97_CHECK(error.step() == 5);
        R97_CHECK(error.kind().find("loss") != std::string::npos);
        R97_CHECK(!error.sequences().empty());
        R97_CHECK(error.sequences()[0].find("sequence 0") != std::string::npos);
        R97_CHECK(std::string(error.what()).find("step 5") != std::string::npos);
    }
    R97_CHECK(stopped);
    R97_CHECK(harness.trainer.state().step == 4);

    // The checkpoints after the last good step do not exist, and the final one was not written.
    R97_CHECK(fs::exists(checkpoint_dir(dir.path, 2)));
    R97_CHECK(fs::exists(checkpoint_dir(dir.path, 4)));
    R97_CHECK(!fs::exists(checkpoint_dir(dir.path, 6)));
    R97_CHECK(!fs::exists(checkpoint_dir(dir.path, 10)));

    // The live weights are the weights of the last good checkpoint.
    const TrainerState good = r97::read_trainer_state(
        (fs::path(checkpoint_dir(dir.path, 4)) / "trainer.state").string());
    R97_CHECK(same_bits(good.master, harness.params));
}

R97_TEST(trainer_non_finite_heldout_loss_keeps_the_step_it_measured) {
    TempDir dir;
    Config config = base_config(dir.path, 4);
    config.eval_every = 1;
    config.checkpoint_every = 1;

    StepFn eval = [](const Batch&, const ParamSet&, ParamSet*) {
        return StepResult{std::numeric_limits<double>::quiet_NaN(), 0};
    };

    Harness harness(config, toy_step, eval, {toy_batch({0, 1})});
    bool stopped = false;
    try {
        harness.trainer.run();
    } catch (const NonFiniteError& error) {
        stopped = true;
        R97_CHECK(error.step() == 1);
        R97_CHECK(error.kind().find("held-out") != std::string::npos);
        R97_CHECK(!error.sequences().empty());
        R97_CHECK(error.sequences()[0].find("held-out batch 0 sequence 0") !=
                  std::string::npos);
    }
    R97_CHECK(stopped);
    R97_CHECK(harness.trainer.state().step == 1);
    // The checkpoint for the good step is written before the held-out pass that throws.
    R97_CHECK(fs::exists(checkpoint_dir(dir.path, 1)));
    R97_CHECK(!fs::exists(checkpoint_dir(dir.path, 2)));
}

R97_TEST(trainer_non_finite_gradient_stops_and_keeps_the_last_good_checkpoint) {
    TempDir dir;
    Config config = base_config(dir.path, 10);
    config.checkpoint_every = 2;

    auto calls = std::make_shared<int>(0);
    StepFn step = [calls](const Batch& batch, const ParamSet& params, ParamSet* grads) {
        ++*calls;
        StepResult result = toy_step(batch, params, grads);
        if (*calls == 3 && grads != nullptr)
            grads->at(0).value[1] = std::numeric_limits<float>::infinity();
        return result;
    };

    Harness harness(config, step, {}, {});
    bool stopped = false;
    try {
        harness.trainer.run();
    } catch (const NonFiniteError& error) {
        stopped = true;
        R97_CHECK(error.step() == 3);
        R97_CHECK(error.kind().find("gradient") != std::string::npos);
        R97_CHECK(!error.sequences().empty());
        R97_CHECK(error.sequences()[0].find("sequence 0") != std::string::npos);
    }
    R97_CHECK(stopped);
    R97_CHECK(harness.trainer.state().step == 2);
    R97_CHECK(fs::exists(checkpoint_dir(dir.path, 2)));
    R97_CHECK(!fs::exists(checkpoint_dir(dir.path, 4)));
    R97_CHECK(!fs::exists(checkpoint_dir(dir.path, 10)));

    // The non-finite gradient never reached the optimiser: the checkpoint at step 2 is the
    // current state, moments included.
    const TrainerState good = r97::read_trainer_state(
        (fs::path(checkpoint_dir(dir.path, 2)) / "trainer.state").string());
    R97_CHECK(same_bits(good.master, harness.params));
    R97_CHECK(good.optimiser == harness.trainer.state().optimiser);
}
