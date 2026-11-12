// r97/trainer.h -- the training loop, and the configuration it reads.
//
// The loop knows how a step is scheduled, logged, evaluated and checkpointed; it does not know
// what a step computes. It takes a StepFn, a ParamSet, an Optimiser, a Batcher and (when there is
// a held-out set) another StepFn. It never reads the objective's or the delta's name: both live
// inside the StepFn it is given.
//
// docs/USAGE.md owns the configuration keys and their schema, docs/TRAINING.md the log line and
// the loop, docs/CHECKPOINT.md the checkpoint layout and the state file. This header only states
// the shapes.
#pragma once

#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "r97/batch.h"
#include "r97/batcher.h"
#include "r97/config.h"
#include "r97/optim.h"
#include "r97/params.h"
#include "r97/step.h"

namespace r97 {

// A configuration key that is unknown, missing a value, or outside its range. `key()` is the
// key as it was written, so a caller can point at it.
class ConfigError : public std::runtime_error {
public:
    ConfigError(const std::string& key, const std::string& message);

    const std::string& key() const noexcept { return key_; }

private:
    std::string key_;
};

// Parse `key = value` lines (blank lines and lines whose first non-space character is `#` are
// ignored); `source` names the file in an error. Throws ConfigError for an unknown key or a
// malformed value. Does not range-check the result: call validate_config() for that.
Config parse_config(const std::string& text, const std::string& source = "<config>");

// The same, from a file. Throws std::runtime_error when the file cannot be read.
Config read_config(const std::string& path);

// One flag's value, and a whole flag list ("--key=value" or "--key value"). Unknown keys throw
// ConfigError. apply_config_flags() ignores nothing: a non-flag argument is an error.
void apply_config_flag(Config& config, const std::string& key, const std::string& value);
void apply_config_flags(Config& config, const std::vector<std::string>& args);

// Every range the configuration schema documents. Throws ConfigError naming the key that is wrong.
void validate_config(const Config& config);

// Read the file (when `path` is not empty) over `defaults`, apply the flags over that, then
// validate. This is what a command line calls; `--config` itself is the caller's flag, not a
// configuration key. A command's own fallbacks go in `defaults`, so a key the file sets is never
// overwritten by the command and a flag still wins over both.
Config load_config(const std::string& path = "",
                   const std::vector<std::string>& flags = {},
                   const Config& defaults = Config{});

// The optimiser settings the loop assumes for this configuration: the learning rate, weight
// decay, warm-up, accumulation, and the cosine horizon (steps / accumulation, or no horizon when
// the run is measured in epochs). The caller builds the Optimiser with this, so the learning rate
// the loop logs is the one it applies.
AdamWConfig optimiser_config(const Config& config);

// A checkpoint was trained on one base and the caller asks to continue it on another.
class BaseIdMismatch : public std::runtime_error {
public:
    BaseIdMismatch(const std::string& checkpoint_base_id, const std::string& requested_base_id,
                   const std::string& checkpoint_dir);

    const std::string& checkpoint_base_id() const noexcept { return checkpoint_base_id_; }
    const std::string& requested_base_id() const noexcept { return requested_base_id_; }
    const std::string& checkpoint_dir() const noexcept { return checkpoint_dir_; }

private:
    std::string checkpoint_base_id_;
    std::string requested_base_id_;
    std::string checkpoint_dir_;
};

// A loss or a gradient that is not finite. `kind()` says which ("loss", "gradient of 'w'",
// "held-out loss"); `sequences()` names every sequence of the batch that produced it.
class NonFiniteError : public std::runtime_error {
public:
    NonFiniteError(int64_t step, const std::string& kind,
                   const std::vector<std::string>& sequences);

    int64_t step() const noexcept { return step_; }
    const std::string& kind() const noexcept { return kind_; }
    const std::vector<std::string>& sequences() const noexcept { return sequences_; }

private:
    int64_t step_;
    std::string kind_;
    std::vector<std::string> sequences_;
};

// What a trainer state file holds, in memory. docs/CHECKPOINT.md owns its on-disk encoding.
struct TrainerState {
    std::string base_id;                  // the caller's opaque base identifier
    int64_t step = 0;                     // completed loop steps
    Batcher::Position position;           // where the batcher's next next() reads
    uint64_t rng_state = 0;               // the loop's random state after this step
    ParamSet master;                      // the master weights, f32, updated in place
    std::vector<uint8_t> optimiser;       // Optimiser::serialise() at this step
};

void write_trainer_state(const std::string& path, const TrainerState& state);
TrainerState read_trainer_state(const std::string& path);

// What checkpointing writes into `<checkpoint>/adapter`. The loop owns no adapter format: the
// adapter-files code supplies this and writes the directory the serving radiance reads. `dir`
// already exists when the writer is called.
using AdapterWriter = std::function<void(const ParamSet& params, const std::string& dir)>;

struct TrainerOptions {
    Config config;
    std::vector<Batch> heldout;        // evaluation batches; empty means no held-out loss
    std::ostream* terminal = nullptr;  // nullptr writes no terminal lines; the JSON file is
                                       // always written
    AdapterWriter write_adapter;      // required: the adapter-files code
};

// The loop over one StepFn. The caller owns the objects passed by reference and keeps them alive.
class Trainer {
public:
    Trainer(StepFn step, StepFn eval, ParamSet& params, Optimiser& optimiser, Batcher& batcher,
            TrainerOptions options);
    ~Trainer();

    Trainer(const Trainer&) = delete;
    Trainer& operator=(const Trainer&) = delete;
    Trainer(Trainer&&) noexcept;
    Trainer& operator=(Trainer&&) noexcept;

    // Continue from a checkpoint directory: refuse a different base_id, then restore the master
    // weights, optimiser state, step, batcher position and random state, in that order.
    void resume(const std::string& checkpoint_dir);

    // Run to the configured horizon (`steps` or `epochs`). Returns the steps taken.
    int64_t run();

    // Run at most `count` more steps, and stop at the horizon. `count >= 0`. Returns the steps
    // taken. This is how a caller stops a run early and how a resume test halves one.
    int64_t run_steps(int64_t count);

    // Write `<out>/checkpoint-<step>` now: the adapter directory and trainer.state. Returns the
    // directory. Writes through a temporary directory, so a failure leaves no half checkpoint.
    std::string save_checkpoint();

    // The state a checkpoint would hold now.
    TrainerState state() const;

    const Config& config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace r97
