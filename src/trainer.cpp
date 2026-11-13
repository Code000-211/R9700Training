// src/trainer.cpp -- the training loop, configuration and checkpoint state of docs/TRAINING.md.
//
// The loop is deliberately blind to the objective and the delta: a StepFn gives it a loss and a
// gradient, and everything else is scheduling. That keeps the two things a checkpoint can restore
// without knowing what they mean -- the master weights and the optimiser's bytes -- the only
// model-shaped state here.
#include "r97/trainer.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <system_error>
#include <utility>

namespace r97 {
namespace {

namespace fs = std::filesystem;

using Clock = std::chrono::steady_clock;

double seconds(Clock::duration span) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(span).count();
}

std::string trim(const std::string& text) {
    size_t begin = 0;
    size_t end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) ++begin;
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
    return text.substr(begin, end - begin);
}

std::string lower(std::string text) {
    for (char& c : text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return text;
}

[[noreturn]] void refuse(const std::string& key, const std::string& message) {
    throw ConfigError(key, message);
}

int64_t parse_integer(const std::string& key, const std::string& value,
                      const std::string& where) {
    if (value.empty()) refuse(key, "needs a value" + where);
    int64_t result = 0;
    const auto [end, error] =
        std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc() || end != value.data() + value.size())
        refuse(key, "'" + value + "' is not an integer" + where);
    return result;
}

double parse_real(const std::string& key, const std::string& value, const std::string& where) {
    if (value.empty()) refuse(key, "needs a value" + where);
    errno = 0;
    char* end = nullptr;
    const double result = std::strtod(value.c_str(), &end);
    if (end == value.c_str() || *end != '\0' || errno == ERANGE || !std::isfinite(result))
        refuse(key, "'" + value + "' is not a finite number" + where);
    return result;
}

bool parse_boolean(const std::string& key, const std::string& value, const std::string& where) {
    const std::string text = lower(trim(value));
    if (text == "true" || text == "1" || text == "yes" || text == "on") return true;
    if (text == "false" || text == "0" || text == "no" || text == "off") return false;
    refuse(key, "'" + value + "' is not a boolean (true or false)" + where);
}

std::vector<std::string> parse_targets(const std::string& key, const std::string& value,
                                       const std::string& where) {
    std::vector<std::string> targets;
    if (trim(value).empty()) return targets;
    size_t begin = 0;
    while (true) {
        const size_t comma = value.find(',', begin);
        const std::string target =
            trim(value.substr(begin, comma == std::string::npos ? std::string::npos
                                                               : comma - begin));
        if (target.empty()) refuse(key, "has an empty target" + where);
        targets.push_back(target);
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return targets;
}

void set_key(Config& config, const std::string& key, const std::string& value,
             const std::string& where) {
    if (key == "model") {
        config.model = value;
    } else if (key == "data") {
        config.data = value;
    } else if (key == "heldout") {
        config.heldout = value;
    } else if (key == "out") {
        config.out = value;
    } else if (key == "objective") {
        config.objective = value;
    } else if (key == "base_id") {
        config.base_id = value;
    } else if (key == "delta.kind") {
        config.delta.kind = value;
    } else if (key == "delta.rank") {
        config.delta.rank = parse_integer(key, value, where);
    } else if (key == "delta.alpha") {
        config.delta.alpha = parse_real(key, value, where);
    } else if (key == "delta.targets") {
        config.delta.targets = parse_targets(key, value, where);
    } else if (key == "delta.gate") {
        config.delta.gate = parse_boolean(key, value, where);
    } else if (key == "lr") {
        config.lr = parse_real(key, value, where);
    } else if (key == "weight_decay") {
        config.weight_decay = parse_real(key, value, where);
    } else if (key == "warmup_steps") {
        config.warmup_steps = parse_integer(key, value, where);
    } else if (key == "token_budget") {
        config.token_budget = parse_integer(key, value, where);
    } else if (key == "accumulation") {
        config.accumulation = parse_integer(key, value, where);
    } else if (key == "steps") {
        config.steps = parse_integer(key, value, where);
    } else if (key == "epochs") {
        config.epochs = parse_integer(key, value, where);
    } else if (key == "seed") {
        config.seed = parse_integer(key, value, where);
    } else if (key == "eval_every") {
        config.eval_every = parse_integer(key, value, where);
    } else if (key == "checkpoint_every") {
        config.checkpoint_every = parse_integer(key, value, where);
    } else {
        refuse(key, "unknown key" + where);
    }
}

std::string number(double value) {
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

// A JSON string literal. A path with a quote or a backslash must not break run.json.
std::string quoted(const std::string& text) {
    std::string out = "\"";
    for (const char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    static const char* digits = "0123456789abcdef";
                    out += "\\u00";
                    out += digits[(static_cast<unsigned char>(c) >> 4) & 0xf];
                    out += digits[static_cast<unsigned char>(c) & 0xf];
                } else {
                    out += c;
                }
        }
    }
    out += '"';
    return out;
}

uint64_t splitmix64(uint64_t state) {
    state += 0x9e3779b97f4a7c15ull;
    state = (state ^ (state >> 30)) * 0xbf58476d1ce4e5b9ull;
    state = (state ^ (state >> 27)) * 0x94d049bb133111ebull;
    return state ^ (state >> 31);
}

ParamSet zero_like(const ParamSet& params) {
    ParamSet zeros;
    zeros.reserve(params.size());
    for (const Param& param : params)
        zeros.push_back(Param{param.name, param.shape,
                              std::vector<float>(param.value.size(), 0.0f)});
    return zeros;
}

int64_t batch_tokens(const Batch& batch) {
    int64_t tokens = 0;
    for (const TokenSeq& sequence : batch) tokens += static_cast<int64_t>(sequence.ids.size());
    return tokens;
}

std::string describe_sequence(const TokenSeq& sequence) {
    std::ostringstream out;
    out << "len " << sequence.ids.size() << ", ids";
    const size_t shown = std::min<size_t>(sequence.ids.size(), 8);
    for (size_t i = 0; i < shown; ++i) out << ' ' << sequence.ids[i];
    if (sequence.ids.size() > shown) out << " ...";
    return out.str();
}

std::vector<std::string> describe_batch(const Batch& batch, const std::string& prefix) {
    std::vector<std::string> named;
    named.reserve(batch.size());
    for (size_t i = 0; i < batch.size(); ++i)
        named.push_back(prefix + "sequence " + std::to_string(i) + " (" +
                        describe_sequence(batch[i]) + ")");
    return named;
}

std::string join(const std::vector<std::string>& parts, const std::string& separator) {
    std::string text;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i != 0) text += separator;
        text += parts[i];
    }
    return text;
}

// The state file. The encoding is the one docs/CHECKPOINT.md documents: a fixed magic, then
// lengths and values in the machine's byte order. The bytes are only ever read by this file.
constexpr char kStateMagic[8] = {'R', '9', '7', 'T', 'R', 'N', '0', '1'};
constexpr uint32_t kStateVersion = 1;

class StateWriter {
public:
    explicit StateWriter(const std::string& path)
        : path_(path), out_(path, std::ios::binary | std::ios::trunc) {
        if (!out_) throw std::runtime_error("r97: trainer: cannot write '" + path + "'");
    }

    void raw(const void* data, size_t size) {
        out_.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        if (!out_) throw std::runtime_error("r97: trainer: cannot write '" + path_ + "'");
    }

    void u32(uint32_t value) { raw(&value, sizeof(value)); }
    void u64(uint64_t value) { raw(&value, sizeof(value)); }
    void i64(int64_t value) { raw(&value, sizeof(value)); }
    void f32(float value) { raw(&value, sizeof(value)); }

    void text(const std::string& value) {
        u64(static_cast<uint64_t>(value.size()));
        raw(value.data(), value.size());
    }

    void finish() {
        out_.flush();
        if (!out_) throw std::runtime_error("r97: trainer: cannot write '" + path_ + "'");
    }

private:
    std::string path_;
    std::ofstream out_;
};

class StateReader {
public:
    explicit StateReader(const std::string& path) : path_(path) {
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in) throw std::runtime_error("r97: trainer: cannot read '" + path + "'");
        const std::streamoff size = in.tellg();
        if (size < 0) throw std::runtime_error("r97: trainer: cannot read '" + path + "'");
        bytes_.resize(static_cast<size_t>(size));
        in.seekg(0);
        if (!bytes_.empty())
            in.read(reinterpret_cast<char*>(bytes_.data()), size);
        if (!in) throw std::runtime_error("r97: trainer: cannot read '" + path + "'");
    }

    size_t remaining() const { return bytes_.size() - position_; }

    void need(size_t size) const {
        if (size > remaining())
            throw std::runtime_error("r97: trainer: '" + path_ + "': truncated trainer state");
    }

    void raw(void* data, size_t size) {
        need(size);
        std::memcpy(data, bytes_.data() + position_, size);
        position_ += size;
    }

    uint32_t u32() {
        uint32_t value = 0;
        raw(&value, sizeof(value));
        return value;
    }
    uint64_t u64() {
        uint64_t value = 0;
        raw(&value, sizeof(value));
        return value;
    }
    int64_t i64() {
        int64_t value = 0;
        raw(&value, sizeof(value));
        return value;
    }
    float f32() {
        float value = 0.0f;
        raw(&value, sizeof(value));
        return value;
    }

    std::string text() {
        const uint64_t size = u64();
        if (size > bytes_.size() - position_)
            throw std::runtime_error("r97: trainer: '" + path_ + "': truncated trainer state");
        std::string value(static_cast<size_t>(size), '\0');
        raw(value.data(), value.size());
        return value;
    }

private:
    std::string path_;
    std::vector<uint8_t> bytes_;
    size_t position_ = 0;
};

}  // namespace

// ------------------------------------------------------------------ configuration

ConfigError::ConfigError(const std::string& key, const std::string& message)
    : std::runtime_error(key.empty() ? "r97: config: " + message
                                     : "r97: config: '" + key + "': " + message),
      key_(key) {}

namespace {

void parse_config_into(Config& config, const std::string& text, const std::string& source) {
    std::istringstream in(text);
    std::string line;
    int64_t line_number = 0;
    while (std::getline(in, line)) {
        ++line_number;
        const std::string stripped = trim(line);
        if (stripped.empty() || stripped.front() == '#') continue;
        const std::string where = " in " + source + ":" + std::to_string(line_number);
        const size_t equals = stripped.find('=');
        if (equals == std::string::npos) {
            const size_t head_end = stripped.find_first_of(" \t");
            refuse(trim(stripped.substr(0, head_end)), "expected 'key = value'" + where);
        }
        const std::string key = trim(stripped.substr(0, equals));
        if (key.empty()) refuse("", "missing a key before '='" + where);
        set_key(config, key, trim(stripped.substr(equals + 1)), where);
    }
}

void read_config_into(const std::string& path, Config& config) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("r97: config: cannot read '" + path + "'");
    std::ostringstream text;
    text << in.rdbuf();
    if (!in.good() && !in.eof())
        throw std::runtime_error("r97: config: cannot read '" + path + "'");
    parse_config_into(config, text.str(), path);
}

}  // namespace

Config parse_config(const std::string& text, const std::string& source) {
    Config config;
    parse_config_into(config, text, source);
    return config;
}

Config read_config(const std::string& path) {
    Config config;
    read_config_into(path, config);
    return config;
}

void apply_config_flag(Config& config, const std::string& key, const std::string& value) {
    if (key.empty()) refuse(key, "a flag needs a key");
    set_key(config, key, value, " (flag --" + key + ")");
}

void apply_config_flags(Config& config, const std::vector<std::string>& args) {
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& argument = args[i];
        if (argument.rfind("--", 0) != 0 || argument.size() == 2)
            refuse("", "unexpected argument '" + argument + "'");
        const std::string body = argument.substr(2);
        const size_t equals = body.find('=');
        const std::string key = equals == std::string::npos ? body : body.substr(0, equals);
        std::string value;
        if (equals != std::string::npos) {
            value = body.substr(equals + 1);
        } else {
            if (i + 1 >= args.size()) refuse(key, "flag --" + key + " needs a value");
            value = args[++i];
        }
        apply_config_flag(config, key, value);
    }
}

void validate_config(const Config& config) {
    auto require = [](bool okay, const std::string& key, const std::string& message) {
        if (!okay) throw ConfigError(key, message);
    };

    require(!trim(config.objective).empty(), "objective", "must not be empty");
    require(!trim(config.delta.kind).empty(), "delta.kind", "must not be empty");
    require(config.delta.rank > 0, "delta.rank",
            "must be > 0 (got " + std::to_string(config.delta.rank) + ")");
    require(std::isfinite(config.delta.alpha) && config.delta.alpha > 0.0, "delta.alpha",
            "must be a finite value > 0 (got " + number(config.delta.alpha) + ")");
    for (const std::string& target : config.delta.targets)
        require(!trim(target).empty(), "delta.targets", "has an empty target");
    require(std::isfinite(config.lr) && config.lr > 0.0, "lr",
            "must be a finite value > 0 (got " + number(config.lr) + ")");
    require(std::isfinite(config.weight_decay) && config.weight_decay >= 0.0, "weight_decay",
            "must be a finite value >= 0 (got " + number(config.weight_decay) + ")");
    require(config.warmup_steps >= 0, "warmup_steps",
            "must be >= 0 (got " + std::to_string(config.warmup_steps) + ")");
    require(config.token_budget > 0, "token_budget",
            "must be > 0 (got " + std::to_string(config.token_budget) + ")");
    require(config.accumulation >= 1, "accumulation",
            "must be >= 1 (got " + std::to_string(config.accumulation) + ")");
    require(config.steps >= 0, "steps",
            "must be >= 0 (got " + std::to_string(config.steps) + ")");
    require(config.epochs >= 0, "epochs",
            "must be >= 0 (got " + std::to_string(config.epochs) + ")");
    require((config.steps > 0) != (config.epochs > 0), "steps",
            "steps and epochs: set exactly one of them (got steps = " +
                std::to_string(config.steps) + ", epochs = " + std::to_string(config.epochs) +
                ")");
    require(config.seed >= 0, "seed",
            "must be >= 0 (got " + std::to_string(config.seed) + ")");
    require(config.eval_every >= 0, "eval_every",
            "must be >= 0 (got " + std::to_string(config.eval_every) + ")");
    require(config.checkpoint_every >= 0, "checkpoint_every",
            "must be >= 0 (got " + std::to_string(config.checkpoint_every) + ")");
    require(!trim(config.out).empty(), "out", "must not be empty");
}

Config load_config(const std::string& path, const std::vector<std::string>& flags,
                   const Config& defaults) {
    Config config = defaults;
    if (!path.empty()) read_config_into(path, config);
    apply_config_flags(config, flags);
    validate_config(config);
    return config;
}

AdamWConfig optimiser_config(const Config& config) {
    AdamWConfig adamw;
    adamw.lr = static_cast<float>(config.lr);
    adamw.weight_decay = static_cast<float>(config.weight_decay);
    adamw.warmup_steps = static_cast<int>(config.warmup_steps);
    adamw.accum_steps = static_cast<int>(config.accumulation);
    // A run measured in epochs has no known horizon, so the cosine schedule holds after warm-up.
    adamw.total_steps =
        config.steps > 0
            ? static_cast<int>((config.steps + config.accumulation - 1) / config.accumulation)
            : 0;
    return adamw;
}

// ------------------------------------------------------------------ errors

BaseIdMismatch::BaseIdMismatch(const std::string& checkpoint_base_id,
                               const std::string& requested_base_id,
                               const std::string& checkpoint_dir)
    : std::runtime_error("r97: cannot resume checkpoint '" + checkpoint_dir +
                         "': it was trained on base_id '" + checkpoint_base_id +
                         "', this run's base_id is '" + requested_base_id + "'"),
      checkpoint_base_id_(checkpoint_base_id),
      requested_base_id_(requested_base_id),
      checkpoint_dir_(checkpoint_dir) {}

NonFiniteError::NonFiniteError(int64_t step, const std::string& kind,
                               const std::vector<std::string>& sequences)
    : std::runtime_error("r97: non-finite " + kind + " at step " + std::to_string(step) +
                         "; sequences: " +
                         (sequences.empty() ? std::string("(none)")
                                            : join(sequences, "; "))),
      step_(step),
      kind_(kind),
      sequences_(sequences) {}

// ------------------------------------------------------------------ the state file

void write_trainer_state(const std::string& path, const TrainerState& state) {
    const std::string temporary = path + ".tmp";
    {
        StateWriter out(temporary);
        out.raw(kStateMagic, sizeof(kStateMagic));
        out.u32(kStateVersion);
        out.u32(0);  // reserved
        out.text(state.base_id);
        out.i64(state.step);
        out.i64(state.position.epoch);
        out.i64(state.position.step);
        out.u64(state.rng_state);
        out.i64(static_cast<int64_t>(state.master.size()));
        for (const Param& param : state.master) {
            out.text(param.name);
            out.i64(static_cast<int64_t>(param.shape.size()));
            for (const int64_t dimension : param.shape) out.i64(dimension);
            out.i64(static_cast<int64_t>(param.value.size()));
            if (!param.value.empty())
                out.raw(param.value.data(), param.value.size() * sizeof(float));
        }
        out.u64(static_cast<uint64_t>(state.optimiser.size()));
        if (!state.optimiser.empty()) out.raw(state.optimiser.data(), state.optimiser.size());
        out.finish();
    }

    std::error_code error;
    fs::rename(temporary, path, error);
    if (error) {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw std::runtime_error("r97: trainer: cannot replace '" + path + "': " +
                                 error.message());
    }
}

TrainerState read_trainer_state(const std::string& path) {
    StateReader in(path);
    char magic[sizeof(kStateMagic)] = {};
    in.raw(magic, sizeof(magic));
    if (std::memcmp(magic, kStateMagic, sizeof(magic)) != 0)
        throw std::runtime_error("r97: trainer: '" + path + "': not a trainer state");
    if (in.u32() != kStateVersion)
        throw std::runtime_error("r97: trainer: '" + path + "': unsupported trainer state version");
    in.u32();  // reserved

    TrainerState state;
    state.base_id = in.text();
    state.step = in.i64();
    state.position.epoch = in.i64();
    state.position.step = in.i64();
    state.rng_state = in.u64();

    if (state.step < 0 || state.position.epoch < 0 || state.position.step < 0)
        throw std::runtime_error("r97: trainer: '" + path + "': negative step or position");

    const int64_t count = in.i64();
    // Each parameter needs at least a name length, a rank and a value count, so the count
    // cannot exceed the bytes left; this also keeps a corrupt count from allocating wildly.
    if (count < 0 || static_cast<uint64_t>(count) > in.remaining() / 24)
        throw std::runtime_error("r97: trainer: '" + path + "': bad parameter count");
    state.master.reserve(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        Param param;
        param.name = in.text();
        const int64_t rank = in.i64();
        if (rank < 0 || rank > 8)
            throw std::runtime_error("r97: trainer: '" + path + "': bad parameter rank");
        int64_t elements = 1;
        for (int64_t d = 0; d < rank; ++d) {
            const int64_t dimension = in.i64();
            if (dimension < 0)
                throw std::runtime_error("r97: trainer: '" + path + "': bad parameter shape");
            if (dimension != 0 &&
                elements > std::numeric_limits<int64_t>::max() / dimension)
                throw std::runtime_error("r97: trainer: '" + path + "': parameter too large");
            elements *= dimension;
            param.shape.push_back(dimension);
        }
        const int64_t values = in.i64();
        if (values < 0 || values != elements)
            throw std::runtime_error(
                "r97: trainer: '" + path + "': parameter shape and size differ");
        param.value.resize(static_cast<size_t>(values));
        if (values != 0)
            in.raw(param.value.data(), static_cast<size_t>(values) * sizeof(float));
        state.master.push_back(std::move(param));
    }

    const uint64_t optimiser_size = in.u64();
    if (optimiser_size > in.remaining())
        throw std::runtime_error("r97: trainer: '" + path + "': truncated optimiser state");
    state.optimiser.resize(static_cast<size_t>(optimiser_size));
    if (!state.optimiser.empty()) in.raw(state.optimiser.data(), state.optimiser.size());

    return state;
}

// ------------------------------------------------------------------ the loop

struct Trainer::Impl {
    StepFn step_fn;
    StepFn eval_fn;
    ParamSet& params;
    Optimiser& optimiser;
    Batcher& batcher;
    TrainerOptions options;
    Config config;
    AdamWConfig adamw;
    std::ofstream log;
    int64_t step = 0;
    uint64_t rng_state = 0;
    int64_t last_saved_step = -1;
    int64_t last_eval_step = -1;
    bool config_written = false;

    Impl(StepFn step, StepFn eval, ParamSet& parameters, Optimiser& optimiser_,
         Batcher& batcher_, TrainerOptions options_)
        : step_fn(std::move(step)),
          eval_fn(std::move(eval)),
          params(parameters),
          optimiser(optimiser_),
          batcher(batcher_),
          options(std::move(options_)),
          config(options.config),
          adamw(optimiser_config(config)),
          rng_state(static_cast<uint64_t>(config.seed)) {
        if (!step_fn) throw std::invalid_argument("r97: trainer: the training StepFn is empty");
        if (!options.write_adapter)
            throw std::invalid_argument(
                "r97: trainer: write_adapter is required (the adapter-files code)");
        if (!options.heldout.empty() && !eval_fn)
            throw std::invalid_argument("r97: trainer: a held-out set needs an evaluation StepFn");
        if (batcher.sequences().empty())
            throw std::invalid_argument("r97: trainer: the batcher has no sequences");

        std::error_code error;
        fs::create_directories(config.out, error);
        if (error)
            throw std::runtime_error("r97: trainer: cannot create output directory '" + config.out +
                                     "': " + error.message());
        log.open((fs::path(config.out) / "train.jsonl").string(), std::ios::app);
        if (!log)
            throw std::runtime_error("r97: trainer: cannot open '" +
                                     (fs::path(config.out) / "train.jsonl").string() + "'");
    }

    bool horizon_reached() const {
        if (config.steps > 0) return step >= config.steps;
        return batcher.position().epoch >= config.epochs;
    }

    // The run's configuration, beside the log: the objective's and the delta's names as the caller
    // resolved them, the schedule the loop runs, and the base_id every checkpoint carries. Written
    // once when the loop first runs, so a refused resume -- which throws before this -- leaves the
    // configuration it refused in place.
    void write_run_config() {
        if (config_written) return;
        std::ostringstream json;
        json << "{\"model\":" << quoted(config.model) << ",\"data\":" << quoted(config.data)
             << ",\"heldout\":" << quoted(config.heldout)
             << ",\"out\":" << quoted(config.out)
             << ",\"objective\":" << quoted(config.objective)
             << ",\"delta.kind\":" << quoted(config.delta.kind)
             << ",\"delta.rank\":" << config.delta.rank
             << ",\"delta.alpha\":" << number(config.delta.alpha)
             << ",\"delta.targets\":[";
        for (size_t i = 0; i < config.delta.targets.size(); ++i) {
            if (i > 0) json << ',';
            json << quoted(config.delta.targets[i]);
        }
        json << "]"
             << ",\"delta.gate\":" << (config.delta.gate ? "true" : "false")
             << ",\"lr\":" << number(config.lr)
             << ",\"weight_decay\":" << number(config.weight_decay)
             << ",\"warmup_steps\":" << config.warmup_steps
             << ",\"token_budget\":" << config.token_budget
             << ",\"accumulation\":" << config.accumulation
             << ",\"steps\":" << config.steps << ",\"epochs\":" << config.epochs
             << ",\"seed\":" << config.seed << ",\"eval_every\":" << config.eval_every
             << ",\"checkpoint_every\":" << config.checkpoint_every
             << ",\"base_id\":" << quoted(config.base_id) << "}\n";

        const fs::path target = fs::path(config.out) / "run.json";
        const fs::path temporary = fs::path(config.out) / "run.json.tmp";
        {
            std::ofstream out(temporary.string(), std::ios::trunc);
            if (!out)
                throw std::runtime_error("r97: trainer: cannot write '" + temporary.string() +
                                         "'");
            out << json.str();
            if (!out)
                throw std::runtime_error("r97: trainer: cannot write '" + temporary.string() +
                                         "'");
        }
        std::error_code error;
        fs::rename(temporary, target, error);
        if (error) {
            std::error_code ignored;
            fs::remove(temporary, ignored);
            throw std::runtime_error("r97: trainer: cannot replace '" + target.string() +
                                     "': " + error.message());
        }
        config_written = true;
    }

    TrainerState make_state() const {
        TrainerState state;
        state.base_id = config.base_id;
        state.step = step;
        state.position = batcher.position();
        state.rng_state = rng_state;
        state.master = params;
        state.optimiser = optimiser.serialise();
        return state;
    }

    std::string save_checkpoint() {
        std::ostringstream name;
        name << "checkpoint-" << std::setw(9) << std::setfill('0') << step;
        const fs::path target = fs::path(config.out) / name.str();
        const fs::path temporary = fs::path(config.out) / (name.str() + ".tmp");

        std::error_code error;
        fs::remove_all(temporary, error);
        error.clear();
        fs::create_directories(temporary / "adapter", error);
        if (error)
            throw std::runtime_error("r97: trainer: cannot create '" + temporary.string() +
                                     "': " + error.message());
        try {
            // The adapter-files code owns the directory's contents; the state file is ours.
            options.write_adapter(params, (temporary / "adapter").string());
            write_trainer_state((temporary / "trainer.state").string(), make_state());
        } catch (...) {
            std::error_code ignored;
            fs::remove_all(temporary, ignored);
            throw;
        }

        fs::remove_all(target, error);
        error.clear();
        fs::rename(temporary, target, error);
        if (error) {
            std::error_code ignored;
            fs::remove_all(temporary, ignored);
            throw std::runtime_error("r97: trainer: cannot replace '" + target.string() +
                                     "': " + error.message());
        }
        last_saved_step = step;
        return target.string();
    }

    void log_train(int64_t epoch, double loss, int64_t scored, int64_t tokens, double lr,
                   double grad_norm, double tokens_per_s, double step_s, double update_s,
                   bool applied) {
        std::ostringstream json;
        json << "{\"kind\":\"train\",\"step\":" << step << ",\"epoch\":" << epoch
             << ",\"loss\":" << number(loss) << ",\"scored\":" << scored
             << ",\"lr\":" << number(lr) << ",\"grad_norm\":" << number(grad_norm)
             << ",\"tokens\":" << tokens << ",\"tokens_per_s\":" << number(tokens_per_s)
             << ",\"step_s\":" << number(step_s) << ",\"update_s\":" << number(update_s)
             << ",\"updated\":" << (applied ? "true" : "false") << "}\n";
        log << json.str();
        log.flush();
        if (options.terminal != nullptr) {
            const std::string horizon =
                config.steps > 0 ? std::to_string(config.steps)
                                 : "epochs " + std::to_string(config.epochs);
            *options.terminal << "step " << step << '/' << horizon << " epoch " << epoch
                              << " loss " << number(loss) << " lr " << number(lr) << " |g| "
                              << number(grad_norm) << " " << tokens << " tok "
                              << number(tokens_per_s) << " tok/s step " << number(step_s)
                              << "s update " << number(update_s) << "s\n";
        }
    }

    void log_eval(double loss, int64_t scored, int64_t tokens, double tokens_per_s,
                  double eval_s, size_t batches) {
        std::ostringstream json;
        json << "{\"kind\":\"eval\",\"step\":" << step << ",\"epoch\":"
             << batcher.position().epoch << ",\"loss\":" << number(loss)
             << ",\"scored\":" << scored << ",\"tokens\":" << tokens
             << ",\"tokens_per_s\":" << number(tokens_per_s) << ",\"eval_s\":" << number(eval_s)
             << ",\"batches\":" << batches << "}\n";
        log << json.str();
        log.flush();
        if (options.terminal != nullptr)
            *options.terminal << "eval step " << step << " loss " << number(loss) << " "
                              << scored << " scored tokens " << number(tokens_per_s) << " tok/s\n";
    }

    void evaluate() {
        if (options.heldout.empty()) return;
        const Clock::time_point started = Clock::now();
        double loss = 0.0;
        int64_t scored = 0;
        int64_t tokens = 0;
        for (size_t i = 0; i < options.heldout.size(); ++i) {
            const Batch& batch = options.heldout[i];
            const StepResult result = eval_fn(batch, params, nullptr);
            if (!std::isfinite(result.loss))
                throw NonFiniteError(step, "held-out loss",
                                     describe_batch(batch, "held-out batch " +
                                                               std::to_string(i) + " "));
            loss += result.loss;
            scored += result.scored;
            tokens += batch_tokens(batch);
        }
        const double eval_s = seconds(Clock::now() - started);
        const double mean = scored > 0 ? loss / static_cast<double>(scored) : 0.0;
        const double tokens_per_s =
            eval_s > 0.0 ? static_cast<double>(tokens) / eval_s : 0.0;
        log_eval(mean, scored, tokens, tokens_per_s, eval_s, options.heldout.size());
        last_eval_step = step;
    }

    // One micro-step: one StepFn call per device, the gradients averaged, the optimiser seeing
    // the average. False means the batcher was at an epoch's end and nothing was consumed.
    bool take_step() {
        std::vector<Batch> step_batches;
        const int64_t epoch = batcher.position().epoch;
        if (!batcher.next(&step_batches)) {
            if (batcher.steps_in_epoch(epoch) <= 0)
                throw std::runtime_error("r97: trainer: batcher epoch " +
                                         std::to_string(epoch) + " has no steps");
            batcher.seek(Batcher::Position{epoch + 1, 0});
            return false;
        }
        if (step_batches.empty())
            throw std::runtime_error("r97: trainer: the batcher returned an empty step");

        const Clock::time_point started = Clock::now();
        ParamSet grads = zero_like(params);
        double loss = 0.0;
        int64_t scored = 0;
        int64_t tokens = 0;
        const float device_scale = 1.0f / static_cast<float>(step_batches.size());

        for (size_t device = 0; device < step_batches.size(); ++device) {
            const Batch& batch = step_batches[device];
            const std::string prefix = "device " + std::to_string(device) + " ";
            ParamSet part = zero_like(params);
            const StepResult result = step_fn(batch, params, &part);
            if (!std::isfinite(result.loss))
                throw NonFiniteError(step + 1, "loss", describe_batch(batch, prefix));
            loss += result.loss;
            scored += result.scored;
            tokens += batch_tokens(batch);

            if (part.size() != params.size())
                throw std::runtime_error(
                    "r97: trainer: the StepFn's gradient has a different parameter count");
            for (size_t p = 0; p < part.size(); ++p) {
                if (part[p].name != params[p].name || part[p].shape != params[p].shape ||
                    part[p].value.size() != params[p].value.size())
                    throw std::runtime_error(
                        "r97: trainer: the StepFn's gradient does not match parameter '" +
                        params[p].name + "'");
                for (size_t i = 0; i < part[p].value.size(); ++i) {
                    if (!std::isfinite(part[p].value[i]))
                        throw NonFiniteError(step + 1, "gradient of '" + params[p].name + "'",
                                             describe_batch(batch, prefix));
                    grads[p].value[i] += part[p].value[i] * device_scale;
                }
            }
        }

        // Summing finite floats can still overflow; check the average before it reaches the
        // optimiser's moments.
        for (const Param& param : grads) {
            for (const float value : param.value) {
                if (!std::isfinite(value)) {
                    std::vector<std::string> named;
                    for (size_t device = 0; device < step_batches.size(); ++device) {
                        const std::vector<std::string> one = describe_batch(
                            step_batches[device], "device " + std::to_string(device) + " ");
                        named.insert(named.end(), one.begin(), one.end());
                    }
                    throw NonFiniteError(step + 1, "gradient of '" + param.name + "'", named);
                }
            }
        }

        const Clock::time_point computed = Clock::now();
        const float grad_norm = clip_global_norm(grads, 0.0f, optimiser.threads());
        if (!std::isfinite(grad_norm)) {
            std::vector<std::string> named;
            for (size_t device = 0; device < step_batches.size(); ++device) {
                const std::vector<std::string> one = describe_batch(
                    step_batches[device], "device " + std::to_string(device) + " ");
                named.insert(named.end(), one.begin(), one.end());
            }
            throw NonFiniteError(step + 1, "gradient norm", named);
        }

        const bool applied = optimiser.step(grads);
        const Clock::time_point updated = Clock::now();

        ++step;
        rng_state = splitmix64(rng_state);
        const double step_s = seconds(computed - started);
        const double update_s = seconds(updated - computed);
        const double elapsed = step_s + update_s;
        const double tokens_per_s =
            elapsed > 0.0 ? static_cast<double>(tokens) / elapsed : 0.0;
        const int lr_step =
            applied ? optimiser.updates_done() : optimiser.updates_done() + 1;
        const float lr = scheduled_lr(adamw, std::max(1, lr_step));
        log_train(epoch, loss, scored, tokens, static_cast<double>(lr),
                  static_cast<double>(grad_norm), tokens_per_s, step_s, update_s, applied);
        return true;
    }
};

Trainer::Trainer(StepFn step, StepFn eval, ParamSet& params, Optimiser& optimiser,
                 Batcher& batcher, TrainerOptions options) {
    validate_config(options.config);
    impl_ = std::make_unique<Impl>(std::move(step), std::move(eval), params, optimiser, batcher,
                                   std::move(options));
}

Trainer::~Trainer() = default;
Trainer::Trainer(Trainer&&) noexcept = default;
Trainer& Trainer::operator=(Trainer&&) noexcept = default;

void Trainer::resume(const std::string& checkpoint_dir) {
    Impl& trainer = *impl_;
    const fs::path state_path = fs::path(checkpoint_dir) / "trainer.state";
    const TrainerState state = read_trainer_state(state_path.string());

    // Refuse before touching the weights: an adapter trained on one base must not be continued
    // silently on another.
    if (state.base_id != trainer.config.base_id)
        throw BaseIdMismatch(state.base_id, trainer.config.base_id, checkpoint_dir);

    if (state.master.size() != trainer.params.size())
        throw std::runtime_error("r97: trainer: checkpoint '" + checkpoint_dir +
                                 "' has a different parameter count");
    for (size_t i = 0; i < state.master.size(); ++i) {
        if (state.master[i].name != trainer.params[i].name ||
            state.master[i].shape != trainer.params[i].shape ||
            state.master[i].value.size() != trainer.params[i].value.size())
            throw std::runtime_error("r97: trainer: checkpoint '" + checkpoint_dir +
                                     "' does not match parameter '" + trainer.params[i].name +
                                     "'");
    }

    trainer.params = state.master;
    trainer.optimiser.restore(state.optimiser);
    trainer.batcher.seek(state.position);
    trainer.step = state.step;
    trainer.rng_state = state.rng_state;
    trainer.last_saved_step = state.step;
    trainer.last_eval_step = -1;
}

int64_t Trainer::run() { return run_steps(std::numeric_limits<int64_t>::max()); }

int64_t Trainer::run_steps(int64_t count) {
    if (count < 0)
        throw std::invalid_argument("r97: trainer: run_steps needs a non-negative count");
    Impl& trainer = *impl_;
    trainer.write_run_config();
    int64_t taken = 0;
    while (taken < count && !trainer.horizon_reached()) {
        if (!trainer.take_step()) continue;
        ++taken;
        // Checkpoint before evaluating: a held-out pass that throws must not lose the good step
        // it was measuring.
        if (trainer.config.checkpoint_every > 0 &&
            trainer.step % trainer.config.checkpoint_every == 0)
            trainer.save_checkpoint();
        if (trainer.config.eval_every > 0 && trainer.step % trainer.config.eval_every == 0)
            trainer.evaluate();
    }
    if (taken > 0) {
        if (trainer.last_saved_step != trainer.step) trainer.save_checkpoint();
        if (trainer.config.eval_every > 0 && trainer.last_eval_step != trainer.step)
            trainer.evaluate();
    }
    return taken;
}

std::string Trainer::save_checkpoint() { return impl_->save_checkpoint(); }

TrainerState Trainer::state() const { return impl_->make_state(); }

const Config& Trainer::config() const { return impl_->config; }

}  // namespace r97
