// src/config_schema.cpp -- the configuration schema, as `r97train config` prints it.
//
// One row per key of r97/config.h, in the order docs/USAGE.md's table lists them. Each row's
// `text` reads the key out of a Config, so the default column is whatever config.h sets and the
// resolved column is whatever a run will use; nothing here keeps a second copy of a default.
// tests/cmake/config_schema_test.sh holds the rows and that table equal, cell for cell.
#include "r97/config_schema.h"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace r97 {
namespace {

// A number as short as it reads back exactly: 32 for 32.0, 1e-04 for 1e-4, 0 for 0.0.
std::string number_text(double value) {
    char buffer[64] = {};
    const std::to_chars_result result = std::to_chars(buffer, buffer + sizeof(buffer), value);
    if (result.ec != std::errc()) return "?";
    return std::string(buffer, result.ptr);
}

std::string text_text(const std::string& value) { return value.empty() ? "empty" : value; }

std::string integer_text(int64_t value) { return std::to_string(value); }

std::string boolean_text(bool value) { return value ? "true" : "false"; }

std::string list_text(const std::vector<std::string>& values) {
    std::string text;
    for (const std::string& value : values) {
        if (!text.empty()) text += ", ";
        text += value;
    }
    return text.empty() ? "empty" : text;
}

// The rows, in the document's order. The last column is the document's own sentence, markup
// stripped: it is compared there, so it says the range and the meaning in one line.
const std::vector<ConfigKey>& rows() {
    static const std::vector<ConfigKey> kRows = {
        {"model", ConfigType::String, [](const Config& c) { return text_text(c.model); },
         "the container to train against; the loop does not read it"},
        {"data", ConfigType::String, [](const Config& c) { return text_text(c.data); },
         "the training sequences; the loop does not read it"},
        {"heldout", ConfigType::String, [](const Config& c) { return text_text(c.heldout); },
         "the held-out sequences; the caller turns them into TrainerOptions::heldout"},
        {"out", ConfigType::String, [](const Config& c) { return text_text(c.out); },
         "required: the output directory for the log and checkpoints"},
        {"objective", ConfigType::String, [](const Config& c) { return text_text(c.objective); },
         "non-empty; the name is the StepFn's business, the loop only carries it"},
        {"base_id", ConfigType::String, [](const Config& c) { return text_text(c.base_id); },
         "opaque caller identifier stored in every checkpoint; resume refuses a different one"},
        {"delta.kind", ConfigType::String,
         [](const Config& c) { return text_text(c.delta.kind); },
         "non-empty; the delta's name, not read by the loop"},
        {"delta.rank", ConfigType::Integer,
         [](const Config& c) { return integer_text(c.delta.rank); },
         "> 0: the rank of the trainable factors"},
        {"delta.alpha", ConfigType::Real,
         [](const Config& c) { return number_text(c.delta.alpha); },
         "finite, > 0: the LoRA scale, applied as alpha / rank"},
        {"delta.targets", ConfigType::List,
         [](const Config& c) { return list_text(c.delta.targets); },
         "comma-separated non-empty module names; empty means the delta's default"},
        {"delta.gate", ConfigType::Boolean,
         [](const Config& c) { return boolean_text(c.delta.gate); },
         "true/false (also 1/0, yes/no, on/off): train a gate beside the sites"},
        {"lr", ConfigType::Real, [](const Config& c) { return number_text(c.lr); },
         "finite, > 0: the peak learning rate; must be set"},
        {"weight_decay", ConfigType::Real,
         [](const Config& c) { return number_text(c.weight_decay); },
         "finite, >= 0: the optimiser's weight decay"},
        {"warmup_steps", ConfigType::Integer,
         [](const Config& c) { return integer_text(c.warmup_steps); },
         ">= 0, counted in optimiser updates"},
        {"token_budget", ConfigType::Integer,
         [](const Config& c) { return integer_text(c.token_budget); },
         "> 0: the batcher's tokens per sub-batch; must be set"},
        {"accumulation", ConfigType::Integer,
         [](const Config& c) { return integer_text(c.accumulation); },
         ">= 1: micro-steps per optimiser update; must be set"},
        {"steps", ConfigType::Integer, [](const Config& c) { return integer_text(c.steps); },
         ">= 0: the loop-step horizon"},
        {"epochs", ConfigType::Integer, [](const Config& c) { return integer_text(c.epochs); },
         ">= 0: the epoch horizon"},
        {"seed", ConfigType::Integer, [](const Config& c) { return integer_text(c.seed); },
         ">= 0: the batcher shuffle and the loop's random state"},
        {"eval_every", ConfigType::Integer,
         [](const Config& c) { return integer_text(c.eval_every); },
         ">= 0: evaluate after every N steps; 0 disables evaluation"},
        {"checkpoint_every", ConfigType::Integer,
         [](const Config& c) { return integer_text(c.checkpoint_every); },
         ">= 0: checkpoint after every N steps; 0 means only when a run stops"},
    };
    return kRows;
}

}  // namespace

std::string_view config_type_name(ConfigType type) {
    switch (type) {
        case ConfigType::String: return "string";
        case ConfigType::Integer: return "integer";
        case ConfigType::Real: return "number";
        case ConfigType::Boolean: return "boolean";
        case ConfigType::List: return "list";
    }
    return "string";
}

const std::vector<ConfigKey>& config_schema() { return rows(); }

std::string config_key_text(const ConfigKey& key, const Config& config) { return key.text(config); }

std::string config_schema_text() {
    const std::vector<ConfigKey>& schema = rows();
    const Config defaults;

    size_t name_width = std::string("key").size();
    size_t type_width = std::string("type").size();
    size_t default_width = std::string("default").size();
    std::vector<std::string> texts;
    texts.reserve(schema.size());
    for (const ConfigKey& key : schema) {
        name_width = std::max(name_width, key.name.size());
        type_width = std::max(type_width, config_type_name(key.type).size());
        texts.push_back(config_key_text(key, defaults));
        default_width = std::max(default_width, texts.back().size());
    }

    // Two spaces between the columns: a reader separates them, and so does the table test.
    const auto pad = [](const std::string& text, size_t width) {
        return text + std::string(width - text.size(), ' ');
    };

    std::ostringstream out;
    out << pad("key", name_width) << "  " << pad("type", type_width) << "  "
        << pad("default", default_width) << "  " << "valid range and meaning" << "\n";
    for (size_t i = 0; i < schema.size(); ++i) {
        out << pad(std::string(schema[i].name), name_width) << "  "
            << pad(std::string(config_type_name(schema[i].type)), type_width) << "  "
            << pad(texts[i], default_width) << "  " << schema[i].range_and_meaning << "\n";
    }
    return out.str();
}

std::string config_values_text(const Config& config) {
    std::ostringstream out;
    for (const ConfigKey& key : rows()) {
        out << key.name << " = ";
        const std::string text = config_key_text(key, config);
        // An empty value is an empty string or an empty list; write it as nothing after the '='.
        out << (text == "empty" ? std::string() : text) << "\n";
    }
    return out.str();
}

}  // namespace r97
