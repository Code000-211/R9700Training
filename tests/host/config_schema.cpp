// tests/host/config_schema.cpp -- the configuration schema against the parser and config.h.
//
// The schema is what `r97train config` prints and what docs/USAGE.md prints; the document side is
// tests/cmake/config_schema_test.sh's, because it needs the binary and the doc. Here the schema is
// checked against the three things the binary cannot check itself: every row's default is the value
// config.h sets (a default lives there and nowhere else), every row names a key the parser accepts
// and sets the field the key is for, and the rows are as many as config.h has members (a member
// added to Config without its row fails here, as a row without its member does).
// Label: host.
#include "../support/test.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "r97/config.h"
#include "r97/config_schema.h"
#include "r97/trainer.h"

using namespace r97;

namespace {

bool same(const Config& a, const Config& b) {
    return a.model == b.model && a.data == b.data && a.heldout == b.heldout && a.out == b.out &&
           a.objective == b.objective && a.base_id == b.base_id && a.delta.kind == b.delta.kind &&
           a.delta.rank == b.delta.rank && a.delta.alpha == b.delta.alpha &&
           a.delta.targets == b.delta.targets && a.delta.gate == b.delta.gate && a.lr == b.lr &&
           a.weight_decay == b.weight_decay && a.warmup_steps == b.warmup_steps &&
           a.token_budget == b.token_budget && a.accumulation == b.accumulation &&
           a.steps == b.steps && a.epochs == b.epochs && a.seed == b.seed &&
           a.eval_every == b.eval_every && a.checkpoint_every == b.checkpoint_every;
}

// The text a config file writes for a key whose shown default is `empty`: nothing after the '='.
std::string file_value(const std::string& text) { return text == "empty" ? std::string() : text; }

// A value an initialiser takes for any member: it converts to whatever the member's type is, so an
// aggregate is probed without naming its members.
struct AnyValue {
    template <class T>
    operator T() const {
        return T{};
    }
};

template <class T, std::size_t... I>
consteval bool brace_fits(std::index_sequence<I...>) {
    return requires { T{((void)I, AnyValue{})...}; };
}

// The number of members T's brace initialisation takes: one initialiser per member fits and one more
// is "too many initializers", so the count is one past the last N that fits.
template <class T, std::size_t N = 0>
consteval std::size_t member_count() {
    if constexpr (N < 64 && brace_fits<T>(std::make_index_sequence<N>{})) {
        return member_count<T, N + 1>();
    } else {
        return N - 1;
    }
}

}  // namespace

R97_TEST(config_schema_defaults_are_config_h) {
    const Config defaults;
    for (const ConfigKey& key : config_schema()) {
        // The row's type is one of the schema's five names: the document's column is that word.
        const std::string type(config_type_name(key.type));
        R97_CHECK(type == "string" || type == "integer" || type == "number" ||
                  type == "boolean" || type == "list");
        R97_CHECK(!key.range_and_meaning.empty());

        // Writing the row's own default back through the parser gives the default Config: the
        // shown value is the one config.h sets, for every key.
        const std::string text =
            std::string(key.name) + " = " + file_value(config_key_text(key, defaults));
        const Config parsed = parse_config(text);
        if (!same(parsed, defaults))
            throw std::runtime_error("default drifted for " + std::string(key.name));
    }
}

R97_TEST(config_schema_keys_set_their_field) {
    // One value per key that is not its default, and a check that it landed where the key says.
    struct Case {
        const char* key;
        const char* value;
        bool (*landed)(const Config&);
    };
    const Case cases[] = {
        {"model", "/srv/models/m.rad",
         [](const Config& c) { return c.model == "/srv/models/m.rad"; }},
        {"data", "/data/train.jsonl",
         [](const Config& c) { return c.data == "/data/train.jsonl"; }},
        {"heldout", "/data/heldout.jsonl",
         [](const Config& c) { return c.heldout == "/data/heldout.jsonl"; }},
        {"out", "/runs/a", [](const Config& c) { return c.out == "/runs/a"; }},
        {"objective", "kld", [](const Config& c) { return c.objective == "kld"; }},
        {"base_id", "qwen3.8", [](const Config& c) { return c.base_id == "qwen3.8"; }},
        {"delta.kind", "bias", [](const Config& c) { return c.delta.kind == "bias"; }},
        {"delta.rank", "4", [](const Config& c) { return c.delta.rank == 4; }},
        {"delta.alpha", "8.5", [](const Config& c) { return c.delta.alpha == 8.5; }},
        {"delta.targets", "q_proj, v_proj",
         [](const Config& c) {
             return c.delta.targets == std::vector<std::string>{"q_proj", "v_proj"};
         }},
        {"delta.gate", "true", [](const Config& c) { return c.delta.gate; }},
        {"lr", "2.5e-5", [](const Config& c) { return c.lr == 2.5e-5; }},
        {"weight_decay", "0.25", [](const Config& c) { return c.weight_decay == 0.25; }},
        {"warmup_steps", "3", [](const Config& c) { return c.warmup_steps == 3; }},
        {"token_budget", "1024", [](const Config& c) { return c.token_budget == 1024; }},
        {"accumulation", "4", [](const Config& c) { return c.accumulation == 4; }},
        {"steps", "11", [](const Config& c) { return c.steps == 11; }},
        {"epochs", "2", [](const Config& c) { return c.epochs == 2; }},
        {"seed", "9", [](const Config& c) { return c.seed == 9; }},
        {"eval_every", "5", [](const Config& c) { return c.eval_every == 5; }},
        {"checkpoint_every", "6", [](const Config& c) { return c.checkpoint_every == 6; }},
    };

    // Every key the schema lists is in the table, and every key in the table is in the schema: a
    // new key must be given its row and its value here, which is where a key that parses but has
    // no schema row would be noticed.
    R97_CHECK(std::size(cases) == config_schema().size());

    // config.h holds one member per key (`delta` counts as its five, in DeltaConfig), so the two
    // counts together are the number of keys. A member added to either struct without its schema
    // row, its case in the table above, and its row in docs/USAGE.md fails here rather than
    // passing every test; a member that is removed or renamed no longer compiles in `same()`.
    R97_CHECK(std::is_aggregate_v<Config>);
    R97_CHECK(std::is_aggregate_v<DeltaConfig>);
    R97_CHECK(member_count<Config>() + member_count<DeltaConfig>() - 1 == std::size(cases));

    for (const Case& c : cases) {
        const bool listed = std::any_of(config_schema().begin(), config_schema().end(),
                                        [&c](const ConfigKey& k) { return k.name == c.key; });
        if (!listed) throw std::runtime_error(std::string("no schema row for ") + c.key);

        const Config config = parse_config(std::string(c.key) + " = " + c.value);
        if (!c.landed(config)) throw std::runtime_error(std::string("key did not land: ") + c.key);
    }
}

R97_TEST(config_values_text_round_trips) {
    Config config;
    config.model = "/srv/models/m.rad";
    config.data = "/data/train.jsonl";
    config.heldout = "/data/heldout.jsonl";
    config.out = "/runs/a";
    config.objective = "ce";
    config.base_id = "qwen3.8";
    config.delta.kind = "lora";
    config.delta.rank = 8;
    config.delta.alpha = 16.0;
    config.delta.targets = {"q_proj", "v_proj"};
    config.delta.gate = true;
    config.lr = 1e-4;
    config.weight_decay = 0.01;
    config.warmup_steps = 20;
    config.token_budget = 4096;
    config.accumulation = 2;
    config.steps = 1000;
    config.epochs = 0;
    config.seed = 7;
    config.eval_every = 100;
    config.checkpoint_every = 100;

    // What `r97train config --config` prints is a config file: the parser reads it back unchanged.
    const Config back = parse_config(config_values_text(config));
    R97_CHECK(same(back, config));

    // The schema's own text names every key with its type, default and sentence.
    const std::string schema = config_schema_text();
    for (const ConfigKey& key : config_schema()) {
        R97_CHECK(schema.find(std::string(key.name) + " ") != std::string::npos);
        R97_CHECK(schema.find(std::string(config_type_name(key.type))) != std::string::npos);
        R97_CHECK(schema.find(std::string(key.range_and_meaning)) != std::string::npos);
    }
}
