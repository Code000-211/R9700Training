// r97/config_schema.h -- the configuration schema: one row per key of r97/config.h.
//
// The schema is the table `docs/USAGE.md` prints and the one `r97train config` prints, and
// tests/cmake/config_schema_test.sh keeps the two equal: the key, its type, its default and the
// sentence that carries its valid range and its meaning.
//
// A row does not restate a default. It shows the Config a default constructor makes, so config.h
// stays the one place a default lives and a default that moves there moves here too.
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "r97/config.h"

namespace r97 {

// How a key's value is written in a config file. `List` is comma-separated.
enum class ConfigType { String, Integer, Real, Boolean, List };

// The schema's word for a type, the one the document's `type` column uses: `string`, `integer`,
// `number`, `boolean`, `list`.
std::string_view config_type_name(ConfigType type);

struct ConfigKey {
    std::string_view name;                // as a config file and a `--flag` write it
    ConfigType type;                      // what config_type_name() prints
    std::string (*text)(const Config&);   // this key's value, as a config file writes it
    std::string_view range_and_meaning;   // its valid range and what reads it
};

// Every key, in the order docs/USAGE.md's table lists them.
const std::vector<ConfigKey>& config_schema();

// One key's value in `config`, as a config file would write it.
std::string config_key_text(const ConfigKey& key, const Config& config);

// The schema as `r97train config` prints it with no argument: a header, then one line per key.
std::string config_schema_text();

// `key = value`, one line per key in schema order: a configuration the parser reads back to the
// same Config. What `r97train config --config <file>` prints.
std::string config_values_text(const Config& config);

}  // namespace r97
