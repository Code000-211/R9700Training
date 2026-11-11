// src/delta_registry.cpp -- the trainable kinds by name, behind r97/delta_registry.h.
//
// A kind registers from its own file (a static object, run at load), and make_delta builds the
// kind a config names. Two registrations of one kind are a programming error rather than a
// silent choice between them: which factory won would depend on the link order.
#include "r97/delta_registry.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>

namespace r97 {
namespace {

struct Table {
    std::map<std::string, DeltaFactory> factories;

    // Deliberately never destroyed: kinds register from other files' static initialisers and a
    // test reads the table from an atexit handler, so which of the two is constructed first -- and
    // therefore which is destroyed first -- is not something this file may depend on.
    static Table& get() {
        static Table* t = new Table;
        return *t;
    }
};

std::string kinds_text() {
    std::string text;
    for (const std::string& kind : delta_kinds()) {
        if (!text.empty()) text += ", ";
        text += "'" + kind + "'";
    }
    return text.empty() ? std::string("none") : text;
}

}  // namespace

void delta_register(const char* kind, DeltaFactory factory) {
    if (!kind || !*kind) throw std::invalid_argument("r97: delta_register: an empty kind");
    if (!factory)
        throw std::invalid_argument(std::string("r97: delta_register: '") + kind + "' has no factory");
    auto [it, inserted] = Table::get().factories.emplace(kind, factory);
    if (!inserted) {
        std::fprintf(stderr, "r97: delta kind '%s' is registered twice\n", kind);
        std::abort();
    }
}

std::vector<std::string> delta_kinds() {
    std::vector<std::string> kinds;
    kinds.reserve(Table::get().factories.size());
    for (const auto& kv : Table::get().factories) kinds.push_back(kv.first);  // map order: sorted
    return kinds;
}

std::shared_ptr<Delta> make_delta(const std::string& kind) {
    const auto& table = Table::get().factories;
    const auto it = table.find(kind);
    if (it == table.end())
        throw std::invalid_argument("r97: make_delta: unknown kind '" + kind +
                                    "'; registered kinds: " + kinds_text());
    return it->second();
}

}  // namespace r97
