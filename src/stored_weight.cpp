// src/stored_weight.cpp -- the store behind src/stored_weight.h.
//
// A function-local static, like the adjoint and family registries, so it exists before the first
// row asks for a weight whatever order the translation units initialise in, and so a host test
// that never decodes one pays nothing.
#include "stored_weight.h"

#include <map>
#include <mutex>
#include <utility>

namespace r97 {
namespace {

struct Key {
    const void* enc = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    bool operator<(const Key& o) const {
        if (enc != o.enc) return enc < o.enc;
        if (rows != o.rows) return rows < o.rows;
        return cols < o.cols;
    }
};

struct Store {
    std::mutex mu;
    std::map<Key, StoredWeight> weights;
};

Store& store() {
    static Store s;
    return s;
}

}  // namespace

void stored_weight_put(const void* enc, StoredWeight weight) {
    if (enc == nullptr || weight.rows <= 0 || weight.cols <= 0) return;
    Store& s = store();
    std::lock_guard<std::mutex> lock(s.mu);
    s.weights[Key{enc, weight.rows, weight.cols}] = std::move(weight);
}

const StoredWeight* stored_weight_find(const void* enc, int64_t rows, int64_t cols) {
    if (enc == nullptr) return nullptr;
    Store& s = store();
    std::lock_guard<std::mutex> lock(s.mu);
    const auto it = s.weights.find(Key{enc, rows, cols});
    return it == s.weights.end() ? nullptr : &it->second;
}

int64_t stored_weight_count() {
    Store& s = store();
    std::lock_guard<std::mutex> lock(s.mu);
    return static_cast<int64_t>(s.weights.size());
}

int64_t stored_weight_bytes() {
    Store& s = store();
    std::lock_guard<std::mutex> lock(s.mu);
    int64_t bytes = 0;
    for (const auto& entry : s.weights)
        bytes += static_cast<int64_t>(entry.second.codes.size()) +
                 static_cast<int64_t>(entry.second.scale.size()) * 2;
    return bytes;
}

void stored_weight_clear() {
    Store& s = store();
    std::lock_guard<std::mutex> lock(s.mu);
    s.weights.clear();
}

}  // namespace r97
