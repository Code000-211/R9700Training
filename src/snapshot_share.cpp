// src/snapshot_share.cpp -- the values a step widens once and every block reads. See
// include/r97/snapshot_share.h.
#include "r97/snapshot_share.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace r97 {
namespace {

int64_t payload_bytes(const AdValue& v) {
    return static_cast<int64_t>(v.value.size()) * static_cast<int64_t>(sizeof(double));
}

}  // namespace

std::string snapshot_key(const Operand& op) {
    // The fields are separated by a byte no buffer name, dtype or number holds, so two different
    // reads cannot build one key by running their fields together.
    constexpr char kField = '\x1f';
    std::string key = op.buffer;
    key += kField;
    key += op.dtype;
    key += kField;
    for (int64_t d : op.shape) {
        key += std::to_string(d);
        key += ',';
    }
    key += kField;
    key += std::to_string(op.row_lo);
    key += ',';
    key += std::to_string(op.rows);
    key += kField;
    for (size_t d = 0; d < op.view_shape.size(); ++d) {
        key += std::to_string(op.view_shape[d]);
        key += ':';
        key += std::to_string(d < op.view_stride.size() ? op.view_stride[d] : 0);
        key += ',';
    }
    key += kField;
    key += std::to_string(op.view_off);
    return key;
}

std::shared_ptr<const AdValue> SharedSnapshots::find(const std::string& key) const {
    const auto it = by_key_.find(key);
    return it == by_key_.end() ? nullptr : it->second;
}

std::shared_ptr<const AdValue> SharedSnapshots::insert(const std::string& key, AdValue value) {
    const auto it = by_key_.find(key);
    if (it != by_key_.end()) return it->second;
    std::shared_ptr<const AdValue> held = std::make_shared<const AdValue>(std::move(value));
    by_key_.emplace(key, held);
    return held;
}

void SharedSnapshots::bind(int64_t position, size_t operand,
                           std::shared_ptr<const AdValue> value) {
    if (!value)
        throw std::invalid_argument("r97: snapshot share: binding a null value at position " +
                                    std::to_string(position) + ", operand " +
                                    std::to_string(operand));
    bindings_[{position, operand}] = std::move(value);
}

const AdValue* SharedSnapshots::at(int64_t position, size_t operand) const {
    const auto it = bindings_.find({position, operand});
    if (it != bindings_.end()) return it->second.get();
    return base_ ? base_->at(position, operand) : nullptr;
}

int64_t SharedSnapshots::bytes() const {
    int64_t total = 0;
    for (const auto& entry : by_key_) total += payload_bytes(*entry.second);
    return total;
}

}  // namespace r97
