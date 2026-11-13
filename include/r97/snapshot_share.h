// r97/snapshot_share.h -- the forward values a step reads in every block, widened once.
//
// A step's frozen operands -- a block's weights, a norm's scale, a table -- are read by every
// block and written by none of them, so every block widens the same bytes to the same f64 values.
// On the real container one block's q/k/v/o, gate_up and down weights are ~3.7 GiB of f64, and a
// 64-block step widened them 64 times. `SharedSnapshots` holds each such value once, keyed by what
// makes two reads the same read (`snapshot_key`), and answers every (position, operand) key of the
// walk's store that reads it; a caller that knows the buffer is never written (the device join
// checks the step's write set) can therefore pull and widen a frozen operand once per step.
//
// The walk itself is unchanged: it asks its SnapshotProvider for (position, operand) and copies
// what it gets, so a shared value is the same bits a per-block widen produced.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "r97/backward.h"
#include "r97/op.h"

namespace r97 {

// The identity of a forward value two reads share: the buffer, the dtype, the shape, the row
// window and the view. Two operands with the same key describe the same read of the same buffer,
// so one widen may answer both -- but only a caller that knows nothing wrote the buffer between
// them may share it.
std::string snapshot_key(const Operand& op);

// A snapshot store over values a step reads many times and writes never, over the per-block store
// that answers everything else. `find`/`insert` are keyed by `snapshot_key`; `bind` puts a shared
// value under one (position, operand) key of the walk; `at` answers from the bindings first and
// from the base provider after that.
class SharedSnapshots final : public SnapshotProvider {
public:
    SharedSnapshots() = default;
    SharedSnapshots(const SharedSnapshots&) = delete;
    SharedSnapshots& operator=(const SharedSnapshots&) = delete;

    // The value held for `key`, or null. A later read of a frozen operand asks this first: when it
    // answers, the caller neither pulls the bytes back nor widens them again.
    std::shared_ptr<const AdValue> find(const std::string& key) const;

    // Hold `value` under `key` and return it. The first caller inserts; a later one gets the value
    // already held, so two reads of one operand cannot disagree about its bits.
    std::shared_ptr<const AdValue> insert(const std::string& key, AdValue value);

    // Answer the walk's (position, operand) with `value`. A null value is refused: a binding that
    // answers nothing is a snapshot the walk would refuse later without naming this call.
    void bind(int64_t position, size_t operand, std::shared_ptr<const AdValue> value);

    // Where every key this store does not hold is answered from: one block's own snapshots. Null
    // (the default) answers only the bindings.
    void set_base(const SnapshotProvider* base) { base_ = base; }

    const AdValue* at(int64_t position, size_t operand) const override;

    // The distinct values held, and the (position, operand) keys answered from this store.
    size_t values() const { return by_key_.size(); }
    size_t size() const { return bindings_.size(); }

    // The payload of the held values, each counted once: what the step holds beside the per-block
    // store, not a per-block copy of it.
    int64_t bytes() const;

private:
    std::map<std::string, std::shared_ptr<const AdValue>> by_key_;
    std::map<std::pair<int64_t, size_t>, std::shared_ptr<const AdValue>> bindings_;
    const SnapshotProvider* base_ = nullptr;
};

}  // namespace r97
