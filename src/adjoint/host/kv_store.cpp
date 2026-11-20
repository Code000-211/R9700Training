// src/adjoint/host/kv_store.cpp -- kv_store's f64 twin and host adjoint.
//
// The qk family's math lives in qk.h. This is the registry row that puts kv_store on the tape:
// the pool write is the op's output, and the k/v cotangents come back from the pool cotangent.
#include "r97/adjoint.h"
#include "qk.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace r97 {
namespace {

int64_t kv_param(const OpRecord& rec, const std::string& name, int64_t dflt) {
    for (const auto& p : rec.params)
        if (p.first == name && std::holds_alternative<int64_t>(p.second))
            return std::get<int64_t>(p.second);
    return dflt;
}

adjoint::qk::KvStoreShape kv_shape(const OpRecord& rec, const std::vector<AdValue>& reads,
                                   const AdValue& pool) {
    adjoint::qk::KvStoreShape s;
    s.tokens = reads.empty() || reads[0].shape.empty() ? 0 : reads[0].shape[0];
    s.kv_heads = kv_param(rec, "n_head_kv", 0);
    s.head_dim = kv_param(rec, "head_dim", 0);
    s.block_size = kv_param(rec, "block_size", 16);
    s.n_blocks = pool.shape.empty() ? 0 : pool.shape[0];
    if (s.kv_heads <= 0 || s.head_dim <= 0 || s.n_blocks <= 0)
        throw std::runtime_error("kv_store: incomplete pool geometry");
    return s;
}

const Operand* first_write(const OpRecord& rec) {
    for (const Operand& o : rec.operands)
        if (o.written) return &o;
    return nullptr;
}

std::vector<int32_t> slots_of(const AdValue& slot) {
    std::vector<int32_t> out;
    out.reserve(slot.value.size());
    for (double v : slot.value) out.push_back((int32_t)v);
    return out;
}

// A record's slot table names one pool row a token. A table that is all one value cannot be one:
// the forward writes every token's k/v into the same row, so the backward's scatter would fold
// that row's cotangent once per token into every token's k and v -- an inflation of exactly the
// token count, silent because the arithmetic is finite. The engine never issues one; a walk whose
// snapshot of the table is zero, short or stale does. Refuse by name and leave the numbers to the
// caller, as the rows do for every other shape a record cannot mean.
void require_distinct_slots(const std::vector<int32_t>& slots, const OpRecord& rec) {
    if (slots.size() < 2) return;
    const int32_t first = slots[0];
    for (size_t t = 1; t < slots.size(); ++t)
        if (slots[t] != first) return;
    int64_t negative = 0;
    for (int32_t s : slots)
        if (s < 0) ++negative;
    if (negative == static_cast<int64_t>(slots.size())) return;  // a padded record: all -1
    throw std::runtime_error("kv_store: op at position " + std::to_string(rec.position) +
                             ": the slot table is constant (" + std::to_string(first) + " for all " +
                             std::to_string(slots.size()) +
                             " tokens), so the store would write every token into one pool row -- "
                             "the snapshot of the slot operand ('" + rec.operands[2].buffer +
                             "') is not the forward's table");
}

void kv_store_twin(const OpRecord& rec, const std::vector<AdValue>& reads,
                   std::vector<AdValue>* out) {
    if (reads.size() < 3) throw std::runtime_error("kv_store: missing operands");
    const Operand* w = first_write(rec);
    if (!w) throw std::runtime_error("kv_store: no pool write");
    AdValue pool;
    pool.dtype = w->dtype;
    pool.shape = w->shape;
    int64_t n = 1;
    for (int64_t d : pool.shape) n *= d;
    pool.value.assign((size_t)n, 0.0);
    // An in-place pool is also a read: the record marks the write over the pool it reads, and the
    // read list carries the pool's old value last. The store overwrites the rows its slots name
    // and leaves the rest, so the post-op slot value starts from the old pool.
    if (w->in_place && reads.size() == 4 && reads[3].shape == w->shape)
        pool.value = reads[3].value;
    const adjoint::qk::KvStoreShape shape = kv_shape(rec, reads, pool);
    const std::vector<int32_t> slot = slots_of(reads[2]);
    require_distinct_slots(slot, rec);
    adjoint::qk::kv_store_forward_f64(reads[0].value.data(), reads[1].value.data(), slot.data(),
                                      shape, pool.value.data());
    out->push_back(std::move(pool));
}

void kv_store_adjoint(const AdjointCall& call) {
    if (call.fwd.size() < 4 || call.cot.empty()) return;
    const AdValue& k = call.fwd[0];
    const AdValue& v = call.fwd[1];
    const AdValue& slot = call.fwd[2];
    const AdValue& pool = call.fwd[3];
    const adjoint::qk::KvStoreShape shape = kv_shape(call.rec, {k, v, slot}, pool);
    const std::vector<int32_t> slots = slots_of(slot);
    require_distinct_slots(slots, call.rec);
    AdValue& dk = call.dc->at(0);
    AdValue& dv = call.dc->at(1);
    dk.dtype = k.dtype;
    dk.shape = k.shape;
    dk.value.assign(k.value.size(), 0.0);
    dv.dtype = v.dtype;
    dv.shape = v.shape;
    dv.value.assign(v.value.size(), 0.0);
    adjoint::qk::kv_store_adjoint_f64(call.cot[0].value.data(), slots.data(), shape,
                                      dk.value.data(), dv.value.data());

    // The in-place pool's own slot: the store overwrites the rows its slots name, so those rows
    // of the old pool are dead and their cotangent is the k's and v's -- dk and dv above. Only
    // the pool the write leaves alone carries the pool's cotangent on. The rows the write
    // covers, by the same (block, offset) arithmetic the forward scatters with.
    for (size_t j = 0; j < call.rec.operands.size(); ++j)
        if (call.rec.operands[j].written && call.rec.operands[j].in_place) {
            AdValue& dp = call.dc->at(j);
            if (dp.value.size() != call.cot[0].value.size()) continue;
            const int64_t pool_rows = shape.n_blocks * shape.block_size;
            std::vector<char> covered((size_t)pool_rows, 0);
            for (int32_t s : slots)
                if (s >= 0 && s < pool_rows) covered[(size_t)s] = 1;
            for (int64_t row = 0; row < pool_rows; ++row) {
                if (covered[(size_t)row]) continue;
                const int64_t block = row / shape.block_size;
                const int64_t off = row % shape.block_size;
                for (int64_t h = 0; h < shape.kv_heads; ++h) {
                    const int64_t base =
                        ((block * shape.kv_heads + h) * shape.block_size + off) * 2 * shape.head_dim;
                    for (int64_t d = 0; d < shape.head_dim; ++d) {
                        dp.value[(size_t)(base + d)] +=
                            call.cot[0].value[(size_t)(base + d)];
                        dp.value[(size_t)(base + shape.head_dim + d)] +=
                            call.cot[0].value[(size_t)(base + shape.head_dim + d)];
                    }
                }
            }
        }
}

}  // namespace

R97_REGISTER_TWIN("kv_store", kv_store_twin)
R97_REGISTER_ADJOINT_FF("kv_store", host, kv_store_adjoint, fills_in_place_slots, 2)  // slots is discrete

}  // namespace r97
