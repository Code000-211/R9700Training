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
    const adjoint::qk::KvStoreShape shape = kv_shape(rec, reads, pool);
    const std::vector<int32_t> slot = slots_of(reads[2]);
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
}

}  // namespace

R97_REGISTER_TWIN("kv_store", kv_store_twin)
R97_REGISTER_ADJOINT("kv_store", host, kv_store_adjoint)

}  // namespace r97
