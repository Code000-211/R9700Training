// src/adjoint/host/rope.cpp -- the rope op's f64 twin and host adjoint.
//
// The qk family's math lives in qk.h; this file is the registry row that puts the rope op on the
// tape, so a recorded graph can be walked without a device. The table is constant: only the input
// carries a cotangent.
#include "r97/adjoint.h"
#include "qk.h"

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace r97 {
namespace {

int64_t rope_param(const OpRecord& rec, const std::string& name, int64_t dflt) {
    for (const auto& p : rec.params)
        if (p.first == name && std::holds_alternative<int64_t>(p.second))
            return std::get<int64_t>(p.second);
    return dflt;
}

double rope_double(const OpRecord& rec, const std::string& name, double dflt) {
    for (const auto& p : rec.params) {
        if (p.first != name) continue;
        if (std::holds_alternative<int64_t>(p.second)) return (double)std::get<int64_t>(p.second);
        if (std::holds_alternative<std::string>(p.second)) {
            try { return std::stod(std::get<std::string>(p.second)); } catch (...) { return dflt; }
        }
    }
    return dflt;
}

adjoint::qk::RopeShape rope_shape(const OpRecord& rec, const AdValue& qkv) {
    adjoint::qk::RopeShape s;
    s.tokens = qkv.shape.empty() ? 0 : qkv.shape[0];
    s.head_dim = rope_param(rec, "head_dim", 0);
    s.rotary_dim = rope_param(rec, "rotary_dim", s.head_dim);
    s.n_head = rope_param(rec, "n_head", 0);
    s.n_head_kv = rope_param(rec, "n_head_kv", 0);
    if (s.head_dim <= 0) {
        if (qkv.shape.size() != 2 || s.n_head + 2 * s.n_head_kv <= 0)
            throw std::runtime_error("rope: cannot infer the head width");
        s.head_dim = qkv.shape[1] / (s.n_head + 2 * s.n_head_kv);
    }
    return s;
}

std::vector<int32_t> rope_positions(const AdValue& p) {
    std::vector<int32_t> out;
    out.reserve(p.value.size());
    for (double v : p.value) out.push_back((int32_t)v);
    return out;
}

void rope_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
               std::vector<AdValue>* out) {
    if (operands.size() < 2) throw std::runtime_error("rope: missing operands");
    const AdValue& qkv = operands[0];
    const AdValue& pos = operands[1];
    const adjoint::qk::RopeShape shape = rope_shape(rec, qkv);
    const std::vector<int32_t> positions = rope_positions(pos);
    const int64_t rows = rope_param(rec, "M", shape.tokens);
    const adjoint::qk::RopeTable table = adjoint::qk::rope_table_f64(
        rows, shape.rotary_dim, rope_double(rec, "theta", 10000.0),
        rope_double(rec, "scale", 1.0));
    AdValue o;
    o.dtype = qkv.dtype;
    o.shape = qkv.shape;
    o.value.assign(qkv.value.size(), 0.0);
    adjoint::qk::rope_forward_f64(qkv.value.data(), positions.data(), shape, table, o.value.data());
    out->push_back(std::move(o));
}

void rope_adjoint(const AdjointCall& call) {
    if (call.fwd.size() < 2 || call.cot.empty()) return;
    const AdValue& qkv = call.fwd[0];
    const AdValue& pos = call.fwd[1];
    const adjoint::qk::RopeShape shape = rope_shape(call.rec, qkv);
    const std::vector<int32_t> positions = rope_positions(pos);
    const int64_t rows = rope_param(call.rec, "M", shape.tokens);
    const adjoint::qk::RopeTable table = adjoint::qk::rope_table_f64(
        rows, shape.rotary_dim, rope_double(call.rec, "theta", 10000.0),
        rope_double(call.rec, "scale", 1.0));
    AdValue& d = call.dc->at(0);
    if (d.value.size() < qkv.value.size()) d.value.resize(qkv.value.size(), 0.0);
    adjoint::qk::rope_adjoint_f64(call.cot[0].value.data(), positions.data(), shape, table,
                                  d.value.data());

}

}  // namespace

R97_REGISTER_TWIN("rope", rope_twin)
R97_REGISTER_ADJOINT_F("rope", host, rope_adjoint, 1)  // positions is discrete

}  // namespace r97
