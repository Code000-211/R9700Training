// A private op of the reference step: y[:, off:off+W] += inc.
//
// The LoRA expand of a fused weight produces its own module's [M, W] block; this is how it lands
// in the fused output without widening the module's B or losing the column offset.

#include "r97/adjoint.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace r97 {
namespace {

int64_t numel(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 0;
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

int64_t param_int(const OpRecord& rec, const char* key, int64_t fallback) {
    for (const OpParam& p : rec.params)
        if (p.first == key) {
            if (const int64_t* v = std::get_if<int64_t>(&p.second)) return *v;
        }
    return fallback;
}

void ref_add_cols_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                       std::vector<AdValue>* out) {
    const AdValue& y = operands[0];
    const AdValue& inc = operands[1];
    const int64_t off = param_int(rec, "off", 0);
    if (y.shape.empty() || inc.shape.size() != 2 || inc.shape[0] != y.shape[0])
        throw std::runtime_error("ref_add_cols: shape mismatch");
    const int64_t row = numel(y.shape) / y.shape[0];
    if (off + inc.shape[1] > row)
        throw std::runtime_error("ref_add_cols: increment past the row");
    AdValue o = y;
    for (int64_t m = 0; m < inc.shape[0]; ++m)
        for (int64_t c = 0; c < inc.shape[1]; ++c)
            o.value[(size_t)(m * row + off + c)] += inc.value[(size_t)(m * inc.shape[1] + c)];
    out->push_back(std::move(o));
}

void ref_add_cols_adjoint(const AdjointCall& call) {
    const AdValue& y = call.fwd[0];
    const AdValue& inc = call.fwd[1];
    const AdValue& dy = call.cot[0];
    const int64_t off = param_int(call.rec, "off", 0);
    AdValue& dy_in = call.dc->at(0);
    AdValue& dinc = call.dc->at(1);
    for (size_t i = 0; i < dy_in.value.size(); ++i) dy_in.value[i] += dy.value[i];
    const int64_t row = numel(y.shape) / y.shape[0];
    for (int64_t m = 0; m < inc.shape[0]; ++m)
        for (int64_t c = 0; c < inc.shape[1]; ++c)
            dinc.value[(size_t)(m * inc.shape[1] + c)] +=
                dy.value[(size_t)(m * row + off + c)];
}

}  // namespace

R97_REGISTER_TWIN("ref_add_cols", ref_add_cols_twin)
R97_REGISTER_ADJOINT("ref_add_cols", host, ref_add_cols_adjoint)

}  // namespace r97
