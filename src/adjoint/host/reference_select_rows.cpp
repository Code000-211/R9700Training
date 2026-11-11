// A private op of the reference step: y[r, :] = x[idx[r], :].
//
// The forward writes one logits row per token (the BOS row included); the boundary has the
// batch's row space, one row per non-BOS token, so this is where the two row spaces meet.

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

AdValue make_value(std::string dtype, std::vector<int64_t> shape, std::vector<double> value) {
    AdValue v;
    v.dtype = std::move(dtype);
    v.shape = std::move(shape);
    v.value = std::move(value);
    if (int64_t(v.value.size()) != numel(v.shape))
        throw std::runtime_error("ref_select_rows: value does not fill its shape");
    return v;
}

void ref_select_rows_twin(const OpRecord&, const std::vector<AdValue>& operands,
                          std::vector<AdValue>* out) {
    const AdValue& x = operands[0];
    const AdValue& idx = operands[1];
    if (x.shape.size() != 2 || idx.value.empty())
        throw std::runtime_error("ref_select_rows: bad operands");
    const int64_t M = (int64_t)idx.value.size();
    const int64_t V = x.shape[1];
    AdValue y = make_value(x.dtype, {M, V}, std::vector<double>((size_t)(M * V), 0.0));
    for (int64_t r = 0; r < M; ++r) {
        const int64_t src = (int64_t)idx.value[(size_t)r];
        if (src < 0 || src >= x.shape[0])
            throw std::runtime_error("ref_select_rows: row out of range");
        for (int64_t c = 0; c < V; ++c)
            y.value[(size_t)(r * V + c)] = x.value[(size_t)(src * V + c)];
    }
    out->push_back(std::move(y));
}

void ref_select_rows_adjoint(const AdjointCall& call) {
    const AdValue& idx = call.fwd[1];
    const AdValue& dy = call.cot[0];
    AdValue& dx = call.dc->at(0);
    const int64_t V = dy.shape[1];
    for (int64_t r = 0; r < (int64_t)idx.value.size(); ++r) {
        const int64_t dst = (int64_t)idx.value[(size_t)r];
        for (int64_t c = 0; c < V; ++c)
            dx.value[(size_t)(dst * V + c)] += dy.value[(size_t)(r * V + c)];
    }
}

}  // namespace

R97_REGISTER_TWIN("ref_select_rows", ref_select_rows_twin)
R97_REGISTER_ADJOINT_F("ref_select_rows", host, ref_select_rows_adjoint, 1)

}  // namespace r97
