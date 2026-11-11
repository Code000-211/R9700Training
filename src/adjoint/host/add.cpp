// src/adjoint/host/add.cpp -- the worked example: y = (a + b) / div.
//
// Schema: reads a and b, writes y. Broadcasting follows the schema: an operand
// carrying exactly one row's width against a taller output broadcasts along
// rows, so a per-channel bias b reads the sum of the cotangent over its rows.
//
// Optional parameter `div` (an integer or a decimal string, default 1): the
// adjoint carries the 1/div through both routes. Add has no trainable leaves, so
// the gradient set stays empty.
#include "r97/adjoint.h"

#include <cstdlib>
#include <string>
#include <utility>
#include <variant>

namespace r97 {
namespace {

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

/* A numeric parameter as f64. A parameter the record does not carry, or one that
 * is not a finite non-zero number, leaves the fallback in place: a twin is part
 * of the op's contract and must not throw. */
double param_num(const OpRecord& rec, const char* name, double fallback) {
    for (const OpParam& p : rec.params) {
        if (p.first != name) continue;
        if (auto* i = std::get_if<int64_t>(&p.second))
            return *i == 0 ? fallback : (double)*i;
        const std::string& s = std::get<std::string>(p.second);
        char* end = nullptr;
        const double v = std::strtod(s.c_str(), &end);
        if (end && *end == '\0' && v != 0.0) return v;
        return fallback;
    }
    return fallback;
}

/* The width b broadcasts along when it carries exactly one row of a: 0 when it
 * carries every element. */
int64_t row_width(const AdValue& a, const AdValue& b) {
    const int64_t n = (int64_t)a.value.size();
    const int64_t bw = (int64_t)b.value.size();
    if (bw == n) return 0;
    if (a.shape.size() < 2 || a.shape[0] <= 0) return -1;
    const int64_t width = numel(a.shape) / a.shape[0];
    return bw == width ? width : -1;
}

}  // namespace

/* Twin: the forward in f64 -- a + b, broadcast as the schema says, divided by
 * `div`. An operand whose shape does not meet the op's contract yields no output,
 * which the checkers report against what they expected. */
void add_twin(const OpRecord& rec, const std::vector<AdValue>& op, std::vector<AdValue>* out) {
    if (op.size() < 2) return;
    const AdValue& a = op[0];
    const AdValue& b = op[1];
    const int64_t n = (int64_t)a.value.size();
    const int64_t width = row_width(a, b);
    if (width < 0) return;
    const double inv = 1.0 / param_num(rec, "div", 1.0);

    AdValue y;
    y.dtype = a.dtype;
    y.shape = a.shape;
    y.value.assign((size_t)n, 0.0);
    if (width == 0) {
        for (int64_t i = 0; i < n; ++i) y.value[(size_t)i] = a.value[(size_t)i] + b.value[(size_t)i];
    } else {
        for (int64_t i = 0; i < n; ++i)
            y.value[(size_t)i] = a.value[(size_t)i] + b.value[(size_t)(i % width)];
    }
    for (double& v : y.value) v *= inv;
    out->push_back(std::move(y));
}

/* Host adjoint, in f64. The record's schema puts a at slot 0, b at slot 1 and y
 * at slot 2, so the cotangent of y is cot[0] and the cotangents to add to are
 * dc->at(0) and dc->at(1). Both are added to, never overwritten: other ops share
 * the operand. */
void add_host_adjoint(const AdjointCall& call) {
    const AdValue& y_cot = call.cot[0];
    AdValue& a_cot = call.dc->at(0);
    AdValue& b_cot = call.dc->at(1);
    const size_t n = a_cot.value.size();
    const double inv = 1.0 / param_num(call.rec, "div", 1.0);

    /* a: no broadcast, the cotangent flows straight through. */
    for (size_t i = 0; i < n; ++i) a_cot.value[i] += y_cot.value[i] * inv;
    /* b: straight through when b matches y element for element, the row sum when
     * it broadcasts along rows. */
    const size_t width = b_cot.value.size();
    if (width == n) {
        for (size_t i = 0; i < n; ++i) b_cot.value[i] += y_cot.value[i] * inv;
    } else if (width > 0 && n % width == 0) {
        for (size_t i = 0; i < n; ++i) b_cot.value[i % width] += y_cot.value[i] * inv;
    }
    /* Add has no trainable leaves: call.grads stays untouched. */
}

R97_REGISTER_TWIN("add", add_twin)
R97_REGISTER_ADJOINT("add", host, add_host_adjoint)

}  // namespace r97
