// src/adjoint/host/quant.cpp -- the quant family's host twin and host adjoint: quant_act_i8g
// (groups of an activation to int8 codes, one scale per group) and had_quant_act_i8g (the same
// after a width-group Hadamard rotation).
//
// Rounding has zero derivative almost everywhere, so the adjoint is a definition rather than a
// derivative of the forward: the straight-through estimator of docs/adjoints/quant.md. What that
// doc says, and this file implements:
//
//   the cotangent of the codes passes through the rounding where the code is inside the grid
//   (|u| <= 127, u = v/s), is dropped where the clamp engaged, and is then rotated back with
//   W^T = W; the scale's cotangent is dropped, the scale being a constant.
//
// Finite differences cannot differentiate a step function, so each op also registers a *frozen*
// twin under `<op>_ste`: the forward linearised at a base point carried as the record's second read
// operand. Its slope is the adjoint above, which is what makes the check of docs/adjoints/README.md
// possible; no walk appends a `<op>_ste` record.
#include "r97/adjoint.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace r97 {
namespace {

// The top of the int8 code grid. The scale is the group's amax over it, so the largest value of
// every group maps to this code.
constexpr double kTop = 127.0;

// The ops this file serves: the two of the family, and the linearised twin of each.
enum class Op { quant, had, quant_ste, had_ste };

Op op_of(const std::string& name) {
    if (name == "quant_act_i8g") return Op::quant;
    if (name == "had_quant_act_i8g") return Op::had;
    if (name == "quant_act_i8g_ste") return Op::quant_ste;
    if (name == "had_quant_act_i8g_ste") return Op::had_ste;
    throw std::invalid_argument("quant: no op of this family is named '" + name + "'");
}

bool is_rotated(Op op) { return op == Op::had || op == Op::had_ste; }
bool is_frozen(Op op) { return op == Op::quant_ste || op == Op::had_ste; }

// -- the record's parameters -------------------------------------------------

const int64_t* int_param(const OpRecord& rec, const char* key) {
    for (const OpParam& p : rec.params) {
        if (p.first != key) continue;
        const int64_t* v = std::get_if<int64_t>(&p.second);
        if (v == nullptr)
            throw std::invalid_argument("quant: parameter '" + std::string(key) +
                                        "' is not an integer");
        return v;
    }
    return nullptr;
}

int64_t required(const OpRecord& rec, const char* key) {
    const int64_t* v = int_param(rec, key);
    if (v == nullptr)
        throw std::invalid_argument("quant: the record has no '" + std::string(key) +
                                    "' parameter");
    return *v;
}

int64_t optional(const OpRecord& rec, const char* key, int64_t fallback) {
    const int64_t* v = int_param(rec, key);
    return v != nullptr ? *v : fallback;
}

// The shapes of one call, read the way the kernel reads them.
struct Geometry {
    int64_t n = 0;     // columns of the activation
    int64_t group = 0; // the effective group width
    int64_t ngrp = 0;  // groups per row
    bool rotated = false;
};

Geometry geometry(Op op, const OpRecord& rec) {
    Geometry g;
    g.rotated = is_rotated(op);
    g.n = required(rec, "n");
    if (g.n <= 0) throw std::invalid_argument("quant: 'n' must be positive");
    const int64_t t = optional(rec, "group", 0);
    if (g.rotated) {
        // A rotation exists only for a power of two, and a group must not straddle the end of a
        // row: the kernel refuses both, and so does the twin.
        if (t <= 0 || (t & (t - 1)) != 0 || g.n % t != 0)
            throw std::invalid_argument(
                "had_quant_act_i8g: 'group' must be a power of two that divides 'n'");
        g.group = t;
    } else {
        // One scale per row for a group that is absent or wider than the row, and a short last
        // group when the group does not divide the row.
        g.group = (t <= 0 || t > g.n) ? g.n : t;
    }
    g.ngrp = (g.n + g.group - 1) / g.group;
    return g;
}

// -- the quantiser -----------------------------------------------------------

// The unnormalised Walsh-Hadamard transform of one group, the kernel's own loop: W W = t I.
void fwht(double* v, int64_t t) {
    for (int64_t h = 1; h < t; h <<= 1)
        for (int64_t i0 = 0; i0 < t; i0 += 2 * h)
            for (int64_t j = i0; j < i0 + h; ++j) {
                const double a = v[j];
                const double b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
}

// Everything the forward knows about one call: the values the quantiser saw (`v`), the scale of
// every group, the codes, and where the clamp engaged. `v`, `code` and `live` are
// piecewise constant in the activation, which is why the frozen twin below holds them at a base
// point and why the adjoint reads them off the forward's own values.
struct Point {
    int64_t rows = 0;
    int64_t n = 0;
    int64_t group = 0;
    int64_t ngrp = 0;
    bool rotated = false;
    std::vector<double> v;       // [rows, n], rotated for the Hadamard op
    std::vector<double> scale;   // [rows, ngrp]
    std::vector<double> code;    // [rows, n]
    std::vector<uint8_t> live;   // 1 where the clamp did not engage
};

Point quantise(const std::vector<double>& x, int64_t rows, const Geometry& geo) {
    Point p;
    p.rows = rows;
    p.n = geo.n;
    p.group = geo.group;
    p.ngrp = geo.ngrp;
    p.rotated = geo.rotated;
    p.v = x;
    p.scale.assign(static_cast<size_t>(rows * geo.ngrp), 0.0);
    p.code.assign(static_cast<size_t>(rows * geo.n), 0.0);
    p.live.assign(static_cast<size_t>(rows * geo.n), 0);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t g = 0; g < geo.ngrp; ++g) {
            const int64_t k0 = g * geo.group;
            const int64_t w = std::min(geo.group, geo.n - k0);  // a short last group
            double* v = p.v.data() + r * geo.n + k0;
            if (geo.rotated) fwht(v, geo.group);  // n % group == 0, so every block is full
            double amax = 0.0;
            for (int64_t i = 0; i < w; ++i) amax = std::max(amax, std::fabs(v[i]));
            // The zero group has no direction to scale: one scale, codes zero, nothing clamped.
            const double s = amax > 0.0 ? amax * (1.0 / kTop) : 1.0;
            p.scale[static_cast<size_t>(r * geo.ngrp + g)] = s;
            for (int64_t i = 0; i < w; ++i) {
                const size_t j = static_cast<size_t>(r * geo.n + k0 + i);
                const double u = v[i] * (1.0 / s);
                const double c = std::nearbyint(u);  // nearest, ties to even
                p.code[j] = std::min(kTop, std::max(-kTop, c));
                p.live[j] = std::fabs(u) <= kTop ? 1 : 0;
            }
        }
    }
    return p;
}

// The scale a column is stored against.
double scale_of(const Point& p, int64_t r, int64_t col) {
    return p.scale[static_cast<size_t>(r * p.ngrp + col / p.group)];
}

// -- outputs -----------------------------------------------------------------

std::string write_dtype(const OpRecord& rec, size_t index, const char* fallback) {
    size_t w = 0;
    for (const Operand& o : rec.operands) {
        if (!o.written) continue;
        if (w++ == index) return o.dtype.empty() ? std::string(fallback) : o.dtype;
    }
    return fallback;
}

AdValue make_value(const OpRecord& rec, size_t index, const char* dtype, int64_t rows, int64_t cols,
                   std::vector<double> value) {
    AdValue a;
    a.dtype = write_dtype(rec, index, dtype);
    a.shape = {rows, cols};
    a.value = std::move(value);
    return a;
}

// -- the twin ----------------------------------------------------------------

// The forward in f64, and (under `<op>_ste`) its linearisation at a base point: with `operands`
// = (x, x0), every output is held at what the forward wrote at x0 except for the rounding
// residual, which is taken straight through in the units of the codes,
//
//     codes(x)  =  c0 + live0 ⊙ (v(x) − v0) / s0,        scale(x)  =  s0,
//
// so d codes / dx = live0 ⊙ dv/dx / s0 at x0. The product s·codes is the issue's twin
// x_hat = W x + const; it is divided by the scale here because the cotangent the tape hands back
// is the codes' own (docs/adjoints/quant.md).
void quant_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                std::vector<AdValue>* out) {
    const Op op = op_of(rec.op);
    const Geometry geo = geometry(op, rec);
    if (operands.empty()) throw std::invalid_argument("quant: the op reads an activation");
    const std::vector<double>& x = operands[0].value;
    if (x.size() % static_cast<size_t>(geo.n) != 0)
        throw std::invalid_argument("quant: the activation is not a whole number of rows of 'n'");
    const int64_t rows = static_cast<int64_t>(x.size()) / geo.n;

    if (!is_frozen(op)) {
        const Point fwd = quantise(x, rows, geo);
        out->push_back(make_value(rec, 0, "i8", rows, geo.n, fwd.code));
        out->push_back(make_value(rec, 1, "f32", rows, geo.ngrp, fwd.scale));
        return;
    }

    if (operands.size() < 2)
        throw std::invalid_argument(
            "quant: the straight-through twin reads the base point as its second operand");
    if (operands[1].value.size() != x.size())
        throw std::invalid_argument("quant: the base point and the activation have different sizes");
    const Point base = quantise(operands[1].value, rows, geo);
    const Point at = quantise(x, rows, geo);  // only `at.v`, the rotated activation, is used

    std::vector<double> codes(static_cast<size_t>(rows * geo.n), 0.0);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t j = 0; j < geo.n; ++j) {
            const size_t i = static_cast<size_t>(r * geo.n + j);
            const double s = scale_of(base, r, j);
            codes[i] = base.live[i] != 0 ? base.code[i] + (at.v[i] - base.v[i]) / s : base.code[i];
        }
    }
    out->push_back(make_value(rec, 0, "i8", rows, geo.n, std::move(codes)));
    out->push_back(make_value(rec, 1, "f32", rows, geo.ngrp, base.scale));
}

// -- the adjoint -------------------------------------------------------------

// dL/dx = W^T (live ⊙ cot_codes / s), the codes' cotangent taken through the rounding and dropped
// where the code saturated, then rotated back. W^T = W for the Walsh order, and the scale's
// cotangent is dropped: the scale is a constant.
void quant_adjoint(const AdjointCall& call) {
    const Op op = op_of(call.rec.op);
    const Geometry geo = geometry(op, call.rec);
    size_t reads = 0;
    for (const Operand& o : call.rec.operands)
        if (!o.written) ++reads;
    // The schema: the activation, the codes the op wrote, the scales it wrote. The frozen twin
    // reads its base point between the activation and the writes, so neither index is written
    // down here -- the writes are found from the record.
    if (reads == 0 || call.fwd.size() < reads + 2)
        throw std::invalid_argument(
            "quant: the record must read the activation and write the codes and the scale");
    if (call.cot.size() < 2)
        throw std::invalid_argument(
            "quant: the record writes the codes and the scale, so the walk hands two cotangents");
    if (call.dc == nullptr || call.dc->size() < reads)
        throw std::invalid_argument("quant: the walk handed no room for the activation's cotangent");

    const AdValue& x = call.fwd[0];
    const AdValue& scale = call.fwd[reads + 1];
    const AdValue& g = call.cot[0];  // the codes' cotangent; the scale's, cot[1], is dropped
    if (x.value.size() % static_cast<size_t>(geo.n) != 0)
        throw std::invalid_argument("quant: the activation is not a whole number of rows of 'n'");
    const int64_t rows = static_cast<int64_t>(x.value.size()) / geo.n;
    if (g.value.size() != x.value.size())
        throw std::invalid_argument("quant: the codes' cotangent has the activation's shape");
    if (scale.value.size() != static_cast<size_t>(rows * geo.ngrp))
        throw std::invalid_argument("quant: the scale does not have one value per (row, group)");

    // The values the quantiser saw, for the mask, from the activation the walk hands back.
    std::vector<double> v(x.value);
    if (geo.rotated)
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t gi = 0; gi < geo.ngrp; ++gi)
                fwht(v.data() + r * geo.n + gi * geo.group, geo.group);

    std::vector<double> d(g.value);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t gi = 0; gi < geo.ngrp; ++gi) {
            const int64_t k0 = gi * geo.group;
            const int64_t w = std::min(geo.group, geo.n - k0);
            const double s = scale.value[static_cast<size_t>(r * geo.ngrp + gi)];
            // The family's scale is amax/127, or 1 for the zero group: always positive.
            if (!(s > 0.0))
                throw std::invalid_argument("quant: the scale of a group must be positive");
            for (int64_t i = 0; i < w; ++i) {
                const size_t j = static_cast<size_t>(r * geo.n + k0 + i);
                // The clamp's own derivative: zero where it engaged, one inside the grid.
                d[j] = std::fabs(v[j] * (1.0 / s)) > kTop ? 0.0 : d[j] / s;
            }
            if (geo.rotated) fwht(d.data() + r * geo.n + k0, geo.group);
        }
    }

    AdValue& dx = call.dc->at(0);
    for (size_t i = 0; i < dx.value.size() && i < d.size(); ++i) dx.value[i] += d[i];
}

}  // namespace

// The family registers one twin and one adjoint function under each op's name, and the linearised
// twin under `<op>_ste` with its base point (schema position 1) frozen, for the finite-difference
// check. A walk appends records under the two real names only.
R97_REGISTER_TWIN("quant_act_i8g_ste", quant_twin)
R97_REGISTER_ADJOINT_F("quant_act_i8g_ste", host, quant_adjoint, 1)
R97_REGISTER_TWIN("had_quant_act_i8g_ste", quant_twin)
R97_REGISTER_ADJOINT_F("had_quant_act_i8g_ste", host, quant_adjoint, 1)
R97_REGISTER_TWIN("quant_act_i8g", quant_twin)
R97_REGISTER_ADJOINT("quant_act_i8g", host, quant_adjoint)
R97_REGISTER_TWIN("had_quant_act_i8g", quant_twin)
R97_REGISTER_ADJOINT("had_quant_act_i8g", host, quant_adjoint)

}  // namespace r97
