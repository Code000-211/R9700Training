// src/adjoint/host/quant.cpp -- the quant family's host twins and host adjoints. quant_act_i8g
// turns groups of an activation into int8 codes and had_quant_act_i8g does the same after a
// width-group Hadamard rotation; rmsnorm_quant_fp8 and gated_quant_fp8 are the int8 trunk's fused
// E4M3 producers, which carry the residual add and the norm or the gated product in the same op.
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

#include "rad_plugin.h"  // bf16 and fp8 e4m3, the fused quantiser's two grids

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

// -- the fused fp8 producers -------------------------------------------------
//
// The int8 trunk's norms and its gated MLP do their rounding inside the producer:
// `rmsnorm_quant_fp8` writes the normalised bf16 activation and its E4M3 codes, and
// `gated_quant_fp8` writes the bf16 product and its codes. Both quantise per 128 columns with the
// same scale the int8 rows use above, so the straight-through rule is the same too; what is new is
// the differentiable part around it, a norm or an activation gate. The code cotangent is divided
// by the group's scale because the tape hands back the codes' own cotangent, not `s * codes`.
constexpr double kFp8Top = 448.0;  // E4M3's largest finite magnitude

double bf16_round(double x) {
    return static_cast<double>(rad_bf16_to_f32(rad_f32_to_bf16(static_cast<float>(x))));
}

std::string string_param(const OpRecord& rec, const char* key, const char* fallback) {
    for (const OpParam& p : rec.params)
        if (p.first == key) {
            if (const std::string* s = std::get_if<std::string>(&p.second)) return *s;
            throw std::invalid_argument("quant: parameter '" + std::string(key) +
                                        "' is not a string");
        }
    return fallback;
}

double double_param(const OpRecord& rec, const char* key, double fallback) {
    for (const OpParam& p : rec.params) {
        if (p.first != key) continue;
        if (const int64_t* i = std::get_if<int64_t>(&p.second)) return static_cast<double>(*i);
        if (const std::string* s = std::get_if<std::string>(&p.second)) {
            try {
                return std::stod(*s);
            } catch (const std::exception&) {
                break;
            }
        }
        throw std::invalid_argument("quant: parameter '" + std::string(key) +
                                    "' is not a number");
    }
    return fallback;
}

struct Fp8Geometry {
    int64_t n = 0;
    int64_t group = 0;
    int64_t ngrp = 0;
};

Fp8Geometry fp8_geometry(const OpRecord& rec) {
    Fp8Geometry g;
    g.n = required(rec, "n");
    if (g.n <= 0) throw std::invalid_argument("quant: 'n' must be positive");
    const int64_t t = optional(rec, "group", 0);
    g.group = (t <= 0 || t > g.n) ? g.n : t;
    g.ngrp = (g.n + g.group - 1) / g.group;
    return g;
}

struct Fp8Point {
    int64_t rows = 0;
    int64_t n = 0;
    int64_t group = 0;
    int64_t ngrp = 0;
    std::vector<double> value;  // what the quantiser saw, bf16-rounded
    std::vector<double> scale;  // [rows, ngrp]
    std::vector<double> code;   // [rows, n], E4M3 values (not the raw bytes)
    std::vector<uint8_t> live;  // 1 where the clamp did not engage
};

Fp8Point quantise_fp8(const std::vector<double>& v, int64_t rows, const Fp8Geometry& geo) {
    Fp8Point p;
    p.rows = rows;
    p.n = geo.n;
    p.group = geo.group;
    p.ngrp = geo.ngrp;
    p.value = v;
    p.scale.assign(static_cast<size_t>(rows * geo.ngrp), 0.0);
    p.code.assign(static_cast<size_t>(rows * geo.n), 0.0);
    p.live.assign(static_cast<size_t>(rows * geo.n), 0);
    // The scale and the code are float arithmetic, as `q8_group` does them: a group's amax times
    // one over the format's top, then the reciprocal. Two codes that sit within a float ulp of a
    // rounding boundary are one code apart, so the twin has to take the reciprocal the way the
    // kernel does or the check would see a whole E4M3 step of difference rather than a bit.
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t gi = 0; gi < geo.ngrp; ++gi) {
            const int64_t k0 = gi * geo.group;
            const int64_t w = std::min(geo.group, geo.n - k0);
            float amax = 0.0f;
            for (int64_t i = 0; i < w; ++i)
                amax = std::max(amax, std::fabs(static_cast<float>(
                                           v[static_cast<size_t>(r * geo.n + k0 + i)])));
            const float top = static_cast<float>(kFp8Top);
            const float s = amax > 0.0f ? amax * (1.0f / top) : 1.0f;
            const float inv = 1.0f / s;
            p.scale[static_cast<size_t>(r * geo.ngrp + gi)] = static_cast<double>(s);
            for (int64_t i = 0; i < w; ++i) {
                const size_t j = static_cast<size_t>(r * geo.n + k0 + i);
                const float u = static_cast<float>(v[j]) * inv;
                const uint8_t bits = rad_f32_to_fp8e4m3(u);
                p.code[j] = static_cast<double>(rad_fp8e4m3_to_f32(bits));
                p.live[j] = std::fabs(u) <= top ? 1 : 0;
            }
        }
    }
    return p;
}

double fp8_scale_of(const Fp8Point& p, int64_t r, int64_t col) {
    return p.scale[static_cast<size_t>(r * p.ngrp + col / p.group)];
}

int schema_index(const OpRecord& rec, const char* name) {
    for (size_t i = 0; i < rec.operands.size(); ++i)
        if (rec.operands[i].buffer == name) return static_cast<int>(i);
    return -1;
}

int write_index(const OpRecord& rec, int at) {
    int w = 0;
    for (const Operand& o : rec.operands)
        if (o.written && w++ == at) return w - 1;
    return -1;
}

const AdValue* cot_of(const AdjointCall& call, int write) {
    if (write < 0 || static_cast<size_t>(write) >= call.cot.size()) return nullptr;
    return &call.cot[static_cast<size_t>(write)];
}

// The bf16 activation the norm or the gate produced, and the fp8 point around it.
struct Fp8Activation {
    Fp8Point point;
    std::vector<double> value;  // [rows, n], in the bf16 activation's own units
};

// The scale and the clamp mask are the only parts of the quantiser the adjoint reads; they are
// properties of the forward, so they come back from the written scale and the activation.
void add_fp8_codes(const AdValue& codes, const AdValue& scale, int64_t rows, const Fp8Geometry& geo,
                   const std::vector<double>& activation, std::vector<double>* d) {
    if (codes.value.size() != static_cast<size_t>(rows * geo.n))
        throw std::invalid_argument("quant: the fp8 codes do not have the activation's shape");
    if (scale.value.size() != static_cast<size_t>(rows * geo.ngrp))
        throw std::invalid_argument("quant: the fp8 scale does not have one value per (row, group)");
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t gi = 0; gi < geo.ngrp; ++gi) {
            const int64_t k0 = gi * geo.group;
            const int64_t w = std::min(geo.group, geo.n - k0);
            const double s = scale.value[static_cast<size_t>(r * geo.ngrp + gi)];
            if (!(s > 0.0))
                throw std::invalid_argument("quant: the scale of a group must be positive");
            for (int64_t i = 0; i < w; ++i) {
                const size_t j = static_cast<size_t>(r * geo.n + k0 + i);
                if (std::fabs(activation[j] / s) <= kFp8Top)
                    (*d)[j] += codes.value[j] / s;
            }
        }
    }
}

// Whether the record asks for the linearised twin: `<op>_ste`, whose base point is the record's
// last read operand (the schema is frozen, so the base point is appended after everything else).
// The forward is then held at the base point's scale and taken straight through in the units of its
// codes,
//
//     codes(x) = c0 + live(x) * (v(x) - v0) / s0,   scale(x) = s0,
//
// with v the activation and no rounding step; the residual and the activation come back as their
// f64 values, so a finite difference sees the norm's or the gate's Jacobian and the quantiser's
// slope rather than a bf16 step.
bool is_linearised(const std::string& op) {
    return op.size() > 4 && op.compare(op.size() - 4, 4, "_ste") == 0;
}

// `rmsnorm_quant_fp8`: y = bf16(rmsnorm(r) * gain), r = x or bf16(x + residual), then y's E4M3
// codes. The residual is in-place, so its output cotangent and the norm's input cotangent both go
// back to its old value and to x.
void rmsnorm_quant_fp8_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                            std::vector<AdValue>* out) {
    const Fp8Geometry geo = fp8_geometry(rec);
    const bool linearised = is_linearised(rec.op);
    const bool has_residual = rec.operands.size() > 1 &&
                              rad_dtype_bits(rad_dtype_parse(rec.operands[1].dtype.c_str())) > 0 &&
                              !rec.operands[1].shape.empty();
    if (operands.empty()) throw std::invalid_argument("quant: rmsnorm_quant_fp8 reads an activation");
    const std::vector<double>& x = operands[0].value;
    if (x.size() % static_cast<size_t>(geo.n) != 0)
        throw std::invalid_argument("quant: the activation is not a whole number of rows of 'n'");
    const int64_t rows = static_cast<int64_t>(x.size()) / geo.n;
    const std::vector<double>* resid = has_residual ? &operands[1].value : nullptr;
    const std::vector<double>& weight = operands[has_residual ? 2 : 1].value;
    if (weight.size() != static_cast<size_t>(geo.n))
        throw std::invalid_argument("quant: the norm gain does not have 'n' values");
    if (resid != nullptr && resid->size() != x.size())
        throw std::invalid_argument("quant: the residual and the activation have different sizes");

    const double eps = double_param(rec, "eps", 1e-6);
    const double wadd = double_param(rec, "wadd", 0.0);
    // The linearised twin has no rounding step at all, so its finite differences are the norm's
    // Jacobian and the quantiser's slope; the forward's own twin rounds as the kernel does.
    std::vector<double> r(x.size());
    for (size_t i = 0; i < r.size(); ++i) {
        const double sum = resid != nullptr ? x[i] + (*resid)[i] : x[i];
        r[i] = linearised ? sum : bf16_round(sum);
    }

    std::vector<double> y(x.size());
    for (int64_t m = 0; m < rows; ++m) {
        double ss = 0.0;
        for (int64_t j = 0; j < geo.n; ++j) {
            const double v = r[static_cast<size_t>(m * geo.n + j)];
            ss += v * v;
        }
        const double rs = 1.0 / std::sqrt(ss / static_cast<double>(geo.n) + eps);
        for (int64_t j = 0; j < geo.n; ++j) {
            const size_t i = static_cast<size_t>(m * geo.n + j);
            const double ny = r[i] * rs * (weight[static_cast<size_t>(j)] + wadd);
            y[i] = linearised ? ny : bf16_round(ny);
        }
    }
    if (linearised) {
        const std::vector<double>& base_y = operands.back().value;
        if (base_y.size() != y.size())
            throw std::invalid_argument("quant: the base point has the activation's shape");
        const Fp8Point b = quantise_fp8(base_y, rows, geo);
        std::vector<double> codes(y.size(), 0.0);
        for (int64_t m = 0; m < rows; ++m) {
            for (int64_t j = 0; j < geo.n; ++j) {
                const size_t i = static_cast<size_t>(m * geo.n + j);
                const double s = fp8_scale_of(b, m, j);
                // The clamp's mask is the adjoint's: it reads where the activation sits now,
                // against the base point's scale, so the two agree on the elements whose code
                // stops responding.
                codes[i] = std::fabs(y[i] / s) <= kFp8Top ? b.code[i] + (y[i] - base_y[i]) / s
                                                          : b.code[i];
            }
        }
        out->push_back(has_residual ? make_value(rec, 0, "bf16", rows, geo.n, r)
                                    : AdValue{"invalid", {}, {}});
        out->push_back(make_value(rec, 1, "fp8_e4m3", rows, geo.n, std::move(codes)));
        out->push_back(make_value(rec, 2, "f32", rows, geo.ngrp, b.scale));
        out->push_back(make_value(rec, 3, "bf16", rows, geo.n, y));
        return;
    }
    const Fp8Point p = quantise_fp8(y, rows, geo);
    out->push_back(has_residual ? make_value(rec, 0, "bf16", rows, geo.n, r)
                                : AdValue{"invalid", {}, {}});
    out->push_back(make_value(rec, 1, "fp8_e4m3", rows, geo.n, p.code));
    out->push_back(make_value(rec, 2, "f32", rows, geo.ngrp, p.scale));
    out->push_back(make_value(rec, 3, "bf16", rows, geo.n, y));
}

void rmsnorm_quant_fp8_adjoint(const AdjointCall& call) {
    const Fp8Geometry geo = fp8_geometry(call.rec);
    // The schema is x, residual, weight, codes, scale, y: the read activation's buffer is the
    // residual stream's name (`h`), not `x`, which is the residual operand it is added into.
    const int ix = 0;
    const int ires = 1;
    const int iw = 2;
    if (call.fwd.size() <= static_cast<size_t>(iw) || call.dc == nullptr ||
        call.dc->size() <= static_cast<size_t>(iw))
        throw std::invalid_argument("quant: rmsnorm_quant_fp8 needs its activation and gain");
    const AdValue& x = call.fwd[static_cast<size_t>(ix)];
    const AdValue& weight = call.fwd[static_cast<size_t>(iw)];
    if (x.value.size() % static_cast<size_t>(geo.n) != 0)
        throw std::invalid_argument("quant: the activation is not a whole number of rows of 'n'");
    const int64_t rows = static_cast<int64_t>(x.value.size()) / geo.n;
    if (weight.value.size() != static_cast<size_t>(geo.n))
        throw std::invalid_argument("quant: the norm gain does not have 'n' values");

    const bool has_residual = call.rec.operands.size() > static_cast<size_t>(ires) &&
                              rad_dtype_bits(rad_dtype_parse(call.rec.operands[ires].dtype.c_str())) > 0 &&
                              !call.rec.operands[ires].shape.empty();
    const AdValue& residual_out = call.fwd[static_cast<size_t>(ires)];
    std::vector<double> r(x.value);
    if (has_residual) {
        if (residual_out.value.size() != r.size())
            throw std::invalid_argument("quant: the residual output has the activation's shape");
        r = residual_out.value;
    }
    const AdValue& y = call.fwd[5];
    if (y.value.size() != r.size())
        throw std::invalid_argument("quant: the normalised activation has the activation's shape");

    const double eps = double_param(call.rec, "eps", 1e-6);
    const double wadd = double_param(call.rec, "wadd", 0.0);
    std::vector<double> rs(static_cast<size_t>(rows), 0.0);
    std::vector<double> d(r.size(), 0.0);
    for (int64_t m = 0; m < rows; ++m) {
        double ss = 0.0;
        for (int64_t j = 0; j < geo.n; ++j) {
            const double v = r[static_cast<size_t>(m * geo.n + j)];
            ss += v * v;
        }
        rs[static_cast<size_t>(m)] =
            1.0 / std::sqrt(ss / static_cast<double>(geo.n) + eps);
    }

    // The output activation's cotangent is already in y's units; the codes' is divided by their
    // scale and kept only where the fp8 clamp did not engage.
    const AdValue* dy = cot_of(call, 3);
    const AdValue* dq = cot_of(call, 1);
    if (dy != nullptr) {
        if (dy->value.size() != d.size())
            throw std::invalid_argument("quant: the activation's cotangent has y's shape");
        for (size_t i = 0; i < d.size(); ++i) d[i] += dy->value[i];
    }
    if (dq != nullptr) add_fp8_codes(*dq, call.fwd[4], rows, geo, y.value, &d);

    const AdValue* dres = cot_of(call, 0);
    std::vector<double> dres_out;
    if (has_residual && dres != nullptr) {
        if (dres->value.size() != d.size())
            throw std::invalid_argument("quant: the residual cotangent has the activation's shape");
        dres_out = dres->value;
    }

    std::vector<double> dx(d.size(), 0.0);
    std::vector<double> dw(static_cast<size_t>(geo.n), 0.0);
    for (int64_t m = 0; m < rows; ++m) {
        const double scale = rs[static_cast<size_t>(m)];
        double drs = 0.0;
        for (int64_t j = 0; j < geo.n; ++j) {
            const size_t i = static_cast<size_t>(m * geo.n + j);
            const double gain = weight.value[static_cast<size_t>(j)] + wadd;
            drs += d[i] * r[i] * gain;
            dx[i] += d[i] * scale * gain;
            dw[static_cast<size_t>(j)] += d[i] * r[i] * scale;
        }
        const double rs3 = scale * scale * scale;
        for (int64_t j = 0; j < geo.n; ++j) {
            const size_t i = static_cast<size_t>(m * geo.n + j);
            dx[i] -= drs * rs3 * r[i] / static_cast<double>(geo.n);
        }
    }
    for (size_t i = 0; i < dx.size(); ++i) {
        call.dc->at(static_cast<size_t>(ix)).value[i] += dx[i];
        if (has_residual) {
            call.dc->at(static_cast<size_t>(ires)).value[i] += dx[i];
            if (!dres_out.empty()) {
                call.dc->at(static_cast<size_t>(ix)).value[i] += dres_out[i];
                call.dc->at(static_cast<size_t>(ires)).value[i] += dres_out[i];
            }
        }
    }
    for (size_t j = 0; j < dw.size(); ++j)
        call.dc->at(static_cast<size_t>(iw)).value[j] += dw[j];
}

// `gated_quant_fp8`: p = bf16(act(gate) * up), then p's E4M3 codes. The activation is two halves
// of one row, so both cotangents go back into the same buffer.
void gated_quant_fp8_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                          std::vector<AdValue>* out) {
    const Fp8Geometry geo = fp8_geometry(rec);
    const bool linearised = is_linearised(rec.op);
    if (operands.empty()) throw std::invalid_argument("quant: gated_quant_fp8 reads an activation");
    const std::vector<double>& gu = operands[0].value;
    const int64_t width = 2 * geo.n;
    if (gu.size() % static_cast<size_t>(width) != 0)
        throw std::invalid_argument("quant: the gate|up activation is not a whole number of rows");
    const int64_t rows = static_cast<int64_t>(gu.size()) / width;
    const bool sigmoid = string_param(rec, "act", "silu") == "sigmoid";

    std::vector<double> p(static_cast<size_t>(rows * geo.n), 0.0);
    for (int64_t m = 0; m < rows; ++m) {
        for (int64_t j = 0; j < geo.n; ++j) {
            const double g = gu[static_cast<size_t>(m * width + j)];
            const double u = gu[static_cast<size_t>(m * width + geo.n + j)];
            const double s = 1.0 / (1.0 + std::exp(-g));
            const double a = sigmoid ? s : g * s;
            p[static_cast<size_t>(m * geo.n + j)] = linearised ? a * u : bf16_round(a * u);
        }
    }
    if (linearised) {
        const std::vector<double>& base = operands.back().value;
        if (base.size() != p.size())
            throw std::invalid_argument("quant: the base point has the product's shape");
        const Fp8Point b = quantise_fp8(base, rows, geo);
        std::vector<double> codes(p.size(), 0.0);
        for (int64_t m = 0; m < rows; ++m) {
            for (int64_t j = 0; j < geo.n; ++j) {
                const size_t i = static_cast<size_t>(m * geo.n + j);
                const double s = fp8_scale_of(b, m, j);
                // As above: the mask follows the activation, not the base point's.
                codes[i] = std::fabs(p[i] / s) <= kFp8Top ? b.code[i] + (p[i] - base[i]) / s
                                                          : b.code[i];
            }
        }
        out->push_back(make_value(rec, 0, "fp8_e4m3", rows, geo.n, std::move(codes)));
        out->push_back(make_value(rec, 1, "f32", rows, geo.ngrp, b.scale));
        out->push_back(make_value(rec, 2, "bf16", rows, geo.n, p));
        return;
    }
    const Fp8Point q = quantise_fp8(p, rows, geo);
    out->push_back(make_value(rec, 0, "fp8_e4m3", rows, geo.n, q.code));
    out->push_back(make_value(rec, 1, "f32", rows, geo.ngrp, q.scale));
    out->push_back(make_value(rec, 2, "bf16", rows, geo.n, p));
}

void gated_quant_fp8_adjoint(const AdjointCall& call) {
    const Fp8Geometry geo = fp8_geometry(call.rec);
    if (call.fwd.empty() || call.dc == nullptr || call.dc->empty())
        throw std::invalid_argument("quant: gated_quant_fp8 needs its activation");
    const AdValue& gu = call.fwd[0];
    const int64_t width = 2 * geo.n;
    if (gu.value.size() % static_cast<size_t>(width) != 0)
        throw std::invalid_argument("quant: the gate|up activation is not a whole number of rows");
    const int64_t rows = static_cast<int64_t>(gu.value.size()) / width;
    const bool sigmoid = string_param(call.rec, "act", "silu") == "sigmoid";

    std::vector<double> p(static_cast<size_t>(rows * geo.n), 0.0);
    for (int64_t m = 0; m < rows; ++m) {
        for (int64_t j = 0; j < geo.n; ++j) {
            const double g = gu.value[static_cast<size_t>(m * width + j)];
            const double u = gu.value[static_cast<size_t>(m * width + geo.n + j)];
            const double s = 1.0 / (1.0 + std::exp(-g));
            p[static_cast<size_t>(m * geo.n + j)] = bf16_round((sigmoid ? s : g * s) * u);
        }
    }
    const AdValue& ffn = call.fwd[3];
    if (ffn.value.size() != p.size())
        throw std::invalid_argument("quant: the gated product does not have the activation's shape");

    std::vector<double> dp(p.size(), 0.0);
    const AdValue* dy = cot_of(call, 2);
    if (dy != nullptr) {
        if (dy->value.size() != dp.size())
            throw std::invalid_argument("quant: the product cotangent has the product's shape");
        dp = dy->value;
    }
    const AdValue* dq = cot_of(call, 0);
    if (dq != nullptr) add_fp8_codes(*dq, call.fwd[2], rows, geo, ffn.value, &dp);

    AdValue& dgu = call.dc->at(0);
    for (int64_t m = 0; m < rows; ++m) {
        for (int64_t j = 0; j < geo.n; ++j) {
            const size_t i = static_cast<size_t>(m * geo.n + j);
            const double g = gu.value[static_cast<size_t>(m * width + j)];
            const double u = gu.value[static_cast<size_t>(m * width + geo.n + j)];
            const double s = 1.0 / (1.0 + std::exp(-g));
            const double a = sigmoid ? s : g * s;
            const double da = sigmoid ? s * (1.0 - s) : s * (1.0 + g * (1.0 - s));
            dgu.value[static_cast<size_t>(m * width + j)] += dp[i] * u * da;
            dgu.value[static_cast<size_t>(m * width + geo.n + j)] += dp[i] * a;
        }
    }
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

R97_REGISTER_TWIN("rmsnorm_quant_fp8", rmsnorm_quant_fp8_twin)
R97_REGISTER_ADJOINT("rmsnorm_quant_fp8", host, rmsnorm_quant_fp8_adjoint)
R97_REGISTER_TWIN("gated_quant_fp8", gated_quant_fp8_twin)
R97_REGISTER_ADJOINT("gated_quant_fp8", host, gated_quant_fp8_adjoint)
// The linearised twins append their base point to the schema -- the activation the quantiser saw --
// because positions 1 and 2 are the residual and the written codes there; the frozen index is that
// appended operand's schema position.
R97_REGISTER_TWIN("rmsnorm_quant_fp8_ste", rmsnorm_quant_fp8_twin)
R97_REGISTER_ADJOINT_F("rmsnorm_quant_fp8_ste", host, rmsnorm_quant_fp8_adjoint, 6)
R97_REGISTER_TWIN("gated_quant_fp8_ste", gated_quant_fp8_twin)
R97_REGISTER_ADJOINT_F("gated_quant_fp8_ste", host, gated_quant_fp8_adjoint, 4)

}  // namespace r97
