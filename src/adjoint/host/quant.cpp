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

#include "parallel.h"

#include "rad_plugin.h"  // bf16 and fp8 e4m3, the fused quantiser's two grids

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace r97 {
namespace {

// The unit the parallel for cuts a quantiser pass into: one activation row, whose scale, codes and
// rotation are its own. A call with fewer rows than this stays on the calling thread.
constexpr int64_t kRowChunk = 2;

// The top of the int8 code grid. The scale is the group's amax over it, so the largest value of
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

// -- the fused producers' predicates -----------------------------------------
//
// The rows of the int8 trunk's three fused producers are keyed by a predicate on the record's
// parameters: the rotation the record can take and, for the two gated ones, the activation it
// serves. The device side keys its rows with the same three (src/adjoint/device/quant.hip), so one
// record reaches one row on both sides. A predicate reads the parameters only, because the row
// lookup has no record.

// A numeric parameter: an int, or a decimal string that is the whole of the string. Missing or
// non-numeric leaves false rather than throwing -- a lookup refuses a record it cannot judge.
bool num_param(const std::vector<OpParam>& params, const char* name, double* out) {
    for (const OpParam& p : params) {
        if (p.first != name) continue;
        if (const int64_t* i = std::get_if<int64_t>(&p.second)) {
            *out = static_cast<double>(*i);
            return true;
        }
        const std::string* s = std::get_if<std::string>(&p.second);
        if (s == nullptr || s->empty()) return false;
        char* end = nullptr;
        const double v = std::strtod(s->c_str(), &end);
        if (end == nullptr || *end != '\0') return false;
        *out = v;
        return true;
    }
    return false;
}

// The rotation and the per-group scale are the record's `n` and `group`: a power of two that
// divides the width, no wider than one block's threads. A record outside that is refused at the
// row lookup rather than served approximately.
bool rotated(const std::vector<OpParam>& params) {
    double n = 0.0, group = 0.0;
    if (!num_param(params, "n", &n) || !num_param(params, "group", &group)) return false;
    if (n <= 0.0 || n != std::floor(n)) return false;
    if (group <= 0.0 || group != std::floor(group) || group > 1024.0) return false;
    const int64_t g = static_cast<int64_t>(group);
    if ((g & (g - 1)) != 0 || static_cast<int64_t>(n) % g != 0) return false;
    return true;
}

// The gated producer's activation: silu, its documented first mode and the one the card's swiglu
// sites issue. The sigmoid form would take a different second half and is not served yet.
bool serves_gated(const std::vector<OpParam>& params) {
    if (!rotated(params)) return false;
    for (const OpParam& p : params) {
        if (p.first != "act") continue;
        const std::string* act = std::get_if<std::string>(&p.second);
        return act != nullptr && *act == "silu";
    }
    return true;
}

// The gate's activation: sigmoid, the mode the card's narrow-head gate issues. It is required,
// because the record's default is the schema's to say, not this row's to guess.
bool serves_gate(const std::vector<OpParam>& params) {
    if (!rotated(params)) return false;
    for (const OpParam& p : params) {
        if (p.first != "act") continue;
        const std::string* act = std::get_if<std::string>(&p.second);
        return act != nullptr && *act == "sigmoid";
    }
    return false;
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
    parallel_for(rows, kRowChunk, [&](int64_t r) {
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
    });
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
    std::vector<double> d(g.value);
    parallel_for(rows, kRowChunk, [&](int64_t r) {
        for (int64_t gi = 0; gi < geo.ngrp; ++gi) {
            const int64_t k0 = gi * geo.group;
            const int64_t w = std::min(geo.group, geo.n - k0);
            if (geo.rotated) fwht(v.data() + r * geo.n + k0, geo.group);
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
    });

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

// -- the fused int8 producers ------------------------------------------------
//
// The int8 trunk's norm and its gates round inside the producer: `rmsnorm_had_quant_i8g`,
// `gated_had_quant_i8g` and `gate_had_quant_i8g` apply an elementwise part and hand the result to
// the quantiser above. Their forward is the parts' forwards composed and their adjoint the parts'
// host adjoints composed, in the order the kernel undoes them (docs/adjoints/quant.md). The parts
// are the elementwise family's rows (src/adjoint/host/elementwise.cpp) and the quantiser's own
// under `had_quant_act_i8g`, found through the registry so there is one definition of each.

// The elementwise op a fused producer applies before the rotation.
enum class Part { rmsnorm, silu_mul, sigmoid_mul };

const char* part_op(Part p) {
    switch (p) {
        case Part::rmsnorm: return "rmsnorm";
        case Part::silu_mul: return "silu_mul";
        case Part::sigmoid_mul: return "act_mul";
    }
    return "";
}

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return shape.empty() ? 0 : n;
}

// A shape as `[a,b]`, for the refusals that have to say what the tape carried.
std::string shape_text(const std::vector<int64_t>& shape) {
    std::string s = "[";
    for (size_t d = 0; d < shape.size(); ++d) s += (d == 0 ? "" : ",") + std::to_string(shape[d]);
    return s + "]";
}

// The quantiser's own record over the part's output: the fused producer's intermediate is the
// quantiser's activation, so the part is served by the family's existing twin and adjoint under
// had_quant_act_i8g. Under `<op>_ste` the quantiser's own linearised row is used instead, and its
// schema is the activation, the base point it was linearised at, the codes, the scale
// (`quant_twin` reads the base point as the record's second operand, :296).
OpRecord had_quant_record(int64_t rows, int64_t n, int64_t group, bool linearised) {
    OpRecord q;
    q.op = linearised ? "had_quant_act_i8g_ste" : "had_quant_act_i8g";
    q.params = {{"M", rows}, {"n", n}, {"group", group}, {"dtype", std::string("bf16")}};
    q.operands = {
        Operand{"x", "bf16", {rows, n}, 0, rows, false},
    };
    if (linearised)
        q.operands.push_back(Operand{"base", "bf16", {rows, n}, 0, rows, false});
    q.operands.push_back(Operand{"q", "i8", {rows, n}, 0, rows, true});
    q.operands.push_back(Operand{"scale", "f32", {rows, n / group}, 0, rows, true});
    return q;
}

// The cotangent slot of a written operand: `cot` holds one entry per write, in the record's write
// order, so this is the write's index among the writes -- the device row's rule
// (`src/adjoint/device/quant.hip:531`). Reading `cot[0]` positionally is wrong on the live fused
// rmsnorm record, which writes the normed value in place *before* the codes: cot[0] there is a slot
// no consumer fills, and the quantiser's part then contributes nothing.
int64_t cot_slot(const OpRecord& rec, int64_t operand) {
    int64_t at = 0;
    for (int64_t j = 0; j < (int64_t)rec.operands.size(); ++j) {
        if (!rec.operands[j].written) continue;
        if (j == operand) return at;
        ++at;
    }
    return -1;
}

// The codes and the scale among the record's writes, found by dtype and shape as the device row
// finds them (`src/adjoint/device/quant.hip:492-523`). A position would be wrong under `_ste`,
// whose base point is appended after the writes and so moves what counting the reads returns.
void written_codes_and_scale(const AdjointCall& call, int64_t* codes_at, int64_t* scale_at) {
    *codes_at = -1;
    *scale_at = -1;
    const std::vector<Operand>& ops = call.rec.operands;
    for (int64_t j = 0; j < (int64_t)ops.size(); ++j) {
        const Operand& o = ops[(size_t)j];
        if (!o.written || o.dtype.empty() || o.shape.empty()) continue;
        if (*codes_at < 0 && (o.dtype == "i8" || o.dtype == "fp8_e4m3" || o.dtype == "fp8_e5m2") &&
            o.shape.size() == 2)
            *codes_at = j;
        else if (*scale_at < 0 && o.dtype == "f32" && o.shape.size() == 2)
            *scale_at = j;
    }
}

// The forward of each fused producer, as the parts' twins composed: the elementwise part over the
// record's read operands, then the quantiser over its output.
void fused_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                std::vector<AdValue>* out, Part part) {
    const int64_t group = required(rec, "group");
    const bool linearised = is_linearised(rec.op);

    std::vector<AdValue> y;
    TwinFn elem = twin_find(part_op(part), rec.params, nullptr);
    if (elem == nullptr)
        throw std::invalid_argument(std::string("quant: no twin for ") + part_op(part));
    elem(rec, operands, &y);
    if (y.empty() || y[0].shape.empty() || y[0].shape[0] <= 0)
        throw std::invalid_argument(std::string("quant: ") + part_op(part) +
                                    " wrote no [rows, n] intermediate for the quantiser");
    // The quantiser's width is the intermediate's value count over its rows, not the record's `n`:
    // the live gate record declares `n=256` while its codes are 6144 wide, and the elementwise
    // part hands back its own shape (`[rows, heads, dim]`) whose flattened width is the codes'
    // width. The device row reads the same fact off the codes; this is the forward's half of it.
    const int64_t rows = y[0].shape[0];
    if ((int64_t)y[0].value.size() % rows != 0)
        throw std::invalid_argument(std::string("quant: ") + part_op(part) + " wrote " +
                                    std::to_string(y[0].value.size()) + " values over " +
                                    std::to_string(rows) + " rows, which is not a whole width");
    const int64_t n = (int64_t)y[0].value.size() / rows;

    const OpRecord q = had_quant_record(rows, n, group, linearised);
    TwinFn quant = twin_find(q.op, q.params, nullptr);
    if (quant == nullptr)
        throw std::invalid_argument(std::string("quant: no twin for ") + q.op);
    // The linearised quantiser reads its base point as the second operand; the fused record names
    // it last, and `operands` arrives in read order, so it is `operands.back()`.
    std::vector<AdValue> qin = {y[0]};
    if (linearised) {
        if (operands.size() < 2 || operands.back().value.size() != y[0].value.size())
            throw std::invalid_argument(std::string("quant: ") + rec.op +
                                        ": the base point does not have the activation's shape");
        qin.push_back(operands.back());
    }
    quant(q, qin, out);
}

// The adjoint of each fused producer: the parts' host adjoints, in the order the kernel undoes
// them. One read operand's cotangent is the quantiser's output cotangent; the elementwise part's
// adjoint then folds it into the read operands' slots, which is where the device kernel must land
// too.
void fused_adjoint(const AdjointCall& call, Part part) {
    const bool linearised = is_linearised(call.rec.op);
    size_t reads = 0;
    for (const Operand& o : call.rec.operands)
        if (!o.written) ++reads;
    if (reads == 0 || call.fwd.size() < reads + 2)
        throw std::invalid_argument(
            "quant: the record must read the activation and write the codes and the scale");
    if (call.cot.size() < 2)
        throw std::invalid_argument(
            "quant: the record writes the codes and the scale, so the walk hands two cotangents");
    if (call.dc == nullptr || call.dc->size() < reads)
        throw std::invalid_argument("quant: the walk handed no room for the activation's cotangent");
    // The codes and the scale by dtype and shape, never by position -- under `_ste` the base point
    // is appended after the writes, so counting the reads lands one slot past the codes.
    int64_t codes_at = -1, scale_at = -1;
    written_codes_and_scale(call, &codes_at, &scale_at);
    if (codes_at < 0 || scale_at < 0)
        throw std::invalid_argument(std::string("quant: ") + call.rec.op +
                                    ": the record does not write the codes and the scale");
    const AdValue& codes = call.fwd[static_cast<size_t>(codes_at)];
    const AdValue& scale = call.fwd[static_cast<size_t>(scale_at)];

    // The intermediate the elementwise part wrote, which the quantiser's adjoint differentiates.
    std::vector<AdValue> y;
    TwinFn elem = twin_find(part_op(part), call.rec.params, nullptr);
    if (elem == nullptr)
        throw std::invalid_argument(std::string("quant: no twin for ") + part_op(part));
    elem(call.rec, call.fwd, &y);

    // The widths are the operands', as the device row reads them: the live gate record declares
    // `n=256` while its codes are 6144 wide over a 48-group scale, and its elementwise
    // intermediate carries its own shape (`[rows, heads, dim]`) with the codes' value count. A
    // rank test on that intermediate refused a record the device row serves; the value count is
    // the fact both sides need.
    if (codes.shape.size() != 2 || codes.shape[0] <= 0 || codes.shape[1] <= 0)
        throw std::invalid_argument(std::string("quant: ") + call.rec.op + ": the codes are " +
                                    shape_text(codes.shape) + ", not [rows, n]");
    if (scale.shape.size() != 2 || scale.shape[0] != codes.shape[0] || scale.shape[1] <= 0 ||
        codes.shape[1] % scale.shape[1] != 0)
        throw std::invalid_argument(std::string("quant: ") + call.rec.op + ": the codes are " +
                                    shape_text(codes.shape) + " and the scale " +
                                    shape_text(scale.shape) +
                                    ", which is not a group scale over the codes");
    const int64_t rows = codes.shape[0];
    const int64_t n = codes.shape[1];
    const int64_t group = n / scale.shape[1];
    const int64_t total = rows * n;
    if (y.empty() || (int64_t)y[0].value.size() != total) {
        std::string detail = " intermediate=" + (y.empty() ? std::string("none")
                                                           : shape_text(y[0].shape) + " with " +
                                                                 std::to_string(y[0].value.size()) +
                                                                 " values") +
                             " codes=" + shape_text(codes.shape) + " (" + std::to_string(total) +
                             " values)";
        for (size_t j = 0; j < call.rec.operands.size(); ++j) {
            const Operand& o = call.rec.operands[j];
            detail += " [" + std::to_string(j) + "] '" + o.buffer + "' " + o.dtype + " " +
                      shape_text(o.shape) + (o.written ? " written" : "");
        }
        throw std::invalid_argument(std::string("quant: ") + call.rec.op + ": the " +
                                    part_op(part) +
                                    " intermediate does not carry the codes' values" + detail);
    }

    const OpRecord q = had_quant_record(rows, n, group, linearised);
    std::vector<AdValue> qfwd = {y[0]};
    if (linearised) {
        if (call.fwd.empty() || call.fwd.back().value.size() != y[0].value.size())
            throw std::invalid_argument(std::string("quant: ") + call.rec.op +
                                        ": the base point does not have the activation's shape");
        qfwd.push_back(call.fwd.back());
    }
    qfwd.push_back(codes);
    qfwd.push_back(scale);
    const int64_t codes_cot_at = cot_slot(call.rec, codes_at);
    const int64_t scale_cot_at = cot_slot(call.rec, scale_at);
    if (codes_cot_at < 0 || (size_t)codes_cot_at >= call.cot.size() || scale_cot_at < 0 ||
        (size_t)scale_cot_at >= call.cot.size())
        throw std::invalid_argument(std::string("quant: ") + call.rec.op + ": the codes' write " +
                                    std::to_string(codes_at) + " and the scale's " +
                                    std::to_string(scale_at) + " need cotangent slots " +
                                    std::to_string(codes_cot_at) + " and " +
                                    std::to_string(scale_cot_at) + " of the " +
                                    std::to_string(call.cot.size()) + " the walk handed");
    std::vector<AdValue> qcot = {call.cot[(size_t)codes_cot_at], call.cot[(size_t)scale_cot_at]};
    std::vector<AdValue> qdc;
    for (const Operand& o : q.operands) {
        AdValue v;
        v.dtype = o.dtype;
        v.shape = o.shape;
        v.value.assign(static_cast<size_t>(numel(o.shape)), 0.0);
        qdc.push_back(std::move(v));
    }
    AdjointCall qcall{q, qfwd, qcot, &qdc, nullptr, ""};
    AdjointFn quant = adjoint_find(q.op, Side::host, q.params, nullptr);
    if (quant == nullptr)
        throw std::invalid_argument("quant: no host adjoint for had_quant_act_i8g");
    quant(qcall);

    // R97_DUMP_QUANT=1: the twin's intermediate against the codes the tape's own kernel wrote.
    // Radiance's kernel table has no host entry for the int8 fused producers (`rmsnorm_had_quant_i8g`
    // and its two siblings), so `check_kernel` cannot check their twins and their FD differentiates a
    // row against the twin it is testing. The tape is the oracle instead: re-quantising the twin's
    // intermediate must reproduce the codes and the scale the kernel actually wrote, and `code_err`
    // is the largest disagreement. A non-zero one means the row's mask and its dx are taken at an
    // activation the kernel never had.
    if (const char* dq = std::getenv("R97_DUMP_QUANT"); dq != nullptr && dq[0] == '1') {
        std::vector<AdValue> twin_out;
        std::vector<AdValue> qin = {y[0]};
        if (linearised && !call.fwd.empty()) qin.push_back(call.fwd.back());
        const TwinFn qt = twin_find(q.op, q.params, nullptr);
        double code_err = 0.0, scale_rel = 0.0;
        int64_t code_bad = 0;
        int64_t hist[10] = {0};
        if (qt != nullptr) {
            qt(q, qin, &twin_out);
            // `code_bad` counts any nonzero difference, so it cannot tell a uniform sub-LSB
            // mismatch (which flips nearly every code) from a structural one. The histogram can: a
            // tight 1-2 with a few outliers is a small systematic difference in the gain or the
            // precision, a broad spread is what a layout difference would need.
            if (!twin_out.empty() && twin_out[0].value.size() == codes.value.size())
                for (size_t i = 0; i < codes.value.size(); ++i) {
                    const double d = std::fabs(twin_out[0].value[i] - codes.value[i]);
                    if (d > 0.0) ++code_bad;
                    if (d > code_err) code_err = d;
                    ++hist[d == 0.0 ? 0
                                    : d <= 1.0 ? 1
                                    : d <= 2.0 ? 2
                                    : d <= 4.0 ? 3
                                    : d <= 8.0 ? 4
                                    : d <= 16.0 ? 5
                                    : d <= 32.0 ? 6
                                    : d <= 64.0 ? 7
                                    : d <= 128.0 ? 8 : 9];
                }
            // A *second* Hadamard on top of the one the twin already applies: `geometry()` sets
            // `rotated = is_rotated(op)` and `Op::had` is rotated, so the plain comparison above is
            // the single-rotation arm. This arm is therefore a control for "twice rotated", not an
            // "unrotated" arm -- the gated producers collapse here (99%) and match above (5.9%),
            // which is what makes it useful.
            double rot_err = 0.0;
            int64_t rot_bad = 0;
            {
                std::vector<AdValue> rot = {y[0]};
                const int64_t rn = n;
                const int64_t rgrp = group;
                for (int64_t r = 0; r < rows; ++r)
                    for (int64_t gi = 0; gi < rn / rgrp; ++gi) {
                        const int64_t k0 = gi * rgrp;
                        const int64_t w = std::min(rgrp, rn - k0);
                        if (w == rgrp) fwht(rot[0].value.data() + r * rn + k0, rgrp);
                    }
                std::vector<AdValue> rot_out;
                qt(q, rot, &rot_out);
                if (!rot_out.empty() && rot_out[0].value.size() == codes.value.size())
                    for (size_t i = 0; i < codes.value.size(); ++i) {
                        const double d = std::fabs(rot_out[0].value[i] - codes.value[i]);
                        if (d > 0.0) ++rot_bad;
                        if (d > rot_err) rot_err = d;
                    }
            }
            // The kernel's own activation, reconstructed from its codes and scale: `codes * scale` is
            // the rotated activation and `fwht` is unnormalised (H*H = n*I), so the inverse is
            // `fwht(.)/group`. Comparing this to `y[0]` tests the twin's *activation* without going
            // through the twin's quantiser at all, which is what separates "the norm is wrong" from
            // "the quantiser's scale is wrong".
            double deq_err = 0.0, deq_scale = 0.0;
            {
                const int64_t ngrp = n / group;
                std::vector<double> dq(codes.value.size());
                for (size_t i = 0; i < dq.size(); ++i) dq[i] = (double)codes.value[i];
                for (int64_t r = 0; r < rows; ++r)
                    for (int64_t gi = 0; gi < ngrp; ++gi) {
                        const int64_t k0 = gi * group;
                        const double sc = scale.value[(size_t)(r * ngrp + gi)];
                        for (int64_t i = 0; i < group; ++i)
                            dq[(size_t)(r * n + k0 + i)] *= sc;
                        fwht(dq.data() + r * n + k0, group);
                        for (int64_t i = 0; i < group; ++i)
                            dq[(size_t)(r * n + k0 + i)] /= (double)group;
                    }
                for (size_t i = 0; i < dq.size() && i < y[0].value.size(); ++i) {
                    const double d = std::fabs(dq[i] - y[0].value[i]);
                    if (d > deq_err) deq_err = d;
                    if (std::fabs(dq[i]) > deq_scale) deq_scale = std::fabs(dq[i]);
                }
                // The kernel's activation against the tensors around it: its own input (`op[0]`) and
                // the in-place write. Whichever it matches names what the kernel actually computed,
                // and whichever it does not names the snapshot the twin should have been given.
                double vs_in = 0.0, vs_wr = 0.0, in_vs_wr = 0.0, in_scale = 0.0;
                for (size_t i = 0; i < dq.size() && i < call.fwd[0].value.size(); ++i) {
                    const double a = std::fabs(dq[i] - call.fwd[0].value[i]);
                    if (a > vs_in) vs_in = a;
                    if (std::fabs(call.fwd[0].value[i]) > in_scale)
                        in_scale = std::fabs(call.fwd[0].value[i]);
                    if (i < call.fwd.back().value.size()) {
                        const double b = std::fabs(dq[i] - call.fwd.back().value[i]);
                        if (b > vs_wr) vs_wr = b;
                        const double c = std::fabs(call.fwd[0].value[i] - call.fwd.back().value[i]);
                        if (c > in_vs_wr) in_vs_wr = c;
                    }
                }
                // The twin's activation against the kernel's by magnitude: if the twin's is a scalar
                // multiple of the kernel's, `ymax/deqmax` is that multiple and the difference is one
                // factor (a scale or a gain), not a different tensor.
                double ymax = 0.0, dqmax = 0.0;
                for (size_t i = 0; i < dq.size() && i < y[0].value.size(); ++i) {
                    if (std::fabs(y[0].value[i]) > ymax) ymax = std::fabs(y[0].value[i]);
                    if (std::fabs(dq[i]) > dqmax) dqmax = std::fabs(dq[i]);
                }
                // Per-row magnitude ratio. A `s` that differs per row shows as a spread here; a
                // single elementwise difference (a gain vector) leaves every row's ratio the same
                // only when the max sits in the same column, so also report the ratio of the row
                // sums, which a per-row scale moves exactly.
                double rmin = 1e300, rmax = -1e300, sum_y = 0.0, sum_d = 0.0;
                for (int64_t r = 0; r < rows; ++r) {
                    double ry = 0.0, rd = 0.0;
                    for (int64_t i = 0; i < n; ++i) {
                        ry += std::fabs(y[0].value[(size_t)(r * n + i)]);
                        rd += std::fabs(dq[(size_t)(r * n + i)]);
                    }
                    sum_y += ry;
                    sum_d += rd;
                    if (rd > 0.0) {
                        const double q = ry / rd;
                        rmin = std::min(rmin, q);
                        rmax = std::max(rmax, q);
                    }
                }
                std::fprintf(stderr,
                             "DEQORACLE pos=%lld deq_vs_in=%.6g deq_vs_write=%.6g in_vs_write=%.6g "
                             "in_max=%.6g ymax=%.6g deqmax=%.6g y_over_deq=%.6g row_ratio=[%.6g,%.6g] "
                             "sum_ratio=%.6g\n",
                             (long long)call.rec.position,
                             in_scale > 0.0 ? vs_in / in_scale : vs_in,
                             in_scale > 0.0 ? vs_wr / in_scale : vs_wr,
                             in_scale > 0.0 ? in_vs_wr / in_scale : in_vs_wr, in_scale, ymax, dqmax,
                             dqmax > 0.0 ? ymax / dqmax : 0.0, rmin, rmax,
                             sum_d > 0.0 ? sum_y / sum_d : 0.0);
            }
            std::fprintf(stderr, "DUMPROT pos=%lld op=%s rot_err=%.6g rot_bad=%lld/%zu deq_rel=%.6g\n",
                         (long long)call.rec.position, call.rec.op.c_str(), rot_err,
                         (long long)rot_bad, codes.value.size(),
                         deq_scale > 0.0 ? deq_err / deq_scale : deq_err);
            if (twin_out.size() > 1 && twin_out[1].value.size() == scale.value.size())
                for (size_t i = 0; i < scale.value.size(); ++i) {
                    const double t = std::fabs(twin_out[1].value[i] - scale.value[i]);
                    const double r = std::fabs(scale.value[i]) > 0.0
                                         ? t / std::fabs(scale.value[i])
                                         : t;
                    if (r > scale_rel) scale_rel = r;
                }
        }
        // The sharper comparison: the activation the kernel wrote. A fused producer writes the
        // part's output in place (the norm's `h`, the gate's product), and that IS the intermediate
        // the twin recomputes, so the two must agree up to bf16 rounding.
        // The activation the kernel wrote, at *every* written operand of the same size: a record
        // carries the residual and the branch at the same shape, so reporting the largest error
        // would name the wrong tensor. The best match is the part's own output, and each candidate
        // is printed so the pairing is visible rather than assumed.
        double act_err = 0.0, act_scale = 0.0;
        int64_t act_at = -1;
        std::string act_all;
        for (size_t i = 0; i < call.fwd.size(); ++i) {
            const AdValue& f = call.fwd[i];
            if (i >= call.rec.operands.size() || !call.rec.operands[i].written) continue;
            if (f.value.size() != y[0].value.size()) continue;
            double e = 0.0, sc = 0.0;
            for (size_t j = 0; j < f.value.size(); ++j) {
                const double d = std::fabs((double)f.value[j] - y[0].value[j]);
                if (d > e) e = d;
                if (std::fabs((double)f.value[j]) > sc) sc = std::fabs((double)f.value[j]);
            }
            char one[96];
            std::snprintf(one, sizeof(one), "%zu:%s:%.3g ", i,
                          call.rec.operands[i].buffer.c_str(), sc > 0.0 ? e / sc : e);
            act_all += one;
            if (act_at < 0 || e < act_err) { act_err = e; act_scale = sc; act_at = (int64_t)i; }
        }
        std::fprintf(stderr,
                     "DUMPTWIN pos=%lld op=%s twin=%d codes=%zu code_err=%.6g code_bad=%lld/%zu "
                     "scale_rel=%.6g act_at=%lld act_err=%.6g act_rel=%.6g "
                     "hist=[%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld] best_of=[%s]\n",
                     (long long)call.rec.position, call.rec.op.c_str(), qt != nullptr ? 1 : 0,
                     codes.value.size(), code_err, (long long)code_bad, codes.value.size(),
                     scale_rel, (long long)act_at, act_err,
                     act_scale > 0.0 ? act_err / act_scale : act_err,
                     (long long)hist[0], (long long)hist[1], (long long)hist[2],
                     (long long)hist[3], (long long)hist[4], (long long)hist[5],
                     (long long)hist[6], (long long)hist[7], (long long)hist[8],
                     (long long)hist[9], act_all.c_str());
        std::fflush(stderr);
    }

    // The elementwise adjoint takes the parts' operands and the intermediate they wrote; it adds
    // its cotangents into the slots the parts' read operands hold. The two-way rule lives in that
    // row, so the read set the caller proved travels with it, as the device row hands its own.
    std::vector<AdValue> efwd(call.fwd.begin(),
                              call.fwd.begin() + static_cast<std::ptrdiff_t>(reads));
    efwd.push_back(y[0]);
    std::vector<AdValue> ecot = {qdc[0]};
    AdjointCall ecall{call.rec, efwd, ecot, call.dc, nullptr, ""};
    ecall.read_rows = call.read_rows;
    ecall.row_count = call.row_count;
    AdjointFn elem_adj = adjoint_find(part_op(part), Side::host, call.rec.params, nullptr);
    if (elem_adj == nullptr)
        throw std::invalid_argument(std::string("quant: no host adjoint for ") + part_op(part));
    elem_adj(ecall);

    // R97_DUMP_QUANT=1: the same stage sums the device row prints, so a host gradient that came
    // back zero names its stage instead of the whole row. `quant_dx` is the cotangent the quantiser
    // handed the elementwise part; `dx` is what the walk's slot holds after both parts ran.
    if (const char* dq = std::getenv("R97_DUMP_QUANT"); dq != nullptr && dq[0] == '1') {
        const auto sum = [](const AdValue& v) {
            double s = 0.0;
            for (double x : v.value) s += std::fabs(x);
            return s;
        };
        std::fprintf(stderr,
                     "DUMPQUANT side=host op=%s pos=%lld y=%.6g codes_cot=%.6g scale_cot=%.6g "
                     "quant_dx=%.6g dx=%.6g\n",
                     call.rec.op.c_str(), (long long)call.rec.position, sum(y[0]),
                     sum(call.cot[(size_t)codes_cot_at]), sum(call.cot[(size_t)scale_cot_at]),
                     sum(qdc[0]),
                     call.dc->empty() ? 0.0 : sum(call.dc->at(0)));
    }
}

void rmsnorm_had_quant_i8g_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                                std::vector<AdValue>* out) {
    fused_twin(rec, operands, out, Part::rmsnorm);
}
void gated_had_quant_i8g_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                              std::vector<AdValue>* out) {
    fused_twin(rec, operands, out, Part::silu_mul);
}
void gate_had_quant_i8g_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                             std::vector<AdValue>* out) {
    fused_twin(rec, operands, out, Part::sigmoid_mul);
}

void rmsnorm_had_quant_i8g_adjoint(const AdjointCall& call) { fused_adjoint(call, Part::rmsnorm); }
void gated_had_quant_i8g_adjoint(const AdjointCall& call) { fused_adjoint(call, Part::silu_mul); }
void gate_had_quant_i8g_adjoint(const AdjointCall& call) { fused_adjoint(call, Part::sigmoid_mul); }

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

// The fused int8 producers: the parts' twins and host adjoints composed, keyed by the same
// predicates the device side keys its rows with (src/adjoint/device/quant.hip). The norm's gain is
// frozen (schema position 1), so the walk asks for x's cotangent only, as the elementwise row
// does.
R97_REGISTER_TWIN_IF("rmsnorm_had_quant_i8g", rmsnorm_had_quant_i8g_twin, rotated)
R97_REGISTER_ADJOINT_IF_F("rmsnorm_had_quant_i8g", host, rmsnorm_had_quant_i8g_adjoint, rotated, 1)
R97_REGISTER_TWIN_IF("gated_had_quant_i8g", gated_had_quant_i8g_twin, serves_gated)
R97_REGISTER_ADJOINT_IF("gated_had_quant_i8g", host, gated_had_quant_i8g_adjoint, serves_gated)
R97_REGISTER_TWIN_IF("gate_had_quant_i8g", gate_had_quant_i8g_twin, serves_gate)
R97_REGISTER_ADJOINT_IF("gate_had_quant_i8g", host, gate_had_quant_i8g_adjoint, serves_gate)

// The three producers' linearised rows, for the finite-difference check. The base point is the
// activation the quantiser saw -- the part's own output -- appended after the codes and the scale,
// so one frozen index is that appended operand's schema position: 4 for the norm (`x, w, codes,
// scale, base`), 3 for the gated one (`gate_up, codes, scale, base`), 4 for the gate (`a, b, codes,
// scale, base`). The norm also freezes its gain at 1, as the plain row above does: the walk never
// asks for it, so the row leaves that cotangent at zero and the finite differences must not measure
// it either (`check_adjoint` skips frozen slots). The quantiser the composition reaches is then its
// own `_ste` row, which reads the base point as its second operand.
R97_REGISTER_TWIN_IF("rmsnorm_had_quant_i8g_ste", rmsnorm_had_quant_i8g_twin, rotated)
R97_REGISTER_ADJOINT_IF_F("rmsnorm_had_quant_i8g_ste", host, rmsnorm_had_quant_i8g_adjoint, rotated, 1, 4)
R97_REGISTER_TWIN_IF("gated_had_quant_i8g_ste", gated_had_quant_i8g_twin, serves_gated)
R97_REGISTER_ADJOINT_IF_F("gated_had_quant_i8g_ste", host, gated_had_quant_i8g_adjoint, serves_gated, 3)
R97_REGISTER_TWIN_IF("gate_had_quant_i8g_ste", gate_had_quant_i8g_twin, serves_gate)
R97_REGISTER_ADJOINT_IF_F("gate_had_quant_i8g_ste", host, gate_had_quant_i8g_adjoint, serves_gate, 4)
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
