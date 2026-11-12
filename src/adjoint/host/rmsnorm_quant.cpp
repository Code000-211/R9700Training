// The host twin and host adjoint of rmsnorm_quant_fp8.
//
// The op adds the residual in place, normalises the sum, narrows `src * rs * (w + wadd)` to bf16,
// writes that value as `out_bf16` when asked, and quantises it per group to an 8-bit code with a
// scale (fp8_e4m3, top 448; or i8 when the code feeds an integer GEMM, top 127).
//
// The adjoint is the rmsnorm family's (docs/adjoints/qk.md) composed with the quantiser's
// straight-through rule (docs/adjoints/quant.md): the scale takes no cotangent, the code's
// cotangent is divided by it, and the bf16 narrowing is the identity. The residual is in place and
// is an input to the norm as well as a written value, so its old value takes the sum's whole
// cotangent: the write's cotangent from the readers plus the norm's own path.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rad_plugin.h"
#include "r97/adjoint.h"

namespace r97 {
namespace {

int64_t param_int(const std::vector<OpParam>& params, const char* name, int64_t fallback) {
    for (const OpParam& p : params) {
        if (p.first != name) continue;
        if (const int64_t* i = std::get_if<int64_t>(&p.second)) return *i;
        if (const std::string* s = std::get_if<std::string>(&p.second)) {
            char* end = nullptr;
            const long long v = std::strtoll(s->c_str(), &end, 10);
            if (end && *end == '\0') return (int64_t)v;
        }
    }
    return fallback;
}

double param_f64(const std::vector<OpParam>& params, const char* name, double fallback) {
    for (const OpParam& p : params) {
        if (p.first != name) continue;
        if (const int64_t* i = std::get_if<int64_t>(&p.second)) return (double)*i;
        if (const std::string* s = std::get_if<std::string>(&p.second)) {
            char* end = nullptr;
            const double v = std::strtod(s->c_str(), &end);
            if (end && *end == '\0') return v;
        }
    }
    return fallback;
}

bool is_fp8(const std::string& dtype) {
    return dtype == "fp8_e4m3" || dtype == "fp8_e5m2" || dtype == "f8";
}

double bf16(double x) {
    return (double)rad_bf16_to_f32(rad_f32_to_bf16((float)x));
}

double round8(double u, const std::string& dtype) {
    if (is_fp8(dtype)) {
        if (dtype == "fp8_e5m2") return (double)rad_fp8e5m2_to_f32(rad_f32_to_fp8e5m2((float)u));
        return (double)rad_fp8e4m3_to_f32(rad_f32_to_fp8e4m3((float)u));
    }
    return std::nearbyint(u);
}

// Where the schema's operands sit: the residual is optional and, when present, is the in-place
// operand at position 1; the rest follow it.
// The engine's record always has the residual's slot at position 1, even when the block has no
// residual: the slot is then a written, empty placeholder. A record a test builds may leave the
// slot out; then the weight follows the activation directly.
struct Layout {
    bool res = false, res_slot = false, out = false;
    int64_t rows = 0, n = 0, group = 1, ng = 1;
    double eps = 1e-6, wadd = 0.0;
    size_t w = 0, q = 0, s = 0, ob = 0;
    std::string qdtype;
    double top = 448.0;

    Layout(const OpRecord& rec, const std::vector<AdValue>& in)
        : eps(param_f64(rec.params, "eps", 1e-6)), wadd(param_f64(rec.params, "wadd", 0.0)) {
        n = param_int(rec.params, "n", in.empty() || in[0].shape.empty() ? 0 : in[0].shape.back());
        rows = in.empty() || n <= 0 ? 0 : (int64_t)(in[0].value.size() / (size_t)n);
        group = param_int(rec.params, "group", n);
        if (group <= 0 || group > n) group = n;
        ng = n > 0 ? (n + group - 1) / group : 0;
        const bool has_res_slot = rec.operands.size() >= 6;
        // The residual's write takes a cotangent when the record says the op writes it; a record
        // that only reads it (the finite-difference check) keeps the same operand positions but
        // has no write cotangent.
        res_slot = has_res_slot && rec.operands[1].written;
        res = in.size() > 2 && in[1].value.size() >= (size_t)(rows * n) &&
              in[1].value.size() == in[0].value.size();
        if (has_res_slot) {
            w = 2;
            q = 3;
            s = 4;
            ob = 5;
            out = rec.operands.size() > ob && rec.operands[ob].written;
        } else {
            w = 1;
            q = 2;
            s = 3;
            ob = 4;
            out = rec.operands.size() > ob && rec.operands[ob].written;
        }
        qdtype = rec.operands[q].dtype;
        top = qdtype == "i8" ? 127.0 : 448.0;
    }
};

double scale_at(const AdValue& s, int64_t row, int64_t gi, int64_t ng) {
    if (s.value.empty()) return 1.0;
    const int64_t idx = s.shape.size() >= 2 ? row * ng + gi : row;
    if (idx < 0 || (size_t)idx >= s.value.size()) return 1.0;
    return s.value[(size_t)idx];
}

// The sum the norm reads, row by row: the residual's old value plus x when there is one, narrowed
// the way the kernel's store narrows it.
std::vector<double> source(const OpRecord& rec, const std::vector<AdValue>& in, const Layout& l) {
    std::vector<double> src((size_t)(l.rows * l.n));
    const AdValue& x = in[0];
    const AdValue* r = l.res ? &in[1] : nullptr;
    for (size_t i = 0; i < src.size(); ++i)
        src[i] = r ? bf16(x.value[i] + r->value[i]) : x.value[i];
    return src;
}

// src -> (normalised, narrowed) values.
std::vector<double> normalised(const std::vector<double>& src, const AdValue& w, const Layout& l,
                               std::vector<double>* rs_out = nullptr) {
    std::vector<double> y(src.size());
    for (int64_t r = 0; r < l.rows; ++r) {
        double ss = 0.0;
        for (int64_t i = 0; i < l.n; ++i) {
            const double v = src[(size_t)(r * l.n + i)];
            ss += v * v;
        }
        const double rs = 1.0 / std::sqrt(ss / (double)l.n + l.eps);
        if (rs_out) rs_out->push_back(rs);
        for (int64_t i = 0; i < l.n; ++i) {
            const double s = src[(size_t)(r * l.n + i)];
            y[(size_t)(r * l.n + i)] = bf16(s * rs * (w.value[(size_t)i] + l.wadd));
        }
    }
    return y;
}

void rmsnorm_twin(const OpRecord& rec, const std::vector<AdValue>& in, std::vector<AdValue>* out) {
    const Layout l(rec, in);
    const std::vector<double> src = source(rec, in, l);
    const std::vector<double> y = normalised(src, in.back(), l);

    std::vector<double> q((size_t)(l.rows * l.n)), s((size_t)(l.rows * l.ng), 1.0);
    for (int64_t r = 0; r < l.rows; ++r)
        for (int64_t gi = 0; gi < l.ng; ++gi) {
            const int64_t lo = gi * l.group, hi = std::min(l.n, lo + l.group);
            double amax = 0.0;
            for (int64_t i = lo; i < hi; ++i)
                amax = std::max(amax, std::fabs(y[(size_t)(r * l.n + i)]));
            const double sc = amax > 0.0 ? amax / l.top : 1.0;
            s[(size_t)(r * l.ng + gi)] = sc;
            for (int64_t i = lo; i < hi; ++i) {
                const size_t e = (size_t)(r * l.n + i);
                q[e] = round8(std::fmax(-l.top, std::fmin(l.top, y[e] / sc)), l.qdtype);
            }
        }

    AdValue qv;
    qv.dtype = l.qdtype;
    qv.shape = {l.rows, l.n};
    qv.value = std::move(q);
    AdValue sv;
    sv.dtype = "f32";
    sv.shape = {l.rows, l.ng};
    sv.value = std::move(s);
    out->push_back(std::move(qv));
    out->push_back(std::move(sv));
    if (l.out) {
        AdValue ov;
        ov.dtype = "bf16";
        ov.shape = {l.rows, l.n};
        ov.value = y;
        out->push_back(std::move(ov));
    }
}

void rmsnorm_adjoint(const AdjointCall& call) {
    const Layout l(call.rec, call.fwd);
    const AdValue& x = call.fwd[0];
    const AdValue* res = l.res ? &call.fwd[1] : nullptr;
    const AdValue& w = call.fwd[l.w];
    const AdValue& s = call.fwd[l.s];

    size_t c = 0;
    // The residual's slot takes a cotangent even when it is the engine's empty placeholder.
    const AdValue* dres = l.res_slot ? &call.cot[c++] : nullptr;
    if (!l.res) dres = nullptr;
    const AdValue& dq = call.cot[c++];
    ++c;  // the scale takes no cotangent
    const AdValue* dout = l.out ? &call.cot[c++] : nullptr;

    // A linearised record carries the base point after its own operands; the Jacobian is taken
    // there, so the norm's source, gain and mask come from it rather than from the perturbed
    // reads.
    const bool ste = call.rec.op == "rmsnorm_quant_fp8_ste" && call.fwd.size() > 8;
    std::vector<AdValue> base_in;
    Layout bl = l;
    const AdValue* bw = &w;
    if (ste) {
        base_in = {call.fwd[6], call.fwd[7], call.fwd[8]};
        bl.n = l.n;
        bl.rows = l.rows;
        bl.group = l.group;
        bl.ng = l.ng;
        bl.eps = l.eps;
        bl.wadd = l.wadd;
        bl.res = true;
        bl.w = 2;
        bw = &call.fwd[8];
    }
    const std::vector<double> src = ste ? source(call.rec, base_in, bl) : source(call.rec, call.fwd, l);
    std::vector<double> rs;
    const std::vector<double> y = normalised(src, *bw, ste ? bl : l, &rs);

    AdValue& dx = call.dc->at(0);
    AdValue& dr = l.res ? call.dc->at(1) : dx;
    AdValue& dw = call.dc->at(l.w);

    std::vector<double> tot(src.size());  // the sum's cotangent, both paths
    for (int64_t r = 0; r < l.rows; ++r) {
        // The norm's path: t_i is the cotangent of src_i * rs * gain_i before the gain.
        std::vector<double> t((size_t)l.n, 0.0);
        double d_rs = 0.0;
        for (int64_t gi = 0; gi < l.ng; ++gi) {
            const double sc = scale_at(s, r, gi, l.ng);
            const int64_t lo = gi * l.group, hi = std::min(l.n, lo + l.group);
            for (int64_t i = lo; i < hi; ++i) {
                const size_t e = (size_t)(r * l.n + i);
                const double live = std::fabs(y[e] / sc) <= l.top ? 1.0 : 0.0;
                double d = live * dq.value[e] / sc;
                if (dout) d += dout->value[e];
                t[(size_t)i] = d;
                d_rs += d * src[e] * (bw->value[(size_t)i] + l.wadd);
            }
        }
        const double rsv = rs[(size_t)r];
        for (int64_t i = 0; i < l.n; ++i) {
            const double gain = bw->value[(size_t)i] + l.wadd;
            const double dsrc = t[(size_t)i] * rsv * gain - (rsv * rsv * rsv / (double)l.n) *
                                                                  src[(size_t)(r * l.n + i)] * d_rs;
            dw.value[(size_t)i] += t[(size_t)i] * rsv * src[(size_t)(r * l.n + i)];
            const size_t e = (size_t)(r * l.n + i);
            tot[e] = dsrc + (dres ? dres->value[e] : 0.0);
            dx.value[e] += tot[e];
            if (l.res) dr.value[e] += tot[e];
        }
    }
}

// The linearised twin: it reads the base point as three frozen operands after the schema's own
// (x0, res0, w0) and writes the base's code and value plus the Jacobian's action from that point.
// The Jacobian is the norm's, with the bf16 narrowing and the quantiser straight through.
void rmsnorm_ste_twin(const OpRecord& rec, const std::vector<AdValue>& in, std::vector<AdValue>* out) {
    const Layout l(rec, in);
    // The base point's reads: the record's reads are x, res, w, x0, res0, w0.
    const size_t nb = in.size();
    if (nb < 6) {
        out->clear();
        return;
    }
    const AdValue& x0 = in[nb - 3];
    const AdValue& r0 = in[nb - 2];
    const AdValue& w0 = in[nb - 1];

    std::vector<AdValue> base_in{x0, r0, w0};
    Layout bl(rec, base_in);
    bl.n = l.n;
    bl.rows = l.rows;
    bl.group = l.group;
    bl.ng = l.ng;
    bl.eps = l.eps;
    bl.wadd = l.wadd;
    bl.res = true;
    bl.w = 2;
    bl.q = 3;
    bl.s = 4;
    bl.ob = 5;
    const std::vector<double> src = source(rec, base_in, bl);
    std::vector<double> rs;
    const std::vector<double> y = normalised(src, w0, bl, &rs);

    // The base's code, scale and mask.
    std::vector<double> q0((size_t)(l.rows * l.n)), s0((size_t)(l.rows * l.ng), 1.0);
    std::vector<double> live((size_t)(l.rows * l.n), 0.0);
    for (int64_t r = 0; r < l.rows; ++r)
        for (int64_t gi = 0; gi < l.ng; ++gi) {
            const int64_t lo = gi * l.group, hi = std::min(l.n, lo + l.group);
            double amax = 0.0;
            for (int64_t i = lo; i < hi; ++i)
                amax = std::max(amax, std::fabs(y[(size_t)(r * l.n + i)]));
            const double sc = amax > 0.0 ? amax / l.top : 1.0;
            s0[(size_t)(r * l.ng + gi)] = sc;
            for (int64_t i = lo; i < hi; ++i) {
                const size_t e = (size_t)(r * l.n + i);
                const double u = y[e] / sc;
                live[e] = std::fabs(u) <= l.top ? 1.0 : 0.0;
                q0[e] = round8(std::fmax(-l.top, std::fmin(l.top, u)), l.qdtype);
            }
        }

    // The Jacobian's action on the difference from the base.
    std::vector<double> q(q0.size()), s((size_t)(l.rows * l.ng), 0.0), ob(y.size(), 0.0);
    for (int64_t r = 0; r < l.rows; ++r) {
        const double rsv = rs[(size_t)r];
        double d_rs = 0.0;
        for (int64_t i = 0; i < l.n; ++i) {
            const size_t e = (size_t)(r * l.n + i);
            const double dsrc = (in[0].value[e] - x0.value[e]) + (in[1].value[e] - r0.value[e]);
            d_rs += src[e] * dsrc;
        }
        d_rs *= -(rsv * rsv * rsv / (double)l.n);
        for (int64_t i = 0; i < l.n; ++i) {
            const size_t e = (size_t)(r * l.n + i);
            const double dsrc = (in[0].value[e] - x0.value[e]) + (in[1].value[e] - r0.value[e]);
            const double gain = w0.value[(size_t)i] + l.wadd;
            const double dy = dsrc * rsv * gain + src[e] * d_rs * gain +
                              src[e] * rsv * (in[2].value[(size_t)i] - w0.value[(size_t)i]);
            q[e] = q0[e] + live[e] * dy / s0[(size_t)(r * l.ng + i / l.group)];
            ob[e] = y[e] + dy;
        }
        for (int64_t gi = 0; gi < l.ng; ++gi) s[(size_t)(r * l.ng + gi)] = s0[(size_t)(r * l.ng + gi)];
    }
    AdValue qv;
    qv.dtype = l.qdtype;
    qv.shape = {l.rows, l.n};
    qv.value = std::move(q);
    AdValue sv;
    sv.dtype = "f32";
    sv.shape = {l.rows, l.ng};
    sv.value = std::move(s);
    out->push_back(std::move(qv));
    out->push_back(std::move(sv));
    if (l.out) {
        AdValue ov;
        ov.dtype = "bf16";
        ov.shape = {l.rows, l.n};
        ov.value = std::move(ob);
        out->push_back(std::move(ov));
    }
}

}  // namespace

R97_REGISTER_TWIN("rmsnorm_quant_fp8", rmsnorm_twin)
R97_REGISTER_ADJOINT("rmsnorm_quant_fp8", host, rmsnorm_adjoint)
R97_REGISTER_TWIN("rmsnorm_quant_fp8_ste", rmsnorm_ste_twin)
R97_REGISTER_ADJOINT_F("rmsnorm_quant_fp8_ste", host, rmsnorm_adjoint, 6, 7, 8)

}  // namespace r97
