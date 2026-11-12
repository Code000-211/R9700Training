// The host twin and host adjoint of quant_act_fp8: a per-group 8-bit code and scale of an
// activation that is already in whatever dtype the engine computed it in (the record's input may
// be bf16 or, through a view, a flat [rows, n] slice).
//
// This is the quant family's rule (docs/adjoints/quant.md) at fp8's top: the scale takes no
// cotangent, the code's cotangent is divided by the scale, and a code at the clamp carries none.
// The code is fp8_e4m3 (top 448) or i8 when the plugin feeds an integer GEMM (top 127).
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

bool is_fp8(const std::string& dtype) {
    return dtype == "fp8_e4m3" || dtype == "fp8_e5m2" || dtype == "f8";
}

double round8(double u, const std::string& dtype) {
    if (is_fp8(dtype)) {
        if (dtype == "fp8_e5m2") return (double)rad_fp8e5m2_to_f32(rad_f32_to_fp8e5m2((float)u));
        return (double)rad_fp8e4m3_to_f32(rad_f32_to_fp8e4m3((float)u));
    }
    return std::nearbyint(u);
}

struct Geometry {
    int64_t rows = 0, n = 0, group = 1, ng = 1;
    std::string qdtype;
    double top = 448.0;

    Geometry(const OpRecord& rec, const std::vector<AdValue>& in) {
        n = param_int(rec.params, "n", in.empty() ? 0 : in[0].shape.back());
        if (n <= 0) n = 1;
        int64_t total = 0;
        for (int64_t d : in[0].shape) total = total == 0 ? d : total * d;
        if (total < n) total = n;
        rows = total / n;
        group = param_int(rec.params, "group", n);
        if (group <= 0 || group > n) group = n;
        ng = n > 0 ? (n + group - 1) / group : 0;
        qdtype = rec.operands[1].dtype;
        top = qdtype == "i8" ? 127.0 : 448.0;
    }
};

double scale_at(const AdValue& s, int64_t row, int64_t gi, int64_t ng) {
    if (s.value.empty()) return 1.0;
    const int64_t idx = s.shape.size() >= 2 ? row * ng + gi : row;
    if (idx < 0 || (size_t)idx >= s.value.size()) return 1.0;
    return s.value[(size_t)idx];
}

void quant_twin(const OpRecord& rec, const std::vector<AdValue>& in, std::vector<AdValue>* out) {
    const Geometry g(rec, in);
    std::vector<double> q((size_t)(g.rows * g.n), 0.0), s((size_t)(g.rows * g.ng), 1.0);
    for (int64_t r = 0; r < g.rows; ++r)
        for (int64_t gi = 0; gi < g.ng; ++gi) {
            const int64_t lo = gi * g.group, hi = std::min(g.n, lo + g.group);
            double amax = 0.0;
            for (int64_t i = lo; i < hi; ++i)
                amax = std::max(amax, std::fabs(in[0].value[(size_t)(r * g.n + i)]));
            const double sc = amax > 0.0 ? amax / g.top : 1.0;
            s[(size_t)(r * g.ng + gi)] = sc;
            for (int64_t i = lo; i < hi; ++i) {
                const size_t e = (size_t)(r * g.n + i);
                q[e] = round8(std::fmax(-g.top, std::fmin(g.top, in[0].value[e] / sc)), g.qdtype);
            }
        }
    AdValue qv;
    qv.dtype = g.qdtype;
    qv.shape = {g.rows, g.n};
    qv.value = std::move(q);
    AdValue sv;
    sv.dtype = "f32";
    sv.shape = {g.rows, g.ng};
    sv.value = std::move(s);
    out->push_back(std::move(qv));
    out->push_back(std::move(sv));
}

void quant_adjoint(const AdjointCall& call) {
    const Geometry g(call.rec, call.fwd);
    const AdValue& s = call.fwd[2];
    const AdValue& dq = call.cot[0];
    AdValue& dx = call.dc->at(0);
    for (int64_t r = 0; r < g.rows; ++r)
        for (int64_t gi = 0; gi < g.ng; ++gi) {
            const double sc = scale_at(s, r, gi, g.ng);
            const int64_t lo = gi * g.group, hi = std::min(g.n, lo + g.group);
            for (int64_t i = lo; i < hi; ++i) {
                const size_t e = (size_t)(r * g.n + i);
                const double u = call.fwd[0].value[e] / sc;
                if (std::fabs(u) > g.top) continue;
                dx.value[e] += dq.value[e] / sc;
            }
        }
}

}  // namespace

R97_REGISTER_TWIN("quant_act_fp8", quant_twin)
R97_REGISTER_ADJOINT("quant_act_fp8", host, quant_adjoint)

}  // namespace r97
