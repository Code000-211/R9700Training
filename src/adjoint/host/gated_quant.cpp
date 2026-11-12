// The host twin and host adjoint of gated_quant_fp8.
//
// The op is one fused pass: silu(gate) * up, narrowed to bf16, quantised per group of
// `group` elements to an 8-bit code with a scale, and the bf16 value kept beside it. The code is
// fp8_e4m3 (`top` = 448), or i8 when the plugin feeds an integer GEMM (`top` = 127). The group
// scale and the code's rounding are straight-through: the adjoint returns the cotangent of the
// pre-quant value divided by the scale, and the scale operand takes none (docs/adjoints/quant.md).
// The bf16 value the op writes is the same pre-quant value, so its cotangent is added whole.
//
// The linearised twin (`gated_quant_fp8_ste`) keeps the scale and the bf16 narrowing at the base
// point and writes the value as `v0 + (p - p0)`, so finite differences see the slope the adjoint
// claims. At the base point it agrees with the forward: v0 is already bf16.
#include <cmath>
#include <cstdint>
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

const std::string* param_str(const std::vector<OpParam>& params, const char* name) {
    for (const OpParam& p : params) {
        if (p.first != name) continue;
        if (const std::string* s = std::get_if<std::string>(&p.second)) return s;
    }
    return nullptr;
}

bool is_fp8(const std::string& dtype) {
    return dtype == "fp8_e4m3" || dtype == "fp8_e5m2" || dtype == "f8";
}

// The op's activation and its derivative, by the parameters the kernel reads: `act` names it
// ("silu", "sigmoid"), `mode` is the integer spelling libr4d passes.
struct Act {
    bool sigmoid = false;
    double operator()(double x) const {
        const double s = 1.0 / (1.0 + std::exp(-x));
        return sigmoid ? s : x * s;
    }
    double deriv(double x) const {
        const double s = 1.0 / (1.0 + std::exp(-x));
        return sigmoid ? s * (1.0 - s) : s * (1.0 + x * (1.0 - s));
    }
};

Act act_of(const std::vector<OpParam>& params) {
    Act a;
    const std::string* name = param_str(params, "act");
    a.sigmoid = name ? *name == "sigmoid" : param_int(params, "mode", 0) == 1;
    return a;
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

// The value the op holds per output element: silu(gate) * up in f64.
std::vector<double> gated(const AdValue& gate_up, int64_t n, Act act) {
    const int64_t rows = gate_up.shape[0];
    std::vector<double> p((size_t)(rows * n));
    const int64_t w = 2 * n;
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t i = 0; i < n; ++i) {
            const size_t g = (size_t)(r * w + i), u = (size_t)(r * w + n + i);
            p[(size_t)(r * n + i)] = act(gate_up.value[g]) * gate_up.value[u];
        }
    return p;
}

// The group count and that group's scale, from the s operand and the parameters.
struct Groups {
    int64_t n = 0, group = 1, ng = 1;
    explicit Groups(const OpRecord& rec, int64_t n_) : n(n_) {
        group = param_int(rec.params, "group", n);
        if (group <= 0) group = n;
        ng = n > 0 ? (n + group - 1) / group : 0;
    }
    int64_t of(int64_t col) const { return group > 0 ? col / group : 0; }
    int64_t begin(int64_t gi) const { return gi * group; }
    int64_t end() const { return n; }
};

double scale_at(const AdValue& s, int64_t row, int64_t gi, int64_t ng) {
    if (s.value.empty()) return 1.0;
    const int64_t idx = s.shape.size() >= 2 ? row * ng + gi : row;
    if (idx < 0 || (size_t)idx >= s.value.size()) return 1.0;
    return s.value[(size_t)idx];
}

// What the op writes, in schema order: the code, the scale, and the bf16 value when the op has it.
struct Layout {
    int64_t rows = 0, n = 0;
    bool has_out = false;
    bool ste = false;
    size_t q = 0, s = 0, ob = 0;
    int64_t n_writes() const { return has_out ? 3 : 2; }

    Layout(const OpRecord& rec, const std::vector<AdValue>& in) {
        rows = in.empty() ? 0 : in[0].shape[0];
        n = param_int(rec.params, "n", 0);
        if (n <= 0 && !in.empty() && in[0].shape.size() >= 2) n = in[0].shape[1] / 2;
        ste = rec.op == "gated_quant_fp8_ste";
        q = 0;
        while (q < rec.operands.size() && !rec.operands[q].written) ++q;
        s = q + 1;
        ob = s + 1;
        has_out = rec.operands.size() > ob && rec.operands[ob].written;
    }
};

// The forward in f64, at the schema's read operands. The code's value is stored decoded: fp8
// dtypes hold a number, i8 ones an integer code, the same convention the other twins use.
void gated_twin(const OpRecord& rec, const std::vector<AdValue>& in, std::vector<AdValue>* out) {
    Layout lay(rec, in);
    Groups grp(rec, lay.n);
    const Act act = act_of(rec.params);
    const std::string qdtype = rec.operands[lay.q].dtype;
    const double top = qdtype == "i8" ? 127.0 : 448.0;

    std::vector<double> v = gated(in[0], lay.n, act);
    std::vector<double> s((size_t)(lay.rows * grp.ng), 1.0);
    std::vector<double> code((size_t)(lay.rows * lay.n), 0.0);

    // The base point of the linearised twin, when present.
    std::vector<double> v0, s0, q0, live;
    if (lay.ste) {
        const std::vector<double> p0 = gated(in[1], lay.n, act);
        v0.resize(p0.size());
        for (size_t i = 0; i < p0.size(); ++i) v0[i] = bf16(p0[i]);
        s0.assign(s.size(), 1.0);
        q0.assign(code.size(), 0.0);
        live.assign(code.size(), 1.0);
        for (int64_t r = 0; r < lay.rows; ++r)
            for (int64_t gi = 0; gi < grp.ng; ++gi) {
                double amax = 0.0;
                for (int64_t i = grp.begin(gi); i < std::min(grp.end(), grp.begin(gi) + grp.group); ++i)
                    amax = std::max(amax, std::fabs(v0[(size_t)(r * lay.n + i)]));
                s0[(size_t)(r * grp.ng + gi)] = amax > 0.0 ? amax / top : 1.0;
                const double sc = s0[(size_t)(r * grp.ng + gi)];
                for (int64_t i = grp.begin(gi); i < std::min(grp.end(), grp.begin(gi) + grp.group); ++i) {
                    const size_t e = (size_t)(r * lay.n + i);
                    const double u = v0[e] / sc;
                    live[e] = std::fabs(u) <= top ? 1.0 : 0.0;
                    q0[e] = round8(std::fmax(-top, std::fmin(top, u)), qdtype);
                }
            }
    }

    for (int64_t r = 0; r < lay.rows; ++r) {
        for (int64_t gi = 0; gi < grp.ng; ++gi) {
            double amax = 0.0;
            for (int64_t i = grp.begin(gi); i < std::min(grp.end(), grp.begin(gi) + grp.group); ++i)
                amax = std::max(amax, std::fabs(v[(size_t)(r * lay.n + i)]));
            const double sc = amax > 0.0 ? amax / top : 1.0;
            s[(size_t)(r * grp.ng + gi)] = sc;
            for (int64_t i = grp.begin(gi); i < std::min(grp.end(), grp.begin(gi) + grp.group); ++i) {
                const size_t e = (size_t)(r * lay.n + i);
                const double u = std::fmax(-top, std::fmin(top, v[e] / sc));
                code[e] = round8(u, qdtype);
            }
        }
    }

    AdValue q;
    q.dtype = qdtype;
    q.shape = {(int64_t)lay.rows, lay.n};
    q.value.resize((size_t)(lay.rows * lay.n));
    AdValue sc;
    sc.dtype = "f32";
    sc.shape = {(int64_t)lay.rows, grp.ng};
    sc.value = s;
    std::vector<AdValue> writes;
    writes.push_back(std::move(q));
    writes.push_back(std::move(sc));
    if (lay.has_out) {
        AdValue o;
        o.dtype = "bf16";
        o.shape = {(int64_t)lay.rows, lay.n};
        o.value.resize((size_t)(lay.rows * lay.n));
        writes.push_back(std::move(o));
    }

    // The linearised twin writes the base's code and value plus the slope from the base point:
    // the same slope the adjoint claims. At the base point it is the forward.
    const std::vector<double> p0 = lay.ste ? gated(in[1], lay.n, act) : std::vector<double>{};
    for (int64_t r = 0; r < lay.rows; ++r)
        for (int64_t gi = 0; gi < grp.ng; ++gi) {
            const double sc = lay.ste ? s0[(size_t)(r * grp.ng + gi)] : s[(size_t)(r * grp.ng + gi)];
            for (int64_t i = grp.begin(gi); i < std::min(grp.end(), grp.begin(gi) + grp.group); ++i) {
                const size_t e = (size_t)(r * lay.n + i);
                if (lay.ste) {
                    writes[0].value[e] = q0[e] + live[e] * (v[e] - p0[e]) / sc;
                    if (lay.has_out) writes[2].value[e] = v0[e] + (v[e] - p0[e]);
                } else {
                    writes[0].value[e] = code[e];
                    if (lay.has_out) writes[2].value[e] = v[e];
                }
            }
        }
    if (lay.ste) writes[1].value = s0;
    *out = std::move(writes);
}

void gated_adjoint(const AdjointCall& call) {
    const Layout lay(call.rec, call.fwd);
    Groups grp(call.rec, lay.n);
    const Act act = act_of(call.rec.params);
    const std::string qdtype = call.rec.operands[lay.q].dtype;
    const double top = qdtype == "i8" ? 127.0 : 448.0;
    const AdValue& gate_up = call.fwd[0];
    const AdValue& s = call.fwd[lay.s];
    const AdValue& vf = call.fwd[lay.ob];
    const AdValue& dq = call.cot[0];
    const AdValue& dout = lay.has_out ? call.cot[2] : AdValue{};
    AdValue& dgate = call.dc->at(0);

    // The value the mask reads: the bf16 operand when the op writes it, else recomputed.
    std::vector<double> own;
    if (!lay.has_out && !lay.ste) {
        own = gated(gate_up, lay.n, act);
        for (double& x : own) x = bf16(x);
    }
    // The linearised twin's base point: the mask is the one its code was written with.
    std::vector<double> base;
    if (lay.ste) {
        base = gated(call.fwd[1], lay.n, act);
        for (double& x : base) x = bf16(x);
    }

    for (int64_t r = 0; r < lay.rows; ++r)
        for (int64_t gi = 0; gi < grp.ng; ++gi) {
            const double sc = scale_at(s, r, gi, grp.ng);
            for (int64_t i = grp.begin(gi); i < std::min(grp.end(), grp.begin(gi) + grp.group); ++i) {
                const size_t e = (size_t)(r * lay.n + i);
                const double v = lay.ste ? base[e] : (lay.has_out ? vf.value[e] : own[e]);
                const double live = std::fabs(v / sc) <= top ? 1.0 : 0.0;
                double d = live * dq.value[e] / sc;
                if (lay.has_out) d += dout.value[e];
                const size_t g = (size_t)(r * 2 * lay.n + i), u = g + (size_t)lay.n;
                dgate.value[g] += d * gate_up.value[u] * act.deriv(gate_up.value[g]);
                dgate.value[u] += d * act(gate_up.value[g]);
            }
        }
}

}  // namespace

R97_REGISTER_TWIN("gated_quant_fp8_ste", gated_twin)
R97_REGISTER_ADJOINT_F("gated_quant_fp8_ste", host, gated_adjoint, 1)
R97_REGISTER_TWIN("gated_quant_fp8", gated_twin)
R97_REGISTER_ADJOINT("gated_quant_fp8", host, gated_adjoint)

}  // namespace r97
