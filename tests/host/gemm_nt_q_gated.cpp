// tests/host/gemm_nt_q_gated.cpp -- the gated narrow fold's host twin and adjoint.
//
// K = 256, N = 512, M = 4, group = 128: the fold's schema at a size the band serves, out_bf16
// bound and unbound. The twin is held to a plain f64 reference of the same chain -- the grouped
// GEMM, the gate, the rotation and the quantiser -- and the adjoint to the reference's
// straight-through cotangents at 1e-9 relative: the activation's, in the codes' basis (the row
// applies the activation's own group scale), the weight's in its codes and its scale's per
// (row, K block). The seed carries nonzero
// cotangents on the codes and on out_bf16 at once, and on the scale, which the row drops. A gate
// that loses its factor on the gate half's cotangent must be seen. Label: host.
#include "support/adjoint_check.h"
#include "support/test.h"

#include "r97/adjoint.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace r97;
using namespace r97::adjoint_check;

namespace {

constexpr int64_t kM = 4, kN = 512, kK = 256, kGroup = 128;
constexpr int64_t kFused = kN / 2;
constexpr int64_t kGroupsK = kK / kGroup;
constexpr int64_t kGroupsN = kFused / kGroup;
constexpr double kTol = 1e-9;

OpRecord make_record(bool bound) {
    OpRecord rec;
    rec.op = "gemm_nt_q_gated";
    rec.params = {{"M", kM},
                  {"N", kN},
                  {"K", kK},
                  {"group", kGroup},
                  {"dtype", std::string("i8a8h")},
                  {"act", std::string("silu")}};
    const auto add = [&](const char* name, const char* dtype, int64_t rows, int64_t cols,
                         bool written) {
        Operand o;
        o.buffer = name;
        o.dtype = dtype;
        o.shape = {rows, cols};
        o.written = written;
        rec.operands.push_back(std::move(o));
    };
    add("a", "i8", kM, kK, false);
    add("a_scale", "f32", kM, kGroupsK, false);
    add("b", "i8", kN, kK, false);
    add("b_scale", "bf16", kN, kGroupsK, false);
    add("q", "i8", kM, kFused, true);
    add("scale", "f32", kM, kGroupsN, true);
    if (bound) add("out_bf16", "bf16", kM, kFused, true);
    return rec;
}

AdValue av(const std::string& dtype, std::vector<int64_t> shape, std::vector<double> value) {
    AdValue v;
    v.dtype = dtype;
    v.shape = std::move(shape);
    v.value = std::move(value);
    return v;
}

// The unnormalised Walsh-Hadamard of one block, as the kernel and the row do it.
void fwht(double* v, int64_t t) {
    for (int64_t h = 1; h < t; h <<= 1)
        for (int64_t i0 = 0; i0 < t; i0 += 2 * h)
            for (int64_t j = i0; j < i0 + h; ++j) {
                const double a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
}

double silu(double g) { return g / (1.0 + std::exp(-g)); }

double silu_prime(double g) {
    const double s = 1.0 / (1.0 + std::exp(-g));
    return s * (1.0 + g * (1.0 - s));
}

// The plain f64 forward: the grouped GEMM, the gate, the rotation and the quantiser.
struct Ref {
    std::vector<double> zg, zu, p;  // the halves and the product, [M, n]
    std::vector<double> v, code;    // the rotated product and its codes, [M, n]
    std::vector<double> scale;      // [M, n / group]
};

Ref ref_forward(const std::vector<double>& a, const std::vector<double>& as,
                const std::vector<double>& b, const std::vector<double>& bs) {
    Ref r;
    r.zg.assign(kM * kFused, 0.0);
    r.zu.assign(kM * kFused, 0.0);
    for (int64_t m = 0; m < kM; ++m) {
        for (int64_t j = 0; j < kFused; ++j) {
            double g_acc = 0.0, u_acc = 0.0;
            for (int64_t g = 0; g < kGroupsK; ++g) {
                const int64_t k0 = g * kGroup;
                double s_g = 0.0, s_u = 0.0;
                for (int64_t k = k0; k < k0 + kGroup; ++k) {
                    s_g += a[static_cast<size_t>(m * kK + k)] * b[static_cast<size_t>(j * kK + k)];
                    s_u += a[static_cast<size_t>(m * kK + k)] *
                           b[static_cast<size_t>((kFused + j) * kK + k)];
                }
                g_acc += s_g * as[static_cast<size_t>(m * kGroupsK + g)] *
                         bs[static_cast<size_t>(j * kGroupsK + g)];
                u_acc += s_u * as[static_cast<size_t>(m * kGroupsK + g)] *
                         bs[static_cast<size_t>((kFused + j) * kGroupsK + g)];
            }
            r.zg[static_cast<size_t>(m * kFused + j)] = g_acc;
            r.zu[static_cast<size_t>(m * kFused + j)] = u_acc;
        }
    }
    r.p.resize(kM * kFused);
    for (size_t i = 0; i < r.p.size(); ++i) r.p[i] = silu(r.zg[i]) * r.zu[i];
    r.v = r.p;
    r.code = r.p;
    r.scale.assign(kM * kGroupsN, 0.0);
    for (int64_t m = 0; m < kM; ++m) {
        for (int64_t gi = 0; gi < kGroupsN; ++gi) {
            const size_t k0 = static_cast<size_t>(m * kFused + gi * kGroup);
            fwht(r.v.data() + k0, kGroup);
            double amax = 0.0;
            for (int64_t i = 0; i < kGroup; ++i)
                amax = std::max(amax, std::fabs(r.v[k0 + i]));
            const double s = amax > 0.0 ? amax * (1.0 / 127.0) : 1.0;
            r.scale[static_cast<size_t>(m * kGroupsN + gi)] = s;
            for (int64_t i = 0; i < kGroup; ++i)
                r.code[k0 + i] = std::min(127.0, std::max(-127.0,
                                                          std::nearbyint(r.v[k0 + i] * (1.0 / s))));
        }
    }
    return r;
}

// The reference's straight-through cotangents: `gate_factor` false drops the gate's slope on the
// gate half, so a row that loses it is what the check compares against.
struct RefGrad {
    std::vector<double> da;  // [M, K], the codes' basis: the row applies a_scale
    std::vector<double> dw;  // [N, K], the weight's codes
    std::vector<double> ds;  // [N, K / group]
};

RefGrad ref_grad(const Ref& r, const std::vector<double>& a, const std::vector<double>& as,
                 const std::vector<double>& b, const std::vector<double>& bs,
                 const std::vector<double>& gq, const std::vector<double>& gbf,
                 bool gate_factor) {
    // dy = the codes' cotangent, the mask at the forward's codes over the group's scale,
    // rotated back, plus out_bf16's own.
    std::vector<double> dy = gq;
    for (int64_t m = 0; m < kM; ++m) {
        for (int64_t gi = 0; gi < kGroupsN; ++gi) {
            const size_t k0 = static_cast<size_t>(m * kFused + gi * kGroup);
            const double s = r.scale[static_cast<size_t>(m * kGroupsN + gi)];
            for (int64_t i = 0; i < kGroup; ++i) {
                const double code = r.code[k0 + i];
                const bool top = code == 127.0 || code == -127.0;
                const bool live = !top || std::fabs(r.v[k0 + i] * (1.0 / s)) <= 127.0;
                dy[k0 + i] = live ? dy[k0 + i] / s : 0.0;
            }
            fwht(dy.data() + k0, kGroup);
        }
    }
    for (size_t i = 0; i < dy.size(); ++i) dy[i] += gbf[i];
    std::vector<double> dzg(dy.size()), dzu(dy.size());
    for (size_t i = 0; i < dy.size(); ++i) {
        dzg[i] = dy[i] * r.zu[i] * (gate_factor ? silu_prime(r.zg[i]) : 1.0);
        dzu[i] = dy[i] * silu(r.zg[i]);
    }
    RefGrad g;
    g.da.assign(kM * kK, 0.0);
    for (int64_t m = 0; m < kM; ++m)
        for (int64_t k = 0; k < kK; ++k) {
            const int64_t gg = k / kGroup;
            double acc = 0.0;
            for (int64_t j = 0; j < kFused; ++j) {
                const size_t i = static_cast<size_t>(m * kFused + j);
                acc += dzg[i] * (b[static_cast<size_t>(j * kK + k)] *
                                 bs[static_cast<size_t>(j * kGroupsK + gg)]) +
                       dzu[i] * (b[static_cast<size_t>((kFused + j) * kK + k)] *
                                 bs[static_cast<size_t>((kFused + j) * kGroupsK + gg)]);
            }
            // The row hands back the codes' cotangent: the dequantised product times the
            // activation's own group scale, the same factor this reference applies to `a`.
            g.da[static_cast<size_t>(m * kK + k)] =
                acc * as[static_cast<size_t>(m * kGroupsK + gg)];
        }
    g.dw.assign(kN * kK, 0.0);
    g.ds.assign(kN * kGroupsK, 0.0);
    for (int64_t m = 0; m < kM; ++m)
        for (int64_t j = 0; j < kFused; ++j)
            for (int64_t gg = 0; gg < kGroupsK; ++gg) {
                const double sa = as[static_cast<size_t>(m * kGroupsK + gg)];
                const size_t i = static_cast<size_t>(m * kFused + j);
                for (int64_t k = gg * kGroup; k < (gg + 1) * kGroup; ++k) {
                    g.dw[static_cast<size_t>(j * kK + k)] +=
                        dzg[i] * (a[static_cast<size_t>(m * kK + k)] * sa);
                    g.dw[static_cast<size_t>((kFused + j) * kK + k)] +=
                        dzu[i] * (a[static_cast<size_t>(m * kK + k)] * sa);
                    g.ds[static_cast<size_t>(j * kGroupsK + gg)] +=
                        dzg[i] * a[static_cast<size_t>(m * kK + k)];
                    g.ds[static_cast<size_t>((kFused + j) * kGroupsK + gg)] +=
                        dzu[i] * a[static_cast<size_t>(m * kK + k)];
                }
            }
    return g;
}

// The largest |got - want| over the larger magnitude, a floor where both sit near zero.
double max_err(const std::vector<double>& got, const std::vector<double>& want) {
    double m = 0.0;
    for (size_t i = 0; i < want.size(); ++i)
        m = std::max(m, std::fabs(got[i] - want[i]) /
                            (std::max(std::fabs(got[i]), std::fabs(want[i])) + 1e-12));
    return m;
}

void all_zero(const AdValue& o) {
    for (const double x : o.value) R97_CHECK(x == 0.0);
}

void run_case(bool bound, uint64_t seed) {
    const OpRecord rec = make_record(bound);
    TwinFn twin = twin_find(rec.op, rec.params, nullptr);
    R97_CHECK(twin != nullptr);
    AdjointFn adj = adjoint_find(rec.op, Side::host, rec.params, nullptr);
    R97_CHECK(adj != nullptr);

    uint64_t s = seed;
    std::vector<double> a(kM * kK), as_(kM * kGroupsK), b(kN * kK), bs(kN * kGroupsK);
    for (auto& x : a) x = std::nearbyint(urand(&s) * 120.0);
    for (auto& x : as_) x = 0.05 + 0.45 * (1.0 + urand(&s)) / 2.0;
    for (auto& x : b) x = std::nearbyint(urand(&s) * 120.0);
    for (auto& x : bs) x = 0.05 + 0.45 * (1.0 + urand(&s)) / 2.0;

    const std::vector<AdValue> reads{av("i8", {kM, kK}, a), av("f32", {kM, kGroupsK}, as_),
                                     av("i8", {kN, kK}, b), av("bf16", {kN, kGroupsK}, bs)};
    std::vector<AdValue> out;
    twin(rec, reads, &out);
    R97_CHECK(out.size() == (bound ? 3 : 2));
    const Ref ref = ref_forward(a, as_, b, bs);
    double code_max = 0.0;
    for (size_t i = 0; i < ref.code.size(); ++i)
        code_max = std::max(code_max, std::fabs(out[0].value[i] - ref.code[i]));
    R97_CHECK(code_max <= 1.0);  // one code step, a rounding tie at most
    R97_CHECK(max_err(out[1].value, ref.scale) <= kTol);
    if (bound) R97_CHECK(max_err(out[2].value, ref.p) <= kTol);

    std::vector<double> gq(out[0].value.size()), gs(out[1].value.size()), gbf(out[0].value.size());
    for (auto& x : gq) x = urand(&s);
    for (auto& x : gs) x = urand(&s);  // the scale's cotangent: the row drops it
    for (auto& x : gbf) x = bound ? urand(&s) : 0.0;

    std::vector<AdValue> fwd = reads;
    fwd.push_back(out[0]);
    fwd.push_back(out[1]);
    if (bound) fwd.push_back(out[2]);
    std::vector<AdValue> dc;
    dc.reserve(fwd.size());
    for (const AdValue& o : fwd)
        dc.emplace_back(av(o.dtype, o.shape, std::vector<double>(o.value.size(), 0.0)));

    std::vector<AdValue> cot = {av("i8", {kM, kFused}, gq), av("f32", {kM, kGroupsN}, gs)};
    if (bound) cot.push_back(av("bf16", {kM, kFused}, gbf));

    AdjointCall call{rec, fwd, cot, &dc, nullptr, nullptr};
    adj(call);

    const std::vector<double> gbf_ref = bound ? gbf : std::vector<double>(kM * kFused, 0.0);
    const RefGrad want = ref_grad(ref, a, as_, b, bs, gq, gbf_ref, true);
    const double e_a = max_err(dc[0].value, want.da);
    const double e_w = max_err(dc[2].value, want.dw);
    const double e_s = max_err(dc[3].value, want.ds);
    all_zero(dc[1]);          // a_scale is frozen
    all_zero(dc[4]);          // the write slots the row does not answer
    all_zero(dc[5]);
    if (bound) all_zero(dc[6]);
    R97_CHECK(std::max({e_a, e_w, e_s}) <= kTol);

    // The gate's factor, seen: a reference that drops the gate's slope on the gate half differs
    // from the row's weight cotangent there, while the reference above agrees to the tolerance.
    const RefGrad wrong = ref_grad(ref, a, as_, b, bs, gq, gbf_ref, false);
    double guard = 0.0;
    for (int64_t j = 0; j < kFused; ++j)
        for (int64_t k = 0; k < kK; ++k)
            guard = std::max(guard, std::fabs(dc[2].value[static_cast<size_t>(j * kK + k)] -
                                              wrong.dw[static_cast<size_t>(j * kK + k)]));
    R97_CHECK(guard > 1e-3);

    const double m = std::max({e_a, e_w, e_s});
    std::printf("gemm_nt_q_gated host row: max |err| %.3e <= %.0e (K=%lld N=%lld M=%lld "
                "group=%lld, out_bf16 %s), dropped-gate guard %.3e > 1e-03\n",
                m, kTol, (long long)kK, (long long)kN, (long long)kM, (long long)kGroup,
                bound ? "bound" : "unbound", guard);
}

}  // namespace

R97_TEST(gemm_nt_q_gated_host) {
    run_case(true, 0x9e3779b97f4a7c15ull);
    run_case(false, 0x2545f4914f6cdd1dull);
}

// The row freezes a_scale alone: the quantiser's own constant, as gemm_nt_q freezes its scales.
// The weight and its scale answer, because the fold's weight is not the frozen one.
R97_TEST(gemm_nt_q_gated_frozen_position) {
    const OpRecord rec = make_record(true);
    const std::vector<int> frozen = adjoint_frozen(rec.op, Side::host, rec.params);
    R97_CHECK(frozen.size() == 1 && frozen[0] == 1);
}
