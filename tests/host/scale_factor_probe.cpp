// tests/host/scale_factor_probe.cpp -- the activation scale the int8 GEMM's adjoint keeps.
//
// One op, CPU only, no card: `gemm_nt_q` with int8 codes and a real per-group activation scale.
// The op's own forward (the twin, `gemm_nt_q_twin`) dequantises `a` with `a_scale`:
//
//     y[m, n] = sum_k (a[m, k] * a_scale[m, k/gsize]) * (b[n, k] * b_scale[n, k/gsize])
//
// so dL/da[m, k] = a_scale[m, k/gsize] * sum_n dy[m, n] * b[n, k] * b_scale[n, k/gsize].
//
// The walk's code operands carry the CODES' cotangent: every downstream quantiser's straight-through
// adjoint divides the code cotangent by its own group scale, so the cotangent this row owes `a` is
// the derivative above -- the dequantised `dx` times the activation's own scale:
//
//     dx_code[m, k] += a_scale[m, k/gsize] * sum_n dy[m, n] * b[n, k] * b_scale[n, k/gsize]
//
// The host family row `gemm_nt_q_adjoint` (src/adjoint/host/linear.cpp:216) multiplies by `a_scale`,
// and the device row `linear_dx_adjoint` (src/adjoint/device/linear.hip) implements the same rule.
// A row that omitted the factor would come back `1/a_scale` too large, and the quantiser above --
// which divides by `a_scale` on a value never multiplied by it -- would amplify the walk by `1/sa`
// per quantiser -> GEMM hop.
//
// This file is the regression test for that factor. It central-differences the op's own forward
// w.r.t. single elements of `a` at a realistic shape and a REAL, non-unit `a_scale`, then requires
// the adjoint's `dc` for the same element to agree. Remove the `* scale_at(as, ...)` from the row
// again and the ratio lands on the bug's own signature: adj/fd = 1/a_scale, off by 50-87% here.
//
// Label: host.
#include "support/test.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "r97/adjoint.h"

namespace {

using namespace r97;

uint64_t mix(uint64_t* s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

double uni(uint64_t* s) {
    return static_cast<double>(mix(s) >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
}

double rel(double a, double b) {
    const double d = std::fabs(a) > std::fabs(b) ? std::fabs(a) : std::fabs(b);
    return std::fabs(a - b) / (d < 1e-30 ? 1e-30 : d);
}

// One gemm_nt_q record, the shape the walk's live records have: int8 codes, an f32 activation
// scale per (row, group), int8 weight codes and a bf16 weight scale.
OpRecord record(int64_t M, int64_t N, int64_t K, int64_t gsize) {
    OpRecord rec;
    rec.op = "gemm_nt_q";
    rec.params = {{"M", M}, {"N", N}, {"K", K}, {"group", gsize}, {"dtype", "i8a8h"}};
    rec.operands = {
        {"a", "i8", {M, K}, 0, M, false},
        {"a_scale", "f32", {M, K / gsize}, 0, M, false},
        {"b", "i8", {N, K}, 0, N, false},
        {"b_scale", "bf16", {N, K / gsize}, 0, N, false},
        {"y", "f32", {M, N}, 0, M, true},
    };
    return rec;
}

struct Values {
    std::vector<AdValue> in;  // a, a_scale, b, b_scale
    AdValue dy;               // the cotangent of y
};

Values values_of(const OpRecord& rec, uint64_t seed) {
    Values v;
    const int64_t M = rec.operands[0].shape[0], K = rec.operands[0].shape[1];
    const int64_t N = rec.operands[2].shape[0];
    for (const Operand& o : rec.operands) {
        if (o.written) continue;
        AdValue a;
        a.dtype = o.dtype;
        a.shape = o.shape;
        int64_t n = 1;
        for (int64_t d : o.shape) n *= d;
        a.value.resize(static_cast<std::size_t>(n));
        for (double& x : a.value) {
            const uint64_t r = mix(&seed);
            if (o.buffer == "a" || o.buffer == "b") {
                x = static_cast<double>(static_cast<int>(r % 201) - 100);  // int8 codes
            } else {
                // Positive, bf16-exact, and the size a real grid gives: 0.125 .. 0.5. Never 1.0 --
                // a unit scale would make the factor this test exists for invisible.
                x = static_cast<double>((r >> 8) % 4 + 1) / 8.0;
            }
        }
        v.in.push_back(std::move(a));
    }
    v.dy.dtype = "f32";
    v.dy.shape = {M, N};
    v.dy.value.resize(static_cast<std::size_t>(M * N));
    for (double& x : v.dy.value) x = uni(&seed);
    return v;
}

// L(t): the twin's y, dotted with dy, after a[..] += delta at one element of `a`.
double loss_at(const OpRecord& rec, const Values& v, TwinFn twin, int64_t m, int64_t k,
               double delta) {
    std::vector<AdValue> x = v.in;
    x[0].value[static_cast<std::size_t>(m * rec.operands[0].shape[1] + k)] += delta;
    std::vector<AdValue> y;
    twin(rec, x, &y);
    if (y.size() != 1)
        throw std::runtime_error("the twin wrote " + std::to_string(y.size()) + " operands");
    double s = 0.0;
    for (std::size_t e = 0; e < y[0].value.size(); ++e) s += y[0].value[e] * v.dy.value[e];
    return s;
}

struct Probe {
    int64_t m = 0, k = 0, g = 0;
    double scale = 0.0;   // a_scale[m, k/gsize]
    double closed = 0.0;  // dL/da[m,k] in closed form from the twin
    double fd = 0.0;      // central difference of the twin
    double adj = 0.0;     // the adjoint's dc for the same element
    double ratio() const { return adj / fd; }
};

struct Report {
    double worst_adj_vs_fd = 0.0;    // max over the probed elements: the row's dx vs the difference
    double worst_closed_vs_fd = 0.0; // the same for the closed form: proves the reference
    double mean_ratio = 0.0;         // mean adj/fd (1 when the row keeps a_scale, 1/s when it drops)
    double max_scale_dev = 0.0;      // max |a_scale - 1| over the probed elements (0 would be blind)
};

Report probe(const char* who, const OpRecord& rec, uint64_t seed) {
    std::string err;
    TwinFn twin = twin_find(rec.op, rec.params, &err);
    if (!twin) throw std::runtime_error("no twin: " + err);
    AdjointFn adj = adjoint_find(rec.op, Side::host, rec.params, &err);
    if (!adj) throw std::runtime_error("no adjoint: " + err);

    Values v = values_of(rec, seed);
    const int64_t M = rec.operands[0].shape[0], K = rec.operands[0].shape[1];
    const int64_t N = rec.operands[2].shape[0];
    const int64_t groups = rec.operands[1].shape[1];
    const int64_t gsize = K / groups;

    // The forward over the whole schema, as the walk hands it to an adjoint.
    std::vector<AdValue> out_vals;
    twin(rec, v.in, &out_vals);
    std::vector<AdValue> fwd(rec.operands.size());
    {
        std::size_t r = 0, w = 0;
        for (std::size_t i = 0; i < rec.operands.size(); ++i)
            fwd[i] = rec.operands[i].written ? out_vals[w++] : v.in[r++];
    }
    std::vector<AdValue> cot{v.dy};
    std::vector<AdValue> dc(rec.operands.size());
    {
        std::size_t r = 0;
        for (std::size_t i = 0; i < rec.operands.size(); ++i) {
            if (rec.operands[i].written) continue;
            dc[i] = v.in[r++];
            dc[i].value.assign(dc[i].value.size(), 0.0);
        }
    }
    AdjointCall call{rec, fwd, cot, &dc, nullptr, ""};
    adj(call);

    Report rep;
    double sum_ratio = 0.0;
    int64_t n_probes = 0;
    const int64_t ks[] = {0, gsize / 2, gsize, gsize + 3, 2 * gsize};
    for (int64_t m = 0; m < (M < 3 ? M : 3); ++m)
        for (const int64_t k : ks) {
            if (k >= K) continue;
            Probe p;
            p.m = m;
            p.k = k;
            p.g = k / gsize;
            p.scale = v.in[1].value[static_cast<std::size_t>(m * groups + p.g)];
            double closed = 0.0;
            for (int64_t n = 0; n < N; ++n)
                closed += v.dy.value[static_cast<std::size_t>(m * N + n)] *
                          v.in[2].value[static_cast<std::size_t>(n * K + k)] *
                          v.in[3].value[static_cast<std::size_t>(n * groups + p.g)] * p.scale;
            p.closed = closed;
            const double eps = 1e-3;
            p.fd = (loss_at(rec, v, twin, m, k, +eps) - loss_at(rec, v, twin, m, k, -eps)) /
                   (2.0 * eps);
            p.adj = dc[0].value[static_cast<std::size_t>(m * K + k)];
            const double e = rel(p.adj, p.fd);
            if (e > rep.worst_adj_vs_fd) rep.worst_adj_vs_fd = e;
            const double ec = rel(p.closed, p.fd);
            if (ec > rep.worst_closed_vs_fd) rep.worst_closed_vs_fd = ec;
            const double sd = std::fabs(p.scale - 1.0);
            if (sd > rep.max_scale_dev) rep.max_scale_dev = sd;
            sum_ratio += p.ratio();
            ++n_probes;
            std::printf("  %-6s a[%lld,%lld] g%lld s=%.4f  closed %+.6f  fd %+.6f  adj %+.6f  "
                        "adj/fd %.4f\n",
                        who, (long long)p.m, (long long)p.k, (long long)p.g, p.scale, p.closed,
                        p.fd, p.adj, p.ratio());
        }
    rep.mean_ratio = sum_ratio / static_cast<double>(n_probes);
    std::printf("%s: mean adj/fd %.4f, worst |adj-fd|/max %.3g, worst |closed-fd|/max %.3g, "
                "max |a_scale-1| %.3f\n",
                who, rep.mean_ratio, rep.worst_adj_vs_fd, rep.worst_closed_vs_fd,
                rep.max_scale_dev);
    return rep;
}

}  // namespace

// The regression test: the row's dx is the derivative of the op's own forward, i.e. the codes'
// cotangent, which carries the activation's group scale. It fails if the row drops `a_scale`.
R97_TEST(scale_factor_probe_keeps_the_activation_scale) {
    const OpRecord rec = record(3, 64, 384, 128);
    const Report r = probe("row", rec, 0x5ca1eull);
    // A real scale, not the unit one that would make the row's factor invisible.
    R97_CHECK(r.max_scale_dev > 0.4);
    // The central difference measures the twin's own derivative, so it is a trustworthy reference.
    R97_CHECK(r.worst_closed_vs_fd < 1e-7);
    if (!(r.worst_adj_vs_fd < 1e-6))
        throw std::runtime_error(
            "gemm_nt_q's adjoint does not match the finite difference of the op's own forward: it "
            "is missing the activation scale (dx is 1/a_scale too large). worst relative error " +
            std::to_string(r.worst_adj_vs_fd) + ", mean adj/fd " + std::to_string(r.mean_ratio) +
            " (1 when the row keeps a_scale)");
}
