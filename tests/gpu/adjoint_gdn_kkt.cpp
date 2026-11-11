// tests/gpu/adjoint_gdn_kkt.cpp -- the device adjoint of the gated delta net's in-chunk solve
// against the host adjoint.
//
//   * `gdn_kkt_solve` is registered on the card and agrees with src/adjoint/host/gdn_kkt.cpp
//     within the harness's bf16 tolerance at the fixture's delta-net geometry (2 key heads over 4
//     value heads, 16 lanes, chunk 64, docs/FIXTURES.md) and at the real model's (5120 hidden, 16
//     key heads over 48 value heads, 128 lanes, chunk 64, docs/DESIGN.md);
//   * a chunk is solved on its own operands, so the fixture covers below one chunk, exactly one,
//     and a short last chunk; the real model covers 1024 tokens as one sequence, 4096 tokens as
//     one sequence, and each as a batch of unequal lengths, because cu_seqlens addresses rows and
//     a batch must not hand a chunk another sequence's keys;
//   * the 4096-token call's largest error against the f64 host adjoint is printed, with how many
//     elements sit outside the tolerance, and beside it the same solve with Y and Et rounded to
//     bf16 as they are formed -- the arithmetic the kernel does not do, and the number
//     docs/kernels/gdn_kkt.md cites for whether bf16 would suffice;
//   * a deliberately wrong kernel (one that skips dg's second term) is rejected rather than
//     accepted as plausible.
//
// Needs one R9700 (gfx1201, 32 GB) or similar; runs under the GPU test guard. No container is read:
// the real shapes are numbers from the design, so this test does not carry the `model` label.
#include "support/device_check.h"
#include "support/gpu_guard.h"
#include "support/test.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace r97;
using device_check::Case;
using device_check::urand;

// The delta net's geometry: key heads over value heads, lanes per key, the chunk the solve inverts
// by forward substitution, and the lengths of the sequences in the batch.
struct Geometry {
    int64_t k_heads;
    int64_t heads;
    int64_t lanes;
    int64_t chunk;
    std::vector<int64_t> lens;
};

int64_t rows_of(const Geometry& g) {
    return std::accumulate(g.lens.begin(), g.lens.end(), int64_t(0));
}

std::string name_of(const Geometry& g) {
    std::string out;
    for (size_t i = 0; i < g.lens.size(); ++i) {
        if (i) out += "+";
        out += std::to_string(g.lens[i]);
    }
    out += " x " + std::to_string(g.heads) + " heads of " + std::to_string(g.lanes) + ", chunk " +
           std::to_string(g.chunk);
    if (g.lens.size() > 1) out += " batch";
    return out;
}

Operand read_operand(const char* buffer, const char* dtype, std::vector<int64_t> shape) {
    return Operand{buffer, dtype, std::move(shape), 0, 0, false};
}

Operand written_operand(const char* buffer, const char* dtype, std::vector<int64_t> shape) {
    return Operand{buffer, dtype, std::move(shape), 0, 0, true};
}

// bf16's own grid, so the keys the card reads at f32 are the keys the twin computed with.
double on_bf16(double v) {
    const float f = (float)v;
    unsigned int u;
    std::memcpy(&u, &f, sizeof u);
    const unsigned int lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    u &= 0xffff0000u;
    float out;
    std::memcpy(&out, &u, sizeof out);
    return (double)out;
}

/* The op's schema: four reads, then A, which the forward writes and the twin fills on the way in.
 * The keys are bf16's grid and unit norm, beta a sigmoid of the gate, and g the log of a decay
 * that is cumsummed inside each chunk -- the near-identity block a trained delta net solves. */
Case kkt_case(const Geometry& g, uint64_t seed) {
    const int64_t rows = rows_of(g);
    std::vector<double> keys((size_t)(rows * g.k_heads * g.lanes));
    std::vector<double> betas((size_t)(rows * g.heads));
    std::vector<double> decays((size_t)(rows * g.heads));

    for (int64_t t = 0; t < rows; ++t)
        for (int64_t hk = 0; hk < g.k_heads; ++hk) {
            double* row = &keys[(size_t)((t * g.k_heads + hk) * g.lanes)];
            double norm = 0.0;
            for (int64_t d = 0; d < g.lanes; ++d) {
                row[d] = urand(&seed);
                norm += row[d] * row[d];
            }
            norm = std::sqrt(norm);
            for (int64_t d = 0; d < g.lanes; ++d) row[d] = on_bf16(row[d] / norm);
        }

    for (int64_t t = 0; t < rows; ++t)
        for (int64_t h = 0; h < g.heads; ++h)
            betas[(size_t)(t * g.heads + h)] = 0.05 + 0.9 * (0.5 + 0.5 * urand(&seed));

    int64_t bos = 0;
    for (const int64_t len : g.lens) {
        for (int64_t h = 0; h < g.heads; ++h) {
            for (int64_t c0 = 0; c0 < len; c0 += g.chunk) {
                const int64_t L = std::min(g.chunk, len - c0);
                double acc = 0.0;
                for (int64_t i = 0; i < L; ++i) {
                    acc -= 0.2 + 0.6 * (0.5 + 0.5 * urand(&seed));
                    decays[(size_t)((bos + c0 + i) * g.heads + h)] = acc;
                }
            }
        }
        bos += len;
    }

    std::vector<int64_t> cu_shape{(int64_t)g.lens.size() + 1};
    std::vector<double> cu((size_t)cu_shape[0], 0.0);
    int64_t at = 0;
    for (size_t s = 0; s < g.lens.size(); ++s) {
        at += g.lens[s];
        cu[s + 1] = (double)at;
    }

    Case c;
    c.rec.op = "gdn_kkt_solve";
    c.rec.params = {{"M", rows}, {"head_k", g.lanes}, {"chunk", g.chunk}};
    c.rec.operands = {
        read_operand("k", "bf16", {rows, g.k_heads, g.lanes}),
        read_operand("beta", "f32", {rows, g.heads}),
        read_operand("g", "f32", {rows, g.heads}),
        read_operand("cu_seqlens", "i32", cu_shape),
        written_operand("A", "bf16", {rows, g.heads, g.chunk}),
    };
    c.in = {
        AdValue{"bf16", {rows, g.k_heads, g.lanes}, std::move(keys)},
        AdValue{"f32", {rows, g.heads}, std::move(betas)},
        AdValue{"f32", {rows, g.heads}, std::move(decays)},
        AdValue{"i32", cu_shape, std::move(cu)},
    };
    c.cot = {device_check::random_value("bf16", {rows, g.heads, g.chunk}, &seed)};
    c.label = name_of(g);
    return c;
}

/* -- the measurements ------------------------------------------------------ */

struct Error {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double l2 = 0.0;      // ||device - host|| / ||host||, over the three cotangents together
    int64_t outside = 0;  // elements past the harness's tolerance
    double worst = 0.0;   // |d - h| / (kAbsTol + kRelTol * |h|) at its tightest
};

// Every element of the three gradient slots, measured against the host adjoint: the largest
// absolute and relative error, the normwise relative error, and how many elements are outside the
// tolerance the harness holds a device kernel to (which is the tolerance in the doc).
Error error_between(const std::vector<AdValue>& d, const std::vector<AdValue>& h, size_t slots) {
    Error e;
    double num = 0.0, den = 0.0;
    for (size_t slot = 0; slot < slots; ++slot)
        for (size_t i = 0; i < h[slot].value.size(); ++i) {
            const double hv = h[slot].value[i];
            const double diff = std::abs(d[slot].value[i] - hv);
            num += diff * diff;
            den += hv * hv;
            e.max_abs = std::max(e.max_abs, diff);
            const double rel = diff / (std::abs(hv) + 1e-12);
            if (rel > e.max_rel) e.max_rel = rel;
            const double tol = device_check::kAbsTol + device_check::kRelTol * std::abs(hv);
            if (diff > tol) ++e.outside;
            if (diff / tol > e.worst) e.worst = diff / tol;
        }
    e.l2 = std::sqrt(num / (den > 0.0 ? den : 1.0));
    return e;
}

// The device adjoint and the host adjoint on the same forward values and cotangent, and the
// largest error between them.
Error compare_device_to_host(const Case& c) {
    std::string error;
    const AdjointFn device = adjoint_find(c.rec.op, Side::device, c.rec.params, &error);
    const AdjointFn host = adjoint_find(c.rec.op, Side::host, c.rec.params, &error);
    if (!device || !host) throw std::runtime_error(error);

    std::vector<AdValue> fwd, host_dc;
    AdjointCall host_call = device_check::make_call(c, &fwd, &host_dc);
    host(host_call);

    std::vector<AdValue> device_dc = host_dc;
    for (AdValue& v : device_dc) std::fill(v.value.begin(), v.value.end(), 0.0);
    AdjointCall device_call{c.rec, fwd, c.cot, &device_dc, nullptr, ""};
    device(device_call);

    return error_between(device_dc, host_dc, 3);
}

/* The same transposed solve with Y and Et rounded to bf16 as they are formed, everything else in
 * f64: the arithmetic a bf16 solve would run, so the doc can say what it would cost. Y is stored
 * packed (row p holds columns 0..p), as the kernel holds it. */
std::vector<AdValue> solve_in_bf16(const AdjointCall& call, const Geometry& g) {
    const std::vector<double>& k = call.fwd[0].value;
    const std::vector<double>& beta = call.fwd[1].value;
    const std::vector<double>& gate = call.fwd[2].value;
    const std::vector<double>& A = call.fwd[4].value;
    const std::vector<double>& Ac = call.cot[0].value;
    const int64_t rows = rows_of(g);
    const int64_t K = g.lanes, C = g.chunk, H = g.heads, HK = g.k_heads;
    const int64_t grp = H / HK;

    std::vector<AdValue> out;
    out.push_back(AdValue{"bf16", {rows, HK, K}, std::vector<double>((size_t)(rows * HK * K), 0.0)});
    out.push_back(AdValue{"f32", {rows, H}, std::vector<double>((size_t)(rows * H), 0.0)});
    out.push_back(AdValue{"f32", {rows, H}, std::vector<double>((size_t)(rows * H), 0.0)});
    out.push_back(AdValue{"i32", {(int64_t)g.lens.size() + 1},
                          std::vector<double>((size_t)g.lens.size() + 1, 0.0)});
    std::vector<double>& dk = out[0].value;
    std::vector<double>& dbeta = out[1].value;
    std::vector<double>& dg = out[2].value;

    int64_t bos = 0;
    for (const int64_t len : g.lens) {
        for (int64_t h = 0; h < H; ++h) {
            const int64_t hk = h / grp;
            for (int64_t c0 = 0; c0 < len; c0 += C) {
                const int64_t L = std::min(C, len - c0);
                std::vector<double> Y((size_t)(L * L), 0.0);
                for (int64_t p = 0; p < L; ++p)
                    for (int64_t l = 0; l <= p; ++l) {
                        double acc = 0.0;
                        for (int64_t j = 0; j <= l; ++j)
                            acc += Ac[(size_t)(((bos + c0 + p) * H + h) * C + j)] *
                                   A[(size_t)(((bos + c0 + l) * H + h) * C + j)];
                        Y[(size_t)(p * L + l)] = on_bf16(acc);
                    }
                for (int64_t i = 0; i < L; ++i)
                    for (int64_t l = 0; l < i; ++l) {
                        double et = 0.0;
                        for (int64_t p = i; p < L; ++p)
                            et -= A[(size_t)(((bos + c0 + p) * H + h) * C + i)] *
                                  Y[(size_t)(p * L + l)];
                        Y[(size_t)(i * L + l)] = on_bf16(et);
                    }
                for (int64_t i = 0; i < L; ++i) {
                    const size_t ti = (size_t)((bos + c0 + i) * H + h);
                    const double bi = beta[ti];
                    const double gi = gate[ti];
                    double beta_acc = 0.0, dgi = 0.0;
                    for (int64_t l = 0; l < i; ++l) {
                        const double et = Y[(size_t)(i * L + l)];
                        if (et == 0.0) continue;
                        const size_t tl = (size_t)((bos + c0 + l) * H + h);
                        const double phi = std::exp(gi - gate[tl]);
                        double dot = 0.0;
                        for (int64_t d = 0; d < K; ++d)
                            dot += k[(size_t)(((bos + c0 + i) * HK + hk) * K + d)] *
                                   k[(size_t)(((bos + c0 + l) * HK + hk) * K + d)];
                        const double cij = dot * phi;
                        const double m = bi * cij;
                        beta_acc += et * cij;
                        dgi += et * m;
                        dg[tl] -= et * m;
                        const double coeff = et * bi * phi;
                        for (int64_t d = 0; d < K; ++d) {
                            dk[(size_t)(((bos + c0 + i) * HK + hk) * K + d)] +=
                                coeff * k[(size_t)(((bos + c0 + l) * HK + hk) * K + d)];
                            dk[(size_t)(((bos + c0 + l) * HK + hk) * K + d)] +=
                                coeff * k[(size_t)(((bos + c0 + i) * HK + hk) * K + d)];
                        }
                    }
                    dbeta[ti] += beta_acc;
                    dg[ti] += dgi;
                }
            }
        }
        bos += len;
    }
    return out;
}

// The wrong kernel: dg comes back with its sign thrown away, so every token whose decay sum is
// negative is wrong by twice its value while dk and dbeta stay right. The harness must catch it,
// in dg's slot.
void wrong_gdn_kkt_adjoint(const AdjointCall& call) {
    std::string error;
    const AdjointFn right = adjoint_find(call.rec.op, Side::host, call.rec.params, &error);
    if (!right) return;
    std::vector<AdValue> dc = *call.dc;
    for (AdValue& v : dc) std::fill(v.value.begin(), v.value.end(), 0.0);
    AdjointCall tmp{call.rec, call.fwd, call.cot, &dc, nullptr, ""};
    right(tmp);
    for (size_t slot = 0; slot < call.dc->size(); ++slot)
        for (size_t i = 0; i < call.dc->at(slot).value.size(); ++i) {
            const double v = dc[slot].value[i];
            call.dc->at(slot).value[i] += slot == 2 ? std::fabs(v) : v;
        }
}

/* -- the run --------------------------------------------------------------- */

struct Timed {
    Case c;
    int reps;
};

void require_registered() {
    if (!op_rows("gdn_kkt_solve").device_adjoint)
        throw std::runtime_error("no device adjoint for 'gdn_kkt_solve'");
}

std::vector<Timed> fixture_cases() {
    const Geometry geo{2, 4, 16, 64, {}};
    Geometry g = geo;
    std::vector<Timed> out;
    g.lens = {5};
    out.push_back({kkt_case(g, 0x9d000001ull), 20});
    g.lens = {64};
    out.push_back({kkt_case(g, 0x9d000002ull), 20});
    g.lens = {101};  // two chunks, the last a short 37
    out.push_back({kkt_case(g, 0x9d000003ull), 20});
    g.lens = {5, 64, 150};  // each sequence's chunk grid starts at its own cu row
    out.push_back({kkt_case(g, 0x9d000004ull), 20});
    // No grouping -- one key head per value head -- with wider keys.
    g = {2, 2, 32, 64, {90}};
    out.push_back({kkt_case(g, 0x9d000005ull), 20});
    return out;
}

// The real model: 16 key heads over 48 value heads, 128 lanes, chunk 64 (docs/DESIGN.md), 1024 and
// 4096 tokens, each as one sequence and as a batch of unequal lengths.
std::vector<Timed> real_cases() {
    const Geometry g{16, 48, 128, 64, {}};
    std::vector<Timed> out;
    Geometry one = g;
    one.lens = {1024};
    out.push_back({kkt_case(one, 0x9d010001ull), 5});
    one.lens = {1024, 512, 2560};
    out.push_back({kkt_case(one, 0x9d010002ull), 5});
    one.lens = {4096};
    out.push_back({kkt_case(one, 0x9d010003ull), 3});
    one.lens = {1000, 3000, 96};
    out.push_back({kkt_case(one, 0x9d010004ull), 3});
    return out;
}

void run() {
    const auto options = device_check::guard_options(int64_t(8) << 30);
    const r97test::DeviceGuard guard(options);
    device_check::require_device(options.device);
    (void)device_check::load_plugin();

    require_registered();

    std::vector<Timed> cases = fixture_cases();
    std::vector<Timed> real = real_cases();
    const Timed big = real[2];  // 4096 tokens, one sequence
    for (Timed& t : real) cases.push_back(std::move(t));

    for (const Timed& t : cases) {
        const device_check::Report r = device_check::check(t.c);
        device_check::require(r);
        R97_CHECK(r.scratch_peak > 0);  // the kernel took its device buffers from the allocator
        std::printf("%s: device == host, scratch peak %lld KiB\n", t.c.label.c_str(),
                    (long long)(r.scratch_peak / 1024));
        if (r97test::bench()) device_check::print_bench(t.c, t.reps);
    }

    // The 4096-token single sequence against the f64 host adjoint, and the same solve with its
    // transposed solve rounded to bf16. The doc quotes both.
    const Error f32 = compare_device_to_host(big.c);
    std::printf("gdn_kkt_solve %s: f32 solve max abs %.6g, max rel %.6g, relative L2 %.6g, "
                "outside tolerance %lld, worst %.3g of it\n",
                big.c.label.c_str(), f32.max_abs, f32.max_rel, f32.l2, (long long)f32.outside,
                f32.worst);

    std::vector<AdValue> fwd, host_dc;
    std::string error;
    const AdjointFn host = adjoint_find(big.c.rec.op, Side::host, big.c.rec.params, &error);
    AdjointCall host_call = device_check::make_call(big.c, &fwd, &host_dc);
    host(host_call);
    const std::vector<AdValue> bf16 = solve_in_bf16(host_call, Geometry{16, 48, 128, 64, {4096}});
    const Error e = error_between(bf16, host_dc, 3);
    std::printf("gdn_kkt_solve %s: bf16 solve max abs %.6g, max rel %.6g, relative L2 %.6g, "
                "outside tolerance %lld, worst %.3g of it\n",
                big.c.label.c_str(), e.max_abs, e.max_rel, e.l2, (long long)e.outside, e.worst);

    // A kernel whose dg keeps a token's own term but not its neighbours' must be rejected, in
    // dg's slot.
    const Case small = kkt_case(Geometry{2, 4, 16, 64, {101}}, 0x9d00ffffull);
    const device_check::Report wrong =
        device_check::check(small, wrong_gdn_kkt_adjoint);
    R97_CHECK(!wrong.ok);
    R97_CHECK(wrong.first_slot == 2);
    std::printf("rejected the wrong kernel: %s\n", wrong.note.c_str());
}

}  // namespace

R97_TEST(adjoint_gdn_kkt_device) { run(); }
