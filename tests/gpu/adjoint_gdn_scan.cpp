// tests/gpu/adjoint_gdn_scan.cpp -- the device adjoint of the gated delta-rule chunk scan.
//
// `gdn_chunk_scan` carries a [K, V] state from chunk to chunk of a delta-rule sequence, and the
// state's cotangent is carried the same way: over 4096 tokens that is 64 state updates per value
// head. The card's kernel (src/adjoint/device/gdn_scan.hip) keeps the state and its cotangent at
// f32 that whole way and rounds only the gradients that leave it. Its reference is the op's own
// host row (src/adjoint/host/gdn_scan.cpp) -- the card issues the scan as one op, so there is no
// fused op for the reference to compose -- and this file holds the kernel to it within the
// harness's bf16 tolerance on the same random inputs.
//
// The shapes are the tiny fixture's gdn geometry (docs/FIXTURES.md; two key heads and four value
// heads of 16, 4-token chunks) and the real model's (docs/DESIGN.md; hidden 5120, K = V = 128, 16
// key heads, 48 value heads, 64-token chunks), at 1024 and 4096 tokens, for one sequence and for a
// batch of unequal lengths. No container is read: the real shapes are numbers from the design, so
// this test does not carry the `model` label.
//
// The data is the real pipeline's, because the scan is only well conditioned with it: `gdn_prep`
// L2-normalises q and k per head, so the keys' products are cosines and the delta rule's transition
// stays bounded over 64 chunks; `gdn_kkt_solve` writes each chunk's `A = (I + M)^{-1}` from those
// cosines, the gate's decay and beta; and the gate is a within-chunk cumsum of small negative log
// decays. `A` with random entries of order one would grow the state by tens of times a chunk, and
// then an f32 kernel could not agree with an f64 reference at all: the comparison would measure the
// case's conditioning rather than the kernel.
//
// The test also prints what docs/kernels/gdn_scan.md quotes: the largest deviation from the f64
// host row at the real shapes, and, under `--bench`, the time per call beside the forward twin's,
// and the scratch peak.
#include "r97/adjoint.h"

#include "support/device_check.h"
#include "support/gpu_guard.h"
#include "support/test.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace r97;
using device_check::numel;

// One `gdn_chunk_scan` geometry. `k` and `v` are the per-head widths, `hk` and `hv` the head
// counts, `chunk` the chunk length, and `scale` the record's scale parameter; `lens` is one length
// per sequence in the batch. `slots` is the state cache's slot count (0 for one slot per sequence)
// and `sidx` the per-sequence slot indices (empty for no `state_idx` read).
struct ScanShape {
    const char* name;
    int64_t k;
    int64_t v;
    int64_t hk;
    int64_t hv;
    int64_t chunk;
    double scale;
    std::vector<int64_t> lens;
    int64_t slots = 0;
    std::vector<int64_t> sidx;

    int64_t tokens() const {
        int64_t t = 0;
        for (int64_t n : lens) t += n;
        return t;
    }
    int64_t seqs() const { return (int64_t)lens.size(); }
    int64_t state_slots() const { return slots > 0 ? slots : seqs(); }
};

// The tiny fixture's gdn geometry, and a state pool deeper than the batch: sequence 0 keeps slot 2
// and sequence 1 slot 0, so a kernel that reads or writes the wrong slot is caught.
std::vector<ScanShape> fixture_shapes() {
    return {
        {"tiny-fixture-short", 16, 16, 2, 4, 4, 0.5, {3}},
        {"ragged-chunks", 16, 16, 2, 4, 4, 0.5, {2, 4, 13}},
        {"state-pool", 16, 16, 2, 4, 4, 0.5, {5, 4}, 3, {2, 0}},
        {"state-pool-unsorted", 16, 16, 2, 4, 4, 0.5, {9, 1, 4}, 4, {3, 0, 2}},
    };
}

// The real model's gdn geometry at the two token counts the issue names. The batch of unequal
// lengths puts three sequences across chunk boundaries in different ways: 448 tokens is exactly 7
// chunks, and 322 and 254 leave a short last chunk of 2 and of 62 tokens.
std::vector<ScanShape> real_shapes() {
    const double scale = 1.0 / std::sqrt(128.0);
    return {
        {"1024-one-sequence", 128, 128, 16, 48, 64, scale, {1024}},
        {"1024-ragged", 128, 128, 16, 48, 64, scale, {448, 322, 254}},
        {"4096-one-sequence", 128, 128, 16, 48, 64, scale, {4096}},
    };
}

// A value as the card holds it. Both sides read the case's values, but the card reads them as the
// dtype the operand declares, so a value the tape could not hold would show up as an error of the
// comparison rather than of the kernel.
double round_dtype(double x, const std::string& dtype) {
    float f = (float)x;
    if (dtype == "bf16") {
        unsigned int u;
        std::memcpy(&u, &f, sizeof u);
        if ((u & 0x7fffffffu) <= 0x7f800000u) {
            const unsigned int lsb = (u >> 16) & 1u;
            u += 0x7fffu + lsb;
            u &= 0xffff0000u;
        }
        std::memcpy(&f, &u, sizeof f);
    }
    return (double)f;
}

// A parameter's text, exactly representable: the tape carries integers and strings and drops a
// double, so the f64 parameters travel as strings.
std::string f64_text(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
}

OpRecord scan_rec(const ScanShape& s) {
    const int64_t t = s.tokens();
    const int64_t seq = s.seqs();
    const int64_t slots = s.state_slots();
    OpRecord rec;
    rec.op = "gdn_chunk_scan";
    rec.params = {{"M", t},
                  {"head_k", s.k},
                  {"head_v", s.v},
                  {"chunk", s.chunk},
                  {"scale", f64_text(s.scale)}};
    rec.n = t;
    // Schema positions 0 to 9; the optional state_idx is position 10, after the two writes.
    rec.operands = {
        {"q", "bf16", {t, s.hk, s.k}, 0, t, false},
        {"k", "bf16", {t, s.hk, s.k}, 0, t, false},
        {"v", "bf16", {t, s.hv, s.v}, 0, t, false},
        {"A", "bf16", {t, s.hv, s.chunk}, 0, t, false},
        {"g", "f32", {t, s.hv}, 0, t, false},
        {"beta", "f32", {t, s.hv}, 0, t, false},
        {"h0", "f32", {slots, s.hv, s.v, s.k}, 0, slots, false},
        {"cu", "i32", {seq + 1}, 0, 1, false},
        {"o", "bf16", {t, s.hv, s.v}, 0, t, true},
        {"ht", "f32", {slots, s.hv, s.v, s.k}, 0, slots, true},
    };
    if (!s.sidx.empty()) rec.operands.push_back({"state_idx", "i32", {seq}, 0, 1, false});
    return rec;
}

// A's value: `gdn_kkt_solve`'s (I + M)^{-1} for each chunk and value head, with
// M[i][l] = beta_i <k_i, k_l> e^{g_i - g_l} for l < i and zeros elsewhere. The op writes zero above
// the diagonal, so the case does too.
void fill_kkt(AdValue* A, const AdValue& k, const AdValue& beta, const AdValue& g,
              const ScanShape& s) {
    const int64_t L = s.chunk, K = s.k, hv = s.hv, grp = s.hv / s.hk;
    std::vector<double> kc((size_t)(L * K)), M((size_t)(L * L)), Ac((size_t)(L * L));
    int64_t bos = 0;
    for (int64_t n = 0; n < s.seqs(); ++n) {
        for (int64_t c0 = 0; c0 < s.lens[n]; c0 += L) {
            const int64_t len = std::min(L, s.lens[n] - c0);
            for (int64_t h = 0; h < hv; ++h) {
                const int64_t kh = h / grp;
                for (int64_t i = 0; i < len; ++i) {
                    const int64_t ti = bos + c0 + i;
                    const double gi = g.value[(size_t)(ti * hv + h)];
                    const double bi = beta.value[(size_t)(ti * hv + h)];
                    for (int64_t d = 0; d < K; ++d)
                        kc[(size_t)(i * K + d)] = k.value[(size_t)((ti * s.hk + kh) * K + d)];
                    for (int64_t l = 0; l < len; ++l) {
                        double dot = 0.0;
                        if (l < i) {
                            const int64_t tl = bos + c0 + l;
                            for (int64_t d = 0; d < K; ++d)
                                dot += kc[(size_t)(i * K + d)] *
                                       k.value[(size_t)((tl * s.hk + kh) * K + d)];
                            dot *= bi * std::exp(gi - g.value[(size_t)(tl * hv + h)]);
                        }
                        M[(size_t)(i * L + l)] = dot;
                    }
                }
                // (I + M)^{-1} by forward substitution: A[i][j] = d_ij - sum_{l<i} M[i][l] A[l][j].
                for (int64_t i = 0; i < len; ++i) {
                    for (int64_t j = 0; j <= i; ++j) {
                        double acc = (i == j) ? 1.0 : 0.0;
                        for (int64_t l = j; l < i; ++l)
                            acc -= M[(size_t)(i * L + l)] * Ac[(size_t)(l * L + j)];
                        Ac[(size_t)(i * L + j)] = acc;
                    }
                }
                for (int64_t i = 0; i < len; ++i)
                    for (int64_t j = 0; j < L; ++j)
                        A->value[(size_t)(((bos + c0 + i) * hv + h) * L + j)] =
                            j <= i ? round_dtype(Ac[(size_t)(i * L + j)], "bf16") : 0.0;
            }
        }
        bos += s.lens[n];
    }
}

// The values the harness hands both sides. q, k and v are drawn from `seed` and rounded to bf16; q
// and k are L2-normalised per head from the rounded values, as gdn_prep does. The data that names
// structure is not drawn: `cu` holds the sequence boundaries, h0 is the zero state a prefill starts
// from, g is the within-chunk cumsum of small negative log decays, beta is a sigmoid of a small
// input, and A is gdn_kkt_solve's inverse. The cotangents are the writes' (o, then ht).
device_check::Case scan_case(const ScanShape& s, std::string label, uint64_t seed) {
    device_check::Case c;
    c.label = std::move(label);
    c.rec = scan_rec(s);
    const int64_t t = s.tokens(), hv = s.hv, hk = s.hk, K = s.k, V = s.v;

    for (const Operand& o : c.rec.operands) {
        if (o.written) continue;
        AdValue val;
        val.dtype = o.dtype;
        val.shape = o.shape;
        val.value.resize((size_t)numel(o.shape));
        for (double& x : val.value) x = device_check::urand(&seed) * 2.0 - 1.0;
        c.in.push_back(std::move(val));
    }
    // The reads in schema order: q, k, v, A, g, beta, h0, cu (, state_idx).
    AdValue& q = c.in[0];
    AdValue& k = c.in[1];
    AdValue& v = c.in[2];
    AdValue& A = c.in[3];
    AdValue& g = c.in[4];
    AdValue& beta = c.in[5];
    AdValue& h0 = c.in[6];
    AdValue& cu = c.in[7];

    cu.value[0] = 0.0;
    for (size_t n = 0; n < s.lens.size(); ++n) cu.value[n + 1] = cu.value[n] + (double)s.lens[n];
    h0.value.assign(h0.value.size(), 0.0);
    if (c.in.size() > 8)
        for (size_t n = 0; n < s.sidx.size(); ++n) c.in[8].value[n] = (double)s.sidx[n];

    // q and k: rounded, then L2-normalised per (token, key head) from the rounded values, then
    // rounded again, which is what the scan reads off the tape.
    for (int64_t tok = 0; tok < t; ++tok) {
        for (int64_t h = 0; h < hk; ++h) {
            for (AdValue* val : {&q, &k}) {
                double* row = &val->value[(size_t)((tok * hk + h) * K)];
                double ss = 0.0;
                for (int64_t d = 0; d < K; ++d) {
                    row[d] = round_dtype(row[d], "bf16");
                    ss += row[d] * row[d];
                }
                const double inv = 1.0 / std::sqrt(ss + 1e-6);
                for (int64_t d = 0; d < K; ++d) row[d] = round_dtype(row[d] * inv, "bf16");
            }
        }
    }

    // The gate: a within-chunk cumsum of log decays of a few per cent a token, so a 64-token
    // chunk decays by a factor of order e^{-2} and the state keeps a memory of its earlier chunks.
    // beta is a sigmoid of a small input, as the architecture issues it.
    for (int64_t n = 0; n < s.seqs(); ++n) {
        int64_t bos = 0;
        for (int64_t m = 0; m < n; ++m) bos += s.lens[m];
        std::vector<double> run((size_t)hv, 0.0);
        for (int64_t i = 0; i < s.lens[n]; ++i) {
            if (i % s.chunk == 0) run.assign((size_t)hv, 0.0);
            for (int64_t h = 0; h < hv; ++h) {
                const size_t at = (size_t)((bos + i) * hv + h);
                run[(size_t)h] -= 0.005 + 0.045 * std::fabs(g.value[at]);
                g.value[at] = round_dtype(run[(size_t)h], "f32");
            }
        }
    }
    for (double& x : beta.value) x = round_dtype(1.0 / (1.0 + std::exp(-2.0 * x)), "f32");

    fill_kkt(&A, k, beta, g, s);

    c.cot.push_back(device_check::random_value("bf16", {t, hv, V}, &seed));
    c.cot.push_back(device_check::random_value("f32", {s.state_slots(), hv, V, K}, &seed));

    // Every value at the dtype its operand declares, so the f64 reference sees the card's numbers.
    for (size_t i = 0; i < c.in.size(); ++i)
        for (double& x : c.in[i].value) x = round_dtype(x, c.rec.operands[i].dtype);
    for (AdValue& val : c.cot)
        for (double& x : val.value) x = round_dtype(x, val.dtype);
    return c;
}

// The largest deviation between the card's gradients and the f64 host row's over every operand the
// op reads: the number docs/kernels/gdn_scan.md quotes.
struct Deviation {
    double abs = 0.0;
    double rel = 0.0;
    std::string where;
};

Deviation deviation(const device_check::Case& c) {
    std::vector<AdValue> fwd_d, dc_d, fwd_h, dc_h;
    AdjointCall cd = device_check::make_call(c, &fwd_d, &dc_d);
    AdjointCall ch = device_check::make_call(c, &fwd_h, &dc_h);
    AdjointFn device = adjoint_find(c.rec.op, Side::device, c.rec.params, nullptr);
    AdjointFn host = adjoint_find(c.rec.op, Side::host, c.rec.params, nullptr);
    if (!device || !host) throw std::runtime_error(c.label + ": a registry row is missing");
    device(cd);
    host(ch);

    Deviation d;
    for (size_t slot = 0; slot < c.rec.operands.size() && slot < dc_d.size(); ++slot) {
        for (size_t i = 0; i < dc_d[slot].value.size(); ++i) {
            const double x = dc_d[slot].value[i];
            const double y = dc_h[slot].value[i];
            const double diff = std::fabs(x - y);
            if (diff > d.abs) {
                d.abs = diff;
                d.where = c.rec.operands[slot].buffer;
            }
            // A gradient that is itself near zero says nothing about a relative error.
            if (std::fabs(y) >= 1e-3) d.rel = std::max(d.rel, diff / std::fabs(y));
        }
    }
    return d;
}

// A wrong kernel: the gate's cotangent is dropped. The gate enters the scan through
// exp(g_i - g_j) across a chunk's whole triangle, the term of the reverse pass a kernel is most
// likely to leave out, so a comparison that accepted this row would accept anything.
void gate_dropped(const AdjointCall& call) {
    AdjointFn real = adjoint_find(call.rec.op, Side::device, call.rec.params, nullptr);
    if (!real) return;
    real(call);
    if (call.dc && call.dc->size() > 4)
        call.dc->at(4).value.assign(call.dc->at(4).value.size(), 0.0);
}

// One case: the harness's comparison against the host row, then the deviation the doc quotes.
void run_case(const ScanShape& s, uint64_t seed, bool measure) {
    const device_check::Case c = scan_case(s, s.name, seed);
    const device_check::Report r = device_check::check(c);
    if (!r.ok) throw std::runtime_error("gdn_chunk_scan " + std::string(s.name) + ": " + r.note);
    std::printf("gdn_chunk_scan %s: device == host, scratch peak %lld KiB\n", s.name,
                (long long)(r.scratch_peak / 1024));
    if (measure) {
        const Deviation d = deviation(c);
        std::printf("gdn_chunk_scan %s: max |device - host| %.3g (%s), max rel %.3g\n", s.name,
                    d.abs, d.where.c_str(), d.rel);
    }
    std::fflush(stdout);
}

void run_fixture() {
    const auto options = device_check::guard_options(int64_t(1) << 30);
    const r97test::DeviceGuard guard(options);
    device_check::require_device(options.device);
    (void)device_check::load_plugin();

    // h0, cu and state_idx are frozen: the walk asks for no cotangent there.
    const r97::OpRows rows = r97::op_rows("gdn_chunk_scan");
    R97_CHECK(rows.device_adjoint);
    R97_CHECK(rows.device_frozen == (std::vector<int>{6, 7, 10}));

    for (const ScanShape& s : fixture_shapes()) run_case(s, 0x9d1c0ffeeull, false);

    // The comparison has teeth: a kernel that drops the gate's cotangent is rejected there.
    const device_check::Report wrong =
        device_check::check(scan_case(fixture_shapes()[1], "gate dropped", 0x9d1c0ffeeull),
                            gate_dropped);
    R97_CHECK(!wrong.ok);
    R97_CHECK(wrong.first_slot == 4);
    std::printf("rejected the kernel that dropped the gate: %s\n", wrong.note.c_str());
}

void run_real() {
    const auto options = device_check::guard_options(int64_t(2) << 30);
    const r97test::DeviceGuard guard(options);
    device_check::require_device(options.device);
    (void)device_check::load_plugin();

    for (const ScanShape& s : real_shapes()) run_case(s, 0x5eed0badull, true);

    if (r97test::bench()) {
        // Both sides are measured over the same case, and the reference is f64 on the host: the
        // 4096-token case gets one repetition and the shorter ones a few.
        for (const ScanShape& s : real_shapes()) {
            const device_check::Case c = scan_case(s, s.name, 0x5eed0badull);
            device_check::print_bench(c, s.tokens() >= 4096 ? 1 : 3);
        }
    }
}

}  // namespace

R97_TEST(adjoint_gdn_scan_fixture) { run_fixture(); }

R97_TEST(adjoint_gdn_scan_real) { run_real(); }
