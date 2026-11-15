// tests/host/gdn_gradient_probe.cpp -- the per-element question the walk's growth curve cannot
// answer: is an op's adjoint the derivative of THAT OP'S OWN forward, element by element?
//
// The method is the one that found the int8 GEMM's missing activation scale
// (tests/host/scale_factor_probe.cpp): build the op's record, run its f64 twin for the forward,
// pick one element of one input, central-difference `sum over the writes of (y * dy)` with respect
// to that element, and divide the adjoint's `dc` for the same element by that difference. A row that
// is the derivative lands on adj/fd = 1; a row that dropped (or added) a factor s lands on s or 1/s.
// A host-vs-device comparison cannot see this class, because both twins share the factor.
//
// Two ops are probed here, and the difference between them is the point:
//
//   gdn_chunk_scan       the assignment's original candidate: the walk's `gdn_o` cotangent grows
//                        ~1e4 across the pair of records that straddle it. Measured here at
//                        fixture scale and over three chunks, with the live configuration's zero
//                        state cotangent and with a nonzero one.
//   gdn_gated_rmsnorm    the record that actually sits between those two measurements (step 1's
//                        finding): `gdn_o` is written by the scan, then read AND written by the
//                        rmsnorm, so the ~1e4 is this op's Jacobian on the cotangent it is handed.
//                        Probed here at the live channel width (128) on rows whose mean square is
//                        driven down to eps' own scale, which is where an RMS norm's conditioning
//                        lives and where its existing finite-difference test never goes (that one
//                        draws x uniform in [-1,1), i.e. rms ~ 0.58, sc ~ 1.7).
//
// The harness proves itself with a deliberately wrong dy: the adjoint is handed twice the cotangent
// the difference scores, and every probed ratio must come back exactly 2 -- not 1. Without that
// line, "adj/fd = 1" would also be what a harness that scored nothing would print.
//
// Label: host.
#include "../support/adjoint_check.h"
#include "../support/test.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

using namespace r97;
using namespace r97::adjoint_check;

namespace {

int64_t numel_of(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

/* One element to probe: a schema position, an element of it, the central difference's step, and
 * what the element is, so a report line reads without the fixture in view. `scored` is false for a
 * probe whose ratio is not a pass/fail criterion -- a frozen operand, or A's unread upper
 * triangle -- and which the test asserts about directly instead. */
struct Where {
    int schema = 0;
    int64_t elem = 0;
    double eps = 1e-6;
    bool scored = true;
    std::string note;
};

/* What one probe measured. `ratio` is adj/fd: 1 when the row is the derivative of the op's own
 * forward at this element, s when the row multiplied by a factor it should not have, 1/s when it
 * dropped one. */
struct Ratio {
    std::string note;
    double fd = 0.0;
    double adj = 0.0;
    double ratio = 0.0;
};

struct ProbeReport {
    std::vector<Ratio> ratios;
    double worst = 0.0;       // worst |adj - fd| / max(|adj|, |fd|) over the scored probes
    // The same over the probes whose difference is within an order of magnitude of the loss scale.
    // A probe whose |fd| is 1e-4 of the loss it is read out of spends nine orders of the difference
    // on cancellation, so its ratio is the finite difference's own noise, not the row's answer;
    // this subset is where the ratio can be read to 1e-6.
    double worst_big = 0.0;
    double amp = 0.0;         // max |dc| over the reads / max |cot| over the writes: the op's own
                              // buffer-max amplification, the number the walk's dump reports
    double max_abs_fd = 0.0;  // largest |fd| probed: proves the perturbations moved something
};

/* The cotangent the difference scores and the adjoint is handed, drawn from the seed alone so a
 * second call reproduces it exactly (the wrong-dy check doubles this list). `zero_at` names the
 * writes whose cotangent is zero, which is how the live configuration is reproduced: the walk's
 * record reports `ht` with no cotangent at all (nothing reads the final state of a prefill), and a
 * variant that only zeroed the write's *value* would still draw a random cotangent for it. */
std::vector<AdValue> cotangent_of(const std::vector<AdValue>& out_spec, uint64_t seed,
                                  const std::vector<int>& zero_at = {}) {
    uint64_t s = seed;
    std::vector<AdValue> cot(out_spec.size());
    for (size_t o = 0; o < out_spec.size(); ++o) {
        cot[o] = out_spec[o];
        cot[o].value.assign((size_t)numel_of(cot[o].shape), 0.0);
        for (double& v : cot[o].value) v = urand(&s);
    }
    for (int w : zero_at)
        if (w >= 0 && (size_t)w < cot.size())
            cot[(size_t)w].value.assign(cot[(size_t)w].value.size(), 0.0);
    return cot;
}

/* The whole probe for one record. `in` are the read operands in schema order and `out_spec` the
 * written ones (dtype and shape; their values come from the twin). When `adj_cot` is given, the
 * adjoint is handed THAT cotangent while the difference still scores the true one -- which is how
 * the harness proves it can see a wrong dy. `zero_cot` names the writes whose cotangent is zero
 * (see cotangent_of). */
ProbeReport probe(const OpRecord& rec, const std::vector<AdValue>& in,
                  const std::vector<AdValue>& out_spec, uint64_t seed,
                  const std::vector<Where>& where, const std::vector<AdValue>* adj_cot = nullptr,
                  const std::vector<int>& zero_cot = {}) {
    std::string err;
    const TwinFn twin = twin_find(rec.op, rec.params, &err);
    if (!twin) fail("no twin for '" + rec.op + "': " + err);
    const AdjointFn adj = adjoint_find(rec.op, Side::host, rec.params, &err);
    if (!adj) fail("no host adjoint for '" + rec.op + "': " + err);
    const std::vector<int> frozen = adjoint_frozen(rec.op, Side::host, rec.params);

    std::vector<AdValue> out_vals;
    twin(rec, in, &out_vals);
    if (out_vals.size() != out_spec.size())
        fail("the twin wrote " + std::to_string(out_vals.size()) + " operands, the record has " +
             std::to_string(out_spec.size()));

    // The forward over the whole schema, and each read's slot in `in`.
    std::vector<AdValue> fwd(rec.operands.size());
    std::vector<int> in_at(rec.operands.size(), -1);
    {
        size_t r = 0, w = 0;
        for (size_t i = 0; i < rec.operands.size(); ++i) {
            if (rec.operands[i].written) {
                fwd[i] = out_vals[w++];
                continue;
            }
            if (r >= in.size()) fail("the record reads more operands than the check was given");
            fwd[i] = in[r];
            in_at[i] = (int)r++;
        }
        if (r != in.size())
            fail("the record reads " + std::to_string(r) + " operands, the check was given " +
                 std::to_string(in.size()));
    }

    const std::vector<AdValue> cot = cotangent_of(out_spec, seed, zero_cot);
    std::vector<AdValue> dc(rec.operands.size());
    {
        size_t r = 0;
        for (size_t i = 0; i < rec.operands.size(); ++i) {
            if (rec.operands[i].written) continue;
            dc[i] = in[r++];
            dc[i].value.assign(dc[i].value.size(), 0.0);
        }
    }
    const std::vector<AdValue>& handed = adj_cot ? *adj_cot : cot;
    AdjointCall call{rec, fwd, handed, &dc, nullptr, ""};
    adj(call);

    // A row that writes a frozen operand has left the contract: those slots carry no payload in
    // the walk, so a write there is a gradient that silently vanishes.
    for (int f : frozen) {
        if (f < 0 || (size_t)f >= dc.size()) continue;
        for (double v : dc[(size_t)f].value)
            if (v != 0.0)
                fail("the row wrote the frozen operand at schema position " + std::to_string(f) +
                     " ('" + rec.operands[(size_t)f].buffer + "')");
    }

    ProbeReport rep;
    double max_dc = 0.0, max_cot = 0.0;
    for (size_t i = 0; i < dc.size(); ++i)
        for (double v : dc[i].value)
            if (std::isfinite(v)) max_dc = std::max(max_dc, std::fabs(v));
    for (const AdValue& c : cot)
        for (double v : c.value)
            if (std::isfinite(v)) max_cot = std::max(max_cot, std::fabs(v));
    rep.amp = max_cot > 0.0 ? max_dc / max_cot : 0.0;

    // L(t): the twin's writes, dotted with the cotangent, after in[slot].value[elem] += delta.
    const auto loss_at = [&](const Where& at, double delta) {
        const int slot = in_at[(size_t)at.schema];
        if (slot < 0) fail("the probe names a written operand: " + at.note);
        std::vector<AdValue> x = in;
        x[(size_t)slot].value[(size_t)at.elem] += delta;
        std::vector<AdValue> y;
        twin(rec, x, &y);
        double l = 0.0;
        for (size_t o = 0; o < y.size() && o < cot.size(); ++o)
            for (size_t e = 0; e < y[o].value.size() && e < cot[o].value.size(); ++e)
                l += y[o].value[e] * cot[o].value[e];
        return l;
    };

    for (const Where& at : where) {
        Ratio r;
        r.note = at.note;
        r.fd = (loss_at(at, +at.eps) - loss_at(at, -at.eps)) / (2.0 * at.eps);
        r.adj = dc[(size_t)at.schema].value[(size_t)at.elem];
        r.ratio = r.fd != 0.0 ? r.adj / r.fd : 0.0;
        rep.max_abs_fd = std::max(rep.max_abs_fd, std::fabs(r.fd));
        if (at.scored) {
            const double e = rel_err(r.adj, r.fd);
            rep.worst = std::max(rep.worst, e);
            if (std::fabs(r.fd) > 1.0) rep.worst_big = std::max(rep.worst_big, e);
        }
        rep.ratios.push_back(std::move(r));
    }
    return rep;
}

/* ---------------------------------------------------------------- gdn_chunk_scan */

struct ScanShape {
    const char* name;
    int64_t k, v, hk, hv, chunk;
    double scale;
    std::vector<int64_t> lens;

    int64_t tokens() const {
        int64_t t = 0;
        for (int64_t n : lens) t += n;
        return t;
    }
    int64_t seqs() const { return (int64_t)lens.size(); }
};

std::string f64_text(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
}

/* The record the walk's live rows carry, at a fixture's geometry: q, k, v, A, g, beta, the zero
 * initial state, the sequence boundaries, then the two writes. No `state_idx` (one slot per
 * sequence). `with_scale` false is the record the tape really carries: it holds integers and
 * strings and no double, so `scale` is absent and both the twin and the row fall back to the
 * architecture's 1/sqrt(head_k) (src/adjoint/host/gdn_scan.cpp:141). */
OpRecord scan_rec(const ScanShape& s, bool with_scale = true) {
    const int64_t t = s.tokens();
    OpRecord rec;
    rec.op = "gdn_chunk_scan";
    rec.params = {{"M", t}, {"head_k", s.k}, {"head_v", s.v}, {"chunk", s.chunk}};
    if (with_scale) rec.params.push_back({"scale", f64_text(s.scale)});
    rec.n = t;
    rec.operands = {
        {"q", "bf16", {t, s.hk, s.k}, 0, t, false},
        {"k", "bf16", {t, s.hk, s.k}, 0, t, false},
        {"v", "bf16", {t, s.hv, s.v}, 0, t, false},
        {"A", "bf16", {t, s.hv, s.chunk}, 0, t, false},
        {"g", "f32", {t, s.hv}, 0, t, false},
        {"beta", "f32", {t, s.hv}, 0, t, false},
        {"h0", "f32", {s.seqs(), s.hv, s.v, s.k}, 0, s.seqs(), false},
        {"cu", "i32", {s.seqs() + 1}, 0, 1, false},
        {"o", "bf16", {t, s.hv, s.v}, 0, t, true},
        {"ht", "f32", {s.seqs(), s.hv, s.v, s.k}, 0, s.seqs(), true},
    };
    return rec;
}

struct Case {
    std::vector<AdValue> in;
    std::vector<AdValue> out;
};

/* The live configuration's data: q, k, v and A random, `g` a within-chunk cumsum of negative
 * increments (so no exponent grows), `beta` a sigmoid, `h0` the zero state a prefill enters with,
 * and A unit lower triangular with random off-diagonals. */
Case scan_case(const OpRecord& rec, const ScanShape& s, uint64_t seed) {
    Case c;
    for (const Operand& o : rec.operands) {
        AdValue val;
        val.dtype = o.dtype;
        val.shape = o.shape;
        val.value.resize((size_t)numel_of(o.shape));
        for (double& x : val.value) x = urand(&seed);
        if (o.written) {
            c.out.push_back(std::move(val));
            continue;
        }
        if (o.buffer == "cu") {
            val.value[0] = 0.0;
            for (size_t n = 0; n < s.lens.size(); ++n)
                val.value[n + 1] = val.value[n] + (double)s.lens[n];
        } else if (o.buffer == "h0") {
            val.value.assign(val.value.size(), 0.0);
        } else if (o.buffer == "g") {
            std::vector<double> run((size_t)s.hv, 0.0);
            for (size_t n = 0; n < s.lens.size(); ++n) {
                int64_t bos = 0;
                for (size_t m = 0; m < n; ++m) bos += s.lens[m];
                for (int64_t t = 0; t < s.lens[n]; ++t) {
                    if (t % s.chunk == 0) run.assign((size_t)s.hv, 0.0);
                    for (int64_t h = 0; h < s.hv; ++h) {
                        const size_t at = (size_t)((bos + t) * s.hv + h);
                        run[(size_t)h] += -0.05 - 0.5 * std::fabs(val.value[at]);
                        val.value[at] = run[(size_t)h];
                    }
                }
            }
        } else if (o.buffer == "beta") {
            for (double& x : val.value) x = 1.0 / (1.0 + std::exp(-2.0 * x));
        } else if (o.buffer == "A") {
            for (size_t n = 0; n < s.lens.size(); ++n) {
                int64_t bos = 0;
                for (size_t m = 0; m < n; ++m) bos += s.lens[m];
                for (int64_t t = 0; t < s.lens[n]; ++t) {
                    const int64_t i = t % s.chunk;
                    for (int64_t h = 0; h < s.hv; ++h) {
                        const size_t base = (size_t)(((bos + t) * s.hv + h) * s.chunk);
                        val.value[base + (size_t)i] = 1.0;
                    }
                }
            }
        }
        c.in.push_back(std::move(val));
    }
    return c;
}

std::vector<Where> scan_where(const ScanShape& s) {
    const int64_t T = s.tokens();
    const int64_t H = s.hv, K = s.k, V = s.v, C = s.chunk;
    std::vector<Where> w;
    char note[96];
    const auto qk = [&](int schema, int64_t t, int64_t hk, int64_t d, const char* name) {
        Where x;
        x.schema = schema;
        x.elem = (t * s.hk + hk) * K + d;
        std::snprintf(note, sizeof note, "%s[t%lld,hk%lld,d%lld]", name, (long long)t,
                      (long long)hk, (long long)d);
        x.note = note;
        w.push_back(x);
    };
    const auto vv = [&](int64_t t, int64_t h, int64_t e) {
        Where x;
        x.schema = 2;
        x.elem = (t * H + h) * V + e;
        std::snprintf(note, sizeof note, "v[t%lld,h%lld,e%lld]", (long long)t, (long long)h,
                      (long long)e);
        x.note = note;
        w.push_back(x);
    };
    const auto aa = [&](int64_t t, int64_t h, int64_t i, int64_t j, bool read) {
        Where x;
        x.schema = 3;
        x.elem = (t * H + h) * C + i * C + j;
        x.scored = read;
        std::snprintf(note, sizeof note, "A[t%lld,h%lld,i%lld,j%lld]%s", (long long)t, (long long)h,
                      (long long)i, (long long)j, read ? "" : " (upper: unread)");
        x.note = note;
        w.push_back(x);
    };
    const auto row = [&](int schema, int64_t t, int64_t h, const char* name) {
        Where x;
        x.schema = schema;
        x.elem = t * H + h;
        std::snprintf(note, sizeof note, "%s[t%lld,h%lld]", name, (long long)t, (long long)h);
        x.note = note;
        w.push_back(x);
    };
    // Two tokens of q and k, one in the first chunk and one past a chunk boundary (so the state
    // carry is on the path the difference sees), and one in the short last chunk.
    qk(0, 0, 0, 0, "q");
    qk(0, 4 < T ? 4 : T - 1, 0, 2 < K ? 2 : 0, "q");
    qk(1, 1, 0, 1 < K ? 1 : 0, "k");
    qk(1, T - 1, 0, K - 1, "k");
    vv(0, 0, 0);
    vv(T > 2 ? T - 2 : T - 1, H - 1, V - 1);
    // A's element (row, col) is read only where `row` is the token's LOCAL index in its chunk
    // (load_chunk stores A[tj][i] for i <= local j = t % chunk), so a probe must sit on a token
    // whose local index is the row it names; otherwise the forward never reads it and the probe
    // scores a zero on both sides. The third probe is A's upper triangle, which is exactly that:
    // it is unread by construction and both sides must come back zero.
    {
        const int64_t t = 4 < T ? 4 : T - 1;  // past a chunk boundary: the state carry is live
        aa(t, 0, t % C, 1, true);
    }
    aa(T - 2, H - 1, (T - 2) % C, 0, true);
    aa(1 < T ? 1 : 0, 0, (1 < T ? 1 : 0) % C, (1 < T ? 1 : 0) % C + 1, false);
    row(4, T > 3 ? 3 : T - 1, 0, "g");              // first token of the second chunk
    row(5, T > 4 ? 4 : T - 1, H - 1, "beta");
    {
        // The entering state: the schema's S0. It is frozen -- the prefill starts from zero and the
        // cotangent leaving the first chunk is dropped -- so this probe is a design assertion, not
        // a scored ratio: the difference must be non-zero (the twin reads it) and the row must
        // leave its cotangent at zero.
        Where x;
        x.schema = 6;
        x.elem = 0;
        x.scored = false;
        x.note = "h0[slot0,h0,e0,d0] (frozen: prefill's zero state)";
        w.push_back(x);
    }
    return w;
}

/* The probe whose note carries the given marker. A control probe (a frozen operand, an unread
 * triangle) is asserted about by name rather than by index, so a fixture change cannot silently
 * point the assertion at some other element. */
const Ratio& by_note(const ProbeReport& r, const char* needle) {
    for (const Ratio& x : r.ratios)
        if (x.note.find(needle) != std::string::npos) return x;
    fail(std::string("no probe matching '") + needle + "'");
    return r.ratios[0];
}

void print_report(const char* op, const char* shape, const ProbeReport& r) {
    std::printf("%s %s: buffer-max amplification |dc|/|dy| = %.4g, worst |adj-fd|/max %.3g "
                "(%.3g over the probes with |fd| > 1)\n",
                op, shape, r.amp, r.worst, r.worst_big);
    for (const Ratio& x : r.ratios)
        std::printf("  %-40s fd %+.6g  adj %+.6g  adj/fd %+.6f\n", x.note.c_str(), x.fd, x.adj,
                    x.ratio);
}

}  // namespace

/* The scan's adjoint against the derivative of the scan's own forward, element by element: q, k, v,
 * A, g and beta, with the writes' cotangents live. Three chunks, so the state carry is exercised,
 * and the live configuration (a zero `ht` cotangent) alongside a random one. */
R97_TEST(gdn_gradient_probe_chunk_scan_matches_the_ops_own_derivative) {
    const ScanShape shapes[] = {
        {"three-chunks", 4, 4, 1, 2, 3, 0.5, {8}},
        {"ragged", 4, 6, 1, 2, 3, 0.25, {2, 7}},
        // The gpu test's off-tape shape (tests/gpu/adjoint_gdn_scan.cpp:91): head_k 16, so the
        // architecture's scale is 0.25 -- the value the record does not carry.
        {"off-tape-arch-scale", 16, 16, 2, 4, 4, 0.25, {3}},
    };
    for (const ScanShape& s : shapes) {
        const OpRecord rec = scan_rec(s);
        const Case c = scan_case(rec, s, 0x9d1c0ffee1ull);
        const std::vector<Where> where = scan_where(s);

        // The live configuration: the walk's record reports `ht` with no cotangent at all (nothing
        // reads a prefill's final state), so write 1 -- `ht` -- is handed zeros.
        const ProbeReport zero_state = probe(rec, c.in, c.out, 0x5eed0badull, where, nullptr, {1});
        print_report("gdn_chunk_scan", s.name, zero_state);

        // And with the state's cotangent live, so the reverse scan's state channel is on the path.
        const ProbeReport both = probe(rec, c.in, c.out, 0x5eed0badull, where);
        print_report("gdn_chunk_scan (ht live)", s.name, both);

        // The `scale`-less record, which is the one the tape hands the walk. Where the fixture's
        // explicit scale IS the architecture's 1/sqrt(head_k) (k=4 -> 0.5, k=16 -> 0.25) the absent
        // run must reproduce the explicit one value for value: that is what proves the fallback is
        // read by the twin AND the row, rather than one of them taking 1.0 or the other's scale.
        // `ragged`'s explicit 0.25 at k=4 is deliberately NOT the fallback, so it is the control
        // that a run with no `scale` does not silently inherit the fixture's.
        const double arch = 1.0 / std::sqrt((double)s.k);
        if (std::fabs(s.scale - arch) < 1e-12) {
            const OpRecord bare = scan_rec(s, false);
            const Case bc = scan_case(bare, s, 0x9d1c0ffee1ull);
            const ProbeReport off = probe(bare, bc.in, bc.out, 0x5eed0badull, where, nullptr, {1});
            print_report("gdn_chunk_scan (scale absent)", s.name, off);
            for (size_t i = 0; i < off.ratios.size(); ++i) {
                const double a = zero_state.ratios[i].adj, b = off.ratios[i].adj;
                if (!(std::fabs(a - b) <= 1e-9 * std::max(1.0, std::fabs(a))))
                    fail(std::string("gdn_chunk_scan ") + s.name + ": " + off.ratios[i].note +
                         ": the absent-`scale` run's cotangent " + std::to_string(b) +
                         " is not the explicit 1/sqrt(head_k) run's " + std::to_string(a));
            }
            if (!(off.worst < 1e-6))
                fail(std::string("gdn_chunk_scan ") + s.name +
                     ": with `scale` absent the adjoint is not the derivative of the forward: "
                     "worst relative error " +
                     std::to_string(off.worst));
        }

        // The control that the two variants are actually different runs: if zeroing `ht`'s
        // cotangent changed no answer, the state channel was never on the path and the pair above
        // would be one measurement printed twice.
        int64_t moved = 0;
        for (size_t i = 0; i < zero_state.ratios.size(); ++i)
            if (zero_state.ratios[i].adj != both.ratios[i].adj) ++moved;
        if (moved == 0)
            fail(std::string("gdn_chunk_scan ") + s.name +
                 ": zeroing ht's cotangent changed no cotangent, so the state channel is not on "
                 "the path this check claims to cover");
        std::printf("gdn_chunk_scan %s: %lld of %zu cotangents moved with ht's cotangent live\n",
                    s.name, (long long)moved, zero_state.ratios.size());

        for (const ProbeReport* r : {&zero_state, &both}) {
            if (!(r->worst < 1e-6))
                fail(std::string("gdn_chunk_scan ") + s.name +
                     ": the adjoint is not the derivative of the op's own forward: worst relative "
                     "error " +
                     std::to_string(r->worst));
            if (!(r->max_abs_fd > 1e-6))
                fail(std::string("gdn_chunk_scan ") + s.name +
                     ": every probed difference was zero, so the check scored nothing");
            // The frozen state and the unread upper triangle: zero cotangent, as the schema says.
            const Ratio& h0 = by_note(*r, "h0[slot0");
            if (h0.adj != 0.0)
                fail("gdn_chunk_scan: the row wrote h0's cotangent, and h0 is frozen");
            if (!(std::fabs(h0.fd) > 1e-9))
                fail("gdn_chunk_scan: perturbing h0 moved nothing, so the twin does not read the "
                     "entering state and the freeze assertion is vacuous");
            const Ratio& upper = by_note(*r, "(upper: unread)");
            if (upper.adj != 0.0 || upper.fd != 0.0)
                fail("gdn_chunk_scan: A's upper triangle took a cotangent or moved the forward");
        }
    }
}

/* ------------------------------------------------------------ gdn_gated_rmsnorm */

struct RmsShape {
    const char* name;
    int64_t rows;
    int64_t channels;
    const char* act;
    double w_max;               // the norm weight's magnitude: w[i] = +/- w_max
    std::vector<double> row_ms; // mean(x^2) per row
};

OpRecord rms_rec(const RmsShape& s) {
    OpRecord rec;
    rec.op = "gdn_gated_rmsnorm";
    rec.params = {
        {"M", s.rows}, {"channels", s.channels}, {"act", std::string(s.act)}, {"dtype", "bf16"}};
    rec.n = s.rows;
    rec.operands = {
        {"x", "bf16", {s.rows, s.channels}, 0, s.rows, false},
        {"z", "bf16", {s.rows, s.channels}, 0, s.rows, false},
        {"w", "f32", {s.channels}, 0, 1, false},
        {"o", "bf16", {s.rows, s.channels}, 0, s.rows, true},
    };
    return rec;
}

/* x per row drawn random and then scaled so the row's mean square is exactly the one asked for --
 * that mean is what sets sc = rsqrt(mean + eps) and therefore the row's conditioning; z uniform in
 * [-2, 2]; w alternating sign at the given magnitude. */
Case rms_case(const OpRecord& rec, const RmsShape& s, uint64_t seed) {
    Case c;
    for (const Operand& o : rec.operands) {
        AdValue val;
        val.dtype = o.dtype;
        val.shape = o.shape;
        val.value.resize((size_t)numel_of(o.shape));
        for (double& x : val.value) x = urand(&seed);
        if (o.written) {
            c.out.push_back(std::move(val));
            continue;
        }
        if (o.buffer == "x") {
            for (int64_t r = 0; r < s.rows; ++r) {
                double ss = 0.0;
                for (int64_t i = 0; i < s.channels; ++i) {
                    const double v = val.value[(size_t)(r * s.channels + i)];
                    ss += v * v;
                }
                const double want = (double)s.channels * s.row_ms[(size_t)r];
                const double k = ss > 0.0 ? std::sqrt(want / ss) : 0.0;
                for (int64_t i = 0; i < s.channels; ++i) val.value[(size_t)(r * s.channels + i)] *= k;
            }
        } else if (o.buffer == "z") {
            for (double& x : val.value) x *= 2.0;
        } else if (o.buffer == "w") {
            for (int64_t i = 0; i < s.channels; ++i)
                val.value[(size_t)i] = (i % 2 == 0 ? 1.0 : -1.0) * s.w_max;
        }
        c.in.push_back(std::move(val));
    }
    return c;
}

std::vector<Where> rms_where(const RmsShape& s) {
    std::vector<Where> w;
    char note[128];
    for (int64_t r = 0; r < s.rows; ++r) {
        const double m = s.row_ms[(size_t)r];
        // x's step is relative: a row at mean square 1e-8 has elements near 1e-4, and an absolute
        // 1e-6 there would be a 1% perturbation of the element the difference is taken about.
        for (const int64_t i : {int64_t{0}, s.channels / 2}) {
            Where x;
            x.schema = 0;
            x.elem = r * s.channels + i;
            const double xv = std::fabs(std::sqrt(m));
            x.eps = std::max(1e-6 * xv, 1e-11);
            std::snprintf(note, sizeof note, "x[r%lld,i%lld] m=%.0e", (long long)r, (long long)i, m);
            x.note = note;
            w.push_back(x);
        }
        for (const int64_t i : {int64_t{0}, s.channels / 2}) {
            Where x;
            x.schema = 1;
            x.elem = r * s.channels + i;
            x.eps = 1e-6;
            std::snprintf(note, sizeof note, "z[r%lld,i%lld] m=%.0e", (long long)r, (long long)i, m);
            x.note = note;
            w.push_back(x);
        }
    }
    return w;
}

/* The rmsnorm's adjoint against the derivative of its own forward, at the live channel width and on
 * rows whose mean square walks down to eps' own scale (1e-6). This is where the op's Jacobian is
 * large: sc is capped at 1/sqrt(eps) = 1000, so a near-zero row times a norm weight of magnitude 10
 * multiplies its cotangent by up to ~1e4 -- the size of the walk's unexplained jump. If the row is
 * the exact derivative, adj/fd stays 1 while that amplification happens, and the jump is
 * conditioning rather than a bug; if it is not, the ratio names the factor. */
R97_TEST(gdn_gradient_probe_gated_rmsnorm_matches_the_ops_own_derivative) {
    const RmsShape shapes[] = {
        {"live-conditioning", 6, 128, "silu", 10.0, {1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e-6}},
        {"ordinary-rows", 3, 128, "silu", 1.0, {0.3, 1.0, 3.0}},
    };
    for (const RmsShape& s : shapes) {
        const OpRecord rec = rms_rec(s);
        const Case c = rms_case(rec, s, 0x9d1c0ffee1ull);

        // The forward itself, against radiance's own kernel at this fixture: a twin that is right
        // about a formula the kernel does not run would make the adjoint check agree with itself.
        const KernelReport k = check_kernel(rec, c.in, c.out, 0.02);
        if (!k.ok) fail(std::string("gdn_gated_rmsnorm ") + s.name + ": " + k.note);
        std::printf("gdn_gated_rmsnorm %s: kernel vs twin max err %.3g\n", s.name, k.max_err);

        const ProbeReport r = probe(rec, c.in, c.out, 0x5eed0badull, rms_where(s));
        print_report("gdn_gated_rmsnorm", s.name, r);
        // Two criteria, because the probes are not equally well conditioned. Where the difference is
        // of the loss's own size (|fd| > 1) the ratio is readable to 1e-6; the rest are probes whose
        // derivative is 1e-3..1e-4 of the loss they were read out of, where 1e-4 is already ten
        // orders tighter than any factor this test exists to catch.
        if (!(r.worst_big < 1e-6))
            fail(std::string("gdn_gated_rmsnorm ") + s.name +
                 ": the adjoint is not the derivative of the op's own forward: worst relative "
                 "error " +
                 std::to_string(r.worst_big) + " over the well-conditioned probes");
        if (!(r.worst < 1e-4))
            fail(std::string("gdn_gated_rmsnorm ") + s.name +
                 ": the adjoint is not the derivative of the op's own forward: worst relative "
                 "error " +
                 std::to_string(r.worst));
        if (!(r.max_abs_fd > 1e-9))
            fail(std::string("gdn_gated_rmsnorm ") + s.name +
                 ": every probed difference was zero, so the check scored nothing");
    }
}

/* The harness's own proof: hand the adjoint a cotangent the difference does not score and the ratio
 * must move by exactly that factor. A harness that scored nothing, or that read the adjoint's `dc`
 * from the wrong element, would print 1 here too -- and that is the failure mode this line rules
 * out. */
R97_TEST(gdn_gradient_probe_detects_a_wrong_cotangent) {
    const ScanShape s = {"wrong-dy", 4, 4, 1, 2, 3, 0.5, {8}};
    const OpRecord rec = scan_rec(s);
    const Case c = scan_case(rec, s, 0x9d1c0ffee1ull);
    std::vector<Where> where = scan_where(s);
    where.resize(4);  // q, q, k, k: four elements with four different non-zero differences

    const ProbeReport right = probe(rec, c.in, c.out, 0x5eed0badull, where);
    if (!(right.worst < 1e-6))
        fail("the true cotangent's ratios are not 1, so the wrong-dy check would mean nothing");

    std::vector<AdValue> twice = cotangent_of(c.out, 0x5eed0badull);
    for (AdValue& v : twice)
        for (double& x : v.value) x *= 2.0;
    const ProbeReport wrong = probe(rec, c.in, c.out, 0x5eed0badull, where, &twice);
    print_report("gdn_chunk_scan", "wrong dy (2x)", wrong);
    for (const Ratio& r : wrong.ratios) {
        if (!(std::fabs(r.ratio - 2.0) < 1e-6))
            fail("a doubled cotangent did not double the ratio: " + r.note + " gave adj/fd " +
                 std::to_string(r.ratio) + " (2 is the wrong dy's signature, 1 means the harness "
                 "scores something other than what it hands the row)");
        if (!(std::fabs(r.ratio - 1.0) > 0.5))
            fail("a doubled cotangent still read as correct: the check cannot see a wrong dy");
    }
    // And the same metric the row checks assert on is the one that moved: a row that multiplied its
    // cotangent by two would put this run's own `worst` at 0.5, nine orders past the 1e-6 the row
    // checks accept, so those criteria are live rather than decorative.
    if (!(wrong.worst > 1e-6))
        fail("the wrong-dy run's worst |adj-fd|/max is below the row checks' tolerance, so their "
             "criteria would not catch a wrong row");
    std::printf("wrong dy: worst |adj-fd|/max %.3g (the row checks accept 1e-6)\n", wrong.worst);
}
