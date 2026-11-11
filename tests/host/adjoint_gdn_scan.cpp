// tests/host/adjoint_gdn_scan.cpp -- the gdn_scan family (src/adjoint/host/gdn_scan.cpp)
// through both checks of tests/support/adjoint_check.h: the f64 twin against radiance's
// reference kernel for gdn_chunk_scan, and the host adjoint against central finite
// differences of that twin. The shapes cover a sequence shorter than a chunk, exactly one
// chunk, several chunks and a ragged batch of different lengths; the first is the tiny
// fixture's gdn geometry. One shape indexes a state cache larger than the batch through the
// optional `state_idx` read, and one test pins the scale the architecture issues when a
// record off the tape carries none.
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

/* bf16 carries 8 significand bits, so a kernel that rounds as it goes lands within
 * about 2^-9 relative of the f64 twin; 0.02 leaves the sums room without hiding an
 * op that computes something else. */
const double bf16_tol = 0.02;

/* The issue fixes this tighter than the family standard: the twin is exact in f64
 * and the adjoint was validated against central differences to 1e-9 relative. */
const double fd_tol = 1e-8;

int64_t numel_of(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

/* One gdn_chunk_scan geometry. `k` and `v` are the per-head widths, `hk` and `hv`
 * the head counts, `chunk` the gate's chunk length, `scale` the record's scale
 * parameter; `lens` is one sequence length per sequence in the batch. `slots` is
 * the state cache's slot count (0 for one slot per sequence) and `sidx` the
 * per-sequence slot indices (empty for no `state_idx` operand). */
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

/* The first is the tiny fixture's gdn geometry -- two key heads and four value
 * heads of 16, 4-token chunks -- with a sequence shorter than its chunk. The
 * others put a sequence exactly on a chunk boundary, one across several chunks,
 * and a ragged batch whose three lengths cover shorter, equal and several. Built per
 * call, like everything else the tests hand the checkers. */
std::vector<ScanShape> scan_shapes() {
    return {
        {"tiny-fixture-short", 16, 16, 2, 4, 4, 0.5, {3}},
        {"one-chunk", 8, 8, 2, 2, 5, 0.25, {5}},
        {"several-chunks", 4, 6, 1, 2, 3, 0.125, {11}},
        {"ragged", 8, 8, 2, 2, 5, 0.75, {2, 7, 13}},
        /* A state pool deeper than the batch: sequence 0 keeps slot 2 and
         * sequence 1 slot 0, so slot 1 stays untouched and a twin that ignores
         * `state_idx` writes and reads the wrong slot. */
        {"state-pool", 8, 8, 1, 2, 4, 0.5, {5, 4}, 3, {2, 0}},
    };
}

/* A parameter's text, exactly representable: the f64 parameters travel as strings
 * on the tape, which carries integers and strings and drops a double. */
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
    /* The optional read is schema position 10, after the two writes. */
    if (!s.sidx.empty())
        rec.operands.push_back({"state_idx", "i32", {seq}, 0, 1, false});
    return rec;
}

struct Case {
    std::vector<AdValue> in;
    std::vector<AdValue> out;
};

/* What the checkers are given. q, k, v and A are drawn from `seed`; the data that
 * names structure is not: `cu` holds the sequence boundaries, h0 is the zero
 * initial state this family serves, g is a within-chunk cumsum of negative gate
 * increments (so no exponent grows), beta is a sigmoid, and A is unit lower
 * triangular with random off-diagonal entries. A's upper triangle is left random
 * as well: the op never reads it, and the finite-difference run must then report
 * a zero cotangent there. */
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
        } else if (o.buffer == "state_idx") {
            for (size_t n = 0; n < s.sidx.size(); ++n) val.value[n] = (double)s.sidx[n];
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
                        const double inc = -0.05 - 0.5 * std::fabs(val.value[at]);
                        run[(size_t)h] += inc;
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
                        if (i < s.chunk) val.value[base + (size_t)i] = 1.0;
                    }
                }
            }
        }
        c.in.push_back(std::move(val));
    }
    return c;
}

void require_kernel(const ScanShape& s, const KernelReport& r) {
    if (!r.ok) fail(std::string("gdn_chunk_scan ") + s.name + ": " + r.note);
    std::printf("gdn_chunk_scan %s: kernel max err %.3g\n", s.name, r.max_err);
}

void require_fd(const ScanShape& s, const FdReport& r) {
    if (!r.ok) fail(std::string("gdn_chunk_scan ") + s.name + ": " + r.note);
    std::printf("gdn_chunk_scan %s: adjoint rel err %.2e\n", s.name, r.rel);
}

}  // namespace

R97_TEST(adjoint_gdn_chunk_scan_kernel_agrees_with_the_twin) {
    for (const ScanShape& s : scan_shapes()) {
        const OpRecord rec = scan_rec(s);
        const Case c = scan_case(rec, s, 0x9d1c0ffee1ull);
        require_kernel(s, check_kernel(rec, c.in, c.out, bf16_tol));
    }
}

R97_TEST(adjoint_gdn_chunk_scan_agrees_with_finite_differences) {
    for (const ScanShape& s : scan_shapes()) {
        const OpRecord rec = scan_rec(s);
        const Case c = scan_case(rec, s, 0x5eed0badull);
        require_fd(s, check_adjoint(rec, c.in, c.out, fd_tol));
    }
}

/* The architecture issues this op with `scale = 1/sqrt(head_k)`, and the tape
 * drops an f64 parameter, so a record off a recorded step carries no `scale` at
 * all. The twin must then use the architecture's value -- the one the kernel ran
 * with -- and not the op's own default of 1, or every scan's output is
 * sqrt(head_k) too large. */
R97_TEST(adjoint_gdn_chunk_scan_defaults_scale_to_the_architectures) {
    const ScanShape s = {"one-chunk", 8, 8, 2, 2, 5, 0.25, {5}};
    const OpRecord rec = scan_rec(s);
    const Case c = scan_case(rec, s, 0x5ca1e5a11ull);

    std::string note;
    const TwinFn twin = twin_find(rec.op, rec.params, &note);
    if (!twin) fail(note);

    auto at_scale = [&](bool present, double v) {
        OpRecord r = rec;
        r.params.erase(std::remove_if(r.params.begin(), r.params.end(),
                                      [](const OpParam& p) { return p.first == "scale"; }),
                       r.params.end());
        if (present) r.params.push_back({"scale", f64_text(v)});
        return r;
    };
    auto twin_out = [&](const OpRecord& r) {
        std::vector<AdValue> y;
        twin(r, c.in, &y);
        return y;
    };

    const double arch = 1.0 / std::sqrt((double)s.k);
    const std::vector<AdValue> absent = twin_out(at_scale(false, 0.0));
    const std::vector<AdValue> explicit_arch = twin_out(at_scale(true, arch));
    const std::vector<AdValue> explicit_one = twin_out(at_scale(true, 1.0));
    if (absent.size() != explicit_arch.size() || absent.empty() || absent[0].value.empty())
        fail("the twin's output count changed with the scale");

    double worst = 0.0;
    for (size_t e = 0; e < absent[0].value.size(); ++e)
        worst = std::max(worst, std::fabs(absent[0].value[e] - explicit_arch[0].value[e]));
    if (worst > 1e-12)
        fail("a record without `scale` is not the one at 1/sqrt(head_k): off by " +
             std::to_string(worst));

    /* o is linear in the scale, so the default-1 twin is the arch's times sqrt(head_k).
     * A twin that ignores `scale` entirely would match the arch's output instead. */
    double linear = 0.0;
    for (size_t e = 0; e < absent[0].value.size(); ++e)
        linear = std::max(linear, std::fabs(explicit_one[0].value[e] - absent[0].value[e] / arch));
    if (linear > 1e-9)
        fail("the twin's default-1 output is not the arch's times sqrt(head_k): off by " +
             std::to_string(linear));

    std::printf("gdn_chunk_scan default scale: 1/sqrt(%lld) = %.6g, o off by %.2e\n",
                (long long)s.k, arch, linear);
}
