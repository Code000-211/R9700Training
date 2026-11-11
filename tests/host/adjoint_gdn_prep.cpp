// tests/host/adjoint_gdn_prep.cpp -- the gdn_prep family (src/adjoint/host/gdn_prep.cpp)
// through both checks of tests/support/adjoint_check.h: each op's f64 twin against
// radiance's reference kernel for that op, and each host adjoint against finite differences
// of its twin. Three shapes apiece, one of them the tiny fixture's gdn geometry. Then the
// three gdn ops that only prepare decode state: docs/adjoints/gdn-math.md says no prefill
// loss reads what they write, and this family registers them as stops.
#include "support/adjoint_check.h"
#include "support/test.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace r97;
using namespace r97::adjoint_check;

namespace {

/* bf16 carries 8 significand bits, so a kernel that rounds as it goes lands within
 * about 2^-9 relative of the f64 twin; 0.02 leaves the sum's rounding and the store
 * room without hiding an op that computes something else. */
const double bf16_tol = 0.02;

/* The twin is exact in f64, so central differences reproduce its directional
 * derivative far inside 1e-6 relative. */
const double fd_tol = 1e-6;

int64_t numel_of(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

/* One gdn_conv_prep geometry. `k` and `v` are the per-head widths, `hk` and `hv`
 * the head counts, `chunk` the gate's chunk length and `width` the convolution's;
 * `lens` is one sequence length per sequence in the batch. */
struct ConvShape {
    const char* name;
    int64_t k;
    int64_t v;
    int64_t hk;
    int64_t hv;
    int64_t width;
    int64_t chunk;
    std::vector<int64_t> lens;

    int64_t conv_dim() const { return 2 * hk * k + hv * v; }
    int64_t tokens() const {
        int64_t t = 0;
        for (int64_t n : lens) t += n;
        return t;
    }
    int64_t seqs() const { return (int64_t)lens.size(); }
};

/* The tiny fixture's gdn geometry is the container's: two key heads and four value
 * heads of 16, a 4-wide convolution, 4-token chunks, so the conv row is the
 * fixture's whole hidden width of 128. The other two keep the same op but move
 * everything else: two sequences with one shorter than the convolution's window,
 * and a single head with the widths the other way round.
 *
 * Built per call, like everything else the tests hand the checkers, rather than once
 * in a namespace-scope object. */
std::vector<ConvShape> conv_shapes() {
    return {
        {"tiny-fixture", 16, 16, 2, 4, 4, 4, {8}},
        {"ragged-short", 8, 8, 2, 2, 3, 2, {5, 1}},
        {"one-head", 4, 8, 1, 1, 2, 7, {3}},
    };
}

OpRecord conv_rec(const ConvShape& s) {
    const int64_t t = s.tokens();
    const int64_t cw = s.conv_dim();
    const int64_t seq = s.seqs();
    const int64_t keep = s.width - 1;
    OpRecord rec;
    rec.op = "gdn_conv_prep";
    rec.params = {{"M", t},
                  {"head_k", s.k},
                  {"head_v", s.v},
                  {"chunk", s.chunk},
                  {"conv_width", s.width},
                  {"dtype", "bf16"}};
    rec.n = t;
    rec.operands = {
        {"x", "bf16", {t, cw}, 0, t, false},
        {"w", "bf16", {cw, s.width}, 0, cw, false},
        {"b", "bf16", {cw}, 0, 1, false},
        {"A_log", "f32", {s.hv}, 0, 1, false},
        {"dt_bias", "f32", {s.hv}, 0, 1, false},
        {"conv_state", "bf16", {seq, cw, keep}, 0, seq, true},
        {"cu", "i32", {seq + 1}, 0, 1, false},
        {"a", "bf16", {t, s.hv}, 0, t, false},
        {"b_gate", "bf16", {t, s.hv}, 0, t, false},
        {"cache_idx", "i32", {seq}, 0, seq, false},
        {"has_init", "i32", {seq}, 0, seq, false},
        {"q", "bf16", {t, s.hk, s.k}, 0, t, true},
        {"k", "bf16", {t, s.hk, s.k}, 0, t, true},
        {"v", "bf16", {t, s.hv, s.v}, 0, t, true},
        {"g", "f32", {t, s.hv}, 0, t, true},
        {"beta", "f32", {t, s.hv}, 0, t, true},
    };
    return rec;
}

struct Case {
    std::vector<AdValue> in;
    std::vector<AdValue> out;
};

/* What the checkers are given. Every operand is drawn from `seed`, except the
 * three that name structure rather than data: `cu` holds the sequence boundaries,
 * `cache_idx` gives each sequence its own slot, and `has_init` is 0 because the
 * forward this family serves is a prefill from position 0, where the carried
 * convolution state is zero. */
Case conv_case(const OpRecord& rec, const ConvShape& s, uint64_t seed) {
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
        } else if (o.buffer == "cache_idx") {
            for (size_t n = 0; n < val.value.size(); ++n) val.value[n] = (double)n;
        } else if (o.buffer == "has_init") {
            val.value.assign(val.value.size(), 0.0);
        }
        c.in.push_back(std::move(val));
    }
    return c;
}

/* gdn_gated_rmsnorm works one row per (token, head), so a shape is its channel
 * count and its row count. The first is the fixture's: its 4 value heads over 8
 * tokens of 16 channels, the value head width. Built per call, like the
 * convolution's shapes, and for the same reason. */
struct RmsShape {
    const char* name;
    int64_t rows;
    int64_t channels;
    const char* act;
};

std::vector<RmsShape> rms_shapes() {
    return {
        {"tiny-fixture", 32, 16, "silu"},
        {"wide", 37, 192, "silu"},
        {"narrow-sigmoid", 3, 8, "sigmoid"},
    };
}

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

Case rms_case(const OpRecord& rec, uint64_t seed) {
    Case c;
    for (const Operand& o : rec.operands) {
        AdValue val;
        val.dtype = o.dtype;
        val.shape = o.shape;
        val.value.resize((size_t)numel_of(o.shape));
        for (double& x : val.value) x = urand(&seed);
        (o.written ? c.out : c.in).push_back(std::move(val));
    }
    return c;
}

void require_kernel(const ConvShape& s, const KernelReport& r) {
    if (!r.ok) fail(std::string("gdn_conv_prep ") + s.name + ": " + r.note);
}

void require_kernel(const RmsShape& s, const KernelReport& r) {
    if (!r.ok) fail(std::string("gdn_gated_rmsnorm ") + s.name + ": " + r.note);
}

void require_fd(const ConvShape& s, const FdReport& r) {
    if (!r.ok) fail(std::string("gdn_conv_prep ") + s.name + ": " + r.note);
}

void require_fd(const RmsShape& s, const FdReport& r) {
    if (!r.ok) fail(std::string("gdn_gated_rmsnorm ") + s.name + ": " + r.note);
}

}  // namespace

R97_TEST(adjoint_gdn_conv_prep_kernel_agrees_with_the_twin) {
    for (const ConvShape& s : conv_shapes()) {
        const OpRecord rec = conv_rec(s);
        const Case c = conv_case(rec, s, 0x9d1c0ffee1ull);
        require_kernel(s, check_kernel(rec, c.in, c.out, bf16_tol));
    }
}

R97_TEST(adjoint_gdn_conv_prep_agrees_with_finite_differences) {
    for (const ConvShape& s : conv_shapes()) {
        const OpRecord rec = conv_rec(s);
        const Case c = conv_case(rec, s, 0x5eed0badull);
        require_fd(s, check_adjoint(rec, c.in, c.out, fd_tol));
    }
}

R97_TEST(adjoint_gdn_gated_rmsnorm_kernel_agrees_with_the_twin) {
    for (const RmsShape& s : rms_shapes()) {
        const OpRecord rec = rms_rec(s);
        const Case c = rms_case(rec, 0x1234abcdull);
        require_kernel(s, check_kernel(rec, c.in, c.out, bf16_tol));
    }
}

R97_TEST(adjoint_gdn_gated_rmsnorm_agrees_with_finite_differences) {
    for (const RmsShape& s : rms_shapes()) {
        const OpRecord rec = rms_rec(s);
        const Case c = rms_case(rec, 0x7777aaaaull);
        require_fd(s, check_adjoint(rec, c.in, c.out, fd_tol));
    }
}

/* The decode ops write decode rows' q, k, v, g, beta and the caches, and no prefill
 * row's loss reads any of it (docs/adjoints/gdn-math.md, section 5). So each is
 * registered as a stop: the lookup succeeds, so a deliberate stop reads differently
 * from a family nobody has written, and the row adds nothing to any cotangent. */
R97_TEST(adjoint_gdn_decode_ops_are_stops) {
    const std::vector<OpParam> p = {{"M", int64_t{4}}, {"dtype", std::string("bf16")}};
    const char* ops[] = {"gdn_conv_update", "gdn_recurrent_update", "gdn_conv_recurrent_update"};
    for (const char* op : ops) {
        std::string error;
        const AdjointFn adj = adjoint_find(op, Side::host, p, &error);
        R97_CHECK(adj != nullptr);
        R97_CHECK(error.empty());

        /* No forward in the loss path runs them, so nothing to twin. */
        std::string twin_error;
        R97_CHECK(twin_find(op, p, &twin_error) == nullptr);

        OpRecord rec;
        rec.op = op;
        rec.params = p;
        const std::vector<AdValue> fwd(2, AdValue{"bf16", {2, 2}, {1.0, 2.0, 3.0, 4.0}});
        const std::vector<AdValue> cot(1, AdValue{"bf16", {2, 2}, {1.0, 1.0, 1.0, 1.0}});
        std::vector<AdValue> dc(2, AdValue{"bf16", {2, 2}, std::vector<double>(4, 0.0)});
        adj(AdjointCall{rec, fwd, cot, &dc, nullptr, ""});
        for (const AdValue& d : dc)
            for (double v : d.value) R97_CHECK(v == 0.0);
    }
}
