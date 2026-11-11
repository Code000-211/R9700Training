// tests/host/adjoint_gdn_kkt.cpp -- `gdn_kkt_solve` (src/adjoint/host/gdn_kkt.cpp) through both
// checks of tests/support/adjoint_check.h: its f64 twin against radiance's reference kernel, and
// its host adjoint against central finite differences of the twin. Then the two things the issue
// names that the checks alone cannot say: a chunk's cotangent depends on its own chunk and no
// other, so a batch of sequences of different lengths answers each sequence the way running that
// sequence alone does; and cu_seqlens is frozen, so nothing is answered for it.
//
// The shapes are the tiny fixture's delta-net geometry (2 key heads over 4 value heads, 16 lanes,
// chunk 64, tests/support/fixture.h) with sequence lengths below, at and above the op's chunk size
// and a short last chunk, a batch of sequences of different lengths, and one head geometry without
// grouping (one key head per value head) and wider keys.
//
// The family's tolerance is 1e-8 relative in f64 and not the standard 1e-6: the solve runs along a
// whole chunk and the rounding of one row is carried into the next, so only a check that tight can
// tell the backward's transposed solve from a near miss.
#include "support/adjoint_check.h"
#include "support/test.h"

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

/* k and A are bf16 in the device row, and the twin is exact in f64 over the values it is given.
 * The keys are filled on bf16's own grid (8 significand bits; entries are below 1 in magnitude),
 * so the kernel and the twin read the same keys and the tolerance covers the arithmetic and A's
 * bf16 store, not a hidden quantisation of the input. */
const double bf16_tol = 0.02;

/* The issue's tolerance for this family: errors compound over a sequence, so 1e-8 and not the
 * standard 1e-6. */
const double fd_tol = 1e-8;

/* The step the check below takes, with half of it for the refinement. The shared checker's 1e-6 is
 * what the standard 1e-6 tolerance needs, and at that step this op's second-order term alone is
 * 1.4e-8 of the answer: a 64-token solve is a 64-deep recurrence, so along a direction over eight
 * thousand operands its third derivative is large. Refining the step cancels that term; the step
 * itself is then only large enough to keep the twin's f64 roundoff out of the difference, and every
 * step from 3e-5 to 1.5e-4 answers the same to about 1e-9. */
const double fd_eps = 5e-5;

int64_t count(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (const int64_t d : shape) n *= d;
    return n;
}

/* One geometry the op has to hold for: the key and value head counts, the key's lanes, the op's
 * chunk size (A's last dimension), and each sequence's length. */
struct Shape {
    const char* name;
    int64_t k_heads;
    int64_t heads;
    int64_t lanes;
    int64_t chunk;
    std::vector<int64_t> lens;
};

/* What the checkers are given: the record, the read operands' values in schema order, and the
 * written operand's dtype and shape. */
struct Case {
    OpRecord rec;
    std::vector<AdValue> in;
    std::vector<AdValue> out;
    int64_t rows = 0;
};

AdValue operand(std::vector<int64_t> shape, const char* dtype) {
    AdValue v;
    v.dtype = dtype;
    v.shape = std::move(shape);
    v.value.resize((size_t)count(v.shape));
    return v;
}

void fill_random(AdValue* v, uint64_t* seed) {
    for (double& x : v->value) x = urand(seed);
}

/* The nearest bf16 value, for a magnitude below 1: bf16 keeps 8 significand bits, so a multiple of
 * 2^-9 is one. */
double bf16(double x) { return std::round(x * 512.0) / 512.0; }

int64_t rows_of(const Shape& s) {
    int64_t n = 0;
    for (const int64_t len : s.lens) n += len;
    return n;
}

/* A call as a training step records one, over values the caller supplies: rows of keys (one unit
 * row per token and key head), of beta (a sigmoid, so strictly between 0 and 1), and of the log
 * decay (a per-token decrease, cumsummed inside each chunk and starting again at every chunk
 * boundary, which is what GATE_CUMSUM gives the op). cu_seqlens bounds each sequence. */
Case build_case(const Shape& shape, const std::vector<double>& keys,
                const std::vector<double>& betas, const std::vector<double>& decays) {
    Case c;
    c.rows = rows_of(shape);

    AdValue k = operand({c.rows, shape.k_heads, shape.lanes}, "bf16");
    k.value = keys;
    AdValue beta = operand({c.rows, shape.heads}, "f32");
    beta.value = betas;
    AdValue g = operand({c.rows, shape.heads}, "f32");
    g.value = decays;
    AdValue cu = operand({(int64_t)shape.lens.size() + 1}, "i32");
    int64_t at = 0;
    cu.value[0] = 0.0;
    for (size_t s = 0; s < shape.lens.size(); ++s) {
        at += shape.lens[s];
        cu.value[s + 1] = (double)at;
    }

    c.rec.op = "gdn_kkt_solve";
    c.rec.params = {
        {"M", c.rows},
        {"head_k", shape.lanes},
        {"chunk", shape.chunk},
    };
    c.rec.operands.push_back({"k", "bf16", k.shape, 0, c.rows, false});
    c.in.push_back(std::move(k));
    c.rec.operands.push_back({"beta", "f32", beta.shape, 0, c.rows, false});
    c.in.push_back(std::move(beta));
    c.rec.operands.push_back({"g", "f32", g.shape, 0, c.rows, false});
    c.in.push_back(std::move(g));
    c.rec.operands.push_back({"cu_seqlens", "i32", cu.shape, 0, 0, false});
    c.in.push_back(std::move(cu));

    AdValue A = operand({c.rows, shape.heads, shape.chunk}, "bf16");
    c.rec.operands.push_back({"A", "bf16", A.shape, 0, c.rows, true});
    c.out.push_back(std::move(A));
    return c;
}

/* A case of the same geometry with freshly drawn keys, gates and decay. */
Case make_case(const Shape& shape, uint64_t seed) {
    uint64_t s = seed;
    std::vector<double> keys((size_t)(rows_of(shape) * shape.k_heads * shape.lanes));
    std::vector<double> betas((size_t)(rows_of(shape) * shape.heads));
    std::vector<double> decays((size_t)(rows_of(shape) * shape.heads));

    /* Unit key rows, on bf16's grid: the delta rule normalises its keys, so <k_i, k_j> is a cosine
     * and the block M stays well conditioned for a full chunk. */
    for (int64_t t = 0; t < rows_of(shape); ++t)
        for (int64_t hk = 0; hk < shape.k_heads; ++hk) {
            double* row = &keys[(size_t)((t * shape.k_heads + hk) * shape.lanes)];
            double norm = 0.0;
            for (int64_t d = 0; d < shape.lanes; ++d) {
                row[d] = urand(&s);
                norm += row[d] * row[d];
            }
            norm = std::sqrt(norm);
            for (int64_t d = 0; d < shape.lanes; ++d) row[d] = bf16(row[d] / norm);
        }

    /* beta is a sigmoid of the gate, and g is the log of the decay, so it only falls inside a
     * chunk: with both, A = (I + M)^{-1} is the near-identity a trained block's is. */
    for (int64_t t = 0; t < rows_of(shape); ++t)
        for (int64_t h = 0; h < shape.heads; ++h)
            betas[(size_t)(t * shape.heads + h)] = 0.05 + 0.9 * (0.5 + 0.5 * urand(&s));

    int64_t bos = 0;
    for (const int64_t len : shape.lens) {
        for (int64_t h = 0; h < shape.heads; ++h) {
            for (int64_t c0 = 0; c0 < len; c0 += shape.chunk) {
                const int64_t L = std::min(shape.chunk, len - c0);
                double acc = 0.0;
                for (int64_t i = 0; i < L; ++i) {
                    acc -= 0.2 + 0.6 * (0.5 + 0.5 * urand(&s));
                    decays[(size_t)((bos + c0 + i) * shape.heads + h)] = acc;
                }
            }
        }
        bos += len;
    }
    return build_case(shape, keys, betas, decays);
}

/* The tokens of sequence `s` of `big`, as a case of their own: the same rows of keys, beta and g,
 * and cu_seqlens {0, len}. Used to show that a batch answers each sequence the way that sequence
 * alone is answered. */
Case slice_case(const Case& big, const Shape& source, int64_t s) {
    int64_t bos = 0;
    for (int64_t i = 0; i < s; ++i) bos += source.lens[(size_t)i];
    const int64_t len = source.lens[(size_t)s];
    Shape one = source;
    one.lens = {len};

    std::vector<double> keys;
    const std::vector<double>& all_k = big.in[0].value;
    keys.assign(all_k.begin() + (size_t)(bos * source.k_heads * source.lanes),
                all_k.begin() + (size_t)((bos + len) * source.k_heads * source.lanes));
    std::vector<double> betas(big.in[1].value.begin() + (size_t)(bos * source.heads),
                              big.in[1].value.begin() + (size_t)((bos + len) * source.heads));
    std::vector<double> decays(big.in[2].value.begin() + (size_t)(bos * source.heads),
                               big.in[2].value.begin() + (size_t)((bos + len) * source.heads));
    return build_case(one, keys, betas, decays);
}

/* One cotangent per operand: zeros, then a random cotangent on the op's one write. */
std::vector<AdValue> zero_cotangents(const Case& c) {
    std::vector<AdValue> dc;
    for (const Operand& o : c.rec.operands) {
        AdValue v;
        v.dtype = o.dtype;
        v.shape = o.shape;
        v.value.assign((size_t)count(o.shape), 0.0);
        dc.push_back(std::move(v));
    }
    return dc;
}

/* The adjoint run directly on a case, with the cotangent the caller gives: the reads' slots of dc
 * come back filled, the writes' stay zero. */
std::vector<AdValue> run_adjoint(const Case& c, const std::vector<AdValue>& cot) {
    std::vector<AdValue> dc = zero_cotangents(c);
    const AdjointFn fn = adjoint_find(c.rec.op, Side::host, c.rec.params, nullptr);
    R97_CHECK(fn != nullptr);
    const AdjointCall call{c.rec, c.in, cot, &dc, nullptr, ""};
    fn(call);
    return dc;
}


/* Central finite differences of the twin against the adjoint's directional derivative, as
 * check_adjoint asks it: every read operand, a random cotangent on the write, a random direction,
 * frozen operands left alone. Two differences, at `eps` and at `eps/2`, are combined as
 * (4 D(eps/2) - D(eps)) / 3: the second-order term of a symmetric difference cancels in that
 * combination, which leaves the twin's roundoff as the only error of order eps and lets the check
 * speak to 1e-8. */
FdReport check_adjoint_step(const OpRecord& rec, const std::vector<AdValue>& in,
                            const std::vector<AdValue>& out_spec, double rel_tol, double eps) {
    FdReport rep;
    size_t n_reads = 0, n_writes = 0;
    operand_counts(rec, &n_reads, &n_writes);
    if (n_reads != in.size() || n_writes != out_spec.size()) {
        rep.note = "record reads " + std::to_string(n_reads) + " and writes " +
                   std::to_string(n_writes) + "; the check was given " +
                   std::to_string(in.size()) + " and " + std::to_string(out_spec.size());
        return rep;
    }
    std::string err;
    const TwinFn twin = twin_find(rec.op, rec.params, &err);
    if (!twin) {
        rep.note = err;
        return rep;
    }
    const AdjointFn adj = adjoint_find(rec.op, Side::host, rec.params, &err);
    if (!adj) {
        rep.note = err;
        return rep;
    }
    const std::vector<int> frozen = adjoint_frozen(rec.op, Side::host, rec.params);

    std::vector<AdValue> out_vals;
    twin(rec, in, &out_vals);
    if (out_vals.size() != out_spec.size()) {
        rep.note = "twin produced " + std::to_string(out_vals.size()) + " outputs, expected " +
                   std::to_string(out_spec.size());
        return rep;
    }
    std::vector<AdValue> fwd(rec.operands.size());
    {
        size_t r = 0, w = 0;
        for (size_t i = 0; i < rec.operands.size(); ++i)
            fwd[i] = rec.operands[i].written ? out_vals[w++] : in[r++];
    }

    uint64_t seed = 0xfeedface1234ull;
    std::vector<AdValue> cot;
    for (const AdValue& spec : out_spec) {
        AdValue c = spec;
        c.value.assign((size_t)count(c.shape), 0.0);
        for (double& v : c.value) v = urand(&seed);
        cot.push_back(std::move(c));
    }

    std::vector<AdValue> dc(rec.operands.size());
    {
        size_t r = 0;
        for (size_t i = 0; i < rec.operands.size(); ++i) {
            if (rec.operands[i].written) continue;
            dc[i] = in[r++];
            dc[i].value.assign(dc[i].value.size(), 0.0);
        }
    }
    const AdjointCall call{rec, fwd, cot, &dc, nullptr, ""};
    adj(call);

    double analytic = 0.0;
    std::vector<std::vector<double>> dir(rec.operands.size());
    for (size_t i = 0; i < rec.operands.size(); ++i) {
        if (rec.operands[i].written) continue;
        if (std::find(frozen.begin(), frozen.end(), (int)i) != frozen.end()) continue;
        dir[i].resize(fwd[i].value.size());
        for (double& v : dir[i]) v = urand(&seed);
        for (size_t e = 0; e < dir[i].size(); ++e) analytic += dc[i].value[e] * dir[i][e];
    }

    auto slope = [&](double h) {
        double d = 0.0;
        for (const double sign : {1.0, -1.0}) {
            std::vector<AdValue> x = in;
            size_t at = 0;
            for (size_t i = 0; i < rec.operands.size(); ++i) {
                if (rec.operands[i].written) continue;
                for (size_t e = 0; e < dir[i].size(); ++e) x[at].value[e] += sign * h * dir[i][e];
                ++at;
            }
            std::vector<AdValue> y;
            twin(rec, x, &y);
            double loss = 0.0;
            for (size_t o = 0; o < y.size(); ++o)
                for (size_t e = 0; e < y[o].value.size(); ++e)
                    loss += y[o].value[e] * cot[o].value[e];
            d += sign * loss;
        }
        return d / (2.0 * h);
    };
    const double coarse = slope(eps);
    const double fine = slope(eps / 2.0);
    rep.analytic = analytic;
    rep.numeric = (4.0 * fine - coarse) / 3.0;
    rep.rel = rel_err(rep.analytic, rep.numeric);
    rep.ok = close_enough(rep.analytic, rep.numeric, rel_tol);
    if (!rep.ok)
        rep.note = "adjoint says " + std::to_string(rep.analytic) + ", finite differences say " +
                   std::to_string(rep.numeric);
    return rep;
}

void adjoint_gdn_kkt_test() {
    const std::vector<Shape> shapes = {
        /* the fixture's delta-net geometry, one chunk's worth of sequence or less */
        {"fixture, below one chunk", 2, 4, 16, 64, {5}},
        {"fixture, exactly one chunk", 2, 4, 16, 64, {64}},
        /* above the chunk size, with a short last chunk of 37 tokens */
        {"fixture, two chunks and a short last", 2, 4, 16, 64, {101}},
        /* a batch of sequences of different lengths: short, exactly one chunk, and two chunks with
         * a short last one, so each sequence's chunk grid starts again at its own cu row */
        {"fixture, three sequences", 2, 4, 16, 64, {5, 64, 150}},
        /* no grouping -- one key head per value head -- with wider keys, still two chunks */
        {"no grouping, two chunks", 2, 2, 32, 64, {90}},
    };

    for (const Shape& s : shapes) {
        const Case c = make_case(s, 0x5eed);

        const KernelReport k = check_kernel(c.rec, c.in, c.out, bf16_tol);
        std::printf("gdn_kkt_solve %s: kernel max err %.6g\n", s.name, k.max_err);
        require(k);

        const FdReport fd = check_adjoint_step(c.rec, c.in, c.out, fd_tol, fd_eps);
        std::printf("gdn_kkt_solve %s: adjoint analytic %.10g numeric %.10g rel err %.3g\n",
                    s.name, fd.analytic, fd.numeric, fd.rel);
        require(fd);
    }
}

/* A chunk is solved on its own operands, so no cotangent crosses a chunk or a sequence boundary:
 * the batch of three sequences answers token for token what each sequence on its own answers, and
 * cu_seqlens -- which addresses those rows -- is frozen, so it gets no cotangent at all. */
void adjoint_gdn_kkt_batch_test() {
    const Shape shape{"fixture, three sequences", 2, 4, 16, 64, {5, 64, 150}};
    const Case big = make_case(shape, 0x1234);
    R97_CHECK(big.in.size() == 4);

    /* The adjoint row this op registers: a twin, a host adjoint, and cu_seqlens frozen. */
    R97_CHECK(twin_find("gdn_kkt_solve", big.rec.params, nullptr) != nullptr);
    const OpRows rows = op_rows("gdn_kkt_solve");
    R97_CHECK(rows.twin);
    R97_CHECK(rows.host_adjoint);
    R97_CHECK(rows.host_frozen == std::vector<int>({3}));

    uint64_t seed = 0xbeef;
    AdValue cot = operand(big.out[0].shape, "bf16");
    fill_random(&cot, &seed);
    const std::vector<AdValue> dc = run_adjoint(big, {cot});

    /* cu_seqlens addresses rows: a frozen operand, and this adjoint never writes its slot. */
    R97_CHECK(dc.size() == 5);
    for (const double v : dc[3].value) R97_CHECK(v == 0.0);

    /* Each sequence alone, with the same values and its own slice of the cotangent. */
    int64_t bos = 0;
    for (size_t s = 0; s < shape.lens.size(); ++s) {
        const int64_t len = shape.lens[s];
        const Case mine = slice_case(big, shape, (int64_t)s);
        AdValue my_cot = operand(mine.out[0].shape, "bf16");
        my_cot.value.assign(cot.value.begin() + (size_t)(bos * shape.heads * shape.chunk),
                            cot.value.begin() +
                                (size_t)((bos + len) * shape.heads * shape.chunk));
        const std::vector<AdValue> one_dc = run_adjoint(mine, {my_cot});

        /* The batch's cotangent of these tokens, element for element, is the answer the sequence
         * on its own gave: the same sums in the same order, so equal up to the compiler's
         * reassociation. */
        for (size_t slot : {size_t{0}, size_t{1}, size_t{2}}) {
            const int64_t width = slot == 0 ? shape.k_heads * shape.lanes : shape.heads;
            const size_t base = (size_t)(bos * width);
            const size_t n = (size_t)(len * width);
            for (size_t e = 0; e < n; ++e) {
                const double mine_v = one_dc[slot].value[e];
                const double batch_v = dc[slot].value[base + e];
                R97_CHECK(std::fabs(batch_v - mine_v) <=
                          1e-12 * (1.0 + std::fabs(mine_v)));
            }
        }
        std::printf("gdn_kkt_solve batch: sequence %zu of %zu tokens answers on its own\n", s,
                    shape.lens.size());
        bos += len;
    }
}

/* A record the op cannot compute -- here the record's head_k is not the key's last dimension --
 * gets no output from the twin, and both checks say so rather than comparing against nothing. */
void adjoint_gdn_kkt_reject_test() {
    const Shape shape{"fixture, exactly one chunk", 2, 4, 16, 64, {64}};
    const Case c = make_case(shape, 0xf00d);
    OpRecord bad = c.rec;
    for (auto& p : bad.params)
        if (p.first == "head_k") p.second = int64_t{32};  // the keys carry 16 lanes

    const TwinFn twin = twin_find("gdn_kkt_solve", bad.params, nullptr);
    R97_CHECK(twin != nullptr);
    std::vector<AdValue> out;
    twin(bad, c.in, &out);
    R97_CHECK(out.empty());

    const KernelReport k = check_kernel(bad, c.in, c.out, bf16_tol);
    R97_CHECK(!k.ok);
    R97_CHECK(!k.note.empty());
    const FdReport fd = check_adjoint(bad, c.in, c.out, fd_tol);
    R97_CHECK(!fd.ok);
    R97_CHECK(!fd.note.empty());
    std::printf("gdn_kkt_solve rejects a head_k that is not the key's last dimension: %s\n",
                fd.note.c_str());
}

}  // namespace

R97_TEST(adjoint_gdn_kkt) { adjoint_gdn_kkt_test(); }
R97_TEST(adjoint_gdn_kkt_batch) { adjoint_gdn_kkt_batch_test(); }
R97_TEST(adjoint_gdn_kkt_reject) { adjoint_gdn_kkt_reject_test(); }
