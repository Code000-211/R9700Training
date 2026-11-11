// tests/host/adjoint_attention.cpp -- `attn_paged` (src/adjoint/host/attention.cpp) through both
// checks of tests/support/adjoint_check.h: its f64 twin against radiance's reference kernel, and
// its host adjoint against finite differences of the twin. Then the addressing the issue asks
// for: the pool's cotangent sits on the rows the step wrote -- one per sequence and key position,
// found through the block table -- and nowhere else.
//
// The shapes are the fixture's head geometry (4 query heads over 2 key/value heads of dimension
// 32, tests/support/fixture.h) and the grouped-query geometry a Qwen3.8-27B block has (24 over 4
// of dimension 256), a single sequence and a batch of three sequences of different lengths.
#include "support/adjoint_check.h"
#include "support/test.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace r97;
using namespace r97::adjoint_check;

namespace {

/* bf16 carries 8 significand bits, so a kernel that rounds as it goes lands within about 2^-9
 * relative of the f64 twin; 0.02 leaves the softmax's weighting and the bf16 store room. */
const double bf16_tol = 0.02;

/* The twin is exact in f64, so central differences reproduce its directional derivative far
 * inside 1e-6 relative. */
const double fd_tol = 1e-6;

int64_t count(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (const int64_t d : shape) n *= d;
    return n;
}

/* One shape the family has to hold for: the head geometry, the pool's block size, the sliding
 * window (0 = the whole visible prefix), each sequence's length, and whether the record carries
 * cu_seqlens. */
struct Shape {
    const char* name;
    int64_t q_heads;
    int64_t kv_heads;
    int64_t head_dim;
    int64_t block_size;
    int64_t window;
    std::vector<int64_t> lens;
    bool ragged;
};

/* What the checkers are given: the record, the read operands' values in schema order, and the
 * written operands' dtypes and shapes. */
struct Case {
    OpRecord rec;
    std::vector<AdValue> in;
    std::vector<AdValue> out;
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

/* A call of the op as a training step records one: each sequence's keys and values are the pool
 * rows of that step, one row per position, and the block table addresses them. Each sequence gets
 * as many blocks as its length needs plus a table entry the step did not fill, which stays -1 and
 * whose pool rows hold values no query may read. A record that leaves the optional operands out
 * (equal-length sequences, no fp8 pool) carries five operands; the ones it keeps sit at their
 * schema positions. */
Case make_case(const Shape& shape, uint64_t seed) {
    const int64_t n_seq = (int64_t)shape.lens.size();
    const int64_t hd = shape.head_dim;
    const int64_t kh = shape.kv_heads;
    int64_t rows = 0;
    for (const int64_t n : shape.lens) rows += n;

    int64_t blocks = 0;
    int64_t widest = 0;
    for (const int64_t n : shape.lens) {
        const int64_t b = (n + shape.block_size - 1) / shape.block_size;
        blocks += b;
        if (b > widest) widest = b;
    }
    const int64_t width = widest + 1;

    AdValue q = operand({rows, shape.q_heads, hd}, "bf16");
    fill_random(&q, &seed);
    AdValue kv = operand({blocks, kh, shape.block_size, 2 * hd}, "bf16");
    fill_random(&kv, &seed);

    AdValue table = operand({n_seq, width}, "i32");
    int64_t first_block = 0;
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t mine = (shape.lens[(size_t)s] + shape.block_size - 1) / shape.block_size;
        for (int64_t b = 0; b < width; ++b)
            table.value[(size_t)(s * width + b)] = b < mine ? first_block + b : -1;
        first_block += mine;
    }

    AdValue seqused = operand({n_seq}, "i32");
    for (int64_t s = 0; s < n_seq; ++s) seqused.value[(size_t)s] = (double)shape.lens[(size_t)s];

    AdValue cu = operand({n_seq + 1}, "i32");
    for (int64_t s = 0; s < n_seq; ++s)
        cu.value[(size_t)s + 1] = cu.value[(size_t)s] + (double)shape.lens[(size_t)s];

    AdValue kdesc = operand({n_seq * kh}, "f32");
    AdValue vdesc = operand({n_seq * kh}, "f32");
    for (size_t i = 0; i < kdesc.value.size(); ++i) {
        kdesc.value[i] = 1.0 + 0.4 * urand(&seed);
        vdesc.value[i] = 1.0 + 0.4 * urand(&seed);
    }

    AdValue y = operand({rows, shape.q_heads, hd}, "bf16");

    char scale[64];
    std::snprintf(scale, sizeof scale, "%.17g", 1.0 / std::sqrt((double)hd));
    Case c;
    c.rec.op = "attn_paged";
    c.rec.params = {
        {"head_dim", hd},     {"gqa", shape.q_heads / kh}, {"block_size", shape.block_size},
        {"causal", int64_t{1}}, {"window", shape.window},  {"scale", std::string(scale)},
    };
    c.rec.position = 3;

    c.rec.operands.push_back({"q", "bf16", q.shape, 0, rows, false});
    c.in.push_back(std::move(q));
    c.rec.operands.push_back({"kv_cache", "bf16", kv.shape, 0, 0, false});
    c.in.push_back(std::move(kv));
    c.rec.operands.push_back({"block_table", "i32", table.shape, 0, 0, false});
    c.in.push_back(std::move(table));
    c.rec.operands.push_back({"seqused", "i32", seqused.shape, 0, 0, false});
    c.in.push_back(std::move(seqused));
    if (shape.ragged) {
        c.rec.operands.push_back({"k_descale", "f32", kdesc.shape, 0, 0, false});
        c.in.push_back(std::move(kdesc));
        c.rec.operands.push_back({"v_descale", "f32", vdesc.shape, 0, 0, false});
        c.in.push_back(std::move(vdesc));
        c.rec.operands.push_back({"cu_seqlens", "i32", cu.shape, 0, 0, false});
        c.in.push_back(std::move(cu));
    }
    c.rec.operands.push_back({"out", "bf16", y.shape, 0, rows, true});
    c.out.push_back(std::move(y));
    return c;
}

void adjoint_attention_test() {
    const std::vector<Shape> shapes = {
        /* the fixture's head geometry: 4 query heads over 2 of dimension 32 */
        {"fixture, one sequence", 4, 2, 32, 8, 0, {12}, true},
        {"fixture, three sequences", 4, 2, 32, 4, 0, {5, 9, 14}, true},
        /* the grouped-query geometry a Qwen3.8-27B block has */
        {"grouped query, one sequence", 24, 4, 256, 16, 0, {7}, true},
        {"grouped query, three sequences", 24, 4, 256, 5, 0, {3, 6, 10}, true},
        /* a sliding window, and the equal-length case whose record carries no cu_seqlens */
        {"fixture, sliding window", 4, 2, 32, 3, 4, {6, 6}, true},
        {"fixture, uniform lengths", 4, 2, 32, 4, 0, {6, 6}, false},
    };

    for (const Shape& s : shapes) {
        const Case c = make_case(s, 0x5eed);

        /* (1) the twin against radiance's host kernel -- where the record carries every operand
         * the op's schema names, since libref reads them positionally and a record that leaves one
         * out gives it nothing to read -- (2) the adjoint against finite differences of the twin. */
        if (c.in.size() == 7) {
            const KernelReport k = check_kernel(c.rec, c.in, c.out, bf16_tol);
            std::printf("attn_paged %s: kernel max err %.6g\n", s.name, k.max_err);
            require(k);
        } else {
            std::printf("attn_paged %s: twin and adjoint only, the record leaves the optional "
                        "operands out\n",
                        s.name);
        }
        const FdReport fd = check_adjoint(c.rec, c.in, c.out, fd_tol);
        std::printf("attn_paged %s: adjoint analytic %.6g numeric %.6g rel err %.3g\n", s.name,
                    fd.analytic, fd.numeric, fd.rel);
        require(fd);
    }
}

/* The pool's cotangent is the issue's own statement of what the adjoint returns: one cotangent per
 * row the step wrote, addressed by sequence and key position, and none on a row no query read. Run
 * the adjoint directly and read it back through the block table. */
void adjoint_attention_pool_address_test() {
    const Shape shape{"fixture, three sequences", 4, 2, 32, 4, 0, {5, 9, 14}, true};
    const Case c = make_case(shape, 0x1234);
    R97_CHECK(c.in.size() == 7);

    /* The adjoint row this op registers: a twin, a host adjoint, and the integer operands frozen. */
    R97_CHECK(twin_find("attn_paged", c.rec.params, nullptr) != nullptr);
    const OpRows rows = op_rows("attn_paged");
    R97_CHECK(rows.twin);
    R97_CHECK(rows.host_adjoint);
    R97_CHECK(rows.host_frozen == std::vector<int>({2, 3, 4, 5, 6}));

    const AdjointFn fn = adjoint_find("attn_paged", Side::host, c.rec.params, nullptr);
    R97_CHECK(fn != nullptr);

    /* One cotangent per operand: zeros, then a random cotangent on the output. */
    std::vector<AdValue> dc;
    for (const Operand& o : c.rec.operands) {
        AdValue v;
        v.dtype = o.dtype;
        v.shape = o.shape;
        v.value.assign((size_t)count(o.shape), 0.0);
        dc.push_back(std::move(v));
    }
    std::vector<AdValue> cot;
    uint64_t seed = 0xbeef;
    AdValue ycot = operand(c.out[0].shape, "bf16");
    fill_random(&ycot, &seed);
    cot.push_back(std::move(ycot));

    const AdjointCall call{c.rec, c.in, cot, &dc, nullptr, ""};
    fn(call);

    const int64_t hd = shape.head_dim;
    const int64_t kh = shape.kv_heads;
    const int64_t width = (int64_t)c.in[2].shape[1];
    const AdValue& dkv = dc[1];
    const int64_t n_seq = (int64_t)shape.lens.size();

    auto slot_of = [&](int64_t s, int64_t kh_, int64_t j) -> int64_t {
        const int64_t blk = (int64_t)c.in[2].value[(size_t)(s * width + j / shape.block_size)];
        R97_CHECK(blk >= 0);
        return ((blk * kh + kh_) * shape.block_size + j % shape.block_size) * 2 * hd;
    };

    /* Every (sequence, key position) the step wrote moved, in the key's share of the row and in
     * the value's. */
    double least_k = 1e300;
    double least_v = 1e300;
    for (int64_t s = 0; s < n_seq; ++s) {
        for (int64_t j = 0; j < shape.lens[(size_t)s]; ++j) {
            for (int64_t h = 0; h < kh; ++h) {
                const int64_t slot = slot_of(s, h, j);
                double k_moved = 0.0;
                double v_moved = 0.0;
                for (int64_t d = 0; d < hd; ++d) {
                    k_moved += std::fabs(dkv.value[(size_t)(slot + d)]);
                    v_moved += std::fabs(dkv.value[(size_t)(slot + hd + d)]);
                }
                least_k = std::min(least_k, k_moved);
                least_v = std::min(least_v, v_moved);
            }
        }
    }
    /* A block's rows past the sequence's last position were never written and never read, so no
     * query can have sent a cotangent there. */
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t mine = (shape.lens[(size_t)s] + shape.block_size - 1) / shape.block_size;
        for (int64_t j = shape.lens[(size_t)s]; j < mine * shape.block_size; ++j) {
            for (int64_t h = 0; h < kh; ++h) {
                const int64_t slot = slot_of(s, h, j);
                for (int64_t d = 0; d < 2 * hd; ++d)
                    R97_CHECK(dkv.value[(size_t)(slot + d)] == 0.0);
            }
        }
    }
    std::printf("attn_paged pool cotangent: smallest key row %.6g, smallest value row %.6g, "
                "unwritten rows zero\n",
                least_k, least_v);
    R97_CHECK(least_k > 0.0);
    R97_CHECK(least_v > 0.0);
}

/* A record the op cannot compute -- here `q`'s head count is not `gqa` times the pool's -- gets no
 * output from the twin, and both checks say so rather than comparing against nothing. */
void adjoint_attention_reject_test() {
    const Shape shape{"fixture, three sequences", 4, 2, 32, 4, 0, {5, 9, 14}, true};
    const Case c = make_case(shape, 0xf00d);
    OpRecord bad = c.rec;
    for (auto& p : bad.params)
        if (p.first == "gqa") p.second = int64_t{3};  // 4 query heads over 2 key/value heads is 2

    const TwinFn twin = twin_find("attn_paged", bad.params, nullptr);
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
    std::printf("attn_paged rejects a head count that is not gqa times the pool's: %s\n",
                fd.note.c_str());
}

}  // namespace

R97_TEST(adjoint_attention) { adjoint_attention_test(); }
R97_TEST(adjoint_attention_pool_address) { adjoint_attention_pool_address_test(); }
R97_TEST(adjoint_attention_reject) { adjoint_attention_reject_test(); }
