// tests/host/adjoint_linear.cpp -- the frozen-linears family through the adjoint checker.
//
// The tiny fixture's dimensions (hidden 128, MLP 256, vocabulary 320) are one of the three shapes
// every op runs. The int8 linear's adjoint is checked in the dequantised activation basis: its
// activation scale is a constant of the quantiser, not a factor of this GEMM's input cotangent.
// Label: host.
#include "support/adjoint_check.h"
#include "support/test.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

using namespace r97;
using namespace r97::adjoint_check;

namespace {

const double bf16_tol = 1e-2;
const double fd_tol = 1e-6;

uint64_t mix(uint64_t* s) {
    uint64_t x = (*s += 0x9e3779b97f4a7c15ull);
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

double unit(uint64_t* s) { return static_cast<double>(mix(s) >> 11) * (1.0 / 9007199254740992.0); }

struct Case {
    std::vector<AdValue> in;
    std::vector<AdValue> out;
};

AdValue value_of(const Operand& o) {
    AdValue v;
    v.dtype = o.dtype;
    v.shape = o.shape;
    int64_t n = 1;
    for (int64_t d : o.shape) n *= d;
    v.value.assign(static_cast<std::size_t>(n), 0.0);
    return v;
}

bool has(const std::string& s, const char* needle) { return s.find(needle) != std::string::npos; }

// Values chosen so the host kernel sees exactly what the twin sees. Integer codes survive an i8
// store; scales are positive, as the grid requires; indices select rows of the operand they index.
Case make_case(const OpRecord& rec, uint64_t seed) {
    Case c;
    int64_t index_rows = 1;
    for (const Operand& o : rec.operands) {
        if (!o.written && (o.buffer == "idx" || o.buffer == "tokens")) {
            for (const Operand& src : rec.operands) {
                if (!src.written && src.buffer != o.buffer && !src.shape.empty()) {
                    index_rows = src.shape[0];
                    break;
                }
            }
        }
    }
    for (const Operand& o : rec.operands) {
        AdValue v = value_of(o);
        if (o.written) {
            c.out.push_back(std::move(v));
            continue;
        }
        for (double& x : v.value) {
            const uint64_t r = mix(&seed);
            if (o.buffer == "idx" || o.buffer == "tokens") {
                x = static_cast<double>(r % static_cast<uint64_t>(index_rows > 0 ? index_rows : 1));
                if ((r >> 8) % 11 == 0) x = -1.0;  // a row the gather leaves zero
            } else if (o.dtype == "i8" || o.dtype == "u8") {
                x = static_cast<double>(static_cast<int>(r % 9) - 4);
            } else if (has(o.buffer, "scale")) {
                /* A power-of-two fraction, so a bf16 scale store is exact. */
                x = (static_cast<double>((r >> 16) % 16) + 1.0) / 256.0;
            } else if (o.dtype == "bf16") {
                /* Multiples of 2^-7 are bf16-exact here, so the kernel and twin
                 * start from the same inputs even after a cancellation-heavy dot. */
                x = (static_cast<double>(static_cast<int>(r % 257) - 128)) / 128.0;
            } else {
                x = 2.0 * unit(&seed) - 1.0;
            }
        }
        c.in.push_back(std::move(v));
    }
    return c;
}

OpRecord gemm_nt_rec(int64_t M, int64_t N, int64_t K) {
    OpRecord rec;
    rec.op = "gemm_nt";
    rec.params = {{"M", M}, {"N", N}, {"K", K}, {"dtype", "bf16"}};
    rec.operands = {
        {"a", "bf16", {M, K}, 0, M, false},
        {"b", "bf16", {N, K}, 0, N, false},
        {"y", "f32", {M, N}, 0, M, true},
    };
    return rec;
}

OpRecord gemm_nt_q_rec(int64_t M, int64_t N, int64_t K) {
    OpRecord rec;
    rec.op = "gemm_nt_q";
    rec.params = {{"M", M}, {"N", N}, {"K", K}, {"group", 128}, {"dtype", "i8a8h"}};
    rec.operands = {
        {"a", "i8", {M, K}, 0, M, false},
        {"a_scale", "f32", {M, K / 128}, 0, M, false},
        {"b", "i8", {N, K}, 0, N, false},
        {"b_scale", "bf16", {N, K / 128}, 0, N, false},
        {"y", "f32", {M, N}, 0, M, true},
    };
    return rec;
}

OpRecord logits_gemm_rec(int64_t M, int64_t n_vocab, int64_t n_embd) {
    OpRecord rec;
    rec.op = "logits_gemm";
    rec.params = {{"M", M}, {"n_vocab", n_vocab}, {"n_embd", n_embd}, {"dtype", "bf16"}};
    rec.operands = {
        {"x", "bf16", {M, n_embd}, 0, M, false},
        {"lm_head", "bf16", {n_vocab, n_embd}, 0, n_vocab, false},
        {"logits", "f32", {M, n_vocab}, 0, M, true},
    };
    return rec;
}

OpRecord gather_rows_rec(int64_t M, int64_t rows, int64_t n) {
    OpRecord rec;
    rec.op = "gather_rows";
    rec.params = {{"M", M}, {"n", n}, {"dtype", "bf16"}};
    rec.operands = {
        {"x", "bf16", {rows, n}, 0, rows, false},
        {"idx", "i32", {M}, 0, M, false},
        {"y", "bf16", {M, n}, 0, M, true},
    };
    return rec;
}

OpRecord embed_lookup_rec(int64_t M, int64_t n_vocab, int64_t n_embd) {
    OpRecord rec;
    rec.op = "embed_lookup";
    rec.params = {{"M", M}, {"n_embd", n_embd}, {"n_vocab", n_vocab}, {"dtype", "bf16"}};
    rec.operands = {
        {"tokens", "i32", {M}, 0, M, false},
        {"wte", "bf16", {n_vocab, n_embd}, 0, n_vocab, false},
        {"x", "bf16", {M, n_embd}, 0, M, true},
    };
    return rec;
}

void check(const OpRecord& rec, uint64_t seed, int unit_scale_read = -1) {
    Case c = make_case(rec, seed);
    const KernelReport k = check_kernel(rec, c.in, c.out, bf16_tol);
    std::printf("%s kernel max err %.3g\n", rec.op.c_str(), k.max_err);
    require(k);
    // The int8 GEMM's adjoint is in the dequantised activation basis, so the finite-difference
    // check holds the activation scale at one. The kernel check above keeps the real scale.
    if (unit_scale_read >= 0)
        for (double& x : c.in[static_cast<std::size_t>(unit_scale_read)].value) x = 1.0;
    const FdReport f = check_adjoint(rec, c.in, c.out, fd_tol, seed ^ 0x51ed);
    std::printf("%s adjoint rel err %.3g\n", rec.op.c_str(), f.rel);
    require(f);
}

}  // namespace

R97_TEST(adjoint_linear_gemm_nt) {
    check(gemm_nt_rec(4, 256, 128), 0x1001);   // the tiny fixture's MLP width
    check(gemm_nt_rec(1, 128, 256), 0x1002);
    check(gemm_nt_rec(7, 320, 384), 0x1003);
}

R97_TEST(adjoint_linear_gemm_nt_q) {
    /* Reads are a, a_scale, b, b_scale: the scale at read index 1. */
    check(gemm_nt_q_rec(4, 256, 128), 0x2001, 1);  // the tiny fixture's int8 linear
    check(gemm_nt_q_rec(1, 128, 256), 0x2002, 1);
    check(gemm_nt_q_rec(7, 320, 384), 0x2003, 1);
}

R97_TEST(adjoint_linear_logits_gemm) {
    check(logits_gemm_rec(4, 320, 128), 0x3001);  // the tiny fixture's vocabulary
    check(logits_gemm_rec(1, 128, 256), 0x3002);
    check(logits_gemm_rec(9, 320, 384), 0x3003);
}

R97_TEST(adjoint_linear_gather_rows) {
    check(gather_rows_rec(4, 17, 128), 0x4001);  // the tiny fixture's width
    check(gather_rows_rec(17, 256, 37), 0x4002);
    check(gather_rows_rec(37, 320, 384), 0x4003);
}

/* The frozen column of docs/adjoints/linear.md, operand for operand: the walk asks for exactly
 * the cotangents the page says it does not, and the positions it registers are schema positions.
 * A row that froze the wrong operand would still pass a finite-difference check whenever the
 * check's own cotangent happens to be zero, so pin the positions here. */
R97_TEST(adjoint_linear_frozen_positions_match_the_doc) {
    const OpRecord gemm_nt = gemm_nt_rec(2, 3, 4);
    const OpRecord gemm_nt_q = gemm_nt_q_rec(2, 3, 128);
    const OpRecord logits_gemm = logits_gemm_rec(2, 16, 8);
    const OpRecord gather_rows = gather_rows_rec(2, 5, 4);
    const OpRecord embed_lookup = embed_lookup_rec(2, 16, 8);
    const std::pair<const OpRecord*, std::vector<int>> rows[] = {
        {&gemm_nt, {1}},              /* b */
        {&gemm_nt_q, {1, 2, 3}},      /* a_scale, b, b_scale */
        {&logits_gemm, {1}},          /* lm_head */
        {&gather_rows, {1}},          /* idx */
        {&embed_lookup, {0, 1}},      /* tokens, wte */
    };
    for (const auto& row : rows) {
        const OpRecord& rec = *row.first;
        const std::vector<int> frozen = adjoint_frozen(rec.op, Side::host, rec.params);
        std::printf("%s: %zu frozen operands\n", rec.op.c_str(), frozen.size());
        R97_CHECK(frozen == row.second);
    }
}

R97_TEST(adjoint_linear_embed_lookup) {
    check(embed_lookup_rec(4, 320, 128), 0x5001);  // the tiny fixture's vocabulary
    check(embed_lookup_rec(1, 128, 256), 0x5002);
    check(embed_lookup_rec(9, 320, 384), 0x5003);
}

R97_TEST(adjoint_linear_embed_lookup_is_a_stop) {
    const OpRecord rec = embed_lookup_rec(4, 320, 128);
    const Case c = make_case(rec, 0x5001);
    const KernelReport k = check_kernel(rec, c.in, c.out, bf16_tol);
    std::printf("embed_lookup kernel max err %.3g\n", k.max_err);
    require(k);

    std::string err;
    TwinFn twin = twin_find(rec.op, rec.params, &err);
    R97_CHECK(twin != nullptr);
    AdjointFn adj = adjoint_find(rec.op, Side::host, rec.params, &err);
    R97_CHECK(adj != nullptr);
    const std::vector<int> frozen = adjoint_frozen(rec.op, Side::host, rec.params);
    R97_CHECK(frozen == std::vector<int>({0, 1}));

    static std::vector<AdValue> fwd;
    fwd.clear();
    fwd.resize(rec.operands.size());
    std::size_t r = 0, w = 0;
    for (const Operand& o : rec.operands) {
        fwd[&o - rec.operands.data()] = o.written ? c.out[w++] : c.in[r++];
    }
    std::vector<AdValue> out;
    twin(rec, c.in, &out);
    R97_CHECK(out.size() == 1);
    for (std::size_t i = 0, wi = 0; i < rec.operands.size(); ++i) {
        if (rec.operands[i].written) fwd[i] = out[wi++];
    }
    std::vector<AdValue> cot = out;
    for (AdValue& a : cot) for (double& x : a.value) x = 0.25;
    std::vector<AdValue> dc = fwd;
    for (AdValue& a : dc) a.value.assign(a.value.size(), 0.0);
    AdjointCall call{rec, fwd, cot, &dc, nullptr, "test"};
    adj(call);
    for (const AdValue& a : dc)
        for (double x : a.value) R97_CHECK(x == 0.0);
}

// gemm_nt_q's adjoint is a cotangent for the dequantised activation. Perturbing the stored code by
// eps / a_scale changes the dequantised activation by exactly eps, which is the direction the
// adjoint describes.
R97_TEST(adjoint_linear_gemm_nt_q_dequant_basis) {
    const OpRecord rec = gemm_nt_q_rec(3, 5, 128);
    const Case c = make_case(rec, 0x6001);
    std::string err;
    TwinFn twin = twin_find(rec.op, rec.params, &err);
    R97_CHECK(twin != nullptr);
    AdjointFn adj = adjoint_find(rec.op, Side::host, rec.params, &err);
    R97_CHECK(adj != nullptr);

    std::vector<AdValue> fwd(rec.operands.size());
    std::size_t r = 0, w = 0;
    for (std::size_t i = 0; i < rec.operands.size(); ++i)
        fwd[i] = rec.operands[i].written ? c.out[w++] : c.in[r++];
    std::vector<AdValue> out;
    twin(rec, c.in, &out);
    R97_CHECK(out.size() == 1);
    for (std::size_t i = 0, wi = 0; i < rec.operands.size(); ++i)
        if (rec.operands[i].written) fwd[i] = out[wi++];

    std::vector<AdValue> cot = out;
    uint64_t seed = 0x6002;
    for (AdValue& a : cot) for (double& x : a.value) x = 2.0 * unit(&seed) - 1.0;
    std::vector<AdValue> dc = fwd;
    for (AdValue& a : dc) a.value.assign(a.value.size(), 0.0);
    AdjointCall call{rec, fwd, cot, &dc, nullptr, "test"};
    adj(call);

    const std::vector<int64_t>& shape = fwd[0].shape;
    const int64_t K = shape[1];
    std::vector<double> dir(fwd[0].value.size());
    for (double& x : dir) x = 2.0 * unit(&seed) - 1.0;

    auto loss = [&](double eps, double sign) {
        std::vector<AdValue> in = c.in;
        for (std::size_t e = 0; e < in[0].value.size(); ++e) {
            const double scale = in[1].value[static_cast<std::size_t>((e / K) * (K / 128) + (e % K) / 128)];
            in[0].value[e] += sign * eps * dir[e] / scale;
        }
        std::vector<AdValue> y;
        twin(rec, in, &y);
        double sum = 0.0;
        for (std::size_t e = 0; e < y[0].value.size(); ++e) sum += y[0].value[e] * cot[0].value[e];
        return sum;
    };
    const double eps = 1e-5;
    const double numeric = (loss(eps, 1.0) - loss(eps, -1.0)) / (2.0 * eps);
    double analytic = 0.0;
    for (std::size_t e = 0; e < dir.size(); ++e) analytic += dc[0].value[e] * dir[e];
    const double rel = std::fabs(numeric - analytic) / std::fmax(1.0, std::fabs(analytic));
    std::printf("gemm_nt_q dequant basis rel err %.3g\n", rel);
    R97_CHECK(rel < fd_tol);
}
