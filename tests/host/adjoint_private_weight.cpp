// tests/host/adjoint_private_weight.cpp -- the w8a8_fwht128 weight's stored layout, decoded by
// hand, against the f64 row.
//
// The real container hands `gemm_nt_q` its weight and its scale as plugin-private operands: no f64
// values, one pointer each into the engine's two planes. A device row reads them in the engine's
// layout -- frag64 codes and a t16 scale, src/adjoint/device/linear.hip's frag_go/t16_go -- so the
// thing that can be wrong is the decode, and a decode that only agrees with itself proves nothing.
// This case therefore writes the planes the way the pin writes them (libr4d/r4d_fp8_frag.h's
// frag_src walk and r4d_rad_relayout_i8_block's t16 loop, both quoted below), reads them back with
// the inverse the kernel uses, and then asks the question the walk asks: does the f64 row's dx on
// the canonical weight equal a plain f64 product of the weight this decode produced?
//
// The two directions are written from different structures -- the encoder walks the pin's
// (tile, lane, fragment, byte) order, the decoder computes an address from (row, k) -- so a
// misread of the permutation moves a code or a scale and the two dx part.
//
// The tolerance is the f64 one (1e-9 relative L2): both sides are f64 arithmetic on the same
// integer codes and power-of-two scales, so only the summation order differs.
// Label: host.
#include "r97/adjoint.h"
#include "r97/backward.h"
#include "r97/op.h"
#include "support/test.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using namespace r97;

namespace {

// The pin's fragment, as libr4d/r4d_fp8_frag.h states it: a (16 row, 64 k) tile of 1024 bytes laid
// out as 32 lanes x 32 bytes, lane l's 32 bytes being four 8-byte fragments whose bytes 8t cover
// k = t*16 + (l>>4)*8 + j of row l&15.
constexpr int kFragRows = 16;
constexpr int kFragK = 16;
constexpr int kSuperK = 64;

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

// A bf16 bit pattern's f64 value, the way the kernel reads the scale plane.
double of_bf16(uint16_t h) {
    const uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return static_cast<double>(f);
}

// A power-of-two scale's bf16 bit pattern (round half away from zero, the store's own rule).
uint16_t to_bf16_bits(double x) {
    const float f = static_cast<float>(x);
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    u += 0x8000u;
    return static_cast<uint16_t>(u >> 16);
}

// The encoder: the pin's own walk, frag_src then frag_walk, writing each 8-byte group of a row of
// the canonical [N][K] codes into the stored plane. `at` advances 8 bytes per group, exactly as
// r4d_rad_relayout_i8_block's frag_walk does.
std::vector<int8_t> encode_codes(const std::vector<int8_t>& code, int64_t N, int64_t K) {
    std::vector<int8_t> out(static_cast<std::size_t>(N * K), 0);
    int64_t at = 0;
    for (int64_t rt = 0; rt < N / kFragRows; ++rt)
        for (int64_t ks = 0; ks < K / kSuperK; ++ks)
            for (int lane = 0; lane < 32; ++lane)
                for (int t = 0; t < 4; ++t, at += 8) {
                    // frag_src: row = rt*16 + (lane & 15), k = ks*64 + t*16 + (lane >> 4)*8 + j.
                    const int64_t row = rt * kFragRows + (lane & 15);
                    const int64_t k0 = ks * kSuperK + t * kFragK + (lane >> 4) * 8;
                    for (int j = 0; j < 8; ++j)
                        out[static_cast<std::size_t>(at + j)] =
                            code[static_cast<std::size_t>(row * K + k0 + j)];
                }
    return out;
}

// The decoder: the kernel's inverse (src/adjoint/device/linear.hip's frag_go), the byte offset of
// the 8-byte group that starts at column k (k % 8 == 0) of `row`.
int64_t frag_at(int64_t K, int64_t row, int64_t k) {
    const int64_t kk = k & 63;
    const int64_t lane = (row & 15) + 16 * ((kk & 15) >> 3);
    return ((row >> 4) * (K >> 6) + (k >> 6)) * 32 * 32 + lane * 32 + (kk >> 4) * 8;
}

// The encoder's scale plane: r4d_rad_relayout_i8_block's loop, d[((r/16)*G + g)*16 + r%16] = s.
std::vector<uint16_t> encode_scales(const std::vector<double>& scale, int64_t N, int64_t G) {
    std::vector<uint16_t> out(static_cast<std::size_t>(N * G), 0);
    for (int64_t r = 0; r < N; ++r)
        for (int64_t g = 0; g < G; ++g)
            out[static_cast<std::size_t>(((r / 16) * G + g) * 16 + r % 16)] =
                to_bf16_bits(scale[static_cast<std::size_t>(r * G + g)]);
    return out;
}

// The decoder's scale index: the tile-major [N/16][K/128][16] plane, the kernel's t16_go.
int64_t t16_at(int64_t G, int64_t row, int64_t g) {
    return ((row >> 4) * G + g) * 16 + (row % 16);
}

uint64_t mix(uint64_t* s) {
    uint64_t x = (*s += 0x9e3779b97f4a7c15ull);
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// Both sides are f64 over the same integers and powers of two, so only the summation order differs.
const double rel_tol = 1e-9;

}  // namespace

// The stored layout round-trips, and the weight the decode produces gives the f64 row's dx.
R97_TEST(adjoint_private_weight_stored_layout_matches_the_f64_row) {
    const int64_t M = 4, N = 32, K = 256;
    const int64_t group = 128, G = K / group;

    // Canonical codes (small integers, exact in i8) and per-(row, 128-K) scales that are powers of
    // two, so a bf16 scale store is exact and the two sides start from the same numbers.
    std::vector<int8_t> code(static_cast<std::size_t>(N * K));
    std::vector<double> scale(static_cast<std::size_t>(N * G));
    uint64_t seed = 0x9a8a0001ull;
    for (std::size_t i = 0; i < code.size(); ++i)
        code[i] = static_cast<int8_t>(static_cast<int>(mix(&seed) % 9) - 4);
    for (std::size_t i = 0; i < scale.size(); ++i)
        scale[i] = static_cast<double>((mix(&seed) >> 16) % 16 + 1) / 256.0;

    // The engine's two planes, written the pin's way, then read back with the kernel's inverse.
    const std::vector<int8_t> codes_stored = encode_codes(code, N, K);
    const std::vector<uint16_t> scales_stored = encode_scales(scale, N, G);

    std::vector<double> weight(static_cast<std::size_t>(N * K));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const int8_t c = codes_stored[static_cast<std::size_t>(frag_at(K, n, k) + (k % 8))];
            const double s =
                of_bf16(scales_stored[static_cast<std::size_t>(t16_at(G, n, k / group))]);
            R97_CHECK(c == code[static_cast<std::size_t>(n * K + k)]);
            R97_CHECK(std::fabs(s - scale[static_cast<std::size_t>(n * G + k / group)]) < 1e-30);
            weight[static_cast<std::size_t>(n * K + k)] = static_cast<double>(c) * s;
        }

    // The cotangent, uniform, and the reference product dx = dy * W in f64.
    std::vector<double> dy(static_cast<std::size_t>(M * N));
    for (double& x : dy)
        x = 2.0 * (static_cast<double>(mix(&seed) >> 11) / 9007199254740992.0) - 1.0;
    std::vector<double> dx_ref(static_cast<std::size_t>(M * K), 0.0);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t k = 0; k < K; ++k) {
            double acc = 0.0;
            for (int64_t n = 0; n < N; ++n)
                acc += dy[static_cast<std::size_t>(m * N + n)] *
                       weight[static_cast<std::size_t>(n * K + k)];
            dx_ref[static_cast<std::size_t>(m * K + k)] = acc;
        }

    // The f64 host row, on the canonical codes and scales: the same record the walk builds, with
    // the weight and its scale as the values a readable operand would carry.
    OpRecord rec;
    rec.op = "gemm_nt_q";
    rec.params = {{"M", M}, {"N", N}, {"K", K}, {"group", group}, {"dtype", "i8a8h"}};
    rec.operands = {
        {"a", "i8", {M, K}, 0, M, false},
        {"a_scale", "f32", {M, G}, 0, M, false},
        {"b", "i8", {N, K}, 0, N, false},
        {"b_scale", "bf16", {N, G}, 0, N, false},
        {"y", "f32", {M, N}, 0, M, true},
    };
    std::vector<AdValue> fwd(rec.operands.size());
    fwd[0].dtype = "i8";
    fwd[0].shape = {M, K};
    fwd[0].value.assign(static_cast<std::size_t>(M * K), 0.0);
    fwd[1].dtype = "f32";
    fwd[1].shape = {M, G};
    fwd[1].value.assign(static_cast<std::size_t>(M * G), 1.0);
    fwd[2].dtype = "i8";
    fwd[2].shape = {N, K};
    fwd[2].value.assign(code.begin(), code.end());
    fwd[3].dtype = "bf16";
    fwd[3].shape = {N, G};
    fwd[3].value = scale;
    fwd[4].dtype = "f32";
    fwd[4].shape = {M, N};
    fwd[4].value.assign(static_cast<std::size_t>(M * N), 0.0);
    std::vector<AdValue> cot(1);
    cot[0].dtype = "f32";
    cot[0].shape = {M, N};
    cot[0].value = dy;
    std::vector<AdValue> dc(rec.operands.size());
    for (std::size_t i = 0; i < dc.size(); ++i) {
        dc[i].dtype = rec.operands[i].dtype;
        dc[i].shape = rec.operands[i].shape;
        dc[i].value.assign(static_cast<std::size_t>(numel(rec.operands[i].shape)), 0.0);
    }

    std::string why;
    const AdjointFn host = adjoint_find("gemm_nt_q", Side::host, rec.params, &why);
    R97_CHECK(host != nullptr);
    if (host == nullptr) {
        std::printf("adjoint_private_weight: no host row: %s\n", why.c_str());
        return;
    }
    AdjointCall call{rec, fwd, cot, &dc, nullptr, ""};
    host(call);

    const std::vector<double>& dx = dc[0].value;
    double num = 0.0, den = 0.0, worst = 0.0;
    for (std::size_t i = 0; i < dx.size(); ++i) {
        const double d = dx[i] - dx_ref[i];
        num += d * d;
        den += dx_ref[i] * dx_ref[i];
        worst = std::max(worst, std::fabs(d));
    }
    const double rel = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    std::printf("adjoint_private_weight: stored-layout decode vs f64 row, rel L2 %.3g, max abs "
                "%.3g\n",
                rel, worst);
    R97_CHECK(rel < rel_tol);
}

// The kill switch: with R97_NO_STORED_WEIGHTS=1 `snapshot()` gives a plugin-private operand neither
// values nor a pointer, so the walk refuses it by name -- the behaviour before the pointer was
// threaded, kept as the way to take the stored path out of a run. One operand is unspannable (`b`)
// and its scale is readable, so the refusal names `b` deterministically even though the guard's
// units run in parallel.
//
// The *default* path's loud decline -- an encoding that classifies but has no kernel built, or a
// plugin-private operand with no encoding at all -- is the device row's, so it is covered where the
// row runs: `adjoint_iqr_plane_unbuilt_form_refuses` and
// `adjoint_private_weight_device_refuses_without_an_encoding` in tests/gpu/. A host test cannot
// reach `linear.hip`: the row lives in libr97.so, not in libr97.a.
R97_TEST(adjoint_private_weight_kill_switch_refuses_by_name) {
    const int64_t M = 2, N = 32, K = 256;
    const int64_t group = 128, G = K / group;
    OpRecord rec;
    rec.op = "gemm_nt_q";
    rec.params = {{"M", M}, {"N", N}, {"K", K}, {"group", group}, {"dtype", "i8a8h"}};
    rec.operands = {
        {"a", "i8", {M, K}, 0, M, false},
        {"a_scale", "f32", {M, G}, 0, M, false},
        {"b", "i8", {N, K}, 0, N, false},
        {"b_scale", "bf16", {N, G}, 0, N, false},
        {"y", "f32", {M, N}, 0, M, true},
    };
    StoredSnapshots snaps;
    for (std::size_t i = 0; i < rec.operands.size(); ++i) {
        AdValue v;
        v.dtype = rec.operands[i].dtype;
        v.shape = rec.operands[i].shape;
        v.value.assign(static_cast<std::size_t>(numel(v.shape)), 0.0);
        if (rec.operands[i].buffer == "b") {
            // What snapshot() returns with the kill switch set: the shape, no values, no pointer
            // (tests/host/snapshot_view.cpp asserts that against snapshot() itself, including the
            // pointer the default path hands over).
            v.value.clear();
            v.raw = nullptr;
        }
        snaps.set(rec.position, i, std::move(v));
    }
    BoundaryNames names;
    names.add("y", 0, M);
    std::vector<Seed> seeds;
    seeds.emplace_back();
    seeds.back().buffer = "y";
    seeds.back().row_lo = 0;
    seeds.back().row_hi = M;
    seeds.back().cotangent.assign(static_cast<std::size_t>(M * N), 0.5f);
    BackwardOptions bo;
    bo.side = Side::host;

    std::string what;
    try {
        backward_block({rec}, snaps, names, seeds, {}, bo);
    } catch (const std::exception& e) {
        what = e.what();
    }
    std::printf("adjoint_private_weight: gate off refuses: %s\n", what.c_str());
    R97_CHECK(!what.empty());
    R97_CHECK(what.find("operand 'b'") != std::string::npos);
    R97_CHECK(what.find("0 values") != std::string::npos);
}

// The stored path is the default: a plugin-private operand reaches a row with the tape's pointer
// (`raw`) and no f64 values, and the walk admits it so a device row can decode the engine's
// layout. A host row that indexed the empty value would read out of bounds; one whose size check
// met it first (the fold's) would return silently and leave the activation's cotangent zero.
// Each host linear row the device step's host fallback can serve on a stored trunk therefore
// refuses a stored operand by name. The pointer's byte is a canary: the refusal happens before
// any read, so a regression to indexing it still throws out of range rather than reading it.
R97_TEST(host_linear_rows_refuse_a_stored_weight_by_name) {
    const int64_t M = 2, N = 32, K = 256;
    const int64_t group = 128, G = K / group;
    const int canary = 0;

    auto stored_case = [&](OpRecord rec, const std::string& weight, const std::string& seed) {
        StoredSnapshots snaps;
        for (std::size_t i = 0; i < rec.operands.size(); ++i) {
            AdValue v;
            v.dtype = rec.operands[i].dtype;
            v.shape = rec.operands[i].shape;
            v.value.assign(static_cast<std::size_t>(numel(v.shape)), 0.0);
            if (rec.operands[i].buffer == weight) {
                // What the default path of snapshot() hands over: the shape, the tape's pointer,
                // no values (src/join_common.h; tests/host/snapshot_view.cpp pins it).
                v.value.clear();
                v.raw = &canary;
            }
            snaps.set(rec.position, i, std::move(v));
        }
        BoundaryNames names;
        for (const Operand& o : rec.operands)
            if (o.written) names.add(o.buffer, 0, o.rows);
        std::vector<Seed> seeds;
        seeds.emplace_back();
        seeds.back().buffer = seed;
        seeds.back().row_lo = 0;
        seeds.back().row_hi = M;
        std::size_t n = 0;
        for (const Operand& o : rec.operands)
            if (o.buffer == seed) n = static_cast<std::size_t>(numel(o.shape));
        R97_CHECK(n != 0);
        seeds.back().cotangent.assign(n, 0.5f);
        BackwardOptions bo;
        bo.side = Side::host;

        std::string what;
        try {
            backward_block({rec}, snaps, names, seeds, {}, bo);
        } catch (const std::exception& e) {
            what = e.what();
        }
        std::printf("host_linear_rows: %s stored %s refuses: %s\n", rec.op.c_str(),
                    weight.c_str(), what.c_str());
        R97_CHECK(what.find("op '" + rec.op + "'") != std::string::npos);
        R97_CHECK(what.find("operand '" + weight + "'") != std::string::npos);
        R97_CHECK(what.find("plugin-private") != std::string::npos);
    };

    {
        OpRecord rec;
        rec.op = "gemm_nt";
        rec.params = {{"M", M}, {"N", N}, {"K", K}, {"dtype", "bf16"}};
        rec.operands = {{"a", "bf16", {M, K}, 0, M, false},
                        {"b", "bf16", {N, K}, 0, N, false},
                        {"y", "bf16", {M, N}, 0, M, true}};
        stored_case(rec, "b", "y");
    }
    {
        OpRecord rec;
        rec.op = "gemm_nt_q";
        rec.params = {{"M", M}, {"N", N}, {"K", K}, {"group", group}, {"dtype", "i8a8h"}};
        rec.operands = {{"a", "i8", {M, K}, 0, M, false},
                        {"a_scale", "f32", {M, G}, 0, M, false},
                        {"b", "i8", {N, K}, 0, N, false},
                        {"b_scale", "bf16", {N, G}, 0, N, false},
                        {"y", "f32", {M, N}, 0, M, true}};
        stored_case(rec, "b", "y");
    }
    {
        // The fold's shape check runs before its stored check, so this record has to pass it:
        // N/2 = 16 over group 16.
        OpRecord rec;
        rec.op = "gemm_nt_q_gated";
        rec.params = {{"M", M}, {"N", N}, {"K", K}, {"group", 16}, {"act", "silu"},
                      {"dtype", "i8a8h"}};
        rec.operands = {{"a", "i8", {M, K}, 0, M, false},
                        {"a_scale", "f32", {M, K / 16}, 0, M, false},
                        {"b", "i8", {N, K}, 0, N, false},
                        {"b_scale", "bf16", {N, K / 16}, 0, N, false},
                        {"q", "i8", {M, N / 2}, 0, M, true},
                        {"scale", "f32", {M, 1}, 0, M, true}};
        stored_case(rec, "b", "q");
    }
    {
        OpRecord rec;
        rec.op = "logits_gemm";
        rec.params = {{"n_embd", K}, {"n_vocab", N}};
        rec.operands = {{"x", "bf16", {M, K}, 0, M, false},
                        {"lm_head", "bf16", {N, K}, 0, N, false},
                        {"y", "bf16", {M, N}, 0, M, true}};
        stored_case(rec, "lm_head", "y");
    }
}
