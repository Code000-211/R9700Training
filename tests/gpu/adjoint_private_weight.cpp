// tests/gpu/adjoint_private_weight.cpp -- the w8a8_fwht128 stored planes through the device row.
//
// The real container hands `gemm_nt_q` its weight and its scale as plugin-private operands: no f64
// values, one device pointer each, the engine's own layout. The device row reads them in place
// (frag64 codes and a t16 scale; src/adjoint/device/linear.hip's frag_go/t16_go). This case builds
// both planes the pin's way, hands the row exactly the operands the tape would -- the shape and the
// pointer, no values -- and holds its dx to the f64 product of the weight those planes decode to.
// It is the fast proof of the kernel's indexing; the model tier's device_step case is the same path
// on the whole container.
//
// The planes are written by the pin's own walk (libr4d/r4d_fp8_frag.h's frag_src and
// r4d_rad_relayout_i8_block's t16 loop) and read back here with an inverse written from (row, k),
// so an indexing error moves a code or a scale and dx parts.
//
// Label: gpu.
#include "r97/adjoint.h"
#include "r97/op.h"
#include "rad_device.h"
#include "support/device_check.h"
#include "support/gpu_guard.h"
#include "support/test.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace r97;

namespace {

constexpr int kFragRows = 16;
constexpr int kFragK = 16;
constexpr int kSuperK = 64;

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

double of_bf16(uint16_t h) {
    const uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return static_cast<double>(f);
}

uint16_t to_bf16_bits(double x) {
    const float f = static_cast<float>(x);
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    u += 0x8000u;
    return static_cast<uint16_t>(u >> 16);
}

// The pin's walk: frag_src over (row tile, 64-k superblock, lane, fragment), eight bytes a group.
std::vector<int8_t> encode_codes(const std::vector<int8_t>& code, int64_t N, int64_t K) {
    std::vector<int8_t> out(static_cast<std::size_t>(N * K), 0);
    int64_t at = 0;
    for (int64_t rt = 0; rt < N / kFragRows; ++rt)
        for (int64_t ks = 0; ks < K / kSuperK; ++ks)
            for (int lane = 0; lane < 32; ++lane)
                for (int t = 0; t < 4; ++t, at += 8) {
                    const int64_t row = rt * kFragRows + (lane & 15);
                    const int64_t k0 = ks * kSuperK + t * kFragK + (lane >> 4) * 8;
                    for (int j = 0; j < 8; ++j)
                        out[static_cast<std::size_t>(at + j)] =
                            code[static_cast<std::size_t>(row * K + k0 + j)];
                }
    return out;
}

// The kernel's inverse: the byte offset of the 8-byte group starting at column k of `row`.
int64_t frag_at(int64_t K, int64_t row, int64_t k) {
    const int64_t kk = k & 63;
    const int64_t lane = (row & 15) + 16 * ((kk & 15) >> 3);
    return ((row >> 4) * (K >> 6) + (k >> 6)) * 32 * 32 + lane * 32 + (kk >> 4) * 8;
}

// The encoder's scale plane: d[((r/16)*G + g)*16 + r%16] = s[r*G + g].
std::vector<uint16_t> encode_scales(const std::vector<double>& scale, int64_t N, int64_t G) {
    std::vector<uint16_t> out(static_cast<std::size_t>(N * G), 0);
    for (int64_t r = 0; r < N; ++r)
        for (int64_t g = 0; g < G; ++g)
            out[static_cast<std::size_t>(((r / 16) * G + g) * 16 + r % 16)] =
                to_bf16_bits(scale[static_cast<std::size_t>(r * G + g)]);
    return out;
}

int64_t t16_at(int64_t G, int64_t row, int64_t g) { return ((row >> 4) * G + g) * 16 + (row % 16); }

uint64_t mix(uint64_t* s) {
    uint64_t x = (*s += 0x9e3779b97f4a7c15ull);
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

}  // namespace

R97_TEST(adjoint_private_weight_device_reads_the_stored_planes) {
    const r97test::GuardOptions o = device_check::guard_options(int64_t(64) << 20);
    const r97test::DeviceGuard guard(o);
    device_check::require_device(o.device);
    (void)device_check::load_plugin();

    const int64_t M = 4, N = 32, K = 256;
    const int64_t group = 128, G = K / group;

    // Canonical codes and power-of-two scales: exact in i8 and in bf16, so the two sides start
    // from the same numbers and only the summation order and one bf16 rounding differ.
    std::vector<int8_t> code(static_cast<std::size_t>(N * K));
    std::vector<double> scale(static_cast<std::size_t>(N * G));
    uint64_t seed = 0x5150a000ull;
    for (std::size_t i = 0; i < code.size(); ++i)
        code[i] = static_cast<int8_t>(static_cast<int>(mix(&seed) % 9) - 4);
    for (std::size_t i = 0; i < scale.size(); ++i)
        scale[i] = static_cast<double>((mix(&seed) >> 16) % 16 + 1) / 256.0;

    // The engine's two planes, then the weight they decode to, the reference the row must meet.
    const std::vector<int8_t> codes_stored = encode_codes(code, N, K);
    const std::vector<uint16_t> scales_stored = encode_scales(scale, N, G);
    std::vector<double> weight(static_cast<std::size_t>(N * K));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const int8_t c = codes_stored[static_cast<std::size_t>(frag_at(K, n, k) + (k % 8))];
            const double s =
                of_bf16(scales_stored[static_cast<std::size_t>(t16_at(G, n, k / group))]);
            R97_CHECK(c == code[static_cast<std::size_t>(n * K + k)]);
            weight[static_cast<std::size_t>(n * K + k)] = static_cast<double>(c) * s;
        }

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

    // The two planes on the card, in the engine's layout: what the tape's pointers name.
    void* const dev_codes = rad_dev_alloc(static_cast<int64_t>(N * K), RAD_MEM_DEVICE);
    void* const dev_scales = rad_dev_alloc(static_cast<int64_t>(N * G * 2), RAD_MEM_DEVICE);
    R97_CHECK(dev_codes != nullptr && dev_scales != nullptr);
    RadStream stream = nullptr;
    R97_CHECK(rad_stream_create(&stream, 0) == 0);
    R97_CHECK(rad_memcpy_async(dev_codes, codes_stored.data(), static_cast<int64_t>(N * K),
                               stream) == 0);
    R97_CHECK(rad_memcpy_async(dev_scales, scales_stored.data(), static_cast<int64_t>(N * G * 2),
                               stream) == 0);
    R97_CHECK(rad_stream_sync(stream) == 0);

    // The record the walk builds, and its operands exactly as the tape hands them: the weight and
    // its scale carry a shape and a device pointer and no values.
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
    fwd[2].dtype = "w8a8_fwht128";
    fwd[2].shape = {N, K};
    fwd[2].raw = dev_codes;  // no values: the plugin-private operand the tape records
    fwd[3].dtype = "w8a8_fwht128";
    fwd[3].shape = {N, G};
    fwd[3].raw = dev_scales;
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
    const AdjointFn device = adjoint_find("gemm_nt_q", Side::device, rec.params, &why);
    R97_CHECK(device != nullptr);
    if (device == nullptr) {
        std::printf("adjoint_private_weight: no device row: %s\n", why.c_str());
        return;
    }
    AdjointCall call{rec, fwd, cot, &dc, nullptr, ""};
    device(call);

    const std::vector<double>& dx = dc[0].value;
    double num = 0.0, den = 0.0, worst = 0.0;
    for (std::size_t i = 0; i < dx.size(); ++i) {
        const double d = dx[i] - dx_ref[i];
        num += d * d;
        den += dx_ref[i] * dx_ref[i];
        worst = std::max(worst, std::fabs(d));
    }
    const double rel = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    std::printf("adjoint_private_weight: device stored planes vs f64 product, rel L2 %.3g, max abs "
                "%.3g\n",
                rel, worst);
    // The kernel accumulates in f32 and rounds dx once to bf16 (one unit roundoff is 2^-9 of the
    // element), so the relative L2 is held to the bf16 figure the family uses.
    R97_CHECK(rel <= 1e-2);

    rad_dev_free(dev_codes, RAD_MEM_DEVICE);
    rad_dev_free(dev_scales, RAD_MEM_DEVICE);
}
