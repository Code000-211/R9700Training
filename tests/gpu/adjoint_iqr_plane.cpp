// tests/gpu/adjoint_iqr_plane.cpp -- an IQR trunk weight's planes through the device row.
//
// The qwen3.8-27b-iq5r container's trunk is not int8 codes: it is an IQR form at ~5.07 bpw, so the
// int8 pair's N*K-byte plane is not it. The tape hands `gemm_nt_q` such a weight and its scale as
// plugin-private operands -- the weight's declared shape and one pointer each -- and the form rides
// beside them (AdValue::enc, patch 0124). The device row derives the form from that encoding
// (src/adjoint/device/iqr_plane.h), decodes the codes into the int8 fragment plane the dx kernel
// reads with libr4d's own trellis decoder, and writes the form's scale into the tile-major bf16
// plane that kernel reads (plane_index.h's frag_go/t16_go).
//
// This case writes those planes the format's way -- the lane segments and the scale planes of
// libr4d/r4d_iqr.h's layouts, restated here from the format's own description, so the addresses are
// written twice: once by the encoder and once from (row, k) -- and holds the row's dx to the f64
// product of the weight they decode to. A misread permutation moves a code or a scale and dx parts,
// and a form classified wrongly reads the wrong width.
//
// Two forms, one per scale shape the container's family uses: a table trellis on the tcq_rb grid
// (the block's own bf16 scale, the row's factor already in the codes) under an e4m3 level, and a
// computed trellis (kTrellisMad, so the plane's A/B words and the signed code path) under one f32 a
// row times one f32 a 128-K block. Every scale is a power of two, so the emitted bf16 plane is the
// plane's own product exactly and only the f32 accumulation and the one bf16 rounding of dx differ
// from the f64 reference.
//
// Label: gpu.
#include "r97/adjoint.h"
#include "r97/op.h"

#include "rad_device.h"
#include "rad_encoding.h"
#include "rad_types.h"

#include "support/device_check.h"
#include "support/gpu_guard.h"
#include "support/test.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

using namespace r97;

namespace {

int64_t align256(int64_t b) { return (b + 255) / 256 * 256; }

// The tile-major element index of (row, block of K) in a scale plane: libr4d/r4d_iqr.h's scale
// buffer, [N/16][K/128][16], which is the order plane_index.h's t16_go addresses.
int64_t t16_at(int64_t G, int64_t row, int64_t g) { return ((row >> 4) * G + g) * 16 + (row % 16); }

// The byte offset of the 8-byte group that starts at column k (k % 8 == 0) of `row` in the int8
// fragment plane: plane_index.h's frag_go, restated here so the diagnostic reads the plane the way
// the kernel does.
int64_t frag_at(int64_t K, int64_t row, int64_t k) {
    const int64_t kk = k & 63;
    const int64_t lane = (row & 15) + 16 * ((kk & 15) >> 3);
    return (((row >> 4) * (K >> 6) + (k >> 6)) * 32 + lane) * 32 + ((kk >> 4) * 8);
}

// The OCP E4M3 value of a byte: one sign, four exponent bits biased by 7, three mantissa bits,
// subnormals m/8 * 2^-6, and 0x7F/0xFF the format's only NaN. This case's planes hold 0x38 (1.0)
// and 0x30 (0.5), and the rest of the rule is here so the reading is the format's, not a table's.
double e4m3(uint8_t b) {
    const int e = (b >> 3) & 0xF, m = b & 0x7;
    const double mag = e == 0 ? (m / 8.0) * std::pow(2.0, -6)
                              : (e == 0xF ? (m == 0x7 ? std::nan("") : std::pow(2.0, 8))
                                          : (1.0 + m / 8.0) * std::pow(2.0, e - 7));
    return (b & 0x80) != 0 ? -mag : mag;
}

float f32_at(const std::vector<uint8_t>& p, int64_t off) {
    float f;
    std::memcpy(&f, p.data() + off, sizeof f);
    return f;
}

uint16_t u16_at(const std::vector<uint8_t>& p, int64_t off) {
    return static_cast<uint16_t>(p[static_cast<size_t>(off)] |
                                 (static_cast<uint16_t>(p[static_cast<size_t>(off + 1)]) << 8));
}

double of_bf16(uint16_t h) {
    const uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return static_cast<double>(f);
}

// A power-of-two scale's bf16 pattern. Every scale here is a power of two, so the store's rounding
// rule never decides a value.
uint16_t to_bf16_bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    u += 0x8000u;
    return static_cast<uint16_t>(u >> 16);
}

// `n` <= 32 bits of a little-endian bit stream from bit `off`: a row's codes are packed lowest
// first from bit 0 of its first byte (rad_encoding.h's PLANES).
uint32_t bits_at(const std::vector<uint8_t>& p, int64_t off, int n) {
    uint64_t w = 0;
    for (int i = 0; i < 5; ++i) {
        const int64_t at = (off >> 3) + i;
        w |= static_cast<uint64_t>(at >= 0 && at < static_cast<int64_t>(p.size())
                                       ? p[static_cast<size_t>(at)]
                                       : 0)
             << (8 * i);
    }
    return static_cast<uint32_t>((w >> (off & 7)) & (n == 32 ? 0xFFFFFFFFu : ((1u << n) - 1u)));
}

uint64_t mix(uint64_t* s) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    return *s;
}

// One case's form: the code bits, the state bits, the rows a scale (0 = one a row; above 0, the
// tcq_rb grid), whether the state's values are computed rather than read from a table, and whether
// the scale is an e4m3 level under one f32 (else one f32 a row times one f32 a 128-K block).
//
// `enc_L` overrides the state bits the *encoding* declares (0: the planes' own), so a form that
// classifies but has no kernel built is reachable; `refuse` expects the row to throw on it.
struct Case {
    int B, L, lrb;
    bool mad, e4m3;
    int enc_L = 0;
    bool refuse = false;
};

// Builds the form's two planes, reads the weight they decode to out of them by hand, and holds the
// device row's dx to the f64 product of that weight.
void iqr_case(const Case& cs) {
    const r97test::GuardOptions o = device_check::guard_options(int64_t(64) << 20);
    const r97test::DeviceGuard guard(o);
    device_check::require_device(o.device);
    (void)device_check::load_plugin();

    const int64_t M = 4, N = 32, K = 256;
    const int64_t nkb = K / 128;
    const int B = cs.B, L = cs.L;
    const int lrun = 5;            // the run: 32 rows, so two 16-row tiles
    const int64_t run_tiles = (int64_t(1) << lrun) / 16;  // the run's 16-row tiles
    const int64_t lane_bytes = (64 * B) / 8;
    const int64_t unit_bytes = 32 * lane_bytes;  // a unit is 32 lane segments
    const int64_t unit_n = (N / 16) * nkb;

    // ---- the weight buffer: the codes, then the form's aux -- the table, or a computed trellis's
    // 2^L values followed by its A and B.
    const int64_t codes_n = unit_n * unit_bytes;
    const int64_t aux_off = align256(codes_n + 16);
    const int64_t aux_n = cs.mad ? (int64_t(1) << L) + 8 : (int64_t(1) << L);
    const int64_t w_bytes = aux_off + align256(aux_n + 16);
    std::vector<uint8_t> W(static_cast<size_t>(w_bytes), 0);
    uint64_t seed = 0x1a70c0deull;
    for (int64_t i = 0; i < codes_n; ++i)
        W[static_cast<size_t>(i)] = static_cast<uint8_t>(mix(&seed));
    if (cs.mad) {
        // A = 1, B = 126: bytesum((st + 126) & 0x3F3F3F3F) - 126 is the state's value, and every
        // value lands in [-126, 126], which the codes of an int8 grid must.
        const uint32_t ab[2] = {1u, 126u};
        std::memcpy(W.data() + aux_off + (int64_t(1) << L), ab, sizeof ab);
    } else {
        for (int64_t i = 0; i < (int64_t(1) << L); ++i)  // small values: the codes of a grid
            W[static_cast<size_t>(aux_off + i)] =
                static_cast<uint8_t>(static_cast<int8_t>((i % 9) - 4));
    }

    // ---- the scale buffer in r4d_iqr.h's order: e4m3 per [1 x 128], then the f32 level and the
    // block's bf16; or one f32 a row, then one f32 a 128-K block.
    const int64_t s1_n = cs.e4m3 ? N * nkb : N * 4;
    const int64_t s2_off = align256(s1_n);
    const int64_t blk_off = cs.e4m3 ? s2_off + align256(16) : s2_off + align256(nkb * 4);
    const int64_t blk_n = cs.lrb != 0 ? (N >> cs.lrb) * nkb * 2 : 0;
    const int64_t s_bytes = blk_off + (blk_n != 0 ? align256(blk_n) : 0);
    std::vector<uint8_t> S(static_cast<size_t>(s_bytes), 0);
    if (cs.e4m3) {
        for (int64_t i = 0; i < s1_n; ++i)  // 1.0 / 0.5, the two e4m3 levels this case uses
            S[static_cast<size_t>(i)] = (i % 2) != 0 ? 0x38 : 0x30;
        const float level = 2.0f;
        std::memcpy(S.data() + s2_off, &level, sizeof level);
        if (blk_n != 0) {
            const uint16_t two = to_bf16_bits(2.0f);
            for (int64_t i = 0; i < blk_n / 2; ++i)
                std::memcpy(S.data() + blk_off + 2 * i, &two, sizeof two);
        }
    } else {
        for (int64_t r = 0; r < N; ++r) {
            const float v = (r % 2) != 0 ? 2.0f : 1.0f;
            std::memcpy(S.data() + 4 * r, &v, sizeof v);
        }
        for (int64_t g = 0; g < nkb; ++g) {
            const float v = (g % 2) != 0 ? 2.0f : 1.0f;
            std::memcpy(S.data() + s2_off + 4 * g, &v, sizeof v);
        }
    }

    // ---- the encoding the tape carries beside the pointer: which form these planes are.
    RadEncoding enc;
    rad_enc_clear(&enc);
    rad_enc_copy_str(enc.scheme, "trellis");  // the scheme libr4d's classify reads a trellis as
    rad_enc_copy_str(enc.transform, "fwht128");  // the hook's transform for an i8a8h operand
    rad_enc_add_plane(&enc, "codes", cs.B == 2 ? RAD_U2 : RAD_U5, 1, 1);
    rad_enc_add_table(&enc, "trellis", RAD_I8, int64_t(1) << (cs.enc_L != 0 ? cs.enc_L : L), 1);
    if (cs.mad) rad_enc_add_table(&enc, "trellis.mad", RAD_U32, 2, 1);
    if (cs.e4m3) {
        rad_enc_add_plane(&enc, "scale", RAD_F8E4M3, 1, 128);
        rad_enc_add_plane(&enc, "scale.1", RAD_F32, 0, 0);
    } else {
        rad_enc_add_plane(&enc, "scale", RAD_F32, 1, 0);
        rad_enc_add_plane(&enc, "scale.col", RAD_F32, 0, 128);
    }
    // The trellis plane carries the run and the rows a scale in its block (r4d_iqr_layout.cpp:101).
    enc.plane[1].block[0] = int64_t(1) << lrun;
    enc.plane[1].block[1] = cs.lrb != 0 ? (int64_t(1) << cs.lrb) : 0;

    // ---- the reference: the row's own scale, the block's fold, and the ratio the codes were
    // chosen against (r4d_iqr.h's own_scale / fold_scale / rb_ratio, as the row's decoder reads
    // them back).
    std::vector<double> own(static_cast<size_t>(N * nkb), 1.0);
    std::vector<double> fold(static_cast<size_t>(N * nkb), 1.0);
    for (int64_t r = 0; r < N; ++r)
        for (int64_t kb = 0; kb < nkb; ++kb) {
            const size_t i = static_cast<size_t>(r * nkb + kb);
            if (cs.e4m3) {
                const double level = static_cast<double>(f32_at(S, s2_off));
                const double e4 = e4m3(S[static_cast<size_t>(t16_at(nkb, r, kb))]);
                own[i] = of_bf16(to_bf16_bits(static_cast<float>(level * e4)));
                fold[i] = cs.lrb != 0
                              ? of_bf16(u16_at(S, blk_off + 2 * ((r >> cs.lrb) * nkb + kb)))
                              : own[i];
            } else {
                fold[i] = static_cast<double>(f32_at(S, 4 * r)) * f32_at(S, s2_off + 4 * kb);
            }
        }
    std::vector<double> scale(static_cast<size_t>(N * nkb));
    for (size_t i = 0; i < scale.size(); ++i) scale[i] = fold[i];

    // The codes: column e of a lane's segment is 16 B-bit codes (the tile's row 15 first) and then
    // the same column's first bits in the unit above -- the run's last tile at a run's first,
    // tail-biting. A row's state is the L bits from its own code.
    //
    // `bits_at` takes a *bit* offset (the codes plane is one bit stream, r4d_iqr.h:30-45), so a
    // unit's byte offset is scaled to bits here: `seg`/`pseg` are bit offsets of the lane segment,
    // and col/in_col below are bit offsets within it. Mixing the two reads the plane an eighth of
    // the way in, which is a whole different unit's codes -- the first run of this case measured
    // rel L2 1.46 that way.
    const uint32_t* ab = cs.mad
                             ? reinterpret_cast<const uint32_t*>(W.data() + aux_off + (int64_t(1) << L))
                             : nullptr;
    std::vector<int8_t> code(static_cast<size_t>(N * K));
    for (int64_t r = 0; r < N; ++r)
        for (int64_t k = 0; k < K; ++k) {
            const int64_t rt = r / 16, within = r % 16, kb = k / 128;
            const int64_t lane = (k % 128) / 4, e = k % 4;
            const int64_t rp = (rt % run_tiles == 0) ? rt + run_tiles - 1 : rt - 1;
            const int64_t seg = ((rt * nkb + kb) * unit_bytes + lane * lane_bytes) * 8;
            const int64_t pseg = ((rp * nkb + kb) * unit_bytes + lane * lane_bytes) * 8;
            const int64_t q = 15 - within;  // the stream position of this row's code (bottom-up)
            const int64_t col = e * 16 * B;  // where this column's stream starts in the segment
            const int64_t in_col = q * B;    // the row's own code within it
            uint32_t st;
            if (in_col + L <= 16 * B) {
                st = bits_at(W, seg + col + in_col, L);
            } else {
                const uint32_t lo =
                    bits_at(W, seg + col + in_col, static_cast<int>(16 * B - in_col));
                const uint32_t hi =
                    bits_at(W, pseg + col, static_cast<int>(in_col + L - 16 * B));
                st = lo | (hi << (16 * B - in_col));
            }
            int v;
            if (cs.mad) {
                uint32_t h = static_cast<uint32_t>(st * ab[0]) + ab[1];
                h &= 0x3F3F3F3Fu;
                v = static_cast<int>((h & 0xFFu) + ((h >> 8) & 0xFFu) + ((h >> 16) & 0xFFu) +
                                     ((h >> 24) & 0xFFu)) -
                    126;
            } else {
                v = static_cast<int>(static_cast<int8_t>(W[static_cast<size_t>(aux_off + st)]));
            }
            if (cs.lrb != 0) {  // rint(v * own / B), libquant's rb_round (r4d_selftest.cpp:6847-6849)
                const double ratio = own[static_cast<size_t>(r * nkb + kb)] /
                                     fold[static_cast<size_t>(r * nkb + kb)];
                // Round half to even, not away from zero: the device's `rows` rounds the product
                // with the hardware's nearest-even, and the two differ exactly on the halves
                // (|v| = 1 at ratio 0.5), which is a handful of large elements rather than noise.
                v = static_cast<int>(std::nearbyint(static_cast<double>(v) * ratio));
            }
            R97_CHECK(v >= -128 && v <= 127);
            code[static_cast<size_t>(r * K + k)] = static_cast<int8_t>(v);
        }

    std::vector<double> weight(static_cast<size_t>(N * K));
    for (int64_t r = 0; r < N; ++r)
        for (int64_t k = 0; k < K; ++k)
            weight[static_cast<size_t>(r * K + k)] =
                static_cast<double>(code[static_cast<size_t>(r * K + k)]) *
                scale[static_cast<size_t>(r * nkb + k / 128)];

    std::vector<double> dy(static_cast<size_t>(M * N));
    for (double& x : dy)
        x = 2.0 * (static_cast<double>(mix(&seed) >> 11) / 9007199254740992.0) - 1.0;
    const bool dbg = std::getenv("R97_IQR_DEBUG") != nullptr;
    if (dbg) {  // one-hot at n = 0: dx[m,k] is the decoded weight's row 0, element by element
        std::fill(dy.begin(), dy.end(), 0.0);
        for (int64_t m = 0; m < M; ++m) dy[static_cast<size_t>(m * N)] = 1.0;
    }
    std::vector<double> dx_ref(static_cast<size_t>(M * K), 0.0);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t k = 0; k < K; ++k) {
            double acc = 0.0;
            for (int64_t n = 0; n < N; ++n)
                acc += dy[static_cast<size_t>(m * N + n)] * weight[static_cast<size_t>(n * K + k)];
            dx_ref[static_cast<size_t>(m * K + k)] = acc;
        }

    // ---- the two planes on the card, where the tape's pointers name them.
    void* const dev_w = rad_dev_alloc(w_bytes, RAD_MEM_DEVICE);
    void* const dev_s = rad_dev_alloc(s_bytes, RAD_MEM_DEVICE);
    R97_CHECK(dev_w != nullptr && dev_s != nullptr);
    RadStream stream = nullptr;
    R97_CHECK(rad_stream_create(&stream, 0) == 0);
    R97_CHECK(rad_memcpy_async(dev_w, W.data(), w_bytes, stream) == 0);
    R97_CHECK(rad_memcpy_async(dev_s, S.data(), s_bytes, stream) == 0);
    R97_CHECK(rad_stream_sync(stream) == 0);

    // The record and its operands exactly as the tape hands them: the weight and its scale carry a
    // shape, a device pointer and the encoding, and no values.
    OpRecord rec;
    rec.op = "gemm_nt_q";
    rec.params = {{"M", M}, {"N", N}, {"K", K}, {"group", int64_t(128)}, {"dtype", "i8a8h"}};
    rec.operands = {
        {"a", "i8", {M, K}, 0, M, false},
        {"a_scale", "f32", {M, nkb}, 0, M, false},
        {"b", "plugin-private", {N, K}, 0, N, false},
        {"b_scale", "plugin-private", {N, K}, 0, N, false},
        {"y", "f32", {M, N}, 0, M, true},
    };
    std::vector<AdValue> fwd(rec.operands.size());
    fwd[0].dtype = "i8";
    fwd[0].shape = {M, K};
    fwd[0].value.assign(static_cast<size_t>(M * K), 0.0);
    fwd[1].dtype = "f32";
    fwd[1].shape = {M, nkb};
    fwd[1].value.assign(static_cast<size_t>(M * nkb), 1.0);
    fwd[2].dtype = "plugin-private";
    fwd[2].shape = {N, K};
    fwd[2].raw = dev_w;
    fwd[2].enc = &enc;
    fwd[3].dtype = "plugin-private";
    fwd[3].shape = {N, K};  // the record declares the scale with the weight's shape
    fwd[3].raw = dev_s;
    fwd[3].enc = &enc;
    fwd[4].dtype = "f32";
    fwd[4].shape = {M, N};
    fwd[4].value.assign(static_cast<size_t>(M * N), 0.0);
    std::vector<AdValue> cot(1);
    cot[0].dtype = "f32";
    cot[0].shape = {M, N};
    cot[0].value = dy;
    std::vector<AdValue> dc(rec.operands.size());
    for (size_t i = 0; i < dc.size(); ++i) {
        dc[i].dtype = rec.operands[i].dtype;
        dc[i].shape = rec.operands[i].shape;
        int64_t n = 1;
        for (int64_t d : rec.operands[i].shape) n *= d;
        dc[i].value.assign(static_cast<size_t>(n), 0.0);
    }

    std::string why;
    const AdjointFn device = adjoint_find("gemm_nt_q", Side::device, rec.params, &why);
    R97_CHECK(device != nullptr);
    if (device == nullptr) {
        std::printf("adjoint_iqr_plane: no device row: %s\n", why.c_str());
        return;
    }
    AdjointCall call{rec, fwd, cot, &dc, nullptr, ""};
    if (cs.refuse) {
        // A form the encoding classifies but this build has no kernel for must not come back as a
        // zero dx: the row names it and throws. `iqr_plane_supported` is the classification, not a
        // promise that a kernel exists, so the decode's own refusal is the one that has to be loud.
        std::string what;
        bool threw = false;
        try {
            device(call);
        } catch (const std::exception& e) {
            threw = true;
            what = e.what();
        }
        std::printf("adjoint_iqr_plane: trellis b%d L%d lrb%d, encoding says L%d: %s\n", cs.B, L,
                    cs.lrb, cs.enc_L, threw ? what.c_str() : "NOT REFUSED");
        R97_CHECK(threw);
        R97_CHECK(what.find("iqr dx declined") != std::string::npos);
        R97_CHECK(what.find("did not decode") != std::string::npos);
        R97_CHECK(what.find("L" + std::to_string(cs.enc_L)) != std::string::npos);
        rad_dev_free(dev_w, RAD_MEM_DEVICE);
        rad_dev_free(dev_s, RAD_MEM_DEVICE);
        return;
    }
    device(call);

    const std::vector<double>& dx = dc[0].value;
    double num = 0.0, den = 0.0, worst = 0.0;
    for (size_t i = 0; i < dx.size(); ++i) {
        const double d = dx[i] - dx_ref[i];
        num += d * d;
        den += dx_ref[i] * dx_ref[i];
        worst = std::max(worst, std::fabs(d));
    }
    const double rel = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    if (dbg) {
        // The decode's own plane, through the plugin, against the reference codes: this says
        // whether the codes or the scale is what disagrees, without going through dx.
        using DecodeFn = bool (*)(const void*, uint32_t, uint32_t, const void*, const void*, void*,
                                  void*);
        void* const lib = dlopen(R97_DEVICE_PLUGIN, RTLD_NOW | RTLD_NOLOAD);
        auto dec = lib == nullptr
                       ? static_cast<DecodeFn>(nullptr)
                       : reinterpret_cast<DecodeFn>(
                             dlsym(lib, "_ZN3r976device16iqr_plane_decodeEPKvjjS2_S2_PvS3_"));
        if (dec == nullptr) {
            std::printf("  plane: no decode entry point (%s)\n", dlerror());
        } else {
            void* const dev = rad_dev_alloc(N * K, RAD_MEM_DEVICE);
            std::vector<uint8_t> got(static_cast<size_t>(N * K), 0);
            const bool ok = dec(&enc, static_cast<uint32_t>(N), static_cast<uint32_t>(K), dev_w,
                                dev_s, dev, stream);
            R97_CHECK(rad_stream_sync(stream) == 0);
            R97_CHECK(rad_memcpy_async(got.data(), dev, N * K, stream) == 0);
            R97_CHECK(rad_stream_sync(stream) == 0);
            int bad = 0;
            for (int64_t r = 0; r < N && bad < 6; ++r)
                for (int64_t k = 0; k < K && bad < 6; ++k) {
                    const int8_t g =
                        static_cast<int8_t>(got[static_cast<size_t>(frag_at(K, r, k) + (k % 8))]);
                    if (g != code[static_cast<size_t>(r * K + k)]) {
                        std::printf("  plane r%lld k%lld: decode %d, reference %d\n", (long long)r,
                                    (long long)k, (int)g,
                                    (int)code[static_cast<size_t>(r * K + k)]);
                        ++bad;
                    }
                }
            std::printf("  plane: decode %s, %s\n", ok ? "ran" : "declined",
                        bad == 0 ? "matches the reference codes" : "disagrees");
            rad_dev_free(dev, RAD_MEM_DEVICE);
        }
        for (int64_t k = 0; k < 24; ++k)
            std::printf("  k %2lld got %8.3f | r0 %8.3f r15 %8.3f r1 %8.3f r14 %8.3f\n",
                        (long long)k, dx[static_cast<size_t>(k)],
                        weight[static_cast<size_t>(0 * K + k)],
                        weight[static_cast<size_t>(15 * K + k)],
                        weight[static_cast<size_t>(1 * K + k)],
                        weight[static_cast<size_t>(14 * K + k)]);
    }
    std::printf("adjoint_iqr_plane: %s b%d L%d lrb%d scale=%s, dx vs the f64 product, rel L2 "
                "%.3g, max abs %.3g\n",
                cs.mad ? "trellis.mad" : "trellis", cs.B, cs.L, cs.lrb, cs.e4m3 ? "e4m3" : "row-f32",
                rel, worst);
    // f32 accumulation and one bf16 rounding of dx: the figure the linear family's rows are held to.
    R97_CHECK(rel <= 1e-2);

    rad_dev_free(dev_w, RAD_MEM_DEVICE);
    rad_dev_free(dev_s, RAD_MEM_DEVICE);
}

}  // namespace

R97_TEST(adjoint_iqr_plane_trellis_run_e4m3) { iqr_case(Case{5, 14, 5, false, true}); }
R97_TEST(adjoint_iqr_plane_trellis_mad_row_f32) { iqr_case(Case{2, 16, 0, true, false}); }
// L 13 classifies (r4d_iqr_layout.cpp:127 admits L <= 14) and no kernel in R4D_IQR_FORMS is built
// at it, so the row must name the form and throw rather than leave dx at zero.
R97_TEST(adjoint_iqr_plane_unbuilt_form_refuses) {
    iqr_case(Case{5, 14, 0, false, true, 13, true});
}
