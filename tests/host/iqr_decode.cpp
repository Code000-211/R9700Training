// tests/host/iqr_decode.cpp -- the IQR decode against libquant's own host decode.
//
// The device kernel and the host implementation in src/iqr_decode.h are only as good as the
// reference they claim to follow, and no card is needed to check that: libquant's
// lq::grid_tcq_decode decodes the same stored planes. This test builds synthetic IQR weights in
// every form the recipe stores (a computed mad codebook or a stored one, f32/bf16/e4m3 scales, a
// second level, and tcq_rb row blocks) and holds our decode to libquant's values bit for bit.
//
// It also checks the parts the container does not exercise: a chunk that starts at a run boundary
// and a weight whose last run is short, and the chunk's output against the whole weight's.
//
// libquant is required, not optional: a library or symbol that cannot be found fails the test,
// because a run whose reference comparison never happened must not read as one that passed (the
// libref load of tests/host/adjoint_qk.cpp fails the same way).
#include "../../src/iqr_decode.h"

#include "../../third_party/radiance/libquant/lq_grid.h"
#include "../../third_party/radiance/abi/rad_plugin.h"

#include "support/test.h"

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {

using lq_decode_fn = int (*)(const RadEncoding*, int64_t, int64_t, const void* const*, int64_t,
                             int64_t, float*);

// The libquant symbols are C++ ones (namespace lq); dlsym needs their mangled names.
constexpr const char* kGridTcqDecode = "_ZN2lq15grid_tcq_decodeEPK11RadEncodingllPKPKvllPf";

// The affine grid's reference is grid_decode, one value at a time, so it takes libquant's Grid and
// BlockScale rather than an encoding.
constexpr const char* kGridDecode = "_ZN2lq11grid_decodeERKNS_4GridERKNS_10BlockScaleEi";
using lq_grid_decode_fn = float (*)(const lq::Grid&, const lq::BlockScale&, int);

lq_decode_fn g_lq = nullptr;
lq_grid_decode_fn g_grid = nullptr;

// The reference library sits where the radiance build leaves it. The host test runs from the build
// directory (ctest) or from wherever a person invokes it, so look up and down the tree too.
std::string find_libquant() {
    const char* env = std::getenv("R97_LIBQUANT");
    if (env && *env) return env;
    std::vector<std::string> dirs = { ".", "..", "../..", "../../.." };
    for (const std::string& d : dirs) {
        std::string p = d + "/radiance-build/radiance_home/quantizers/libquant.so";
        if (FILE* f = std::fopen(p.c_str(), "rb")) { std::fclose(f); return p; }
    }
    return {};
}

// Loads libquant and takes the two reference decodes out of it. The handle stays open for the rest
// of the run: the function pointers below are called until the test ends.
void load_libquant() {
    const std::string path = find_libquant();
    if (path.empty())
        throw std::runtime_error(
            "test: cannot find libquant; set R97_LIBQUANT to the radiance build's "
            "radiance_home/quantizers/libquant.so (also looked up from '.', '..', '../..' and "
            "'../../..')");
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) throw std::runtime_error("test: cannot load libquant at " + path + ": " + dlerror());
    void* p = dlsym(h, kGridTcqDecode);
    if (!p)
        throw std::runtime_error("test: " + path + " has no lq::grid_tcq_decode (" +
                                 kGridTcqDecode + "); has its signature changed?");
    std::memcpy(&g_lq, &p, sizeof(p));
    p = dlsym(h, kGridDecode);
    if (!p)
        throw std::runtime_error("test: " + path + " has no lq::grid_decode (" + kGridDecode +
                                 "); has its signature changed?");
    std::memcpy(&g_grid, &p, sizeof(p));
}

struct Synth {
    int bits = 4, state_bits = 16, run_rows = 256, rb = 1, mad = 0;
    int scale_dtype = R97_IQR_F32;
    bool rowf = false;      // one f32 scale a row, with scale.col (set to ones)
    bool second = false;    // a scale.1 second level (e4m3's form)
    bool partial = false;   // a last run shorter than run_rows
    int64_t rows = 0, cols = 256;

    RadEncoding enc{};
    std::vector<std::vector<uint8_t>> plane;
    std::vector<const void*> ptr;
    std::vector<int8_t> lut;
    int64_t bc = 0;
    int64_t nkb = 0;

    // our description, pointing into `plane`
    R97IqrDecodeDesc desc{};

    void build() {
        nkb = cols / 128;
        bc = rowf ? cols : 128;
        rad_enc_clear(&enc);
        rad_enc_copy_str(enc.scheme, "trellis");
        rad_enc_add_plane(&enc, "codes", bits == 3 ? RAD_U3 : bits == 4 ? RAD_U4 : bits == 5 ? RAD_U5 : RAD_U6, 1, 1);
        rad_enc_add_plane(&enc, "scale", scale_dtype == R97_IQR_F32 ? RAD_F32
                                        : scale_dtype == R97_IQR_BF16 ? RAD_BF16
                                                                      : RAD_F8E4M3,
                         1, rowf ? 0 : 128);
        if (second) rad_enc_add_plane(&enc, "scale.1", RAD_F32, 0, 0);
        const int tp = rad_enc_add_table(&enc, "trellis", RAD_I8, (int64_t)1 << state_bits, 1);
        enc.plane[tp].block[0] = run_rows;
        enc.plane[tp].block[1] = rb;
        if (mad) rad_enc_add_table(&enc, "trellis.mad", RAD_U32, 2, 1);
        if (rowf) rad_enc_add_plane(&enc, "scale.col", RAD_F32, 0, 128);

        plane.assign((size_t)enc.n_planes, {});
        const int64_t S = (int64_t)1 << state_bits;
        for (int i = 0; i < enc.n_planes; ++i) {
            int64_t r = 0, c = 0;
            rad_enc_plane_dims(&enc.plane[i], rows, cols, &r, &c);
            plane[(size_t)i].assign((size_t)(r * rad_enc_row_bytes(enc.plane[i].dtype, c)), 0);
        }
        ptr.resize((size_t)enc.n_planes);
        for (int i = 0; i < enc.n_planes; ++i) ptr[(size_t)i] = plane[(size_t)i].data();

        const int ic = rad_enc_find(&enc, "codes"), is = rad_enc_find(&enc, "scale");
        const int it = rad_enc_find(&enc, "trellis"), im = rad_enc_find(&enc, "trellis.mad");
        const int i2 = rad_enc_find(&enc, "scale.1"), icf = rad_enc_find(&enc, "scale.col");

        // Codes: a deterministic stream, no structure the decoder could accidentally rely on.
        uint32_t z = 0x9E3779B9u ^ (uint32_t)(bits * 131 + state_bits * 17 + run_rows + rb * 7 + cols);
        auto rnd = [&]() {
            z ^= z << 13; z ^= z >> 17; z ^= z << 5;
            return z;
        };
        const int64_t crow = rad_enc_row_bytes(enc.plane[ic].dtype, cols);
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t k = 0; k < cols; ++k)
                rad_store_bits(plane[(size_t)ic].data() + r * crow, bits, k,
                               rnd() & ((1u << bits) - 1u));

        // The codebook.
        const uint32_t A = (rnd() & 0xFFFFFFu) | 1u, B = rnd();
        if (mad) {
            uint32_t* ab = (uint32_t*)plane[(size_t)im].data();
            ab[0] = A;
            ab[1] = B;
        }
        for (int64_t st = 0; st < S; ++st) {
            if (mad) {
                const uint32_t h = (((uint32_t)st & 0xFFFFFFu) * A + B) & 0x3F3F3F3Fu;
                plane[(size_t)it][(size_t)st] =
                    (uint8_t)(int8_t)((int)((h & 0xFFu) + ((h >> 8) & 0xFFu) + ((h >> 16) & 0xFFu) +
                                            (h >> 24)) - 126);
            } else {
                plane[(size_t)it][(size_t)st] = (uint8_t)(int8_t)((int)(rnd() % 253) - 126);
            }
        }

        // Scales. Keep them positive and in a narrow band, as a weight's are.
        const int64_t srow = rad_enc_row_bytes(enc.plane[is].dtype, rowf ? 1 : nkb);
        for (int64_t r = 0; r < rows; ++r) {
            const int64_t n = rowf ? 1 : nkb;
            for (int64_t g = 0; g < n; ++g) {
                const float v = 0.0005f + (float)(rnd() % 1000) * 0.00001f;
                uint8_t* p = plane[(size_t)is].data() + r * srow;
                if (scale_dtype == R97_IQR_F32) ((float*)p)[g] = v;
                else if (scale_dtype == R97_IQR_BF16) ((uint16_t*)p)[g] = r97_iqr_f32_to_bf16(v);
                else {
                    // e4m3: a positive value in [2^-6, 1); the exact bits do not matter, the
                    // decoder and libquant read the same byte.
                    p[g] = (uint8_t)(0x20u | (uint8_t)(rnd() & 0x1Fu));
                }
            }
        }
        if (second) {
            // One f32 for the whole tensor, exact in bf16 after the product's settle.
            ((float*)plane[(size_t)i2].data())[0] = 0.625f;
        }
        if (rowf) {
            float* cf = (float*)plane[(size_t)icf].data();
            for (int64_t g = 0; g < nkb; ++g) cf[g] = 1.0f;  // the decoder leaves it to the caller
        }

        desc = R97IqrDecodeDesc{};
        desc.rows_all = rows;
        desc.cols = cols;
        desc.row0 = 0;
        desc.rows = rows;
        desc.bits = bits;
        desc.state_bits = state_bits;
        desc.run_rows = run_rows;
        desc.rb = rb;
        desc.mad = mad;
        desc.round_mad = mad || rowf;
        desc.mad_a = A;
        desc.mad_b = B;
        desc.lut = (const int8_t*)plane[(size_t)it].data();
        desc.codes = plane[(size_t)ic].data();
        desc.codes_row_bytes = rad_enc_row_bytes(enc.plane[ic].dtype, cols);
        desc.scale = plane[(size_t)is].data();
        desc.scale_dtype = scale_dtype;
        desc.scale_block_cols = rowf ? 0 : 128;
        desc.scale_row_bytes = rad_enc_row_bytes(enc.plane[is].dtype, rowf ? 1 : nkb);
        if (second) {
            desc.scale2 = plane[(size_t)i2].data();
            desc.scale2_dtype = R97_IQR_F32;
            desc.scale2_row_bytes = 4;
        }
    }
};

// The decode's f32 values, run by run and column by column, the way the kernel's column loop runs
// them. `values` is [row0, row0+rows) x cols, the shape libquant writes.
void decode_values(const R97IqrDecodeDesc& d, std::vector<float>* values,
                   std::vector<int8_t>* codes, std::vector<uint16_t>* scales) {
    values->assign((size_t)(d.rows * d.cols), 0.0f);
    codes->assign((size_t)(d.rows * d.cols), 0);
    scales->assign((size_t)(d.rows * (d.cols / 128)), 0);
    for (int64_t run = d.row0; run < d.row0 + d.rows; run += d.run_rows) {
        int64_t end = std::min(run + d.run_rows, d.rows_all);
        end = std::min(end, d.row0 + d.rows);
        const int32_t c = (d.state_bits + d.bits - 1) / d.bits;
        const uint32_t mask = (1u << d.state_bits) - 1u;
        for (int64_t j = 0; j < d.cols; ++j) {
            uint32_t st = 0;
            for (int32_t q = 0; q < c; ++q) {
                const int64_t t = end - c + q;
                st = ((st << d.bits) |
                      r97_iqr_load_bits(d.codes + (t - d.row0) * d.codes_row_bytes, d.bits, j)) & mask;
            }
            const int64_t kb = j >> 7;
            float cur = 0.0f;
            for (int64_t r = run; r < end; ++r) {
                st = ((st << d.bits) |
                      r97_iqr_load_bits(d.codes + (r - d.row0) * d.codes_row_bytes, d.bits, j)) & mask;
                float sc = 0.0f;
                const int32_t code = r97_iqr_value(d, st, r, kb, &cur, &sc);
                (*values)[(size_t)((r - d.row0) * d.cols + j)] = (float)code * sc;
                (*codes)[(size_t)((r - d.row0) * d.cols + j)] = (int8_t)code;
                if ((j & 127) == 0)
                    (*scales)[(size_t)((r - d.row0) * (d.cols / 128) + kb)] =
                        r97_iqr_f32_to_bf16(sc);
            }
        }
    }
}

void expect(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

void one_case(const char* name, int bits, int L, int rb, int mad, int scale_dtype, bool rowf,
              bool second, int64_t cols, bool partial) {
    Synth s;
    s.bits = bits;
    s.state_bits = L;
    s.run_rows = 256;
    s.rb = rb;
    s.mad = mad;
    s.scale_dtype = scale_dtype;
    s.rowf = rowf;
    s.second = second;
    s.partial = partial;
    s.cols = cols;
    s.rows = 3 * s.run_rows + (partial ? s.run_rows / 2 : 0);
    s.build();

    const std::string nm = std::string("decode ") + name + " codes=u" + std::to_string(bits) +
                           " L=" + std::to_string(L) + " rb=" + std::to_string(rb) +
                           " " + (mad ? "mad" : "lut") + " " +
                           (rowf ? "f32-row" : scale_dtype == R97_IQR_BF16 ? "bf16-128" :
                                   scale_dtype == R97_IQR_F32 ? "f32-128" : "e4m3") +
                           (second ? "+s2" : "") + " cols=" + std::to_string(cols) +
                           (partial ? " short-last" : "");

    std::vector<int8_t> codes;
    std::vector<uint16_t> scales;
    std::vector<float> values;
    decode_values(s.desc, &values, &codes, &scales);

    std::vector<int8_t> hcodes((size_t)(s.rows * s.cols), 0);
    std::vector<uint16_t> hscales((size_t)(s.rows * (s.cols / 128)), 0);
    R97IqrDecodeDesc hd = s.desc;
    hd.out_codes = hcodes.data();
    hd.out_scale = hscales.data();
    expect(r97_iqr_validate(hd) == 0, nm + ": the description is valid");
    expect(r97_iqr_decode_host(hd) == 0, nm + ": the host path returns");
    expect(hcodes == codes, nm + ": the host path's codes are the column loop's");
    expect(hscales == scales, nm + ": the host path's scales are the column loop's");

    std::vector<float> ref((size_t)(s.rows * s.cols), 0.0f);
    const int rc = g_lq(&s.enc, s.rows, s.cols, s.ptr.data(), 0, s.rows, ref.data());
    expect(rc == 0, nm + ": libquant decodes it, rc=" + std::to_string(rc));
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        // A zero is a zero: rb rounding turns a negative-zero value into the int8 code 0,
        // and the code times a positive block scale cannot carry the sign back.
        if (ref[i] == 0.0f && values[i] == 0.0f) continue;
        if (std::memcmp(&ref[i], &values[i], sizeof(float)) != 0) {
            if (!bad) first = i;
            ++bad;
        }
    }
    expect(bad == 0, nm + ": the values are libquant's bits (" + std::to_string(bad) +
                         " differ, first at " + std::to_string(first) + ": libquant " +
                         std::to_string(ref[first]) + ", ours " + std::to_string(values[first]) +
                         ")");

    if (s.rows > s.run_rows) {
        // A chunk that starts at a run boundary: rows [run_rows, 2*run_rows).
        R97IqrDecodeDesc cd = s.desc;
        cd.row0 = s.run_rows;
        cd.rows = s.run_rows;
        cd.codes = s.desc.codes + cd.row0 * cd.codes_row_bytes;
        cd.scale = s.desc.scale + cd.row0 * cd.scale_row_bytes;
        if (s.desc.scale2 && s.desc.scale2_block_rows > 0)
            cd.scale2 = s.desc.scale2 + (cd.row0 / s.desc.scale2_block_rows) * cd.scale2_row_bytes;
        std::vector<int8_t> cc((size_t)(cd.rows * cd.cols), 0);
        std::vector<uint16_t> cs((size_t)(cd.rows * (cd.cols / 128)), 0);
        cd.out_codes = cc.data();
        cd.out_scale = cs.data();
        expect(r97_iqr_decode_host(cd) == 0, nm + ": the chunk decodes");
        expect(std::memcmp(cc.data(), codes.data() + s.run_rows * s.cols, cc.size()) == 0,
               nm + ": the chunk's codes are the whole's");
        expect(std::memcmp(cs.data(), scales.data() + s.run_rows * (s.cols / 128),
                           cs.size() * sizeof(uint16_t)) == 0,
               nm + ": the chunk's scales are the whole's");
    }

    // A chunk at the very end, with the short run, and a chunk that is not a multiple of a run
    // before the end must be refused.
    if (partial) {
        R97IqrDecodeDesc bad = s.desc;
        bad.rows = s.rows - 100;  // ends inside the last (short) run, before the tensor's end
        expect(r97_iqr_validate(bad) != 0, nm + ": a chunk ending inside a short run is refused");
    }
}

// The affine grid, which the container's drafter head stores: unsigned codes under an integer zero
// a block, no trellis above them and no codebook. libquant's reference is grid_decode, one value at
// a time: (code - zero[r, block]) * scale.
void one_affine_case(const char* name, int bits, int zero_bits, int scale_dtype, int64_t cols) {
    const int64_t rows = 300;
    const int64_t nkb = cols / 128;
    const int64_t crow = (cols * bits + 7) / 8;
    const int64_t zrow = (nkb * zero_bits + 7) / 8;
    const int64_t srow = nkb * 2;   // f16 or bf16
    const uint32_t sdt = scale_dtype == R97_IQR_F16 ? RAD_F16 : RAD_BF16;

    const std::string nm = std::string("decode affine ") + name + " codes=u" + std::to_string(bits) +
                           " zero=u" + std::to_string(zero_bits) + " " +
                           (scale_dtype == R97_IQR_F16 ? "f16-128" : "bf16-128") +
                           " cols=" + std::to_string(cols);

    std::vector<uint8_t> codes((size_t)(rows * crow));
    std::vector<uint8_t> zero((size_t)(rows * zrow));
    std::vector<uint8_t> scale((size_t)(rows * srow));
    uint32_t seed = 0x9E3779B9u;
    auto rnd = [&]() { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed; };
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t j = 0; j < cols; ++j)
            rad_store_bits(codes.data() + r * crow, bits, j, rnd() & ((1u << bits) - 1u));
        for (int64_t g = 0; g < nkb; ++g) {
            // A zero the code's own width, as a quantiser stores one: the centred code stays on
            // the int8 grid.
            rad_store_bits(zero.data() + r * zrow, zero_bits, g, rnd() % (1u << bits));
            rad_store_f32(scale.data() + r * srow, sdt, g, 0.0004f + (float)(rnd() % 997) * 2.0e-5f);
        }
    }

    R97IqrDecodeDesc d{};
    d.rows_all = rows;
    d.cols = cols;
    d.row0 = 0;
    d.rows = rows;
    d.affine = 1;
    d.bits = bits;
    d.run_rows = 1;
    d.rb = 1;
    d.codes = codes.data();
    d.codes_row_bytes = crow;
    d.zero = zero.data();
    d.zero_bits = zero_bits;
    d.zero_row_bytes = zrow;
    d.scale = scale.data();
    d.scale_dtype = scale_dtype;
    d.scale_block_cols = 128;
    d.scale_row_bytes = srow;

    std::vector<int8_t> hcodes((size_t)(rows * cols), 0);
    std::vector<uint16_t> hscales((size_t)(rows * nkb), 0);
    R97IqrDecodeDesc hd = d;
    hd.out_codes = hcodes.data();
    hd.out_scale = hscales.data();
    expect(r97_iqr_validate(hd) == 0, nm + ": the description is valid");
    expect(r97_iqr_decode_host(hd) == 0, nm + ": the host path returns");

    // The grid: the code less its block's zero, and one bf16 scale the 128 columns.
    std::vector<float> values((size_t)(rows * cols), 0.0f);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t j = 0; j < cols; ++j) {
            const int code = (int)rad_load_bits(codes.data() + r * crow, bits, j);
            const int zed = (int)rad_load_bits(zero.data() + r * zrow, zero_bits, j >> 7);
            const int8_t want = (int8_t)(code - zed);
            expect(hcodes[(size_t)(r * cols + j)] == want,
                   nm + ": the int8 grid is the code less the block's zero");
            values[(size_t)(r * cols + j)] = (float)want * r97_iqr_block_scale(d, r, j >> 7);
        }
        for (int64_t g = 0; g < nkb; ++g)
            expect(hscales[(size_t)(r * nkb + g)] ==
                       r97_iqr_f32_to_bf16(r97_iqr_block_scale(d, r, g)),
                   nm + ": the scale is the block's");
    }

    lq::Grid grid{};
    grid.asym = true;
    size_t bad = 0, first = 0;
    float first_ref = 0.0f;
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t j = 0; j < cols; ++j) {
            const int64_t kb = j >> 7;
            lq::BlockScale b;
            b.s = r97_iqr_block_scale(d, r, kb);
            b.zero = (int)rad_load_bits(zero.data() + r * zrow, zero_bits, kb);
            const float ref = g_grid(grid, b, (int)rad_load_bits(codes.data() + r * crow, bits, j));
            const size_t i = (size_t)(r * cols + j);
            if (std::memcmp(&ref, &values[i], sizeof(float)) != 0) {
                if (!bad) { first = i; first_ref = ref; }
                ++bad;
            }
        }
    }
    expect(bad == 0, nm + ": the values are libquant's bits (" + std::to_string(bad) +
                         " differ, first at " + std::to_string(first) + ": libquant " +
                         std::to_string(first_ref) + ", ours " + std::to_string(values[first]) +
                         ")");

    // A chunk of the rows, as a vocabulary-sized head decodes it.
    R97IqrDecodeDesc cd = d;
    cd.row0 = 128;
    cd.rows = 100;
    cd.codes = d.codes + cd.row0 * d.codes_row_bytes;
    cd.zero = d.zero + cd.row0 * d.zero_row_bytes;
    cd.scale = d.scale + cd.row0 * d.scale_row_bytes;
    std::vector<int8_t> cc((size_t)(cd.rows * cd.cols), 0);
    std::vector<uint16_t> cs((size_t)(cd.rows * nkb), 0);
    cd.out_codes = cc.data();
    cd.out_scale = cs.data();
    expect(r97_iqr_decode_host(cd) == 0, nm + ": the chunk decodes");
    expect(std::memcmp(cc.data(), hcodes.data() + 128 * cols, cc.size()) == 0,
           nm + ": the chunk's codes are the whole's");
    expect(std::memcmp(cs.data(), hscales.data() + 128 * nkb, cs.size() * sizeof(uint16_t)) == 0,
           nm + ": the chunk's scales are the whole's");

    // The zero is the affine form's own plane: without it there is nothing to decode.
    R97IqrDecodeDesc nz = d;
    nz.zero = nullptr;
    expect(r97_iqr_validate(nz) != 0, nm + ": an affine description without a zero is refused");
}

}  // namespace

R97_TEST(iqr_decode) {
    load_libquant();

    // Every form the container stores: computed and stored codebooks, the three scale layouts,
    // and the code widths the head's k3..k6 use.
    for (int bits : { 3, 4, 5, 6 }) {
        one_case("trunk", bits, 16, 1, 1, R97_IQR_F32, true, false, 256, false);
        one_case("stored", bits, 16, 1, 0, R97_IQR_BF16, false, false, 256, false);
    }
    one_case("embedding", 4, 12, 1, 0, R97_IQR_F8E4M3, false, true, 256, false);
    one_case("f32-128", 4, 16, 1, 1, R97_IQR_F32, false, false, 512, false);
    one_case("e4m3-128", 5, 12, 1, 0, R97_IQR_F8E4M3, false, true, 128, false);
    one_case("mad-rb", 4, 16, 8, 1, R97_IQR_F32, true, false, 256, false);
    one_case("lut-rb", 5, 16, 16, 0, R97_IQR_BF16, false, false, 256, false);
    one_case("rb-f32-row", 4, 16, 8, 0, R97_IQR_F32, true, false, 256, false);
    one_case("short-last-mad", 4, 16, 1, 1, R97_IQR_F32, true, false, 256, true);
    one_case("short-last-embedding", 4, 12, 1, 0, R97_IQR_F8E4M3, false, true, 256, true);

    // A column count that leaves a partial workgroup on the card (5 * 128 = 640, not a multiple
    // of the kernel's 256 threads).
    one_case("partial-workgroup", 4, 16, 1, 1, R97_IQR_F32, true, false, 640, false);

    // The affine grid of the drafter head (u2 codes, a u8 zero and an f16 scale the 128 columns),
    // a narrower zero plane, and a bf16 scale.
    one_affine_case("drafter", 2, 8, R97_IQR_F16, 512);
    one_affine_case("drafter-narrow-zero", 2, 4, R97_IQR_F16, 384);
    one_affine_case("drafter-bf16", 4, 8, R97_IQR_BF16, 256);
}
