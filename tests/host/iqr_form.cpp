// tests/host/iqr_form.cpp -- the stored form a host row derives from an encoding.
//
// src/iqr_form.h is what lets a host row differentiate a trunk weight the tape stores in a
// plugin-private encoding: the join reads the container's canonical planes and turns the entry's
// RadEncoding into the description r97_iqr_decode_host already decodes (tests/host/iqr_decode.cpp
// holds that decode to libquant bit for bit). What this test pins is the derivation -- which plane
// is what, the trellis's state count and run, the scale's layout, and the one factor the
// description cannot carry -- against the encodings the real container writes, built here by hand
// so the test needs no container.
#include "iqr_form.h"

#include "rad_encoding.h"
#include "rad_types.h"
#include "support/test.h"

#include <cstring>
#include <vector>

namespace {
namespace r97_test {

void expect(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

// One plane's bytes, built in memory.
struct Plane {
    std::vector<uint8_t> bytes;
    int64_t bytes_per_row = 0;
};

int32_t plane_index(const RadEncoding& e, const char* role) {
    for (int32_t i = 0; i < e.n_planes; ++i)
        if (std::strcmp(e.plane[i].role, role) == 0) return i;
    return -1;
}

// The encodings this test builds, and the planes they point at. `stored` is the array
// r97_iqr_desc_from_planes reads; it must stay alive for the decode.
struct Built {
    RadEncoding enc;
    std::vector<Plane> planes;
    std::vector<r97::R97IqrStoredPlane> stored;

    void seal() {
        stored.assign(planes.size(), r97::R97IqrStoredPlane{});
        for (size_t i = 0; i < planes.size(); ++i) {
            stored[i].data = planes[i].bytes.empty() ? nullptr : planes[i].bytes.data();
            stored[i].bytes = static_cast<int64_t>(planes[i].bytes.size());
        }
    }
};

// The trunk's form, exactly as the container writes it (rad-info's spelling):
//   trellis:codes=u6[1x1],scale=f32[1x*],trellis=i8{65536x1},trellis.mad=u32{2x1},
//   scale.col=f32[*x128]/fwht128
// The trellis is a TABLE that carries its run in block[0] (256 rows) -- not a plane the affine
// helpers build, which is why this test builds it plane by plane.
Built trunk_form(int64_t rows, int64_t cols, bool with_mad, bool with_scale_col) {
    const int64_t nkb = cols / 128;
    Built b;
    rad_enc_clear(&b.enc);
    rad_enc_copy_str(b.enc.scheme, "trellis");
    rad_enc_copy_str(b.enc.transform, "fwht128");
    rad_enc_add_plane(&b.enc, "codes", RAD_U6, 1, 1);
    rad_enc_add_plane(&b.enc, "scale", RAD_F32, 1, 0);
    const int it = rad_enc_add_table(&b.enc, "trellis", RAD_I8, 65536, 1);
    b.enc.plane[it].block[0] = 256;
    b.enc.plane[it].block[1] = 1;
    if (with_mad) rad_enc_add_table(&b.enc, "trellis.mad", RAD_U32, 2, 1);
    if (with_scale_col) rad_enc_add_plane(&b.enc, "scale.col", RAD_F32, 0, 128);

    b.planes.resize(static_cast<size_t>(b.enc.n_planes));
    for (int32_t i = 0; i < b.enc.n_planes; ++i) {
        const RadEncPlane& p = b.enc.plane[i];
        int64_t prows = 0, pcols = 0;
        rad_enc_plane_dims(&p, rows, cols, &prows, &pcols);
        b.planes[static_cast<size_t>(i)].bytes_per_row = rad_enc_row_bytes(p.dtype, pcols);
        b.planes[static_cast<size_t>(i)].bytes.assign(
            static_cast<size_t>(prows * b.planes[static_cast<size_t>(i)].bytes_per_row), 0);
    }
    // The states: an int8 a state, a value no decode can produce by accident.
    Plane& lut = b.planes[static_cast<size_t>(it)];
    for (size_t i = 0; i < lut.bytes.size(); ++i) lut.bytes[i] = static_cast<uint8_t>(i * 37 + 11);
    // A, B for the computed codebook.
    if (with_mad) {
        const int im = plane_index(b.enc, "trellis.mad");
        const uint32_t ab[2] = {0x00ABCDEFu, 0x00000007u};
        std::memcpy(b.planes[static_cast<size_t>(im)].bytes.data(), ab, sizeof ab);
    }
    // The codes: u6 packed, every code distinct enough that a wrong bit offset shows.
    Plane& codes = b.planes[0];
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t j = 0; j < cols; ++j) {
            const uint32_t v = static_cast<uint32_t>((r * 13 + j * 7) % 64);
            const int64_t o = j * 6;
            uint8_t* row = codes.bytes.data() + r * codes.bytes_per_row;
            row[o >> 3] |= static_cast<uint8_t>(v << (o & 7));
            if ((o & 7) + 6 > 8) row[(o >> 3) + 1] |= static_cast<uint8_t>(v >> (8 - (o & 7)));
        }
    // The scale: one f32 a row, distinct so the fold's operand order is visible.
    const int is = plane_index(b.enc, "scale");
    float* srow = reinterpret_cast<float*>(b.planes[static_cast<size_t>(is)].bytes.data());
    for (int64_t r = 0; r < rows; ++r) srow[r] = 0.0001f * static_cast<float>(r + 1);
    // The per-128 factor: two different values, so a fold that ignores it is not a rounding away.
    if (with_scale_col) {
        const int icf = plane_index(b.enc, "scale.col");
        float* cf = reinterpret_cast<float*>(b.planes[static_cast<size_t>(icf)].bytes.data());
        for (int64_t kb = 0; kb < nkb; ++kb) cf[kb] = 2.0f + static_cast<float>(kb);
    }
    b.seal();
    return b;
}

// The stored form, as the container writes blk.63.attn_output.weight:
//   trellis:codes=u6[1x1],scale=bf16[1x128],trellis=i8{65536x1},trellis.mad=u32{2x1}
Built bf16_form(int64_t rows, int64_t cols) {
    const int64_t nkb = cols / 128;
    Built b;
    rad_enc_clear(&b.enc);
    rad_enc_copy_str(b.enc.scheme, "trellis");
    rad_enc_copy_str(b.enc.transform, "fwht128");
    rad_enc_add_plane(&b.enc, "codes", RAD_U6, 1, 1);
    rad_enc_add_plane(&b.enc, "scale", RAD_BF16, 1, 128);
    const int it = rad_enc_add_table(&b.enc, "trellis", RAD_I8, 65536, 1);
    b.enc.plane[it].block[0] = 256;
    b.enc.plane[it].block[1] = 1;
    rad_enc_add_table(&b.enc, "trellis.mad", RAD_U32, 2, 1);
    b.planes.resize(static_cast<size_t>(b.enc.n_planes));
    for (int32_t i = 0; i < b.enc.n_planes; ++i) {
        const RadEncPlane& p = b.enc.plane[i];
        int64_t prows = 0, pcols = 0;
        rad_enc_plane_dims(&p, rows, cols, &prows, &pcols);
        b.planes[static_cast<size_t>(i)].bytes.assign(
            static_cast<size_t>(prows * rad_enc_row_bytes(p.dtype, pcols)), 0);
    }
    for (size_t i = 0; i < b.planes[static_cast<size_t>(it)].bytes.size(); ++i)
        b.planes[static_cast<size_t>(it)].bytes[i] = static_cast<uint8_t>(i * 29 + 5);
    const int is = plane_index(b.enc, "scale");
    uint16_t* sc = reinterpret_cast<uint16_t*>(b.planes[static_cast<size_t>(is)].bytes.data());
    for (int64_t i = 0; i < rows * nkb; ++i) sc[i] = r97_iqr_f32_to_bf16(0.5f + 0.25f * i);
    b.seal();
    return b;
}

// The drafter head's affine grid: u4 codes, an integer zero a 128 columns, a bf16 scale, no
// trellis and no codebook.
Built affine_form(int64_t rows, int64_t cols) {
    const int64_t nkb = cols / 128;
    Built b;
    rad_enc_clear(&b.enc);
    rad_enc_copy_str(b.enc.scheme, "affine");
    rad_enc_add_plane(&b.enc, "codes", RAD_U4, 1, 1);
    rad_enc_add_plane(&b.enc, "scale", RAD_BF16, 1, 128);
    rad_enc_add_plane(&b.enc, "zero", RAD_U8, 1, 128);
    b.planes.resize(static_cast<size_t>(b.enc.n_planes));
    for (int32_t i = 0; i < b.enc.n_planes; ++i) {
        const RadEncPlane& p = b.enc.plane[i];
        int64_t prows = 0, pcols = 0;
        rad_enc_plane_dims(&p, rows, cols, &prows, &pcols);
        b.planes[static_cast<size_t>(i)].bytes_per_row = rad_enc_row_bytes(p.dtype, pcols);
        b.planes[static_cast<size_t>(i)].bytes.assign(
            static_cast<size_t>(prows * b.planes[static_cast<size_t>(i)].bytes_per_row), 0);
    }
    Plane& codes = b.planes[0];
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t j = 0; j < cols; ++j) {
            const uint8_t v = static_cast<uint8_t>((r * 5 + j * 3) & 15);
            uint8_t* row = codes.bytes.data() + r * codes.bytes_per_row;
            row[j >> 1] |= static_cast<uint8_t>(v << ((j & 1) * 4));
        }
    const int iz = plane_index(b.enc, "zero");
    Plane& zero = b.planes[static_cast<size_t>(iz)];
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t kb = 0; kb < nkb; ++kb)
            zero.bytes[static_cast<size_t>(r * zero.bytes_per_row + kb)] =
                static_cast<uint8_t>(kb + 1);
    const int is = plane_index(b.enc, "scale");
    uint16_t* sc = reinterpret_cast<uint16_t*>(b.planes[static_cast<size_t>(is)].bytes.data());
    for (int64_t i = 0; i < rows * nkb; ++i) sc[i] = r97_iqr_f32_to_bf16(0.125f * (i + 1));
    b.seal();
    return b;
}

void check_trunk() {
    const int64_t rows = 512, cols = 256;  // two runs, two 128-column blocks
    const int64_t nkb = cols / 128;
    Built b = trunk_form(rows, cols, true, true);
    R97IqrDecodeDesc d{};
    expect(r97::r97_iqr_desc_from_planes(&b.enc, b.stored.data(), rows, cols, &d),
           "the trunk form is a description");
    expect(d.bits == 6, "the trunk's codes are six bits");
    expect(d.state_bits == 16, "the trellis holds 2^16 states");
    expect(d.run_rows == 256, "the trellis covers 256 rows a run");
    expect(d.rb == 1, "one scale a row");
    expect(d.mad == 1, "the codebook is computed");
    expect(d.round_mad == 1, "the computed codebook rounds its codes");
    expect(d.scale_dtype == R97_IQR_F32, "the trunk's scale is f32");
    expect(d.scale_block_cols == 0, "the trunk's scale is one value a row");
    expect(d.scale2 == nullptr, "the trunk has no second level");
    expect(d.affine == 0, "the trunk is not an affine grid");

    r97::R97IqrDecoded dec;
    expect(r97::r97_iqr_decode_planes(&b.enc, b.stored.data(), rows, cols, &dec),
           "the trunk form decodes");
    expect(dec.codes.size() == static_cast<size_t>(rows * cols), "the codes cover the weight");
    expect(dec.scale.size() == static_cast<size_t>(rows * nkb), "one scale a 128 columns");

    // The fold the description cannot carry: value = bf16(row f32 * the 128-column factor), one
    // rounding. The two factors are 2 and 3, so a fold that dropped the factor, applied it twice,
    // or took it from the wrong row would differ.
    const float* srow = reinterpret_cast<const float*>(
        b.planes[static_cast<size_t>(plane_index(b.enc, "scale"))].bytes.data());
    const float* cf = reinterpret_cast<const float*>(
        b.planes[static_cast<size_t>(plane_index(b.enc, "scale.col"))].bytes.data());
    int64_t bad = 0;
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t kb = 0; kb < nkb; ++kb) {
            const uint16_t want = r97_iqr_f32_to_bf16(srow[r] * cf[kb]);
            if (dec.scale[static_cast<size_t>(r * nkb + kb)] != want) ++bad;
        }
    expect(bad == 0, "every stored scale is the row's factor times the block's (" +
                         std::to_string(bad) + " differ)");
    // And the factor is not a no-op at these values: bf16(row) alone is a different bit pattern.
    expect(dec.scale[0] != r97_iqr_f32_to_bf16(srow[0]),
           "the block's factor changes the stored scale");

    // A computed codebook with no scale.col still rounds: round_mad follows mad alone.
    Built no_cf = trunk_form(rows, cols, true, false);
    R97IqrDecodeDesc d2{};
    expect(r97::r97_iqr_desc_from_planes(&no_cf.enc, no_cf.stored.data(), rows, cols, &d2),
           "the trunk form without scale.col is a description");
    expect(d2.round_mad == 1, "a computed codebook rounds without scale.col too");
    // A stored codebook beside scale.col rounds as the computed one does: libquant's mad form.
    Built lut_cf = trunk_form(rows, cols, false, true);
    R97IqrDecodeDesc d3{};
    expect(r97::r97_iqr_desc_from_planes(&lut_cf.enc, lut_cf.stored.data(), rows, cols, &d3),
           "a stored codebook with scale.col is a description");
    expect(d3.mad == 0 && d3.round_mad == 1, "scale.col makes a stored codebook round");
    expect(d3.lut != nullptr, "the stored codebook is the trellis table");
}

void check_bf16() {
    const int64_t rows = 512, cols = 256;
    const int64_t nkb = cols / 128;
    Built b = bf16_form(rows, cols);
    R97IqrDecodeDesc d{};
    expect(r97::r97_iqr_desc_from_planes(&b.enc, b.stored.data(), rows, cols, &d),
           "the bf16 form is a description");
    expect(d.scale_dtype == R97_IQR_BF16, "the stored form's scale is bf16");
    expect(d.scale_block_cols == 128, "one scale a 128 columns");
    expect(d.scale_row_bytes == nkb * 2, "a scale row is one bf16 a 128 columns");
    expect(d.affine == 0, "the stored form is not an affine grid");
    expect(d.scale2 == nullptr, "the stored form has no second level");

    r97::R97IqrDecoded dec;
    expect(r97::r97_iqr_decode_planes(&b.enc, b.stored.data(), rows, cols, &dec),
           "the bf16 form decodes");
    // No scale.col, so the stored scale is the plane's own bits.
    const uint16_t* sc = reinterpret_cast<const uint16_t*>(
        b.planes[static_cast<size_t>(plane_index(b.enc, "scale"))].bytes.data());
    expect(std::memcmp(dec.scale.data(), sc, dec.scale.size() * sizeof(uint16_t)) == 0,
           "the stored form's scales are the plane's own");
    for (int8_t c : dec.codes) {
        expect(c >= -127 && c <= 127, "a decoded code is on the int8 grid");
        if (c != 0) break;
    }
}

void check_affine() {
    const int64_t rows = 256, cols = 256;
    const int64_t nkb = cols / 128;
    Built b = affine_form(rows, cols);
    R97IqrDecodeDesc d{};
    expect(r97::r97_iqr_desc_from_planes(&b.enc, b.stored.data(), rows, cols, &d),
           "the affine grid is a description");
    expect(d.affine == 1, "the affine grid decodes as one");
    expect(d.bits == 4, "its codes are four bits");
    expect(d.zero_bits == 8, "its zero is a byte");
    expect(d.zero_row_bytes == nkb, "a zero row is one byte a 128 columns");
    expect(d.run_rows == 1 && d.rb == 1, "an affine row stands alone");
    expect(d.scale2 == nullptr, "the affine grid has no second level");

    r97::R97IqrDecoded dec;
    expect(r97::r97_iqr_decode_planes(&b.enc, b.stored.data(), rows, cols, &dec),
           "the affine grid decodes");
    const Plane& cp = b.planes[0];
    const Plane& zp = b.planes[static_cast<size_t>(plane_index(b.enc, "zero"))];
    int64_t bad = 0;
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t j = 0; j < cols; ++j) {
            const uint32_t code =
                r97_iqr_load_bits(cp.bytes.data() + r * cp.bytes_per_row, 4, j);
            const uint32_t zero =
                r97_iqr_load_bits(zp.bytes.data() + r * zp.bytes_per_row, 8, j / 128);
            if (dec.codes[static_cast<size_t>(r * cols + j)] !=
                static_cast<int8_t>(static_cast<int32_t>(code) - static_cast<int32_t>(zero)))
                ++bad;
        }
    expect(bad == 0, "every affine code is its stored code less the block's zero (" +
                         std::to_string(bad) + " differ)");
}

void check_refusals() {
    const int64_t rows = 256, cols = 256;
    // A plane the decode needs, missing: the trellis is what makes the state a code.
    Built no_trellis = trunk_form(rows, cols, true, true);
    no_trellis.enc.n_planes = 2;  // codes and scale only
    R97IqrDecodeDesc d{};
    expect(!r97::r97_iqr_desc_from_planes(&no_trellis.enc, no_trellis.stored.data(), rows, cols,
                                          &d),
           "an encoding with no trellis and no zero is refused");
    // A trellis whose state count is not a power of two.
    Built odd = trunk_form(rows, cols, true, true);
    odd.enc.plane[plane_index(odd.enc, "trellis")].extent[0] = 1000;
    expect(!r97::r97_iqr_desc_from_planes(&odd.enc, odd.stored.data(), rows, cols, &d),
           "a trellis of 1000 states is refused");
    // The per-128 factor beside a scale that is already one a 128 columns: the two layouts
    // contradict, and neither the device row nor this one has a form for the pair. The description
    // is derivable -- it cannot know what the factor means -- so the refusal is the decode's.
    Built both = bf16_form(rows, cols);
    rad_enc_add_plane(&both.enc, "scale.col", RAD_F32, 0, 128);
    both.planes.resize(static_cast<size_t>(both.enc.n_planes));
    both.planes.back().bytes.assign(static_cast<size_t>(cols / 128 * 4), 0);
    both.seal();
    r97::R97IqrDecoded both_dec;
    expect(!r97::r97_iqr_decode_planes(&both.enc, both.stored.data(), rows, cols, &both_dec),
           "a bf16 block scale beside scale.col is refused");
    // A plane shorter than the weight: the bytes are the caller's, and a short one is a wrong read.
    Built short_codes = trunk_form(rows, cols, true, true);
    short_codes.stored[0].bytes /= 2;
    expect(!r97::r97_iqr_desc_from_planes(&short_codes.enc, short_codes.stored.data(), rows, cols,
                                          &d),
           "a short codes plane is refused");
    // A column count the decode does not tile.
    Built ragged = trunk_form(rows, cols, true, true);
    expect(!r97::r97_iqr_desc_from_planes(&ragged.enc, ragged.stored.data(), rows, 250, &d),
           "250 columns is refused");
    // No planes at all.
    expect(!r97::r97_iqr_desc_from_planes(nullptr, ragged.stored.data(), rows, cols, &d),
           "no encoding is refused");
}

}  // namespace r97_test
}  // namespace

R97_TEST(iqr_form) {
    r97_test::check_trunk();
    r97_test::check_bf16();
    r97_test::check_affine();
    r97_test::check_refusals();
}
