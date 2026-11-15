// src/iqr_form.h -- a stored IQR weight's encoding, read as the decode's own description.
//
// The tape hands a plugin-private weight as one pointer into the engine's VRAM, in the form
// libr4d's layout hook rearranged it into (unit-major lane segments, the scale tile-major:
// r4d_iqr.h). That is the device row's to read, and a host row cannot: the pointer is not a host
// address and the layout is not the canonical one.
//
// The canonical planes are still in the container, and that is what a host row reads. This header
// is the one place that turns an encoding plus those planes into an R97IqrDecodeDesc -- the
// description r97_iqr_decode_host already decodes and tests/host/iqr_decode.cpp already holds to
// libquant bit for bit -- so the trainer carries no second trellis and no second container reader.
//
// What the description cannot say is the scale.col factor: a trellis under one f32 a row times one
// f32 a 128-K block (libr4d's kRowF32) keeps the block's factor out of the weight's side, and the
// forward folds it into the activation's own per-128 scale. A GEMM's adjoint needs the weight's
// value, so r97_iqr_decode_planes applies that factor itself, to the same bf16 the device row's
// scale plane holds (iqr_plane.hip's eff_scale: bf16(row * factor), one rounding).
//
// The form is not named by the op or the container: it is derived from the encoding's planes,
// exactly as libr4d's classify derives it (r4d_iqr_layout.cpp:83-158), and anything this file
// cannot derive is refused rather than decoded as some other shape.
#pragma once

#include "iqr_decode.h"

#include "rad_encoding.h"
#include "rad_types.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace r97 {

// One of the encoding's planes as the container holds it: the bytes and their length. A plane the
// caller does not hold is `{nullptr, 0}`.
struct R97IqrStoredPlane {
    const uint8_t* data = nullptr;
    int64_t bytes = 0;
};

namespace iqr_detail {

// The width of an unsigned packed code plane, or 0 for a dtype the IQR forms do not store.
inline int code_bits(uint32_t dt) {
    switch (dt) {
        case RAD_U1: case RAD_U2: case RAD_U3: case RAD_U4: case RAD_U5: case RAD_U6:
        case RAD_U7: case RAD_U8: case RAD_U9: case RAD_U10: case RAD_U11: case RAD_U12:
            return rad_dtype_bits(dt);
        default:
            return 0;
    }
}

inline int32_t log2_exact(int64_t n) {
    int32_t l = 0;
    while ((int64_t(1) << l) < n) ++l;
    return (int64_t(1) << l) == n ? l : -1;
}

// The decode's scale element type for a plane's dtype, or -1.
inline int32_t scale_type(uint32_t dt) {
    switch (dt) {
        case RAD_F32:    return R97_IQR_F32;
        case RAD_BF16:   return R97_IQR_BF16;
        case RAD_F8E4M3: return R97_IQR_F8E4M3;
        case RAD_F16:    return R97_IQR_F16;
        default:         return -1;
    }
}

inline int plane_of(const RadEncoding& enc, const char* role) {
    for (int i = 0; i < enc.n_planes; ++i)
        if (std::strcmp(enc.plane[i].role, role) == 0) return i;
    return -1;
}

inline bool held(const R97IqrStoredPlane* plane, int i, int64_t need) {
    return i >= 0 && plane[i].data != nullptr && plane[i].bytes >= need;
}

}  // namespace iqr_detail

// The description of the whole weight [rows, cols] from `enc`'s stored planes: `plane[i]` is
// encoding plane i's bytes, and the output pointers are left to the caller. False when the encoding
// is not a form this decode reads, or a plane the decode needs is missing or short.
inline bool r97_iqr_desc_from_planes(const RadEncoding* enc, const R97IqrStoredPlane* plane,
                                     int64_t rows, int64_t cols, R97IqrDecodeDesc* out) {
    using namespace iqr_detail;
    if (enc == nullptr || plane == nullptr || out == nullptr || rows <= 0 || cols <= 0) return false;
    if ((cols & 127) != 0) return false;
    const int64_t nkb = cols / 128;
    const int ic = plane_of(*enc, "codes");
    const int is = plane_of(*enc, "scale");
    const int it = plane_of(*enc, "trellis");
    const int iz = plane_of(*enc, "zero");
    if (ic < 0 || is < 0 || (it < 0 && iz < 0)) return false;

    R97IqrDecodeDesc d{};
    d.rows_all = rows;
    d.cols = cols;
    d.row0 = 0;
    d.rows = rows;
    d.bits = code_bits(enc->plane[ic].dtype);
    if (d.bits <= 0) return false;
    d.codes_row_bytes = rad_enc_row_bytes(enc->plane[ic].dtype, cols);
    if (!held(plane, ic, rows * d.codes_row_bytes)) return false;

    if (iz >= 0) {
        // The affine grid: an integer zero a block beside the scale and no trellis above it, so a
        // row is a run of its own.
        if (it >= 0) return false;
        d.zero_bits = rad_dtype_bits(enc->plane[iz].dtype);
        if (d.zero_bits < 1 || d.zero_bits > 8) return false;
        if (enc->plane[iz].block[0] != enc->plane[is].block[0] ||
            enc->plane[iz].block[1] != enc->plane[is].block[1])
            return false;
        d.affine = 1;
        d.run_rows = 1;
        d.rb = 1;
        d.zero_row_bytes =
            rad_enc_row_bytes(enc->plane[iz].dtype, enc->plane[iz].block[1] > 0 ? nkb : 1);
        if (!held(plane, iz, rows * d.zero_row_bytes)) return false;
        d.zero = plane[iz].data;
    } else {
        d.state_bits = log2_exact(enc->plane[it].extent[0]);
        if (d.state_bits < 1) return false;
        d.run_rows = static_cast<int32_t>(enc->plane[it].block[0]);
        d.rb = enc->plane[it].block[1] > 0 ? static_cast<int32_t>(enc->plane[it].block[1]) : 1;
        if (d.run_rows < 1 || d.rb < 1) return false;
    }

    d.scale_dtype = scale_type(enc->plane[is].dtype);
    if (d.scale_dtype < 0) return false;
    d.scale_block_cols = enc->plane[is].block[1];
    d.scale_row_bytes =
        rad_enc_row_bytes(enc->plane[is].dtype, d.scale_block_cols > 0 ? nkb : 1);
    if (!held(plane, is, rows * d.scale_row_bytes)) return false;

    const int is2 = plane_of(*enc, "scale.1");
    if (is2 >= 0) {
        if (iz >= 0) return false;
        d.scale2_dtype = scale_type(enc->plane[is2].dtype);
        if (d.scale2_dtype < 0) return false;
        d.scale2_block_rows = enc->plane[is2].block[0];
        d.scale2_block_cols = enc->plane[is2].block[1];
        d.scale2_row_bytes =
            rad_enc_row_bytes(enc->plane[is2].dtype, d.scale2_block_cols > 0 ? nkb : 1);
        const int64_t rows2 = d.scale2_block_rows > 0 ? (rows + d.scale2_block_rows - 1) /
                                                           d.scale2_block_rows
                                                     : 1;
        if (!held(plane, is2, rows2 * d.scale2_row_bytes)) return false;
        d.scale2 = plane[is2].data;
    }

    const int im = plane_of(*enc, "trellis.mad");
    d.mad = im >= 0;
    d.round_mad = d.mad || plane_of(*enc, "scale.col") >= 0;
    if (it >= 0) {
        // The state values, one int8 a state, or the computed code's A and B.
        const int64_t states = enc->plane[it].extent[0];
        if (!held(plane, it, states)) return false;
        d.lut = reinterpret_cast<const int8_t*>(plane[it].data);
        if (d.mad) {
            if (!held(plane, im, 8)) return false;
            const uint32_t* ab = reinterpret_cast<const uint32_t*>(plane[im].data);
            d.mad_a = ab[0];
            d.mad_b = ab[1];
        }
    }
    if (iz >= 0 && !held(plane, iz, rows * d.zero_row_bytes)) return false;

    d.codes = plane[ic].data;
    d.scale = plane[is].data;
    *out = d;
    return true;
}

// The decoded weight in the two grids a `gemm_nt_q` reads: `codes` is [rows, cols] int8 row-major
// and `scale` is [rows, cols/128] bf16 bits, so a value of the weight is codes[i] * scale[i/cols,
// (i%cols)/128] -- the pair the device row's dx kernel reads from its own planes.
struct R97IqrDecoded {
    std::vector<int8_t> codes;
    std::vector<uint16_t> scale;
};

// Decode the whole weight [rows, cols] from `enc`'s stored planes. The scale.col factor, which the
// description cannot carry, is folded here into the same bf16 the device row's scale plane holds.
inline bool r97_iqr_decode_planes(const RadEncoding* enc, const R97IqrStoredPlane* plane,
                                  int64_t rows, int64_t cols, R97IqrDecoded* out) {
    using namespace iqr_detail;
    if (out == nullptr) return false;
    R97IqrDecodeDesc d{};
    if (!r97_iqr_desc_from_planes(enc, plane, rows, cols, &d)) return false;
    const int64_t nkb = cols / 128;
    out->codes.assign(static_cast<size_t>(rows * cols), 0);
    out->scale.assign(static_cast<size_t>(rows * nkb), 0);
    d.out_codes = out->codes.data();
    d.out_scale = out->scale.data();
    if (r97_iqr_validate(d) != 0) return false;
    if (r97_iqr_decode_host(d) != 0) return false;

    const int icf = plane_of(*enc, "scale.col");
    if (icf < 0) return true;
    // One f32 a row times one f32 a 128-K block, the product rounded to bf16 once: what the device
    // row's eff_scale computes for the kRowF32 form.
    const RadEncPlane& cf = enc->plane[icf];
    if (cf.dtype != RAD_F32 || !held(plane, icf, nkb * 4)) return false;
    if (d.scale_dtype != R97_IQR_F32 || d.scale_block_cols > 0) return false;
    const float* factor = reinterpret_cast<const float*>(plane[icf].data);
    const float* row = reinterpret_cast<const float*>(plane[plane_of(*enc, "scale")].data);
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t kb = 0; kb < nkb; ++kb)
            out->scale[static_cast<size_t>(r * nkb + kb)] =
                r97_iqr_f32_to_bf16(row[r] * factor[kb]);
    return true;
}

}  // namespace r97
