// src/iqr_decode.h -- decode a stored IQR weight to the int8 grid the transposed linear reads.
//
// The trunk of qwen3.8-27b-iq4r-s-v4.rad is stored as IQR codes: a tail-biting trellis down each
// column, one code a row, with a block scale. radiance decodes such a weight where its forward
// GEMM reads it (libr4d/r4d_iqr_dec.h), but the backward needs the weight itself, in the plain
// int8 [N, K] grid a `gemm_nt_q` reads, with one bf16 scale per 128 columns.
//
// This is that decode, as one plain description (R97IqrDecodeDesc), a host implementation to check
// it against libquant (iqr_decode_host), and the state math the device kernel shares with it
// (every function here is __host__ __device__ when hipcc compiles it). The code is libquant's
// grid_tcq_decode (libquant/lq_grid.cpp): the same tail-biting seam, the same mad LUT, the same
// two-level scale, and the same tcq_rb rounding.
//
// What is not applied: the fwht transform (the int8 grid is the codes' basis, not W) and the
// scale.col factor (the forward folds it into the activation's per-128 scale, so the weight side
// keeps the codes libquant's tcq_colf multiplies it into). Callers that need W in the logical
// basis apply the transform themselves.
#pragma once

#include <math.h>
#include <stdint.h>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define R97_IQR_HD __host__ __device__ inline
#else
#define R97_IQR_HD inline
#endif

// The stored scale element types an IQR weight uses. The numbers are the kernel ABI: the test and
// the kernel both include this header, so they cannot drift.
enum R97IqrScaleType {
    R97_IQR_F32    = 0,
    R97_IQR_BF16   = 1,
    R97_IQR_F8E4M3 = 2,
    R97_IQR_F16    = 3,
};

// One stored IQR weight and one range of its output rows.
//
// The plane pointers point at the caller's copy of the stored planes, and `codes` and `scale`
// start at row `row0` (the kernel reads rows relative to `row0`). `lut` and the optional
// `scale2` are whole planes. `out_codes` is [rows, cols] int8 row-major and `out_scale` is
// [rows, cols/128] bf16, the two operands a `gemm_nt_q` over the decoded weight reads.
//
// `row0` must be a multiple of `run_rows`; every chunk is whole runs but the weight's last. The
// caller provides both output buffers (the scratch): rows*cols bytes and rows*(cols/128)*2 bytes.
struct R97IqrDecodeDesc {
    // The whole weight and the range to decode.
    int64_t rows_all;   // N, the weight's output rows
    int64_t cols;       // K, its input columns; a multiple of 128
    int64_t row0;       // first output row, a multiple of run_rows
    int64_t rows;       // rows to decode; whole runs but the weight's last

    // The code plane: `bits` per code, packed along a row lowest bits first; `state_bits` = log2
    // of the trellis's state count; `run_rows` = the rows one tail-biting run covers; `rb` = the
    // rows one int8 scale covers (1 when there is no tcq_rb).
    int32_t bits;
    int32_t state_bits;
    int32_t run_rows;
    int32_t rb;

    // The codebook: `mad` computes state values (trellis.mad's A and B), otherwise `lut` holds
    // one int8 value a state.
    int32_t   mad;
    int32_t   round_mad;  // tcq_rb's rounding: libquant uses its mad form when scale.col is
                          // present too, whatever the codebook
    uint32_t  mad_a;
    uint32_t  mad_b;
    const int8_t* lut;

    const uint8_t* codes;
    int64_t        codes_row_bytes;

    // The first-level scale plane, TILED by (1, `scale_block_cols`) or (1, cols) when 0.
    const uint8_t* scale;
    int32_t        scale_dtype;
    int64_t        scale_block_cols;
    int64_t        scale_row_bytes;

    // The second level (scale.1), one f32 a block of the first: block rows/cols of 0 mean the
    // whole tensor, the same convention radiance's plane blocks use.
    const uint8_t* scale2;
    int32_t        scale2_dtype;
    int64_t        scale2_row_bytes;
    int64_t        scale2_block_rows;
    int64_t        scale2_block_cols;

    // Outputs, caller-provided.
    int8_t*  out_codes;
    uint16_t* out_scale;   // bf16 bits
};

// ---------------------------------------------------------------- scalar formats

R97_IQR_HD uint16_t r97_iqr_f32_to_bf16(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((u >> 16) | 0x0040u);  // a quiet NaN
    const uint32_t round = 0x7FFFu + ((u >> 16) & 1u);                            // nearest, even
    return (uint16_t)((u + round) >> 16);
}

R97_IQR_HD float r97_iqr_bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

R97_IQR_HD float r97_iqr_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, u;
    if (e == 0) {
        if (m == 0) u = sign;
        else {
            e = 127 - 15 + 1;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            m &= 0x3FFu;
            u = sign | (e << 23) | (m << 13);
        }
    } else if (e == 31) {
        u = sign | 0x7F800000u | (m << 13);
    } else {
        u = sign | ((e + 127 - 15) << 23) | (m << 13);
    }
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

// OCP FP8 E4M3, the form libquant uses for a trellis's first-level scale (radiance's
// rad_fp8e4m3_to_f32, the same bits).
R97_IQR_HD float r97_iqr_e4m3_to_f32(uint8_t b) {
    const uint32_t s = (uint32_t)(b >> 7) << 31;
    uint32_t e = (uint32_t)(b >> 3) & 0xFu;
    uint32_t m = (uint32_t)b & 0x7u;
    uint32_t u;
    if (e == 0u) {
        if (m == 0u) {
            u = s;
        } else {
            int sh = 0;
            while (!(m & 0x8u)) { m <<= 1; ++sh; }
            m &= 0x7u;
            u = s | ((uint32_t)(127 - 6 - sh) << 23) | (m << 20);
        }
    } else if (e == 0xFu && m == 0x7u) {
        u = s | 0x7FC00000u;  // the single NaN encoding
    } else {
        u = s | ((e - 7u + 127u) << 23) | (m << 20);
    }
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

R97_IQR_HD float r97_iqr_load_scalar(const uint8_t* base, int32_t dtype, int64_t i) {
    switch (dtype) {
        case R97_IQR_F32:    return ((const float*)base)[i];
        case R97_IQR_BF16:   return r97_iqr_bf16_to_f32(((const uint16_t*)base)[i]);
        case R97_IQR_F8E4M3: return r97_iqr_e4m3_to_f32(base[i]);
        case R97_IQR_F16:    return r97_iqr_f16_to_f32(((const uint16_t*)base)[i]);
        default:             return 0.0f;
    }
}

// A stored code row's element `i`, the packing abi/rad_types.h describes.
R97_IQR_HD uint32_t r97_iqr_load_bits(const uint8_t* row, int32_t bits, int64_t i) {
    const int64_t o = i * (int64_t)bits;
    const int64_t byte = o >> 3;
    const int sh = (int)(o & 7);
    uint32_t w = row[byte];
    if (sh + bits > 8)  w |= (uint32_t)row[byte + 1] << 8;
    if (sh + bits > 16) w |= (uint32_t)row[byte + 2] << 16;
    if (sh + bits > 24) w |= (uint32_t)row[byte + 3] << 24;
    return (w >> sh) & ((1u << bits) - 1u);
}

// ---------------------------------------------------------------- one value

// The state's value on the int8 grid: trellis.mad's computed LUT or the stored one.
R97_IQR_HD int32_t r97_iqr_state_value(const R97IqrDecodeDesc& d, uint32_t st) {
    if (d.mad) {
        const uint32_t h = (((uint32_t)(st & 0xFFFFFFu)) * (d.mad_a & 0xFFFFFFu) + d.mad_b) & 0x3F3F3F3Fu;
        const uint32_t v = (h & 0xFFu) + ((h >> 8) & 0xFFu) + ((h >> 16) & 0xFFu) + ((h >> 24) & 0xFFu);
        return (int32_t)v - 126;
    }
    return (int32_t)d.lut[st];
}

// The scale of row r's 128-column block `kb`, with the second level settled to bf16 as libquant
// does. `kb` is the index into the output's per-128 scales.
R97_IQR_HD float r97_iqr_block_scale(const R97IqrDecodeDesc& d, int64_t r, int64_t kb) {
    const int64_t bc = d.scale_block_cols > 0 ? d.scale_block_cols : d.cols;
    const int64_t gb = bc == d.cols ? 0 : (kb * 128) / bc;
    const uint8_t* row = d.scale + (r - d.row0) * d.scale_row_bytes;
    float v = r97_iqr_load_scalar(row, d.scale_dtype, gb);
    if (d.scale2) {
        const int64_t i2 = d.scale2_block_rows > 0 ? (r / d.scale2_block_rows) : 0;
        const int64_t j2 = d.scale2_block_cols > 0 ? (gb * bc) / d.scale2_block_cols : 0;
        const float s2 = r97_iqr_load_scalar(d.scale2 + i2 * d.scale2_row_bytes, d.scale2_dtype, j2);
        v = r97_iqr_bf16_to_f32(r97_iqr_f32_to_bf16(v * s2));
    }
    return v;
}

// The one int8 scale an rb block of rows shares: the largest of its scales (libquant starts the
// maximum at zero, so a block of non-positive scales decodes to zeros).
R97_IQR_HD float r97_iqr_block_max(const R97IqrDecodeDesc& d, int64_t r, int64_t kb) {
    const int64_t rb = d.rb;
    int64_t r0 = (r / rb) * rb;
    int64_t r1 = r0 + rb;
    if (r1 > d.rows_all) r1 = d.rows_all;
    float m = 0.0f;
    for (int64_t q = r0; q < r1; ++q) {
        const float v = r97_iqr_block_scale(d, q, kb);
        m = v > m ? v : m;
    }
    return m;
}

// libquant's rb_round: round-to-nearest on the int8 grid, with tcq_code=mad's magic-number form.
R97_IQR_HD int32_t r97_iqr_rb_round(int32_t v, float ratio, int32_t mad) {
    if (mad) return (int32_t)(fmaf((float)v, ratio, 12582912.0f) - 12582912.0f);
    return (int32_t)rintf((float)v * ratio);
}

// Decode one state at row r to its int8 code and the f32 scale the code multiplies. `cur` is the
// rb block's scale, carried across rows by the caller (only when rb > 1).
R97_IQR_HD int32_t r97_iqr_value(const R97IqrDecodeDesc& d, uint32_t st, int64_t r, int64_t kb,
                                 float* cur, float* scale) {
    const int32_t v = r97_iqr_state_value(d, st);
    if (d.rb > 1) {
        if (r % d.rb == 0) *cur = r97_iqr_block_max(d, r, kb);
        const float sr = r97_iqr_block_scale(d, r, kb);
        *scale = *cur;
        if (*cur <= 0.0f) return 0;
        return r97_iqr_rb_round(v, sr / *cur, d.round_mad);
    }
    *scale = r97_iqr_block_scale(d, r, kb);
    return v;
}

// ---------------------------------------------------------------- a run of one column

// Decode rows [run_start, run_end) of column j. run_start is a multiple of run_rows, and run_end
// is the run's end: the whole run, or the weight's last, shorter one.
R97_IQR_HD void r97_iqr_decode_column(const R97IqrDecodeDesc& d, int64_t run_start, int64_t run_end,
                                      int64_t j) {
    const int32_t c = (d.state_bits + d.bits - 1) / d.bits;
    const uint32_t mask = (1u << d.state_bits) - 1u;
    // The seam: the state before the run's first row is the state after its last.
    uint32_t st = 0;
    for (int32_t q = 0; q < c; ++q) {
        const int64_t t = run_end - c + q;
        st = ((st << d.bits) | r97_iqr_load_bits(d.codes + (t - d.row0) * d.codes_row_bytes, d.bits, j)) & mask;
    }
    const int64_t kb = j >> 7;
    const int64_t nkb = d.cols >> 7;
    float cur = 0.0f;
    for (int64_t r = run_start; r < run_end; ++r) {
        st = ((st << d.bits) | r97_iqr_load_bits(d.codes + (r - d.row0) * d.codes_row_bytes, d.bits, j)) & mask;
        float sc;
        const int32_t v = r97_iqr_value(d, st, r, kb, &cur, &sc);
        d.out_codes[(r - d.row0) * d.cols + j] = (int8_t)v;
        if ((j & 127) == 0) d.out_scale[(r - d.row0) * nkb + kb] = r97_iqr_f32_to_bf16(sc);
    }
}

// ---------------------------------------------------------------- the description's rules

// 0 when the kernel can decode this range, 1 when it cannot. The caller checks this too, so a
// bad range is an error rather than a wrong answer.
R97_IQR_HD int32_t r97_iqr_validate(const R97IqrDecodeDesc& d) {
    if (!d.codes || !d.out_codes || !d.out_scale) return 1;
    if (d.rows <= 0 || d.row0 < 0 || d.rows_all <= 0 || d.rows > d.rows_all - d.row0) return 1;
    if (d.cols <= 0 || (d.cols & 127) != 0) return 1;
    if (d.bits < 1 || d.bits > 16 || d.state_bits < 1 || d.state_bits > 24) return 1;
    if (d.run_rows <= 0 || d.rb <= 0 || d.run_rows % d.rb != 0) return 1;
    if (d.row0 % d.run_rows != 0) return 1;
    if (d.rows % d.run_rows != 0 && d.row0 + d.rows != d.rows_all) return 1;
    const int64_t bc = d.scale_block_cols > 0 ? d.scale_block_cols : d.cols;
    if (bc != 128 && bc != d.cols) return 1;
    if (!d.mad && !d.lut) return 1;
    if (d.scale2 && (d.scale2_block_rows < 0 || d.scale2_block_cols < 0)) return 1;
    // The output has one scale a 128-column block, so the second level must be constant over one.
    if (d.scale2 && d.scale2_block_cols > 0 && d.scale2_block_cols % 128 != 0) return 1;
    const int32_t c = (d.state_bits + d.bits - 1) / d.bits;
    for (int64_t r0 = d.row0; r0 < d.row0 + d.rows; r0 += d.run_rows) {
        int64_t r1 = r0 + d.run_rows;
        if (r1 > d.rows_all) r1 = d.rows_all;
        if (r1 - r0 < c) return 1;  // no seam to seed the run from
    }
    return 0;
}

// The host implementation: the same loop the device kernel runs, one column after another. The
// tests check it against libquant's grid_tcq_decode, and the device kernel against it.
inline int32_t r97_iqr_decode_host(const R97IqrDecodeDesc& d) {
    if (r97_iqr_validate(d)) return 1;
    for (int64_t r0 = d.row0; r0 < d.row0 + d.rows; r0 += d.run_rows) {
        int64_t r1 = r0 + d.run_rows;
        if (r1 > d.rows_all) r1 = d.rows_all;
        for (int64_t j = 0; j < d.cols; ++j) r97_iqr_decode_column(d, r0, r1, j);
    }
    return 0;
}

// The device kernel's entry point, in libr97. It launches asynchronously; r97_iqr_decode_wait
// waits for the launched work. Returns 0, or 1 when the description is outside the kernel's
// rules (r97_iqr_validate).
extern "C" int r97_iqr_decode(const R97IqrDecodeDesc* desc);
extern "C" int r97_iqr_decode_wait(void);
