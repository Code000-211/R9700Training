// src/adjoint/device/iqr_plane.h -- a plugin-private IQR weight plane, decoded on the card.
//
// The container's trunk is not int8 codes: its weights are IQR forms (libr4d/r4d_iqr.h), a
// tail-biting trellis or a table, with the scale form the encoding names. The forward decodes each
// unit where it reads it, with libr4d's own device decoders (r4d_iqr_dec.h), and this is the same
// arithmetic called by the backward -- not a second implementation of the trellis.
//
// The decode writes the int8 frag64 order the i8a8h row already reads (plane_index.h), so the
// linear row's dx runs unchanged over a decoded plane.
#pragma once

#include <cstdint>

namespace r97 {
namespace device {

// Decodes `enc`'s weight plane (W, S at N rows and K columns) into `out`, N*K bytes in the int8
// fragment order. Returns false -- and touches nothing -- when the encoding is not a form this file
// implements, so the caller declines the record rather than reading a plane it cannot decode.
bool iqr_plane_decode(const void* enc, uint32_t N, uint32_t K, const void* W, const void* S,
                      void* out, void* stream);

// Writes the form's scale in the plane the i8a8h row reads: N rows by K/128 blocks of bf16 in the
// tile-major order plane_index.h's t16_go addresses -- the row's own scale for a bf16 form, the
// product an E4M3 level or a per-row f32 with its block factors stands for otherwise. The record
// declares the scale with the weight's shape, so its geometry is the form's, never the record's.
bool iqr_plane_scale(const void* enc, uint32_t N, uint32_t K, const void* S, void* out,
                     void* stream);

// Whether `enc` is a form this file decodes: the classification the two calls above apply, without
// touching a device. A caller branches on it before it sizes anything.
bool iqr_plane_supported(const void* enc);

// The form's name, for the diagnostic that prints what a plane was classified as.
const char* iqr_plane_form(const void* enc);

}  // namespace device
}  // namespace r97
