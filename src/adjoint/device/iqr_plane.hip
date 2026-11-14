// src/adjoint/device/iqr_plane.hip -- a plugin-private IQR weight plane, decoded on the card.
//
// The container's trunk is not int8 codes. Its weights are IQR forms (libr4d/r4d_iqr.h): a
// tail-biting trellis or a table at B bits a value, a state carried across a 128-K unit, and the
// scale form the encoding names. The forward decodes each unit where it reads it with libr4d's own
// device decoders, so the backward calls the same arithmetic rather than carry a second trellis:
// r4d_iqr_dev::Dec's fetch/rows/unzero and r4d_iqr_dev::frag, the exact sequence
// libr4d/r4d_gemm_iqr.hip:728-732 runs. A wrong dx therefore disagrees with the forward instead of
// agreeing with itself.
//
// What this file states itself is only the classification -- which form an encoding's planes are --
// because libr4d's classify is file-static (libr4d/r4d_iqr_layout.cpp:83-158). It is ported rule for
// rule and cites the lines; a wrong classification makes the decode read the wrong width, which
// tests/host/adjoint_iqr_plane.cpp pins by requiring a wrong form to change dx.
//
// The decoded values are written in the int8 frag64 order the i8a8h row already reads
// (plane_index.h), so the linear row's dx runs unchanged over a decoded plane. One wave decodes one
// unit -- 16 rows by 128 columns -- through libr4d's own LDS transpose.
#include "iqr_plane.h"

#include "plane_index.h"

#include "r4d_iqr.h"
#include "r4d_iqr_dec.h"

#include "rad_device.h"
#include "rad_encoding.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace r97 {
namespace device {
namespace {

using namespace r4d_iqr;
using namespace r4d_iqr_dev;

// ---- the classification -----------------------------------------------------------------------
// Ported from libr4d/r4d_iqr_layout.cpp:31-158, whose helpers and rules these are. `entries` is not
// carried: the decoder reads the form's table out of the joined weight buffer itself
// (r4d_iqr_dec.h:385-402 takes the aux plane's offset from the form and the geometry).

struct IqrForm {
    uint32_t kind = 0, bits = 0, lv = 0, L = 0, lrb = 0, lrun = 0, scale = 0;
};

int log2_exact(int64_t v) {  // r4d_iqr_layout.cpp:31
    if (v <= 0 || (v & (v - 1))) return -1;
    int l = 0;
    while ((static_cast<int64_t>(1) << l) < v) ++l;
    return l;
}

int unsigned_bits(uint32_t dt) {  // r4d_iqr_layout.cpp:38
    static const uint32_t kU[] = {RAD_U1, RAD_U2, RAD_U3, RAD_U4, RAD_U5, RAD_U6, RAD_U7, RAD_U8};
    for (int i = 0; i < 8; ++i)
        if (dt == kU[i]) return i + 1;
    return 0;
}

int signed_bits(uint32_t dt) {  // r4d_iqr_layout.cpp:44
    static const uint32_t kS[] = {RAD_I2, RAD_I3, RAD_I4, RAD_I5, RAD_I6, RAD_I7};
    for (int i = 0; i < 6; ++i)
        if (dt == kS[i]) return i + 2;
    return 0;
}

// r4d_iqr_layout.cpp:53-79: the scale plane a form reads -- one bf16 per [1 x 128], or E4M3 per
// [1 x 128] under an f32 or bf16 second level, or one f32 a row with an optional column factor.
bool scale_form(const RadEncoding* enc, int want_planes, uint32_t* out) {
    const RadEncPlane* sc = rad_enc_plane(enc, "scale");
    const RadEncPlane* s2 = rad_enc_plane(enc, "scale.1");
    if (sc && sc->kind == RAD_PLANE_TILED && sc->block[0] == 1 && sc->block[1] == 0 &&
        sc->dtype == RAD_F32 && !s2) {
        const RadEncPlane* cf = rad_enc_plane(enc, "scale.col");
        if (cf && (cf->kind != RAD_PLANE_TILED || cf->dtype != RAD_F32 || cf->block[0] != 0 ||
                   cf->block[1] != 128))
            return false;
        if (enc->n_planes != want_planes + (cf ? 1 : 0)) return false;
        *out = kRowF32;
        return true;
    }
    if (!sc || sc->kind != RAD_PLANE_TILED || sc->block[0] != 1 || sc->block[1] != 128) return false;
    if (!s2) {
        if (sc->dtype != RAD_BF16 || enc->n_planes != want_planes) return false;
        *out = kBf16;
        return true;
    }
    if (sc->dtype != RAD_F8E4M3 || enc->n_planes != want_planes + 1) return false;
    if (s2->kind != RAD_PLANE_TILED || (s2->dtype != RAD_F32 && s2->dtype != RAD_BF16) ||
        s2->block[1] != 0 || (s2->block[0] != 0 && s2->block[0] != 1))
        return false;
    *out = s2->block[0] == 0 ? kE4m3Tensor : kE4m3Row;
    return true;
}

// r4d_iqr_layout.cpp:83-158: which IQR form `enc` is at transform `tr`. Int8 codes are not one --
// the int8 GEMMs read them as they are -- and neither is anything this file cannot decode.
bool classify(const RadEncoding* enc, const char* tr, IqrForm* f) {
    if (!enc || std::strncmp(enc->transform, tr, RAD_ENC_STR) != 0) return false;
    const RadEncPlane& cp = enc->plane[0];
    if (std::strncmp(cp.role, "codes", RAD_ENC_STR) != 0) return false;
    if (rad_enc_is(enc, "trellis")) {
        const RadEncPlane* tp = rad_enc_plane(enc, "trellis");
        const RadEncPlane* mp = rad_enc_plane(enc, "trellis.mad");
        const int b = unsigned_bits(cp.dtype);
        uint32_t sc = 0;
        if (!tp || !b || cp.block[0] != 1 || cp.block[1] != 1 || !scale_form(enc, mp ? 4 : 3, &sc))
            return false;
        if (mp && (mp->kind != RAD_PLANE_TABLE || mp->dtype != RAD_U32 || mp->extent[0] != 2 ||
                   mp->extent[1] != 1))
            return false;
        const int L = log2_exact(tp->extent[0]);
        const int lrun = log2_exact(tp->block[0]);
        const int lrb = tp->block[1] > 1 ? log2_exact(tp->block[1]) : 0;
        if (tp->kind != RAD_PLANE_TABLE || tp->dtype != RAD_I8 || tp->extent[1] != 1 || L <= b ||
            L > (mp ? 16 : 14) || lrun < 4 || lrb < 0 || lrb > lrun)
            return false;
        f->kind = mp ? kTrellisMad : kTrellis;
        f->bits = static_cast<uint32_t>(b);
        f->L = static_cast<uint32_t>(L);
        f->lrun = static_cast<uint32_t>(lrun);
        f->lrb = static_cast<uint32_t>(lrb);
        f->scale = sc;
        return span(Form{f->kind, f->bits, 0, f->L, f->lrb, f->scale, f->lrun}) <= 16;
    }
    if (!rad_enc_is(enc, "affine")) return false;
    uint32_t sc = 0;
    if (cp.block[0] != 1) return false;
    if (cp.block[1] > 1) {
        const RadEncPlane* gr = rad_enc_plane(enc, "grid");
        const int b = unsigned_bits(cp.dtype) ? unsigned_bits(cp.dtype)
                    : cp.dtype == RAD_U9 ? 9 : cp.dtype == RAD_U10 ? 10 : cp.dtype == RAD_U11 ? 11
                    : cp.dtype == RAD_U12 ? 12 : 0;
        const int lv = log2_exact(cp.block[1]);
        if (!gr || !b || lv < 1 || lv > 2 || !scale_form(enc, 3, &sc) || sc != kBf16 ||
            gr->kind != RAD_PLANE_TABLE || gr->dtype != RAD_I8 || gr->extent[1] != cp.block[1] ||
            gr->extent[0] < 1 || gr->extent[0] > (static_cast<int64_t>(1) << b) ||
            (static_cast<int64_t>(1) << b) * cp.block[1] > 16384)
            return false;
        f->kind = kGrid;
        f->bits = static_cast<uint32_t>(b);
        f->lv = static_cast<uint32_t>(lv);
        f->scale = kBf16;
        return true;
    }
    if (const int b = signed_bits(cp.dtype)) {
        if (!scale_form(enc, 2, &sc) || sc != kBf16) return false;
        f->kind = kSigned;
        f->bits = static_cast<uint32_t>(b);
        return true;
    }
    const int b = unsigned_bits(cp.dtype);
    if (!b || b > 8 || (b == 8 && !rad_enc_plane(enc, "zero"))) return false;
    if (rad_enc_plane(enc, "zero")) {
        if (!scale_form(enc, 3, &sc) || sc != kBf16) return false;
        f->kind = kZero;
        f->bits = static_cast<uint32_t>(b);
        return true;
    }
    if (const RadEncPlane* tp = rad_enc_plane(enc, "table")) {
        if (!scale_form(enc, 3, &sc) || sc != kBf16 || tp->kind != RAD_PLANE_TABLE ||
            tp->dtype != RAD_I8 || tp->extent[0] != 1 || tp->extent[1] < 2 ||
            tp->extent[1] > (static_cast<int64_t>(1) << b))
            return false;
        f->kind = kTable;
        f->bits = static_cast<uint32_t>(b);
        return true;
    }
    return false;
}

// ---- the decode --------------------------------------------------------------------------------

// One wave decodes one unit: 16 rows by 128 columns. libr4d's decoder reads the unit's codes and
// its scale, and `frag` transposes the decoded int8 values into the fragment order the i8a8h row
// reads -- the same call the forward makes (libr4d/r4d_gemm_iqr.hip:728-732).
template <class F>
__global__ __launch_bounds__(32) void iqr_decode_kernel(Ctx c0, uint8_t* __restrict__ out,
                                                       uint32_t nkb, uint32_t K) {
    using D = Dec<F>;
    __shared__ __attribute__((aligned(16))) int8_t s_tab[D::kTab];
    __shared__ __attribute__((aligned(16))) uint8_t tr[16u * kTrPitch];
    const uint32_t lane = threadIdx.x & 31u;
    Ctx c = c0;
    c.tab = s_tab;
    D::stage(s_tab, c, threadIdx.x, blockDim.x);
    __syncthreads();
    const uint32_t units = (c.N / 16u) * nkb;
    for (uint32_t u = blockIdx.x; u < units; u += gridDim.x) {
        const uint32_t rt = u / nkb, kb = u % nkb;
        constexpr bool UNS = D::MAD && F::LRB == 0;
        typename D::Raw cur = D::fetch(c, rt, kb, lane, true);
        uint32_t w[16];
        D::template rows<UNS>(cur, c, rb_ratio<F>(c, rt * 16u + (lane & 15u), kb), w);
        D::unzero(c, rt, kb, w);  // the zero form's bias; a no-op for every other form
        r4d_fp8::Stage8 s[2];
        frag(w, tr, lane, s);
        // The unit's values, eight contiguous k at a time, at the int8 fragment plane's addresses.
        const int row = static_cast<int>(rt * 16u + (lane & 15u));
        const uint32_t col = (lane >> 4) * 8u;
#pragma unroll
        for (uint32_t h = 0; h < 2; ++h) {
            const uint32_t* q = reinterpret_cast<const uint32_t*>(&s[h]);
#pragma unroll
            for (uint32_t t = 0; t < 4; ++t) {
                const int k = static_cast<int>(kb * 128u + h * 64u + t * 16u + col);
                *reinterpret_cast<uint2*>(out + frag_go(static_cast<int>(K), row, k)) =
                    make_uint2(q[2 * t], q[2 * t + 1]);
            }
        }
        // `tr` is the block's; the wave-scope fences inside frag do not order a second wave.
        __syncthreads();
    }
}

template <class F>
bool launch(const IqrForm& f, uint32_t N, uint32_t K, const void* W, const void* S, void* out,
            void* stream) {
    Ctx c{};
    c.W = static_cast<const uint8_t*>(W);
    c.S = static_cast<const uint8_t*>(S);
    c.N = N;
    c.nkb = K / 128u;
    // FormT's word() carries lrun 0, so the run's tiles come from the encoding, as the forward's
    // launcher takes them (libr4d/r4d_gemm_iqr.hip:835).
    c.run_tiles = is_trellis(F::KIND) ? (1u << f.lrun) / 16u : 1u;
    if (c.run_tiles == 0 || N % 16u != 0 || c.nkb == 0) return false;
    const uint32_t units = (N / 16u) * c.nkb;
    hipLaunchKernelGGL((iqr_decode_kernel<F>), dim3(units), dim3(32), 0,
                       static_cast<hipStream_t>(stream), c, static_cast<uint8_t*>(out), c.nkb, K);
    return hipGetLastError() == hipSuccess;
}

bool dispatch(const IqrForm& f, uint32_t N, uint32_t K, const void* W, const void* S, void* out,
              void* stream) {
// The parameters carry a trailing underscore because the preprocessor substitutes inside `f.lv`
// and `f.lrb`: a parameter named `lv` would rewrite the member access itself.
#define R97_IQR_TRY(K_, B_, LV_, L_, LRB_, SC_)                                                    \
    if (f.kind == (K_) && f.bits == (B_) && f.lv == (LV_) && f.L == (L_) && f.lrb == (LRB_) &&     \
        f.scale == (SC_))                                                                          \
        return launch<FormT<K_, B_, LV_, L_, LRB_, SC_>>(f, N, K, W, S, out, stream);
    R4D_IQR_FORMS(R97_IQR_TRY)
#undef R97_IQR_TRY
    return false;
}

}  // namespace

bool iqr_plane_decode(const void* enc, uint32_t N, uint32_t K, const void* W, const void* S,
                      void* out, void* stream) {
    if (enc == nullptr || W == nullptr || S == nullptr || out == nullptr) return false;
    const RadEncoding* e = static_cast<const RadEncoding*>(enc);
    // The transform the engine's own layout hook passes for an i8a8h op (r4d_iqr_layout.cpp:212).
    const bool rot = std::strcmp(e->transform, "fwht128") == 0;
    IqrForm f;
    if (!classify(e, rot ? "fwht128" : "", &f)) return false;
    return dispatch(f, N, K, W, S, out, stream);
}

const char* iqr_plane_form(const void* enc) {
    static const char* kKind[] = {"?", "signed", "zero", "table", "grid", "trellis", "trellis.mad"};
    static const char* kScale[] = {"bf16", "e4m3-tensor", "e4m3-row", "row-f32"};
    static thread_local char text[96];
    if (enc == nullptr) return "none";
    const RadEncoding* e = static_cast<const RadEncoding*>(enc);
    const bool rot = std::strcmp(e->transform, "fwht128") == 0;
    IqrForm f;
    if (!classify(e, rot ? "fwht128" : "", &f)) return "unclassified";
    std::snprintf(text, sizeof text, "%s b%u L%u lrb%u lrun%u scale=%s",
                  f.kind < 7 ? kKind[f.kind] : "?", f.bits, f.L, f.lrb, f.lrun,
                  f.scale < 4 ? kScale[f.scale] : "?");
    return text;
}

}  // namespace device
}  // namespace r97
