// The IQR decode kernel on the card: every form the first container stores, decoded to the int8
// grid and checked against the host decode byte for byte, then radiance's int8 GEMM over that grid
// against its IQR GEMM over the same codes. Real tensors are the point, so this is a model test
// (support/model_case.h): it resolves the container first thing, skips (exit 77) when it is absent,
// and only then takes the card.

#include <dlfcn.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rad_device.h"
#include "rad_encoding.h"
#include "rad_plugin.h"
#include "radfile.h"

#include "iqr_decode.h"
#include "support/device_check.h"
#include "support/gpu_guard.h"
#include "support/model_case.h"
#include "support/model_dir.h"

namespace {

void expect(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

struct Device {
    RadStream stream = nullptr;
    ~Device() {
        if (stream) rad_stream_destroy(stream);
    }
};

void* device_alloc(int64_t bytes) {
    void* p = rad_dev_alloc(bytes, RAD_MEM_DEVICE);
    if (!p) throw std::runtime_error(std::string("rad_dev_alloc: ") + rad_dev_last_error());
    return p;
}

void device_free(void*& p) {
    if (p) rad_dev_free(p, RAD_MEM_DEVICE);
    p = nullptr;
}

void upload(void* dst, const void* src, int64_t bytes, RadStream s) {
    if (bytes <= 0) return;
    if (rad_memcpy_async(dst, src, bytes, s) != 0)
        throw std::runtime_error(std::string("rad_memcpy_async: ") + rad_dev_last_error());
}

void download(void* dst, const void* src, int64_t bytes, RadStream s) {
    if (bytes <= 0) return;
    if (rad_memcpy_async(dst, src, bytes, s) != 0)
        throw std::runtime_error(std::string("rad_memcpy_async: ") + rad_dev_last_error());
}

int plane_of(const RadEncoding& e, const char* role) {
    for (int i = 0; i < (int)e.n_planes; ++i)
        if (std::strcmp(e.plane[i].role, role) == 0) return i;
    return -1;
}

int scale_type(uint32_t dtype) {
    switch (dtype) {
        case RAD_F32: return R97_IQR_F32;
        case RAD_BF16: return R97_IQR_BF16;
        case RAD_F8E4M3: return R97_IQR_F8E4M3;
        case RAD_F16: return R97_IQR_F16;
        default: return -1;
    }
}

int32_t log2_exact(int64_t n) {
    int32_t l = 0;
    while ((int64_t(1) << l) < n) ++l;
    if ((int64_t(1) << l) != n) return -1;
    return l;
}

// The whole-weight description: plane pointers at row 0, the output pointers left to the caller.
R97IqrDecodeDesc describe(const rad::RadFile& f, const RadFileEntry& e, bool with_out) {
    const RadEncoding& enc = f.encoding(e);
    const int ic = plane_of(enc, "codes");
    const int is = plane_of(enc, "scale");
    const int it = plane_of(enc, "trellis");
    const int iz = plane_of(enc, "zero");
    const std::string nm = f.str(e.name);
    expect(ic >= 0 && is >= 0 && (it >= 0 || iz >= 0), nm + ": the IQR planes");

    R97IqrDecodeDesc d{};
    d.rows_all = e.shape[0];
    d.cols = e.shape[1];
    d.row0 = 0;
    d.rows = d.rows_all;

    int sg = 0;
    const int packed = rad_dtype_packed_int(enc.plane[ic].dtype, &sg);
    expect(packed > 0 && !sg, nm + ": packed unsigned codes");
    d.bits = rad_dtype_bits(enc.plane[ic].dtype);
    if (iz >= 0) {
        // The affine grid (the drafter head): a zero a block beside the scale and no trellis above
        // it, so a row is a run of its own.
        expect(it < 0, nm + ": one form, not a trellis and a zero");
        // The zero is a whole byte a block (RAD_U8) or a packed unsigned field whose width is its
        // own, as radiance writes both.
        int zsg = 0;
        const int zpacked = rad_dtype_packed_int(enc.plane[iz].dtype, &zsg);
        expect(!zsg && (zpacked > 0 || enc.plane[iz].dtype == RAD_U8), nm + ": unsigned zeros");
        d.zero_bits = rad_dtype_bits(enc.plane[iz].dtype);
        expect(d.zero_bits >= 1 && d.zero_bits <= 8, nm + ": a zero of 8 bits at most");
        expect(enc.plane[iz].block[0] == enc.plane[is].block[0] &&
                   enc.plane[iz].block[1] == enc.plane[is].block[1],
               nm + ": the zero is tiled like the scale");
        d.affine = 1;
        d.run_rows = 1;
        d.rb = 1;
    } else {
        d.state_bits = log2_exact(enc.plane[it].extent[0]);
        expect(d.state_bits > 0, nm + ": a power-of-two trellis");
        d.run_rows = (int32_t)enc.plane[it].block[0];
        expect(d.run_rows > 0, nm + ": a trellis run");
        d.rb = enc.plane[it].block[1] > 0 ? (int32_t)enc.plane[it].block[1] : 1;
    }

    d.scale_dtype = scale_type(enc.plane[is].dtype);
    expect(d.scale_dtype >= 0, nm + ": a scale format");
    d.scale_block_cols = enc.plane[is].block[1];
    const int64_t nkb = d.cols / 128;
    const int64_t srow = rad_enc_row_bytes(enc.plane[is].dtype,
                                           d.scale_block_cols > 0 ? nkb : 1);
    d.scale_row_bytes = srow;
    d.codes_row_bytes = rad_enc_row_bytes(enc.plane[ic].dtype, d.cols);
    if (iz >= 0) {
        d.zero_row_bytes = rad_enc_row_bytes(enc.plane[iz].dtype,
                                             d.scale_block_cols > 0 ? nkb : 1);
        // The zero is a plane in the scale's order, one entry a row and block: a decode that walks
        // rows depends on that, so say so rather than read past the plane.
        expect((int64_t)f.plane(e, iz).bytes >= d.rows_all * d.zero_row_bytes,
               nm + ": a zero a row of blocks");
    }

    const int is2 = plane_of(enc, "scale.1");
    if (is2 >= 0) {
        expect(iz < 0, nm + ": the affine grid has no second level");
        d.scale2_dtype = scale_type(enc.plane[is2].dtype);
        expect(d.scale2_dtype >= 0, nm + ": a scale.1 format");
        d.scale2_block_rows = enc.plane[is2].block[0];
        d.scale2_block_cols = enc.plane[is2].block[1];
        d.scale2_row_bytes = rad_enc_row_bytes(enc.plane[is2].dtype,
                                               d.scale2_block_cols > 0 ? nkb : 1);
    }

    const int im = plane_of(enc, "trellis.mad");
    const int icf = plane_of(enc, "scale.col");
    d.mad = im >= 0;
    d.round_mad = d.mad || icf >= 0;
    if (it >= 0) d.lut = (const int8_t*)f.data(f.plane(e, it));
    if (im >= 0) {
        const uint32_t* ab = (const uint32_t*)f.data(f.plane(e, im));
        d.mad_a = ab[0];
        d.mad_b = ab[1];
    }
    d.codes = (const uint8_t*)f.data(f.plane(e, ic));
    d.scale = (const uint8_t*)f.data(f.plane(e, is));
    if (iz >= 0) d.zero = (const uint8_t*)f.data(f.plane(e, iz));
    if (is2 >= 0) d.scale2 = (const uint8_t*)f.data(f.plane(e, is2));

    if (with_out) {
        d.out_codes = nullptr;
        d.out_scale = nullptr;
    }
    return d;
}

// The description of rows [row0, row0+rows): the codes and scale planes start at row0, as the
// kernel reads them relative to it.
R97IqrDecodeDesc slice(R97IqrDecodeDesc d, int64_t row0, int64_t rows) {
    d.row0 = row0;
    d.rows = rows;
    d.codes += row0 * d.codes_row_bytes;
    d.scale += row0 * d.scale_row_bytes;
    if (d.zero) d.zero += row0 * d.zero_row_bytes;
    return d;
}

int64_t code_bytes(const R97IqrDecodeDesc& d) { return d.rows * d.codes_row_bytes; }
int64_t scale_bytes(const R97IqrDecodeDesc& d) { return d.rows * d.scale_row_bytes; }
int64_t zero_bytes(const R97IqrDecodeDesc& d) { return d.zero ? d.rows * d.zero_row_bytes : 0; }
int64_t out_code_bytes(const R97IqrDecodeDesc& d) { return d.rows * d.cols; }
int64_t out_scale_bytes(const R97IqrDecodeDesc& d) {
    return d.rows * (d.cols / 128) * (int64_t)sizeof(uint16_t);
}

void compare(const std::vector<int8_t>& want, const std::vector<uint16_t>& wsc,
             const std::vector<int8_t>& got, const std::vector<uint16_t>& gsc,
             const char* what) {
    for (size_t i = 0; i < want.size(); ++i)
        expect(got[i] == want[i], std::string(what) + ": code " + std::to_string(i) +
                                      " is " + std::to_string((int)got[i]) + ", want " +
                                      std::to_string((int)want[i]));
    for (size_t i = 0; i < wsc.size(); ++i)
        expect(gsc[i] == wsc[i], std::string(what) + ": scale " + std::to_string(i) + " differs");
}

void decode_case(const std::string& so, rad::RadFile& f, const RadFileEntry& e, int64_t row0,
                 int64_t rows) {
    const R97IqrDecodeDesc whole = describe(f, e, false);
    R97IqrDecodeDesc hd = slice(whole, row0, rows);

    std::vector<int8_t> want((size_t)out_code_bytes(hd));
    std::vector<uint16_t> wsc((size_t)out_scale_bytes(hd) / sizeof(uint16_t));
    hd.out_codes = want.data();
    hd.out_scale = wsc.data();
    expect(r97_iqr_validate(hd) == 0,
           std::string(f.str(e.name)) + ": the host description is valid");
    expect(r97_iqr_decode_host(hd) == 0, std::string(f.str(e.name)) + ": the host decode");

    // The device copies are exactly the bytes the host path reads: the chunk of codes and scales,
    // the whole trellis and its mad, and the whole scale.1 (a scalar in this container).
    Device dev;
    if (rad_stream_create(&dev.stream, 0) != 0)
        throw std::runtime_error(std::string("rad_stream_create: ") + rad_dev_last_error());

    void* dcodes = device_alloc(code_bytes(hd));
    void* dscale = device_alloc(scale_bytes(hd));
    void* dzero = hd.zero ? device_alloc(zero_bytes(hd)) : nullptr;
    void* dlut = hd.lut ? device_alloc(int64_t(1) << hd.state_bits) : nullptr;
    void* dmad = hd.mad ? device_alloc(8) : nullptr;
    const int is2 = plane_of(f.encoding(e), "scale.1");
    const int64_t s2bytes = is2 >= 0 ? (int64_t)f.plane(e, is2).bytes : 0;
    void* ds2 = hd.scale2 ? device_alloc(s2bytes) : nullptr;
    void* dout = device_alloc(out_code_bytes(hd));
    void* dsc = device_alloc(out_scale_bytes(hd));

    upload(dcodes, hd.codes, code_bytes(hd), dev.stream);
    upload(dscale, hd.scale, scale_bytes(hd), dev.stream);
    if (hd.zero) upload(dzero, hd.zero, zero_bytes(hd), dev.stream);
    if (hd.lut) upload(dlut, hd.lut, int64_t(1) << hd.state_bits, dev.stream);
    if (hd.mad) {
        uint32_t ab[2] = { hd.mad_a, hd.mad_b };
        upload(dmad, ab, 8, dev.stream);
    }
    if (hd.scale2) upload(ds2, hd.scale2, s2bytes, dev.stream);
    if (rad_stream_sync(dev.stream) != 0)
        throw std::runtime_error(std::string("rad_stream_sync: ") + rad_dev_last_error());

    R97IqrDecodeDesc dd = hd;
    dd.codes = (const uint8_t*)dcodes;
    dd.scale = (const uint8_t*)dscale;
    dd.zero = (const uint8_t*)dzero;
    dd.lut = (const int8_t*)dlut;
    dd.scale2 = (const uint8_t*)ds2;
    if (hd.mad) {
        dd.mad_a = hd.mad_a;
        dd.mad_b = hd.mad_b;
    }
    dd.out_codes = (int8_t*)dout;
    dd.out_scale = (uint16_t*)dsc;

    using DecodeFn = int (*)(const R97IqrDecodeDesc*, R97IqrStream);
    void* h = dlopen(so.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) throw std::runtime_error(std::string("dlopen: ") + dlerror());
    DecodeFn dec = (DecodeFn)dlsym(h, "r97_iqr_decode");
    if (!dec) throw std::runtime_error("the device plugin has no IQR decode");
    // No hidden stream: a call with none is refused rather than quietly run on the legacy one.
    expect(dec(&dd, nullptr) == 1, std::string(f.str(e.name)) + ": no stream, no launch");
    // The kernel launches on the stream this test made, so the same stream's sync is what orders
    // the download after it and what the timing below measures.
    expect(dec(&dd, dev.stream) == 0, std::string(f.str(e.name)) + ": the device decode starts");
    if (rad_stream_sync(dev.stream) != 0)
        throw std::runtime_error(std::string("rad_stream_sync: ") + rad_dev_last_error());

    std::vector<int8_t> got((size_t)out_code_bytes(hd));
    std::vector<uint16_t> gsc((size_t)out_scale_bytes(hd) / sizeof(uint16_t));
    download(got.data(), dout, (int64_t)got.size(), dev.stream);
    download(gsc.data(), dsc, (int64_t)gsc.size() * (int64_t)sizeof(uint16_t), dev.stream);
    if (rad_stream_sync(dev.stream) != 0)
        throw std::runtime_error(std::string("rad_stream_sync: ") + rad_dev_last_error());
    dlclose(h);

    compare(want, wsc, got, gsc, f.str(e.name));
    std::printf("%s rows [%lld, %lld): device == host\n", f.str(e.name),
                (long long)row0, (long long)(row0 + rows));

    device_free(dcodes);
    device_free(dscale);
    device_free(dzero);
    device_free(dlut);
    device_free(dmad);
    device_free(ds2);
    device_free(dout);
    device_free(dsc);
}

// Time the device decode alone: the planes are uploaded once, then the launch and its wait are
// timed, so the number is the kernel's.
void time_case(const std::string& so, rad::RadFile& f, const RadFileEntry& e, int64_t rows,
               const char* label) {
    R97IqrDecodeDesc hd = describe(f, e, false);
    hd.rows = rows > 0 ? rows : e.shape[0];
    hd.row0 = 0;
    if (hd.rows > hd.rows_all) hd.rows = hd.rows_all;

    Device dev;
    if (rad_stream_create(&dev.stream, 0) != 0)
        throw std::runtime_error(std::string("rad_stream_create: ") + rad_dev_last_error());
    void* dcodes = device_alloc(code_bytes(hd));
    void* dscale = device_alloc(scale_bytes(hd));
    void* dzero = hd.zero ? device_alloc(zero_bytes(hd)) : nullptr;
    void* dlut = hd.lut ? device_alloc(int64_t(1) << hd.state_bits) : nullptr;
    void* dmad = hd.mad ? device_alloc(8) : nullptr;
    const int is2 = plane_of(f.encoding(e), "scale.1");
    const int64_t s2bytes = is2 >= 0 ? (int64_t)f.plane(e, is2).bytes : 0;
    void* ds2 = hd.scale2 ? device_alloc(s2bytes) : nullptr;
    void* dout = device_alloc(out_code_bytes(hd));
    void* dsc = device_alloc(out_scale_bytes(hd));
    upload(dcodes, hd.codes, code_bytes(hd), dev.stream);
    upload(dscale, hd.scale, scale_bytes(hd), dev.stream);
    if (hd.zero) upload(dzero, hd.zero, zero_bytes(hd), dev.stream);
    if (hd.lut) upload(dlut, hd.lut, int64_t(1) << hd.state_bits, dev.stream);
    if (hd.mad) {
        uint32_t ab[2] = { hd.mad_a, hd.mad_b };
        upload(dmad, ab, 8, dev.stream);
    }
    if (hd.scale2) upload(ds2, hd.scale2, s2bytes, dev.stream);
    R97IqrDecodeDesc dd = hd;
    dd.codes = (const uint8_t*)dcodes;
    dd.scale = (const uint8_t*)dscale;
    dd.zero = (const uint8_t*)dzero;
    dd.lut = (const int8_t*)dlut;
    dd.scale2 = (const uint8_t*)ds2;
    if (hd.mad) dd.mad_a = hd.mad_a, dd.mad_b = hd.mad_b;
    dd.out_codes = (int8_t*)dout;
    dd.out_scale = (uint16_t*)dsc;
    if (rad_stream_sync(dev.stream) != 0)
        throw std::runtime_error(std::string("rad_stream_sync: ") + rad_dev_last_error());

    using DecodeFn = int (*)(const R97IqrDecodeDesc*, R97IqrStream);
    void* h = dlopen(so.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) throw std::runtime_error(std::string("dlopen: ") + dlerror());
    DecodeFn dec = (DecodeFn)dlsym(h, "r97_iqr_decode");
    if (!dec) throw std::runtime_error("the device plugin has no IQR decode");
    expect(dec(&dd, dev.stream) == 0, "the timed decode starts");
    if (rad_stream_sync(dev.stream) != 0)
        throw std::runtime_error(std::string("rad_stream_sync: ") + rad_dev_last_error());
    double best = 1e30;
    for (int i = 0; i < 5; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        expect(dec(&dd, dev.stream) == 0, "the timed decode starts");
        if (rad_stream_sync(dev.stream) != 0)
            throw std::runtime_error(std::string("rad_stream_sync: ") + rad_dev_last_error());
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
    }
    const double gib = (double)out_code_bytes(hd) / (1024.0 * 1024.0 * 1024.0);
    std::printf("time %s [%lld, %lld): %.3f ms, %.1f GiB/s, %.1f MiB out, %.2f MiB scales\n",
                label, (long long)hd.row0, (long long)(hd.row0 + hd.rows), best,
                gib / (best / 1000.0), (double)out_code_bytes(hd) / (1024.0 * 1024.0),
                (double)out_scale_bytes(hd) / (1024.0 * 1024.0));
    dlclose(h);
    device_free(dcodes); device_free(dscale); device_free(dzero); device_free(dlut);
    device_free(dmad);
    device_free(ds2); device_free(dout); device_free(dsc);
}

// ------------------------------------------------------------------ gemm comparison

struct L4d {
    void* h = nullptr;
    int (*gemm_iqr)(const RadArgs*, RadStream) = nullptr;
    int64_t (*gemm_iqr_scratch)(const RadArgs*) = nullptr;
    int (*gemm_i8)(const RadArgs*, RadStream) = nullptr;
    int64_t (*gemm_i8_scratch)(const RadArgs*) = nullptr;
    int (*layout_iqr)(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*,
                      int, RadLayout*) = nullptr;
    int (*relayout_iqr)(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*,
                        int, void*, int64_t) = nullptr;
    int (*layout_i8)(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*,
                     int, RadLayout*) = nullptr;
    int (*relayout_i8)(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*,
                       int, void*, int64_t) = nullptr;

    ~L4d() {
        if (h) dlclose(h);
    }
};

std::string sibling(const std::string& so, const char* name) {
    const size_t slash = so.find_last_of('/');
    if (slash == std::string::npos) return name;
    return so.substr(0, slash + 1) + name;
}

void load_l4d(L4d& l, const std::string& plugin) {
    // Where libr4d is: the environment's override, or beside the device plugin -- radiance's home
    // kernels/ directory, which is where both modules land. A path that is not there falls through
    // to the next.
    std::vector<std::string> paths;
    const char* env = std::getenv("R97_LIBR4D");
    if (env && *env) paths.push_back(env);
    paths.push_back(sibling(plugin, "libr4d.so"));

    std::string why;
    for (const std::string& path : paths) {
        l.h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (l.h) break;
        why += "\n  " + path + ": " + dlerror();
    }
    if (!l.h) throw std::runtime_error("the IQR GEMM comparison needs libr4d.so:" + why);
#define GET(f, n)                                                                                \
    do {                                                                                         \
        void* p = dlsym(l.h, n);                                                                 \
        if (!p) throw std::runtime_error(std::string("libr4d has no ") + n);                     \
        std::memcpy(&l.f, &p, sizeof(p));                                                        \
    } while (0)
    GET(gemm_iqr, "r4d_rad_gemm_iqr_m64");
    GET(gemm_iqr_scratch, "r4d_rad_gemm_iqr_m64_scratch");
    GET(gemm_i8, "r4d_rad_gemm_i8a8_m16");
    GET(gemm_i8_scratch, "r4d_rad_gemm_i8a8_m16_scratch");
    GET(layout_iqr, "r4d_rad_layout_iqr");
    GET(relayout_iqr, "r4d_rad_relayout_iqr");
    GET(layout_i8, "r4d_rad_layout_i8_block");
    GET(relayout_i8, "r4d_rad_relayout_i8_block");
#undef GET
}

RadTensor tensor(void* data, uint32_t dtype, int64_t r, int64_t c) {
    RadTensor t{};
    t.data = data;
    t.dtype = dtype;
    t.rank = 2;
    t.shape[0] = r;
    t.shape[1] = c;
    t.stride[0] = c;
    t.stride[1] = 1;
    return t;
}

RadTensor stored(void* data, const RadLayout& l) {
    RadTensor t{};
    t.data = data;
    t.dtype = l.dtype;
    t.rank = l.rank;
    for (uint32_t i = 0; i < l.rank && i < RAD_MAX_RANK; ++i) t.shape[i] = l.shape[i];
    rad_tensor_pack(&t);
    return t;
}

std::vector<RadTensor> iqr_planes(const rad::RadFile& f, const RadFileEntry& e,
                                  const int* sel, int n_sel) {
    std::vector<RadTensor> v;
    const RadEncoding& enc = f.encoding(e);
    for (int i = 0; i < n_sel; ++i) {
        const RadEncPlane& p = enc.plane[sel[i]];
        int64_t r = p.extent[0] > 0 ? p.extent[0] : e.shape[0];
        int64_t c = p.extent[1] > 0 ? p.extent[1] : e.shape[1];
        if (p.kind == RAD_PLANE_TILED) {
            r = e.shape[0];
            c = p.block[1] > 0 ? (e.shape[1] + p.block[1] - 1) / p.block[1] : 1;
        }
        v.push_back(tensor(const_cast<uint8_t*>(f.data(f.plane(e, sel[i]))), p.dtype, r, c));
    }
    return v;
}

void gemm_case(L4d& l, const std::string& plugin, rad::RadFile& f, const RadFileEntry& e) {
    const int64_t N = e.shape[0], K = e.shape[1], M = 32, nkb = K / 128;
    const RadEncoding& enc = f.encoding(e);
    const int ic = plane_of(enc, "codes"), is = plane_of(enc, "scale");
    const int it = plane_of(enc, "trellis"), im = plane_of(enc, "trellis.mad");
    expect(ic >= 0 && is >= 0 && it >= 0, std::string(f.str(e.name)) + ": the IQR planes");
    expect(im >= 0, std::string(f.str(e.name)) + ": a trellis.mad GEMM case");

    // The host decode: radiance's own IQR GEMM decodes the same codes, so the int8 reference is
    // exactly the grid the backward will read.
    R97IqrDecodeDesc hd = describe(f, e, false);
    std::vector<int8_t> grid((size_t)(N * K));
    std::vector<uint16_t> gsc((size_t)(N * nkb));
    hd.out_codes = grid.data();
    hd.out_scale = gsc.data();
    expect(r97_iqr_validate(hd) == 0, "the GEMM weight's description is valid");
    expect(r97_iqr_decode_host(hd) == 0, "the GEMM weight decodes on the host");

    // The int8 encoding of that grid, in the same rotated basis and with the same bf16 scales.
    RadEncoding ie = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    rad_enc_copy_str(ie.transform, enc.transform);
    const int64_t iew = rad_enc_plane_bytes(&ie.plane[0], N, K);
    const int64_t ies = rad_enc_plane_bytes(&ie.plane[1], N, K);
    std::vector<uint8_t> iwbuf((size_t)iew), isbuf((size_t)ies);
    std::memcpy(iwbuf.data(), grid.data(), (size_t)(N * K));
    std::memcpy(isbuf.data(), gsc.data(), (size_t)(N * nkb * 2));
    RadTensor iw[1] = { tensor(iwbuf.data(), RAD_I8, N, K) };
    RadTensor isc[1] = { tensor(isbuf.data(), RAD_BF16, N, nkb) };

    const RadParam params[] = {
        RAD_INT("M", M), RAD_INT("N", N), RAD_INT("K", K), RAD_INT("group", 128),
        RAD_STR("dtype", "i8a8h"),
    };
    const int np = (int)(sizeof(params) / sizeof(params[0]));

    RadLayout lw{}, lsw{}, liw{}, lis{};
    int selw[3] = { ic, it, im };
    int sels[1] = { is };
    const std::vector<RadTensor> pw = iqr_planes(f, e, selw, 3);
    const std::vector<RadTensor> ps = iqr_planes(f, e, sels, 1);
    expect(l.layout_iqr(params, np, 2, &enc, selw, pw.data(), 3, &lw) == 0,
           std::string(f.str(e.name)) + ": the IQR weight layout");
    expect(l.layout_iqr(params, np, 3, &enc, sels, ps.data(), 1, &lsw) == 0,
           std::string(f.str(e.name)) + ": the IQR scale layout");
    std::vector<uint8_t> wbuf((size_t)lw.bytes), swbuf((size_t)lsw.bytes);
    expect(l.relayout_iqr(params, np, 2, &enc, selw, pw.data(), 3, wbuf.data(), lw.bytes) == 0,
           std::string(f.str(e.name)) + ": the IQR weight relayout");
    expect(l.relayout_iqr(params, np, 3, &enc, sels, ps.data(), 1, swbuf.data(), lsw.bytes) == 0,
           std::string(f.str(e.name)) + ": the IQR scale relayout");

    int seli[1] = { 0 }, selsi[1] = { 1 };
    expect(l.layout_i8(params, np, 2, &ie, seli, iw, 1, &liw) == 0,
           std::string(f.str(e.name)) + ": the int8 weight layout");
    expect(l.layout_i8(params, np, 3, &ie, selsi, isc, 1, &lis) == 0,
           std::string(f.str(e.name)) + ": the int8 scale layout");
    std::vector<uint8_t> iwst((size_t)liw.bytes), isst((size_t)lis.bytes);
    expect(l.relayout_i8(params, np, 2, &ie, seli, iw, 1, iwst.data(), liw.bytes) == 0,
           std::string(f.str(e.name)) + ": the int8 weight relayout");
    expect(l.relayout_i8(params, np, 3, &ie, selsi, isc, 1, isst.data(), lis.bytes) == 0,
           std::string(f.str(e.name)) + ": the int8 scale relayout");

    // The activations, the same bytes for both rows.
    std::vector<int8_t> a((size_t)(M * K));
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = (int8_t)((int)((i * 1103515245u + 12345u) >> 16) % 255 - 127);
    std::vector<float> asc((size_t)(M * nkb), 0.01f), y((size_t)(M * N), 0.0f);
    std::vector<uint16_t> ybr((size_t)(M * N), 0);

    Device dev;
    if (rad_stream_create(&dev.stream, 0) != 0)
        throw std::runtime_error(std::string("rad_stream_create: ") + rad_dev_last_error());
    void* da = device_alloc(M * K);
    void* das = device_alloc(M * nkb * 4);
    void* dw = device_alloc(lw.bytes);
    void* ds = device_alloc(lsw.bytes);
    void* diw = device_alloc(liw.bytes);
    void* dis = device_alloc(lis.bytes);
    void* dc1 = device_alloc(M * N * 2);
    void* dc2 = device_alloc(M * N * 2);
    upload(da, a.data(), M * K, dev.stream);
    upload(das, asc.data(), M * nkb * 4, dev.stream);
    upload(dw, wbuf.data(), lw.bytes, dev.stream);
    upload(ds, swbuf.data(), lsw.bytes, dev.stream);
    upload(diw, iwst.data(), liw.bytes, dev.stream);
    upload(dis, isst.data(), lis.bytes, dev.stream);
    if (rad_stream_sync(dev.stream) != 0)
        throw std::runtime_error(std::string("rad_stream_sync: ") + rad_dev_last_error());

    RadTensor tiqr[5] = {
        tensor(da, RAD_I8, M, K), tensor(das, RAD_F32, M, nkb),
        stored(dw, lw), stored(ds, lsw), tensor(dc1, RAD_BF16, M, N),
    };
    RadTensor ti8[5] = {
        tensor(da, RAD_I8, M, K), tensor(das, RAD_F32, M, nkb),
        stored(diw, liw), stored(dis, lis), tensor(dc2, RAD_BF16, M, N),
    };
    RadArgs aiqr{};
    aiqr.t = tiqr; aiqr.n_t = 5; aiqr.p = params; aiqr.n_p = np; aiqr.world_size = 1; aiqr.rank = 0;
    RadArgs ai8 = aiqr;
    ai8.t = ti8;
    const int64_t s1 = l.gemm_iqr_scratch(&aiqr), s2 = l.gemm_i8_scratch(&ai8);
    const int64_t scratch = s1 > s2 ? s1 : s2;
    void* dsc = scratch > 0 ? device_alloc(scratch) : nullptr;
    aiqr.scratch = dsc; aiqr.scratch_bytes = scratch;
    ai8.scratch = dsc; ai8.scratch_bytes = scratch;

    expect(l.gemm_iqr(&aiqr, dev.stream) == 0, std::string(f.str(e.name)) + ": the IQR GEMM runs");
    expect(l.gemm_i8(&ai8, dev.stream) == 0, std::string(f.str(e.name)) + ": the int8 GEMM runs");
    std::vector<uint16_t> y2((size_t)(M * N), 0);
    download(ybr.data(), dc1, M * N * 2, dev.stream);
    download(y2.data(), dc2, M * N * 2, dev.stream);
    if (rad_stream_sync(dev.stream) != 0)
        throw std::runtime_error(std::string("rad_stream_sync: ") + rad_dev_last_error());

    double worst = 0.0, peak = 0.0;
    for (int64_t i = 0; i < M * N; ++i) {
        const float x = r97_iqr_bf16_to_f32(ybr[(size_t)i]);
        const float z = r97_iqr_bf16_to_f32(y2[(size_t)i]);
        const double err = std::fabs((double)x - (double)z);
        const double scale = std::fabs((double)z) > 1.0 ? std::fabs((double)z) : 1.0;
        if (err / scale > worst) worst = err / scale;
        if (std::fabs((double)z) > peak) peak = std::fabs((double)z);
    }
    expect(peak > 0.0, std::string(f.str(e.name)) + ": the GEMMs wrote a non-zero output");
    std::printf("gemm %s: worst relative error %.3g, peak %.3g\n", f.str(e.name), worst, peak);
    expect(worst <= 1.0 / 128.0,
           std::string(f.str(e.name)) + ": the int8 GEMM reproduces the IQR GEMM (" +
               std::to_string(worst) + ")");

    device_free(da); device_free(das); device_free(dw); device_free(ds);
    device_free(diw); device_free(dis); device_free(dc1); device_free(dc2); device_free(dsc);
}

}  // namespace

R97_MODEL_TEST(iqr_decode) {
    // A model test resolves the container before anything else: a machine without it skips here,
    // before the card guard takes a card for a run that would decode nothing.
#ifdef R97_MODEL_RAD
    const std::string rad = r97test::container_or_skip(R97_MODEL_RAD);
#else
    const std::string rad = r97test::container_or_skip("");
#endif

    r97test::GuardOptions o;
    const char* device_env = std::getenv("R97_DEVICE");
    o.device = r97test::parse_device(device_env ? device_env : "0");
    o.need_vram = int64_t(4) << 30;
    o.bdf = r97test::hip_bdf(o.device);
    r97test::DeviceGuard guard(o);
    if (rad_dev_count() == 0) throw std::runtime_error("no device: the gpu tier needs a card");
    if (rad_dev_set(o.device) != 0)
        throw std::runtime_error(std::string("rad_dev_set: ") + rad_dev_last_error());

    const std::string plugin = r97::device_check::plugin_path();
    void* h = dlopen(plugin.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) throw std::runtime_error(std::string("dlopen: ") + dlerror());

    rad::RadFile f;
    if (f.open(rad.c_str()) != RAD_OK) throw std::runtime_error("open " + rad);

    // One tensor per form the container stores, and the linear kinds around them: the trunk's
    // mad trellises at each code width and run, the head's two-level e4m3 scale, and the head-fed
    // bf16 scale. A chunk that starts inside the tensor and the short last chunk exercise the seam
    // and the chunked decode the vocabulary projection needs.
    const char* names[] = {
        "blk.0.ffn_gate_up.weight",   // enc 10: u5, T=512, f32 row, scale.col
        "blk.8.ffn_gate_up.weight",   // enc 12: u3, T=512, f32 row, scale.col
        "blk.56.ffn_gate_up.weight",  // enc 13: u5, T=512, f32 row
        "blk.0.ffn_down.weight",      // enc 9: u4, T=1024, f32 row, scale.col
        "blk.3.attn_k.weight",        // enc 5: u4, T=256, f32 row, scale.col
        "blk.0.ssm_inz.weight",       // enc 9, the ssm input projection
        "blk.3.attn_qg.weight",       // enc 9, the q/g projection
        "blk.3.attn_v.weight",        // enc 5, the v projection
        "blk.0.ssm_out.weight",       // enc 11: u5, T=256, bf16 per 128, mad
        "blk.3.attn_output.weight",   // enc 11, the attention output projection
        "token_embd.weight",          // enc 8: u4, T=256, e4m3 per 128, scale.1, L=12
        "output.k3.weight",           // enc 4: u3, T=256, f32 row, scale.col
        "output.k5.weight",           // enc 6: u5, T=256
        "output.k6.weight",           // enc 7: u6, T=256
    };
    for (const char* n : names) {
        const RadFileEntry* e = f.find(n);
        if (!e) continue;
        const int64_t T = f.encoding(*e).plane[plane_of(f.encoding(*e), "trellis")].block[0];
        int64_t rows = T > 512 ? T : (512 / T) * T;
        if (rows <= 0) rows = T;
        if (rows > e->shape[0]) rows = e->shape[0];
        const int64_t row0 = e->shape[0] > 2 * rows ? T : 0;
        decode_case(plugin, f, *e, row0, rows);
        if (e->shape[0] > rows)
            decode_case(plugin, f, *e, e->shape[0] - rows, rows);  // the short last run
    }

    // The head chunks the projection decodes: 130560 rows in the k3..k6 pieces.
    for (const char* n : { "output.k3.weight", "output.k6.weight" }) {
        const RadFileEntry* e = f.find(n);
        if (!e) continue;
        decode_case(plugin, f, *e, 0, 512);
        decode_case(plugin, f, *e, e->shape[0] - 256, 256);
    }

    // The one affine form in the container: the drafter's head, u2 codes under a u8 zero and an f16
    // scale the 128 columns, 248320 rows of them. It decodes in chunks like the vocabulary
    // projection, so a chunk and the short last chunk are both checked.
    const RadFileEntry* draft = f.find("dflash.draft_head.weight");
    if (draft) {
        decode_case(plugin, f, *draft, 0, 8192);
        decode_case(plugin, f, *draft, draft->shape[0] - 512, 512);
    }

    const RadFileEntry* big = f.find("blk.0.ffn_gate_up.weight");
    if (big) time_case(plugin, f, *big, 0, "ffn_gate_up");
    const RadFileEntry* down = f.find("blk.0.ffn_down.weight");
    if (down) time_case(plugin, f, *down, 0, "ffn_down");
    const RadFileEntry* inz = f.find("blk.0.ssm_inz.weight");
    if (inz) time_case(plugin, f, *inz, 0, "ssm_inz");
    const RadFileEntry* out = f.find("blk.0.ssm_out.weight");
    if (out) time_case(plugin, f, *out, 0, "ssm_out");
    for (const char* name : { "blk.3.attn_qg.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight",
                              "blk.3.attn_output.weight" }) {
        const RadFileEntry* e = f.find(name);
        if (e) time_case(plugin, f, *e, 0, name + 6);  // "attn_..." without "blk.3."
    }
    const RadFileEntry* head = f.find("output.k3.weight");
    if (head) time_case(plugin, f, *head, 256, "output.k3 head chunk");
    const RadFileEntry* emb = f.find("token_embd.weight");
    if (emb) time_case(plugin, f, *emb, 256, "token_embd head chunk");
    if (draft) time_case(plugin, f, *draft, 8192, "draft head chunk");

    dlclose(h);

    L4d l;
    load_l4d(l, plugin);
    const RadFileEntry* g = f.find("blk.0.ssm_out.weight");
    if (!g) g = f.find("blk.0.ffn_gate_up.weight");
    if (g) gemm_case(l, plugin, f, *g);
}
