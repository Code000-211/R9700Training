// tests/gpu/iqr_decode_range.cpp -- the affine grid's run is one row, so its launch takes a row
// as a blockIdx.y. A chunk of the drafter's head has more runs than the card's grid.y holds; the
// kernel that mapped a run to a block used to launch nothing over that limit and still return 0.
// This case decodes a half-sized chunk of that head's rows and holds the card to the host decode.
#include "support/device_check.h"
#include "support/gpu_guard.h"
#include "support/test.h"

#include "iqr_decode.h"
#include "rad_device.h"

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

using DecodeFn = int (*)(const R97IqrDecodeDesc*, R97IqrStream);

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

uint32_t next(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

}  // namespace

R97_TEST(iqr_decode_range) {
    r97test::GuardOptions o = r97::device_check::guard_options(int64_t(1) << 30);
    r97test::DeviceGuard guard(o);
    r97::device_check::require_device(o.device);

    // Half of the drafter head's 248320 rows: 124160 affine runs, nearly twice the card's 65536
    // grid.y. The width stays small because it is the range, not the tensor, this case is about.
    const int64_t rows_all = 248320;
    const int64_t row0 = rows_all / 2;
    const int64_t rows = rows_all - row0;
    const int64_t cols = 256;
    const int64_t nkb = cols / 128;
    const int64_t crow = cols * 2 / 8;
    const int64_t zrow = nkb;
    const int64_t srow = nkb * 2;

    std::vector<uint8_t> codes((size_t)(rows * crow));
    std::vector<uint8_t> zero((size_t)(rows * zrow));
    std::vector<uint8_t> scale((size_t)(rows * srow));
    uint32_t seed = 0x9e3779b9u;
    for (uint8_t& b : codes) b = (uint8_t)next(seed);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t k = 0; k < zrow; ++k)
            zero[(size_t)(r * zrow + k)] = (uint8_t)(next(seed) & 3u);
        for (int64_t k = 0; k < nkb; ++k) {
            // f16 1.0, 0.5, 2.0 and 1.5: exact scalars, whatever the decoder rounds to.
            const uint16_t f16[4] = { 0x3c00, 0x3800, 0x4000, 0x3e00 };
            const uint16_t v = f16[next(seed) & 3u];
            std::memcpy(&scale[(size_t)((r * nkb + k) * 2)], &v, 2);
        }
    }

    R97IqrDecodeDesc hd{};
    hd.rows_all = rows_all;
    hd.cols = cols;
    hd.row0 = row0;
    hd.rows = rows;
    hd.affine = 1;
    hd.bits = 2;
    hd.run_rows = 1;
    hd.rb = 1;
    hd.codes = codes.data();
    hd.codes_row_bytes = crow;
    hd.zero = zero.data();
    hd.zero_bits = 8;
    hd.zero_row_bytes = zrow;
    hd.scale = scale.data();
    hd.scale_dtype = R97_IQR_F16;
    hd.scale_block_cols = 128;
    hd.scale_row_bytes = srow;

    std::vector<int8_t> want((size_t)(rows * cols));
    std::vector<uint16_t> wsc((size_t)(rows * nkb));
    hd.out_codes = want.data();
    hd.out_scale = wsc.data();
    expect(r97_iqr_validate(hd) == 0, "the host description is valid");
    expect(r97_iqr_decode_host(hd) == 0, "the host decode");

    const std::string plugin = r97::device_check::plugin_path();
    void* h = dlopen(plugin.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) throw std::runtime_error(std::string("dlopen: ") + dlerror());
    DecodeFn dec = (DecodeFn)dlsym(h, "r97_iqr_decode");
    if (!dec) throw std::runtime_error("the device plugin has no IQR decode");

    Device dev;
    if (rad_stream_create(&dev.stream, 0) != 0)
        throw std::runtime_error(std::string("rad_stream_create: ") + rad_dev_last_error());

    const int64_t code_bytes = (int64_t)codes.size();
    const int64_t zero_bytes = (int64_t)zero.size();
    const int64_t scale_bytes = (int64_t)scale.size();
    const int64_t out_bytes = rows * cols;
    const int64_t out_scale_bytes = rows * nkb * 2;

    void* dcodes = device_alloc(code_bytes);
    void* dzero = device_alloc(zero_bytes);
    void* dscale = device_alloc(scale_bytes);
    void* dout = device_alloc(out_bytes);
    void* dsc = device_alloc(out_scale_bytes);

    upload(dcodes, codes.data(), code_bytes, dev.stream);
    upload(dzero, zero.data(), zero_bytes, dev.stream);
    upload(dscale, scale.data(), scale_bytes, dev.stream);
    // A sentinel makes a launch that writes nothing visible even if every comparison value happens
    // to be a plausible decode.
    if (rad_memset_async(dout, 0x5a, out_bytes, dev.stream) != 0 ||
        rad_memset_async(dsc, 0x5a, out_scale_bytes, dev.stream) != 0)
        throw std::runtime_error(std::string("rad_memset_async: ") + rad_dev_last_error());
    if (rad_stream_sync(dev.stream) != 0)
        throw std::runtime_error(std::string("rad_stream_sync: ") + rad_dev_last_error());

    R97IqrDecodeDesc dd = hd;
    dd.codes = (const uint8_t*)dcodes;
    dd.zero = (const uint8_t*)dzero;
    dd.scale = (const uint8_t*)dscale;
    dd.out_codes = (int8_t*)dout;
    dd.out_scale = (uint16_t*)dsc;

    expect(dec(&dd, dev.stream) == 0, "the device decode starts");
    if (rad_stream_sync(dev.stream) != 0)
        throw std::runtime_error(std::string("rad_stream_sync: ") + rad_dev_last_error());

    std::vector<int8_t> got((size_t)out_bytes);
    std::vector<uint16_t> gsc((size_t)(rows * nkb));
    download(got.data(), dout, out_bytes, dev.stream);
    download(gsc.data(), dsc, out_scale_bytes, dev.stream);
    if (rad_stream_sync(dev.stream) != 0)
        throw std::runtime_error(std::string("rad_stream_sync: ") + rad_dev_last_error());
    dlclose(h);

    int64_t bad = -1;
    for (int64_t i = 0; i < out_bytes; ++i)
        if (got[(size_t)i] != want[(size_t)i]) {
            bad = i;
            break;
        }
    if (bad >= 0) {
        char msg[256];
        std::snprintf(msg, sizeof msg,
                      "codes differ at row %lld of [%lld, %lld), column %lld: device %d, host %d",
                      (long long)(row0 + bad / cols), (long long)row0, (long long)(row0 + rows),
                      (long long)(bad % cols), (int)got[(size_t)bad], (int)want[(size_t)bad]);
        throw std::runtime_error(msg);
    }
    for (int64_t i = 0; i < rows * nkb; ++i)
        expect(gsc[(size_t)i] == wsc[(size_t)i], "the scales differ from the host's");

    std::printf("iqr_decode_range: %lld affine rows starting at %lld: device == host\n",
                (long long)rows, (long long)row0);

    device_free(dcodes);
    device_free(dzero);
    device_free(dscale);
    device_free(dout);
    device_free(dsc);
}
