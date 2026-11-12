// tests/gpu/lora_shrink_step_order_r4d.cpp -- libr4d's lora_shrink reads a step-order
// activation, on the card.
//
// the serving patch `0210` replaces the RAD_E_UNSUPPORTED refusal that libr4d's (and
// libhalo's) lora_shrink once raised with the read itself: when `gemm_dtype`
// names a GEMM whose `quant_act_i8` wrote the codes in the int8 WMMA's step order (w4a8/w8a8/
// w2a8), each column's codes are multiplied by the lora_a factor of that column. The i8a8h
// trunk -- codes in column order -- keeps the plain reading: the gate is the GEMM's encoding,
// not the activation's dtype.
//
// The reference is the f64 dot of the natural codes, each times its group's scale and the pool
// factor of its column, one row at a time (the host kernel of the same op,
// tests/host/lora_shrink_step_order.cpp, already reads either order and is the cross-check the
// tolerance below is loose enough for). Both card rows are held: lora_shrink_scalar and
// lora_shrink_wmma_q (the i8 codes' tiled row), at a geometry the wmma split keeps at one block
// (K 128) and at one that splits K over blocks with its scratch partials (K 2048). A base row
// (slot -1) is left unwritten, and a buffer held in column order but called with a step-order
// `gemm_dtype` must come out wrong: the gate, not the dtype, is what selects the reading.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "rad_abi.h"
#include "rad_device.h"

#include "support/device_check.h"
#include "support/gpu_guard.h"
#include "support/test.h"

namespace {

using namespace r97;  // device_check::guard_options and friends

// The int8 WMMA's step order (libref/ref_common.h, i8_step_pos): the middle two dwords of every
// 16-byte step exchanged. Self-inverse.
int64_t step_pos(int64_t c) {
    return (c & ~int64_t(12)) | ((c & 4) << 1) | ((c & 8) >> 1);
}

uint16_t bf16(float f) {
    const uint32_t u = __builtin_bit_cast(uint32_t, f);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

float bf16f(uint16_t h) { return __builtin_bit_cast(float, (uint32_t)h << 16); }

// A deterministic small value, exact in bf16: a product of two such codes is exact in f32, so
// the f32 accumulators differ from the f64 reference only by their addition order.
int8_t code_of(int m, int k) {
    const int v = ((m * 31 + k * 7) % 13) - 6;  // -6..6
    return (int8_t)v;
}
float a_of(int s, int j, int k) {
    return ((((s * 13 + j * 7 + k * 3) % 7) - 3) * 1.0f);  // -3..3
}

struct Geom {
    int64_t M, K, R, S, G;
};

struct Row {
    const RadKernelInfo* k = nullptr;
};

// Where libr4d is: R97_LIBR4D_SO, or beside this binary's radiance home kernels.
std::string r4d_so_path() {
    if (const char* e = std::getenv("R97_LIBR4D_SO"); e && *e)
        return std::string(e);
    const std::string dir = device_check::exe_dir();
    if (!dir.empty()) {
        for (const char* rel : {"/radiance-build/radiance_home/kernels/libr4d.so",
                                "/kernels/libr4d.so"}) {
            const std::string p = dir + rel;
            if (access(p.c_str(), R_OK) == 0)
                return p;
        }
    }
    throw std::runtime_error("no libr4d.so: set R97_LIBR4D_SO");
}

void load_rows(Row* scalar, Row* wmma) {
    const std::string so = r4d_so_path();
    void* h = dlopen(so.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h)
        throw std::runtime_error("dlopen(" + so + "): " + std::string(dlerror()));
    int (*kcount)(void) = (int (*)(void))dlsym(h, "rad_kernel_count");
    const RadKernelInfo* (*kat)(int) = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    if (!kcount || !kat)
        throw std::runtime_error("libr4d exports no rad_kernel_count / rad_kernel_at");
    const int n = kcount();
    for (int i = 0; i < n; ++i) {
        const RadKernelInfo* k = kat(i);
        if (!k || !k->name)
            continue;
        if (std::strcmp(k->name, "lora_shrink_scalar") == 0)
            scalar->k = k;
        else if (std::strcmp(k->name, "lora_shrink_wmma_q") == 0)
            wmma->k = k;
    }
    if (!scalar->k)
        throw std::runtime_error("libr4d has no row named lora_shrink_scalar");
    if (!wmma->k)
        throw std::runtime_error("libr4d has no row named lora_shrink_wmma_q");
}

// The natural codes (column order), a scale stream and a pool: the test's single source of truth.
struct Bufs {
    std::vector<int8_t> x_col;   // [M,K]
    std::vector<float> scale;    // [M,K/G]
    std::vector<int32_t> slot;   // [M]; the last row is a base row
    std::vector<uint16_t> la;    // [S,R,K] bf16
};

Bufs make(const Geom& g) {
    Bufs b;
    b.x_col.assign((size_t)g.M * g.K, 0);
    for (int64_t m = 0; m < g.M; ++m)
        for (int64_t c = 0; c < g.K; ++c)
            b.x_col[(size_t)(m * g.K + c)] = code_of((int)m, (int)c);
    b.scale.assign((size_t)g.M * (g.K / g.G), 0);
    for (int64_t m = 0; m < g.M; ++m)
        for (int64_t gr = 0; gr < g.K / g.G; ++gr)
            b.scale[(size_t)(m * g.K / g.G + gr)] = 0.25f + 0.125f * (float)gr;
    b.slot = {0, 0, 1, 0, -1};  // slot 1 is a second adapter, so the pool rows differ
    if ((int64_t)b.slot.size() < g.M)
        b.slot.assign((size_t)g.M, 0);
    b.la.assign((size_t)g.S * g.R * g.K, 0);
    for (int64_t s = 0; s < g.S; ++s)
        for (int64_t j = 0; j < g.R; ++j)
            for (int64_t k = 0; k < g.K; ++k)
                b.la[(size_t)(s * g.R * g.K + j * g.K + k)] = bf16(a_of((int)s, (int)j, (int)k));
    return b;
}

// The f64 reference over the natural codes. The base row stays at `sentinel`.
std::vector<double> reference(const Bufs& b, const Geom& g, double sentinel) {
    std::vector<double> ref((size_t)g.M * g.R, sentinel);
    for (int64_t m = 0; m < g.M; ++m) {
        const int s = b.slot[m];
        if (s < 0 || s >= g.S)
            continue;
        for (int64_t j = 0; j < g.R; ++j) {
            double acc = 0.0;
            for (int64_t k = 0; k < g.K; ++k)
                acc += (double)b.x_col[(size_t)(m * g.K + k)] *
                       b.scale[(size_t)(m * g.K / g.G + k / g.G)] *
                       (double)bf16f(b.la[(size_t)(s * g.R * g.K + j * g.K + k)]);
            ref[(size_t)(m * g.R + j)] = acc;
        }
    }
    return ref;
}

// The codes stored the way quant_act_i8 writes them for a step-order site: each row permuted
// through i8_step_pos. Column order leaves the buffer untouched.
std::vector<int8_t> stored(const Bufs& b, const Geom& g, bool step) {
    if (!step)
        return b.x_col;
    std::vector<int8_t> x = b.x_col;
    for (int64_t m = 0; m < g.M; ++m) {
        const int8_t* nat = b.x_col.data() + m * g.K;
        int8_t* row = x.data() + m * g.K;
        for (int64_t c = 0; c < g.K; ++c)
            row[step_pos(c)] = nat[c];
    }
    return x;
}

// One row of the card over the stored codes of `gemm_dtype`, returning t with the base row's
// cells kept at their sentinel (-777.0f, pre-filled).
std::vector<float> run_row(const RadKernelInfo* k, const Bufs& b, const Geom& g, bool step,
                           const char* gemm_dtype, RadStream stream) {
    const std::vector<int8_t> x = stored(b, g, step);

    void* dx = rad_dev_alloc((int64_t)g.M * g.K, RAD_MEM_DEVICE);
    void* ds = rad_dev_alloc((int64_t)g.M * (g.K / g.G) * 4, RAD_MEM_DEVICE);
    void* dsl = rad_dev_alloc((int64_t)g.M * 4, RAD_MEM_DEVICE);
    void* dla = rad_dev_alloc((int64_t)g.S * g.R * g.K * 2, RAD_MEM_DEVICE);
    void* dt = rad_dev_alloc((int64_t)g.M * g.R * 4, RAD_MEM_DEVICE);
    if (!dx || !ds || !dsl || !dla || !dt)
        throw std::runtime_error("rad_dev_alloc: " + std::string(rad_dev_last_error()));
    std::vector<float> t0((size_t)g.M * g.R, -777.0f);  // the sentinel: a base row stays
    if (rad_memcpy_async(dx, x.data(), (int64_t)x.size(), stream) != 0 ||
        rad_memcpy_async(ds, b.scale.data(), (int64_t)b.scale.size() * 4, stream) != 0 ||
        rad_memcpy_async(dsl, b.slot.data(), (int64_t)b.slot.size() * 4, stream) != 0 ||
        rad_memcpy_async(dla, b.la.data(), (int64_t)b.la.size() * 2, stream) != 0 ||
        rad_memcpy_async(dt, t0.data(), (int64_t)t0.size() * 4, stream) != 0)
        throw std::runtime_error("upload: " + std::string(rad_dev_last_error()));

    RadTensor ts[5]{};
    ts[0].data = dx; ts[0].dtype = RAD_I8; ts[0].rank = 2;
    ts[0].shape[0] = g.M; ts[0].shape[1] = g.K; ts[0].stride[0] = g.K; ts[0].stride[1] = 1;
    ts[1].data = ds; ts[1].dtype = RAD_F32; ts[1].rank = 2;
    ts[1].shape[0] = g.M; ts[1].shape[1] = g.K / g.G;
    ts[1].stride[0] = g.K / g.G; ts[1].stride[1] = 1;
    ts[2].data = dsl; ts[2].dtype = RAD_I32; ts[2].rank = 1;
    ts[2].shape[0] = g.M; ts[2].stride[0] = 1;
    ts[3].data = dla; ts[3].dtype = RAD_BF16; ts[3].rank = 3;
    ts[3].shape[0] = g.S; ts[3].shape[1] = g.R; ts[3].shape[2] = g.K;
    ts[3].stride[0] = g.R * g.K; ts[3].stride[1] = g.K; ts[3].stride[2] = 1;
    ts[4].data = dt; ts[4].dtype = RAD_F32; ts[4].rank = 2;
    ts[4].shape[0] = g.M; ts[4].shape[1] = g.R; ts[4].stride[0] = g.R; ts[4].stride[1] = 1;

    const char* c_dtype = "i8";
    RadParam ps[] = {
        RAD_INT("K", g.K), RAD_INT("R", g.R), RAD_INT("S", g.S), RAD_INT("M", g.M),
        RAD_INT("group", g.G), RAD_STR("dtype", c_dtype), RAD_STR("gemm_dtype", gemm_dtype),
    };
    RadArgs a{};
    a.t = ts; a.n_t = 5;
    a.p = ps; a.n_p = (int)(sizeof ps / sizeof ps[0]);
    a.world_size = 1;
    if (k->scratch) {
        const int64_t need = k->scratch(&a);
        if (need > 0) {
            void* d = rad_dev_alloc(need, RAD_MEM_DEVICE);
            if (!d)
                throw std::runtime_error("rad_dev_alloc(scratch): " +
                                         std::string(rad_dev_last_error()));
            a.scratch = d; a.scratch_bytes = need;
        }
    }
    const int rc = k->launch(&a, stream);
    if (rc != RAD_OK)
        throw std::runtime_error(std::string(k->name) + " refused (" + rad_strerror(rc) +
                                 ") the " + gemm_dtype + " site, " +
                                 (step ? "step-order" : "column-order") + " codes, K " +
                                 std::to_string(g.K) + " R " + std::to_string(g.R));
    if (rad_stream_sync(stream) != 0)
        throw std::runtime_error("rad_stream_sync: " + std::string(rad_dev_last_error()));

    std::vector<float> t((size_t)g.M * g.R);
    if (rad_memcpy_async(t.data(), dt, (int64_t)t.size() * 4, stream) != 0 ||
        rad_stream_sync(stream) != 0)
        throw std::runtime_error("download: " + std::string(rad_dev_last_error()));
    rad_dev_free(dx, RAD_MEM_DEVICE);
    rad_dev_free(ds, RAD_MEM_DEVICE);
    rad_dev_free(dsl, RAD_MEM_DEVICE);
    rad_dev_free(dla, RAD_MEM_DEVICE);
    rad_dev_free(dt, RAD_MEM_DEVICE);
    if (a.scratch)
        rad_dev_free(a.scratch, RAD_MEM_DEVICE);
    return t;
}

// The base row's cells must stay at their sentinel; the rest are the row's t.
bool base_row_ok(const std::vector<float>& t, const Bufs& b, const Geom& g, const char* label) {
    for (int64_t m = 0; m < g.M; ++m) {
        const int s = b.slot[m];
        if (s < 0 || s >= g.S)
            for (int64_t j = 0; j < g.R; ++j)
                if (t[(size_t)(m * g.R + j)] != -777.0f)
                    throw std::runtime_error(std::string(label) + ": the base row m=" +
                                             std::to_string(m) + " was written");
    }
    return true;
}

// The heart of the check: the step-order reading and the column-order reading of the SAME row are
// one byte permutation apart in memory, so both kernels sum the same codes in the same order and
// must give bit-identical f32 cells. A wrong permutation, or a gate that unexchanges a
// column-order buffer (or misses a step-order one), breaks this.
void check_equal(const char* label, const std::vector<float>& step, const std::vector<float>& col,
                 const Bufs& b, const Geom& g) {
    base_row_ok(step, b, g, label);
    base_row_ok(col, b, g, label);
    for (size_t i = 0; i < step.size(); ++i)
        if (step[i] != -777.0f && std::memcmp(&step[i], &col[i], sizeof(float)) != 0)
            throw std::runtime_error(std::string(label) + ": cell " + std::to_string(i) +
                                     " is " + std::to_string(step[i]) + " step-order, " +
                                     std::to_string(col[i]) + " column-order -- the unexchange "
                                     "did not line the codes up by column");
}

// The value: the column-order reading (the one that has always been correct) against an f64 dot
// of the natural codes, the base row at its sentinel.
void check_reference(const char* label, const std::vector<float>& t,
                     const std::vector<double>& ref, const Bufs& b, const Geom& g) {
    base_row_ok(t, b, g, label);
    for (int64_t m = 0; m < g.M; ++m) {
        const int s = b.slot[m];
        if (s < 0 || s >= g.S)
            continue;
        for (int64_t j = 0; j < g.R; ++j) {
            const double want = ref[(size_t)(m * g.R + j)];
            const double got = t[(size_t)(m * g.R + j)];
            if (std::fabs(got - want) > 1e-3 + 1e-5 * std::fabs(want))
                throw std::runtime_error(std::string(label) + ": t[" + std::to_string(m) + "][" +
                                         std::to_string(j) + "] = " + std::to_string(got) +
                                         ", the f64 reference is " + std::to_string(want));
        }
    }
}

void run_geom(const Row& scalar, const Row& wmma, RadStream stream, const Geom& g) {
    const Bufs b = make(g);
    const std::vector<double> ref = reference(b, g, -777.0);
    const char* tag = g.K == 128 ? "K=128" : "K=2048";

    // 1. The point of the patch: step-order codes, a step-order gemm_dtype. Each must give
    //    bit-identical cells to the same row's column-order reading.
    for (const Row* row : {&scalar, &wmma}) {
        // The column-order reading of this row: the step-order readings must match it, cell for
        // cell, whatever step-order encoding the site is called under.
        const std::vector<float> col = run_row(row->k, b, g, false, "i8a8h", stream);
        for (const char* gd : {"w8a8", "w4a8", "w2a8"}) {
            const std::string label = std::string(row->k->name) + " " + gd + " " + tag;
            check_equal(label.c_str(), run_row(row->k, b, g, true, gd, stream), col, b, g);
        }
    }
    std::printf("  %s: the card rows unexchange a %s-order site to the column-order cells\n", tag,
                g.K == 128 ? "single-step" : "split-K");

    // 2. The value: the column-order reading (the i8a8h trunk, unchanged by the patch) against an
    //    f64 dot. This is what the step-order cells now equal, cell for cell.
    for (const Row* row : {&scalar, &wmma}) {
        const std::string label = std::string(row->k->name) + " i8a8h " + tag;
        check_reference(label.c_str(), run_row(row->k, b, g, false, "i8a8h", stream), ref, b, g);
    }

    // 3. The gate, not the dtype: the column-order buffer called with a step-order gemm_dtype
    //    unexchanges codes that are not permuted, so it must NOT give the plain reading's cells.
    for (const Row* row : {&scalar, &wmma}) {
        const std::vector<float> plain = run_row(row->k, b, g, false, "i8a8h", stream);
        const std::vector<float> bad = run_row(row->k, b, g, false, "w8a8", stream);
        bool differs = false;
        for (size_t i = 0; i < plain.size(); ++i)
            if (plain[i] != -777.0f && std::memcmp(&plain[i], &bad[i], sizeof(float)) != 0)
                differs = true;
        if (!differs)
            throw std::runtime_error(std::string(row->k->name) +
                                     ": column-order codes read as a step-order site gave the "
                                     "plain reading; the gate did not permute");
    }
    std::printf("  %s: the column-order trunk is untouched; a misnamed site permutes\n", tag);
}

}  // namespace

R97_TEST(lora_shrink_reads_step_order_codes_r4d) {
    using namespace r97;
    r97test::GuardOptions o = device_check::guard_options(int64_t(64) << 20);
    r97test::DeviceGuard guard(o);
    device_check::require_device(o.device);
    RadStream stream = nullptr;
    if (rad_stream_create(&stream, 0) != 0)
        throw std::runtime_error("rad_stream_create: " + std::string(rad_dev_last_error()));
    Row scalar, wmma;
    load_rows(&scalar, &wmma);

    // K 128: the wmma plan's one split (no scratch partials). K 2048, R 16: three K splits with
    // the scratch partials and the split's arrival counters.
    run_geom(scalar, wmma, stream, Geom{5, 128, 16, 2, 128});
    run_geom(scalar, wmma, stream, Geom{5, 2048, 16, 2, 128});
    rad_stream_destroy(stream);
    std::printf("  libr4d's lora_shrink rows read the step order back\n");
}
