// tests/host/adjoint_qk.cpp -- the qk family's f64 twins against radiance's host kernels, and
// their adjoints against central differences of those twins.
#include "../../src/adjoint/host/qk.h"

#include "../support/fixture.h"
#include "../support/test.h"

#include "../../third_party/radiance/abi/rad_abi.h"
#include "../../third_party/radiance/abi/rad_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using r97::adjoint::qk::KvStoreShape;
using r97::adjoint::qk::RopeShape;
using r97::adjoint::qk::RopeTable;

constexpr double kTinyTheta = 10000.0;

struct Rng {
    uint64_t state = 0x9e3779b97f4a7c15ull;
    double next() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return ((double)(state >> 11) / 4503599627370496.0) * 2.0 - 1.0;
    }
};

RadTensor make_tensor(void* data, uint32_t dtype, std::vector<int64_t> shape) {
    RadTensor t{};
    t.data = data;
    t.dtype = dtype;
    t.rank = (uint32_t)shape.size();
    if (t.rank > RAD_MAX_RANK) throw std::runtime_error("test: rank too large");
    for (size_t i = 0; i < shape.size(); ++i) t.shape[i] = shape[i];
    rad_tensor_pack(&t);
    return t;
}

const RadKernelInfo* host_kernel(const char* op) {
    static void* handle = nullptr;
    if (!handle) {
        const std::string path = r97::detail::radiance_home() + "/kernels/libref.so";
        handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) throw std::runtime_error("test: cannot load libref: " + std::string(dlerror()));
    }
    auto count = reinterpret_cast<int (*)()>(dlsym(handle, "rad_kernel_count"));
    auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(handle, "rad_kernel_at"));
    if (!count || !at) throw std::runtime_error("test: libref has no kernel table");
    for (int i = 0; i < count(); ++i) {
        const RadKernelInfo* k = at(i);
        if (k && k->op && std::strcmp(k->op, op) == 0) return k;
    }
    throw std::runtime_error(std::string("test: no host kernel for ") + op);
}

int run_kernel(const char* op, std::vector<RadTensor> tensors, std::vector<RadParam> params) {
    const RadKernelInfo* k = host_kernel(op);
    RadArgs args{};
    args.t = tensors.data();
    args.n_t = (uint32_t)tensors.size();
    args.p = params.data();
    args.n_p = (uint32_t)params.size();
    return k->launch(&args, nullptr);
}

void close_enough(const char* label, const std::vector<double>& got,
                  const std::vector<double>& want, double atol, double rtol) {
    R97_CHECK(got.size() == want.size());
    for (size_t i = 0; i < got.size(); ++i) {
        const double tol = atol + rtol * std::max(std::fabs(got[i]), std::fabs(want[i]));
        if (!(std::fabs(got[i] - want[i]) <= tol)) {
            std::fprintf(stderr, "%s: at %zu got %.17g want %.17g (tol %.3g)\n", label, i,
                         got[i], want[i], tol);
            R97_CHECK(false);
        }
    }
}

template <typename Fn>
void check_gradient(const char* label, const std::vector<double>& x,
                    const std::vector<double>& analytic, Fn scalar) {
    R97_CHECK(x.size() == analytic.size());
    std::vector<double> xp = x, xm = x;
    for (size_t i = 0; i < x.size(); ++i) {
        const double h = 1e-6 * std::max(1.0, std::fabs(x[i]));
        xp[i] = x[i] + h;
        xm[i] = x[i] - h;
        const double fd = (scalar(xp) - scalar(xm)) / (2.0 * h);
        xp[i] = x[i];
        xm[i] = x[i];
        const double scale = std::max({1.0, std::fabs(fd), std::fabs(analytic[i])});
        const double err = std::fabs(fd - analytic[i]) / scale;
        if (!(err <= 1e-6)) {
            std::fprintf(stderr, "%s: at %zu fd %.17g analytic %.17g rel %.3g\n", label, i, fd,
                         analytic[i], err);
            R97_CHECK(false);
        }
    }
}

// ------------------------------------------------------------------ rope

struct RopeCase {
    int64_t tokens;
    int64_t n_head;
    int64_t n_head_kv;
    int64_t head_dim;
    int64_t rotary_dim;
    double theta;
    double scale;
    bool bf16;
};

std::vector<int32_t> rope_positions(int64_t tokens) {
    std::vector<int32_t> p((size_t)tokens);
    for (int64_t t = 0; t < tokens; ++t) p[(size_t)t] = (int32_t)(2 * t + (t % 3));
    return p;
}

void check_rope(const RopeCase& c) {
    const RopeShape shape{c.tokens, c.n_head, c.n_head_kv, c.head_dim, c.rotary_dim};
    const int64_t row = (c.n_head + 2 * c.n_head_kv) * c.head_dim;
    const int64_t n = shape.tokens * row;
    const std::vector<int32_t> positions = rope_positions(c.tokens);
    int32_t max_pos = 0;
    for (int32_t p : positions) max_pos = std::max(max_pos, p);
    const int64_t table_rows = (int64_t)max_pos + 1;
    const RopeTable table = r97::adjoint::qk::rope_table_f64(
        table_rows, c.rotary_dim, c.theta, c.scale);

    // The table itself is rope_table's host kernel, position for position.
    {
        std::vector<int32_t> all((size_t)table_rows);
        std::iota(all.begin(), all.end(), 0);
        std::vector<float> host_cs((size_t)(table_rows * c.rotary_dim));
        RadTensor tp = make_tensor(all.data(), RAD_I32, {table_rows});
        RadTensor tc = make_tensor(host_cs.data(), RAD_F32, {table_rows, c.rotary_dim});
        R97_CHECK(run_kernel("rope_table", {tp, tc},
                             {RAD_INT("M", table_rows), RAD_INT("rot", c.rotary_dim),
                              RAD_F64("theta", c.theta), RAD_F64("scale", c.scale),
                              RAD_STR("dtype", "f32")}) == RAD_OK);
        std::vector<double> got(host_cs.begin(), host_cs.end());
        close_enough("rope_table", got, table.cos_sin, 2e-5, 2e-5);
    }

    // The same input through the host kernel. A bf16 case stores bf16, so its input is narrowed
    // first and the twin sees exactly that value; the comparison is then a bf16 tolerance.
    Rng rng;
    std::vector<double> src((size_t)n);
    for (double& v : src) v = rng.next();
    if (c.bf16)
        for (double& v : src)
            v = (double)rad_bf16_to_f32(rad_f32_to_bf16((float)v));

    std::vector<double> twin((size_t)n);
    r97::adjoint::qk::rope_forward_f64(src.data(), positions.data(), shape, table, twin.data());

    std::vector<double> host_out;
    if (c.bf16) {
        std::vector<uint16_t> host((size_t)n);
        for (int64_t i = 0; i < n; ++i) host[(size_t)i] = rad_f32_to_bf16((float)src[(size_t)i]);
        std::vector<int32_t> host_pos = positions;
        RadTensor t = make_tensor(host.data(), RAD_BF16, {shape.tokens, row});
        RadTensor tp = make_tensor(host_pos.data(), RAD_I32, {shape.tokens});
        R97_CHECK(run_kernel("rope", {t, tp},
                             {RAD_INT("M", shape.tokens), RAD_INT("head_dim", c.head_dim),
                              RAD_INT("n_head", c.n_head),
                              RAD_INT("n_head_kv", c.n_head_kv), RAD_F64("theta", c.theta),
                              RAD_F64("scale", c.scale), RAD_INT("rotary_dim", c.rotary_dim),
                              RAD_STR("mode", "neox")}) == RAD_OK);
        host_out.resize((size_t)n);
        for (int64_t i = 0; i < n; ++i)
            host_out[(size_t)i] = (double)rad_bf16_to_f32(host[(size_t)i]);
    } else {
        std::vector<float> host((size_t)n);
        for (int64_t i = 0; i < n; ++i) host[(size_t)i] = (float)src[(size_t)i];
        std::vector<int32_t> host_pos = positions;
        RadTensor t = make_tensor(host.data(), RAD_F32, {shape.tokens, row});
        RadTensor tp = make_tensor(host_pos.data(), RAD_I32, {shape.tokens});
        R97_CHECK(run_kernel("rope", {t, tp},
                             {RAD_INT("M", shape.tokens), RAD_INT("head_dim", c.head_dim),
                              RAD_INT("n_head", c.n_head),
                              RAD_INT("n_head_kv", c.n_head_kv), RAD_F64("theta", c.theta),
                              RAD_F64("scale", c.scale), RAD_INT("rotary_dim", c.rotary_dim),
                              RAD_STR("mode", "neox")}) == RAD_OK);
        host_out.assign(host.begin(), host.end());
    }
    close_enough(c.bf16 ? "rope bf16" : "rope f32", host_out, twin, c.bf16 ? 2e-2 : 2e-5,
                 c.bf16 ? 2e-2 : 2e-5);

    // Central differences of the twin, seeded with a fixed cotangent.
    std::vector<double> seed((size_t)n);
    for (double& v : seed) v = rng.next();
    std::vector<double> adjoint((size_t)n);
    r97::adjoint::qk::rope_adjoint_f64(seed.data(), positions.data(), shape, table,
                                      adjoint.data());
    auto loss = [&](const std::vector<double>& x) {
        std::vector<double> y(x.size());
        r97::adjoint::qk::rope_forward_f64(x.data(), positions.data(), shape, table, y.data());
        double sum = 0.0;
        for (size_t i = 0; i < y.size(); ++i) sum += seed[i] * y[i];
        return sum;
    };
    check_gradient("rope", src, adjoint, loss);
}

// ------------------------------------------------------------------ kv_store

struct KvCase {
    int64_t tokens;
    int64_t kv_heads;
    int64_t head_dim;
    int64_t block_size;
    int64_t n_blocks;
    bool bf16_pool;
    bool pad_last;
};

void check_kv_store(const KvCase& c) {
    const KvStoreShape shape{c.tokens, c.kv_heads, c.head_dim, c.block_size, c.n_blocks};
    const int64_t pages = (c.tokens + c.block_size - 1) / c.block_size;
    R97_CHECK(pages <= c.n_blocks);
    std::vector<int32_t> block_table((size_t)pages);
    for (int64_t i = 0; i < pages; ++i) block_table[(size_t)i] = (int32_t)(pages - 1 - i);
    std::vector<int32_t> slot((size_t)c.tokens);
    for (int64_t t = 0; t < c.tokens; ++t)
        slot[(size_t)t] = (int32_t)r97::adjoint::qk::kv_slot(
            block_table.data(), pages, c.block_size, 0, t);
    if (c.pad_last) slot.back() = -1;

    const int64_t head_values = c.kv_heads * c.head_dim;
    const int64_t n = c.tokens * head_values;
    const int64_t pool_n = c.n_blocks * c.kv_heads * c.block_size * 2 * c.head_dim;
    Rng rng;
    std::vector<double> k((size_t)n), v((size_t)n);
    for (double& x : k) x = rng.next();
    for (double& x : v) x = rng.next();

    // Forward: the twin's pool against kv_store's.
    std::vector<double> twin_pool((size_t)pool_n, 0.0);
    r97::adjoint::qk::kv_store_forward_f64(k.data(), v.data(), slot.data(), shape,
                                          twin_pool.data());
    std::vector<double> host_pool((size_t)pool_n, 0.0);
    if (c.bf16_pool) {
        std::vector<uint16_t> pool((size_t)pool_n, 0);
        std::vector<float> kf(k.begin(), k.end()), vf(v.begin(), v.end());
        RadTensor kt = make_tensor(kf.data(), RAD_F32, {c.tokens, c.kv_heads, c.head_dim});
        RadTensor vt = make_tensor(vf.data(), RAD_F32, {c.tokens, c.kv_heads, c.head_dim});
        RadTensor st = make_tensor(slot.data(), RAD_I32, {c.tokens});
        RadTensor pt = make_tensor(pool.data(), RAD_BF16,
                                   {c.n_blocks, c.kv_heads, c.block_size, 2 * c.head_dim});
        R97_CHECK(run_kernel("kv_store", {kt, vt, st, pt},
                             {RAD_INT("head_dim", c.head_dim),
                              RAD_INT("block_size", c.block_size),
                              RAD_INT("n_head_kv", c.kv_heads)}) == RAD_OK);
        for (int64_t i = 0; i < pool_n; ++i)
            host_pool[(size_t)i] = (double)rad_bf16_to_f32(pool[(size_t)i]);
    } else {
        std::vector<float> pool((size_t)pool_n, 0.0f);
        std::vector<float> kf(k.begin(), k.end()), vf(v.begin(), v.end());
        RadTensor kt = make_tensor(kf.data(), RAD_F32, {c.tokens, c.kv_heads, c.head_dim});
        RadTensor vt = make_tensor(vf.data(), RAD_F32, {c.tokens, c.kv_heads, c.head_dim});
        RadTensor st = make_tensor(slot.data(), RAD_I32, {c.tokens});
        RadTensor pt = make_tensor(pool.data(), RAD_F32,
                                   {c.n_blocks, c.kv_heads, c.block_size, 2 * c.head_dim});
        R97_CHECK(run_kernel("kv_store", {kt, vt, st, pt},
                             {RAD_INT("head_dim", c.head_dim),
                              RAD_INT("block_size", c.block_size),
                              RAD_INT("n_head_kv", c.kv_heads)}) == RAD_OK);
        host_pool.assign(pool.begin(), pool.end());
    }
    close_enough(c.bf16_pool ? "kv_store bf16" : "kv_store f32", host_pool, twin_pool,
                 c.bf16_pool ? 2e-2 : 2e-5, c.bf16_pool ? 2e-2 : 2e-5);

    // Adjoint: the pool cotangent is laid out by (sequence, position), then routed back.
    std::vector<double> d_pool((size_t)pool_n, 0.0);
    for (int64_t t = 0; t < c.tokens; ++t) {
        if (slot[(size_t)t] < 0) continue;
        for (int64_t h = 0; h < c.kv_heads; ++h) {
            const int64_t block = slot[(size_t)t] / c.block_size;
            const int64_t off = slot[(size_t)t] % c.block_size;
            const int64_t base =
                ((block * c.kv_heads + h) * c.block_size + off) * 2 * c.head_dim;
            for (int64_t d = 0; d < c.head_dim; ++d) {
                d_pool[(size_t)(base + d)] = rng.next();
                d_pool[(size_t)(base + c.head_dim + d)] = rng.next();
            }
        }
    }
    std::vector<double> d_k((size_t)n), d_v((size_t)n);
    r97::adjoint::qk::kv_store_adjoint_f64(d_pool.data(), slot.data(), shape, d_k.data(),
                                          d_v.data());
    auto loss = [&](const std::vector<double>& kk, const std::vector<double>& vv) {
        std::vector<double> pool((size_t)pool_n, 0.0);
        r97::adjoint::qk::kv_store_forward_f64(kk.data(), vv.data(), slot.data(), shape,
                                              pool.data());
        double sum = 0.0;
        for (int64_t i = 0; i < pool_n; ++i) sum += d_pool[(size_t)i] * pool[(size_t)i];
        return sum;
    };
    check_gradient("kv_store k", k, d_k,
                   [&](const std::vector<double>& kk) { return loss(kk, v); });
    check_gradient("kv_store v", v, d_v,
                   [&](const std::vector<double>& vv) { return loss(k, vv); });
}

// ------------------------------------------------------------------ the gate

void check_sigmoid(int64_t n) {
    Rng rng;
    std::vector<double> x((size_t)n), y((size_t)n);
    for (double& v : x) v = 4.0 * rng.next();
    r97::adjoint::qk::sigmoid_forward_f64(x.data(), n, y.data());

    std::vector<float> host_x(x.begin(), x.end()), host_y((size_t)n);
    RadTensor xt = make_tensor(host_x.data(), RAD_F32, {n});
    RadTensor yt = make_tensor(host_y.data(), RAD_F32, {n});
    R97_CHECK(run_kernel("sigmoid", {xt, yt},
                         {RAD_INT("M", 1), RAD_INT("n", n), RAD_STR("dtype", "f32")}) == RAD_OK);
    std::vector<double> host(host_y.begin(), host_y.end());
    close_enough("sigmoid host", host, y, 2e-6, 2e-6);

    std::vector<double> seed((size_t)n), d_x((size_t)n);
    for (double& v : seed) v = rng.next();
    r97::adjoint::qk::sigmoid_adjoint_f64(y.data(), seed.data(), n, d_x.data());
    auto loss = [&](const std::vector<double>& xx) {
        std::vector<double> yy((size_t)n);
        r97::adjoint::qk::sigmoid_forward_f64(xx.data(), n, yy.data());
        double sum = 0.0;
        for (int64_t i = 0; i < n; ++i) sum += seed[(size_t)i] * yy[(size_t)i];
        return sum;
    };
    check_gradient("sigmoid", x, d_x, loss);
}

void check_mul(int64_t rows, int64_t n, bool bcast) {
    Rng rng;
    const int64_t an = rows * n, bn = bcast ? n : rows * n;
    std::vector<double> a((size_t)an), b((size_t)bn), y((size_t)an);
    for (double& v : a) v = rng.next();
    for (double& v : b) v = rng.next();
    r97::adjoint::qk::mul_forward_f64(a.data(), b.data(), rows, n, bcast, y.data());

    std::vector<float> host_a(a.begin(), a.end()), host_b(b.begin(), b.end()),
        host_y((size_t)an);
    RadTensor at = make_tensor(host_a.data(), RAD_F32, {rows, n});
    RadTensor bt = make_tensor(host_b.data(), RAD_F32, bcast ? std::vector<int64_t>{n}
                                                             : std::vector<int64_t>{rows, n});
    RadTensor yt = make_tensor(host_y.data(), RAD_F32, {rows, n});
    R97_CHECK(run_kernel("mul", {at, bt, yt},
                         {RAD_INT("M", rows), RAD_INT("n", n), RAD_STR("dtype", "f32")}) ==
              RAD_OK);
    std::vector<double> host(host_y.begin(), host_y.end());
    close_enough("mul host", host, y, 2e-6, 2e-6);

    std::vector<double> seed((size_t)an), d_a((size_t)an), d_b((size_t)bn);
    for (double& v : seed) v = rng.next();
    r97::adjoint::qk::mul_adjoint_f64(seed.data(), a.data(), b.data(), rows, n, bcast,
                                     d_a.data(), d_b.data());
    auto loss = [&](const std::vector<double>& aa, const std::vector<double>& bb) {
        std::vector<double> yy((size_t)an);
        r97::adjoint::qk::mul_forward_f64(aa.data(), bb.data(), rows, n, bcast, yy.data());
        double sum = 0.0;
        for (int64_t i = 0; i < an; ++i) sum += seed[(size_t)i] * yy[(size_t)i];
        return sum;
    };
    check_gradient("mul a", a, d_a,
                   [&](const std::vector<double>& aa) { return loss(aa, b); });
    check_gradient("mul b", b, d_b,
                   [&](const std::vector<double>& bb) { return loss(a, bb); });
}

}  // namespace

R97_TEST(adjoint_qk_rope) {
    // The tiny fixture's attention geometry: 4 query heads, 2 kv heads, head_dim 32, half rotary.
    check_rope({5, 4, 2, 32, 16, kTinyTheta, 1.0, true});
    // The host issues the q and k halves separately, each with n_head_kv = 0.
    check_rope({5, 4, 0, 32, 16, kTinyTheta, 1.0, false});
    check_rope({5, 2, 0, 32, 16, kTinyTheta, 1.0, false});
    check_rope({3, 3, 1, 8, 4, 1e6, 1.0, false});
    check_rope({7, 2, 2, 16, 16, kTinyTheta, 2.0, false});
}

R97_TEST(adjoint_qk_kv_store) {
    check_kv_store({5, 2, 32, 4, 3, true, false});  // tiny fixture: 5 tokens, 2 kv heads, hd 32
    check_kv_store({7, 3, 16, 8, 2, false, true});
    check_kv_store({9, 1, 8, 3, 4, false, false});
}

R97_TEST(adjoint_qk_gate) {
    check_sigmoid(5 * 4 * 32);  // the tiny fixture's query width
    check_sigmoid(3 * 3 * 8);
    check_sigmoid(7 * 2 * 16);
    check_mul(5, 128, false);  // same-shaped gate, as the attention block issues it
    check_mul(3, 24, false);
    check_mul(7, 32, true);  // mul's one broadcast rule
}
