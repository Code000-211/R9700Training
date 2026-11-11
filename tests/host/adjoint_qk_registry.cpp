// tests/host/adjoint_qk_registry.cpp -- the qk family's registered rows, driven through the
// checker rather than through the family's f64 functions (tests/host/adjoint_qk.cpp does that).
//
// `rope` asks the checker for a kernel and an adjoint at a [tokens, n_head*head_dim] q with the
// tiny fixture's mrope shape and a theta that is not the kernel's default (500000, so a twin that
// reads the wrong key falls back to 10000 and cannot match). It is an in-place op -- operand 0
// is written but also read -- so the check names that operand as in-place. `kv_store` asks for a
// [tokens, n_head_kv, head_dim] k/v and slot list into a [blocks, n_head_kv, block_size,
// 2*head_dim] pool (the k and v halves share the last dimension). Both also pin their rejection: a rope record without a head width, and a kv_store record
// with no pool write.
#include "../support/adjoint_check.h"
#include "../support/test.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using r97::AdValue;
using r97::OpRecord;
using r97::Operand;
namespace ac = r97::adjoint_check;

AdValue values(const char* dtype, std::vector<int64_t> shape, std::vector<double> v) {
    AdValue a;
    a.dtype = dtype;
    a.shape = std::move(shape);
    a.value = std::move(v);
    return a;
}

OpRecord rope_record(int64_t tokens, bool head_width) {
    OpRecord rec;
    rec.op = "rope";
    rec.block = 0;
    rec.position = 0;
    rec.n = tokens;
    rec.params = {{"M", tokens},
                  {"head_dim", int64_t(8)},
                  {"mode", std::string("mrope")},
                  {"n_head", int64_t(2)},
                  {"n_head_kv", int64_t(0)},
                  {"rotary_dim", int64_t(8)},
                  {"scale", std::string("1")},
                  {"theta", std::string("500000")}};
    if (!head_width) {
        rec.params.erase(rec.params.begin() + 1);  // head_dim
        rec.params.erase(rec.params.begin() + 2);  // n_head (now at 2)
        rec.params.erase(rec.params.begin() + 2);  // n_head_kv
    }
    rec.operands = {
        {"q", "bf16", {tokens, 16}, 0, tokens, true},
        {"pos", "i32", {tokens}, 0, tokens, false},
    };
    return rec;
}

OpRecord kv_store_record() {
    const int64_t tokens = 5;
    OpRecord rec;
    rec.op = "kv_store";
    rec.block = 1;
    rec.position = 0;
    rec.n = tokens;
    rec.params = {{"M", tokens},
                  {"block_size", int64_t(4)},
                  {"head_dim", int64_t(8)},
                  {"kv_dtype", std::string("f32")},
                  {"n_head_kv", int64_t(1)}};
    rec.operands = {
        {"k", "f32", {tokens, 1, 8}, 0, tokens, false},
        {"v", "f32", {tokens, 1, 8}, 0, tokens, false},
        {"slots", "i32", {tokens}, 0, tokens, false},
        {"pool", "f32", {3, 1, 4, 16}, 0, 0, true},
    };
    return rec;
}

std::vector<double> ramp(size_t n, double lo = -1.0, double hi = 1.0) {
    std::vector<double> v(n);
    for (size_t i = 0; i < n; ++i) {
        const double t = n == 1 ? 0.0 : double(i) / double(n - 1);
        v[i] = lo + (hi - lo) * t;
    }
    return v;
}

// Values the kernel and the twin both see exactly: a bf16 operand narrows on the way in, so round
// the check' input to bf16 first and the comparison is the output's bf16 rounding alone.
std::vector<double> bf16_exact(std::vector<double> v) {
    for (double& x : v) x = (double)rad_bf16_to_f32(rad_f32_to_bf16((float)x));
    return v;
}

bool refuses(const r97::TwinFn& twin, const OpRecord& rec, const std::vector<AdValue>& in,
             const std::string& text) {
    try {
        std::vector<AdValue> out;
        twin(rec, in, &out);
    } catch (const std::exception& e) {
        return std::string(e.what()).find(text) != std::string::npos;
    }
    return false;
}

}  // namespace

R97_TEST(rope_registry_row_matches_the_host_kernel_and_its_adjoint) {
    const OpRecord rec = rope_record(5, true);
    const std::vector<AdValue> in = {values("bf16", {5, 16}, bf16_exact(ramp(5 * 16))),
                                     values("i32", {5}, {0, 1, 2, 3, 4})};
    const std::vector<AdValue> out = {values("bf16", {5, 16}, {})};

    ac::require(ac::check_kernel(rec, in, out, 2e-2, {0}));
    ac::require(ac::check_adjoint(rec, in, out, 1e-6, 0xfeedface1234ull, {0}));
}

R97_TEST(rope_registry_row_refuses_a_record_without_the_head_width) {
    const OpRecord rec = rope_record(5, false);
    std::string err;
    const r97::TwinFn twin = r97::twin_find(rec.op, rec.params, &err);
    R97_CHECK(twin != nullptr);
    const std::vector<AdValue> in = {values("bf16", {5, 16}, bf16_exact(ramp(5 * 16))),
                                     values("i32", {5}, {0, 1, 2, 3, 4})};
    R97_CHECK(refuses(twin, rec, in, "cannot infer the head width"));
}

R97_TEST(kv_store_registry_row_matches_the_host_kernel_and_its_adjoint) {
    const OpRecord rec = kv_store_record();
    const std::vector<AdValue> in = {
        values("f32", {5, 1, 8}, ramp(5 * 8)),
        values("f32", {5, 1, 8}, ramp(5 * 8, 1.0, -1.0)),
        values("i32", {5}, {0, 1, 2, 4, 5}),
    };
    const std::vector<AdValue> out = {values("f32", {3, 1, 4, 16}, {})};

    ac::require(ac::check_kernel(rec, in, out, 2e-5));
    ac::require(ac::check_adjoint(rec, in, out, 1e-6));
}

R97_TEST(kv_store_registry_row_refuses_a_record_without_a_pool_write) {
    OpRecord rec = kv_store_record();
    rec.operands.pop_back();  // no written pool
    std::string err;
    const r97::TwinFn twin = r97::twin_find(rec.op, rec.params, &err);
    R97_CHECK(twin != nullptr);
    const std::vector<AdValue> in = {
        values("f32", {5, 1, 8}, ramp(5 * 8)),
        values("f32", {5, 1, 8}, ramp(5 * 8, 1.0, -1.0)),
        values("i32", {5}, {0, 1, 2, 4, 5}),
    };
    R97_CHECK(refuses(twin, rec, in, "no pool write"));
}
