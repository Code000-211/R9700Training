// tests/host/adjoint_fills_in_place.cpp -- a row that registers `fills_in_place_slots` owns the
// slot on every path (PR 162's review).
//
// Registering the predicate turns the walk's in-place passthrough off for the row, so two things
// must hold or a cotangent is lost or counted twice without a word:
//
//   * the predicate and the walk agree on what an in-place read is. The walk asks `in_place_read`,
//     which also recognises a `kv:` buffer no separate read names when the record carries no mark
//     (a record built from a program); a predicate that tested the mark alone left that pool to
//     the passthrough, which copies the whole pool cotangent in, the overwritten rows included.
//   * the row fills the slot before any exit. `lora_expand` returned ahead of its fill when it
//     could not serve the adapter's layout; with the passthrough off that dropped the cotangent
//     of the buffer the write replaced for every block below.
//
// The fused norm's host row is held to the same standard on its sizes: a residual cotangent or a
// gradient slot of the wrong size is refused by name, not indexed past its end.

#include "support/test.h"

#include "r97/adjoint.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace r97;

namespace {

AdValue values(const char* dtype, std::vector<int64_t> shape, std::vector<double> v) {
    AdValue a;
    a.dtype = dtype;
    a.shape = std::move(shape);
    a.value = std::move(v);
    return a;
}

std::vector<double> ramp(size_t n, double base) {
    std::vector<double> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = base + 0.125 * (double)(i + 1);
    return v;
}

// Two blocks of two rows, one kv head, two lanes each way; three tokens cover rows 0, 1 and 2,
// so row 3 is the one the store leaves alone.
constexpr int64_t kBlocks = 2, kBlockSize = 2, kHeads = 1, kHeadDim = 2, kTokens = 3;
constexpr int64_t kRowValues = 2 * kHeadDim;                       // k lanes, then v lanes
constexpr int64_t kPoolValues = kBlocks * kHeads * kBlockSize * kRowValues;

OpRecord kv_rec(const char* pool_buffer, bool marked) {
    OpRecord rec;
    rec.op = "kv_store";
    rec.params = {{"head_dim", kHeadDim}, {"block_size", kBlockSize}, {"n_head_kv", kHeads}};
    rec.n = kTokens;
    rec.operands = {
        {"k", "bf16", {kTokens, kHeads, kHeadDim}, 0, kTokens, false},
        {"v", "bf16", {kTokens, kHeads, kHeadDim}, 0, kTokens, false},
        {"slot", "i32", {kTokens}, 0, kTokens, false},
        {pool_buffer, "bf16", {kBlocks, kHeads, kBlockSize, kRowValues}, 0, kBlocks, true}};
    rec.operands[3].in_place = marked;
    return rec;
}

}  // namespace

// The predicate is the walk's rule, not the mark: an unmarked `kv:` pool is the row's too.
R97_TEST(fills_in_place_slots_follow_the_walks_rule) {
    const OpRecord marked = kv_rec("kv:0", true);
    const OpRecord unmarked = kv_rec("kv:0", false);
    const OpRecord plain = kv_rec("pool", false);
    R97_CHECK(fills_in_place_slots(marked, 3));
    R97_CHECK(fills_in_place_slots(unmarked, 3));
    R97_CHECK(in_place_read(unmarked, 3));
    R97_CHECK(!fills_in_place_slots(plain, 3));     // a plain write is not an in-place read
    R97_CHECK(!fills_in_place_slots(marked, 0));    // a read is not a write
    R97_CHECK(!fills_in_place_slots(marked, 9));    // past the record
    R97_CHECK(adjoint_fills("kv_store", Side::host, marked.params) == fills_in_place_slots);
    R97_CHECK(adjoint_fills("lora_expand", Side::host, {}) == fills_in_place_slots);
}

// The kv_store row fills the unmarked `kv:` pool's slot itself: the rows the store overwrote get
// nothing (their cotangent is the k's and the v's), the row it left alone carries the pool's.
R97_TEST(kv_store_fills_an_unmarked_kv_pool) {
    for (const bool marked : {true, false}) {
        const OpRecord rec = kv_rec("kv:0", marked);
        const std::vector<AdValue> fwd = {
            values("bf16", {kTokens, kHeads, kHeadDim}, ramp((size_t)(kTokens * kHeadDim), 1.0)),
            values("bf16", {kTokens, kHeads, kHeadDim}, ramp((size_t)(kTokens * kHeadDim), 2.0)),
            values("i32", {kTokens}, {0.0, 1.0, 2.0}),
            values("bf16", {kBlocks, kHeads, kBlockSize, kRowValues},
                   ramp((size_t)kPoolValues, 3.0))};
        const std::vector<AdValue> cot = {values("bf16", {kBlocks, kHeads, kBlockSize, kRowValues},
                                                 ramp((size_t)kPoolValues, 0.5))};
        std::vector<AdValue> dc(rec.operands.size());
        dc[3] = values("bf16", {kBlocks, kHeads, kBlockSize, kRowValues},
                       std::vector<double>((size_t)kPoolValues, 0.0));

        std::string err;
        const AdjointFn fn = adjoint_find(rec.op, Side::host, rec.params, &err);
        R97_CHECK(fn != nullptr);
        AdjointCall call{rec, fwd, cot, &dc, nullptr, ""};
        fn(call);

        for (int64_t row = 0; row < kBlocks * kBlockSize; ++row)
            for (int64_t e = 0; e < kRowValues; ++e) {
                const size_t at = (size_t)(row * kRowValues + e);
                const double want = row < kTokens ? 0.0 : cot[0].value[at];
                R97_CHECK(dc[3].value[at] == want);
            }
        // The covered rows' cotangent went to k and v instead, once.
        R97_CHECK(dc[0].value.size() == (size_t)(kTokens * kHeadDim));
        R97_CHECK(dc[0].value[0] == cot[0].value[0]);
        R97_CHECK(dc[1].value[0] == cot[0].value[(size_t)kHeadDim]);
    }
}

// The expand's direct path is filled before the row looks at the adapter's layout: a record the
// row cannot serve (here the pool's B has no values) still hands the write's cotangent to the
// buffer it replaced. A slot of another size is refused by name.
R97_TEST(lora_expand_fills_the_direct_path_before_any_exit) {
    constexpr int64_t M = 2, N = 3, R = 2, S = 1;
    OpRecord rec;
    rec.op = "lora_expand";
    rec.params = {{"M", M}, {"N", N}, {"R", R}, {"S", S}, {"dtype", std::string("bf16")}};
    rec.n = M;
    rec.operands = {{"t", "f32", {M, R}, 0, M, false},
                    {"slot", "i32", {M}, 0, M, false},
                    {"lora_b", "bf16", {S, N, R}, 0, S, false},
                    {"scale", "f32", {S, N}, 0, S, false},
                    {"y", "bf16", {M, N}, 0, M, true}};
    rec.operands[4].in_place = true;

    const std::vector<AdValue> fwd = {
        values("f32", {M, R}, ramp((size_t)(M * R), 1.0)),
        values("i32", {M}, {0.0, 0.0}),
        values("bf16", {S, N, R}, {}),  // no values: expand_layout cannot serve the record
        values("f32", {S, N}, ramp((size_t)(S * N), 1.0)),
        values("bf16", {M, N}, ramp((size_t)(M * N), 2.0))};
    const std::vector<AdValue> cot = {values("bf16", {M, N}, ramp((size_t)(M * N), 0.5))};

    std::string err;
    const AdjointFn fn = adjoint_find(rec.op, Side::host, rec.params, &err);
    R97_CHECK(fn != nullptr);

    std::vector<AdValue> dc(rec.operands.size());
    dc[0] = values("f32", {M, R}, std::vector<double>((size_t)(M * R), 0.0));
    dc[4] = values("bf16", {M, N}, std::vector<double>((size_t)(M * N), 0.0));
    AdjointCall call{rec, fwd, cot, &dc, nullptr, ""};
    fn(call);
    R97_CHECK(dc[4].value == cot[0].value);
    for (double v : dc[0].value) R97_CHECK(v == 0.0);  // the layout was not served

    std::vector<AdValue> short_dc(rec.operands.size());
    short_dc[0] = dc[0];
    short_dc[4] = values("bf16", {M, 1}, std::vector<double>((size_t)M, 0.0));
    AdjointCall short_call{rec, fwd, cot, &short_dc, nullptr, ""};
    bool refused = false;
    try {
        fn(short_call);
    } catch (const std::runtime_error& e) {
        refused = std::string(e.what()).find("in-place slot") != std::string::npos;
    }
    R97_CHECK(refused);
}

// The fused norm's elementwise row, on the residual layout: a residual cotangent of the wrong
// size is refused by name instead of read past its end.
R97_TEST(rmsnorm_residual_refuses_a_short_cotangent) {
    constexpr int64_t rows = 2, n = 4;
    OpRecord rec;
    rec.op = "rmsnorm";
    rec.params = {{"M", rows}, {"n", n}, {"dtype", std::string("bf16")}};
    rec.n = rows;
    rec.operands = {{"x", "bf16", {rows, n}, 0, rows, false},
                    {"residual", "bf16", {rows, n}, 0, rows, true},
                    {"y", "bf16", {rows, n}, 0, rows, true}};
    rec.operands[1].in_place = true;

    const std::vector<AdValue> fwd = {values("bf16", {rows, n}, ramp((size_t)(rows * n), 1.0)),
                                      values("bf16", {rows, n}, ramp((size_t)(rows * n), 2.0)),
                                      values("bf16", {rows, n}, ramp((size_t)(rows * n), 3.0))};
    std::string err;
    const AdjointFn fn = adjoint_find(rec.op, Side::host, rec.params, &err);
    R97_CHECK(fn != nullptr);

    const auto run = [&](size_t res_cot_values) {
        const std::vector<AdValue> cot = {
            values("bf16", {rows, n}, std::vector<double>(res_cot_values, 0.25)),
            values("bf16", {rows, n}, ramp((size_t)(rows * n), 0.5))};
        std::vector<AdValue> dc(rec.operands.size());
        dc[0] = values("bf16", {rows, n}, std::vector<double>((size_t)(rows * n), 0.0));
        dc[1] = dc[0];
        AdjointCall call{rec, fwd, cot, &dc, nullptr, ""};
        fn(call);
        return dc;
    };

    // The full-size residual cotangent lands on both summands; the engine's one-value
    // placeholder adds nothing and is not an error.
    const std::vector<AdValue> full = run((size_t)(rows * n));
    const std::vector<AdValue> placeholder = run(1);
    for (size_t i = 0; i < (size_t)(rows * n); ++i) {
        R97_CHECK(full[0].value[i] == full[1].value[i]);
        R97_CHECK(placeholder[0].value[i] == placeholder[1].value[i]);
        R97_CHECK(std::fabs(full[0].value[i] - placeholder[0].value[i] - 0.25) < 1e-12);
    }

    bool refused = false;
    try {
        run(3);
    } catch (const std::runtime_error& e) {
        refused = std::string(e.what()).find("residual's cotangent") != std::string::npos;
    }
    R97_CHECK(refused);
}
