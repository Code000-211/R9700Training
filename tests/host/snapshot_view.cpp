// tests/host/snapshot_view.cpp -- snapshot() of an operand the tape describes by a view.
//
// The walk reads each operand the op saw back through snapshot() (src/join_common.h). An operand
// the tape cannot state as one byte span reaches the walk as a declared shape with no byte count:
// a strided view (patches/training/0120+0123 carry the rank, shape, strides and element offset the
// kernel received) or a dtype with no element size (a plugin-private quantised tensor, which the
// kernel decodes in its own layout). The first must snapshot exactly the values the kernel saw --
// bit for bit the same values a packed copy of the view holds, whether it is read live through the
// strides or from a tape that kept a packed copy. The second has no f64 value at all and must not
// be refused as a lost span (the real container's gemm_nt_q weight operands are exactly that: at
// HEAD the device step refused `blk.63.attn_qg.weight` by name for it). A *readable* operand whose
// span the tape lost stays refused: that gate is what caught the zeroed adapter gradient, and
// tests/host/lora_shrink_step_order.cpp holds its message.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "rad_runtime.h"  // RAD_ISSUE_OBSERVER / RAD_REPLAY_BLOCK gate join_common.h
#include "engine.h"       // the complete rad::Engine behind ForwardDriver's unique_ptr
#include "rad_plugin.h"   // join_common's read_elem
#include "../../src/join_common.h"

#include "support/test.h"

namespace {

// The declared buffer is [kRows, kCols] f32 row-major, values chosen so that every element tells
// itself apart from its neighbours in both axes ([row, col] -> 100*row + col).
constexpr int64_t kRows = 4;
constexpr int64_t kCols = 7;
constexpr int64_t kTake = 3;  // columns the view keeps
constexpr int64_t kOff  = 2;  // the view's first column in the declared buffer

std::vector<float> declared() {
    std::vector<float> b(static_cast<size_t>(kRows * kCols));
    for (int64_t r = 0; r < kRows; ++r)
        for (int64_t c = 0; c < kCols; ++c)
            b[static_cast<size_t>(r * kCols + c)] = static_cast<float>(100 * r + c) * 0.25f;
    return b;
}

// The operand record for the view [kRows, kTake] at column kOff of that buffer: shape and strides
// as the observer hands them over, and the declared shape as the walk's record keeps it.
r97::Operand view_operand() {
    r97::Operand op;
    op.buffer = "gdn_in";
    op.dtype  = "f32";
    op.shape  = {kRows, kCols};
    op.view_shape  = {kRows, kTake};
    op.view_stride = {kCols, 1};
    op.view_off    = kOff;
    return op;
}

// What the kernel saw, in the view's own order, as f64: the reference every case must meet bit for
// bit.
std::vector<double> seen_by_the_kernel(const std::vector<float>& b) {
    std::vector<double> want;
    for (int64_t r = 0; r < kRows; ++r)
        for (int64_t c = 0; c < kTake; ++c)
            want.push_back(static_cast<double>(b[static_cast<size_t>(r * kCols + kOff + c)]));
    return want;
}

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

}  // namespace

// The half the patch carries: an operand the observer describes by rank, shape, strides and
// element offset snapshots to the same values as a packed copy of the view. Both readings of the
// content are exercised: a live buffer, which the snapshot must walk with the strides
// (OperandContent::bytes == 0), and a tape that kept the view's values packed (bytes > 0).
R97_TEST(snapshot_view_is_bit_identical_to_a_packed_copy) {
    const std::vector<float> b = declared();
    const r97::Operand op = view_operand();
    const std::vector<double> want = seen_by_the_kernel(b);

    const r97::AdValue live = r97::snapshot(op, {b.data(), 0, true}, "gdn_gated_rmsnorm");
    R97_CHECK(live.shape == (std::vector<int64_t>{kRows, kTake}));
    R97_CHECK(same_bits(live.value, want));

    // The same view, handed over as the packed copy the host tape keeps of it: the values must
    // still be the view's, not the declared buffer's.
    std::vector<float> packed;
    for (int64_t r = 0; r < kRows; ++r)
        for (int64_t c = 0; c < kTake; ++c)
            packed.push_back(b[static_cast<size_t>(r * kCols + kOff + c)]);
    const r97::AdValue from_copy =
        r97::snapshot(op, {packed.data(), packed.size() * sizeof(float), true}, "gdn_gated_rmsnorm");
    R97_CHECK(from_copy.shape == (std::vector<int64_t>{kRows, kTake}));
    R97_CHECK(same_bits(from_copy.value, want));
}

// The other half: a dtype with no element size has no f64 value to snapshot, so the walk gets the
// shape and no values rather than a refusal. The real container's quantised weight operands are
// this case (the device step refused `blk.63.attn_qg.weight` at HEAD), and no adjoint row may read
// a value that does not exist -- the rows guard on value.empty().
R97_TEST(snapshot_keeps_no_values_for_a_dtype_with_no_element_size) {
    r97::Operand w;
    w.buffer = "blk.63.attn_qg.weight";
    w.dtype  = "plugin-private";
    w.shape  = {16, 8};
    const void* ptr = reinterpret_cast<const void*>(static_cast<uintptr_t>(0x1000));

    const r97::AdValue got = r97::snapshot(w, {ptr, 0, true}, "gemm_nt_q");
    R97_CHECK(got.shape == (std::vector<int64_t>{16, 8}));
    R97_CHECK(got.value.empty());

    // The same when the observer describes it by a view: still no values, still no throw.
    w.view_shape  = {16, 8};
    w.view_stride = {8, 1};
    w.view_off    = 0;
    const r97::AdValue viewed = r97::snapshot(w, {ptr, 0, true}, "gemm_nt_q");
    R97_CHECK(viewed.shape == (std::vector<int64_t>{16, 8}));
    R97_CHECK(viewed.value.empty());
}

// The gate stays: a readable operand the tape handed a pointer for but could not span is refused by
// name, not read as zeros. This is the failure the zeroed adapter gradient hid behind.
R97_TEST(snapshot_refuses_a_readable_operand_with_no_span) {
    r97::Operand s;
    s.buffer = "lora.blk.0.attn_qg.scale";
    s.dtype  = "f32";
    s.shape  = {4};
    const float values[4] = {1.f, 2.f, 3.f, 4.f};

    bool threw = false;
    std::string what;
    try {
        (void)r97::snapshot(s, {values, 0, true}, "lora_expand");
    } catch (const std::runtime_error& e) {
        threw = true;
        what = e.what();
    }
    R97_CHECK(threw);
    R97_CHECK(what.find("lora_expand") != std::string::npos);
    R97_CHECK(what.find("lora.blk.0.attn_qg.scale") != std::string::npos);
    R97_CHECK(what.find("carries no bytes") != std::string::npos);
}
