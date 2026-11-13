// tests/host/lora_shrink_step_order.cpp -- the reference lora_shrink reads a step-order
// activation.
//
// radiance declares lora_shrink with two dtype parameters: `dtype`, the activation's element dtype
// (always "i8" for codes, whatever the site), and `gemm_dtype`, the encoding of the GEMM whose
// activation it reads -- the same string that GEMM carries. It is `gemm_dtype` that says how the
// codes are stored: quant_act_i8 writes them in the int8 WMMA's step order at w4a8/w8a8/w2a8 (the
// middle dwords of every 16-byte step exchanged) and the rotated producers write column order.
//
// patches/serving/0210-lora-int8-trunk.patch makes libref's lora_shrink unexchange the step
// order when `gemm_dtype` names one of those. The int8 fixture's trunk is i8a8h, whose codes are
// already in column order, so the fixture test exercises only the column-order reading; this runs
// the reference kernel directly, with the parameter set radiance declares at such a site: the
// same codes, once in column order and once in step order, must produce the same f32 accumulator,
// and that accumulator must agree with an independent f64 dot in which every code is multiplied
// by its group's scale.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "rad_runtime.h"  // RAD_ISSUE_OBSERVER and RAD_REPLAY_BLOCK gate join_common.h
#include "engine.h"       // the complete rad::Engine behind ForwardDriver's unique_ptr
#include "rad_plugin.h"   // rad_fp8e4m3_to_f32, which join_common's read_elem calls
#include "../../src/join_common.h"

#include "support/adjoint_check.h"
#include "support/test.h"

namespace {

constexpr int64_t kM = 4, kK = 128, kR = 2, kS = 1, kG = 128;

// i8_step_pos from libref/ref_common.h: the middle two dwords of each 16-code step exchanged.
int64_t step_pos(int64_t c) {
    return (c & ~int64_t(12)) | ((c & 4) << 1) | ((c & 8) >> 1);
}

struct Result {
    int rc = RAD_E_INVAL;
    std::vector<float> t;
};

// Run libref's lora_shrink over `codes` (row-major [M,K], column order) as a site whose GEMM is
// `gemm_dtype` reads it. With `step_order` the codes are stored the way quant_act_i8 writes them.
Result shrink(const RadKernelInfo* k, const std::vector<int8_t>& codes,
              const std::vector<float>& scale, const std::vector<float>& a, const char* gemm_dtype,
              bool step_order) {
    std::vector<int8_t> x((size_t)(kM * kK));
    for (int64_t m = 0; m < kM; ++m)
        for (int64_t c = 0; c < kK; ++c)
            x[(size_t)(m * kK + (step_order ? step_pos(c) : c))] = codes[(size_t)(m * kK + c)];

    std::vector<int32_t> slot((size_t)kM, 0);
    std::vector<float> t((size_t)(kM * kR), 0.0f);

    RadTensor tensors[5]{};
    tensors[0].data = x.data();
    tensors[0].dtype = RAD_I8;
    tensors[0].rank = 2;
    tensors[0].shape[0] = kM;
    tensors[0].shape[1] = kK;
    tensors[0].stride[0] = kK;
    tensors[0].stride[1] = 1;

    tensors[1].data = const_cast<float*>(scale.data());
    tensors[1].dtype = RAD_F32;
    tensors[1].rank = 2;
    tensors[1].shape[0] = kM;
    tensors[1].shape[1] = kK / kG;
    tensors[1].stride[0] = kK / kG;
    tensors[1].stride[1] = 1;

    tensors[2].data = slot.data();
    tensors[2].dtype = RAD_I32;
    tensors[2].rank = 1;
    tensors[2].shape[0] = kM;
    tensors[2].stride[0] = 1;

    tensors[3].data = const_cast<float*>(a.data());
    tensors[3].dtype = RAD_F32;
    tensors[3].rank = 3;
    tensors[3].shape[0] = kS;
    tensors[3].shape[1] = kR;
    tensors[3].shape[2] = kK;
    tensors[3].stride[0] = kR * kK;
    tensors[3].stride[1] = kK;
    tensors[3].stride[2] = 1;

    tensors[4].data = t.data();
    tensors[4].dtype = RAD_F32;
    tensors[4].rank = 2;
    tensors[4].shape[0] = kM;
    tensors[4].shape[1] = kR;
    tensors[4].stride[0] = kR;
    tensors[4].stride[1] = 1;

    // `dtype` is the element dtype radiance declares (`rad_dtype_name` of the codes), and
    // `gemm_dtype` the site GEMM's own name: the two are different strings at every coded site.
    RadParam params[] = {
        RAD_INT("K", kK),  RAD_INT("R", kR),
        RAD_INT("S", kS),  RAD_INT("M", kM),
        RAD_INT("group", kG),
        RAD_STR("dtype", "i8"),
        RAD_STR("gemm_dtype", gemm_dtype),
    };
    RadArgs args{};
    args.t = tensors;
    args.n_t = 5;
    args.p = params;
    args.n_p = (int)(sizeof params / sizeof params[0]);
    args.world_size = 1;

    Result r;
    r.rc = k->launch(&args, nullptr);
    r.t = t;
    return r;
}

}  // namespace

R97_TEST(lora_shrink_reads_step_order_codes) {
    const RadKernelInfo* k = r97::adjoint_check::find_host_kernel("lora_shrink");
    R97_CHECK(k != nullptr);

    uint64_t seed = 0x0210c0deull;
    std::vector<int8_t> codes((size_t)(kM * kK));
    std::vector<float> scale((size_t)kM, 0.0f);
    std::vector<float> a((size_t)(kS * kR * kK));
    for (int64_t m = 0; m < kM; ++m) {
        scale[(size_t)m] = 0.003f + (float)(m + 1) * 0.0007f;
        for (int64_t c = 0; c < kK; ++c)
            codes[(size_t)(m * kK + c)] = (int8_t)((int)(r97::adjoint_check::splitmix64(&seed) % 255) - 127);
    }
    for (float& v : a)
        v = (float)r97::adjoint_check::urand(&seed) * 0.25f;

    // The column-order readings: the rotated trunk's i8a8h, and the plain i8a8 codes of
    // quant_act_i8g. Both must agree, bit for bit, with the same kernel run over the codes in
    // each step-order encoding -- w4a8, w8a8 and w2a8 are what quant_act_i8 feeds and what
    // radiance's model of an A in the fwht128 basis is served with.
    const Result column = shrink(k, codes, scale, a, "i8a8h", false);
    R97_CHECK(column.rc == RAD_OK);
    const struct {
        const char* gemm_dtype;
        bool        step_order;
    } cases[] = {
        { "i8a8", false }, { "i8a8h", false }, { "w4a8", true },
        { "w8a8", true },  { "w2a8", true },
    };
    for (const auto& c : cases) {
        const Result got = shrink(k, codes, scale, a, c.gemm_dtype, c.step_order);
        R97_CHECK(got.rc == RAD_OK);
        R97_CHECK(got.t.size() == column.t.size());
        R97_CHECK(std::memcmp(column.t.data(), got.t.data(), column.t.size() * sizeof(float)) == 0);
    }

    // The independent f64 dot: t[m,j] = sum_k (code[m,k] * scale[m,k]) * A[0,j,k].
    for (int64_t m = 0; m < kM; ++m) {
        for (int64_t j = 0; j < kR; ++j) {
            double want = 0.0;
            for (int64_t c = 0; c < kK; ++c)
                want += (double)codes[(size_t)(m * kK + c)] * (double)scale[(size_t)m] *
                        (double)a[(size_t)(j * kK + c)];
            const double got = (double)column.t[(size_t)(m * kR + j)];
            R97_CHECK(std::fabs(got - want) <= 1e-5 * std::max(std::fabs(got), std::fabs(want)) + 1e-6);
        }
    }
}

// join_common's snapshot(): a declared operand the tape handed a pointer for but could not
// describe as one span is an error, not zeros of its declared shape. The silent zeros are what
// zeroed an operand in the device step (n_zero_a = n_zero_b = 399 of 400). Which operand is not
// yet settled -- a plugin-private weight read as zeros cascades its zero dx up the walk, and a zero
// scale on a lora row zeroes dB, dt and hence dA -- but a pointer with no span is refused by name
// either way, so the next run says which. An absent operand (no pointer) and
// an operand the tape does not name (a raw batch field -- the adapter slot array, whose zero is a
// valid slot index) keep the zeros.
R97_TEST(snapshot_refuses_a_declared_operand_with_no_span) {
    r97::Operand scale;
    scale.buffer = "pool.scale";
    scale.dtype = "f32";
    scale.shape = {4};
    const float values[4] = {1.f, 2.f, 3.f, 4.f};

    // A span the tape described: the values it holds.
    const r97::OperandContent spanned{values, sizeof values, true};
    const r97::AdValue got = r97::snapshot(scale, spanned, "lora_expand");
    R97_CHECK(got.shape == std::vector<int64_t>({4}));
    R97_CHECK(got.value.size() == 4 && got.value[2] == 3.0);

    // The operand is there -- a pointer and a declared shape -- but the tape could not span it.
    const r97::OperandContent unspanned{values, 0, true};
    bool threw = false;
    std::string what;
    try {
        (void)r97::snapshot(scale, unspanned, "lora_expand");
    } catch (const std::runtime_error& e) {
        threw = true;
        what = e.what();
    }
    R97_CHECK(threw);
    R97_CHECK(what.find("lora_expand") != std::string::npos);
    R97_CHECK(what.find("pool.scale") != std::string::npos);
    R97_CHECK(what.find("[4]") != std::string::npos);

    // An absent operand hands nothing in: zeros, as before.
    const r97::OperandContent absent{nullptr, 0, false};
    const r97::AdValue none = r97::snapshot(scale, absent, "lora_expand");
    R97_CHECK(none.value.size() == 4 && none.value[0] == 0.0);

    // A raw batch field is unnamed: no refusal, and its zero is a valid adapter slot.
    r97::Operand slot;
    slot.dtype = "i32";
    const r97::OperandContent raw{values, 0, false};
    (void)r97::snapshot(slot, raw, "lora_shrink");
}

// The other half of the ticket: a weight that several delta modules land in is one fused adapter
// on the engine's side (A concatenated along the rank, B block-diagonal), and the walk reports one
// gradient for the pair. The split must give every module its own non-zero slice. Two modules,
// ranks 2 and 3, on one weight -- the shape the device step needs for all 400 modules of the
// container, where 399 came back exactly zero.
R97_TEST(split_adapter_grads_gives_every_module_its_own_slice) {
    const int64_t kIn = 8;
    r97::ParamSet params;
    const auto leaf = [&params](const std::string& name, std::vector<int64_t> shape) {
        r97::Param p;
        p.name = name;
        p.shape = std::move(shape);
        p.value.assign(static_cast<size_t>(r97::numel(p.shape)), 0.0f);
        params.push_back(std::move(p));
    };
    leaf("m0.lora_A", {2, kIn});
    leaf("m0.lora_B", {3, 2});
    leaf("m1.lora_A", {3, kIn});
    leaf("m1.lora_B", {4, 3});

    r97::SiteGroups groups;
    groups["blk.0.qkv"] = {r97::SiteSlice{"m0", 2, kIn, 3, 0, 0},
                           r97::SiteSlice{"m1", 3, kIn, 4, 2, 3}};

    r97::ParamSet grads;
    r97::Param a, b;
    a.name = "@fused:blk.0.qkv.lora_A";
    a.shape = {5, kIn};
    a.value.resize(static_cast<size_t>(5 * kIn));
    for (size_t i = 0; i < a.value.size(); ++i) a.value[i] = 1.0f + float(i);
    b.name = "@fused:blk.0.qkv.lora_B";
    b.shape = {7, 5};
    b.value.resize(7 * 5);
    for (size_t i = 0; i < b.value.size(); ++i) b.value[i] = 100.0f + float(i);
    grads.push_back(a);
    grads.push_back(b);

    r97::split_adapter_grads(&grads, params, groups, "");

    R97_CHECK(r97::find_grad(grads, "@fused:blk.0.qkv.lora_A") == nullptr);
    R97_CHECK(r97::find_grad(grads, "@fused:blk.0.qkv.lora_B") == nullptr);
    const r97::Param* a0 = r97::find_grad(grads, "m0.lora_A");
    const r97::Param* a1 = r97::find_grad(grads, "m1.lora_A");
    const r97::Param* b0 = r97::find_grad(grads, "m0.lora_B");
    const r97::Param* b1 = r97::find_grad(grads, "m1.lora_B");
    R97_CHECK(a0 != nullptr);
    R97_CHECK(a1 != nullptr);
    R97_CHECK(b0 != nullptr);
    R97_CHECK(b1 != nullptr);
    R97_CHECK(a0->shape == std::vector<int64_t>({2, kIn}));
    R97_CHECK(a1->shape == std::vector<int64_t>({3, kIn}));
    R97_CHECK(b0->shape == std::vector<int64_t>({3, 2}));
    R97_CHECK(b1->shape == std::vector<int64_t>({4, 3}));

    // Each module holds its own slice of the fused pair, not a copy of the whole: A's rows 0..1
    // then 2..4, B's block rows 0..2 x rank 0..1 then 3..6 x rank 2..4.
    R97_CHECK(a0->value.front() == 1.0f && a0->value.back() == 16.0f);
    R97_CHECK(a1->value.front() == 17.0f && a1->value.back() == 40.0f);
    R97_CHECK(b0->value.front() == 100.0f && b0->value.back() == 111.0f);
    R97_CHECK(b1->value.front() == 117.0f && b1->value.back() == 134.0f);

    // Every module's gradient is non-zero, and the two modules' are distinct.
    const auto sq = [](const r97::Param& p) {
        double t = 0.0;
        for (float v : p.value) t += double(v) * double(v);
        return t;
    };
    R97_CHECK(sq(*a0) > 0.0 && sq(*a1) > 0.0 && sq(*b0) > 0.0 && sq(*b1) > 0.0);
    R97_CHECK(sq(*a0) != sq(*a1));
    R97_CHECK(sq(*b0) != sq(*b1));
}
