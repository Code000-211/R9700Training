// tests/gpu/adjoint_attention.cpp -- the device adjoint of paged causal attention, and of the
// fusion the card issues in its place, against the host adjoints.
//
// What this file checks (docs/kernels/attention.md carries the kernel itself):
//   * on the fixture's geometry (4 query heads, 2 key-value heads, head_dim 32, docs/FIXTURES.md)
//     and on the real container's (24 query heads over 4 key-value heads, head_dim 256, hidden
//     5120, docs/DESIGN.md), for one sequence and for a batch of unequal lengths, at 1024 and at
//     4096 tokens, the device adjoint matches src/adjoint/host/attention.cpp within the harness's
//     bf16 tolerance -- on both rows, save the fusion's 4096-token single sequence, which is
//     timed rather than compared because its reference is the plain row's again (section 4 of
//     docs/kernels/attention.md);
//   * `attn_paged_gate_quant`, the card's fusion of the attention, the output gate and the
//     quantising producer, matches the composition of the host adjoints of its parts: the host
//     `attn_paged`, `quant_act_i8g`, `mul` and `sigmoid`;
//   * the harness rejects a deliberately wrong kernel rather than accepting anything plausible.
//
// The fusion has no host row of its own, so this file registers its forward twin -- the same
// composition, forwards -- with the r97 registry; without one the harness could not build the
// case's values, and the composition is what the issue names as the reference.
//
// Needs one R9700 (gfx1201, 32 GB) or similar; runs under the GPU test guard and R97_DEVICE
// selects the card. No container is read -- the real geometry is numbers from the design -- so
// this test does not carry the `model` label.
#include "support/device_check.h"
#include "support/gpu_guard.h"
#include "support/test.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace r97;
using device_check::numel;
using device_check::random_value;

// The attention's geometry, without the sequence lengths: the two are what a case is.
struct Geom {
    int64_t q_heads = 0;
    int64_t kv_heads = 0;
    int64_t head_dim = 0;
    int64_t block_size = 0;
};

const Geom kFixture{4, 2, 32, 8};   // tests/fixtures/oracle's head layout, page 8
const Geom kReal{24, 4, 256, 16};   // docs/DESIGN.md: 24 query heads over 4 key-value heads
constexpr int64_t kGroup = 128;     // the fusion's quantiser group (the op's `group` parameter)

std::string label_of(const Geom& g, const std::vector<int64_t>& lens, bool fused) {
    std::string s = std::to_string(g.q_heads) + "/" + std::to_string(g.kv_heads) + "/" +
                    std::to_string(g.head_dim) + " ";
    int64_t rows = 0;
    for (const int64_t l : lens) rows += l;
    if (lens.size() == 1) {
        s += std::to_string(rows) + " one sequence";
    } else {
        s += std::to_string(rows) + " as";
        for (size_t i = 0; i < lens.size(); ++i) s += (i ? "+" : " ") + std::to_string(lens[i]);
    }
    return (fused ? "fused " : "") + s;
}

OpRecord record_of(const Geom& g, bool fused) {
    OpRecord rec;
    rec.op = fused ? "attn_paged_gate_quant" : "attn_paged";
    char scale[64];
    std::snprintf(scale, sizeof scale, "%.17g", 1.0 / std::sqrt((double)g.head_dim));
    rec.params = {
        {"q_len", int64_t{0}},
        {"head_dim", g.head_dim},
        {"gqa", g.q_heads / g.kv_heads},
        {"block_size", g.block_size},
        {"causal", int64_t{1}},
        {"window", int64_t{0}},
        {"scale", std::string(scale)},
    };
    if (fused) {
        rec.params.push_back({"group", kGroup});
        rec.params.push_back({"act", std::string("sigmoid")});
    }
    return rec;
}

void fill(AdValue* v, double lo, double hi, uint64_t* seed) {
    for (double& x : v->value) x = lo + (hi - lo) * device_check::urand(seed);
}

// One case's forward: a full prefill of each sequence (its position window is the whole context),
// a block table that maps every sequence onto its own consecutive pages, and the descales the
// paged kernel reads. The fusion adds the gate it multiplies in and the two operands its
// quantiser writes.
device_check::Case make_case(const Geom& g, const std::vector<int64_t>& lens, bool fused,
                             uint64_t seed) {
    const int64_t n_seq = (int64_t)lens.size();
    const int64_t hd = g.head_dim;
    int64_t rows = 0, width = 0;
    std::vector<int64_t> blocks((size_t)n_seq, 0);
    for (size_t s = 0; s < lens.size(); ++s) {
        rows += lens[s];
        blocks[s] = (lens[s] + g.block_size - 1) / g.block_size;
        if (blocks[s] > width) width = blocks[s];
    }
    int64_t n_blocks = 0;
    for (const int64_t b : blocks) n_blocks += b;

    device_check::Case c;
    c.rec = record_of(g, fused);
    c.label = label_of(g, lens, fused);

    AdValue q = random_value("bf16", {rows, g.q_heads, hd}, &seed);
    AdValue kv = random_value("bf16", {n_blocks, g.kv_heads, g.block_size, 2 * hd}, &seed);
    AdValue table = random_value("i32", {n_seq, width}, &seed);
    AdValue seqused = random_value("i32", {n_seq}, &seed);
    AdValue cu = random_value("i32", {n_seq + 1}, &seed);
    AdValue kd = random_value("f32", {n_seq, g.kv_heads}, &seed);
    AdValue vd = random_value("f32", {n_seq, g.kv_heads}, &seed);
    fill(&kd, 0.5, 2.0, &seed);
    fill(&vd, 0.5, 2.0, &seed);

    /* The pages of a sequence are its own, in order, so a key's page is where the twin's key_slot
     * puts it (src/adjoint/host/attention.cpp). A page past the sequence's last is -1: the
     * forward reads nothing there and no key maps to it. */
    int64_t first_block = 0;
    for (int64_t s = 0; s < n_seq; ++s) {
        seqused.value[(size_t)s] = (double)lens[(size_t)s];
        cu.value[(size_t)s] = (double)(s == 0 ? 0 : (int64_t)cu.value[(size_t)s - 1] +
                                                   lens[(size_t)s - 1]);
        for (int64_t b = 0; b < width; ++b)
            table.value[(size_t)(s * width + b)] = b < blocks[(size_t)s] ? (double)(first_block + b) : -1.0;
        first_block += blocks[(size_t)s];
    }
    cu.value[(size_t)n_seq] = (double)rows;

    c.in = {q, kv, table, seqused, kd, vd, cu};
    c.rec.operands = {
        Operand{"q", "bf16", q.shape, 0, rows, false},
        Operand{"kv_cache", "bf16", kv.shape, 0, n_blocks, false},
        Operand{"block_table", "i32", table.shape, 0, n_seq, false},
        Operand{"seqused", "i32", seqused.shape, 0, 0, false},
        Operand{"k_descale", "f32", kd.shape, 0, 0, false},
        Operand{"v_descale", "f32", vd.shape, 0, 0, false},
        Operand{"cu_seqlens", "i32", cu.shape, 0, 0, false},
    };

    AdValue gate = random_value("bf16", {rows, g.q_heads, hd}, &seed);
    AdValue out_cot = random_value("bf16", {rows, g.q_heads, hd}, &seed);
    c.rec.operands.push_back(Operand{"out", "bf16", {rows, g.q_heads, hd}, 0, rows, true});
    if (!fused) {
        c.cot = {out_cot};
        return c;
    }

    const int64_t m = rows * g.q_heads;
    const int64_t gwidth = std::min<int64_t>(kGroup, hd);
    const int64_t ngrp = (hd + gwidth - 1) / gwidth;
    c.rec.operands.push_back(Operand{"gate", "bf16", gate.shape, 0, rows, false});
    c.rec.operands.push_back(Operand{"q8", "f8e4m3", {m, hd}, 0, m, true});
    c.rec.operands.push_back(Operand{"q8_scale", "f32", {m, ngrp}, 0, m, true});
    c.in.push_back(gate);
    c.cot = {out_cot, random_value("bf16", {m, hd}, &seed),
             random_value("f32", {m, ngrp}, &seed)};
    return c;
}

int64_t param_i64(const OpRecord& rec, const char* name, int64_t fallback) {
    for (const OpParam& p : rec.params) {
        if (p.first != name) continue;
        if (const auto* i = std::get_if<int64_t>(&p.second)) return *i;
        if (const auto* s = std::get_if<std::string>(&p.second))
            return std::strtoll(s->c_str(), nullptr, 10);
        return fallback;
    }
    return fallback;
}

size_t slot_of(const OpRecord& rec, const std::string& name) {
    for (size_t i = 0; i < rec.operands.size(); ++i)
        if (rec.operands[i].buffer == name) return i;
    throw std::runtime_error("record has no operand '" + name + "'");
}

size_t reads_of(const OpRecord& rec) {
    size_t n = 0;
    for (const Operand& o : rec.operands)
        if (!o.written) ++n;
    return n;
}

// The attention's operands alone, as a record `attn_paged`'s own row reads: the fusion's record
// carries them in the same order, with `out` and the gate between them.
OpRecord attn_record_of(const OpRecord& fused) {
    OpRecord rec = fused;
    rec.op = "attn_paged";
    rec.params.erase(std::remove_if(rec.params.begin(), rec.params.end(),
                                    [](const OpParam& p) {
                                        return p.first == "group" || p.first == "act";
                                    }),
                     rec.params.end());
    /* Its seven reads, then the out the gate overwrote: the gate is the fusion's last read, and
     * the quantiser's two writes are not the attention's. */
    const size_t total_reads = reads_of(fused);
    size_t seen = 0;
    rec.operands.clear();
    for (const Operand& o : fused.operands) {
        if (!o.written && ++seen == total_reads) continue;
        if (o.buffer == "q8" || o.buffer == "q8_scale") continue;
        rec.operands.push_back(o);
    }
    return rec;
}

// The fusion's forward twin: the attention, then the gate, then the quantiser, each the host twin
// of the part. Registered for this file's cases only.
// The record stores its outputs at the widest of the dtypes it names, and the adjoint recomputes
// its mask from them, so the twin hands back the stored value: bf16 for the attention's output and
// f32 for the quantiser's scale.
double stored_bf16(double v) {
    float f = static_cast<float>(v);
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    std::memcpy(&f, &u, sizeof(f));
    return static_cast<double>(f);
}

void fused_twin(const OpRecord& rec, const std::vector<AdValue>& in, std::vector<AdValue>* out) {
    const OpRecord attn_rec = attn_record_of(rec);
    const int64_t hd = param_i64(rec, "head_dim", 0);
    TwinFn attn = twin_find("attn_paged", attn_rec.params, nullptr);
    TwinFn sigmoid = twin_find("sigmoid", rec.params, nullptr);
    TwinFn mul = twin_find("mul", rec.params, nullptr);
    if (!attn || !sigmoid || !mul || hd <= 0) return;

    std::vector<AdValue> a, s, y;
    attn(attn_rec, in, &a);                    // the attention's out, [rows, heads, hd]
    if (a.size() != 1) return;
    sigmoid(rec, {in.back()}, &s);             // the gate is the record's last read
    if (s.size() != 1) return;
    OpRecord mrec = rec;
    mrec.op = "mul";
    mrec.operands = {{"a", a[0].dtype, a[0].shape, 0, 0, false},
                     {"s", s[0].dtype, s[0].shape, 0, 0, false},
                     {"y", "bf16", a[0].shape, 0, 0, true}};
    mul(mrec, {a[0], s[0]}, &y);               // y = out * sigmoid(gate), elementwise
    if (y.size() != 1) return;
    for (double& v : y[0].value) v = stored_bf16(v);
    y[0].dtype = "bf16";

    OpRecord qrec = rec;
    qrec.op = "quant_act_i8g";
    qrec.params = {{"n", hd}, {"group", param_i64(rec, "group", hd)}};
    TwinFn quant = twin_find("quant_act_i8g", qrec.params, nullptr);
    if (!quant) return;
    std::vector<AdValue> codes;
    quant(qrec, {y[0]}, &codes);               // the codes and the group's scale
    if (codes.size() != 2) return;
    for (double& v : codes[1].value) v = static_cast<float>(v);
    codes[1].dtype = "f32";

    out->push_back(std::move(y[0]));
    out->push_back(std::move(codes[0]));
    out->push_back(std::move(codes[1]));
}

// The reference: the host adjoints of the fusion's parts, in the order the kernel undoes them.
void fused_host_reference(const AdjointCall& call) {
    const size_t reads = reads_of(call.rec);
    const OpRecord attn_rec = attn_record_of(call.rec);
    const int64_t hd = param_i64(call.rec, "head_dim", 0);
    if (hd <= 0 || call.cot.size() < 3 || call.fwd.size() < reads + 3) return;

    const AdValue& gate = call.fwd[reads - 1];
    const AdValue& y = call.fwd[reads];
    const AdValue& codes = call.fwd[reads + 1];
    const AdValue& scales = call.fwd[reads + 2];

    /* The quantiser's straight-through step first: its codes' cotangent joins the out cotangent,
     * and its scale's is dropped, as the family's host adjoint drops it (docs/adjoints/quant.md). */
    OpRecord qrec = call.rec;
    qrec.op = "quant_act_i8g";
    qrec.params = {{"n", hd}, {"group", param_i64(call.rec, "group", hd)}};
    qrec.operands = {
        Operand{"x", y.dtype, y.shape, 0, 0, false},
        Operand{"codes", codes.dtype, codes.shape, 0, 0, true},
        Operand{"scale", scales.dtype, scales.shape, 0, 0, true},
    };
    std::vector<AdValue> qfwd = {y, codes, scales};
    std::vector<AdValue> qcot = {call.cot[1], call.cot[2]};
    std::vector<AdValue> qdc(3);
    qdc[0].value.assign(y.value.size(), 0.0);
    qdc[1].value.assign(codes.value.size(), 0.0);
    qdc[2].value.assign(scales.value.size(), 0.0);

    AdjointFn quant = adjoint_find("quant_act_i8g", Side::host, qrec.params, nullptr);
    if (!quant) return;
    AdjointCall quant_call{qrec, qfwd, qcot, &qdc, nullptr, ""};
    quant(quant_call);

    AdValue ytot = call.cot[0];
    for (size_t i = 0; i < ytot.value.size(); ++i) ytot.value[i] += qdc[0].value[i];

    /* The gate: y = a * s with s = sigmoid(gate), so the mul gives the attention's cotangent and
     * the sigmoid's, and the attention is what produced `a`. */
    AdValue s = gate;
    for (double& v : s.value) v = 1.0 / (1.0 + std::exp(-v));
    AdValue a = y;
    for (size_t i = 0; i < a.value.size(); ++i) a.value[i] = y.value[i] / s.value[i];

    OpRecord mrec = call.rec;
    mrec.op = "mul";
    mrec.operands = {
        Operand{"a", a.dtype, a.shape, 0, 0, false},
        Operand{"b", s.dtype, s.shape, 0, 0, false},
        Operand{"y", y.dtype, y.shape, 0, 0, true},
    };
    std::vector<AdValue> mfwd = {a, s};
    std::vector<AdValue> mcot = {ytot};
    std::vector<AdValue> mdc(2);
    mdc[0].value.assign(a.value.size(), 0.0);
    mdc[1].value.assign(a.value.size(), 0.0);
    AdjointFn mul = adjoint_find("mul", Side::host, mrec.params, nullptr);
    if (!mul) return;
    AdjointCall mul_call{mrec, mfwd, mcot, &mdc, nullptr, ""};
    mul(mul_call);

    OpRecord srec = call.rec;
    srec.op = "sigmoid";
    srec.operands = {Operand{"x", gate.dtype, gate.shape, 0, 0, false},
                     Operand{"y", s.dtype, s.shape, 0, 0, true}};
    std::vector<AdValue> sfwd = {gate, s};
    std::vector<AdValue> scot = {mdc[1]};
    std::vector<AdValue> sdc(1);
    sdc[0].value.assign(gate.value.size(), 0.0);
    AdjointFn sigmoid = adjoint_find("sigmoid", Side::host, srec.params, nullptr);
    if (!sigmoid) return;
    AdjointCall sig_call{srec, sfwd, scot, &sdc, nullptr, ""};
    sigmoid(sig_call);

    /* The attention itself, on the fusion's own record: it reads the same seven operands and
     * writes the query's and the pool's cotangents. */
    AdjointFn attn = adjoint_find("attn_paged", Side::host, attn_rec.params, nullptr);
    if (!attn) return;
    std::vector<AdValue> acot = {mdc[0]};
    AdjointCall attn_call{attn_rec, call.fwd, acot, call.dc, nullptr, ""};
    attn(attn_call);

    AdValue& dgate = call.dc->at(slot_of(call.rec, "gate"));
    for (size_t i = 0; i < dgate.value.size(); ++i) dgate.value[i] += sdc[0].value[i];
}

// Deliberately wrong: carries the cotangent to the query and drops the pool's, which the harness
// must catch in the pool's slot.
void wrong_attention_adjoint(const AdjointCall& call) {
    const AdjointFn real = adjoint_find(call.rec.op, Side::device, call.rec.params, nullptr);
    if (!real) return;
    real(call);
    if (call.dc->size() > 1)
        for (double& v : call.dc->at(1).value) v = 0.0;  // the pool's rows left empty
}

// The bench runs the op's forward twin -- f64 on the host -- so the real shapes are timed once;
// the fixture's are cheap enough for the harness's default.
int bench_reps(const device_check::Case& c) {
    int64_t rows = 0;
    for (const Operand& o : c.rec.operands)
        if (o.buffer == "q") rows = o.shape.empty() ? 0 : o.shape[0];
    return rows > 64 ? 1 : 20;
}

void run() {
    const auto options = device_check::guard_options(int64_t(1) << 30);
    const r97test::DeviceGuard guard(options);
    device_check::require_device(options.device);
    (void)device_check::load_plugin();

    R97_CHECK(r97::op_rows("attn_paged").device_adjoint);
    R97_CHECK(r97::op_rows("attn_paged_gate_quant").device_adjoint);

    const std::vector<device_check::Case> cases = {
        // The fixture's geometry: one short sequence, then several of unequal length.
        make_case(kFixture, {12}, false, 0x51ed1001ull),
        make_case(kFixture, {5, 9, 14}, false, 0x51ed1002ull),
        // 4096 tokens in one sequence, where every row sees the whole pool: at the fixture's head
        // geometry first, which is cheap, and then at the real one, whose f64 reference is minutes
        // a pass (the twin alone takes 103 s at that shape).
        make_case(kFixture, {4096}, false, 0x51ed1005ull),
        // The real geometry: 1024 tokens as one sequence, and the same count as a batch of
        // unequal lengths, whose pages are laid out one sequence after another.
        make_case(kReal, {1024}, false, 0x51ed1003ull),
        make_case(kReal, {128, 256, 640}, false, 0x51ed1004ull),
        make_case(kReal, {4096}, false, 0x51ed1006ull),
        // 4096 tokens as a batch of unequal lengths at the real geometry: one long sequence and
        // two shorter ones. The shorter two are not a whole number of pages, so each ends part way
        // into one, and their tables are padded with -1 past it. Cheaper than the 4096-token
        // single sequence above -- the reference's work is the sum of the squares, not the square
        // of the sum -- so its comparison is affordable.
        make_case(kReal, {2048, 1300, 748}, false, 0x51ed1007ull),
    };
    const std::vector<device_check::Case> fused_cases = {
        make_case(kFixture, {12}, true, 0x51ed2001ull),
        make_case(kFixture, {5, 9, 14}, true, 0x51ed2002ull),
        make_case(kFixture, {4096}, true, 0x51ed2005ull),
        make_case(kReal, {1024}, true, 0x51ed2003ull),
        make_case(kReal, {128, 256, 640}, true, 0x51ed2004ull),
        make_case(kReal, {2048, 1300, 748}, true, 0x51ed2007ull),
    };

    for (const device_check::Case& c : cases) {
        // The plain row's reference is the host adjoint the family registers under the op's name.
        const device_check::Report r = device_check::check(c);
        device_check::require(r);
        R97_CHECK(r.scratch_peak > 0);  // the kernel took its device buffers from the allocator
        std::printf("attn %s: device == host, scratch peak %lld KiB\n", c.label.c_str(),
                    (long long)(r.scratch_peak / 1024));
        std::fflush(stdout);
        if (r97test::bench()) device_check::print_bench(c, bench_reps(c));
    }
    for (const device_check::Case& c : fused_cases) {
        const device_check::Report r = device_check::check(c, nullptr, fused_host_reference);
        device_check::require(r);
        R97_CHECK(r.scratch_peak > 0);
        std::printf("attn %s: device == the composition of the host adjoints, scratch peak %lld "
                    "KiB\n",
                    c.label.c_str(), (long long)(r.scratch_peak / 1024));
        std::fflush(stdout);
        if (r97test::bench()) device_check::print_bench(c, bench_reps(c));
    }

    /* The fusion at the real geometry and 4096 tokens: timed, not compared. Its reference would be
     * another pass of the plain row's host adjoint at the shape above, which is already minutes of
     * f64; what the fusion adds there is the gate's straight-through pass, linear in the tokens and
     * compared at the shapes above, over the attention the case above compares. */
    if (r97test::bench()) {
        const device_check::Case big_fused = make_case(kReal, {4096}, true, 0x51ed3002ull);
        device_check::print_bench(big_fused, 1);
    }

    // A kernel that drops the pool's cotangent must be rejected, in the pool's slot.
    const device_check::Report wrong = device_check::check(cases[0], wrong_attention_adjoint);
    R97_CHECK(!wrong.ok);
    R97_CHECK(wrong.first_slot == 1);
    std::printf("rejected the wrong kernel: %s\n", wrong.note.c_str());
}

}  // namespace

R97_REGISTER_TWIN("attn_paged_gate_quant", fused_twin)

R97_TEST(adjoint_attention_device) { run(); }
