// tests/gpu/adjoint_gdn_prep.cpp -- the device adjoints of gdn_prep against the host adjoints.
//
// The two ops the family has (docs/kernels/gdn_prep.md, docs/adjoints/gdn_prep.md):
//   * gdn_conv_prep: the 4-wide causal convolution over the projected row, its silu, the l2 norm
//     of q and k, and the per-chunk gate and beta, at the fixture's geometry (hidden 128) and at
//     the real container's (16 key heads, 48 value heads, head dimension 128, convolution width 4,
//     a conv row of 10240); one sequence and a batch of unequal lengths, 1024 and 4096 tokens;
//   * gdn_gated_rmsnorm: the gated RMS norm on the recurrence's output, over the fixture's 16
//     channels and the real 128, at the fixture's row count and at 1024 and 4096 tokens of 48
//     heads;
// both within the harness's bf16 tolerance of the host adjoint on the same values;
//   * the test rejects a kernel that drops one of the gate's routes rather than accepting
//     anything plausible;
//   * the registered rows declare the same frozen operands the host rows do, so the walk never
//     asks either side for a weight's cotangent.
//
// Needs one R9700 (gfx1201, 32 GB) or similar; runs under the GPU test guard. No container is
// read -- the real shapes are numbers from docs/DESIGN.md and docs/ADAPTER.md section 6 -- so this
// test does not carry the `model` label.
#include "support/device_check.h"
#include "support/gpu_guard.h"
#include "support/test.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace r97;

/* The conv op's schema positions, as src/adjoint/host/gdn_prep.cpp names them. */
enum : int {
    kX = 0,
    kW = 1,
    kBias = 2,
    kALog = 3,
    kDtBias = 4,
    kState = 5,
    kCu = 6,
    kA = 7,
    kBGate = 8,
    kSidx = 9,
    kHasInit = 10,
    kQ = 11,
    kK = 12,
    kV = 13,
    kG = 14,
    kBeta = 15,
};

enum : int { nX = 0, nZ = 1, nW = 2, nO = 3 };

AdValue index_value(const char* dtype, std::vector<int64_t> shape, std::vector<double> values) {
    AdValue v;
    v.dtype = dtype;
    v.shape = std::move(shape);
    v.value = std::move(values);
    return v;
}

/* The shared harness lays a forward out as its reads followed by the twin's writes. That is wrong
 * for this family's conv op: it writes its INOUT conv_state at schema position 5 and reads its
 * gate at 7, and both the host adjoint and the kernel index the forward by schema position. Rather
 * than change the shared harness for one family, these helpers do for this file what
 * device_check::make_call, check and print_bench do, with the forward laid out by schema position;
 * the case, the tolerance, the scratch ledger and the guard are still the shared ones. */
AdjointCall make_schema_call(const device_check::Case& c, std::vector<AdValue>* fwd,
                             std::vector<AdValue>* dc) {
    TwinFn twin = twin_find(c.rec.op, c.rec.params, nullptr);
    if (!twin) throw std::runtime_error("no twin for op '" + c.rec.op + "'");

    std::vector<AdValue> out;
    twin(c.rec, c.in, &out);
    fwd->assign(c.rec.operands.size(), AdValue{});
    size_t r = 0;
    size_t w = 0;
    for (size_t i = 0; i < c.rec.operands.size(); ++i) {
        if (c.rec.operands[i].written) {
            if (w < out.size()) (*fwd)[i] = std::move(out[w++]);
        } else if (r < c.in.size()) {
            (*fwd)[i] = c.in[r++];
        }
    }

    dc->clear();
    for (const Operand& op : c.rec.operands) {
        AdValue v;
        v.dtype = op.dtype;
        v.shape = op.shape;
        v.value.assign((size_t)device_check::numel(op.shape), 0.0);
        dc->push_back(std::move(v));
    }
    return AdjointCall{c.rec, *fwd, c.cot, dc, nullptr, ""};
}

/* device_check::check, over make_schema_call instead of the harness's forward layout. */
device_check::Report check_schema(const device_check::Case& c, AdjointFn device = nullptr) {
    device_check::Report r;
    r.label = c.label;

    std::string err;
    if (!device) device = adjoint_find(c.rec.op, Side::device, c.rec.params, &err);
    if (!device) {
        r.ok = false;
        r.note = c.label + ": device adjoint: " + err;
        return r;
    }
    AdjointFn host = adjoint_find(c.rec.op, Side::host, c.rec.params, &err);
    if (!host) {
        r.ok = false;
        r.note = c.label + ": host adjoint: " + err;
        return r;
    }

    std::vector<AdValue> device_fwd, device_dc;
    device::scratch_reset();
    {
        AdjointCall call = make_schema_call(c, &device_fwd, &device_dc);
        device(call);
    }
    const device::ScratchStats stats = device::scratch_stats();
    r.scratch_peak = stats.peak_bytes;
    if (stats.live_bytes != 0) {
        r.ok = false;
        r.note = c.label + ": device kernel leaked " + std::to_string(stats.live_bytes)
                 + " bytes of scratch";
        return r;
    }

    // The host reference sees the same forward values and cotangent as the device; only its dc
    // slots start at zero again.
    std::vector<AdValue> host_dc = device_dc;
    for (AdValue& v : host_dc) std::fill(v.value.begin(), v.value.end(), 0.0);
    AdjointCall host_call{c.rec, device_fwd, c.cot, &host_dc, nullptr, ""};
    host(host_call);

    if (device_dc.size() != host_dc.size()) {
        r.ok = false;
        r.note = c.label + ": device wrote " + std::to_string(device_dc.size())
                 + " dc slots, host " + std::to_string(host_dc.size());
        return r;
    }
    for (size_t slot = 0; slot < device_dc.size(); ++slot) {
        const AdValue& d = device_dc[slot];
        const AdValue& h = host_dc[slot];
        if (d.value.size() != h.value.size()) {
            r.ok = false;
            r.note = c.label + ": dc[" + std::to_string(slot) + "] ("
                     + device_check::op_name(c, slot) + ") has " + std::to_string(d.value.size())
                     + " values, host " + std::to_string(h.value.size());
            return r;
        }
        for (size_t i = 0; i < d.value.size(); ++i) {
            const double dv = d.value[i];
            const double hv = h.value[i];
            const double tol = device_check::kAbsTol + device_check::kRelTol * std::abs(hv);
            if (std::isfinite(dv) && std::abs(dv - hv) <= tol) continue;
            if (r.mismatches++ == 0) {
                r.first_slot = (int64_t)slot;
                r.first_index = (int64_t)i;
                r.first_device = dv;
                r.first_host = hv;
                r.ok = false;
            }
        }
    }
    if (!r.ok) {
        char buf[256];
        std::snprintf(buf, sizeof buf, " (tolerance %.3g + %.3g*|host|)", device_check::kAbsTol,
                      device_check::kRelTol);
        r.note = c.label + ": dc[" + std::to_string(r.first_slot) + "] ("
                 + device_check::op_name(c, r.first_slot) + ") index "
                 + std::to_string(r.first_index) + ": device " + std::to_string(r.first_device)
                 + ", host " + std::to_string(r.first_host) + buf
                 + (r.mismatches > 1 ? ", " + std::to_string(r.mismatches) + " mismatches"
                                     : std::string());
    }
    return r;
}

/* device_check::time_case, over make_schema_call: the adjoint's whole call and the twin. */
device_check::Timing time_schema(const device_check::Case& c, int reps = 20) {
    AdjointFn device = adjoint_find(c.rec.op, Side::device, c.rec.params, nullptr);
    if (!device) throw std::runtime_error("no device adjoint for op '" + c.rec.op + "'");
    TwinFn twin = twin_find(c.rec.op, c.rec.params, nullptr);
    if (!twin) throw std::runtime_error("no twin for op '" + c.rec.op + "'");

    device_check::Timing t;
    if (reps < 1) reps = 1;

    std::vector<AdValue> fwd, dc;
    AdjointCall call = make_schema_call(c, &fwd, &dc);
    device::scratch_reset();
    device(call);  // warm-up: the first launch is where the kernel's module is finalized
    auto zero_dc = [&dc] {
        for (AdValue& v : dc) std::fill(v.value.begin(), v.value.end(), 0.0);
    };
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        zero_dc();
        device(call);
    }
    auto t1 = std::chrono::steady_clock::now();
    t.adjoint_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    t.scratch_peak = device::scratch_stats().peak_bytes;

    std::vector<AdValue> out;
    twin(c.rec, c.in, &out);
    auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        out.clear();
        twin(c.rec, c.in, &out);
    }
    auto t3 = std::chrono::steady_clock::now();
    t.forward_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / reps;
    return t;
}

void print_bench_schema(const device_check::Case& c, int reps = 20) {
    const device_check::Timing t = time_schema(c, reps);
    const double ratio = t.forward_ms > 0.0 ? t.adjoint_ms / t.forward_ms : 0.0;
    std::printf("%-28s %8.3f ms adjoint  %8.3f ms forward twin  %6.2fx forward  "
                "scratch peak %lld KiB\n",
                (c.rec.op + " " + c.label).c_str(), t.adjoint_ms, t.forward_ms, ratio,
                (long long)(t.scratch_peak / 1024));
    std::fflush(stdout);
}

/* One gdn_conv_prep record at a prefill from position 0: the state is written, `has_init` is
 * cold, the gate and beta arrive in their own operands and `x`'s rows carry the conv channels
 * (plus `row_extra` columns past them, which the op does not read but a fused row would have). */
device_check::Case conv_case(int64_t Hg, int64_t H, int64_t K, int64_t V, int64_t width,
                             int64_t chunk, const std::vector<int64_t>& lens, int64_t row_extra,
                             uint64_t seed) {
    device_check::Case c;
    const int64_t conv_dim = 2 * Hg * K + H * V;
    const int64_t xw = conv_dim + row_extra;
    int64_t T = 0;
    std::vector<double> cu{0.0};
    for (int64_t len : lens) {
        T += len;
        cu.push_back((double)T);
    }
    const int64_t S = (int64_t)lens.size();
    const int64_t keep = width - 1;

    c.rec.op = "gdn_conv_prep";
    c.rec.position = 0;
    c.rec.params = {{"head_k", K}, {"head_v", V}, {"conv_width", width}, {"chunk", chunk}};
    c.rec.operands = {
        Operand{"x", "bf16", {T, xw}, 0, T, false},
        Operand{"w", "bf16", {conv_dim, width}, 0, conv_dim, false},
        Operand{"b", "bf16", {conv_dim}, 0, 1, false},
        Operand{"A_log", "f32", {H}, 0, 1, false},
        Operand{"dt_bias", "f32", {H}, 0, 1, false},
        Operand{"conv_state", "bf16", {S, conv_dim, keep}, 0, S, true},
        Operand{"cu", "i32", {S + 1}, 0, S + 1, false},
        Operand{"a", "bf16", {T, H}, 0, T, false},
        Operand{"b_gate", "bf16", {T, H}, 0, T, false},
        Operand{"cache_idx", "i32", {S}, 0, S, false},
        Operand{"has_init", "i32", {S}, 0, S, false},
        Operand{"q", "bf16", {T, Hg, K}, 0, T, true},
        Operand{"k", "bf16", {T, Hg, K}, 0, T, true},
        Operand{"v", "bf16", {T, H, V}, 0, T, true},
        Operand{"g", "f32", {T, H}, 0, T, true},
        Operand{"beta", "f32", {T, H}, 0, T, true},
    };
    std::vector<double> slots((size_t)S);
    for (int64_t n = 0; n < S; ++n) slots[(size_t)n] = (double)n;
    c.in = {
        device_check::random_value("bf16", {T, xw}, &seed),
        device_check::random_value("bf16", {conv_dim, width}, &seed),
        device_check::random_value("bf16", {conv_dim}, &seed),
        device_check::random_value("f32", {H}, &seed),
        device_check::random_value("f32", {H}, &seed),
        index_value("i32", {S + 1}, cu),
        device_check::random_value("bf16", {T, H}, &seed),
        device_check::random_value("bf16", {T, H}, &seed),
        index_value("i32", {S}, slots),
        index_value("i32", {S}, std::vector<double>((size_t)S, 0.0)),
    };
    c.cot = {
        device_check::random_value("bf16", {S, conv_dim, keep}, &seed),
        device_check::random_value("bf16", {T, Hg, K}, &seed),
        device_check::random_value("bf16", {T, Hg, K}, &seed),
        device_check::random_value("bf16", {T, H, V}, &seed),
        device_check::random_value("f32", {T, H}, &seed),
        device_check::random_value("f32", {T, H}, &seed),
    };
    c.label = std::to_string(T) + " tok, " + std::to_string(S) + " seq, " + std::to_string(Hg) + "x"
              + std::to_string(K) + "+" + std::to_string(H) + "x" + std::to_string(V) + ", w"
              + std::to_string(width) + (row_extra > 0 ? " wide row" : "");
    return c;
}

/* One gdn_gated_rmsnorm record: dense [rows, channels] rows, as the op runs them. */
device_check::Case rmsnorm_case(int64_t rows, int64_t channels, const char* act, uint64_t seed) {
    device_check::Case c;
    c.rec.op = "gdn_gated_rmsnorm";
    c.rec.position = 0;
    c.rec.params = {{"channels", channels}, {"act", std::string(act)}};
    c.rec.operands = {
        Operand{"x", "bf16", {rows, channels}, 0, rows, false},
        Operand{"z", "bf16", {rows, channels}, 0, rows, false},
        Operand{"w", "f32", {channels}, 0, 1, false},
        Operand{"o", "bf16", {rows, channels}, 0, rows, true},
    };
    c.in = {device_check::random_value("bf16", {rows, channels}, &seed),
            device_check::random_value("bf16", {rows, channels}, &seed),
            device_check::random_value("f32", {channels}, &seed)};
    c.cot = {device_check::random_value("bf16", {rows, channels}, &seed)};
    c.label = std::to_string(rows) + " x " + std::to_string(channels) + " " + act;
    return c;
}

/* Deliberately wrong: the gate's cotangent is dropped, which the test must catch in slot 7. */
void wrong_conv_prep_adjoint(const AdjointCall& call) {
    AdjointFn fn = adjoint_find("gdn_conv_prep", Side::device, call.rec.params, nullptr);
    if (!fn) return;
    fn(call);
    AdValue& a = call.dc->at(kA);
    std::fill(a.value.begin(), a.value.end(), 0.0);
}

/* Deliberately wrong: the gate operand z's cotangent is dropped, which must be caught in slot 1. */
void wrong_rmsnorm_adjoint(const AdjointCall& call) {
    AdjointFn fn = adjoint_find("gdn_gated_rmsnorm", Side::device, call.rec.params, nullptr);
    if (!fn) return;
    fn(call);
    AdValue& z = call.dc->at(nZ);
    std::fill(z.value.begin(), z.value.end(), 0.0);
}

void run_case(const device_check::Case& c, int reps) {
    const device_check::Report r = check_schema(c);
    device_check::require(r);
    R97_CHECK(r.scratch_peak > 0);  // the kernel took its device buffers from the allocator
    std::printf("%s %s: device == host, scratch peak %lld KiB\n", c.rec.op.c_str(),
                c.label.c_str(), (long long)(r.scratch_peak / 1024));
    if (r97test::bench()) print_bench_schema(c, reps);
}

void run() {
    // The 4096-token cases hold a few hundred megabytes of device scratch, and the harness sizes
    // the guard's request; a gigabyte covers them with room for the engine's own allocations.
    const auto options = device_check::guard_options(int64_t(1) << 30);
    const r97test::DeviceGuard guard(options);
    device_check::require_device(options.device);
    (void)device_check::load_plugin();

    R97_CHECK(r97::op_rows("gdn_conv_prep").device_adjoint);
    R97_CHECK(r97::op_rows("gdn_gated_rmsnorm").device_adjoint);
    // The device rows freeze what the host rows freeze, so the walk asks neither for a weight.
    R97_CHECK(r97::op_rows("gdn_conv_prep").device_frozen ==
              r97::op_rows("gdn_conv_prep").host_frozen);
    R97_CHECK(r97::op_rows("gdn_gated_rmsnorm").device_frozen ==
              r97::op_rows("gdn_gated_rmsnorm").host_frozen);

    // The fixture's geometry (hidden 128), one sequence and a batch of unequal lengths, then the
    // same geometry with a row wider than the conv channels, the shape of a fused row.
    const device_check::Case fixture = conv_case(2, 4, 16, 16, 4, 4, {8}, 0, 0x51ed1001ull);
    const device_check::Case fixture_ragged =
        conv_case(2, 4, 16, 16, 4, 4, {5, 3, 2}, 0, 0x51ed1002ull);
    const device_check::Case fixture_wide =
        conv_case(2, 4, 16, 16, 4, 4, {7}, 32, 0x51ed1003ull);
    // The real container: 16 key heads of 128, 48 value heads of 128, convolution width 4, a
    // conv row of 10240, chunks of 64.
    const device_check::Case real_1024 = conv_case(16, 48, 128, 128, 4, 64, {1024}, 0, 0x51ed1004ull);
    const device_check::Case real_4096 = conv_case(16, 48, 128, 128, 4, 64, {4096}, 0, 0x51ed1005ull);
    const device_check::Case real_ragged =
        conv_case(16, 48, 128, 128, 4, 64, {1500, 1400, 1196}, 0, 0x51ed1006ull);

    run_case(fixture, 20);
    run_case(fixture_ragged, 20);
    run_case(fixture_wide, 20);
    run_case(real_1024, 5);
    // The twin is the whole op in f64 on the host: at 4096 tokens one rep of it costs seconds, so
    // the two biggest cases are timed twice and the doc says so.
    run_case(real_4096, 2);
    run_case(real_ragged, 2);

    // The gated RMS norm: the fixture's 8 tokens of 4 value heads, 16 channels, then 1024 and
    // 4096 tokens of 48 heads (the op's rows, not the tokens: every head is a row of its own).
    const device_check::Case rms_fixture = rmsnorm_case(32, 16, "silu", 0x51ed2001ull);
    const device_check::Case rms_sigmoid = rmsnorm_case(32, 16, "sigmoid", 0x51ed2002ull);
    const device_check::Case rms_1024 = rmsnorm_case(1024 * 48, 128, "silu", 0x51ed2003ull);
    const device_check::Case rms_4096 = rmsnorm_case(4096 * 48, 128, "silu", 0x51ed2004ull);

    run_case(rms_fixture, 20);
    run_case(rms_sigmoid, 20);
    run_case(rms_1024, 5);
    run_case(rms_4096, 3);

    // A kernel that drops the gate's route must be rejected, in the gate's slot.
    const device_check::Report wrong_conv = check_schema(fixture, wrong_conv_prep_adjoint);
    R97_CHECK(!wrong_conv.ok);
    R97_CHECK(wrong_conv.first_slot == kA);
    std::printf("rejected the wrong conv kernel: %s\n", wrong_conv.note.c_str());

    const device_check::Report wrong_rms = check_schema(rms_fixture, wrong_rmsnorm_adjoint);
    R97_CHECK(!wrong_rms.ok);
    R97_CHECK(wrong_rms.first_slot == nZ);
    std::printf("rejected the wrong norm kernel: %s\n", wrong_rms.note.c_str());
}

}  // namespace

R97_TEST(adjoint_gdn_prep_device) { run(); }
