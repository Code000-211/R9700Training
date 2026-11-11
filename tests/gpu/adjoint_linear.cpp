// tests/gpu/adjoint_linear.cpp -- the frozen-linears family on the card.
//
// The device rows are the ones src/adjoint/device/linear.hip registers on Side::device; this file
// asks two questions of them.
//
//   agrees: the device row and the host row, handed the same record and the same values, write the
//   same dx within the harness's bf16 tolerance (tests/support/device_check.h). At the fixture's
//   shapes, and at the real model's hidden width with few rows, that is the whole of dx, compared
//   element by element. At the model's real shapes -- 1024 and 4096 tokens, and a packed batch of
//   unequal lengths -- it is a spread sample of (m,k) probes against the same f64 sum the host
//   adjoint forms: the full comparison there is hundreds of GMAC of f64 on one thread per call.
//
//   measures: every real shape prints one line of adjoint milliseconds, forward-twin milliseconds
//   and scratch, as docs/kernels/README.md asks. The forward twin is called once per shape rather
//   than reps+1 times, because at the widest linear the f64 twin is tens of minutes a call; the
//   adjoint is warmed up and averaged. docs/kernels/linear.md carries the numbers.
//
// The values follow tests/host/adjoint_linear.cpp -- integer codes, bf16-exact bf16 operands,
// power-of-two scales -- so that the two sides start from what the host test's kernel check starts
// from. The cotangent is uniform random, which is the point of the model-shape probes: it exercises
// the bf16 rounding of the gradient, not exactness.
//
// Label: gpu. The kernels launch on the card the guard reserved (R97_DEVICE selects it).
#include "support/device_check.h"
#include "support/test.h"

#include "r97/adjoint.h"
#include "r97/op.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

using namespace r97;

namespace {

const double bf16_tol = 1e-2;  // whole-tensor relative L2 over the probe sample

uint64_t mix(uint64_t* s) { return device_check::splitmix64(s); }

double unit(uint64_t* s) { return device_check::urand(s); }

bool has(const std::string& s, const char* needle) { return s.find(needle) != std::string::npos; }

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

AdValue value_of(const Operand& o) {
    AdValue v;
    v.dtype = o.dtype;
    v.shape = o.shape;
    v.value.assign(static_cast<std::size_t>(numel(o.shape)), 0.0);
    return v;
}

OpRecord gemm_nt_q_rec(int64_t M, int64_t N, int64_t K) {
    OpRecord rec;
    rec.op = "gemm_nt_q";
    rec.params = {{"M", M}, {"N", N}, {"K", K}, {"group", 128}, {"dtype", "i8a8h"}};
    rec.operands = {
        {"a", "i8", {M, K}, 0, M, false},
        {"a_scale", "f32", {M, K / 128}, 0, M, false},
        {"b", "i8", {N, K}, 0, N, false},
        {"b_scale", "bf16", {N, K / 128}, 0, N, false},
        {"y", "f32", {M, N}, 0, M, true},
    };
    return rec;
}

OpRecord gemm_nt_rec(int64_t M, int64_t N, int64_t K) {
    OpRecord rec;
    rec.op = "gemm_nt";
    rec.params = {{"M", M}, {"N", N}, {"K", K}, {"dtype", "bf16"}};
    rec.operands = {
        {"a", "bf16", {M, K}, 0, M, false},
        {"b", "bf16", {N, K}, 0, N, false},
        {"y", "f32", {M, N}, 0, M, true},
    };
    return rec;
}

int operand_pos(const OpRecord& rec, const char* name) {
    for (std::size_t i = 0; i < rec.operands.size(); ++i)
        if (rec.operands[i].buffer == name) return static_cast<int>(i);
    return -1;
}

// A case the kernel was written for: int8 codes as integers, bf16 values exact in bf16, scales
// powers of two, and a uniform cotangent at the linear's output.
device_check::Case linear_case(const OpRecord& rec, uint64_t seed, std::string label) {
    device_check::Case c;
    c.rec = rec;
    c.label = std::move(label);
    for (const Operand& o : rec.operands) {
        if (o.written) continue;
        AdValue v = value_of(o);
        for (double& x : v.value) {
            const uint64_t r = mix(&seed);
            if (o.dtype == "i8") {
                x = static_cast<double>(static_cast<int>(r % 9) - 4);
            } else if (o.dtype == "bf16") {
                x = static_cast<double>(static_cast<int>(r % 257) - 128) / 128.0;  // exact in bf16
            } else if (has(o.buffer, "scale")) {
                x = (static_cast<double>((r >> 16) % 16) + 1.0) / 256.0;  // power of two: exact
            } else {
                x = 2.0 * unit(&seed) - 1.0;
            }
        }
        c.in.push_back(std::move(v));
    }
    for (const Operand& o : rec.operands) {
        if (!o.written) continue;
        AdValue d = value_of(o);
        for (double& x : d.value) x = 2.0 * unit(&seed) - 1.0;
        c.cot.push_back(std::move(d));
    }
    return c;
}

// One call over a case without the forward twin: the reads come from the case, the writes are
// zeroed slots at the schema positions the kernel expects. The model's shapes cannot afford the
// twin, whose f64 GEMM is the better part of an hour at the widest linear.
AdjointCall build_call(const device_check::Case& c, std::vector<AdValue>* fwd, std::vector<AdValue>* dc) {
    fwd->clear();
    std::size_t read = 0;
    for (const Operand& o : c.rec.operands) {
        if (o.written) {
            fwd->push_back(value_of(o));
        } else {
            fwd->push_back(c.in.at(read++));
        }
    }
    dc->clear();
    for (const Operand& o : c.rec.operands) dc->push_back(value_of(o));
    return AdjointCall{c.rec, *fwd, c.cot, dc, nullptr, ""};
}

AdjointFn device_row(const OpRecord& rec) {
    std::string why;
    AdjointFn fn = adjoint_find(rec.op, Side::device, rec.params, &why);
    if (fn == nullptr)
        throw std::runtime_error("device adjoint of " + rec.op + ": " + why +
                                 " (is the plugin loaded?)");
    return fn;
}

double rel_l2(const std::vector<double>& dev, const std::vector<double>& ref) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        num += (dev[i] - ref[i]) * (dev[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    if (den == 0.0) return std::sqrt(num);
    return std::sqrt(num / den);
}

// dx[m,k] = sum_n dy[m,n] * W[n,k], W the codes times their per-block scale (`gemm_nt_q`) or the
// bf16 weight as it stands: the f64 sum the host adjoint forms, for one element.
double probe_ref(const AdjointCall& call, int64_t m, int64_t k) {
    const int ia = operand_pos(call.rec, "a");
    const int ib = operand_pos(call.rec, "b");
    const int ibs = operand_pos(call.rec, "b_scale");
    const int64_t K = call.fwd[static_cast<std::size_t>(ia)].shape[1];
    const int64_t N = call.fwd[static_cast<std::size_t>(ib)].shape[0];
    const int64_t group = K / 128 > 0 ? 128 : K;
    const int64_t groups = (K + group - 1) / group;
    const std::vector<double>& dy = call.cot.front().value;
    const std::vector<double>& b = call.fwd[static_cast<std::size_t>(ib)].value;
    const std::vector<double>* s =
        ibs >= 0 ? &call.fwd[static_cast<std::size_t>(ibs)].value : nullptr;
    const int64_t g = group > 0 ? k / group : 0;
    double acc = 0.0;
    for (int64_t n = 0; n < N; ++n)
        acc += dy[static_cast<std::size_t>(m * N + n)] *
               b[static_cast<std::size_t>(n * K + k)] *
               (s != nullptr ? (*s)[static_cast<std::size_t>(n * groups + g)] : 1.0);
    return acc;
}

// The device row on a spread sample of (m,k) against probe_ref.
double agree_on_probes(const device_check::Case& c, uint64_t seed, int probes) {
    std::vector<AdValue> fwd, dc;
    AdjointCall call = build_call(c, &fwd, &dc);
    device_row(c.rec)(call);
    const int ia = operand_pos(c.rec, "a");
    const std::vector<double>& dx = dc[static_cast<std::size_t>(ia)].value;
    const int64_t K = c.in[static_cast<std::size_t>(ia)].shape[1];
    const int64_t M = c.in[static_cast<std::size_t>(ia)].shape[0];
    std::vector<double> got, want;
    got.reserve(static_cast<std::size_t>(probes));
    want.reserve(static_cast<std::size_t>(probes));
    for (int i = 0; i < probes; ++i) {
        const uint64_t r = mix(&seed);
        const int64_t m = static_cast<int64_t>(r % static_cast<uint64_t>(M));
        const int64_t k = static_cast<int64_t>((r >> 32) % static_cast<uint64_t>(K));
        got.push_back(dx[static_cast<std::size_t>(m * K + k)]);
        want.push_back(probe_ref(call, m, k));
    }
    return rel_l2(got, want);
}

void report(const char* what, const OpRecord& rec, double rel) {
    const int64_t M = *std::get_if<int64_t>(&rec.params[0].second);
    const int64_t N = *std::get_if<int64_t>(&rec.params[1].second);
    const int64_t K = *std::get_if<int64_t>(&rec.params[2].second);
    std::printf("%-22s %-10s M=%5lld N=%6lld K=%6lld  rel %.3g\n", what, rec.op.c_str(),
                static_cast<long long>(M), static_cast<long long>(N), static_cast<long long>(K), rel);
    std::fflush(stdout);
}

// The whole of dx and every frozen position against the host row, through the harness.
void agree_whole(const device_check::Case& c) {
    const device_check::Report r = device_check::check(c);
    device_check::require(r);
    R97_CHECK(r.scratch_peak > 0);
    std::printf("%-34s device == host whole, scratch peak %lld KiB\n", c.label.c_str(),
                static_cast<long long>(r.scratch_peak / 1024));
    std::fflush(stdout);
    if (r97test::bench()) device_check::print_bench(c, 20);
}

struct Timing {
    double adjoint_ms = 0.0;
    double forward_ms = 0.0;
    int64_t scratch_peak = 0;
};

// The harness's measurement, with the forward twin called once instead of reps+1 times: at the
// widest real shape one f64 twin call is over twenty minutes, and the twin is a scalar loop whose
// rate does not depend on a warm cache.
Timing time_call(const device_check::Case& c, int reps) {
    const AdjointFn row = device_row(c.rec);
    const TwinFn twin = twin_find(c.rec.op, c.rec.params, nullptr);
    if (!twin) throw std::runtime_error("no twin for op '" + c.rec.op + "'");

    std::vector<AdValue> fwd, dc;
    AdjointCall call = build_call(c, &fwd, &dc);
    device::scratch_reset();
    row(call);  // warm-up: the first launch is where the kernel's module is finalized
    auto zero_dc = [&dc] {
        for (AdValue& v : dc) std::fill(v.value.begin(), v.value.end(), 0.0);
    };
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        zero_dc();
        row(call);
    }
    const auto t1 = std::chrono::steady_clock::now();

    Timing t;
    t.adjoint_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    t.scratch_peak = device::scratch_stats().peak_bytes;

    std::vector<AdValue> out;
    const auto t2 = std::chrono::steady_clock::now();
    twin(c.rec, c.in, &out);
    const auto t3 = std::chrono::steady_clock::now();
    t.forward_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    return t;
}

void measure(const device_check::Case& c, int reps) {
    const Timing t = time_call(c, reps);
    const double ratio = t.forward_ms > 0.0 ? t.adjoint_ms / t.forward_ms : 0.0;
    std::printf("%-28s %8.3f ms adjoint  %8.3f ms forward twin  %6.2fx forward  "
                "scratch peak %lld KiB\n",
                (c.rec.op + " " + c.label).c_str(), t.adjoint_ms, t.forward_ms, ratio,
                static_cast<long long>(t.scratch_peak / 1024));
    std::fflush(stdout);
    R97_CHECK(t.adjoint_ms > 0.0);
}

// The card from R97_DEVICE, locked for the test's lifetime, with the VRAM the largest call below
// holds at once (dy and the codes staged, dx, and room for the code object).
r97test::GuardOptions guard_options() { return device_check::guard_options(int64_t(2) << 30); }

}  // namespace

R97_TEST(adjoint_linear_device_rows) {
    // What the family registers: the two ops on Side::device, the same frozen positions as their
    // host rows, and the int8 predicate that keeps the kernel to the 128-wide scale grid it serves.
    (void)device_check::load_plugin();
    for (const OpRecord& rec : {gemm_nt_q_rec(4, 256, 128), gemm_nt_rec(4, 256, 128)}) {
        const std::vector<int> dev = adjoint_frozen(rec.op, Side::device, rec.params);
        const std::vector<int> host = adjoint_frozen(rec.op, Side::host, rec.params);
        R97_CHECK(!dev.empty());
        R97_CHECK(dev == host);
        std::printf("%s: device frozen %zu operand(s), the host row's\n", rec.op.c_str(), dev.size());
    }
    OpRecord other = gemm_nt_q_rec(4, 256, 128);  // the same op on a grid the kernel does not serve
    other.params[3].second = int64_t(64);
    std::string why;
    R97_CHECK(adjoint_find(other.op, Side::device, other.params, &why) == nullptr);
    std::printf("gemm_nt_q group 64 is left to the host row: %s\n", why.c_str());
    std::fflush(stdout);
}

R97_TEST(adjoint_linear_device_agrees) {
    const r97test::GuardOptions o = guard_options();
    const r97test::DeviceGuard guard(o);
    (void)device_check::load_plugin();
    device_check::require_device(o.device);
    std::printf("adjoint_linear: card %s\n", guard.card().c_str());

    // The tiny fixture's shapes, whole: the fixture's MLP and vocabulary widths, its eight-channel
    // delta-net projection (padded to sixteen on the card) and its ssm_out width.
    for (const OpRecord& rec : {gemm_nt_q_rec(4, 256, 128), gemm_nt_q_rec(1, 128, 256),
                                gemm_nt_q_rec(7, 320, 384), gemm_nt_q_rec(3, 8, 128),
                                gemm_nt_rec(4, 256, 128), gemm_nt_rec(1, 128, 256),
                                gemm_nt_rec(7, 320, 384), gemm_nt_rec(3, 8, 128),
                                gemm_nt_rec(2, 128, 64)}) {
        const int64_t M = *std::get_if<int64_t>(&rec.params[0].second);
        const int64_t N = *std::get_if<int64_t>(&rec.params[1].second);
        const int64_t K = *std::get_if<int64_t>(&rec.params[2].second);
        agree_whole(linear_case(rec, 0x2000 + static_cast<uint64_t>(N), rec.op + " fixture " +
                                                                         std::to_string(M) + "x" +
                                                                         std::to_string(N) + "x" +
                                                                         std::to_string(K)));
    }

    // The real model's hidden width with few rows, whole: five scale blocks of K, both ops.
    for (const OpRecord& rec : {gemm_nt_q_rec(16, 5120, 5120), gemm_nt_rec(16, 5120, 5120)}) {
        agree_whole(linear_case(rec, 0x2100, rec.op + " hidden width 16x5120x5120 whole"));
    }

    // The real model's linears at one sequence's worth of tokens, on probes: the hidden width, the
    // MLP's intermediate and down widths, the attention projections' widths, and one bf16 GEMM.
    struct Real {
        const char* name;
        OpRecord rec;
    };
    const std::vector<Real> real = {
        {"attn_k/v", gemm_nt_q_rec(1024, 1024, 5120)},
        {"attn_output", gemm_nt_q_rec(1024, 5120, 6144)},
        {"attn_qg", gemm_nt_q_rec(1024, 12288, 5120)},
        {"ssm_inz", gemm_nt_q_rec(1024, 16384, 5120)},
        {"ffn_down", gemm_nt_q_rec(1024, 5120, 17408)},
        {"ffn_gate_up", gemm_nt_q_rec(1024, 34816, 5120)},
        {"ssm_out bf16", gemm_nt_rec(1024, 5120, 6144)},
        // Four thousand and ninety-six tokens of the widest linear, and a batch of unequal lengths
        // whose 4023 tokens leave both the 256-row band and the sixteen rows of one wave's tile a
        // partial tail.
        {"ffn_gate_up 4096", gemm_nt_q_rec(4096, 34816, 5120)},
        {"attn_k/v batch", gemm_nt_q_rec(4023, 1024, 5120)},
    };
    for (const Real& r : real) {
        const double rel = agree_on_probes(linear_case(r.rec, 0x3000, r.name), 0x3001, 512);
        report(r.name, r.rec, rel);
        R97_CHECK(rel <= bf16_tol);
    }
}

R97_TEST(adjoint_linear_device_teeth) {
    // The comparison has teeth: a device kernel that leaves the weight's contribution out of one
    // row of dx is a kernel this harness has to reject.
    const r97test::GuardOptions o = guard_options();
    const r97test::DeviceGuard guard(o);
    (void)device_check::load_plugin();
    device_check::require_device(o.device);

    const device_check::Case c = linear_case(gemm_nt_q_rec(4, 256, 128), 0x5000, "fixture, one row dropped");
    const device_check::Report wrong = device_check::check(c, [](const AdjointCall& call) {
        const AdjointFn right = device_row(call.rec);
        right(call);
        AdValue& dx = call.dc->at(static_cast<std::size_t>(operand_pos(call.rec, "a")));
        std::fill(dx.value.begin(), dx.value.begin() + static_cast<std::ptrdiff_t>(dx.shape[1]), 0.0);
    });
    R97_CHECK(!wrong.ok);
    R97_CHECK(wrong.first_slot == operand_pos(c.rec, "a"));
    std::printf("rejected the kernel that dropped one row: %s\n", wrong.note.c_str());
    std::fflush(stdout);
}

R97_TEST(adjoint_linear_device_measures) {
    const r97test::GuardOptions o = guard_options();
    const r97test::DeviceGuard guard(o);
    (void)device_check::load_plugin();
    device_check::require_device(o.device);
    if (!r97test::bench()) {
        std::printf("adjoint_linear: measurement skipped without --bench (the forward twin is "
                    "minutes a call at these shapes)\n");
        return;
    }
    std::printf("adjoint_linear: card %s, one call per shape, the whole call (upload, kernel, "
                "download)\n", guard.card().c_str());

    // Every linear of one transformer block at 1024 and at 4096 tokens, plus a packed batch of
    // three sequences of unequal length (317 + 1009 + 2697 = 4023 rows), which is what a step's
    // token count looks like. The attention linear runs at 1024 rows for its 12288-wide form
    // (where the projection's own width dominates) and at 4096 for the others.
    measure(linear_case(gemm_nt_q_rec(1024, 1024, 5120), 0x4001, "1024 tok, k/v"), 3);
    measure(linear_case(gemm_nt_q_rec(4096, 1024, 5120), 0x4002, "4096 tok, k/v"), 3);
    measure(linear_case(gemm_nt_q_rec(4096, 5120, 6144), 0x4003, "4096 tok, attn output"), 3);
    measure(linear_case(gemm_nt_q_rec(1024, 12288, 5120), 0x4004, "1024 tok, q/g"), 3);
    measure(linear_case(gemm_nt_q_rec(4096, 12288, 5120), 0x4005, "4096 tok, q/g"), 3);
    measure(linear_case(gemm_nt_q_rec(4096, 16384, 5120), 0x4006, "4096 tok, ssm in/z"), 3);
    measure(linear_case(gemm_nt_q_rec(4096, 5120, 17408), 0x4007, "4096 tok, ffn down"), 3);
    measure(linear_case(gemm_nt_q_rec(1024, 34816, 5120), 0x4008, "1024 tok, ffn gate/up"), 3);
    measure(linear_case(gemm_nt_q_rec(4096, 34816, 5120), 0x4009, "4096 tok, ffn gate/up"), 3);
    measure(linear_case(gemm_nt_q_rec(4023, 34816, 5120), 0x400a, "batch 317+1009+2697, ffn gate/up"),
            3);
    measure(linear_case(gemm_nt_rec(4096, 5120, 6144), 0x400b, "4096 tok, ssm out bf16"), 3);
}
