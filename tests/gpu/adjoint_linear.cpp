// tests/gpu/adjoint_linear.cpp -- the frozen-linears family on the card.
//
// The device rows are the ones src/adjoint/device/linear.hip registers on Side::device; this file
// asks two questions of them.
//
//   agrees: the device row and the host row, handed the same record and the same values, write the
//   same dx within bf16_tol. At the fixture's shapes that is the whole of dx. At the model's it is
//   a spread sample of (m,k) probes against the same f64 sum the host adjoint forms, because the
//   full comparison there is hundreds of GMAC of f64 on one thread per call -- the doc says which
//   shapes carry which check.
//
//   measures: every real shape runs twice and the call's staging and kernel milliseconds and the
//   card memory it held are printed (src/adjoint/device/device.h). docs/kernels/linear.md carries
//   the numbers; no speed target is set, the measurement is the deliverable.
//
// The values follow tests/host/adjoint_linear.cpp -- integer codes, bf16-exact bf16 operands,
// power-of-two scales -- so that the two sides start from what the host test's kernel check starts
// from. The cotangent is uniform random, which is the point of the model-shape probes: it exercises
// the bf16 rounding of the gradient, not exactness.
//
// Label: gpu. The kernels launch on the card the guard reserved (R97_DEVICE selects it).
#include "../../src/adjoint/device/device.h"  // which card, and what the last call cost
#include "support/gpu_guard.h"
#include "support/test.h"

#include "r97/adjoint.h"
#include "r97/op.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using namespace r97;

namespace {

// The registry's table is a function-local static, so whichever translation unit touches it first
// constructs it, and its destructor is registered at that moment -- which has to be before the
// exit-time test runner arms, or the table is gone by the time a test reads it. A test file that
// carries its own registrations gets that for free (tests/host/adjoint_registry.cpp); this one only
// reads rows the library registers, and the library's objects initialise after the test objects on
// the link line, so it asks for the table itself, here, before its first R97_TEST below.
struct Ask_for_the_table {
    Ask_for_the_table() { (void)ops_registered(); }
} ask_for_the_table;

const double bf16_tol = 1e-2;

uint64_t mix(uint64_t* s) {
    uint64_t x = (*s += 0x9e3779b97f4a7c15ull);
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

double unit(uint64_t* s) { return static_cast<double>(mix(s) >> 11) * (1.0 / 9007199254740992.0); }

bool has(const std::string& s, const char* needle) { return s.find(needle) != std::string::npos; }

AdValue value_of(const Operand& o) {
    AdValue v;
    v.dtype = o.dtype;
    v.shape = o.shape;
    int64_t n = 1;
    for (int64_t d : o.shape) n *= d;
    v.value.assign(static_cast<std::size_t>(n), 0.0);
    return v;
}

struct Case {
    std::vector<AdValue> fwd;  // the full schema; the written operands are zero
    std::vector<AdValue> cot;  // one per written operand, in schema order
};

Case make_case(const OpRecord& rec, uint64_t seed) {
    Case c;
    for (const Operand& o : rec.operands) {
        AdValue v = value_of(o);
        if (o.written) {
            AdValue d = v;
            for (double& x : d.value) x = 2.0 * unit(&seed) - 1.0;
            c.cot.push_back(std::move(d));
        } else {
            for (double& x : v.value) {
                const uint64_t r = mix(&seed);
                if (o.dtype == "i8" || o.dtype == "u8") {
                    x = static_cast<double>(static_cast<int>(r % 9) - 4);
                } else if (has(o.buffer, "scale")) {
                    x = (static_cast<double>((r >> 16) % 16) + 1.0) / 256.0;  // power of two: exact
                } else if (o.dtype == "bf16") {
                    x = (static_cast<double>(static_cast<int>(r % 257) - 128)) / 128.0;
                } else {
                    x = 2.0 * unit(&seed) - 1.0;
                }
            }
        }
        c.fwd.push_back(std::move(v));
    }
    return c;
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

AdjointFn side_of(Side side, const OpRecord& rec) {
    std::string why;
    AdjointFn fn = adjoint_find(rec.op, side, rec.params, &why);
    if (fn == nullptr)
        throw std::runtime_error((side == Side::device ? "device" : "host") + std::string(" adjoint of ")
                                 + rec.op + ": " + why);
    return fn;
}

// One adjoint call. `fwd` and `dc` span the full schema -- the writes' slots of dc stay empty, the
// read operands' zeroed -- as the host checker hands them over.
std::vector<AdValue> run(AdjointFn fn, const OpRecord& rec, const Case& c) {
    std::vector<AdValue> dc(rec.operands.size());
    for (std::size_t i = 0; i < rec.operands.size(); ++i) {
        if (rec.operands[i].written) continue;
        dc[i] = c.fwd[i];
        dc[i].value.assign(dc[i].value.size(), 0.0);
    }
    AdjointCall call{rec, c.fwd, c.cot, &dc, nullptr, ""};
    fn(call);
    return dc;
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

// dx[m,k] = sum_n dy[m,n] * b[n,k] * s[n, k/128], the f64 sum the host adjoint forms.
double probe_ref(const OpRecord& rec, const Case& c, int64_t m, int64_t k) {
    const int ia = operand_pos(rec, "a");
    const int ib = operand_pos(rec, "b");
    const int ibs = operand_pos(rec, "b_scale");
    const int64_t K = c.fwd[static_cast<std::size_t>(ia)].shape[1];
    const int64_t N = c.fwd[static_cast<std::size_t>(ib)].shape[0];
    const std::vector<double>& dy = c.cot.front().value;
    const std::vector<double>& b = c.fwd[static_cast<std::size_t>(ib)].value;
    const std::vector<double>* s = ibs >= 0 ? &c.fwd[static_cast<std::size_t>(ibs)].value : nullptr;
    const int64_t groups = K / 128;
    double acc = 0.0;
    for (int64_t n = 0; n < N; ++n)
        acc += dy[static_cast<std::size_t>(m * N + n)] * b[static_cast<std::size_t>(n * K + k)] *
               (s != nullptr ? (*s)[static_cast<std::size_t>(n * groups + k / 128)] : 1.0);
    return acc;
}

// The device row and the host row on one record: the whole of dx, and the frozen positions.
double agree_fully(const OpRecord& rec, uint64_t seed) {
    const Case c = make_case(rec, seed);
    const int ia = operand_pos(rec, "a");
    const std::vector<AdValue> dev = run(side_of(Side::device, rec), rec, c);
    const std::vector<AdValue> host = run(side_of(Side::host, rec), rec, c);
    R97_CHECK(adjoint_frozen(rec.op, Side::device, rec.params) ==
              adjoint_frozen(rec.op, Side::host, rec.params));
    return rel_l2(dev[static_cast<std::size_t>(ia)].value, host[static_cast<std::size_t>(ia)].value);
}

// The same question over a spread sample of (m,k) instead of every column.
double agree_on_probes(const OpRecord& rec, uint64_t seed, int probes) {
    const Case c = make_case(rec, seed);
    const int ia = operand_pos(rec, "a");
    const std::vector<AdValue> dev = run(side_of(Side::device, rec), rec, c);
    const std::vector<double>& dx = dev[static_cast<std::size_t>(ia)].value;
    const int64_t K = c.fwd[static_cast<std::size_t>(ia)].shape[1];
    const int64_t M = c.fwd[static_cast<std::size_t>(ia)].shape[0];
    std::vector<double> got, want;
    got.reserve(static_cast<std::size_t>(probes));
    want.reserve(static_cast<std::size_t>(probes));
    for (int i = 0; i < probes; ++i) {
        const uint64_t r = mix(&seed);
        const int64_t m = static_cast<int64_t>(r % static_cast<uint64_t>(M));
        const int64_t k = static_cast<int64_t>((r >> 32) % static_cast<uint64_t>(K));
        got.push_back(dx[static_cast<std::size_t>(m * K + k)]);
        want.push_back(probe_ref(rec, c, m, k));
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

// One measured call: prints the staging and kernel milliseconds and the scratch the call held.
void measure(const char* what, const OpRecord& rec, uint64_t seed) {
    const Case c = make_case(rec, seed);
    const AdjointFn fn = side_of(Side::device, rec);
    run(fn, rec, c);  // the first launch loads the code object; measure the second
    const std::vector<AdValue> dc = run(fn, rec, c);
    const DeviceAdjointStats s = device_adjoint_stats();
    const int ia = operand_pos(rec, "a");
    const int64_t M = c.fwd[static_cast<std::size_t>(ia)].shape[0];
    const int64_t N = c.cot.front().shape[1];
    const int64_t K = c.fwd[static_cast<std::size_t>(ia)].shape[1];
    const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    const double tflops = s.kernel_ms > 0.0 ? flops / (s.kernel_ms * 1e9) : 0.0;
    std::printf("%-22s %-10s M=%5lld N=%6lld K=%6lld  staging %8.3f ms  kernel %8.3f ms  %6.1f "
                "TFLOP/s  scratch %4lld MiB\n",
                what, rec.op.c_str(), static_cast<long long>(M), static_cast<long long>(N),
                static_cast<long long>(K), s.staging_ms, s.kernel_ms, tflops,
                static_cast<long long>(s.scratch_bytes >> 20));
    std::fflush(stdout);
    R97_CHECK(!dc[static_cast<std::size_t>(ia)].value.empty());
}

// The guard: the card from R97_DEVICE, locked for the test's lifetime, with the VRAM the largest
// call below holds at once (dy and the codes staged, dx, and a little room).
r97test::GuardOptions guard_options() {
    r97test::GuardOptions o;
    const char* env = std::getenv("R97_DEVICE");
    o.device = r97test::parse_device(env != nullptr ? env : "0");
    o.bdf = r97test::hip_bdf(o.device);
    o.need_vram = int64_t(2) << 30;
    return o;
}

}  // namespace

R97_TEST(adjoint_linear_device_rows) {
    // What the family registers, reachable without a card: the two ops on Side::device, the same
    // frozen positions as their host rows, and the int8 predicate that keeps the kernel to the
    // 128-wide scale grid it serves. A missing row is the failure the loader would see; the card is
    // a separate question, asked below.
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
    r97::device_adjoint_set_device(o.device);
    std::printf("adjoint_linear: card %s\n", guard.card().c_str());

    // The tiny fixture's shapes, whole: the MLP and vocabulary widths of tests/host/adjoint_linear.cpp,
    // the fixture's eight-channel delta-net projection (padded to sixteen on the card) and its
    // ssm_out width.
    for (const OpRecord& rec : {gemm_nt_q_rec(4, 256, 128), gemm_nt_q_rec(1, 128, 256),
                                gemm_nt_q_rec(7, 320, 384), gemm_nt_q_rec(3, 8, 128),
                                gemm_nt_rec(4, 256, 128), gemm_nt_rec(1, 128, 256),
                                gemm_nt_rec(7, 320, 384), gemm_nt_rec(3, 8, 128),
                                gemm_nt_rec(2, 128, 64)}) {
        const double rel = agree_fully(rec, 0x2000 + rec.operands.size());
        report("agrees (whole)", rec, rel);
        R97_CHECK(rel <= bf16_tol);
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
        const double rel = agree_on_probes(r.rec, 0x3000, 512);
        report(r.name, r.rec, rel);
        R97_CHECK(rel <= bf16_tol);
    }
}

R97_TEST(adjoint_linear_device_measures) {
    const r97test::GuardOptions o = guard_options();
    const r97test::DeviceGuard guard(o);
    r97::device_adjoint_set_device(o.device);
    std::printf("adjoint_linear: card %s, one call per shape, staging (conversion and transfer) and "
                "kernel separately\n", guard.card().c_str());

    // Every linear of one transformer block at 1024 and at 4096 tokens, plus a packed batch of
    // three sequences of unequal length (317 + 1009 + 2697 = 4023 rows), which is what a step's
    // token count looks like.
    measure("attn_k/v", gemm_nt_q_rec(4096, 1024, 5120), 0x4001);
    measure("attn_output", gemm_nt_q_rec(4096, 5120, 6144), 0x4002);
    measure("attn_qg", gemm_nt_q_rec(1024, 12288, 5120), 0x4003);
    measure("attn_qg", gemm_nt_q_rec(4096, 12288, 5120), 0x4004);
    measure("ssm_inz", gemm_nt_q_rec(4096, 16384, 5120), 0x4005);
    measure("ffn_down", gemm_nt_q_rec(4096, 5120, 17408), 0x4006);
    measure("ffn_gate_up", gemm_nt_q_rec(1024, 34816, 5120), 0x4007);
    measure("ffn_gate_up", gemm_nt_q_rec(4096, 34816, 5120), 0x4008);
    measure("ffn_gate_up batch", gemm_nt_q_rec(4023, 34816, 5120), 0x4009);
    measure("ssm_out bf16", gemm_nt_rec(4096, 5120, 6144), 0x400a);
}
