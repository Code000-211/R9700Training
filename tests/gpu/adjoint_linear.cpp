// tests/gpu/adjoint_linear.cpp -- the frozen-linears family on the card.
//
// The device rows are the ones src/adjoint/device/linear.hip registers on Side::device; this file
// asks three questions of them.
//
//   agrees: the device row and the host row, handed the same record and the same values, write the
//   same dx -- through tests/support/device_check.h, the harness every device adjoint is held to.
//   At the fixture's shapes, and at the real model's hidden width with few rows, every element of
//   both ops is compared. At the model's real shapes -- 1024 and 4096 tokens, and a packed batch of
//   unequal lengths -- no test can pay for the host row of the whole tensor: 4096 rows of the
//   widest linear is about seven hundred GMAC of f64 on one thread for a single call. The case
//   states a spread of rows instead (`Options::rows`), the reference is the host adjoint on a
//   record of those rows, and the harness compares every element of them to the same tolerance.
//   The sample takes rows spread through the tensor, the two rows a 256-row band turns over at, and
//   the last two, so the partial tail of a batch is in it too; rows are whole, so every block of K
//   is.
//
//   The tolerance is the harness's 2% of the host element plus 1e-3, plus a tenth of the RMS of the
//   elements compared: a reduction's bf16 rounding error does not shrink when its sum cancels, so
//   the per-element rule alone would fail a near-zero element of a kernel that is right. The
//   relative L2 over the compared elements is held to 1e-2 as well.
//
//   measures: every real shape prints one line of adjoint milliseconds, forward-twin milliseconds
//   and scratch, as docs/kernels/README.md asks. The forward twin is timed once per shape rather
//   than as often as the adjoint, because at the widest linear the f64 twin is tens of minutes a
//   call; the adjoint is warmed up and averaged over three calls. docs/kernels/linear.md carries
//   the numbers.
//
//   teeth: the wrong kernels below -- the scale grid dropped, and half the reduction dropped -- are
//   run through the same harness, which has to reject them.
//
// The values follow tests/host/adjoint_linear.cpp -- integer codes, bf16-exact bf16 operands,
// power-of-two scales -- so that the two sides start from what the host test's kernel check starts
// from. The cotangent is uniform random, which is the point of the model-shape cases: it exercises
// the bf16 rounding of the gradient, not exactness.
//
// Label: gpu. The kernels launch on the card the guard reserved (R97_DEVICE selects it).
#include "support/device_check.h"
#include "support/test.h"

#include "r97/adjoint.h"
#include "r97/op.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace r97;

namespace {

const double bf16_tol = 1e-2;  // relative L2 over the elements the case compares
const double rms_share = 0.10; // the fraction of the RMS a near-zero element is allowed

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

// The same op over other rows: a host row reads its shape from the record, so the reference below
// builds one of just the rows a case samples.
OpRecord rec_rows(const OpRecord& rec, int64_t M) {
    const int64_t N = *std::get_if<int64_t>(&rec.params[1].second);
    const int64_t K = *std::get_if<int64_t>(&rec.params[2].second);
    return rec.op == "gemm_nt_q" ? gemm_nt_q_rec(M, N, K) : gemm_nt_rec(M, N, K);
}

int64_t rows_of(const OpRecord& rec) { return *std::get_if<int64_t>(&rec.params[0].second); }
int64_t cols_of(const OpRecord& rec) { return *std::get_if<int64_t>(&rec.params[1].second); }
int64_t inner_of(const OpRecord& rec) { return *std::get_if<int64_t>(&rec.params[2].second); }

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

// The rows a model-shape case compares: a spread across the tensor, the two rows the 256-row band
// turns over at, and the last two, so a batch's partial tail is in the sample as well as the whole
// bands. Rows are whole, so every block of K is compared; the reference costs a host row of these
// rows, which is some seconds a row for the widest linear and is why the sample is a handful.
std::vector<int64_t> sampled_rows(int64_t M, int spread = 3) {
    std::vector<int64_t> rows;
    for (int i = 0; i < spread; ++i) rows.push_back(M * i / spread);
    for (int64_t r : {int64_t(255), int64_t(256), M - 2, M - 1})
        if (r >= 0 && r < M) rows.push_back(r);
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows;
}

// The host row of a case's sampled rows: the same arithmetic the registry's host adjoint does, on a
// record of just those rows, scattered back into the slots the harness compares. The rest of dx is
// left as the harness found it (zero), which is what a case that samples rows asks for. `seconds`,
// when given, comes back with how long the reference took.
device_check::HostRef host_ref_on_rows(std::vector<int64_t> rows, double* seconds = nullptr) {
    return [rows, seconds](AdjointCall& call) {
        const auto started = std::chrono::steady_clock::now();
        std::string why;
        const AdjointFn host = adjoint_find(call.rec.op, Side::host, call.rec.params, &why);
        if (host == nullptr) {
            throw std::runtime_error("host adjoint of " + call.rec.op + ": " + why);
        }

        const int ia = operand_pos(call.rec, "a");
        if (ia < 0 || call.cot.empty())
            throw std::runtime_error("case for op '" + call.rec.op + "' has no a or no cotangent");
        const std::vector<int64_t> a_shape = call.fwd[static_cast<std::size_t>(ia)].shape;
        const int64_t K = a_shape[1];

        // The reads: the rows this case samples for the operands the tokens index, the weight and
        // its scale grid whole, since every row of dx reads all of them.
        const OpRecord sub = rec_rows(call.rec, static_cast<int64_t>(rows.size()));
        std::vector<AdValue> fwd, dc;
        for (std::size_t i = 0; i < sub.operands.size(); ++i) {
            const Operand& o = sub.operands[i];
            AdValue v = value_of(o);
            if (!o.written) {
                const AdValue& full = call.fwd[i];
                if (o.buffer == "a" || o.buffer == "a_scale") {
                    const int64_t cols = o.shape.back();
                    for (std::size_t j = 0; j < rows.size(); ++j) {
                        const int64_t from = rows[j] * cols;
                        std::copy_n(full.value.begin() + from, cols,
                                    v.value.begin() + static_cast<int64_t>(j) * cols);
                    }
                } else {
                    v.value = full.value;
                }
            }
            fwd.push_back(std::move(v));
        }

        // The cotangent: the sampled rows of the gradient the kernel was handed.
        std::vector<AdValue> cot;
        {
            const AdValue& full = call.cot.front();
            AdValue d;
            d.dtype = full.dtype;
            d.shape = {static_cast<int64_t>(rows.size()), full.shape[1]};
            d.value.resize((std::size_t)numel(d.shape));
            const int64_t cols = full.shape[1];
            for (std::size_t j = 0; j < rows.size(); ++j) {
                const int64_t from = rows[j] * cols;
                std::copy_n(full.value.begin() + from, cols,
                            d.value.begin() + static_cast<int64_t>(j) * cols);
            }
            cot.push_back(std::move(d));
        }

        dc.resize(sub.operands.size());
        for (std::size_t i = 0; i < sub.operands.size(); ++i) dc[i] = value_of(sub.operands[i]);
        AdjointCall sub_call{sub, fwd, cot, &dc, nullptr, ""};
        host(sub_call);

        AdValue& out = (*call.dc)[static_cast<std::size_t>(ia)];
        const AdValue& got = dc[static_cast<std::size_t>(ia)];
        for (std::size_t j = 0; j < rows.size(); ++j) {
            const int64_t to = rows[j] * K;
            for (int64_t k = 0; k < K; ++k) out.value[static_cast<std::size_t>(to + k)] +=
                got.value[static_cast<std::size_t>(static_cast<int64_t>(j) * K + k)];
        }
        if (seconds != nullptr) {
            *seconds = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - started).count();
        }
    };
}

// -- the wrong kernels the teeth case runs through the same harness ------------------------------

[[noreturn]] void wrong_case(const char* what) {
    throw std::runtime_error(std::string("wrong-kernel case wants ") + what);
}

// dx = dy * W on the codes, with the weight's per-block scale left out: what a kernel that reads
// the codes and forgets the grid returns.
void wrong_no_scale(const AdjointCall& call) {
    const int ia = operand_pos(call.rec, "a");
    const int ib = operand_pos(call.rec, "b");
    if (ia < 0 || ib < 0 || call.cot.empty()) wrong_case("a, b and a cotangent");
    const int64_t M = call.fwd[static_cast<std::size_t>(ia)].shape[0];
    const int64_t K = call.fwd[static_cast<std::size_t>(ia)].shape[1];
    const int64_t N = call.fwd[static_cast<std::size_t>(ib)].shape[0];
    const std::vector<double>& dy = call.cot.front().value;
    const std::vector<double>& b = call.fwd[static_cast<std::size_t>(ib)].value;
    AdValue& dx = (*call.dc)[static_cast<std::size_t>(ia)];
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            double acc = 0.0;
            for (int64_t n = 0; n < N; ++n)
                acc += dy[static_cast<std::size_t>(m * N + n)] *
                       b[static_cast<std::size_t>(n * K + k)];
            dx.value[static_cast<std::size_t>(m * K + k)] += acc;
        }
    }
}

// dx over the first half of the reduction only: what a kernel that misses a block of K returns.
void wrong_half_reduction(const AdjointCall& call) {
    const int ia = operand_pos(call.rec, "a");
    const int ib = operand_pos(call.rec, "b");
    const int ibs = operand_pos(call.rec, "b_scale");
    if (ia < 0 || ib < 0 || call.cot.empty()) wrong_case("a, b and a cotangent");
    const int64_t M = call.fwd[static_cast<std::size_t>(ia)].shape[0];
    const int64_t K = call.fwd[static_cast<std::size_t>(ia)].shape[1];
    const int64_t N = call.fwd[static_cast<std::size_t>(ib)].shape[0];
    const int64_t groups = ibs >= 0 ? call.fwd[static_cast<std::size_t>(ibs)].shape[1] : 0;
    const int64_t group = groups > 0 ? K / groups : K;
    const std::vector<double>& dy = call.cot.front().value;
    const std::vector<double>& b = call.fwd[static_cast<std::size_t>(ib)].value;
    const std::vector<double>* s =
        ibs >= 0 ? &call.fwd[static_cast<std::size_t>(ibs)].value : nullptr;
    AdValue& dx = (*call.dc)[static_cast<std::size_t>(ia)];
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            double acc = 0.0;
            for (int64_t n = 0; n < N / 2; ++n)
                acc += dy[static_cast<std::size_t>(m * N + n)] *
                       b[static_cast<std::size_t>(n * K + k)] *
                       (s != nullptr ? (*s)[static_cast<std::size_t>(n * groups + k / group)]
                                     : 1.0);
            dx.value[static_cast<std::size_t>(m * K + k)] += acc;
        }
    }
}

// -- the two shapes of case ---------------------------------------------------------------------

// The whole of dx through the harness, with the RMS term a cancelling reduction needs.
void agrees_whole(const device_check::Case& c) {
    device_check::Options o;
    o.allow.rms = rms_share;
    const device_check::Report r = device_check::check(c, nullptr, device_check::HostRef(), o);
    device_check::require(r);
    R97_CHECK(r.rel_l2 <= bf16_tol);
    R97_CHECK(r.scratch_peak > 0);
    std::printf("%-52s rel %.4f, every element, scratch peak %lld KiB\n", c.label.c_str(),
                r.rel_l2, static_cast<long long>(r.scratch_peak / 1024));
    std::fflush(stdout);
}

// The rows a model-shape case samples, the same tolerance, the host row of just those rows.
void agrees_rows(const device_check::Case& c) {
    device_check::Options o;
    o.allow.rms = rms_share;
    o.rows = sampled_rows(rows_of(c.rec));
    o.twin = false;  // the twin is the same unaffordable GEMM, and no written slot is read
    double host_seconds = 0.0;
    const device_check::Report r =
        device_check::check(c, nullptr, host_ref_on_rows(o.rows, &host_seconds), o);
    device_check::require(r);
    R97_CHECK(r.rel_l2 <= bf16_tol);
    R97_CHECK(r.scratch_peak > 0);
    std::printf("%-52s rel %.4f, %zu rows of %lld, host row %.1f s, scratch peak %lld KiB\n",
                c.label.c_str(), r.rel_l2, o.rows.size(),
                static_cast<long long>(rows_of(c.rec)), host_seconds,
                static_cast<long long>(r.scratch_peak / 1024));
    std::fflush(stdout);
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
        std::printf("%s: device frozen %zu operand(s), the host row's\n", rec.op.c_str(),
                    dev.size());
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
        const device_check::Case c =
            linear_case(rec, 0x2000 + static_cast<uint64_t>(cols_of(rec)),
                        rec.op + " fixture " + std::to_string(rows_of(rec)) + "x" +
                            std::to_string(cols_of(rec)) + "x" + std::to_string(inner_of(rec)));
        agrees_whole(c);
        if (r97test::bench()) device_check::print_bench(c, 20, 1);
    }

    // The real model's hidden width with few rows, whole: five scale blocks of K, both ops.
    for (const OpRecord& rec : {gemm_nt_q_rec(16, 5120, 5120), gemm_nt_rec(16, 5120, 5120)}) {
        const device_check::Case c =
            linear_case(rec, 0x2100, rec.op + " hidden width 16x5120x5120 whole");
        agrees_whole(c);
        if (r97test::bench()) device_check::print_bench(c, 20, 1);
    }

    // The real model's linears at one and four thousand tokens, on sampled rows: the hidden width,
    // the MLP's intermediate and down widths, the attention projections' widths, one bf16 GEMM, and
    // a batch of unequal lengths.
    const std::vector<OpRecord> real = {
        gemm_nt_q_rec(1024, 1024, 5120),   // attn k/v
        gemm_nt_q_rec(4096, 1024, 5120),
        gemm_nt_q_rec(4096, 5120, 6144),   // attn output
        gemm_nt_q_rec(1024, 12288, 5120),  // attn q/g
        gemm_nt_q_rec(4096, 12288, 5120),
        gemm_nt_q_rec(4096, 16384, 5120),  // ssm in/z
        gemm_nt_q_rec(4096, 5120, 17408),  // ffn down
        gemm_nt_q_rec(4096, 34816, 5120),  // ffn gate/up, the widest
        gemm_nt_rec(4096, 5120, 6144),     // ssm out, bf16
        // Four thousand and twenty-three rows of the widest linear: a batch of unequal lengths
        // whose tail leaves both the 256-row band and the sixteen rows of one wave's tile partial.
        gemm_nt_q_rec(4023, 34816, 5120),
    };
    for (const OpRecord& rec : real) {
        const device_check::Case c =
            linear_case(rec, 0x3000, rec.op + " " + std::to_string(rows_of(rec)) + "x" +
                                         std::to_string(cols_of(rec)) + "x" +
                                         std::to_string(inner_of(rec)));
        agrees_rows(c);
    }
}

R97_TEST(adjoint_linear_device_teeth) {
    // The comparison has teeth: a kernel that drops the weight's scale grid, and one that drops
    // half the reduction, are kernels this harness has to reject.
    const r97test::GuardOptions o = guard_options();
    const r97test::DeviceGuard guard(o);
    (void)device_check::load_plugin();
    device_check::require_device(o.device);

    const std::pair<device_check::Case, AdjointFn> wrong[] = {
        {linear_case(gemm_nt_q_rec(4, 256, 128), 0x5000, "int8, scale grid dropped"),
         wrong_no_scale},
        {linear_case(gemm_nt_rec(4, 256, 128), 0x5001, "bf16, half the reduction"),
         wrong_half_reduction},
        {linear_case(gemm_nt_q_rec(4, 256, 128), 0x5002, "int8, half the reduction"),
         wrong_half_reduction},
    };
    for (const auto& [c, kernel] : wrong) {
        const device_check::Report r = device_check::check(c, kernel);
        R97_CHECK(!r.ok);
        R97_CHECK(r.first_slot == operand_pos(c.rec, "a"));
        std::printf("%s: rejected, %lld mismatches, first at dc[%lld] index %lld\n",
                    c.label.c_str(), static_cast<long long>(r.mismatches),
                    static_cast<long long>(r.first_slot),
                    static_cast<long long>(r.first_index));
        std::fflush(stdout);
    }
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
    // token count looks like. The q/g projection runs at 1024 rows for its 12288-wide form (where
    // its own width dominates) and at 4096 for the others.
    struct Measure {
        OpRecord rec;
        uint64_t seed;
        const char* label;
    };
    const std::vector<Measure> shapes = {
        {gemm_nt_q_rec(1024, 1024, 5120), 0x4001, "1024 tok, k/v"},
        {gemm_nt_q_rec(4096, 1024, 5120), 0x4002, "4096 tok, k/v"},
        {gemm_nt_q_rec(4096, 5120, 6144), 0x4003, "4096 tok, attn output"},
        {gemm_nt_q_rec(1024, 12288, 5120), 0x4004, "1024 tok, q/g"},
        {gemm_nt_q_rec(4096, 12288, 5120), 0x4005, "4096 tok, q/g"},
        {gemm_nt_q_rec(4096, 16384, 5120), 0x4006, "4096 tok, ssm in/z"},
        {gemm_nt_q_rec(4096, 5120, 17408), 0x4007, "4096 tok, ffn down"},
        {gemm_nt_q_rec(1024, 34816, 5120), 0x4008, "1024 tok, ffn gate/up"},
        {gemm_nt_q_rec(4096, 34816, 5120), 0x4009, "4096 tok, ffn gate/up"},
        {gemm_nt_q_rec(4023, 34816, 5120), 0x400a, "batch 317+1009+2697, ffn gate/up"},
        {gemm_nt_rec(4096, 5120, 6144), 0x400b, "4096 tok, ssm out bf16"},
    };
    for (const Measure& m : shapes) {
        const device_check::Timing t =
            device_check::print_bench(linear_case(m.rec, m.seed, m.label), 3, 1);
        R97_CHECK(t.adjoint_ms > 0.0);
        R97_CHECK(t.forward_ms > 0.0);
    }
}
