// tests/gpu/head_ce.cpp -- the head's loss and gradient on the card, against the reference loss.
//
// What the issue asks of the device loss, and where each part is checked:
//   * the row's arithmetic (head_row.h) is the kernel's, so the f32 log-sum-exp, the loss and the
//     gradient are pinned against ce_position without a card;
//   * the chunk loop on the real vocabulary (248320 wide, the container's) agrees with ce_sum, for
//     chunk sizes 1, 64 and 512: the summed loss and every gradient value;
//   * so does the tiny fixture's own recorded logits -- three sequences of 140, 73 and 31 tokens,
//     unscored rows included, vocab 320 (docs/FIXTURES.md) -- against ce_loss and ce_grad;
//   * the Objective form (HeadCeObjective) is CeObjective's loss and cotangent on a tape, so the
//     device loss can stand where the reference does;
//   * logits near 1e30 stay finite, which is what the two-piece log-sum-exp is for.
//
// Needs one R9700 (gfx1201, 32 GB) or similar, under the GPU test guard; no container is read, so
// the file does not carry the `model` label. --bench prints the cost of 1000 positions, the line
// docs/kernels/head.md quotes.
#include "support/bench.h"
#include "support/device_check.h"
#include "support/gpu_guard.h"
#include "support/oracle.h"
#include "support/test.h"
#include "support/vector_tape.h"

#include "adjoint/device/head_row.h"
#include "adjoint/device/scratch.h"

#include "r97/head.h"
#include "r97/loss.h"
#include "r97/params.h"

#include "rad_device.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace r97;

// ------------------------------------------------------------------- comparing against the loss

void require_close(const std::string& what, int64_t index, double got, double want,
                   double rel = device_check::kRelTol, double abs = device_check::kAbsTol) {
    const double tol = abs + rel * std::abs(want);
    if (!(std::abs(got - want) <= tol) || std::isnan(got)) {
        char buf[512];
        std::snprintf(buf, sizeof buf,
                      "%s[%lld]: %g where the reference loss says %g (tolerance %.3g)", what.c_str(),
                      (long long)index, got, want, tol);
        throw std::runtime_error(buf);
    }
}

void require_copy(int rc, const char* what) {
    if (rc != 0)
        throw std::runtime_error(std::string("head_ce: ") + what + ": " + rad_dev_last_error());
}

// ----------------------------------------------------------------------------- the tiny fixture

// The fixture's recorded logits, one row per prediction: a sequence of L tokens has L-1 rows, so
// the reference's last row of a sequence (which predicts nothing) is dropped, as ce_loss counts
// them. The values are the reference's f64 logits rounded to bf16 and widened again, so both sides
// of every comparison see the bits the card sees.
struct FixtureRows {
    Batch batch;
    std::vector<float> logits;  // rows x vocab
    std::vector<int32_t> ids;   // per row: the target, or -1 where the row is not scored
    int64_t rows = 0;
    int64_t vocab = 0;
};

FixtureRows fixture_rows() {
    const std::filesystem::path dir =
        std::filesystem::path(__FILE__).parent_path().parent_path() / "fixtures" / "oracle";
    const oracle::detail::File ref((dir / "reference.safetensors").string());

    FixtureRows out;
    for (int s = 0; s < 3; ++s) {
        const std::string prefix = "seq" + std::to_string(s) + "/";
        const oracle::Tensor& tokens = ref.tensor(prefix + "input_ids");
        const oracle::Tensor& score = ref.tensor(prefix + "score");
        const oracle::Tensor& logits = ref.tensor(prefix + "logits_without_adapter");
        if (logits.shape.size() != 2)
            throw std::runtime_error("head_ce: the fixture's sequence " + std::to_string(s)
                                     + " has a shape this test does not expect");
        if (out.vocab == 0) out.vocab = logits.shape[1];
        if (logits.shape[0] != int64_t(tokens.data.size()) || logits.shape[1] != out.vocab ||
            score.data.size() != tokens.data.size() || logits.shape[0] < 2)
            throw std::runtime_error("head_ce: the fixture's sequence " + std::to_string(s)
                                     + " has a shape this test does not expect");
        TokenSeq seq;
        seq.ids.reserve(tokens.data.size());
        for (double v : tokens.data) seq.ids.push_back(int32_t(v));
        seq.score.reserve(score.data.size());
        for (double v : score.data) seq.score.push_back(uint8_t(v != 0.0));

        // Rows 0..L-2 of the sequence, widened from the reference through bf16.
        for (int64_t row = 0; row + 1 < int64_t(seq.ids.size()); ++row) {
            for (int64_t v = 0; v < out.vocab; ++v) {
                const double want = logits.data[size_t(row * out.vocab + v)];
                out.logits.push_back(head_bf16_to_f32(head_bf16_from_f32(float(want))));
            }
        }
        out.batch.push_back(std::move(seq));
    }
    out.rows = int64_t(out.logits.size()) / out.vocab;
    out.ids.assign(size_t(out.rows), -1);
    for (const ScoredRow& r : ce_grad_rows(out.batch, 0, out.rows)) out.ids[size_t(r.row)] = r.id;
    return out;
}

// ----------------------------------------------------------------------- the walk over the card

struct Walk {
    double loss = 0.0;
    int64_t scored = 0;
    std::vector<float> grad;  // rows x vocab, zeroed where the rows are not scored
};

// Runs the chunk loop over rows the host holds: each chunk is uploaded, the kernel runs on it, and
// its gradient comes back to the row it belongs to. The staging vectors outlive the loop, because
// a chunk's upload is only known to have landed once the loop has synchronised it.
Walk run_loop(const std::vector<float>& logits, const std::vector<int32_t>& ids, int64_t rows,
              int64_t vocab, int64_t chunk, double grad_scale) {
    Walk out;
    out.grad.assign(size_t(rows * vocab), 0.0f);

    std::vector<uint16_t> up;
    std::vector<uint16_t> down;
    int64_t cursor = 0;

    HeadCeWindow window;
    window.rows = rows;
    window.vocab = vocab;
    window.chunk = chunk;
    window.grad_scale = grad_scale;

    const HeadChunkFill fill = [&](HeadChunk& c) {
        HeadChunkRows answer;
        const int64_t n = std::min(c.positions, rows - cursor);
        if (n <= 0) return answer;
        up.resize(size_t(n * vocab));
        for (int64_t i = 0; i < n * vocab; ++i)
            up[size_t(i)] = head_bf16_from_f32(logits[size_t(cursor * vocab + i)]);
        require_copy(rad_memcpy_async(c.logits, up.data(), n * vocab * int64_t(sizeof(uint16_t)),
                                      static_cast<RadStream>(c.stream)),
                     "uploading a chunk's logits");
        answer.rows = n;
        answer.ids.assign(ids.begin() + cursor, ids.begin() + cursor + n);
        return answer;
    };

    const HeadGradSink sink = [&](const HeadChunk& c) {
        const int64_t n = c.positions;
        down.resize(size_t(n * vocab));
        require_copy(rad_memcpy_async(down.data(), c.grad, n * vocab * int64_t(sizeof(uint16_t)),
                                      static_cast<RadStream>(c.stream)),
                     "downloading a chunk's gradient");
        require_copy(rad_stream_sync(static_cast<RadStream>(c.stream)),
                     "waiting for a chunk's gradient");
        for (int64_t i = 0; i < n * vocab; ++i)
            out.grad[size_t(cursor * vocab + i)] = head_bf16_to_f32(down[size_t(i)]);
        cursor += n;
    };

    const HeadCeSum sum = head_ce_loop(window, fill, sink);
    out.loss = sum.loss;
    out.scored = sum.scored;
    return out;
}

// ------------------------------------------------------------------------------- the row's maths

// No card: the row arithmetic the kernel and its checks share, against the reference position.
void row_arithmetic() {
    uint64_t seed = 0x4ce00001ull;
    const struct {
        int64_t rows;
        int64_t vocab;
    } shapes[] = {{4, 320}, {2, 248320}};

    for (const auto& shape : shapes) {
        std::vector<float> logits(size_t(shape.rows * shape.vocab));
        for (float& x : logits) x = float(device_check::urand(&seed) * 2.0 - 1.0);
        std::vector<uint16_t> bf16(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) bf16[i] = head_bf16_from_f32(logits[i]);

        const double scale = 1.0 / double(shape.rows);
        for (int64_t row = 0; row < shape.rows; ++row) {
            const uint16_t* x = bf16.data() + row * shape.vocab;
            const int32_t target = int32_t(device_check::splitmix64(&seed) % uint64_t(shape.vocab));

            HeadRowExp e;
            e.max = head_row_max(x, shape.vocab, 0, 1);
            e.lse = logf(head_row_sumexp(x, shape.vocab, e.max, 0, 1));

            std::vector<float> want(size_t(shape.vocab));
            std::vector<float> wide(size_t(shape.vocab));
            for (int64_t j = 0; j < shape.vocab; ++j) wide[size_t(j)] = head_bf16_to_f32(x[j]);
            const double want_loss = ce_position(wide.data(), shape.vocab, target, want.data(), scale);

            const std::string where = std::to_string(shape.vocab) + "-wide row " + std::to_string(row);
            require_close(where + " loss", 0, head_row_loss(x, e, target), want_loss, 1e-3, 1e-3);
            for (int64_t j = 0; j < shape.vocab; ++j)
                require_close(where + " grad", j, head_row_grad(x, e, j, target, float(scale)),
                              want[size_t(j)], 1e-3, 1e-4);
        }

        // The stable form, on logits a float holds but an exp of does not: the loss stays finite
        // and the whole softmax lands on the maximum, as the reference's f64 does.
        std::vector<uint16_t> extreme(size_t(shape.vocab));
        std::vector<float> extreme_wide(size_t(shape.vocab));
        for (int64_t j = 0; j < shape.vocab; ++j) {
            extreme[size_t(j)] = head_bf16_from_f32(j == 0 ? -1e30f : 1e30f);
            extreme_wide[size_t(j)] = head_bf16_to_f32(extreme[size_t(j)]);
        }
        HeadRowExp e;
        e.max = head_row_max(extreme.data(), shape.vocab, 0, 1);
        e.lse = logf(head_row_sumexp(extreme.data(), shape.vocab, e.max, 0, 1));
        std::vector<float> extreme_want(size_t(shape.vocab));
        for (int32_t target : {int32_t(0), int32_t(1)}) {
            const double want = ce_position(extreme_wide.data(), shape.vocab, target,
                                            extreme_want.data(), 1.0);
            require_close("1e30 row loss", target, head_row_loss(extreme.data(), e, target), want,
                          1e-6, 1e6);
            for (int64_t j = 0; j < shape.vocab; ++j) {
                const float g = head_row_grad(extreme.data(), e, j, target, 1.0f);
                R97_CHECK(std::isfinite(g));
                require_close("1e30 row grad", j, g, extreme_want[size_t(j)], 1e-6, 1e-6);
            }
        }
    }
    std::printf("head_row: the row's arithmetic matches the reference loss\n");
}

// ------------------------------------------------------------------------------- the chunk loop

// The real vocabulary, chunk sizes 1, 64 and 512: the loop's summed loss and every gradient value
// are ce_sum's on the same bits.
void chunks_on_the_real_vocabulary() {
    constexpr int64_t kRows = 3;
    constexpr int64_t kVocab = 248320;  // the container's, docs/DESIGN.md

    uint64_t seed = 0x4ce00002ull;
    std::vector<float> logits(size_t(kRows * kVocab));
    for (float& x : logits) x = float(device_check::urand(&seed) * 2.0 - 1.0);
    std::vector<float> wide(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) wide[i] = head_bf16_to_f32(head_bf16_from_f32(logits[i]));

    std::vector<ScoredRow> scored;
    std::vector<int32_t> ids(size_t(kRows), -1);
    for (int64_t row = 0; row < kRows; ++row) {
        const int32_t id = int32_t(device_check::splitmix64(&seed) % uint64_t(kVocab));
        scored.push_back(ScoredRow{row, id});
        ids[size_t(row)] = id;
    }
    std::vector<float> want(size_t(kRows * kVocab));
    const double scale = 1.0 / double(kRows);
    const CeSum want_sum = ce_sum(wide.data(), kVocab, scored, want.data(), scale);

    for (int64_t chunk : {int64_t(1), int64_t(64), int64_t(512)}) {
        const Walk got = run_loop(wide, ids, kRows, kVocab, chunk, scale);
        const std::string where = "248320-wide, chunk " + std::to_string(chunk);
        R97_CHECK(got.scored == want_sum.scored);
        require_close(where + " loss", 0, got.loss, want_sum.loss);
        for (int64_t i = 0; i < kRows * kVocab; ++i)
            require_close(where + " grad", i, got.grad[size_t(i)], want[size_t(i)]);
        std::printf("head_ce: %s: loss %.6f, gradient agrees\n", where.c_str(), got.loss);
    }
}

// The tiny fixture's recorded logits, its own score mask (unscored rows and all), chunk sizes 1,
// 64 and 512, against ce_loss and ce_grad.
void chunks_on_the_fixture() {
    const FixtureRows f = fixture_rows();
    R97_CHECK(f.rows == 241);
    R97_CHECK(f.vocab == 320);

    const CeSum want_sum = ce_loss(f.batch, f.logits, f.vocab);
    R97_CHECK(want_sum.scored > 0);
    std::vector<float> want(size_t(f.rows * f.vocab), 0.0f);
    ce_grad(f.batch, f.logits.data(), f.vocab, 1.0 / double(want_sum.scored), want.data());

    for (int64_t chunk : {int64_t(1), int64_t(64), int64_t(512)}) {
        const Walk got = run_loop(f.logits, f.ids, f.rows, f.vocab, chunk, 1.0 / double(want_sum.scored));
        const std::string where = "fixture, chunk " + std::to_string(chunk);
        R97_CHECK(got.scored == want_sum.scored);
        require_close(where + " loss", 0, got.loss, want_sum.loss);
        for (int64_t i = 0; i < f.rows * f.vocab; ++i)
            require_close(where + " grad", i, got.grad[size_t(i)], want[size_t(i)]);
        std::printf("head_ce: %s: loss %.6f over %lld scored, gradient agrees\n", where.c_str(),
                    got.loss, (long long)got.scored);
    }
}

// The Objective form stands where the reference does: on a tape of the fixture's logits, one window
// per sequence, HeadCeObjective is CeObjective's loss and cotangent.
void objective_on_the_fixture() {
    const FixtureRows f = fixture_rows();
    int64_t offset = 0;
    for (int64_t seq = 0; seq < int64_t(f.batch.size()); ++seq) {
        const int64_t rows = int64_t(f.batch[size_t(seq)].ids.size()) - 1;
        const std::vector<float> window(f.logits.begin() + offset * f.vocab,
                                        f.logits.begin() + (offset + rows) * f.vocab);
        r97test::VectorTape tape;
        tape.set_batch(f.batch).set_scored(97).set_buffer("logits", {rows, f.vocab}, window);

        const ObjectiveResult want = CeObjective(tape, ParamSet{});
        R97_CHECK(want.seeds.size() == 1);
        for (int64_t chunk : {int64_t(1), int64_t(64), int64_t(512)}) {
            const Objective got = HeadCeObjective(chunk);
            const ObjectiveResult have = got(tape, ParamSet{});
            const std::string where = "fixture window " + std::to_string(seq) + ", chunk "
                                      + std::to_string(chunk);
            R97_CHECK(have.scored == want.scored);
            require_close(where + " loss", 0, have.loss, want.loss);
            R97_CHECK(have.seeds.size() == 1);
            R97_CHECK(have.seeds[0].buffer == "logits");
            R97_CHECK(have.seeds[0].row_lo == want.seeds[0].row_lo);
            R97_CHECK(have.seeds[0].row_hi == want.seeds[0].row_hi);
            R97_CHECK(have.seeds[0].cotangent.size() == want.seeds[0].cotangent.size());
            for (size_t i = 0; i < want.seeds[0].cotangent.size(); ++i)
                require_close(where + " cotangent", int64_t(i), have.seeds[0].cotangent[i],
                              want.seeds[0].cotangent[i]);
        }
        std::printf("head_ce: fixture window %lld (%lld rows, total_scored 97): the Objective "
                    "matches CeObjective\n",
                    (long long)seq, (long long)rows);
        offset += rows;
    }
}

// --------------------------------------------------------------------------------- the outliers

// Logits a float holds and an exp of does not: every value stays finite, and the reference's f64
// answer is still the one the card gives.
void very_large_logits_stay_finite() {
    constexpr int64_t kRows = 2;
    for (int64_t vocab : {int64_t(320), int64_t(248320)}) {
        std::vector<float> logits(size_t(kRows * vocab));
        for (int64_t row = 0; row < kRows; ++row) {
            for (int64_t j = 0; j < vocab; ++j) {
                // One row with a spread a float loses everything below, one row of one value.
                const float v = row == 0 ? (j % 2 == 0 ? 1e30f : -1e30f) : 3e38f;
                logits[size_t(row * vocab + j)] = v;
            }
        }
        std::vector<float> wide(logits.size());
        for (size_t i = 0; i < logits.size(); ++i)
            wide[i] = head_bf16_to_f32(head_bf16_from_f32(logits[i]));

        std::vector<int32_t> ids{0, 1};
        std::vector<ScoredRow> scored{{0, 0}, {1, 1}};
        std::vector<float> want(size_t(kRows * vocab));
        const double scale = 1.0 / double(kRows);
        const CeSum want_sum = ce_sum(wide.data(), vocab, scored, want.data(), scale);

        const Walk got = run_loop(wide, ids, kRows, vocab, kRows, scale);
        R97_CHECK(std::isfinite(got.loss));
        require_close(std::to_string(vocab) + "-wide large loss", 0, got.loss, want_sum.loss);
        for (int64_t i = 0; i < kRows * vocab; ++i) {
            R97_CHECK(std::isfinite(got.grad[size_t(i)]));
            require_close(std::to_string(vocab) + "-wide large grad", i, got.grad[size_t(i)],
                          want[size_t(i)]);
        }
        std::printf("head_ce: %lld-wide logits at 1e30 and 3e38: loss %.6g, gradient finite\n",
                    (long long)vocab, got.loss);
    }
}

// ------------------------------------------------------------------------------------ the cost

// What docs/kernels/head.md quotes: 1000 positions through the loop, and one chunk's memory. The
// upload is counted once, so the line is the head product's side of the join -- the kernel, the
// losses coming back and nothing else.
void bench_the_loop() {
    constexpr int64_t kRows = 256;   // four chunks of the real width, a fifth of 1000 positions
    constexpr int64_t kChunk = kHeadChunkRows;
    constexpr int64_t kVocab = 248320;

    uint64_t seed = 0x4ce00003ull;
    const int64_t chunk_bytes = kChunk * kVocab;
    std::vector<uint16_t> logits(size_t(chunk_bytes), 0);
    for (uint16_t& x : logits) x = head_bf16_from_f32(float(device_check::urand(&seed) * 2.0 - 1.0));
    std::vector<int32_t> ids(size_t(kChunk), 0);

    double loss = 0.0;
    int64_t rows_done = 0;
    bool uploaded = false;
    device::scratch_reset();

    const HeadCeWindow window{kRows, kVocab, kChunk, 1.0 / double(kRows)};
    const HeadChunkFill fill = [&](HeadChunk& c) {
        HeadChunkRows answer;
        const int64_t n = std::min(c.positions, kRows - rows_done);
        if (n <= 0) return answer;
        if (!uploaded) {  // one upload: what the loop costs is the kernel and the losses coming back
            require_copy(rad_memcpy_async(c.logits, logits.data(), n * kVocab * int64_t(sizeof(uint16_t)),
                                          static_cast<RadStream>(c.stream)),
                         "uploading the bench's rows");
            uploaded = true;
        }
        answer.rows = n;
        answer.ids.assign(ids.begin(), ids.begin() + n);
        return answer;
    };
    const HeadGradSink sink = [](const HeadChunk&) {};

    const auto t0 = std::chrono::steady_clock::now();
    const HeadCeSum sum = head_ce_loop(window, fill, sink);
    const auto t1 = std::chrono::steady_clock::now();
    loss = sum.loss;

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const int64_t peak = device::scratch_stats().peak_bytes;
    device::scratch_reset();
    std::printf("head ce: %.3f ms per 1000 positions (%lld rows of %lld, chunk %lld, one upload); "
                "loss %.6g; one chunk's scratch peak %lld KiB\n",
                ms * 1000.0 / double(kRows), (long long)kRows, (long long)kVocab, (long long)kChunk,
                loss, (long long)(peak / 1024));
}

void run() {
    // The row arithmetic and the fixture's rows need no card; the rest does.
    row_arithmetic();

    const auto options = device_check::guard_options();
    const r97test::DeviceGuard guard(options);
    device_check::require_device(options.device);
    (void)device_check::load_plugin();
    R97_CHECK(head_ce_find() != nullptr);

    chunks_on_the_real_vocabulary();
    chunks_on_the_fixture();
    objective_on_the_fixture();
    very_large_logits_stay_finite();
    if (r97test::bench()) bench_the_loop();
}

}  // namespace

R97_TEST(head_ce) { run(); }
