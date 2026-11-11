// tests/host/forward_batch.cpp -- the training forward on the host: a Batch in, the loss of its
// objective out, and the step kept for the backward. Label: host, so no card is needed.
// docs/FORWARD.md says what the driver is for; docs/FIXTURES.md says how a test gets a fixture.
//
// Every value here is checked against radiance's own prefill (tests/support/fixture.h's
// run_prefill), which is what the fixture was made for: over it the driver adds the batch's row
// arithmetic, the objective's windows and the step's record, and a loss of its own nothing.
//
// The driver and the fixture's prefill are both patches on radiance (the training set); a serving
// build compiles this file to nothing rather than being named by a build rule of its own.
#include "r97/forward.h"

#include "support/fixture.h"
#include "support/test.h"

#if R97_FORWARD && R97_FIXTURE_PREFILL

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "r97/loss.h"
#include "r97/model.h"
#include "support/fixture_w8a8.h"

namespace {

using r97::Batch;
using r97::TokenSeq;

// A prompt of byte ids: the tiny tokenizer is byte-level, so ids 0..255 are the bytes.
std::vector<int32_t> prompt(size_t n, int32_t first) {
    std::vector<int32_t> ids(n);
    for (size_t i = 0; i < n; ++i) ids[i] = (first + 7 * (int32_t)i) % 256;
    return ids;
}

// The batch of `sequences`, every token after the first a target.
Batch batch_of(const std::vector<std::vector<int32_t>>& sequences) {
    Batch batch;
    for (const std::vector<int32_t>& ids : sequences) {
        TokenSeq s;
        s.ids = ids;
        s.score.assign(ids.size(), 1);
        s.score[0] = 0;  // nothing predicts the first token
        batch.push_back(std::move(s));
    }
    return batch;
}

// The driver on the fixture's container, one window unless a case asks for chunks.
r97::ForwardConfig forward_config(const r97::Fixture& f) {
    r97::ForwardConfig cfg;
    cfg.model = f.container;
    cfg.radiance_home = r97::detail::radiance_home();
    cfg.window_rows = 1 << 20;
    return cfg;
}

std::unique_ptr<r97::ForwardDriver> open_forward(const r97::ForwardConfig& cfg) {
    std::string err;
    std::unique_ptr<r97::ForwardDriver> driver = r97::ForwardDriver::open(cfg, &err);
    if (!driver) throw std::runtime_error(err);
    return driver;
}

// The model these cases train: the reference loss on the fixture's container. Named here rather
// than taken from the registry, so these cases test the driver's arity and not the `ce` name.
r97::Model ce_model() {
    r97::Model model;
    model.objective = r97::CeObjective;
    return model;
}

// The reference's rows for a batch: L-1 per sequence, the sequences in the batch's order.
std::vector<float> reference_rows(const std::vector<r97::Prefill>& pre, const Batch& batch,
                                  int64_t vocab) {
    std::vector<float> rows;
    for (size_t i = 0; i < batch.size(); ++i) {
        const size_t n = (batch[i].ids.size() - 1) * size_t(vocab);
        rows.insert(rows.end(), pre[i].logits.begin(), pre[i].logits.begin() + n);
    }
    return rows;
}

void same_double(double got, double want, double tol, const std::string& what) {
    if (!(std::fabs(got - want) <= tol * (1.0 + std::fabs(want))))
        throw std::runtime_error(what + ": " + std::to_string(got) + " against " +
                                 std::to_string(want));
}

// Whether `what` appears in the message `fn` throws.
void throws_saying(const std::function<void()>& fn, const std::string& what) {
    try {
        fn();
    } catch (const std::exception& e) {
        if (std::string(e.what()).find(what) == std::string::npos)
            throw std::runtime_error("threw '" + std::string(e.what()) + "', wanted '" + what + "'");
        return;
    }
    throw std::runtime_error("did not throw, wanted '" + what + "'");
}

// An objective this test writes itself: the sum of one logit column over the scored rows of the
// window. Not a loss (it is not normalised and it differentiates nothing), which is the point --
// the driver carries whatever an Objective returns.
class ColumnObjective {
public:
    explicit ColumnObjective(int64_t column) : column_(column) {}

    r97::ObjectiveResult operator()(const r97::ForwardTape& fwd, const r97::ParamSet&) const {
        r97::ObjectiveResult out;
        int64_t lo = 0, hi = 0;
        fwd.window("logits", &lo, &hi);
        if (lo >= hi) return out;

        const int64_t vocab = fwd.shape("logits")[1];
        const std::vector<float> data = fwd.read("logits", lo, hi, /*differentiated=*/false);

        int64_t row = 0;
        for (const TokenSeq& s : fwd.batch()) {
            for (size_t i = 1; i < s.ids.size(); ++i) {
                if (s.score[i] && row >= lo && row < hi) {
                    out.loss += (double)data[size_t(row - lo) * size_t(vocab) + size_t(column_)];
                    ++out.scored;
                }
                ++row;
            }
        }
        return out;
    }

private:
    int64_t column_;
};

// What the column objective should return, straight off the reference's rows.
double column_of(const Batch& batch, const std::vector<float>& rows, int64_t vocab, int64_t column) {
    double sum = 0.0;
    int64_t row = 0;
    for (const TokenSeq& s : batch) {
        for (size_t i = 1; i < s.ids.size(); ++i) {
            if (s.score[i]) sum += (double)rows[size_t(row) * size_t(vocab) + size_t(column)];
            ++row;
        }
    }
    return sum;
}

// An objective that reads the window as differentiated and seeds nothing: the driver has to refuse
// it, since a read the loss depends on must be answered by a seed.
class UnseededObjective {
public:
    r97::ObjectiveResult operator()(const r97::ForwardTape& fwd, const r97::ParamSet&) const {
        r97::ObjectiveResult out;
        int64_t lo = 0, hi = 0;
        fwd.window("logits", &lo, &hi);
        if (lo >= hi) return out;
        fwd.read("logits", lo, hi, /*differentiated=*/true);
        out.loss = 1.0;
        out.scored = 1;
        return out;
    }
};

// One block boundary as a case saw it: the view's shape, and the first few values copied while the
// boundary still held them (the next block writes over the residual).
struct SeenBoundary {
    int64_t block = 0;
    bool begin = true;
    const void* at = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    std::string dtype;
    std::vector<float> head;
};

// The hook the driver calls at every bracket, recording each boundary into `seen`.
std::function<void(const r97::BlockView&)> watch_boundaries(std::vector<SeenBoundary>& seen,
                                                            size_t head) {
    return [&seen, head](const r97::BlockView& view) {
        SeenBoundary b;
        b.block = view.block;
        b.begin = view.begin;
        b.at = view.residual;
        b.rows = view.rows;
        b.cols = view.cols;
        b.dtype = view.dtype;
        // The residual is the container's element type, which for both fixtures is bf16.
        const uint16_t* v = static_cast<const uint16_t*>(view.residual);
        for (size_t i = 0; i < head && int64_t(i) < view.rows * view.cols; ++i)
            b.head.push_back(r97::bf16_to_f32(v[i]));
        seen.push_back(std::move(b));
    };
}

bool any_differ(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return true;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) return true;
    return false;
}

}  // namespace

// The loss of a batch is the reference loss computed from the same prefill's logits: the driver
// moves rows and counts, and the objective owns the arithmetic.
R97_TEST(forward_loss_is_the_reference_loss) {
    const r97::Fixture& f = r97::fixture();
    const std::vector<std::vector<int32_t>> prompts = {prompt(21, 5)};
    const Batch batch = batch_of(prompts);
    const std::vector<r97::Prefill> pre = r97::run_prefill(f, prompts);
    const std::vector<float> rows = reference_rows(pre, batch, f.n_vocab);
    const r97::CeSum want = r97::ce_loss(batch, rows, f.n_vocab);
    R97_CHECK(want.scored == 20);

    std::unique_ptr<r97::ForwardDriver> driver = open_forward(forward_config(f));
    const r97::StepResult got = driver->step(ce_model())(batch, r97::ParamSet(), nullptr);

    same_double(got.loss, want.loss, 0.0, "the forward's loss and the reference's");
    R97_CHECK(got.scored == want.scored);
    R97_CHECK(driver->last().windows() == 1);
    std::printf("  forward: 20 scored tokens, loss %.9f as the reference ok\n", got.loss);
}

// A second objective, written in this file, gives its own value through the same driver: the driver
// holds no loss, only the rows and the windows.
R97_TEST(forward_carries_whatever_objective_it_is_given) {
    const r97::Fixture& f = r97::fixture();
    const std::vector<std::vector<int32_t>> prompts = {prompt(13, 71), prompt(9, 3)};
    const Batch batch = batch_of(prompts);
    const std::vector<r97::Prefill> pre = r97::run_prefill(f, prompts);
    const std::vector<float> rows = reference_rows(pre, batch, f.n_vocab);
    const int64_t column = 7;
    const double want = column_of(batch, rows, f.n_vocab, column);

    std::unique_ptr<r97::ForwardDriver> driver = open_forward(forward_config(f));
    r97::Model model;
    model.objective = ColumnObjective(column);
    const r97::StepResult got = driver->step(model)(batch, r97::ParamSet(), nullptr);

    same_double(got.loss, want, 1e-12, "the column objective's sum and the reference's");
    R97_CHECK(got.scored == 12 + 8);
    R97_CHECK(driver->last().seeds().empty());  // it differentiates nothing, so it seeds nothing
    std::printf("  forward: a column sum of %.6f from the same step ok\n", got.loss);
}

// A batch of three gives the token-weighted mean of the three singles: each sequence runs from
// position 0 in the batch's one step, so its rows are the ones it gets alone, and the hidden state
// the backward reads back is the same in either case.
R97_TEST(forward_batch_of_three_is_the_weighted_mean) {
    const r97::Fixture& f = r97::fixture();
    const std::vector<std::vector<int32_t>> prompts = {prompt(8, 11), prompt(21, 5),
                                                       prompt(40, 29)};
    const r97::ParamSet params;
    const Batch batch = batch_of(prompts);

    // The reference first: radiance keeps one plugin registry for the process and refuses a second
    // live engine, so a driver and run_prefill cannot both be up.
    const std::vector<r97::Prefill> pre = r97::run_prefill(f, prompts);
    const r97::CeSum want = r97::ce_loss(batch, reference_rows(pre, batch, f.n_vocab), f.n_vocab);

    std::unique_ptr<r97::ForwardDriver> driver = open_forward(forward_config(f));
    const r97::StepFn step = driver->step(ce_model());

    // The loss is the sum of the scored positions' cross entropies, so three sequences' sum is the
    // sum of the three they get alone, and their mean is the token-weighted mean of the three means.
    double summed = 0.0;
    int64_t scored = 0;
    std::vector<std::vector<float>> single_hidden;
    for (const std::vector<int32_t>& ids : prompts) {
        const r97::StepResult one = step(batch_of({ids}), params, nullptr);
        R97_CHECK(one.scored == (int64_t)ids.size() - 1);
        summed += one.loss;
        scored += one.scored;
        single_hidden.push_back(driver->last().hidden());
    }

    const r97::StepResult all = step(batch, params, nullptr);
    R97_CHECK(all.scored == scored);
    same_double(all.scored, want.scored, 0.0, "the batch's scored count and the reference's");
    same_double(all.loss, summed, 1e-12, "the batch's loss and the singles'");
    same_double(all.loss / (double)all.scored, summed / (double)scored, 1e-12,
                "the batch's mean and the singles'");

    // The batch's rows are the reference's: the same three prompts in one prefill.
    same_double(all.loss, want.loss, 0.0, "the batch's loss and the reference's");

    // And the hidden state at each sequence's scored rows is the one it gets alone: the batch's
    // rows are mapped onto the sequences, not onto the step.
    const r97::ForwardStep& last = driver->last();
    const size_t cols = size_t(last.hidden_cols());
    R97_CHECK(cols == size_t(f.hidden));
    size_t at = 0;
    for (size_t d = 0; d < prompts.size(); ++d) {
        const size_t n = prompts[d].size() - 1;
        for (size_t i = 0; i < n * cols; ++i)
            same_double(last.hidden()[at + i], single_hidden[d][i], 0.0,
                        "a sequence's hidden state in the batch and alone");
        at += n * cols;
    }
    R97_CHECK(at == last.hidden().size());
    std::printf("  forward: 8, 21 and 40 tokens give %.9f over %lld scored tokens ok\n", all.loss,
                (long long)all.scored);
}

// Two copies of one sequence in one batch come out the same twice: each sequence's attention reads
// only the keys written in the step and its gated-delta-net state starts from zero, so a batch is
// not one long chain.
R97_TEST(forward_batch_is_not_one_chain) {
    const r97::Fixture& f = r97::fixture();
    const std::vector<int32_t> ids = prompt(24, 7);
    const Batch batch = batch_of({ids, ids});  // 46 logits rows
    std::unique_ptr<r97::ForwardDriver> driver = open_forward(forward_config(f));
    const r97::StepResult got = driver->step(ce_model())(batch, r97::ParamSet(), nullptr);
    R97_CHECK(got.scored == 46);

    const std::vector<float> rows = driver->last().read("logits", 0, 46, false);
    const size_t at = 23 * size_t(f.n_vocab);
    R97_CHECK(rows.size() == 2 * at);
    double worst = 0.0;
    for (size_t i = 0; i < at; ++i)
        worst = std::max(worst, std::fabs((double)rows[i] - (double)rows[at + i]));
    std::printf("  forward: a batch's two copies of one sequence differ by %.9g\n", worst);
    same_double(worst, 0.0, 0.0, "the two copies of one sequence in one batch");
}

// A sequence of the fixture's full length runs: 512 tokens, the context the fixture's engine holds.
// A prompt of exactly that length leaves no room for a generation slot, and a prefill-only step
// draws no token, so it fills the context exactly.
R97_TEST(forward_runs_a_sequence_of_the_fixtures_length) {
    const r97::Fixture& f = r97::fixture();
    const std::vector<std::vector<int32_t>> prompts = {prompt(512, 3)};
    const Batch batch = batch_of(prompts);
    const std::vector<r97::Prefill> pre = r97::run_prefill(f, prompts);

    std::unique_ptr<r97::ForwardDriver> driver = open_forward(forward_config(f));
    const r97::StepResult got = driver->step(ce_model())(batch, r97::ParamSet(), nullptr);

    same_double(got.loss, r97::ce_loss(batch, reference_rows(pre, batch, f.n_vocab), f.n_vocab).loss,
                0.0, "512 tokens' loss and the reference's");
    R97_CHECK(got.scored == 511);
    R97_CHECK(std::isfinite(got.loss));

    // The step kept what the backward reads off it: the seeds, and the head's hidden state at each
    // scored position.
    const r97::ForwardStep& last = driver->last();
    R97_CHECK(last.seeds().size() == 1);
    R97_CHECK(last.hidden_rows().size() == 511);
    R97_CHECK(last.hidden_cols() == f.hidden);
    R97_CHECK(last.hidden().size() == 511 * size_t(f.hidden));
    for (float v : last.hidden()) R97_CHECK(std::isfinite(v));
    // Consecutive rows of the head's input: one per scored position, in the batch's row order.
    for (size_t i = 0; i < last.hidden_rows().size(); ++i)
        R97_CHECK(last.hidden_rows()[i] == last.hidden_rows()[0] + (int64_t)i);
    R97_CHECK(last.hidden_rows()[0] >= 0);
    std::printf("  forward: 512 tokens, 511 scored, hidden 511 x %d ok\n", f.hidden);
}

// Windows are chunks of positions and their Results are summed: the same loss, more calls, and a
// seed per window rather than one for the step.
R97_TEST(forward_windows_sum_to_the_whole) {
    const r97::Fixture& f = r97::fixture();
    const std::vector<std::vector<int32_t>> prompts = {prompt(21, 5), prompt(13, 71)};
    const Batch batch = batch_of(prompts);  // 32 logits rows
    const r97::ParamSet params;

    // One driver at a time: radiance keeps a plugin registry for the process and refuses a second
    // live engine, so the whole window has to finish before the chunked one opens.
    int64_t whole_windows = 0;
    int64_t whole_seeds = 0;
    r97::StepResult one;
    {
        std::unique_ptr<r97::ForwardDriver> whole = open_forward(forward_config(f));
        one = whole->step(ce_model())(batch, params, nullptr);
        whole_windows = whole->last().windows();
        whole_seeds = (int64_t)whole->last().seeds().size();
    }
    R97_CHECK(whole_windows == 1);

    r97::ForwardConfig chunked_cfg = forward_config(f);
    chunked_cfg.window_rows = 7;
    std::unique_ptr<r97::ForwardDriver> chunked = open_forward(chunked_cfg);
    const r97::StepResult many = chunked->step(ce_model())(batch, params, nullptr);

    R97_CHECK(chunked->last().windows() == 5);  // 32 rows in chunks of 7
    // The same rows summed in a different order: equal up to the rounding of the sum.
    same_double(many.loss, one.loss, 1e-12, "the chunked loss and the whole window's");
    R97_CHECK(many.scored == one.scored);
    R97_CHECK(whole_seeds == 1);
    R97_CHECK(chunked->last().seeds().size() == 5);
    std::printf("  forward: 5 windows of 7 rows sum to %.9f, as one window does ok\n", many.loss);
}

// check_seeds runs on every Result before anything walks the seeds: an objective whose
// differentiated read has no seed is refused, with the read named.
R97_TEST(forward_checks_the_seeds_of_every_result) {
    const r97::Fixture& f = r97::fixture();
    std::unique_ptr<r97::ForwardDriver> driver = open_forward(forward_config(f));
    r97::Model model;
    model.objective = UnseededObjective();
    const Batch batch = batch_of({prompt(9, 3)});
    throws_saying([&] { driver->step(model)(batch, r97::ParamSet(), nullptr); },
                  "differentiated read of 'logits'");
    std::printf("  forward: a differentiated read with no seed is refused ok\n");
}

// The block hook: one call per bracket, each with the step's own rows, and one residual buffer over
// the whole step -- a block's entry is the previous block's exit, and the block changes what it was
// handed. The int8 fixture, because in this radiance pin only qwen35's block-quantised builds
// bracket their blocks (arch/qwen35_fp8 calls rad_block_begin); qwen35_bf16 does not, so the hook
// is never called on the bf16 container.
R97_TEST(forward_hooks_every_block_boundary) {
    const r97::Fixture& f = r97::w8a8_fixture();
    const std::vector<std::vector<int32_t>> prompts = {prompt(21, 5), prompt(13, 71)};
    const Batch batch = batch_of(prompts);
    const int64_t tokens = 34;

    std::unique_ptr<r97::ForwardDriver> driver = open_forward(forward_config(f));
    std::vector<SeenBoundary> seen;
    driver->set_block_hook(watch_boundaries(seen, 8));
    const r97::StepResult got = driver->step(ce_model())(batch, r97::ParamSet(), nullptr);
    R97_CHECK(got.scored == 32);

    R97_CHECK((int64_t)seen.size() == 2 * f.n_layers);
    for (size_t i = 0; i < seen.size(); ++i) {
        const SeenBoundary& b = seen[i];
        R97_CHECK(b.block == (int64_t)i / 2);
        R97_CHECK(b.begin == (i % 2 == 0));
        R97_CHECK(b.rows == tokens);
        R97_CHECK(b.cols == f.hidden);
        R97_CHECK(b.dtype == "bf16");
        R97_CHECK(b.at != nullptr);
        R97_CHECK(b.head.size() == 8);
    }
    // The residual's buffer need not be the same one at every block (a block-quantised build may
    // ping-pong two), but the values carry from one block's exit to the next block's entry, and
    // each block changes what it was handed.
    for (size_t i = 1; i + 1 < seen.size(); i += 2) {
        R97_CHECK(!any_differ(seen[i].head, seen[i + 1].head));  // exit b is entry b+1
        R97_CHECK(any_differ(seen[i - 1].head, seen[i].head));    // the block changed it
    }
    std::printf("  forward: %lld boundaries over a %lld x %lld residual ok\n",
                (long long)seen.size(), (long long)seen.front().rows,
                (long long)seen.front().cols);
}

// The same on the int8 fixture: the driver runs whatever container it is given, and the loss of its
// step is the reference loss of the same prefill.
R97_TEST(forward_loss_on_the_int8_fixture) {
    const r97::Fixture& f = r97::w8a8_fixture();
    const std::vector<std::vector<int32_t>> prompts = {prompt(21, 5), prompt(13, 71)};
    const Batch batch = batch_of(prompts);
    const std::vector<r97::Prefill> pre = r97::run_prefill(f, prompts);

    std::unique_ptr<r97::ForwardDriver> driver = open_forward(forward_config(f));
    const r97::StepResult got = driver->step(ce_model())(batch, r97::ParamSet(), nullptr);

    same_double(got.loss, r97::ce_loss(batch, reference_rows(pre, batch, f.n_vocab), f.n_vocab).loss,
                0.0, "the int8 fixture's loss and the reference's");
    R97_CHECK(got.scored == 32);
    R97_CHECK(driver->last().hidden_rows().size() == 32);
    R97_CHECK(driver->last().hidden_cols() == f.hidden);
    std::printf("  forward: the int8 fixture's 32 scored tokens give %.9f ok\n", got.loss);
}

// The full length on the int8 fixture too: 512 tokens, the context its engine holds, and the loss
// the reference prefill of the same 512 tokens gives.
R97_TEST(forward_runs_an_int8_sequence_of_the_fixtures_length) {
    const r97::Fixture& f = r97::w8a8_fixture();
    const std::vector<std::vector<int32_t>> prompts = {prompt(512, 7)};
    const Batch batch = batch_of(prompts);
    const std::vector<r97::Prefill> pre = r97::run_prefill(f, prompts);

    std::unique_ptr<r97::ForwardDriver> driver = open_forward(forward_config(f));
    const r97::StepResult got = driver->step(ce_model())(batch, r97::ParamSet(), nullptr);

    same_double(got.loss, r97::ce_loss(batch, reference_rows(pre, batch, f.n_vocab), f.n_vocab).loss,
                0.0, "the int8 fixture's 512 tokens and the reference's");
    R97_CHECK(got.scored == 511);
    R97_CHECK(driver->last().hidden_rows().size() == 511);
    R97_CHECK(driver->last().hidden_cols() == f.hidden);
    std::printf("  forward: the int8 fixture's 511 scored tokens of 512 give %.9f ok\n", got.loss);
}

// What the driver will not do: a pool dtype the backward has no adjoint for, and a batch that does
// not fit one step.
R97_TEST(forward_refuses_a_pool_it_cannot_differentiate) {
    // Nothing here is read: the pool is refused before the container is opened, so the case does
    // not need a fixture.
    r97::ForwardConfig cfg;
    cfg.model = "/r97/no-such-container.rad";
    cfg.radiance_home = "/r97/no-such-home";
    cfg.kernels = {"libref"};
    cfg.kv_dtype = "fp8";
    std::string err;
    R97_CHECK(r97::ForwardDriver::open(cfg, &err) == nullptr);
    R97_CHECK(err.find("bf16") != std::string::npos);
    std::printf("  forward: a key-value pool that is not bf16 is refused ok\n");
}

R97_TEST(forward_refuses_a_batch_that_is_not_one_step) {
    const r97::Fixture& f = r97::fixture();
    // Three sequences of 200 tokens each: 600 against a 512-token step, so the engine would chunk
    // them, and a chunked prompt is a second step's graph.
    const Batch batch = batch_of({prompt(200, 3), prompt(200, 5), prompt(200, 7)});
    std::unique_ptr<r97::ForwardDriver> driver = open_forward(forward_config(f));
    throws_saying([&] { driver->step(ce_model())(batch, r97::ParamSet(), nullptr); },
                  "one step holds");
    std::printf("  forward: 600 tokens over a 512-token step are refused ok\n");
}

#endif  // R97_FORWARD && R97_FIXTURE_PREFILL
