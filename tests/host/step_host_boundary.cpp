// The per-layer check: the cotangent of each layer boundary against central finite differences of
// the driver's own loss with respect to a random perturbation of that boundary's residual.
//
// The join hands back, for every layer boundary, the cotangent it walked to (HostStepReport). This
// file perturbs the boundary itself and asks the engine what the loss does: the driver's block hook
// fires at the bracket of each block, so a test writes `x + h*d` into the live residual buffer of
// boundary b (the entry of block b, or the exit of the last block) and reads the loss through the
// engine's own head and objective. The value must be the derivative the join reported.
//
// Two arithmetic facts shape the tolerances, and both are printed below rather than assumed.
//
//   The forward is bf16. Every activation the perturbation crosses is rounded to a bf16 grid, so
//   the loss is a staircase in h: the central difference picks up a rounding jitter that falls as
//   h grows (the step buys signal), while the second-order truncation of the central difference
//   grows as h. Neither term is small enough at the other's optimum, so a single direction can be
//   fifteen per cent off on the smallest boundary and be right.
//
//   The residual is bf16 too, so a step smaller than the grid's spacing at an element's magnitude
//   is partly lost in the write itself. The hook therefore records the change it actually wrote
//   along the direction and the difference divides by that effective step, which is what makes the
//   small steps the first boundaries need usable at all; the step shrinks with depth, where the
//   layers above amplify a perturbation most.
//
//   So the check is on the mean over several directions, against that mean's own scatter: a
//   direction whose cotangent is wrong pulls the mean off by more than the noise the finite
//   difference measures for itself. The tolerance is `kSigmas` standard errors of the per-direction
//   residuals plus a small allowance for the truncation, and the test prints all three figures per
//   boundary: the cotangent's sum, the finite difference's sum, and the ratio between them.
#include "r97/adapter_io.h"
#include "r97/batch.h"
#include "r97/config.h"
#include "r97/forward.h"
#include "r97/live_adapter.h"
#include "r97/model.h"
#include "r97/step.h"
#include "r97/step_host.h"
#include "support/late_fixture.h"
#include "support/test.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#if R97_FORWARD
namespace {

// The adapter's parameters are pinned to a grid of 2^-11 with |k| <= 64, so a perturbation of one
// grid unit is exact in the f32 parameter and survives the wire's bf16 rounding.
constexpr int kGridShift = 11;
constexpr int32_t kGridMax = 64;

// The step of the central difference, a power of two, chosen per boundary: the residual's bf16
// quantum near one is 2^-8, so a step below that is lost to the rounding of the value it is added
// to, while the second-order term grows with the step, and the layers above a boundary amplify its
// perturbation -- by two orders of magnitude from the last boundary to the first on this fixture --
// so the deeper the boundary the smaller the step that keeps the response linear. The test prints
// the step it used with every figure.
int step_exponent(int64_t boundary, int64_t layers) {
    return std::min(9, 3 + 2 * static_cast<int>(layers - boundary));
}

// Directions per boundary. The mean of the per-direction residuals is checked against kSigmas of
// its own standard error.
constexpr int kDirections = 8;
constexpr double kSigmas = 3.0;
// What the second-order term may contribute to the mean, as a fraction of the mean's magnitude.
constexpr double kTruncation = 0.02;

r97::Batch batch_of(const std::vector<std::vector<int32_t>>& sequences) {
    r97::Batch batch;
    for (const std::vector<int32_t>& ids : sequences) {
        r97::TokenSeq s;
        s.ids = ids;
        s.score.assign(ids.size(), 1);
        s.score[0] = 0;  // the first token predicts nothing
        batch.push_back(std::move(s));
    }
    return batch;
}

std::vector<int32_t> prompt(int32_t seed, int32_t n) {
    std::vector<int32_t> ids(static_cast<size_t>(n));
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = (seed + 7 * int32_t(i)) % 256;
    return ids;
}

double jitter(uint64_t* state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return double((*state >> 11) & ((1ULL << 53) - 1)) / double(1ULL << 52) * 2.0 - 1.0;
}

void pin_to_grid(r97::ParamSet* params, uint64_t seed) {
    uint64_t state = seed;
    for (r97::Param& p : *params)
        for (float& v : p.value) {
            const int32_t k = int32_t(std::llround(jitter(&state) * double(kGridMax)));
            v = float(std::ldexp(double(k), -kGridShift));
        }
}

float read_bf16(const void* p, int64_t i) {
    const uint32_t u = static_cast<uint32_t>(static_cast<const uint16_t*>(p)[i]) << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

uint16_t round_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

// A write into a residual the driver exposes as const: a read hook cannot need to write, and the
// perturbation is this check's own business.
void write_residual(const std::string& dtype, void* p, int64_t i, float f) {
    if (dtype == "bf16") {
        static_cast<uint16_t*>(p)[i] = round_bf16(f);
    } else if (dtype == "f32") {
        static_cast<float*>(p)[i] = f;
    } else {
        throw std::runtime_error("boundary: cannot perturb dtype '" + dtype + "'");
    }
}

// Boundary b is the entry of block b, and boundary `layers` is the exit of the last block.
//
// The residual is bf16, so `x + h*d` is not what lands in it: writing a value rounds it back onto
// the grid, and a step below the grid's spacing at that element's magnitude is partly or wholly
// lost. The hook therefore records what it actually wrote -- the sum over the boundary of the
// change taken along the direction -- and the finite difference divides by that instead of by the
// step it asked for. That is the effective step, and it is what makes a step below one quantum
// usable at all; the test prints it.
struct Perturb {
    int64_t boundary = -1;
    int64_t layers = 0;
    float h = 0.0f;
    const std::vector<float>* dir = nullptr;
    int64_t fired = 0;
    double applied = 0.0;  // sum of (written - old) * direction

    void operator()(const r97::BlockView& v) {
        if (boundary < layers) {
            if (!v.begin || v.block != boundary) return;
        } else {
            if (v.begin || v.block != layers - 1) return;
        }
        const int64_t n = v.rows * v.cols;
        if (int64_t(dir->size()) < n) throw std::runtime_error("boundary: direction too short");
        double sum = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const float old = read_bf16(v.residual, i);
            const float d = (*dir)[static_cast<size_t>(i)];
            const float want = old + h * d;
            write_residual(v.dtype, const_cast<void*>(v.residual), i, want);
            const float got = read_bf16(v.residual, i);
            sum += double(got - old) * double(d);
        }
        applied = sum;
        ++fired;
    }
};

// The hook the join installs records each block's bracket for its own replay. This check replaces
// it with one that only keeps the live adapter fed, so the tape's block hook is free to perturb.
void hook_live_adapter(r97::ForwardDriver* driver, const r97::WireBuffers& wire) {
    std::set<std::string> leaves;
    int64_t max_rank = 0;
    for (const auto& site : wire.sites) {
        const size_t dot = site.module.find_last_of('.');
        leaves.insert(dot == std::string::npos ? site.module : site.module.substr(dot + 1));
        max_rank = std::max(max_rank, site.rank);
    }
    std::vector<std::string> targets(leaves.begin(), leaves.end());
    driver->set_engine_hook([&wire, targets, max_rank](rad::Engine& engine) {
        r97::LiveAdapterConfig cfg{"train", targets, max_rank, 1.0};
        std::string err;
        std::unique_ptr<r97::LiveAdapter> live = r97::LiveAdapter::open(engine, cfg, &err);
        if (!live) throw std::runtime_error("boundary: live adapter: " + err);
        if (live->update(wire, &err) != 0)
            throw std::runtime_error("boundary: live update: " + err);
    });
}

}  // namespace

R97_TEST(host_step_boundary_cotangents_match_finite_differences) {
    const std::string container = r97test::late_container();
    const r97::Batch batch = batch_of({prompt(21, 5), prompt(13, 7), prompt(3, 4)});

    r97::Config cfg;
    cfg.model = container;
    cfg.objective = "ce";
    cfg.delta.kind = "lora";
    cfg.delta.rank = 4;
    cfg.delta.alpha = 8.0;
    const r97::Model model = r97::make_model(cfg);
    const r97::ModelMeta meta = r97::read_model_meta(container);
    r97::ParamSet params = model.delta->init(meta, cfg.delta, 1234);
    pin_to_grid(&params, 0x94d049bb133111ebULL);

    r97::ForwardConfig fcfg;
    fcfg.model = container;
    fcfg.radiance_home = r97::detail::radiance_home();
    fcfg.window_rows = 1 << 20;
    std::string err;
    std::unique_ptr<r97::ForwardDriver> driver = r97::ForwardDriver::open(fcfg, &err);
    if (!driver) throw std::runtime_error(err);

    r97::HostStepReport report;
    const r97::StepFn step = r97::host_step(*driver, model, {}, &report);
    r97::ParamSet grads;
    const r97::StepResult ran = step(batch, params, &grads);
    R97_CHECK(ran.scored > 0);
    R97_CHECK(report.layers > 0);
    R97_CHECK(int64_t(report.boundary_cotangent.size()) == report.layers + 1);

    const int64_t tokens = report.tokens;
    const int64_t hidden = report.hidden;
    const int64_t layers = report.layers;
    const double scored = double(ran.scored);
    std::printf("  host boundary: loss %.9g over %lld scored, %lld tokens, %lld layers,"
                " %d directions\n",
                ran.loss, (long long)ran.scored, (long long)tokens, (long long)layers, kDirections);

    // The join reports cotangents of the batch's mean loss (the objective scales its seeds by
    // 1/scored); the finite difference divides by the same count so the two are the same quantity.
    r97::WireBuffers wire;
    model.delta->to_wire(params, &wire);
    hook_live_adapter(driver.get(), wire);

    std::vector<std::string> misses;
    uint64_t state = 0x9e3779b97f4a7c15ULL;
    for (int64_t b = 0; b <= layers; ++b) {
        const float h = float(std::ldexp(1.0, -step_exponent(b, layers)));
        const std::vector<double>& cot = report.boundary_cotangent[static_cast<size_t>(b)];
        R97_CHECK(int64_t(cot.size()) == tokens * hidden);

        double sum_cot = 0.0, sum_fd = 0.0, sum_sq = 0.0, sum_effective = 0.0;
        double worst = 0.0;
        for (int d = 0; d < kDirections; ++d) {
            std::vector<float> dir(static_cast<size_t>(tokens * hidden));
            for (float& v : dir) v = jitter(&state) < 0.0 ? -1.0f : 1.0f;
            double cot_dot = 0.0;
            for (size_t i = 0; i < dir.size(); ++i) cot_dot += double(dir[i]) * cot[i];

            Perturb up{b, layers, h, &dir};
            Perturb down{b, layers, -h, &dir};
            driver->set_block_hook([&up](const r97::BlockView& v) { up(v); });
            const double loss_up = driver->step(model)(batch, params, nullptr).loss / scored;
            R97_CHECK(up.fired == 1);
            driver->set_block_hook([&down](const r97::BlockView& v) { down(v); });
            const double loss_down = driver->step(model)(batch, params, nullptr).loss / scored;
            R97_CHECK(down.fired == 1);
            driver->set_block_hook(nullptr);

            const double effective = (up.applied - down.applied) /
                                     (2.0 * double(tokens * hidden));
            if (!(effective > 0.0)) throw std::runtime_error("boundary: no step landed");
            const double fd = (loss_up - loss_down) * double(tokens * hidden) /
                              (up.applied - down.applied);
            const double residual = fd - cot_dot;
            sum_cot += cot_dot;
            sum_fd += fd;
            sum_effective += effective;
            sum_sq += residual * residual;
            worst = std::max(worst, std::fabs(residual));
        }
        const double mean_residual = (sum_fd - sum_cot) / double(kDirections);
        // The standard error of that mean from the per-direction residuals, and the allowance the
        // second-order term may take from the cotangent's own magnitude.
        const double variance =
            std::max(0.0, sum_sq / double(kDirections) - mean_residual * mean_residual);
        const double se = std::sqrt(variance) / std::sqrt(double(kDirections));
        const double allowance =
            kSigmas * se + kTruncation * std::fabs(sum_cot) / double(kDirections);
        const double ratio = sum_cot != 0.0 ? sum_fd / sum_cot : 0.0;
        std::printf("  host boundary %lld: step 2^-%d (effective %8.5f) cotangent sum %12.5f"
                    " finite difference sum %12.5f ratio %7.4f mean residual %9.5f allowance"
                    " %9.5f worst %9.5f\n",
                    (long long)b, step_exponent(b, layers), sum_effective / double(kDirections),
                    sum_cot, sum_fd, ratio, mean_residual, allowance, worst);
        if (!(std::fabs(mean_residual) <= allowance)) {
            misses.push_back("boundary " + std::to_string(b) + ": the finite difference of the"
                             " driver's own loss disagrees with the cotangent by " +
                             std::to_string(mean_residual) + ", which is more than the " +
                             std::to_string(allowance) + " the difference resolves for itself");
        }
    }
    if (!misses.empty()) {
        std::string all;
        for (const std::string& m : misses) all += (all.empty() ? "" : "; ") + m;
        throw std::runtime_error(all);
    }
}
#endif

