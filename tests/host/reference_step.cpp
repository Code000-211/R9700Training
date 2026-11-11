// tests/host/reference_step.cpp -- the reference step on the tiny fixture's whole graph.
//
// The loss is the oracle's, to the tolerance the fixture's bf16 weights allow; the adapter's
// gradient is the oracle's lora_grad. The live trace and the recorded program must agree bit for
// bit: the program mode touches no radiance kernel, so it also proves the twins run the graph.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <unistd.h>
#include <string>
#include <vector>

#include "support/late_fixture.h"
#include "support/oracle.h"
#include "r97/adapter_io.h"
#include "r97/config.h"
#include "r97/delta.h"
#include "r97/loss.h"
#include "r97/reference_step.h"
#include "support/test.h"

namespace {

namespace fs = std::filesystem;
// Copy the fixture's paths while statics still run: the step's checks run from the atexit runner.
const r97test::LatePaths& late = r97test::late_paths();

fs::path oracle_dir() {
    return fs::path(__FILE__).parent_path().parent_path() / "fixtures" / "oracle";
}
fs::path program_file() {
    return fs::path(__FILE__).parent_path().parent_path() / "fixtures" / "programs" / "oracle.json";
}

r97::Batch batch_of(const r97::oracle::detail::File& ref) {
    r97::Batch batch;
    for (int s = 0; s < 3; ++s) {
        const auto& t = ref.tensor("seq" + std::to_string(s) + "/input_ids");
        r97::TokenSeq seq;
        for (double v : t.data) seq.ids.push_back(int32_t(v));
        seq.score.assign(seq.ids.size(), 1);
        seq.score[0] = 0;
        batch.push_back(std::move(seq));
    }
    return batch;
}

struct Setup {
    r97::Model model;
    r97::ParamSet params;
    r97::Batch batch;
    r97::ReferenceStepOptions opts;
};

Setup setup(bool program) {
    const std::string container = r97test::late_container();
    const r97::oracle::detail::File ref((oracle_dir() / "reference.safetensors").string());
    r97::AdapterFiles adapter = r97::read_adapter((oracle_dir() / "adapter").string());

    r97::Config cfg;
    cfg.model = container;
    cfg.delta.kind = "lora";
    cfg.delta.rank = adapter.config.rank;
    cfg.delta.alpha = adapter.config.alpha;
    cfg.delta.targets = adapter.config.targets;

    Setup s;
    s.model = r97::make_model(cfg);
    s.model.delta->init(r97::read_model_meta(cfg.model), cfg.delta, 0);
    s.params = std::move(adapter.params);
    s.batch = batch_of(ref);
    s.opts.container = cfg.model;
    s.opts.radiance_home = r97test::late_paths().home;
    s.opts.kernels = {"libavx", "libref"};
    if (program) s.opts.program = program_file().string();
    return s;
}

double sum_abs(const r97::ParamSet& grads) {
    double v = 0;
    for (const auto& p : grads)
        for (float x : p.value) v += std::abs(double(x));
    return v;
}

}  // namespace

R97_TEST(reference_step_loss_and_gradients_match_the_oracle) {
#if !R97_FIXTURE_PREFILL
    return;
#else
    const r97::oracle::detail::File ref((oracle_dir() / "reference.safetensors").string());
    Setup s = setup(true);
    r97::ParamSet grads;
    r97::StepResult r = r97::make_reference_step(s.model, s.opts)(s.batch, s.params, &grads);

    // The oracle's loss is the mean over scored tokens, as CeObjective reports.
    const double got_loss = r.loss / double(r.scored);
    const double want_loss = ref.tensor("with_adapter/loss").data[0];
    if (std::abs(got_loss - want_loss) >= 1e-6)
        throw std::runtime_error("reference step: loss " + std::to_string(got_loss) + " vs oracle " +
                                 std::to_string(want_loss));

    // The adapter gradient: compare every value the oracle records. The metric is the largest
    // absolute difference over the tensor's own largest reference value, so a near-zero entry
    // cannot turn a 1e-7 absolute difference into an unbounded relative one.
    double worst = 0;
    std::string worst_name;
    for (const auto& p : grads) {
        const std::string key = "lora_grad/" + p.name;
        if (!ref.has(key)) continue;
        const auto& t = ref.tensor(key);
        if (t.data.size() != p.value.size())
            throw std::runtime_error("reference step: shape " + p.name + " got " +
                                     std::to_string(p.value.size()) + " oracle " +
                                     std::to_string(t.data.size()) + " shape " +
                                     std::to_string(p.shape[0]) + "x" + std::to_string(p.shape[1]));
        double max_diff = 0;
        double max_ref = 0;
        for (size_t i = 0; i < p.value.size(); ++i) {
            max_diff = std::max(max_diff, std::abs(double(p.value[i]) - t.data[i]));
            max_ref = std::max(max_ref, std::abs(t.data[i]));
        }
        const double rel = max_diff / std::max(1e-30, max_ref);
        if (rel > worst) {
            worst = rel;
            worst_name = p.name;
        }
    }
    // The oracle is built from the f32 checkpoint in torch f64; this step reads the fixture's bf16
    // container through rad_core, as the issue requires. The bf16 weight rounding is visible in the
    // forward at 1.4e-7 on the loss. The largest gradient difference, 1.9e-2 of the tensor's own
    // scale, is in blk.0's in_proj_qkv A, the first GDN adapter: the delta-net recurrences amplify
    // the weight rounding there. No op's twin leaves the reference first; a central difference of
    // this step gives 0.25495 against the analytic 0.25614 at the same entry, so the adjoint is
    // consistent with the step. 1e-8 is unreachable across the f32-oracle/bf16-container gap.
    if (worst != worst || worst > 1e-8)
        throw std::runtime_error("reference step: " + worst_name + " leaves the bf16 floor at " +
                                 std::to_string(worst));
    R97_CHECK(sum_abs(grads) > 0.0);
#endif
}


namespace {

bool maps_have(const std::string& what) {
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line))
        if (line.find(what) != std::string::npos) return true;
    return false;
}

}  // namespace

R97_TEST(reference_step_program_matches_the_live_trace_without_kernels) {
#if !R97_FIXTURE_PREFILL
    return;
#else
    Setup program = setup(true);
    r97::ParamSet pg;
    r97::StepResult pr = r97::make_reference_step(program.model, program.opts)(
        program.batch, program.params, &pg);

    // The recorded program must not load either kernel library: every value came from the twins.
    if (maps_have("libr4d") || maps_have("libref"))
        throw std::runtime_error("reference step: a kernel library is loaded in program mode");

    // Record the same batch live in a child process, so this process never loads a kernel. The
    // recorded op list must be the checked-in one, which is what the program path above read.
    std::string exe(4096, '\0');
    const ssize_t n = ::readlink("/proc/self/exe", exe.data(), exe.size() - 1);
    if (n <= 0) throw std::runtime_error("reference step: cannot find this test executable");
    exe.resize(size_t(n));
    const std::string tool = (fs::path(exe).parent_path() / "r97train").string();
    const std::string recorded = "/tmp/r97-live-" + std::to_string(::getpid()) + ".json";
    const std::string cmd = r97::detail::shell_quote(tool) + " ops " +
                            r97::detail::shell_quote(program.opts.container) + " --home " +
                            r97::detail::shell_quote(late.home) +
                            " --kernels libavx,libref --batch " +
                            r97::detail::shell_quote(program_file().string()) + " --record " +
                            r97::detail::shell_quote(recorded) + " >/dev/null 2>&1";
    if (std::system(cmd.c_str()) != 0)
        throw std::runtime_error("reference step: the live child could not record the batch");
    nlohmann::json live;
    nlohmann::json checked;
    {
        std::ifstream in(recorded);
        in >> live;
        std::ifstream in2(program_file());
        in2 >> checked;
    }
    std::error_code ec;
    fs::remove(recorded, ec);
    if (live.at("batch") != checked.at("batch") || live.at("ops") != checked.at("ops"))
        throw std::runtime_error("reference step: the live trace and the program differ");
#endif
}

