// tests/host/reference_step.cpp -- the reference step on the tiny fixture's whole graph.
//
// The loss is the oracle's (it matches to 1.8e-8); the adapter's gradient is the oracle's
// lora_grad to the floor an f32 parameter puts under it, stated where it is checked. The live
// trace and the recorded program must agree bit for bit: the program mode touches no radiance
// kernel, so it also proves the twins run the graph.
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
// Copy the fixture's paths at static-init time, before the host tier's `main` runs the checks.
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

    // The oracle's loss is the mean over scored tokens, as CeObjective reports. 1e-6 is the
    // stated tolerance; the step lands 1.75e-8 from it, all of it the f32 adapter point below
    // (the fixture's bf16 weights are the oracle's own values, cast to f64).
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
    // 1e-8 is not reached, and no op's twin or adjoint leaves the reference first: the floor is
    // the parameter a StepFn carries. The oracle differentiates the f64 adapter its generator
    // drew (`make_reference.py`'s `add_lora`), while `params.h` stores every value as float and
    // the adapter file is PEFT's f32, so this step is taken at most half an ulp away from the
    // oracle's point, in every entry.
    //
    // That distance accounts for the whole of it. Shifting every adapter value by one ulp moves
    // this step's own loss by 1.1e-8..4.6e-8 relative (the loss above differs from the oracle by
    // 1.75e-8) and its own gradient by 1.3e-4..1.5e-4 (the gradient differs by 7.14e-5, half of
    // that, as half an ulp should). The difference is spread over the whole adapter: 25 of the
    // 50 tensors are above 1e-5, the largest 7.14e-5 at
    // model.layers.1.linear_attn.in_proj_qkv.lora_B, a GDN adapter, whose delta-net recurrences
    // amplify a parameter shift the most. A twin or an adjoint that left the reference stands
    // alone instead, as the KKT adjoint did at 1.9e-2 before it was fixed. Reaching 1e-8 wants an
    // oracle differentiated at the f32 adapter values a StepFn can hold; no twin is the source.
    if (worst != worst || worst > 1e-4)
        throw std::runtime_error("reference step: " + worst_name + " leaves the f32 floor at " +
                                 std::to_string(worst));
    R97_CHECK(sum_abs(grads) > 0.0);
#endif
}


namespace {

// The file behind every mapping in this process, as `/proc/self/maps` spells it. The host tier
// runs all its checks in one process, so what a piece of code loads is the change it makes to
// this list.
std::vector<std::string> loaded_objects() {
    std::vector<std::string> out;
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        const size_t path = line.find('/');
        if (path == std::string::npos) continue;
        out.push_back(line.substr(path));
    }
    return out;
}

bool is_kernel_library(const std::string& object) {
    return object.find("libr4d") != std::string::npos ||
           object.find("libref") != std::string::npos;
}

// The kernel library in this process's maps that `before` did not have, or an empty string.
std::string new_kernel_library(const std::vector<std::string>& before) {
    for (const std::string& object : loaded_objects()) {
        if (!is_kernel_library(object)) continue;
        bool seen = false;
        for (const std::string& old : before) seen = seen || old == object;
        if (!seen) return object;
    }
    return {};
}

}  // namespace

R97_TEST(reference_step_program_matches_the_live_trace_without_kernels) {
#if !R97_FIXTURE_PREFILL
    return;
#else
    // Read this process's mappings before the container is read and the step is run: rad_core's
    // reader and the twins must load neither kernel library. The host binary runs every check in
    // one process and the checks before this one load libref on purpose (`adjoint_lora`,
    // `gdn_layer`), so what is compared is what this test adds; on its own, `before` holds no
    // kernel library and this is literally "the process has loaded neither libr4d nor libref".
    const std::vector<std::string> before = loaded_objects();

    Setup program = setup(true);
    r97::ParamSet pg;
    r97::StepResult pr = r97::make_reference_step(program.model, program.opts)(
        program.batch, program.params, &pg);

    const std::string loaded = new_kernel_library(before);
    if (!loaded.empty())
        throw std::runtime_error("reference step: program mode loaded " + loaded);

    // The live trace over the same batch and weights, here: the plugin's host chain runs only to
    // hand the twins its OpRecords. The two paths must agree on the loss and on every gradient
    // bit for bit, which is what taking nothing but the records means.
    r97::ReferenceStepOptions traced = program.opts;
    traced.program.clear();
    r97::ParamSet lg;
    r97::StepResult lr =
        r97::make_reference_step(program.model, traced)(program.batch, program.params, &lg);
    if (lr.loss != pr.loss || lr.scored != pr.scored || lg.size() != pg.size())
        throw std::runtime_error("reference step: the live trace and the program disagree on loss");
    for (size_t k = 0; k < pg.size(); ++k) {
        if (lg[k].name != pg[k].name || lg[k].value.size() != pg[k].value.size())
            throw std::runtime_error("reference step: the program's " + pg[k].name +
                                     " is not the live trace's");
        for (size_t i = 0; i < pg[k].value.size(); ++i)
            if (std::memcmp(&lg[k].value[i], &pg[k].value[i], sizeof(float)) != 0)
                throw std::runtime_error("reference step: " + pg[k].name +
                                         " differs between the live trace and the program");
    }

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

