// tests/host/loop_reference.cpp -- the training loop on the reference step's real op graph.
//
// The toy loop test proves the schedule, the checkpoint and the resume on a two-parameter step.
// This one repeats the resume equality on the fixture's whole graph: the forward and backward are
// the twins and host adjoints in f64, and the program file means no kernel runs.
#include <filesystem>
#include <unistd.h>
#include <iterator>
#include <fstream>
#include <string>
#include <vector>

#include "support/late_fixture.h"
#include "support/oracle.h"
#include "support/test.h"
#include "r97/adapter_io.h"
#include "r97/batcher.h"
#include "r97/config.h"
#include "r97/delta.h"
#include "r97/model.h"
#include "r97/optim.h"
#include "r97/reference_step.h"
#include "r97/trainer.h"

namespace {

namespace fs = std::filesystem;

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() / ("r97-loop-reference-" + std::to_string(::getpid()));
        fs::remove_all(path);
        fs::create_directories(path);
    }
    ~TempDir() { std::error_code ec; fs::remove_all(path, ec); }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
};
const r97test::LatePaths& late = r97test::late_paths();

fs::path oracle_dir() {
    return fs::path(__FILE__).parent_path().parent_path() / "fixtures" / "oracle";
}
fs::path program_file() {
    return fs::path(__FILE__).parent_path().parent_path() / "fixtures" / "programs" / "oracle.json";
}

r97::Batch oracle_batch() {
    const r97::oracle::detail::File ref((oracle_dir() / "reference.safetensors").string());
    r97::Batch b;
    for (int s = 0; s < 3; ++s) {
        const auto& t = ref.tensor("seq" + std::to_string(s) + "/input_ids");
        r97::TokenSeq seq;
        for (double v : t.data) seq.ids.push_back(int32_t(v));
        seq.score.assign(seq.ids.size(), 1);
        seq.score[0] = 0;
        b.push_back(std::move(seq));
    }
    return b;
}

r97::Config config_for(const std::string& out, int64_t steps) {
    r97::Config cfg;
    cfg.model = r97test::late_container();
    cfg.data = "unused";
    cfg.out = out;
    cfg.delta.kind = "lora";
    cfg.delta.rank = 4;
    cfg.delta.alpha = 8.0;
    cfg.delta.targets = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                         "self_attn.o_proj", "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
                         "linear_attn.out_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"};
    cfg.lr = 1e-3;
    cfg.steps = steps;
    cfg.token_budget = 512;
    cfg.accumulation = 1;
    cfg.seed = 7;
    return cfg;
}

struct Run {
    std::string checkpoint;
};

Run run(const std::string& out, int64_t steps, int64_t stop_at = 0, const std::string& resume = "") {
    r97::Config cfg = config_for(out, steps);
    r97::Model model = r97::make_model(cfg);
    const r97::ModelMeta meta = r97::read_model_meta(cfg.model);
    r97::ParamSet params = model.delta->init(meta, cfg.delta, uint64_t(cfg.seed));

    r97::ReferenceStepOptions opts;
    opts.container = cfg.model;
    opts.program = program_file().string();
    r97::StepFn inner = r97::make_reference_step(model, opts);
    // The recorded program is one fixed three-sequence batch; the loop is what this test exercises,
    // not the batcher's packing (the toy test covers that), so every step gets that whole batch.
    r97::Batch fixed = oracle_batch();
    r97::StepFn step = [inner, fixed](const r97::Batch&, const r97::ParamSet& p, r97::ParamSet* g) {
        return inner(fixed, p, g);
    };
    r97::Batcher batcher(oracle_batch(), {cfg.token_budget, 1, uint64_t(cfg.seed)});
    r97::Optimiser optimiser(params, r97::optimiser_config(cfg));

    r97::AdapterConfig adapter;
    adapter.base_model = cfg.model;
    adapter.rank = cfg.delta.rank;
    adapter.alpha = cfg.delta.alpha;
    adapter.targets = cfg.delta.targets;
    adapter.input_basis = meta.input_basis;
    r97::TrainerOptions options;
    options.config = cfg;
    options.write_adapter = [adapter](const r97::ParamSet& p, const std::string& dir) {
        r97::write_adapter(dir, adapter, p);
    };
    r97::Trainer trainer(step, {}, params, optimiser, batcher, std::move(options));
    if (!resume.empty()) trainer.resume(resume);
    if (stop_at > 0)
        trainer.run_steps(stop_at);
    else
        trainer.run();
    return {trainer.save_checkpoint()};
}

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
}

}  // namespace

R97_TEST(loop_reference_resume_is_bit_identical) {
#if !R97_FIXTURE_PREFILL
    return;
#else
    TempDir straight;
    TempDir resumed;
    constexpr int64_t n = 2;
    const std::string a = run(straight.path, 2 * n).checkpoint;
    const std::string b1 = run(resumed.path, 2 * n, n).checkpoint;
    const std::string b2 = run(resumed.path, 2 * n, n, b1).checkpoint;

    for (const char* name : {"trainer.state", "adapter/adapter_model.safetensors",
                             "adapter/adapter_config.json"}) {
        const std::string left = read_file(fs::path(a) / name);
        const std::string right = read_file(fs::path(b2) / name);
        if (left != right) throw std::runtime_error(std::string("loop reference: ") + name + " differs");
    }
#endif
}
