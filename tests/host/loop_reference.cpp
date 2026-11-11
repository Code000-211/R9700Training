// tests/host/loop_reference.cpp -- the training loop on the reference step's real op graph.
//
// The toy loop test proves the schedule, the checkpoint and the resume on a two-parameter step.
// This one repeats them -- the resume's bit-identical checkpoint, a pending accumulation kept
// across one, and the refusal of a checkpoint trained on another base -- on the fixture's whole
// graph: the forward and backward are the twins and host adjoints in f64, and the program file
// means no kernel runs.
#include <cstring>
#include <filesystem>
#include <unistd.h>
#include <iterator>
#include <fstream>
#include <stdexcept>
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
        // An assistant-only shape: the first half of every sequence is context, the second half
        // is scored. The step has to keep the batch's row space and the scored subset apart.
        seq.score.assign(seq.ids.size(), 0);
        for (size_t i = seq.ids.size() / 2; i < seq.ids.size(); ++i) seq.score[i] = 1;
        b.push_back(std::move(seq));
    }
    return b;
}

r97::Config config_for(const std::string& out, int64_t steps) {
    r97::Config cfg;
    cfg.model = r97test::late_container();
    cfg.data = "unused";
    cfg.out = out;
    cfg.base_id = "base-A";
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

// The recorded program is one fixed three-sequence batch, in its own order; the loop is what this
// test exercises, not the batcher's packing (the toy test covers that), so every step gets that
// whole batch. The batcher still hands a batch of its own -- one sequence of the three, since the
// harness runs with one sequence a batch -- and the step checks that it is drawn from the program's
// three, so a batcher fed something else fails here instead of being silently ignored.
// `model` is the one the harness initialised: the step applies the delta through it, and a
// second model's uninitialised delta would not know its targets.
r97::StepFn fixed_step(const r97::Model& model, const std::string& container) {
    r97::ReferenceStepOptions opts;
    opts.container = container;
    opts.program = program_file().string();
    r97::StepFn inner = r97::make_reference_step(model, opts);
    r97::Batch fixed = oracle_batch();
    return [inner, fixed](const r97::Batch& given, const r97::ParamSet& p, r97::ParamSet* g) {
        if (given.empty())
            throw std::runtime_error("loop_reference: the batcher handed an empty batch");
        for (const r97::TokenSeq& s : given) {
            bool from_program = false;
            for (const r97::TokenSeq& f : fixed) from_program = from_program || f.ids == s.ids;
            if (!from_program)
                throw std::runtime_error(
                    "loop_reference: the batcher handed a sequence the program does not hold");
        }
        return inner(fixed, p, g);
    };
}

r97::TrainerOptions trainer_options(const r97::Config& cfg) {
    const r97::ModelMeta meta = r97::read_model_meta(cfg.model);
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
    return options;
}

// The loop's inputs, owned here in the order the Trainer needs them to live.
struct Harness {
    r97::Config cfg;
    r97::Model model;
    r97::ParamSet params;
    r97::Batcher batcher;
    r97::Optimiser optimiser;
    r97::Trainer trainer;

    explicit Harness(const r97::Config& base)
        : cfg(base),
          model(r97::make_model(cfg)),
          params(model.delta->init(r97::read_model_meta(cfg.model), cfg.delta, uint64_t(cfg.seed))),
          batcher(oracle_batch(), {cfg.token_budget, 1, uint64_t(cfg.seed)}),
          optimiser(params, r97::optimiser_config(cfg)),
          trainer(fixed_step(model, cfg.model), {}, params, optimiser, batcher,
                  trainer_options(cfg)) {}
};

struct Run {
    std::string checkpoint;
    int64_t step = 0;
    int64_t updates = 0;  // optimiser updates applied
    int64_t pending = 0;  // micro-steps of the next update already taken
};

Run finish(Harness& h) {
    return {h.trainer.save_checkpoint(), h.trainer.state().step, h.optimiser.updates_done(),
            h.optimiser.pending()};
}

Run run(const r97::Config& cfg, int64_t stop_at = 0, const std::string& resume = "") {
    Harness h(cfg);
    if (!resume.empty()) h.trainer.resume(resume);
    if (stop_at > 0)
        h.trainer.run_steps(stop_at);
    else
        h.trainer.run();
    return finish(h);
}

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
}

// The checkpoint's adapter and its bookkeeping, byte for byte.
void same_checkpoint(const std::string& a, const std::string& b) {
    for (const char* name : {"trainer.state", "adapter/adapter_model.safetensors",
                             "adapter/adapter_config.json"}) {
        if (read_file(fs::path(a) / name) != read_file(fs::path(b) / name))
            throw std::runtime_error(std::string("loop reference: ") + name + " differs");
    }
}

bool same_params(const r97::ParamSet& a, const r97::ParamSet& b) {
    if (a.size() != b.size()) return false;
    for (size_t k = 0; k < a.size(); ++k) {
        if (a[k].name != b[k].name || a[k].value.size() != b[k].value.size()) return false;
        if (std::memcmp(a[k].value.data(), b[k].value.data(),
                        a[k].value.size() * sizeof(float)) != 0)
            return false;
    }
    return true;
}

}  // namespace

R97_TEST(loop_reference_resume_is_bit_identical) {
#if !R97_FIXTURE_PREFILL
    return;
#else
    TempDir straight;
    TempDir resumed;
    constexpr int64_t n = 2;
    const std::string a = run(config_for(straight.path, 2 * n)).checkpoint;
    const std::string b1 = run(config_for(resumed.path, 2 * n), n).checkpoint;
    const std::string b2 = run(config_for(resumed.path, 2 * n), n, b1).checkpoint;
    same_checkpoint(a, b2);
#endif
}

R97_TEST(loop_reference_resume_keeps_a_pending_accumulation) {
#if !R97_FIXTURE_PREFILL
    return;
#else
    TempDir straight;
    TempDir split;
    constexpr int64_t n = 3;  // odd: one update is applied and one micro-step is pending
    r97::Config one_cfg = config_for(straight.path, 2 * n);
    one_cfg.accumulation = 2;
    one_cfg.checkpoint_every = 0;
    r97::Config two_cfg = config_for(split.path, 2 * n);
    two_cfg.accumulation = 2;
    two_cfg.checkpoint_every = 0;

    const Run one = run(one_cfg);
    const Run two = run(two_cfg, n);
    if (two.step != n || two.updates != 1 || two.pending != 1)
        throw std::runtime_error(
            "loop reference: the stopped run's optimiser is not one update and one micro-step");

    // Resume into the harness first, so the pending micro-step is seen before it is applied.
    Harness h(two_cfg);
    h.trainer.resume(two.checkpoint);
    if (h.trainer.state().step != n || h.optimiser.updates_done() != 1 ||
        h.optimiser.pending() != 1)
        throw std::runtime_error(
            "loop reference: the resume did not keep the pending accumulation");
    h.trainer.run();
    const Run done = finish(h);
    if (done.step != 2 * n)
        throw std::runtime_error("loop reference: the resumed run stopped early");
    same_checkpoint(one.checkpoint, done.checkpoint);
#endif
}

R97_TEST(loop_reference_resume_refuses_a_different_base_id) {
#if !R97_FIXTURE_PREFILL
    return;
#else
    TempDir dir;
    const Run original = run(config_for(dir.path, 1));

    r97::Config other_cfg = config_for((dir.path / "other").string(), 1);
    other_cfg.base_id = "base-B";
    Harness other(other_cfg);
    const r97::ParamSet before = other.params;

    bool refused = false;
    try {
        other.trainer.resume(original.checkpoint);
    } catch (const r97::BaseIdMismatch& error) {
        refused = true;
        if (error.checkpoint_base_id() != "base-A" || error.requested_base_id() != "base-B" ||
            error.checkpoint_dir() != original.checkpoint)
            throw std::runtime_error(
                "loop reference: the refusal names the wrong ids or checkpoint");
    }
    if (!refused) throw std::runtime_error("loop reference: a different base was accepted");
    if (!same_params(before, other.params))
        throw std::runtime_error("loop reference: the refused resume touched the weights");
#endif
}
