// tools/train.cpp -- the `train` subcommand, with the reference step as an option.
//
//   r97train train --model tiny.rad --out DIR [--data chat.jsonl] [--steps N]
//   r97train train --reference --model tiny.rad --out DIR
//
// This is the ordinary training loop over the host step: ForwardDriver and the live adapter on
// radiance's host kernels. With --reference the step is make_reference_step(): the fixture's op
// graph in f64 from the twins and host adjoints, no card and no kernel library. The loop, the
// checkpoint and the resume are the same either way.
#include "commands.h"

#include "r97/adapter_io.h"
#include "r97/batcher.h"
#include "r97/chatdata.h"
#include "r97/config.h"
#include "r97/delta.h"
#include "r97/forward.h"
#include "r97/model.h"
#include "r97/reference_step.h"
#include "r97/step_host.h"
#include "r97/trainer.h"

#if R97_FORWARD
#include "engine.h"  // ForwardDriver's member; the header cannot say it is complete
#endif

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace r97tool {
namespace {

// The plugin home: RADIANCE_HOME, else the one the build keeps beside this binary, as the `loss`
// subcommand finds it.
std::string radiance_home() {
    if (const char* env = std::getenv("RADIANCE_HOME"); env && *env) return env;

    std::error_code ec;
    const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (ec) return {};
    const std::filesystem::path beside = exe.parent_path();
    for (const std::filesystem::path& p :
         {beside / "radiance-build" / "radiance_home", beside / "radiance_home"}) {
        if (std::filesystem::exists(p / "architectures", ec)) return p.string();
    }
    return {};
}

int train_main(int argc, char** argv) {
    std::string config_path;
    std::vector<std::string> flags;
    bool reference = false;
    std::string resume;
    std::string program;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--reference") {
            reference = true;
        } else if (arg == "--program") {
            if (++i >= argc) {
                std::cerr << "usage: r97train train --model <file.rad> --out <dir> [--reference]\n";
                return 2;
            }
            program = argv[i];
        } else if (arg == "--resume") {
            if (++i >= argc) {
                std::cerr << "usage: r97train train --model <file.rad> --out <dir> [--reference]\n";
                return 2;
            }
            resume = argv[i];
        } else if (arg == "--config" || arg == "--config-file") {
            if (++i >= argc) {
                std::cerr << "usage: r97train train --model <file.rad> --out <dir> [--reference]\n";
                return 2;
            }
            config_path = argv[i];
        } else {
            flags.push_back(arg);
        }
    }

    try {
        // The command's fallbacks sit under --config and the other flags: load_config() reads the
        // file over them and applies the flags last, so a file's lr, steps, accumulation and
        // token_budget are used and a flag still wins over both.
        r97::Config fallback;
        fallback.token_budget = 2048;
        fallback.steps = 1;
        fallback.lr = 1e-3;
        fallback.accumulation = 1;
        r97::Config cfg = r97::load_config(config_path, flags, fallback);
        if (cfg.data.empty()) cfg.data = "tests/fixtures/chat_small.jsonl";

        r97::ChatData chat;
        if (!chat.open(cfg.model))
            throw std::runtime_error("train: cannot read the container's tokenizer: " + chat.error());
        r97::Batch sequences;
        r97::ChatDataStats stats;
        if (!chat.read_file(cfg.data, r97::ChatData::kDefaultMaxTokens, &sequences, &stats))
            throw std::runtime_error("train: " + chat.error());
        if (sequences.empty()) throw std::runtime_error("train: no training sequences in " + cfg.data);

        r97::Model model = r97::make_model(cfg);
        const r97::ModelMeta meta = r97::read_model_meta(cfg.model);
        r97::ParamSet params = model.delta->init(meta, cfg.delta, uint64_t(cfg.seed));

        // The forward driver the host step runs its engines through. It must outlive the step,
        // because the step keeps a reference to it across the loop.
        std::unique_ptr<r97::ForwardDriver> driver;
        r97::StepFn step;
        if (reference) {
            r97::ReferenceStepOptions opts;
            opts.container = cfg.model;
            opts.radiance_home = radiance_home();
            opts.kernels = {"libavx", "libref"};
            opts.program = program;
            step = r97::make_reference_step(model, opts);
        } else {
#if !R97_FORWARD
            throw std::runtime_error(
                "train: this build has no forward driver; use --reference with the host step");
#else
            // The fixture's engine budget, raised to the batch the loop cuts: the step runs the
            // whole batch in one prefill, so max_tok has to hold every token of it, and max_ctx
            // every sequence's positions. Same rule as `r97train loss`'s floor.
            const int64_t budget = std::max<int64_t>(512, cfg.token_budget);
            r97::ForwardConfig fwd;
            fwd.model = cfg.model;
            fwd.radiance_home = radiance_home();
            if (fwd.radiance_home.empty())
                throw std::runtime_error("train: no radiance home: set RADIANCE_HOME to the "
                                         "directory holding architectures/, kernels/ and "
                                         "quantizers/");
            fwd.max_tok = budget;
            fwd.max_ctx = budget;
            std::string err;
            driver = r97::ForwardDriver::open(fwd, &err);
            if (!driver) throw std::runtime_error("train: " + err);
            // The loop keeps no boundary values; the backward walks one block at a time.
            r97::HostStepOptions opts;
            opts.keep_boundaries = false;
            opts.reference_head_split = false;
            step = r97::host_step(*driver, model, opts);
#endif
        }

        r97::Batcher batcher(sequences, {cfg.token_budget, 1, uint64_t(cfg.seed)});
        r97::Optimiser optimiser(params, r97::optimiser_config(cfg));
        r97::AdapterConfig adapter;
        adapter.base_model = cfg.model;
        adapter.rank = cfg.delta.rank;
        adapter.alpha = cfg.delta.alpha;
        adapter.targets = cfg.delta.targets;
        adapter.input_basis = meta.input_basis;
        r97::TrainerOptions options;
        options.config = cfg;
        options.terminal = &std::cout;
        options.write_adapter = [adapter](const r97::ParamSet& p, const std::string& dir) {
            r97::write_adapter(dir, adapter, p);
        };
        r97::Trainer trainer(step, {}, params, optimiser, batcher, std::move(options));
        if (!resume.empty()) trainer.resume(resume);
        trainer.run();
        std::cout << "train: wrote " << trainer.save_checkpoint() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "r97train train: " << e.what() << "\n";
        return 1;
    }
}

R97_REGISTER_COMMAND("train", "--model <file.rad> --out <dir> [--reference]", train_main);

}  // namespace
}  // namespace r97tool
