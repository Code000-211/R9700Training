// tools/train.cpp -- the `train` subcommand, with the reference step as an option.
//
//   r97train train --reference --model tiny.rad --out DIR [--data chat.jsonl] [--steps N]
//
// Without --reference this is the ordinary training loop over the host step. With it the step is
// make_reference_step(): the fixture's op graph in f64 from the twins and host adjoints, no card
// and no kernel library. The loop, the checkpoint and the resume are the same either way.
#include "commands.h"

#include "r97/adapter_io.h"
#include "r97/batcher.h"
#include "r97/chatdata.h"
#include "r97/config.h"
#include "r97/delta.h"
#include "r97/model.h"
#include "r97/reference_step.h"
#include "r97/trainer.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
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
                std::cerr << "usage: r97train train --reference --model <file.rad> --out <dir>\n";
                return 2;
            }
            program = argv[i];
        } else if (arg == "--resume") {
            if (++i >= argc) {
                std::cerr << "usage: r97train train --reference --model <file.rad> --out <dir>\n";
                return 2;
            }
            resume = argv[i];
        } else if (arg == "--config" || arg == "--config-file") {
            if (++i >= argc) {
                std::cerr << "usage: r97train train --reference --model <file.rad> --out <dir>\n";
                return 2;
            }
            config_path = argv[i];
        } else {
            flags.push_back(arg);
        }
    }

    try {
        r97::Config cfg = config_path.empty() ? r97::Config{} : r97::read_config(config_path);
        cfg.token_budget = 2048;
        cfg.steps = 1;
        cfg.lr = 1e-3;
        cfg.accumulation = 1;
        r97::apply_config_flags(cfg, flags);
        r97::validate_config(cfg);
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

        r97::StepFn step;
        if (reference) {
            r97::ReferenceStepOptions opts;
            opts.container = cfg.model;
            opts.radiance_home = radiance_home();
            opts.kernels = {"libavx", "libref"};
            opts.program = program;
            step = r97::make_reference_step(model, opts);
        } else {
            throw std::runtime_error("train: the host step is not wired in this build; use --reference");
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

R97_REGISTER_COMMAND("train", "--reference --model <file.rad> --out <dir>", train_main);

}  // namespace
}  // namespace r97tool
