// tools/loss.cpp -- the `loss` subcommand: a chat JSONL file tokenised by the container's own
// chat-data code, run through the training forward driver, and the mean held-out loss of its
// scored tokens printed. With --lora the adapter directory is handed to radiance before the
// driver opens, and radiance's own loader is what reads it.
#include "commands.h"

#include "r97/chatdata.h"
#include "r97/forward.h"
#include "r97/loss.h"
#include "r97/model.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>

#if R97_FORWARD

#include "engine.h"  // the complete rad::Engine behind ForwardDriver's unique_ptr
#include "lora/preload.h"  // the adapter the driver's engines start with

namespace r97tool {
namespace {

struct Args {
    std::string model;
    std::string data;
    std::string lora;    // empty: the base model
    int64_t limit = -1;  // < 0: every conversation in the file
};

bool parse(int argc, char** argv, Args* a) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](std::string* out) {
            if (i + 1 >= argc) return false;
            *out = argv[++i];
            return true;
        };
        if (arg == "--model") {
            if (!next(&a->model)) return false;
        } else if (arg == "--data") {
            if (!next(&a->data)) return false;
        } else if (arg == "--lora") {
            if (!next(&a->lora)) return false;
        } else if (arg == "--limit") {
            std::string n;
            if (!next(&n)) return false;
            try {
                a->limit = std::stoll(n);
            } catch (...) {
                return false;
            }
            if (a->limit < 0) return false;
        } else {
            std::cerr << "r97train loss: unknown argument " << arg << "\n";
            return false;
        }
    }
    return !a->model.empty() && !a->data.empty();
}

// The plugin home: RADIANCE_HOME, else the one the build keeps beside this binary.
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

// One chunk's loss, summed over the forward driver's step. The driver runs a batch in one step, so
// the caller cuts a large file into batches that fit its budgets.
void add_chunk(const r97::StepFn& step, const r97::Batch& batch, double* loss, int64_t* scored) {
    const r97::ParamSet params;
    const r97::StepResult r = step(batch, params, nullptr);
    *loss += r.loss;
    *scored += r.scored;
}

int loss_main(int argc, char** argv) {
    Args args;
    if (!parse(argc, argv, &args)) {
        std::cerr << "usage: r97train loss --model <file.rad> --data <file.jsonl> "
                     "[--lora <dir>] [--limit N]\n";
        return 2;
    }

    r97::ChatData chat;
    if (!chat.open(args.model)) {
        std::cerr << "r97train loss: " << chat.error() << "\n";
        return 3;
    }

    r97::Batch batch;
    r97::ChatDataStats stats;
    if (!chat.read_file(args.data, r97::ChatData::kDefaultMaxTokens, &batch, &stats)) {
        std::cerr << "r97train loss: " << chat.error() << "\n";
        return 3;
    }
    if (args.limit >= 0 && size_t(args.limit) < batch.size()) batch.resize(size_t(args.limit));
    if (batch.empty()) {
        std::cerr << "r97train loss: no conversations to score\n";
        return 3;
    }

    int64_t longest = 0;
    for (const r97::TokenSeq& s : batch) longest = std::max(longest, int64_t(s.ids.size()));

    // A step holds the batch whole. Keep the fixture-sized floor and enlarge it only when a
    // sequence is longer, then cut the file into batches that fit.
    const int64_t token_budget = std::max<int64_t>(512, longest);
    const int64_t context = std::max<int64_t>(512, longest);

    r97::ForwardConfig cfg;
    cfg.model = args.model;
    cfg.radiance_home = radiance_home();
    if (cfg.radiance_home.empty()) {
        std::cerr << "r97train loss: no radiance home: set RADIANCE_HOME to the directory holding "
                     "architectures/, kernels/ and quantizers/\n";
        return 3;
    }
    cfg.max_tok = token_budget;
    cfg.max_ctx = context;
    cfg.max_seqs = 8;

    // The forward config carries no adapter -- the driver is the base model's -- so the run names
    // it to radiance, which loads it with everything --lora loads and keeps it for the prefill
    // steps every engine of this process runs. A step reports a failed engine start by throwing,
    // after radiance has logged the reason (an adapter directory its loader refuses, say), so the
    // measurement runs under a catch and the run ends there rather than at the top of the process.
    double loss = 0.0;
    int64_t scored = 0;
    try {
        if (!args.lora.empty()) rad::lora::preload_adapter("forward", args.lora);

        std::string err;
        std::unique_ptr<r97::ForwardDriver> driver = r97::ForwardDriver::open(cfg, &err);
        if (!driver) {
            std::cerr << "r97train loss: " << err << "\n";
            return 3;
        }

        r97::Model model;
        model.objective = r97::CeObjective;
        const r97::StepFn step = driver->step(model);

        r97::Batch chunk;
        int64_t chunk_tokens = 0;
        for (const r97::TokenSeq& s : batch) {
            const int64_t tokens = int64_t(s.ids.size());
            if (!chunk.empty() &&
                (chunk.size() >= size_t(cfg.max_seqs) || chunk_tokens + tokens > token_budget)) {
                add_chunk(step, chunk, &loss, &scored);
                chunk.clear();
                chunk_tokens = 0;
            }
            chunk.push_back(s);
            chunk_tokens += tokens;
        }
        if (!chunk.empty()) add_chunk(step, chunk, &loss, &scored);
    } catch (const std::exception& e) {
        std::cerr << "r97train loss: " << e.what() << "\n";
        return 3;
    }

    if (scored == 0) {
        std::cerr << "r97train loss: the file has no scored tokens\n";
        return 4;
    }

    std::cout << "loss: " << std::setprecision(12) << loss / double(scored) << "\n"
              << "scored: " << scored << "\n";
    return 0;
}

R97_REGISTER_COMMAND("loss", "--model <file.rad> --data <file.jsonl> [--lora <dir>] [--limit N]",
                     loss_main);

}  // namespace
}  // namespace r97tool

#endif  // R97_FORWARD
