// tools/loss.cpp -- the `loss` subcommand: the reference cross-entropy of a chat file through the
// training forward, on the host kernel tier or on a card.
//
// The sequences are the container's own render of the chat file (r97/chatdata.h), the step is the
// training forward driver (r97/forward.h), and the loss is its CeObjective summed over windows
// (r97/loss_run.h). The device run is the same step with libr4d as the kernel tier and the card
// hidden to everything but the chosen ordinal (r97/device_tier.h): radiance ranks the devices HIP
// shows it, so `--device 3` must run before anything initialises the device layer.
#include "commands.h"

#include "engine.h"           // radiance core/: Engine, the driver's own

#include "r97/chatdata.h"
#include "r97/device_tier.h"
#include "r97/loss_run.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace r97tool {
namespace {

struct Args {
    std::string data;
    std::string model;
    std::string home;
    std::string kernels;
    int device = -1;          // -1: host tier, unless --kernels names the device one
    int64_t max_tokens = 4096;
    int64_t max_seqs = 32;
    int64_t window = 256;
    int64_t limit = 0;        // 0: every conversation the file keeps
    uint64_t seed = 0;
    bool load_vram = false;   // report the loaded footprint with a one-token step first
};

bool need(int argc, char** argv, int* i, std::string* out) {
    if (*i + 1 >= argc) return false;
    *out = argv[++*i];
    return true;
}

bool number(const std::string& s, int64_t* out) {
    try {
        size_t used = 0;
        const long long v = std::stoll(s, &used);
        if (used != s.size()) return false;
        *out = v;
        return true;
    } catch (...) {
        return false;
    }
}

// radiance's kernel list: ',' or ':' separated, first one wins when two carry the same op.
std::vector<std::string> kernel_list(const std::string& list) {
    std::vector<std::string> out;
    std::string cur;
    for (const char c : list) {
        if (c == ',' || c == ':') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else if (c != ' ') {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

bool parse(int argc, char** argv, Args* a) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        std::string v;
        if (arg == "--model") {
            if (!need(argc, argv, &i, &a->model)) return false;
        } else if (arg == "--home" || arg == "--radiance-home") {
            if (!need(argc, argv, &i, &a->home)) return false;
        } else if (arg == "--kernels") {
            if (!need(argc, argv, &i, &a->kernels)) return false;
        } else if (arg == "--device") {
            int64_t n = 0;
            if (!need(argc, argv, &i, &v) || !number(v, &n) || n < 0 || n > 1024) return false;
            a->device = int(n);
        } else if (arg == "--max-tokens") {
            if (!need(argc, argv, &i, &v) || !number(v, &a->max_tokens)) return false;
        } else if (arg == "--max-seqs") {
            if (!need(argc, argv, &i, &v) || !number(v, &a->max_seqs)) return false;
        } else if (arg == "--window") {
            if (!need(argc, argv, &i, &v) || !number(v, &a->window)) return false;
        } else if (arg == "--limit") {
            if (!need(argc, argv, &i, &v) || !number(v, &a->limit)) return false;
        } else if (arg == "--seed") {
            int64_t n = 0;
            if (!need(argc, argv, &i, &v) || !number(v, &n) || n < 0) return false;
            a->seed = uint64_t(n);
        } else if (arg == "--load-vram") {
            a->load_vram = true;
        } else if (arg == "-h" || arg == "--help") {
            return false;
        } else if (!arg.empty() && arg[0] == '-') {
            return false;
        } else if (a->data.empty()) {
            a->data = arg;
        } else {
            return false;
        }
    }
    return !a->data.empty();
}

void usage() {
    std::cerr
        << "usage: r97train loss <chat.jsonl> [--model FILE] [--home DIR] [--device N]\n"
           "       [--kernels LIST] [--max-tokens N] [--max-seqs N] [--window N]\n"
           "       [--limit N] [--seed N] [--load-vram]\n"
           "\n"
           "  --device N      run the step on card N with libr4d (default: the host tier)\n"
           "  --kernels LIST  the kernel tier to run on when --device is not given\n"
           "  --max-tokens N  a step's token budget, and a conversation's length limit\n"
           "  --window N      scored positions one Objective call reads\n"
           "  --load-vram     load once for a one-token step and report that footprint\n";
}

double mib(int64_t bytes) { return double(bytes) / (1024.0 * 1024.0); }

}  // namespace

static int loss_main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, &a)) {
        usage();
        return 2;
    }
    if (a.device >= 0 && !a.kernels.empty()) {
        std::cerr << "r97train loss: --device and --kernels name the same choice; give one\n";
        return 2;
    }
    if (a.device >= 0 && a.max_tokens < 1) {
        std::cerr << "r97train loss: --max-tokens is one token at least\n";
        return 2;
    }

    if (a.model.empty()) a.model = r97::default_model_rad();
    if (a.model.empty()) {
        std::cerr << "r97train loss: no --model, and $R97_MODEL_RAD / $R97_MODEL_DIR names no "
                     "container\n";
        return 2;
    }
    if (a.home.empty()) a.home = r97::default_radiance_home();
    if (a.home.empty()) {
        std::cerr << "r97train loss: no --home, and neither $RADIANCE_HOME nor the build's "
                     "radiance-build/radiance_home exists\n";
        return 2;
    }

    const bool device = a.device >= 0;
    if (device) {
        std::string err;
        if (!r97::select_device(a.device, &err)) {
            std::cerr << "r97train loss: " << err << "\n";
            return 1;
        }
    } else if (a.kernels.empty()) {
        a.kernels = "libavx,libref";
    } else if (a.kernels.find("libr4d") != std::string::npos) {
        // A caller naming the device tier by hand is running on the card already: no selection to
        // make, but the VRAM lines still mean something.
    }

    r97::ChatData chat;
    if (!chat.open(a.model)) {
        std::cerr << "r97train loss: " << chat.error() << "\n";
        return 1;
    }
    r97::Batch sequences;
    r97::ChatDataStats stats;
    if (!chat.read_file(a.data, a.max_tokens, &sequences, &stats)) {
        std::cerr << "r97train loss: " << chat.error() << "\n";
        return 1;
    }
    if (a.limit > 0 && int64_t(sequences.size()) > a.limit)
        sequences.resize(size_t(a.limit));
    if (sequences.empty()) {
        std::cerr << "r97train loss: the chat file holds no conversation this container can score\n";
        return 1;
    }

    int64_t longest = 1;
    for (const r97::TokenSeq& s : sequences)
        longest = std::max<int64_t>(longest, int64_t(s.ids.size()));
    const int64_t budget = std::max<int64_t>(a.max_tokens, 1);

    r97::LossRunConfig run;
    run.forward.model = a.model;
    run.forward.radiance_home = a.home;
    run.forward.kernels = device ? std::vector<std::string>{"libr4d"} : kernel_list(a.kernels);
    run.forward.max_tok = budget;
    run.forward.max_ctx = longest;
    run.forward.max_seqs = std::max<int64_t>(1, std::min<int64_t>(a.max_seqs, int64_t(sequences.size())));
    run.forward.window_rows = std::max<int64_t>(1, a.window);
    run.token_budget = budget;
    run.seed = a.seed;

    const bool want_vram = device || a.kernels.find("libr4d") != std::string::npos;
    if (want_vram && a.load_vram) {
        // One token, the same KV pool and plugin home as the real step: the engine is alive when
        // run_loss samples, so this is the loaded footprint without the 4096-row head buffer.
        r97::LossRunConfig warm = run;
        warm.token_budget = 1;
        warm.sample_vram = true;
        r97::TokenSeq one;
        one.ids.push_back(sequences.front().ids.front());
        one.score.push_back(0);
        r97::LossRunResult loaded;
        std::string err;
        if (!r97::run_loss(warm, std::vector<r97::TokenSeq>{one}, &loaded, &err)) {
            std::cerr << "r97train loss: the load sample failed: " << err << "\n";
            return 1;
        }
        if (loaded.vram_sampled)
            std::printf("vram after loading %lld MiB used of %lld MiB\n",
                        (long long)(mib(loaded.vram_used) + 0.5),
                        (long long)(mib(loaded.vram_total) + 0.5));
    }

    r97::LossRunResult got;
    std::string err;
    std::unique_ptr<r97::VramWatch> watch;
    if (want_vram) watch = std::make_unique<r97::VramWatch>();
    if (!r97::run_loss(run, sequences, &got, &err)) {
        std::cerr << "r97train loss: " << err << "\n";
        return 1;
    }
    const double per_second = got.seconds > 0 ? double(got.tokens) / got.seconds : 0.0;

    std::printf("loss %.9g\n", got.loss);
    std::printf("sum %.9g\n", got.sum);
    std::printf("scored %lld tokens %lld sequences %lld steps %lld windows %lld\n",
                (long long)got.scored, (long long)got.tokens, (long long)got.sequences,
                (long long)got.steps, (long long)got.windows);
    std::printf("forward %.3f s  %.1f tokens/s\n", got.seconds, per_second);
    std::printf("data conversations %lld kept %lld shortened %lld skipped %lld\n",
                (long long)stats.conversations, (long long)stats.kept,
                (long long)stats.shortened, (long long)stats.skipped);
    if (!got.skipped.empty())
        std::printf("over budget %lld sequences\n", (long long)got.skipped.size());
    if (want_vram) {
        if (got.vram_sampled)
            std::printf("vram after first step %lld MiB used of %lld MiB\n",
                        (long long)(mib(got.vram_used) + 0.5),
                        (long long)(mib(got.vram_total) + 0.5));
        if (watch && watch->samples() > 0)
            std::printf("vram peak %lld MiB used of %lld MiB (%d ms samples)\n",
                        (long long)(mib(watch->peak_used()) + 0.5),
                        (long long)(mib(watch->total()) + 0.5), 20);
    }
    return 0;
}

R97_REGISTER_COMMAND("loss", "<chat.jsonl> [--device N]", loss_main);

}  // namespace r97tool
