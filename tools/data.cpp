// tools/data.cpp -- the `data` subcommand: a chat JSONL file rendered through the container's
// own chat template and tokenizer, with the assistant tokens marked. It reports what happened to
// the conversations (kept, shortened, skipped) and their scored token count, and with --show N
// prints the first N of them with each scored token wrapped in [[...]]. With --kld-corpus FILE it
// also writes the same file as radiance's KL corpus (docs/EVAL.md), so the loss it describes can
// be measured by radiance itself.
#include "commands.h"

#include "r97/chatdata.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <string>

namespace r97tool {
namespace {

struct Args {
    std::string model;
    std::string data;
    std::string kld;
    int show = 0;
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
        } else if (arg == "--kld-corpus") {
            if (!next(&a->kld)) return false;
        } else if (arg == "--show") {
            std::string n;
            if (!next(&n)) return false;
            try {
                a->show = std::stoi(n);
            } catch (...) {
                return false;
            }
            if (a->show < 0) return false;
        } else {
            std::cerr << "r97train data: unknown argument " << arg << "\n";
            return false;
        }
    }
    if (a->model.empty() || a->data.empty()) return false;
    return true;
}

// The scored tokens are wrapped so the mask is visible in the rendered text; the rest is what the
// template wrote and the conversation said.
void show(const r97::ChatData& chat, const r97::TokenSeq& seq, size_t index, std::ostream& out) {
    int64_t scored = 0;
    std::string marked;
    for (size_t i = 0; i < seq.ids.size(); ++i) {
        const std::string text =
            chat.token_text(seq.ids[i], /*first=*/i == 0, /*last=*/i + 1 == seq.ids.size());
        if (seq.score[i]) {
            marked += "[[" + text + "]]";
            ++scored;
        } else {
            marked += text;
        }
    }
    out << "conversation " << index + 1 << ": " << seq.ids.size() << " tokens, " << scored
        << " scored\n"
        << marked << "\n";
}

int data_main(int argc, char** argv) {
    Args args;
    if (!parse(argc, argv, &args)) {
        std::cerr << "usage: r97train data --model <file.rad> --data <file.jsonl> [--show N]"
                     " [--kld-corpus <file>]\n";
        return 2;
    }

    r97::ChatData chat;
    if (!chat.open(args.model)) {
        std::cerr << "r97train data: " << chat.error() << "\n";
        return 3;
    }

    r97::Batch batch;
    r97::ChatDataStats stats;
    if (!chat.read_file(args.data, r97::ChatData::kDefaultMaxTokens, &batch, &stats)) {
        std::cerr << "r97train data: " << chat.error() << "\n";
        return 3;
    }

    std::cout << "conversations: " << stats.conversations << "\n"
              << "kept:          " << stats.kept << "\n"
              << "shortened:     " << stats.shortened << "\n"
              << "skipped:       " << stats.skipped << "\n"
              << "tokens:        " << stats.tokens << "\n"
              << "scored:        " << stats.scored << "\n";

    const size_t shown = size_t(args.show) < batch.size() ? size_t(args.show) : batch.size();
    for (size_t i = 0; i < shown; ++i) show(chat, batch[i], i, std::cout);

    if (!args.kld.empty()) {
        // radiance's corpus is one document per contiguous scored run: the run's sequence through
        // its last target, and score_from the row that predicts the run's first target. Its rows
        // are then the run's targets -- exactly the tokens `scored` counts and `loss` is taken
        // over -- which is what lets `tools/kld_loss.py` read the same number back.
        std::ofstream out(args.kld);
        if (!out) {
            std::cerr << "r97train data: cannot write " << args.kld << "\n";
            return 3;
        }
        int64_t documents = 0;
        int64_t rows = 0;
        for (const r97::TokenSeq& seq : batch) {
            if (!seq.score.empty() && seq.score[0]) {
                std::cerr << "r97train data: a sequence's first token is scored, which radiance's "
                             "corpus cannot express (score_from would be -1)\n";
                return 3;
            }
            for (size_t i = 1; i < seq.score.size();) {
                if (!seq.score[i]) {
                    ++i;
                    continue;
                }
                const size_t start = i;  // first target token of this run
                while (i < seq.score.size() && seq.score[i]) ++i;
                const size_t stop = i;  // one past the last target token

                std::string text;
                for (size_t j = 0; j < stop; ++j)
                    text += chat.token_text(seq.ids[j], j == 0, j + 1 == seq.ids.size());
                out << nlohmann::json{{"prompt", text},
                                      {"score_from", int64_t(start - 1)},
                                      {"source", "data:" + std::to_string(documents)}}
                           .dump()
                    << "\n";
                rows += int64_t(stop - start);
                ++documents;
            }
        }
        if (documents == 0) {
            std::cerr << "r97train data: the file has no scored run to give radiance\n";
            return 3;
        }
        std::cout << "kld documents: " << documents << "\n"
                  << "kld rows:      " << rows << "\n";
    }
    return 0;
}

R97_REGISTER_COMMAND("data", "--model <file.rad> --data <file.jsonl> [--show N] [--kld-corpus <file>]",
                     data_main);

}  // namespace
}  // namespace r97tool
