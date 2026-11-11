// tools/data.cpp -- the `data` subcommand: a chat JSONL file rendered through the container's
// own chat template and tokenizer, with the assistant tokens marked. It reports what happened to
// the conversations (kept, shortened, skipped) and their scored token count, and with --show N
// prints the first N of them with each scored token wrapped in [[...]].
#include "commands.h"

#include "r97/chatdata.h"

#include <iostream>
#include <string>

namespace r97tool {
namespace {

struct Args {
    std::string model;
    std::string data;
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
        std::cerr << "usage: r97train data --model <file.rad> --data <file.jsonl> [--show N]\n";
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
    return 0;
}

R97_REGISTER_COMMAND("data", "--model <file.rad> --data <file.jsonl> [--show N]", data_main);

}  // namespace
}  // namespace r97tool
