// src/chatdata.cpp -- ChatData: a chat JSONL file to token ids with an assistant loss mask.
//
// The rendering is rad_core's, not this file's: ChatTemplate::apply is what the radiance server
// runs, and Tokenizer is what it encodes with, so a conversation tokenises here exactly as it
// would there. This file only decides which of those tokens are scored, and what to do with a
// conversation that does not fit.
#include "r97/chatdata.h"

#include "text/chat.h"          // radiance core/text: the server's chat renderer
#include "rad_internal.h"        // radiance: RAD_OK, RAD_E_*, rad_strerror
#include "format/radfile.h"       // radiance core/format: the container's vocab section
#include "text/tokenizer.h"       // radiance core/text: the container's tokenizer

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <utility>

namespace r97 {
namespace {

using nlohmann::ordered_json;

// How many leading tokens two encodings share. A render of a conversation prefix is a token
// prefix of the whole render for a per-message template; where a template differs at its last
// message this stops at the divergence instead of scoring text it did not render.
int common_prefix(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return int(i);
}

bool is_assistant(const std::vector<ChatMessage>& messages, size_t i) {
    return messages[i].role == "assistant";
}

ordered_json json_messages(const std::vector<ChatMessage>& messages, size_t count) {
    ordered_json out = ordered_json::array();
    for (size_t i = 0; i < count && i < messages.size(); ++i)
        out.push_back(ordered_json{{"role", messages[i].role}, {"content", messages[i].content}});
    return out;
}

}  // namespace

struct ChatData::Impl {
    rad::RadFile file;
    std::shared_ptr<const rad::Vocab> vocab;
    rad::Tokenizer tok;
    rad::ChatTemplate tpl;
    mutable std::string error;

    bool ready() const { return vocab != nullptr; }

    // The container's tokenizer, with the server's BOS rule: a tokenizer that states add_bos gets
    // exactly one BOS in front, inserted after encoding rather than through add_special, which
    // would also add an EOS.
    void encode(std::string_view text, std::vector<int32_t>* ids) const {
        ids->clear();
        tok.encode(text, *ids, /*add_special=*/false, /*parse_special=*/true);
        if (vocab->add_bos() && vocab->bos() >= 0 &&
            (ids->empty() || ids->front() != vocab->bos()))
            ids->insert(ids->begin(), vocab->bos());
    }

    bool render_ids(const ordered_json& messages, bool add_generation_prompt,
                    std::vector<int32_t>* ids) const {
        rad::ChatOptions opt;
        opt.add_generation_prompt = add_generation_prompt;
        rad::ChatPrompt prompt;
        const int rc = tpl.apply(messages, ordered_json(), opt, &prompt);
        if (rc < 0) {
            error = std::string("chat template refused the conversation: ") + rad_strerror(rc);
            return false;
        }
        encode(prompt.prompt, ids);
        return true;
    }
};

ChatData::ChatData() : impl_(std::make_unique<Impl>()) {}
ChatData::~ChatData() = default;
ChatData::ChatData(ChatData&&) noexcept = default;
ChatData& ChatData::operator=(ChatData&&) noexcept = default;

bool ChatData::open(const std::string& rad_path) {
    if (!impl_) impl_ = std::make_unique<Impl>();
    impl_->error.clear();
    impl_->vocab.reset();   // a failed reopen leaves no container behind

    int rc = impl_->file.open(rad_path.c_str());
    if (rc < 0) {
        impl_->error = rad_path + ": " + rad_strerror(rc);
        return false;
    }
    const rad::VocabSection vs = impl_->file.vocab();
    if (!vs) {
        impl_->error = rad_path + ": the container carries no vocabulary section";
        return false;
    }
    rad::VocabBuild vb;
    rc = rad::vocab_deserialize(impl_->file.base(), uint64_t(impl_->file.size()), vs.h, vb);
    if (rc < 0) {
        impl_->error = rad_path + ": the vocabulary section does not read: " + rad_strerror(rc);
        return false;
    }
    auto vocab = std::make_shared<rad::Vocab>();
    rc = vocab->load(std::move(vb));
    if (rc < 0) {
        impl_->error = rad_path + ": the vocabulary does not load: " + rad_strerror(rc);
        return false;
    }
    rc = impl_->tok.init(vocab);
    if (rc < 0) {
        impl_->error = rad_path + ": the tokenizer does not build: " + rad_strerror(rc);
        return false;
    }
    rc = impl_->tpl.load_from_vocab(*vocab);
    if (rc < 0) {
        impl_->error = rad_path + ": the chat template does not compile: " + rad_strerror(rc);
        return false;
    }
    impl_->vocab = std::move(vocab);
    return true;
}

ChatDataOutcome ChatData::render(const std::vector<ChatMessage>& messages, int max_tokens,
                                 TokenSeq* out) const {
    if (out) {
        out->ids.clear();
        out->score.clear();
    }
    if (!impl_ || !impl_->ready()) {
        if (impl_) impl_->error = "no container is open";
        return ChatDataOutcome::Error;
    }
    impl_->error.clear();
    if (max_tokens <= 0) {
        impl_->error = "the token limit must be positive";
        return ChatDataOutcome::Error;
    }

    // Keep only through the last assistant turn: messages after it are not part of a training
    // example (a dangling user turn has nothing to score). Then drop trailing turns until the
    // render fits; a conversation whose first exchange does not fit is skipped.
    size_t keep = messages.size();
    bool shortened = false;
    std::vector<int32_t> ids;
    while (true) {
        size_t last = keep;   // the message after the last assistant turn kept
        while (last > 0 && !is_assistant(messages, last - 1)) --last;
        if (last == 0) return ChatDataOutcome::Skipped;   // no assistant turn to score
        keep = last;

        if (!impl_->render_ids(json_messages(messages, keep), /*add_generation_prompt=*/false,
                               &ids))
            return ChatDataOutcome::Error;
        if (ids.size() <= size_t(max_tokens)) break;

        // This render is too long: drop the assistant turn it ends with and try the one before.
        shortened = true;
        keep = last - 1;
    }

    TokenSeq seq;
    seq.ids = ids;
    seq.score.assign(ids.size(), 0);

    // The tokens the model would have generated for assistant turn i are those between the
    // generation prompt that ends at that turn's header (what serving feeds it) and the end of the
    // turn. Their trailing separator (the newline ChatML writes after <|im_end|>) is the
    // template's, not the model's, so the scored span ends at the turn's last control token.
    for (size_t i = 0; i < keep; ++i) {
        if (!is_assistant(messages, i)) continue;

        std::vector<int32_t> header;
        if (!impl_->render_ids(json_messages(messages, i), /*add_generation_prompt=*/true,
                               &header))
            return ChatDataOutcome::Error;   // the template refused an assistant prefix

        std::vector<int32_t> turn;
        if (!impl_->render_ids(json_messages(messages, i + 1), /*add_generation_prompt=*/false,
                               &turn))
            return ChatDataOutcome::Error;

        int start = common_prefix(header, seq.ids);
        int stop = common_prefix(turn, seq.ids);
        if (stop <= start) continue;   // this template does not render a turn we can locate

        for (int j = stop - 1; j >= start; --j) {
            if (impl_->vocab->type(seq.ids[size_t(j)]) == RAD_TT_CONTROL) {
                stop = j + 1;
                break;
            }
        }
        for (int j = start; j < stop; ++j) seq.score[size_t(j)] = 1;
    }

    if (out) *out = std::move(seq);
    return shortened ? ChatDataOutcome::Shortened : ChatDataOutcome::Kept;
}

ChatDataOutcome ChatData::line(std::string_view jsonl, int max_tokens, TokenSeq* out) const {
    if (!impl_) return ChatDataOutcome::Error;
    impl_->error.clear();

    const ordered_json doc = ordered_json::parse(jsonl.begin(), jsonl.end(), nullptr, false);
    if (doc.is_discarded() || !doc.is_object()) {
        impl_->error = "the line is not a JSON object";
        return ChatDataOutcome::Error;
    }
    const auto messages = doc.find("messages");
    if (messages == doc.end() || !messages->is_array()) {
        impl_->error = "the object has no \"messages\" array";
        return ChatDataOutcome::Error;
    }

    std::vector<ChatMessage> chat;
    chat.reserve(messages->size());
    for (const auto& m : *messages) {
        if (!m.is_object()) {
            impl_->error = "a message is not an object";
            return ChatDataOutcome::Error;
        }
        const auto role = m.find("role");
        const auto content = m.find("content");
        if (role == m.end() || !role->is_string()) {
            impl_->error = "a message has no string \"role\"";
            return ChatDataOutcome::Error;
        }
        if (content == m.end() || !content->is_string()) {
            impl_->error = "the \"" + role->get<std::string>() +
                           "\" message has no string \"content\"";
            return ChatDataOutcome::Error;
        }
        chat.push_back(ChatMessage{role->get<std::string>(), content->get<std::string>()});
    }
    return render(chat, max_tokens, out);
}

bool ChatData::read(std::string_view jsonl, int max_tokens, Batch* out,
                    ChatDataStats* stats) const {
    if (!impl_) return false;
    impl_->error.clear();

    size_t pos = 0;
    int64_t line_no = 0;
    while (pos < jsonl.size()) {
        const size_t eol = jsonl.find('\n', pos);
        const std::string_view text =
            jsonl.substr(pos, eol == std::string_view::npos ? std::string_view::npos : eol - pos);
        pos = eol == std::string_view::npos ? jsonl.size() : eol + 1;
        ++line_no;
        if (text.find_first_not_of(" \t\r") == std::string_view::npos) continue;   // blank line

        TokenSeq seq;
        const ChatDataOutcome outcome = line(text, max_tokens, &seq);
        if (outcome == ChatDataOutcome::Error) {
            impl_->error = "line " + std::to_string(line_no) + ": " + impl_->error;
            return false;
        }
        if (stats) ++stats->conversations;
        switch (outcome) {
            case ChatDataOutcome::Skipped:
                if (stats) ++stats->skipped;
                break;
            case ChatDataOutcome::Kept:
            case ChatDataOutcome::Shortened:
                if (stats) {
                    ++stats->kept;
                    if (outcome == ChatDataOutcome::Shortened) ++stats->shortened;
                    stats->tokens += int64_t(seq.ids.size());
                    stats->scored += int64_t(std::count(seq.score.begin(), seq.score.end(), 1));
                }
                if (out) out->push_back(std::move(seq));
                break;
            case ChatDataOutcome::Error:
                break;   // handled above
        }
    }
    return true;
}

bool ChatData::read_file(const std::string& path, int max_tokens, Batch* out,
                         ChatDataStats* stats) const {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        if (impl_) impl_->error = path + ": cannot read the file";
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return read(ss.str(), max_tokens, out, stats);
}

std::string ChatData::token_text(int32_t id, bool first, bool last) const {
    if (!impl_ || !impl_->ready()) return {};
    return impl_->tok.piece_at(id, /*render_special=*/true, rad::Tokenizer::At{first, last});
}

const std::string& ChatData::error() const {
    static const std::string empty;
    return impl_ ? impl_->error : empty;
}

}  // namespace r97
