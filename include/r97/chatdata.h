// r97/chatdata.h -- chat JSONL to token ids with an assistant loss mask.
//
// Training and serving have to tokenise a conversation identically, so this file carries no
// template and no tokenizer of its own: ChatData renders with the container's own chat template
// and tokenizer through rad_core (core/text/chat.h, the code the radiance server runs). What it
// adds is the loss mask: a token is scored when the model would have generated it, which is each
// assistant turn's content and its end-of-turn token, and never the template's role markers or the
// system and user text.
//
// A conversation longer than the limit drops trailing turns until it fits; one whose first
// exchange does not fit is skipped. `r97train data` reports what happened and shows the first
// conversations with their scored tokens marked. docs/CHATDATA.md has the details.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "r97/batch.h"

namespace r97 {

// One message of a chat conversation, as the JSONL file writes it.
struct ChatMessage {
    std::string role;     // "system", "user" or "assistant" -- whatever the template knows
    std::string content;
};

// What a JSONL file's conversations came to. `kept` counts every conversation that produced a
// sequence, `shortened` the ones that fit only after trailing turns were dropped, and `skipped`
// the ones with no assistant turn or a first exchange that does not fit.
struct ChatDataStats {
    int64_t conversations = 0;   // JSONL lines that held a conversation
    int64_t kept = 0;            // conversations that produced a sequence (shortened ones too)
    int64_t shortened = 0;       // conversations that fit only after trailing turns were dropped
    int64_t skipped = 0;         // conversations with no assistant turn or a first exchange too long
    int64_t tokens = 0;          // tokens over the kept sequences
    int64_t scored = 0;          // scored tokens over the kept sequences
};

// What happened to one conversation.
enum class ChatDataOutcome {
    Kept,        // rendered whole; only messages after its last assistant turn were left out
    Shortened,   // fit only after trailing turns were dropped
    Skipped,     // no assistant turn, or even the first exchange is longer than the limit
    Error,       // malformed conversation, unreadable container, or a template that refused it
};

class ChatData {
public:
    static constexpr int kDefaultMaxTokens = 4096;

    ChatData();
    ~ChatData();
    ChatData(ChatData&&) noexcept;
    ChatData& operator=(ChatData&&) noexcept;
    ChatData(const ChatData&) = delete;
    ChatData& operator=(const ChatData&) = delete;

    // Load the container's tokenizer and chat template. False, with the reason in error(), when
    // the file is not a container or carries no vocabulary section.
    bool open(const std::string& rad_path);

    // Render one conversation and score its assistant turns. On Kept and Shortened `out` holds
    // the token ids and a score of the same length. On Error error() says why.
    ChatDataOutcome render(const std::vector<ChatMessage>& messages, int max_tokens,
                           TokenSeq* out) const;

    // Parse one JSONL line holding {"messages": [...]} and render it as render() does.
    ChatDataOutcome line(std::string_view jsonl, int max_tokens, TokenSeq* out) const;

    // Read every conversation of a JSONL document or file, appending the kept sequences to `out`
    // and counting all outcomes into `stats`. False on Error, with the line number in error().
    bool read(std::string_view jsonl, int max_tokens, Batch* out, ChatDataStats* stats) const;
    bool read_file(const std::string& path, int max_tokens, Batch* out, ChatDataStats* stats) const;

    // A token's text after the decoder chain, for printing a sequence with scored tokens marked.
    // `first` and `last` are the neighbouring tokens' positions, as the decoder wants them.
    std::string token_text(int32_t id, bool first, bool last) const;

    // Why the last call failed. Empty when it did not.
    const std::string& error() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace r97
