// tests/host/chatdata_mask.cpp -- where ChatData puts the assistant mask, exactly.
//
// The expected token ids are built here from the fixture's ChatML template as text
// ("<|im_start|>role\ncontent<|im_end|>\n") and its byte-level tokenizer, not from ChatData, so a
// moved mask boundary fails the test. chat_small.jsonl holds the same three conversations for the
// reader side.
#include "r97/chatdata.h"

#include "support/fixture.h"
#include "support/test.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace {

struct Expected {
    std::vector<int32_t> ids;
    std::vector<uint8_t> score;
};

// The fixture's tokenizer maps every byte to its own id, so this is exact for plain text.
void add_text(Expected& e, const std::string& text, uint8_t score) {
    for (const int32_t id : tiny::byte_ids(text)) {
        e.ids.push_back(id);
        e.score.push_back(score);
    }
}

void add_special(Expected& e, int32_t id, uint8_t score) {
    e.ids.push_back(id);
    e.score.push_back(score);
}

// One message as the fixture's ChatML template writes it. An assistant message is scored on its
// content and its end-of-turn token; the role header and the separator after the turn are not.
void add_message(Expected& e, const std::string& role, const std::string& content, bool scored) {
    add_special(e, tiny::kImStart, 0);
    add_text(e, role, 0);
    add_text(e, "\n", 0);
    add_text(e, content, scored ? 1 : 0);
    add_special(e, tiny::kImEnd, scored ? 1 : 0);
    add_text(e, "\n", 0);
}

Expected expected(const std::vector<r97::ChatMessage>& messages) {
    Expected e;
    for (const r97::ChatMessage& m : messages)
        add_message(e, m.role, m.content, m.role == "assistant");
    return e;
}

r97::TokenSeq render(const r97::ChatData& data, const std::vector<r97::ChatMessage>& messages,
                     int max_tokens, r97::ChatDataOutcome want) {
    r97::TokenSeq seq;
    const r97::ChatDataOutcome got = data.render(messages, max_tokens, &seq);
    R97_CHECK(got == want);
    return seq;
}

void check(const r97::TokenSeq& got, const Expected& want) {
    R97_CHECK(got.ids == want.ids);
    R97_CHECK(got.score == want.score);
}

r97::ChatData opened_fixture() {
    r97::ChatData data;
    R97_CHECK(data.open(r97::fixture().container));
    return data;
}

const std::vector<r97::ChatMessage> kOneTurn = {
    {"user", "Hi"},
    {"assistant", "Hello!"},
};

const std::vector<r97::ChatMessage> kThreeTurns = {
    {"user", "Hi"},
    {"assistant", "Hello!"},
    {"user", "What is two plus two?"},
    {"assistant", "Four."},
    {"user", "Thanks"},
    {"assistant", "You are welcome!"},
};

const std::vector<r97::ChatMessage> kWithSystem = {
    {"system", "You are terse."},
    {"user", "Hi"},
    {"assistant", "Hello!"},
};

std::string chat_small_jsonl() {
    return (std::filesystem::path(__FILE__).parent_path() / ".." / "fixtures" / "chat_small.jsonl")
        .string();
}

}  // namespace

R97_TEST(chatdata_mask_one_turn) {
    const r97::ChatData data = opened_fixture();
    const r97::TokenSeq seq =
        render(data, kOneTurn, r97::ChatData::kDefaultMaxTokens, r97::ChatDataOutcome::Kept);
    check(seq, expected(kOneTurn));

    // The answer's six bytes and the end-of-turn token are the only scored tokens.
    R97_CHECK(std::count(seq.score.begin(), seq.score.end(), 1) == 7);
    R97_CHECK(seq.score.front() == 0);
}

R97_TEST(chatdata_mask_three_turns) {
    const r97::ChatData data = opened_fixture();
    const r97::TokenSeq seq =
        render(data, kThreeTurns, r97::ChatData::kDefaultMaxTokens, r97::ChatDataOutcome::Kept);
    check(seq, expected(kThreeTurns));

    // Every assistant turn is scored, and no user text is.
    const Expected want = expected(kThreeTurns);
    R97_CHECK(std::count(seq.score.begin(), seq.score.end(), 1) ==
              std::count(want.score.begin(), want.score.end(), 1));
}

R97_TEST(chatdata_mask_with_system_message) {
    const r97::ChatData data = opened_fixture();
    const r97::TokenSeq seq =
        render(data, kWithSystem, r97::ChatData::kDefaultMaxTokens, r97::ChatDataOutcome::Kept);
    check(seq, expected(kWithSystem));

    // The system text is context, not a target: no scored token starts before the answer.
    R97_CHECK(seq.score.front() == 0);
    R97_CHECK(seq.score[1] == 0);
}

R97_TEST(chatdata_shortens_trailing_turns_until_it_fits) {
    const r97::ChatData data = opened_fixture();
    const Expected one =
        expected(std::vector<r97::ChatMessage>(kThreeTurns.begin(), kThreeTurns.begin() + 2));
    const Expected two =
        expected(std::vector<r97::ChatMessage>(kThreeTurns.begin(), kThreeTurns.begin() + 4));
    const Expected full = expected(kThreeTurns);
    R97_CHECK(full.ids.size() > two.ids.size() && two.ids.size() > one.ids.size());

    // Exactly two turns fit: the third is dropped whole, not cut in the middle.
    const r97::TokenSeq shortened =
        render(data, kThreeTurns, int(two.ids.size()), r97::ChatDataOutcome::Shortened);
    check(shortened, two);

    // One token less and only the first exchange fits; the first exchange alone still fits.
    check(render(data, kThreeTurns, int(two.ids.size()) - 1, r97::ChatDataOutcome::Shortened), one);
    check(render(data, kThreeTurns, int(one.ids.size()), r97::ChatDataOutcome::Shortened), one);

    // A first exchange that does not fit skips the conversation, with no sequence produced.
    r97::TokenSeq seq;
    R97_CHECK(data.render(kThreeTurns, int(one.ids.size()) - 1, &seq) ==
              r97::ChatDataOutcome::Skipped);
    R97_CHECK(seq.ids.empty() && seq.score.empty());
}

R97_TEST(chatdata_reads_the_small_chat_file) {
    const r97::ChatData data = opened_fixture();
    r97::Batch batch;
    r97::ChatDataStats stats;
    R97_CHECK(data.read_file(chat_small_jsonl(), r97::ChatData::kDefaultMaxTokens, &batch, &stats));

    R97_CHECK(stats.conversations == 3);
    R97_CHECK(stats.kept == 3);
    R97_CHECK(stats.shortened == 0);
    R97_CHECK(stats.skipped == 0);
    R97_CHECK(batch.size() == 3);

    const std::vector<std::vector<r97::ChatMessage>> want = {kOneTurn, kThreeTurns, kWithSystem};
    int64_t tokens = 0, scored = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        const Expected e = expected(want[i]);
        check(batch[i], e);
        tokens += int64_t(e.ids.size());
        scored += std::count(e.score.begin(), e.score.end(), 1);
    }
    R97_CHECK(stats.tokens == tokens);
    R97_CHECK(stats.scored == scored);
}

R97_TEST(chatdata_skips_a_conversation_with_no_assistant_turn) {
    const r97::ChatData data = opened_fixture();
    r97::Batch batch;
    r97::ChatDataStats stats;
    R97_CHECK(data.read(R"({"messages": [{"role": "user", "content": "Hi"}]})",
                        r97::ChatData::kDefaultMaxTokens, &batch, &stats));
    R97_CHECK(stats.conversations == 1);
    R97_CHECK(stats.skipped == 1);
    R97_CHECK(stats.kept == 0);
    R97_CHECK(batch.empty());
}

R97_TEST(chatdata_reports_a_malformed_line_and_its_number) {
    const r97::ChatData data = opened_fixture();
    r97::Batch batch;
    r97::ChatDataStats stats;
    R97_CHECK(!data.read("{\"messages\": [{\"role\": \"user\", \"content\": \"Hi\"},"
                         "{\"role\": \"assistant\", \"content\": \"Hello!\"}]}\n"
                         "{\"messages\": 3}\n",
                         r97::ChatData::kDefaultMaxTokens, &batch, &stats));
    R97_CHECK(data.error().rfind("line 2:", 0) == 0);
    R97_CHECK(batch.size() == 1);   // the first line still counted and appended
}

