// tests/model/chatdata_render.cpp -- ChatData tokenises a conversation exactly as radiance's own
// chat rendering does on the real container, and marks the assistant turns it renders.
//
// The reference here is built from rad_core directly (ChatTemplate::load_from_vocab,
// ChatTemplate::apply, Tokenizer::encode), not from ChatData, so a different template, tokenizer
// setting or BOS rule in the trainer shows up as a mismatch. No card: reading a container's vocab
// section and rendering text need none.
#include "r97/chatdata.h"

#include "../support/model_dir.h"
#include "../support/self_check.h"

#include "format/radfile.h"
#include "rad_internal.h"
#include "text/chat.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace {

using nlohmann::ordered_json;

const std::vector<r97::ChatMessage> kMessages = {
    {"system", "You are terse."},
    {"user", "Hi"},
    {"assistant", "Hello!"},
    {"user", "What is two plus two?"},
    {"assistant", "Four."},
};

// radiance's rendering of the same messages, as the server would tokenise it: the template's own
// prompt, the tokenizer, and the BOS a tokenizer that states add_bos gets in front.
std::vector<int32_t> radiance_ids(const std::string& rad,
                                  const std::vector<r97::ChatMessage>& messages) {
    rad::RadFile file;
    R97_CHECK(file.open(rad.c_str()) == RAD_OK);
    const rad::VocabSection vs = file.vocab();
    R97_CHECK(bool(vs));

    rad::VocabBuild vb;
    R97_CHECK(rad::vocab_deserialize(file.base(), uint64_t(file.size()), vs.h, vb) == RAD_OK);
    auto vocab = std::make_shared<rad::Vocab>();
    R97_CHECK(vocab->load(std::move(vb)) == RAD_OK);

    rad::Tokenizer tokenizer;
    R97_CHECK(tokenizer.init(vocab) == RAD_OK);

    rad::ChatTemplate tpl;
    R97_CHECK(tpl.load_from_vocab(*vocab) == RAD_OK);

    ordered_json json = ordered_json::array();
    for (const r97::ChatMessage& m : messages)
        json.push_back(ordered_json{{"role", m.role}, {"content", m.content}});

    rad::ChatOptions opt;   // a complete conversation: no generation prompt is appended
    opt.add_generation_prompt = false;
    rad::ChatPrompt prompt;
    R97_CHECK(tpl.apply(json, ordered_json(), opt, &prompt) == RAD_OK);

    std::vector<int32_t> ids;
    tokenizer.encode(prompt.prompt, ids, /*add_special=*/false, /*parse_special=*/true);
    if (vocab->add_bos() && vocab->bos() >= 0 && (ids.empty() || ids.front() != vocab->bos()))
        ids.insert(ids.begin(), vocab->bos());
    return ids;
}

}  // namespace

static void chatdata_render_matches_radiance_chat_rendering() {
    const std::string rad = r97test::container_or_skip(R97_MODEL_RAD);

    r97::ChatData data;
    R97_CHECK(data.open(rad));

    r97::TokenSeq seq;
    R97_CHECK(data.render(kMessages, r97::ChatData::kDefaultMaxTokens, &seq) ==
              r97::ChatDataOutcome::Kept);

    // The ids are the container's own chat rendering, token for token.
    R97_CHECK(seq.ids == radiance_ids(rad, kMessages));
    R97_CHECK(seq.ids.size() == seq.score.size());
    R97_CHECK(!seq.ids.empty());

    // The mask starts inside the assistant header and stops at the end-of-turn token, so the
    // first token and the template's trailing separator are context.
    R97_CHECK(seq.score.front() == 0);
    R97_CHECK(seq.score.back() == 0);
    R97_CHECK(std::count(seq.score.begin(), seq.score.end(), 1) > 0);

    // The scored text is the assistants' answers, and the users' and system's text is context.
    std::string scored;
    for (size_t i = 0; i < seq.ids.size(); ++i)
        if (seq.score[i]) scored += data.token_text(seq.ids[i], i == 0, i + 1 == seq.ids.size());
    R97_CHECK(scored.find("Hello!") != std::string::npos);
    R97_CHECK(scored.find("Four.") != std::string::npos);
    R97_CHECK(scored.find("What is two plus two?") == std::string::npos);

    // The last scored token is the second answer's end-of-turn token, not whatever the template
    // writes after it (ChatML's newline), which is why the mask does not run to the end.
    const auto scored_end = std::find(seq.score.rbegin(), seq.score.rend(), uint8_t(1)).base();
    const size_t last = size_t(scored_end - seq.score.begin()) - 1;
    R97_CHECK(last + 1 < seq.ids.size());
    R97_CHECK(!seq.score.back());
}

R97_SELF_CHECK(chatdata_render_matches_radiance_chat_rendering)
