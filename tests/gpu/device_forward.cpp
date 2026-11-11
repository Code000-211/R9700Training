// tests/gpu/device_forward.cpp -- the training forward on a card: its loss against radiance's
// own, the tiny int8 fixture's device loss against the host's, and a 4096-token sequence.
//
// The cases, and the shape each one runs:
//
//   device_forward_tiny_fixture  three sequences of 7, 13 and 5 tokens (25 together, 22 scored)
//                                through the tiny w8a8 fixture, on libavx,libref and on libr4d;
//                                the means agree to 1e-4 relative.
//
//   device_loss_matches_kld     32 one-exchange conversations of tests/fixtures/chat_32.jsonl,
//                                each rendered by ChatData and, separately, by the same chat
//                                template and tokenizer written to a `radiance --kld-record`
//                                corpus; the device loss over the scored positions agrees with
//                                the recorded NLL over those positions to 1e-3 relative.
//
//   device_forward_4096         one sequence of exactly 4096 tokens (4095 scored), one prefill
//                                step with max_tok = max_ctx = 4096; the step runs, the loss is
//                                finite, and the case prints the VRAM after loading, the peak of
//                                the step and the prefill tokens per second.
//
// Card and container: the guard takes R97_DEVICE's card, and the model cases skip (exit 77) when
// no container is found. Radiance ranks the devices HIP shows it, so a device test must run with
// ROCR_VISIBLE_DEVICES naming R97_DEVICE's card and R97_DEVICE=0; when the environment does not
// do that, the cases below refuse rather than measure another card's engine.
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

#include "r97/chatdata.h"
#include "r97/device_tier.h"
#include "r97/loss_run.h"

#include "format/radfile.h"
#include "rad_internal.h"
#include "text/chat.h"
#include "text/tokenizer.h"

#include "support/fixture_w8a8.h"
#include "support/gpu_guard.h"
#include "support/model_case.h"
#include "support/model_dir.h"
#include "support/test.h"

namespace {

namespace fs = std::filesystem;
using nlohmann::ordered_json;

// ---------------------------------------------------------------- the card and the container

// The guard for the card R97_DEVICE names, with the card the radiance engine will actually use
// checked to be that card. Radiance's rank 0 is HIP's ordinal 0; when ROCR_VISIBLE_DEVICES does
// not name R97_DEVICE's card, HIP's ordinal 0 is another card, and the test would measure it
// while the guard held a different lock. HIP can say which physical card ordinal 0 is even after
// visibility has hidden it from the engine, so comparing the two bus ids catches that setup.
r97test::GuardOptions card_options(int64_t need_vram) {
    r97test::GuardOptions o;
    o.device = r97test::parse_device(std::getenv("R97_DEVICE") ? std::getenv("R97_DEVICE") : "0");
    o.bdf = r97test::hip_bdf(o.device);
    o.need_vram = need_vram;
    return o;
}

struct Card {
    r97test::DeviceGuard guard;
    explicit Card(int64_t need_vram) : guard(card_options(need_vram)) {
        const int want = card_options(0).device;
        const std::string zero = r97test::hip_bdf(0);
        const std::string named = r97test::hip_bdf(want);
        if (zero.empty() || named.empty() || zero != named)
            throw std::runtime_error(
                "the engine would run on HIP ordinal 0 (" + (zero.empty() ? "?" : zero)
                + "), not R97_DEVICE's card " + std::to_string(want) + " ("
                + (named.empty() ? "?" : named)
                + "); run with ROCR_VISIBLE_DEVICES=" + std::to_string(want)
                + " R97_DEVICE=0 so the two name the same card");
    }
};

std::string quote(const std::string& s) {
    std::string q = "'";
    for (const char c : s) {
        if (c == '\'')
            q += "'\\''";
        else
            q += c;
    }
    return q + "'";
}

std::string radiance_bin(const std::string& home) {
    if (const char* e = std::getenv("R97_RADIANCE_BIN"); e && *e) return e;
    for (const fs::path& p : {fs::path(home) / "../bin/radiance", fs::path(home) / "radiance"})
        if (fs::exists(p)) return p.string();
    return "";
}

std::string temp_dir(const char* tag) {
    std::string tmpl = std::string("/tmp/r97-") + tag + "-XXXXXX";
    if (::mkdtemp(tmpl.data()) == nullptr)
        throw std::runtime_error("cannot make a temporary directory");
    return tmpl;
}

// A relative difference against the reference, with 1e-3 asked for by the issue's first
// acceptance case and 1e-4 for the fixture, where both tiers run the same w8a8 container.
void close_enough(double got, double want, double tol, const std::string& what) {
    const double scale = want != 0.0 ? std::fabs(want) : 1.0;
    if (!(std::fabs(got - want) <= tol * scale))
        throw std::runtime_error(what + ": " + std::to_string(got) + " against "
                                 + std::to_string(want) + " (relative "
                                 + std::to_string(std::fabs(got - want) / scale) + " > "
                                 + std::to_string(tol) + ")");
}

// ---------------------------------------------------------------- the chat template mirror

// ChatData's renderer, kept here so the test can also write the prompt *text* radiance's kld mode
// tokenises. It is the same three calls ChatData makes (ChatTemplate::apply, Tokenizer::encode,
// then the container's BOS rule), and the cases below assert that its ids equal ChatData's, so a
// divergence fails here rather than turning into a loss comparison of different tokens.
class Mirror {
public:
    bool open(const std::string& rad_path) {
        if (file_.open(rad_path.c_str()) < 0) return false;
        const rad::VocabSection vs = file_.vocab();
        if (!vs) return false;
        rad::VocabBuild vb;
        if (rad::vocab_deserialize(file_.base(), uint64_t(file_.size()), vs.h, vb) < 0) return false;
        auto vocab = std::make_shared<rad::Vocab>();
        if (vocab->load(std::move(vb)) < 0) return false;
        if (tok_.init(vocab) < 0) return false;
        if (tpl_.load_from_vocab(*vocab) < 0) return false;
        vocab_ = std::move(vocab);
        return true;
    }

    // The prompt text and ids for a conversation, without a generation prompt: exactly what
    // ChatData::render encodes for its first exchange.
    bool prompt(const std::vector<r97::ChatMessage>& messages, std::string* text,
                std::vector<int32_t>* ids) const {
        ordered_json json = ordered_json::array();
        for (const r97::ChatMessage& m : messages)
            json.push_back(ordered_json{{"role", m.role}, {"content", m.content}});
        rad::ChatOptions opt;
        opt.add_generation_prompt = false;
        rad::ChatPrompt prompt;
        if (tpl_.apply(json, ordered_json(), opt, &prompt) < 0) return false;
        *text = prompt.prompt;
        ids->clear();
        tok_.encode(*text, *ids, /*add_special=*/false, /*parse_special=*/true);
        if (vocab_->add_bos() && vocab_->bos() >= 0 &&
            (ids->empty() || ids->front() != vocab_->bos()))
            ids->insert(ids->begin(), vocab_->bos());
        return true;
    }

private:
    rad::RadFile file_;
    std::shared_ptr<rad::Vocab> vocab_;
    rad::Tokenizer tok_;
    rad::ChatTemplate tpl_;
};

// ---------------------------------------------------------------- kld records

struct KldDoc {
    int64_t tokens = 0;
    int64_t score_from = 0;
    int64_t row0 = 0;
};

struct KldRecord {
    std::vector<int32_t> tokens;
    std::vector<float> rows;  // stride 2: {nll, entropy}
    std::vector<KldDoc> docs;
};

std::string read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

KldRecord read_kld(const std::string& dir) {
    KldRecord r;
    const ordered_json m = ordered_json::parse(read_text(dir + "/kld.json"));
    const std::string tok = read_text(dir + "/tokens.i32");
    if (tok.size() % 4 != 0) throw std::runtime_error("tokens.i32 is not a whole number of i32");
    r.tokens.resize(tok.size() / 4);
    std::memcpy(r.tokens.data(), tok.data(), tok.size());
    const std::string rows = read_text(dir + "/rows.f32");
    const int64_t n_rows = m.at("rows").get<int64_t>();
    if (int64_t(rows.size()) != n_rows * 2 * 4)
        throw std::runtime_error("rows.f32 is " + std::to_string(rows.size())
                                 + " bytes, not rows * 2 * 4");
    r.rows.resize(size_t(n_rows) * 2);
    std::memcpy(r.rows.data(), rows.data(), rows.size());
    for (const auto& d : m.at("docs")) {
        KldDoc doc;
        doc.tokens = d.at("tokens").get<int64_t>();
        doc.score_from = d.at("score_from").get<int64_t>();
        doc.row0 = d.at("row0").get<int64_t>();
        r.docs.push_back(doc);
    }
    return r;
}

int run_command(const std::string& cmd, std::string* log) {
    std::FILE* p = ::popen((cmd + " 2>&1").c_str(), "r");
    if (!p) return -1;
    log->clear();
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, p)) > 0) log->append(buf, n);
    const int status = ::pclose(p);
    return status >= 0 && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

std::vector<std::string> lines(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::vector<std::string> out;
    for (std::string line; std::getline(in, line);)
        if (!line.empty()) out.push_back(line);
    return out;
}

}  // namespace

// ---------------------------------------------------------------- the tiny int8 fixture

namespace {

// n tokens over `lengths`, ids in [0, n_vocab), every position but the first scored.
r97::Batch shaped_sequences(const std::vector<int>& lengths, int32_t n_vocab) {
    r97::Batch out;
    int32_t next = 1;
    for (const int n : lengths) {
        r97::TokenSeq s;
        for (int i = 0; i < n; ++i) {
            s.ids.push_back((next * 37 + 11) % n_vocab);
            s.score.push_back(i == 0 ? 0 : 1);
            ++next;
        }
        out.push_back(s);
    }
    return out;
}

r97::LossRunConfig forward_on(const std::string& container, const std::string& home,
                              const std::vector<std::string>& kernels, int64_t max_tok,
                              int64_t max_ctx, int64_t max_seqs, int64_t window) {
    r97::LossRunConfig c;
    c.forward.model = container;
    c.forward.radiance_home = home;
    c.forward.kernels = kernels;
    c.forward.max_tok = max_tok;
    c.forward.max_ctx = max_ctx;
    c.forward.max_seqs = max_seqs;
    c.forward.window_rows = window;
    c.token_budget = max_tok;
    return c;
}

}  // namespace

R97_TEST(device_forward_tiny_fixture) {
    r97test::GuardOptions o;
    o.device = r97test::parse_device(std::getenv("R97_DEVICE") ? std::getenv("R97_DEVICE") : "0");
    o.bdf = r97test::hip_bdf(o.device);
    o.need_vram = int64_t(3) << 30;
    r97test::DeviceGuard guard(o);

    const r97::Fixture f = r97::w8a8_fixture();
    const std::string home = r97::default_radiance_home();
    R97_CHECK(!home.empty());

    const std::vector<int> lengths = {7, 13, 5};  // 25 tokens, 22 scored
    const r97::Batch seqs = shaped_sequences(lengths, f.n_vocab);

    r97::LossRunResult host, device;
    std::string err;
    R97_CHECK(r97::run_loss(forward_on(f.container, home, {"libavx", "libref"}, 25, 13, 3, 8),
                            seqs, &host, &err));
    R97_CHECK(r97::run_loss(forward_on(f.container, home, {"libr4d"}, 25, 13, 3, 8), seqs,
                            &device, &err));
    R97_CHECK(host.steps == 1 && device.steps == 1);
    R97_CHECK(host.scored == 22 && device.scored == 22);
    std::fprintf(stderr,
                 "tiny fixture: host loss %.9g, device loss %.9g, %.3f s on the card\n",
                 host.loss, device.loss, device.seconds);
    close_enough(device.loss, host.loss, 1e-4, "the tiny fixture's device loss");
}

// ---------------------------------------------------------------- the real container

namespace {

struct Conversation {
    std::vector<r97::ChatMessage> messages;
    r97::TokenSeq seq;
    std::string prompt;    // the chat template's text for the exchange
    int score_from = 0;    // the kld corpus' score_from: the row that predicts the first scored token
};

std::string model_container() { return r97test::container_or_skip(R97_MODEL_RAD); }

// Read tests/fixtures/chat_32.jsonl through ChatData and through the mirror, and check the two
// renders hold the same ids.
std::vector<Conversation> read_conversations(const std::string& container,
                                             const std::string& path) {
    r97::ChatData chat;
    if (!chat.open(container))
        throw std::runtime_error("the chat data does not open: " + chat.error());
    Mirror mirror;
    if (!mirror.open(container)) throw std::runtime_error("the chat template mirror does not open");

    std::vector<Conversation> out;
    for (const std::string& line : lines(path)) {
        const ordered_json j = ordered_json::parse(line);
        Conversation c;
        for (const auto& m : j.at("messages"))
            c.messages.push_back({m.at("role").get<std::string>(),
                                  m.at("content").get<std::string>()});
        if (chat.line(line, 4096, &c.seq) != r97::ChatDataOutcome::Kept)
            throw std::runtime_error("a conversation of " + path + " was not kept: " + chat.error());
        std::vector<int32_t> mirrored;
        if (!mirror.prompt(c.messages, &c.prompt, &mirrored))
            throw std::runtime_error("the chat template mirror refused a conversation");
        if (mirrored != c.seq.ids)
            throw std::runtime_error("the mirror's ids differ from ChatData's: the kld corpus "
                                     "would name different tokens");
        size_t first = 0;
        while (first < c.seq.score.size() && !c.seq.score[first]) ++first;
        if (first == 0 || first >= c.seq.score.size())
            throw std::runtime_error("a conversation has no scored token after its first position");
        c.score_from = int(first) - 1;  // kld's row for position p predicts token p+1
        out.push_back(std::move(c));
    }
    return out;
}

// The device loss mean over the kld rows the conversations' score masks select.
double kld_loss_over_scored(const KldRecord& record,
                            const std::vector<Conversation>& conversations) {
    double sum = 0.0;
    int64_t n = 0;
    int64_t cursor = 0;  // the kld token stream's offset for the next document
    for (size_t d = 0; d < conversations.size(); ++d) {
        const KldDoc& doc = record.docs[d];
        const Conversation& c = conversations[d];
        if (doc.tokens != int64_t(c.seq.ids.size()))
            throw std::runtime_error("document " + std::to_string(d) + ": kld has "
                                     + std::to_string(doc.tokens) + " tokens, ChatData "
                                     + std::to_string(c.seq.ids.size()));
        if (doc.score_from != c.score_from)
            throw std::runtime_error("document " + std::to_string(d) + ": kld scored from "
                                     + std::to_string(doc.score_from) + ", the corpus asked for "
                                     + std::to_string(c.score_from));
        for (int64_t i = 0; i < doc.tokens; ++i)
            if (record.tokens[size_t(cursor + i)] != c.seq.ids[size_t(i)])
                throw std::runtime_error("document " + std::to_string(d)
                                         + ": kld token " + std::to_string(i) + " differs");
        // radiance scores every position p in [score_from, tokens-1); keep the positions whose
        // predicted token this conversation scores.
        for (int64_t p = doc.score_from; p + 1 < doc.tokens; ++p) {
            if (!c.seq.score[size_t(p + 1)]) continue;
            const int64_t row = doc.row0 + (p - doc.score_from);
            sum += record.rows[size_t(row) * 2];
            n += 1;
        }
        cursor += doc.tokens;
    }
    if (n == 0) throw std::runtime_error("the kld record selected no scored position");
    return sum / double(n);
}

std::string write_corpus(const std::vector<Conversation>& conversations, const std::string& dir) {
    const std::string path = dir + "/corpus.jsonl";
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);
    for (const Conversation& c : conversations)
        out << ordered_json{{"prompt", c.prompt}, {"score_from", c.score_from}}.dump() << "\n";
    return path;
}

}  // namespace

R97_MODEL_TEST(device_loss_matches_kld) {
    const std::string container = model_container();
    Card card(int64_t(20) << 30);
    const std::string home = r97::default_radiance_home();
    R97_CHECK(!home.empty());

    const std::string fixture = (fs::path(__FILE__).parent_path().parent_path()
                                 / "fixtures/chat_32.jsonl")
                                    .string();
    const std::vector<Conversation> conversations = read_conversations(container, fixture);
    R97_CHECK(conversations.size() == 32);

    // The corpus, then radiance's own log-probabilities for it, on the same card and kernel tier.
    const std::string bin = radiance_bin(home);
    if (bin.empty())
        throw std::runtime_error("no radiance binary: build the `radiance` target beside the "
                                 "plugin home, or set R97_RADIANCE_BIN");
    const std::string dir = temp_dir("kld");
    const std::string corpus = write_corpus(conversations, dir);
    const std::string record = dir + "/record";
    int64_t longest = 1;
    for (const Conversation& c : conversations)
        longest = std::max<int64_t>(longest, int64_t(c.seq.ids.size()));
    const std::string cmd =
        quote(bin) + " --model " + quote(container) + " --radiance-home " + quote(home)
        + " --kernels libr4d --kld-corpus " + quote(corpus) + " --kld-record " + quote(record)
        + " --kv-cache-dtype bf16 --max-model-len 4096 --max-num-batched-tokens 4096"
          " --max-num-seqs 32 --num-speculative-tokens 0 --mm-max-patches 0";
    std::string log;
    const int status = run_command(cmd, &log);
    if (status != 0)
        throw std::runtime_error("radiance --kld-record failed (" + std::to_string(status)
                                 + "):\n" + cmd + "\n" + log);
    const KldRecord reference = read_kld(record);
    R97_CHECK(reference.docs.size() == 32);

    const double want = kld_loss_over_scored(reference, conversations);

    // The device loss over ChatData's own sequences.
    r97::Batch seqs;
    for (const Conversation& c : conversations) seqs.push_back(c.seq);
    r97::LossRunConfig run = forward_on(container, home, {"libr4d"}, 4096, longest, 32, 256);
    run.sample_vram = true;
    r97::LossRunResult got;
    std::string err;
    if (!r97::run_loss(run, seqs, &got, &err))
        throw std::runtime_error("the device loss failed: " + err);
    R97_CHECK(got.sequences == 32);
    R97_CHECK(got.scored == [&] {
        int64_t n = 0;
        for (const Conversation& c : conversations)
            for (const int32_t s : c.seq.score) n += s ? 1 : 0;
        return n;
    }());

    std::fprintf(stderr,
                 "32 conversations: device loss %.9g, radiance kld %.9g, relative %.3g, "
                 "%lld scored, %.2f s, %.1f tokens/s",
                 got.loss, want, std::fabs(got.loss - want) / (want != 0.0 ? std::fabs(want) : 1.0),
                 (long long)got.scored, got.seconds,
                 got.seconds > 0 ? double(got.tokens) / got.seconds : 0.0);
    if (got.vram_sampled)
        std::fprintf(stderr, ", %.0f MiB used after the step of %.0f MiB",
                     double(got.vram_used) / (1024.0 * 1024.0),
                     double(got.vram_total) / (1024.0 * 1024.0));
    std::fprintf(stderr, "\n");

    close_enough(got.loss, want, 1e-3, "the 32-conversation device loss");
}

R97_MODEL_TEST(device_forward_4096) {
    const std::string container = model_container();
    Card card(int64_t(24) << 30);
    const std::string home = r97::default_radiance_home();
    R97_CHECK(!home.empty());

    // One sequence of exactly 4096 tokens: a real conversation's ids, cycled after their first
    // token (the template's opening control token stays in front), every position but the first
    // scored. 4095 scored positions is the head's whole-vocabulary work at the length bound.
    const std::string fixture = (fs::path(__FILE__).parent_path().parent_path()
                                 / "fixtures/chat_32.jsonl")
                                    .string();
    const std::vector<Conversation> conversations = read_conversations(container, fixture);
    const r97::TokenSeq& first = conversations.front().seq;
    R97_CHECK(first.ids.size() > 1);
    r97::TokenSeq seq;
    seq.ids.push_back(first.ids.front());
    for (int64_t i = 1; i < 4096; ++i)
        seq.ids.push_back(first.ids[size_t(1 + (i - 1) % (first.ids.size() - 1))]);
    seq.score.push_back(0);
    for (int64_t i = 1; i < 4096; ++i) seq.score.push_back(1);

    const r97::LossRunConfig base =
        forward_on(container, home, {"libr4d"}, 4096, 4096, 1, 256);
    const std::vector<r97::TokenSeq> one = {seq};

    // The loaded footprint: the same ForwardConfig (same KV pool, same plugin home) over a
    // one-token sequence; the engine that holds the step is alive when run_loss samples it.
    r97::LossRunConfig warm = base;
    warm.token_budget = 1;
    warm.sample_vram = true;
    r97::LossRunResult loaded;
    std::string err;
    r97::TokenSeq token;
    token.ids = {seq.ids.front()};
    token.score = {0};
    if (!r97::run_loss(warm, std::vector<r97::TokenSeq>{token}, &loaded, &err))
        throw std::runtime_error("the one-token load failed: " + err);

    r97::VramWatch watch;
    r97::LossRunResult got;
    if (!r97::run_loss(base, one, &got, &err))
        throw std::runtime_error("the 4096-token step failed: " + err);
    R97_CHECK(got.steps == 1);
    R97_CHECK(got.sequences == 1);
    R97_CHECK(got.tokens == 4096);
    R97_CHECK(got.scored == 4095);
    R97_CHECK(std::isfinite(got.loss) && got.loss > 0.0);
    R97_CHECK(got.seconds > 0.0);
    R97_CHECK(watch.samples() > 0);

    std::fprintf(stderr,
                 "4096 tokens: loss %.6f over %lld scored, %.2f s (%.1f tokens/s), "
                 "vram after loading %.0f MiB, peak %.0f MiB of %.0f MiB\n",
                 got.loss, (long long)got.scored, got.seconds,
                 double(got.tokens) / got.seconds,
                 double(loaded.vram_used) / (1024.0 * 1024.0),
                 double(watch.peak_used()) / (1024.0 * 1024.0),
                 double(watch.total()) / (1024.0 * 1024.0));
}
