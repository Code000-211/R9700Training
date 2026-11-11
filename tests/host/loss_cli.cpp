// tests/host/loss_cli.cpp -- `r97train loss` and radiance's KL record measure the same held-out
// loss on tests/fixtures/chat_small.jsonl. The corpus radiance scores is the chat file's scored
// runs, each as the prefix of its sequence up to the run: the docs' rows are then exactly the
// rows the trainer scores.
#include "../support/fixture.h"
#include "../support/test.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <vector>

#include "r97/chatdata.h"
#include "r97/forward.h"

#if R97_FORWARD

namespace {

namespace fs = std::filesystem;

struct Run {
    int status = -1;
    std::string output;
};

Run run(const std::string& command) {
    Run r;
    FILE* p = popen((command + " 2>&1").c_str(), "r");
    if (!p) throw std::runtime_error("cannot run: " + command);
    char buf[4096];
    while (size_t n = fread(buf, 1, sizeof buf, p)) r.output.append(buf, n);
    const int status = pclose(p);
    if (status >= 0 && WIFEXITED(status)) r.status = WEXITSTATUS(status);
    return r;
}

std::string quote(const std::string& s) { return r97::detail::shell_quote(s); }

std::string script_path() {
    const fs::path here = fs::absolute(fs::path(__FILE__));
    return (here.parent_path().parent_path().parent_path() / "tools" / "kld_loss.py").string();
}

std::string chat_path() {
    const fs::path here = fs::absolute(fs::path(__FILE__));
    return (here.parent_path().parent_path() / "fixtures" / "chat_small.jsonl").string();
}

double field(const std::string& output, const std::string& name) {
    const std::string key = name + ":";
    const size_t at = output.rfind(key);
    if (at == std::string::npos) throw std::runtime_error("no " + key + " in: " + output);
    return std::stod(output.substr(at + key.size()));
}

void check_close(double got, double want, const std::string& what) {
    const double scale = std::max(std::fabs(want), 1e-12);
    const double rel = std::fabs(got - want) / scale;
    if (rel > 1e-4)
        throw std::runtime_error(what + ": " + std::to_string(got) + " vs " + std::to_string(want) +
                                 " (relative " + std::to_string(rel) + ")");
}

std::vector<int32_t> read_tokens(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path.string());
    std::vector<int32_t> ids;
    int32_t id = 0;
    while (in.read(reinterpret_cast<char*>(&id), sizeof id)) ids.push_back(id);
    return ids;
}

}  // namespace

R97_TEST(loss_cli_matches_radiance_kld) {
    const r97::Fixture& f = r97::fixture();

    r97::ChatData chat;
    if (!chat.open(f.container)) throw std::runtime_error("chat-data: " + chat.error());
    r97::Batch batch;
    r97::ChatDataStats stats;
    if (!chat.read_file(chat_path(), r97::ChatData::kDefaultMaxTokens, &batch, &stats))
        throw std::runtime_error("chat-data: " + chat.error());
    if (batch.empty()) throw std::runtime_error("the chat fixture tokenised to nothing");

    const fs::path work = fs::path(radtest::tmpdir()) / "loss_cli";
    std::error_code ec;
    fs::remove_all(work, ec);
    fs::create_directories(work, ec);
    const fs::path corpus = work / "corpus.jsonl";
    const fs::path record = work / "kld";
    std::ofstream out(corpus);
    if (!out) throw std::runtime_error("cannot write " + corpus.string());

    int64_t expected_rows = 0;
    size_t max_doc_len = 0;
    size_t docs = 0;
    std::vector<int32_t> expected_tokens;

    for (const r97::TokenSeq& seq : batch) {
        if (seq.score.empty() || seq.score[0] != 0)
            throw std::runtime_error("the chat fixture scored its first token");
        for (size_t i = 1; i < seq.score.size();) {
            if (!seq.score[i]) {
                ++i;
                continue;
            }
            const size_t start = i;  // first target token of this run
            while (i < seq.score.size() && seq.score[i]) ++i;
            const size_t stop = i;  // one past the last target token

            // The doc is the sequence's prefix through the run's last target; score_from is the
            // row that predicts the run's first target.
            std::string text;
            for (size_t j = 0; j < stop; ++j)
                text += chat.token_text(seq.ids[j], j == 0, j + 1 == seq.ids.size());
            nlohmann::json doc = {{"prompt", text},
                                  {"score_from", int64_t(start - 1)},
                                  {"source", "chat_small:" + std::to_string(docs)}};
            out << doc.dump() << "\n";

            expected_tokens.insert(expected_tokens.end(), seq.ids.begin(), seq.ids.begin() + stop);
            expected_rows += int64_t(stop - start);
            max_doc_len = std::max(max_doc_len, stop);
            ++docs;
        }
    }
    out.close();
    if (docs == 0) throw std::runtime_error("the chat fixture has no scored runs");

    rad::Config cfg;
    cfg.model            = f.container;
    cfg.radiance_home    = r97::detail::radiance_home();
    cfg.kernel_hierarchy = {"libavx", "libref"};
    cfg.n_spec           = 0;
    cfg.max_tok          = int64_t(std::max<size_t>(512, max_doc_len));
    cfg.max_seqs         = int32_t(std::max<size_t>(8, docs));
    cfg.max_ctx          = int64_t(std::max<size_t>(512, max_doc_len));
    cfg.kld_corpus       = corpus.string();
    cfg.kld_record       = record.string();

    rad::Engine engine(std::move(cfg));
    int s = engine.start();
    if (s < 0)
        throw std::runtime_error("kld record: the engine could not start: " +
                                 std::string(rad_strerror(s)));
    auto watch = std::make_unique<rad::StallWatch>("loss_cli kld", 60000);
    s = engine.serve(std::move(watch));
    if (s < 0)
        throw std::runtime_error("kld record: " + std::string(rad_strerror(s)));

    // The record's token stream is what the test put in, so both sides really score the chat file.
    const std::vector<int32_t> got_tokens = read_tokens(record / "tokens.i32");
    if (got_tokens != expected_tokens)
        throw std::runtime_error("the KL record tokenised the corpus differently from chat-data");

    const std::string python = std::getenv("PYTHON") ? std::getenv("PYTHON") : "python3";
    const Run kld = run(quote(python) + " " + quote(script_path()) + " " + quote(record.string()));
    if (kld.status != 0)
        throw std::runtime_error("tools/kld_loss.py failed:\n" + kld.output);

    const std::string cli_path = (fs::path(r97::detail::build_dir()) / "r97train").string();
    if (!fs::exists(cli_path))
        throw std::runtime_error("r97train is not beside the test binary at " + cli_path);
    const Run cli = run(quote(cli_path) + " loss --model " + quote(f.container) + " --data " +
                        quote(chat_path()));
    if (cli.status != 0) throw std::runtime_error("r97train loss failed:\n" + cli.output);

    const double kld_loss = field(kld.output, "loss");
    const double cli_loss = field(cli.output, "loss");
    if (int64_t(field(kld.output, "scored")) != expected_rows)
        throw std::runtime_error("kld_loss.py scored " + std::to_string(field(kld.output, "scored")) +
                                 " rows, the chat file has " + std::to_string(expected_rows));
    if (int64_t(field(cli.output, "scored")) != expected_rows)
        throw std::runtime_error("r97train loss scored " + std::to_string(field(cli.output, "scored")) +
                                 " tokens, the chat file has " + std::to_string(expected_rows));
    check_close(cli_loss, kld_loss, "r97train loss against the KL record");
}

#endif  // R97_FORWARD
