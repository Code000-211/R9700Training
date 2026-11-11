// tests/host/loss_cli.cpp -- `r97train loss` and radiance's KL record measure the same held-out
// loss on tests/fixtures/chat_small.jsonl. The corpus radiance scores is the chat file's scored
// runs, each as the prefix of its sequence up to the run: the docs' rows are then exactly the
// rows the trainer scores. The second case runs the same comparison through an adapter directory
// the fixture ships, so `--lora` is checked against radiance loading the same directory itself.
#include "../support/fixture.h"

// radiance's tests/rad_test.h, which the fixture header brings in, defines CHECK for its own
// TEST() registry; this file checks through R97_CHECK, so drop the alias before test.h.
#undef CHECK
#include "../support/test.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

std::string chat_path() {
    const fs::path here = fs::absolute(fs::path(__FILE__));
    return (here.parent_path().parent_path() / "fixtures" / "chat_small.jsonl").string();
}

// The fixture's LoRA adapter: PEFT's two files, for the fixture's own checkpoint.
std::string adapter_path() {
    const fs::path here = fs::absolute(fs::path(__FILE__));
    return (here.parent_path().parent_path() / "fixtures" / "oracle" / "adapter").string();
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

// The record's own NLL column, read the way tools/kld_loss.py reads it: kld.json names the format
// and the row count, and rows.f32 holds `rows` rows of little-endian f32, column 0 each row's
// next-token NLL. The host tier never runs Python, so the check reads the record itself.
struct RecordLoss {
    double loss = 0.0;  // the mean over the record's rows
    int64_t rows = 0;
};

RecordLoss record_loss(const fs::path& dir) {
    std::ifstream manifest(dir / "kld.json");
    if (!manifest) throw std::runtime_error("cannot read " + (dir / "kld.json").string());
    nlohmann::json m;
    manifest >> m;
    const std::string format = m.at("format").get<std::string>();
    const int64_t width = format == "radiance-kld-1"     ? 2
                          : format == "radiance-kld-top-1" ? 3
                                                          : 0;
    if (!width) throw std::runtime_error("the KL record's format is '" + format + "'");
    const int64_t rows = m.at("rows").get<int64_t>();
    if (rows <= 0) throw std::runtime_error("the KL record has no rows");

    std::ifstream in(dir / "rows.f32", std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + (dir / "rows.f32").string());
    in.seekg(0, std::ios::end);
    const std::streamoff bytes = in.tellg();
    in.seekg(0, std::ios::beg);
    if (bytes != rows * width * 4)
        throw std::runtime_error("rows.f32 holds " + std::to_string(bytes) + " bytes, not " +
                                 std::to_string(rows * width * 4));
    std::vector<char> raw(static_cast<size_t>(bytes));
    if (!in.read(raw.data(), bytes))
        throw std::runtime_error("cannot read all of " + (dir / "rows.f32").string());

    double sum = 0.0;
    for (int64_t r = 0; r < rows; ++r) {
        const auto* b = reinterpret_cast<const unsigned char*>(raw.data() + r * width * 4);
        const uint32_t bits = uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 |
                              uint32_t(b[3]) << 24;
        float nll = 0.0f;
        std::memcpy(&nll, &bits, sizeof nll);
        if (!std::isfinite(nll))
            throw std::runtime_error("row " + std::to_string(r) + "'s NLL is not finite");
        sum += nll;
    }
    return {sum / double(rows), rows};
}

std::vector<int32_t> read_tokens(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path.string());
    std::vector<int32_t> ids;
    int32_t id = 0;
    while (in.read(reinterpret_cast<char*>(&id), sizeof id)) ids.push_back(id);
    return ids;
}

// The chat file's scored runs, as radiance's own corpus: one document per run, prompt through the
// run's last target, score_from the row that predicts the run's first target.
struct Corpus {
    fs::path dir;                 // the case's working directory
    fs::path file;                // the corpus JSONL
    int64_t rows = 0;             // scored positions
    size_t max_doc_len = 0;       // tokens of the longest document's prompt
    size_t docs = 0;
    std::vector<int32_t> tokens;  // the chat file's tokens through the last scored run
};

Corpus write_corpus(const r97::Fixture& f, const std::string& name) {
    r97::ChatData chat;
    if (!chat.open(f.container)) throw std::runtime_error("chat-data: " + chat.error());
    r97::Batch batch;
    r97::ChatDataStats stats;
    if (!chat.read_file(chat_path(), r97::ChatData::kDefaultMaxTokens, &batch, &stats))
        throw std::runtime_error("chat-data: " + chat.error());
    if (batch.empty()) throw std::runtime_error("the chat fixture tokenised to nothing");

    Corpus c;
    c.dir = fs::path(radtest::tmpdir()) / name;
    std::error_code ec;
    fs::remove_all(c.dir, ec);
    fs::create_directories(c.dir, ec);
    c.file = c.dir / "corpus.jsonl";
    std::ofstream out(c.file);
    if (!out) throw std::runtime_error("cannot write " + c.file.string());

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
                                  {"source", "chat_small:" + std::to_string(c.docs)}};
            out << doc.dump() << "\n";

            c.tokens.insert(c.tokens.end(), seq.ids.begin(), seq.ids.begin() + stop);
            c.rows += int64_t(stop - start);
            c.max_doc_len = std::max(c.max_doc_len, stop);
            ++c.docs;
        }
    }
    if (c.docs == 0) throw std::runtime_error("the chat fixture has no scored runs");
    return c;
}

// The fixture and the corpus, built before main.
//
// The host runner calls an R97_TEST from an exit handler, after the binary's namespace-scope
// objects are destroyed, and by then radiance's own chat-template machinery is gone: ChatData
// cannot open a container any more. So the corpus is built here, at static-initialiser time,
// where ChatData does work (the chat-data checks use it there). The engine at the other end is
// the reverse -- it allocates through radiance's device ledger, whose constructor runs later in
// the binary's static initialisation -- so an engine starts only from the exit-time test. One
// test, two moments; nothing about the comparison changes.
//
// The prepared state is leaked on purpose: it has to outlive the exit-time test, and a static of
// its own would be destroyed before the runner runs.
struct Prepared {
    bool ok = false;
    std::string error;
    r97::Fixture fixture;
    Corpus corpus;
};

Prepared& prepared() {
    static Prepared* const p = [] {
        Prepared* q = new Prepared();
        try {
            q->fixture = r97::detail::make();
            q->corpus = write_corpus(q->fixture, "loss_cli");
            q->ok = true;
        } catch (const std::exception& e) {
            q->error = e.what();
        }
        return q;
    }();
    return *p;
}

[[maybe_unused]] const bool prepared_before_main = (prepared(), true);

// Record the corpus's loss with radiance itself -- through `lora`, when one is named -- run
// `r97train loss` over the same chat file, and check they agree.
void check_against_record(const Prepared& p, const std::string& name, const std::string& lora) {
    const r97::Fixture& f = p.fixture;
    const Corpus&       c = p.corpus;
    const fs::path record = c.dir / ("kld_" + name);

    rad::Config cfg;
    cfg.model            = f.container;
    cfg.radiance_home    = r97::detail::radiance_home();
    cfg.kernel_hierarchy = {"libavx", "libref"};
    cfg.n_spec           = 0;
    cfg.max_tok          = int64_t(std::max<size_t>(512, c.max_doc_len));
    cfg.max_seqs         = int32_t(std::max<size_t>(8, c.docs));
    cfg.max_ctx          = int64_t(std::max<size_t>(512, c.max_doc_len));
    cfg.kld_corpus       = c.file.string();
    cfg.kld_record       = record.string();
    if (!lora.empty()) {
        // radiance's own route: the adapter is one of the startup ones, and every document runs
        // through it (--lora NAME=DIR and --kld-lora NAME on its command line).
        cfg.lora.adapters.emplace_back("forward", lora);
        cfg.kld_lora = "forward";
    }

    rad::Engine engine(std::move(cfg));
    int s = engine.start();
    if (s < 0)
        throw std::runtime_error("kld record: the engine could not start: " +
                                 std::string(rad_strerror(s)));
    auto watch = std::make_unique<rad::StallWatch>("loss_cli kld", 60000);
    s = engine.serve(std::move(watch));
    if (s < 0) throw std::runtime_error("kld record: " + std::string(rad_strerror(s)));

    // The record's token stream is what the test put in, so both sides really score the chat file.
    const std::vector<int32_t> got_tokens = read_tokens(record / "tokens.i32");
    if (got_tokens != c.tokens)
        throw std::runtime_error("the KL record tokenised the corpus differently from chat-data");

    const RecordLoss kld = record_loss(record);

    const std::string cli_path = (fs::path(r97::detail::build_dir()) / "r97train").string();
    if (!fs::exists(cli_path))
        throw std::runtime_error("r97train is not beside the test binary at " + cli_path);
    const std::string lora_flag = lora.empty() ? "" : " --lora " + quote(lora);
    const Run cli = run(quote(cli_path) + " loss --model " + quote(f.container) + " --data " +
                        quote(chat_path()) + lora_flag);
    if (cli.status != 0) throw std::runtime_error("r97train loss failed:\n" + cli.output);

    const double cli_loss = field(cli.output, "loss");
    if (kld.rows != c.rows)
        throw std::runtime_error("the KL record holds " + std::to_string(kld.rows) +
                                 " rows, the chat file has " + std::to_string(c.rows) +
                                 " scored tokens");
    if (int64_t(field(cli.output, "scored")) != c.rows)
        throw std::runtime_error("r97train loss scored " + std::to_string(field(cli.output, "scored")) +
                                 " tokens, the chat file has " + std::to_string(c.rows));
    check_close(cli_loss, kld.loss, "r97train loss against the KL record" +
                                        (lora.empty() ? std::string() : " through --lora"));
}

// Two cases: the base model, and the fixture's adapter through --lora.
R97_TEST(loss_cli_matches_radiance_kld) {
    const Prepared& p = prepared();
    if (!p.ok) throw std::runtime_error(p.error);
    check_against_record(p, "base", "");
}

R97_TEST(loss_cli_lora_matches_radiance_kld) {
    const Prepared& p = prepared();
    if (!p.ok) throw std::runtime_error(p.error);
    check_against_record(p, "lora", adapter_path());
}

// A directory radiance cannot load must end the run with a message and a non-zero status, not take
// the process down: the driver reports a failed engine start by throwing, and loss_main must catch
// it. Radiance logs the reason the loader refused the directory.
R97_TEST(loss_cli_reports_an_unloadable_lora) {
    const Prepared& p = prepared();
    if (!p.ok) throw std::runtime_error(p.error);
    const r97::Fixture& f = p.fixture;
    const std::string cli_path = (fs::path(r97::detail::build_dir()) / "r97train").string();
    if (!fs::exists(cli_path))
        throw std::runtime_error("r97train is not beside the test binary at " + cli_path);

    const fs::path empty = p.corpus.dir / "no_adapter";
    std::error_code ec;
    fs::remove_all(empty, ec);
    fs::create_directories(empty, ec);

    const Run cli = run(quote(cli_path) + " loss --model " + quote(f.container) + " --data " +
                        quote(chat_path()) + " --lora " + quote(empty.string()));
    if (cli.status != 3)
        throw std::runtime_error("an unloadable adapter directory exited " +
                                 std::to_string(cli.status) + ", not 3:\n" + cli.output);
    if (cli.output.find("adapter_config.json") == std::string::npos)
        throw std::runtime_error("the run did not report radiance's loader error:\n" + cli.output);
}

}  // namespace

#endif  // R97_FORWARD
