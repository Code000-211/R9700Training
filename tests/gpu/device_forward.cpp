// tests/gpu/device_forward.cpp -- the training forward on the card, checked three ways, all on the
// real container (the qwen3.8-27b-iq5r-s-v4 one on disk, resolved through support/model_dir.h):
//
//   device_forward_device_tier   the container on the device kernel tier (libr4d, the host tier as
//                                the per-op fallback): the forward loss over the three short
//                                conversations is finite, every scored token is scored, and the
//                                engine's own kernel-selection report names libr4d -- so the
//                                device tier is the one running, not a silent host fallback. (A
//                                device-vs-host comparison is not runnable on this host: a pure
//                                host kernel set fails to declare this container (rad_arch_declare
//                                returned invalid argument), the graph's op forms include several
//                                that no host library implements, and the device tier refuses the
//                                tiny fixture's geometry -- head_dim 32, head_k 16 and n 64 have no
//                                libr4d kernel -- so no model runs on both tiers here. The device
//                                forward's numbers are checked against radiance's own record in
//                                device_forward_model_loss.)
//   device_forward_model_loss    the container's 32 single-turn conversations, the device tier's
//                                forward loss against radiance's own KL record of the same
//                                corpus (both on the device tier), with the token identity the
//                                comparison assumes;
//   device_forward_seq4096 /     one sequence of that many tokens on the container (4096 is the
//   device_forward_seq2048       ticket's shape, 2048 the shorter piece that must keep working):
//                                the loss over every scored token, throughput, and the card's VRAM
//                                at the step's peak. A step is one prefill piece, so the driver
//                                turns radiance's checkpoint interval off -- its default 2048 would
//                                otherwise cap the piece -- and the piece is the budget the caller
//                                sets, not a fixed 2048.
//
// The child/parent split. The engine picks the card only through ROCR_VISIBLE_DEVICES
// (engine_bringup's usable_devices), which the ROCR runtime reads once, at init, and the shared
// r97_gpu_tests binary runs every test in one process where the other gpu tests init ROCR before
// this one does. So no engine work runs in this process at all: the parent forks, and the child
// re-execs this same binary with R97_TEST_ONLY (only this test), R97_DF_OUT (where to write its
// results file) and ROCR_VISIBLE_DEVICES already set -- a fresh process whose ROCR points at the
// card from the first device call. The child does the real work and writes a JSON file the parent
// reads and checks. Each child waits for the card to report its VRAM free (wait_for_card_free)
// before the guard checks it: a previous child's VRAM is returned to the driver asynchronously.
#include "support/gpu_guard.h"
#include "support/model_case.h"
#include "support/model_dir.h"
#include "support/test.h"

#include "r97/chatdata.h"
#include "r97/forward.h"

#if R97_FORWARD

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <utility>
#include <vector>

#include "engine.h"           // radiance's core/; Config, Engine, rad_strerror
#include "util/stallwatch.h"  // radiance's core/; the watch serve() takes

#include "r97/loss.h"
#include "r97/params.h"

namespace {

namespace fs = std::filesystem;

using r97::Batch;
using r97::TokenSeq;

// ------------------------------------------------------------------ the small helpers

const std::string& self_exe() {
    static const std::string exe = [] {
        std::error_code ec;
        const fs::path p = fs::read_symlink("/proc/self/exe", ec);
        if (ec) throw std::runtime_error("device_forward: cannot read /proc/self/exe");
        return p.string();
    }();
    return exe;
}

// The build directory this test binary sits in: radiance-build/ (the plugins, the home) is beside
// it, and so is the converter.
std::string build_dir() {
    std::error_code ec;
    const fs::path p = fs::read_symlink("/proc/self/exe", ec);
    return ec ? std::string() : p.parent_path().string();
}

// The plugin home: RADIANCE_HOME, else the build's radiance_home.
std::string radiance_home() {
    if (const char* e = std::getenv("RADIANCE_HOME"))
        if (*e) return e;
    const std::string h = build_dir() + "/radiance-build/radiance_home";
    std::error_code ec;
    return fs::exists(h + "/architectures", ec) ? h : std::string();
}

std::string quote(const std::string& s) {
    std::string q = "'";
    for (char c : s)
        if (c == '\'') q += "'\\''";
        else q += c;
    return q + "'";
}

// A fresh directory under TMPDIR (else /tmp) for one case's results. mkdtemp, so two copies of the
// test never collide. The test tier does not carry radiance's tests/ path, so this is the
// equivalent of radtest::tmp_path, which the gpu target cannot use.
std::string temp_dir(const std::string& name) {
    const char* base = std::getenv("TMPDIR");
    if (!base || !*base) base = "/tmp";
    std::string path = std::string(base) + "/r97-df-" + name + "-XXXXXX";
    char* d = ::mkdtemp(path.data());
    if (!d)
        throw std::runtime_error("device_forward: cannot make a temp directory under " +
                                 std::string(base) + ": " + std::strerror(errno));
    return d;
}

// Fork, and have the child re-exec this binary running only `test_name`, with R97_DF_ROLE=<role>
// naming which of this test's children it is and, for a device child, ROCR_VISIBLE_DEVICES
// pointing the ROCR runtime at the card before its first init. R97_TEST_ONLY selects this test so
// the re-exec'd copy runs it alone, and -- seeing R97_DF_ROLE -- does the child's work instead of
// forking again. Wait, and end the run as skipped (exit 77) when the child skips (the container
// absent). The parent never opens the card or inits an engine: its ROCR was already init'd by the
// other gpu tests, so only a fresh child process can be pointed at the card.
void run_in_child(const std::string& dir, const char* test_name, const std::string& role,
                  bool device, int64_t tokens = 0) {
    const char* dev = std::getenv("R97_DEVICE");
    const std::string device_num = dev && *dev ? dev : "0";

    const pid_t pid = ::fork();
    if (pid < 0)
        throw std::runtime_error(std::string("device_forward: fork: ") + std::strerror(errno));
    if (pid == 0) {
        // The child: point ROCR at the card (a device child) or at no card (a host child) before
        // any ROCR init, mark this re-exec'd copy as the child that does the work, and hand it the
        // results directory. The re-exec'd process is fresh, so ROCR reads these for the first time.
        // A host child hides the cards from both runtimes -- HIP_VISIBLE_DEVICES empty and
        // ROCR_VISIBLE_DEVICES at no card -- so the arch declares host bands. A visible card makes
        // it declare device bands, which the host tier then cannot serve.
        const std::string hip_visible  = device ? device_num : "";
        const std::string rocr_visible = device ? device_num : "99";
        if (::setenv("HIP_VISIBLE_DEVICES", hip_visible.c_str(), 1) != 0 ||
            ::setenv("ROCR_VISIBLE_DEVICES", rocr_visible.c_str(), 1) != 0 ||
            ::setenv("R97_TEST_ONLY", test_name, 1) != 0 ||
            ::setenv("R97_DF_OUT", dir.c_str(), 1) != 0 ||
            ::setenv("R97_DF_ROLE", role.c_str(), 1) != 0)
            std::_Exit(1);
        // The one-prompt length this child's step uses, when the caller asks for one.
        if (tokens > 0 &&
            ::setenv("R97_DF_TOKENS", std::to_string(tokens).c_str(), 1) != 0)
            std::_Exit(1);
        std::string exe = self_exe();
        char* const argv[2] = {exe.data(), nullptr};
        ::execv(exe.c_str(), argv);
        std::_Exit(1);  // execv failed
    }

    int status = 0;
    if (::waitpid(pid, &status, 0) != pid)
        throw std::runtime_error(std::string("device_forward: ") + test_name + ": waitpid failed: " +
                                 std::strerror(errno));
    if (!WIFEXITED(status))
        throw std::runtime_error(std::string("device_forward: ") + test_name +
                                 "'s child was killed by a signal; the card may be held by another run");
    const int code = WEXITSTATUS(status);
    if (code != 0) {
        // 77 (the container absent) re-skips the way ctest records it; anything else is a failure.
        if (code == r97test::kSkip)
            r97test::skip(std::string("the model container is absent (see the child run of ") +
                          test_name + " above)");
        throw std::runtime_error(std::string("device_forward: ") + test_name +
                                 "'s " + role + " child exited " + std::to_string(code) +
                                 ": its stderr above says why; the card is R97_DEVICE, and the "
                                 "guard names the card's holders when the VRAM is short");
    }
}

// The results directory this (child) process was told to write into.
std::string out_dir() {
    const char* d = std::getenv("R97_DF_OUT");
    if (!d || !*d)
        throw std::runtime_error("device_forward: R97_DF_OUT is unset; the results have nowhere to go");
    return d;
}

// ------------------------------------------------------------------ the forward, by tier

// The chat file the forward-vs-host case scores: the three short conversations the host tier runs
// (tests/host/loss_cli.cpp runs over the same file).
fs::path chat_small() {
    const fs::path here = fs::absolute(fs::path(__FILE__));
    return here.parent_path().parent_path() / "fixtures" / "chat_small.jsonl";
}

// The chat file's sequences: the batch both tiers score.
Batch chat_batch(const std::string& container) {
    r97::ChatData chat;
    if (!chat.open(container))
        throw std::runtime_error("device_forward: chat-data: " + chat.error());
    Batch batch;
    r97::ChatDataStats stats;
    if (!chat.read_file(chat_small().string(), r97::ChatData::kDefaultMaxTokens, &batch, &stats))
        throw std::runtime_error("device_forward: chat-data: " + chat.error());
    if (batch.empty()) throw std::runtime_error("device_forward: the chat fixture tokenised to nothing");
    return batch;
}

r97::ForwardConfig config_for(const std::string& container,
                              const std::vector<std::string>& kernels, int64_t longest) {
    r97::ForwardConfig cfg;
    cfg.model         = container;
    cfg.radiance_home = radiance_home();
    cfg.kernels       = kernels;
    cfg.max_tok       = std::max<int64_t>(512, longest);
    cfg.max_ctx       = std::max<int64_t>(512, longest);
    cfg.max_seqs      = 8;
    cfg.window_rows   = 256;
    return cfg;
}

int64_t longest_of(const Batch& batch) {
    int64_t m = 0;
    for (const TokenSeq& s : batch) m = std::max(m, int64_t(s.ids.size()));
    return m;
}

// The driver's loss and scored count over the batch, chunked by max_tok like tools/loss.cpp.
r97::StepResult drive(const r97::ForwardConfig& cfg, const Batch& batch) {
    std::string err;
    std::unique_ptr<r97::ForwardDriver> driver = r97::ForwardDriver::open(cfg, &err);
    if (!driver) throw std::runtime_error(err);
    r97::Model model;
    model.objective = r97::CeObjective;
    const r97::StepFn step = driver->step(model);

    double loss = 0.0;
    int64_t scored = 0;
    r97::Batch chunk;
    int64_t chunk_tokens = 0;
    for (const TokenSeq& s : batch) {
        const int64_t tokens = int64_t(s.ids.size());
        if (!chunk.empty() &&
            (chunk.size() >= size_t(cfg.max_seqs) || chunk_tokens + tokens > cfg.max_tok)) {
            const r97::StepResult r = step(chunk, r97::ParamSet(), nullptr);
            loss += r.loss;
            scored += r.scored;
            chunk.clear();
            chunk_tokens = 0;
        }
        chunk.push_back(s);
        chunk_tokens += tokens;
    }
    if (!chunk.empty()) {
        const r97::StepResult r = step(chunk, r97::ParamSet(), nullptr);
        loss += r.loss;
        scored += r.scored;
    }
    return {loss, scored};
}

// Stderr to `file` while the child runs its engine, and back again afterwards. Radiance logs its
// startup report -- the kernel tier among it -- to stderr.
class StderrTo {
public:
    explicit StderrTo(const std::string& file) : file_(file) {
        saved_ = ::dup(2);
        if (saved_ < 0)
            throw std::runtime_error(std::string("device_forward: dup: ") + std::strerror(errno));
        fd_ = ::open(file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd_ < 0) {
            ::close(saved_);
            saved_ = -1;
            throw std::runtime_error("device_forward: cannot open the scratch log " + file + ": " +
                                     std::strerror(errno));
        }
        if (::dup2(fd_, 2) < 0)
            throw std::runtime_error(std::string("device_forward: dup2: ") + std::strerror(errno));
    }
    ~StderrTo() {
        ::fflush(nullptr);
        if (saved_ >= 0) {
            ::dup2(saved_, 2);
            ::close(saved_);
        }
        if (fd_ >= 0) ::close(fd_);
    }
    StderrTo(const StderrTo&) = delete;
    StderrTo& operator=(const StderrTo&) = delete;

private:
    int fd_ = -1;
    int saved_ = -1;
    std::string file_;
};

// The "kernel selection" row of the startup report, as captured: true when the row names libr4d
// and no row after it reports libr4d as left out of the load ("and so not loaded:").
bool device_tier_selected(const std::string& log) {
    const std::string marker = "kernel selection";
    size_t at = 0;
    while (at < log.size()) {
        const size_t end = log.find('\n', at);
        const std::string line =
            log.substr(at, end == std::string::npos ? std::string::npos : end - at);
        at = end == std::string::npos ? log.size() : end + 1;
        const size_t sel = line.find(marker);
        if (sel == std::string::npos) continue;
        if (line.find("libr4d", sel) == std::string::npos) return false;
        for (size_t rest = at; rest < log.size();) {
            const size_t rend = log.find('\n', rest);
            const std::string rl =
                log.substr(rest, rend == std::string::npos ? std::string::npos : rend - rest);
            rest = rend == std::string::npos ? log.size() : rend + 1;
            const size_t left = rl.find("and so not loaded: ");
            if (left != std::string::npos && rl.find("libr4d", left + 19) != std::string::npos)
                return false;
        }
        return true;
    }
    return false;
}

// ------------------------------------------------------------------ the card-free wait

// Wait until the selected card reports `need` bytes free. A previous test's child may have just
// exited and the driver can take a moment to return its VRAM to mem_info_vram_used; the guard
// would otherwise refuse on a stale reading of a card that is in fact free.
void wait_for_card_free(const r97test::GuardOptions& o) {
    const std::string sysfs = "/sys/class/drm";
    const std::string card = r97test::card_for_bdf(sysfs, o.bdf);
    for (int i = 0; i < 120; ++i) {  // up to ~60 s
        try {
            const r97test::CardVram v = r97test::read_card_vram(sysfs, card);
            if (v.total - v.used >= o.need_vram) return;
        } catch (const std::exception&) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

// ------------------------------------------------------------------ test 1: the device tier is the one running

// The device child: guard the card, drive the chat file through the device tier, and prove from
// the engine's own report that the device tier was the one selected (a silent host fallback would
// otherwise pass every later comparison by matching itself).
void test1_device() {
    r97test::GuardOptions o;
    const char* de = std::getenv("R97_DEVICE");
    o.device    = r97test::parse_device(de ? de : "0");
    o.need_vram = int64_t(20) << 30;  // the 27B iq5r container: ~15.1 GiB weights, plus KV and arena
    o.bdf       = r97test::hip_bdf(o.device);  // HIP is inited only now: ROCR_VISIBLE_DEVICES is set
    wait_for_card_free(o);
    r97test::DeviceGuard guard(o);

    const std::string container = r97test::container_or_skip();
    const Batch batch = chat_batch(container);

    const std::string dir = out_dir();
    const std::string log = dir + "/device.log";
    const r97::StepResult r = [&] {
        StderrTo to_log(log);
        return drive(config_for(container, {"libr4d", "libavx", "libref"}, longest_of(batch)), batch);
    }();

    std::string captured;
    {
        std::ifstream in(log);
        captured.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    const bool device_tier = device_tier_selected(captured);

    nlohmann::json j = {
        {"name", "device_forward_device_tier"},
        {"loss", r.loss},
        {"scored", r.scored},
        {"device_selected_libr4d", device_tier},
    };
    std::ofstream out(dir + "/device.json");
    if (!out) throw std::runtime_error("device_forward_device_tier: cannot write the device results");
    out << j.dump(2) << "\n";
}

// ------------------------------------------------------------------ test 2: device versus the KL record

// The 32 single-turn conversations, each one exchange: one scored run per conversation, so a
// conversation's scored span is exactly its document's scored rows.
std::vector<std::vector<r97::ChatMessage>> chat32() {
    std::vector<std::vector<r97::ChatMessage>> out;
    for (int i = 0; i < 32; ++i) {
        const int n = 3 * i + 11;
        out.push_back({{"user", "What is the square of " + std::to_string(n) + "?"},
                       {"assistant", "The square of " + std::to_string(n) + " is " +
                                        std::to_string(n * n) + "."}});
    }
    return out;
}

std::string write_chat32(const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("device_forward_model_loss: cannot write " + path);
    for (const auto& m : chat32()) {
        nlohmann::json msgs = nlohmann::json::array();
        for (const r97::ChatMessage& msg : m) {
            nlohmann::json mm;
            mm["role"]    = msg.role;
            mm["content"] = msg.content;
            msgs.push_back(mm);
        }
        nlohmann::json doc;
        doc["messages"] = msgs;
        out << doc.dump() << "\n";
    }
    return path;
}

// The KL record's NLL column, read the way tools/kld_loss.py reads it: kld.json names the format
// and the row count, rows.f32 holds `rows` rows of little-endian f32, column 0 each row's
// next-token NLL. (tests/host/loss_cli.cpp's record_loss, for the radiance-kld-1 width-2 form.)
struct RecordLoss {
    double loss = 0.0;
    int64_t rows = 0;
};

RecordLoss record_loss(const std::string& dir) {
    std::ifstream manifest(dir + "/kld.json");
    if (!manifest) throw std::runtime_error("device_forward_model_loss: cannot read " + dir + "/kld.json");
    nlohmann::json m;
    manifest >> m;
    const std::string format = m.at("format").get<std::string>();
    const int64_t width = format == "radiance-kld-1"       ? 2
                    : format == "radiance-kld-top-1" ? 3
                                                     : 0;
    if (!width) throw std::runtime_error("device_forward_model_loss: the KL record's format is '" + format + "'");
    const int64_t rows = m.at("rows").get<int64_t>();
    if (rows <= 0) throw std::runtime_error("device_forward_model_loss: the KL record has no rows");

    std::ifstream in(dir + "/rows.f32", std::ios::binary);
    if (!in) throw std::runtime_error("device_forward_model_loss: cannot read " + dir + "/rows.f32");
    in.seekg(0, std::ios::end);
    const std::streamoff bytes = in.tellg();
    in.seekg(0, std::ios::beg);
    if (bytes != rows * width * 4)
        throw std::runtime_error("device_forward_model_loss: rows.f32 holds " + std::to_string(bytes) +
                                 " bytes, not " + std::to_string(rows * width * 4));
    std::vector<char> raw(static_cast<size_t>(bytes));
    if (!in.read(raw.data(), bytes))
        throw std::runtime_error("device_forward_model_loss: cannot read all of " + dir + "/rows.f32");

    double sum = 0.0;
    for (int64_t r = 0; r < rows; ++r) {
        const auto* b = reinterpret_cast<const unsigned char*>(raw.data() + r * width * 4);
        const uint32_t bits = uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 |
                              uint32_t(b[3]) << 24;
        float nll = 0.0f;
        std::memcpy(&nll, &bits, sizeof nll);
        if (!std::isfinite(nll))
            throw std::runtime_error("device_forward_model_loss: row " + std::to_string(r) +
                                     "'s NLL is not finite");
        sum += nll;
    }
    return {sum / double(rows), rows};
}

std::vector<int32_t> read_tokens(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("device_forward_model_loss: cannot read " + path);
    std::vector<int32_t> ids;
    int32_t id = 0;
    while (in.read(reinterpret_cast<char*>(&id), sizeof id)) ids.push_back(id);
    return ids;
}

// The child: drive the 32 conversations through the device forward, record the same corpus through
// radiance's KL mode on the same device tier, and report the two losses and the record's token
// stream.
void test2_child() {
    r97test::GuardOptions o;
    const char* de = std::getenv("R97_DEVICE");
    o.device    = r97test::parse_device(de ? de : "0");
    o.need_vram = int64_t(20) << 30;
    o.bdf       = r97test::hip_bdf(o.device);
    wait_for_card_free(o);
    r97test::DeviceGuard guard(o);

    const std::string container = r97test::container_or_skip();
    const std::string home = radiance_home();
    if (home.empty())
        throw std::runtime_error("device_forward_model_loss: no radiance home beside the build");
    const std::string dir = out_dir();

    const std::string chat_file = dir + "/chat32.jsonl";
    write_chat32(chat_file);

    r97::ChatData chat;
    if (!chat.open(container))
        throw std::runtime_error("device_forward_model_loss: chat-data: " + chat.error());
    Batch batch;
    r97::ChatDataStats stats;
    if (!chat.read_file(chat_file, r97::ChatData::kDefaultMaxTokens, &batch, &stats))
        throw std::runtime_error("device_forward_model_loss: chat-data: " + chat.error());
    if (batch.size() != chat32().size())
        throw std::runtime_error("device_forward_model_loss: " + std::to_string(batch.size()) +
                                 " of the 32 conversations did not render");

    // ---- the device forward: the batch through the driver, chunked by max_tok like tools/loss.cpp
    const r97::StepResult fwd =
        drive(config_for(container, {"libr4d", "libavx", "libref"}, longest_of(batch)), batch);
    if (fwd.scored != stats.scored)
        throw std::runtime_error("device_forward_model_loss: the forward scored " +
                                 std::to_string(fwd.scored) + " tokens, the chat file has " +
                                 std::to_string(stats.scored));

    // ---- the KL record: the same 32 conversations as radiance's corpus, one document per scored
    // run, prompt through the run's last target, score_from the row that predicts its first.
    const std::string kld_dir = dir + "/kld";
    {
        std::error_code ec;
        fs::create_directories(kld_dir, ec);
        if (ec) throw std::runtime_error("device_forward_model_loss: cannot make " + kld_dir + ": " + ec.message());
    }
    const std::string corpus = kld_dir + "/corpus.jsonl";
    std::vector<int32_t> expect_tokens;
    int64_t rows = 0;
    size_t max_doc_len = 0;
    size_t docs = 0;
    {
        std::ofstream out(corpus);
        if (!out) throw std::runtime_error("device_forward_model_loss: cannot write " + corpus);
        size_t d = 0;
        for (const TokenSeq& seq : batch) {
            if (seq.score.empty() || seq.score[0] != 0)
                throw std::runtime_error("device_forward_model_loss: conversation " + std::to_string(d) +
                                         " scored its first token");
            // Every scored run of the conversation is one document: the prompt is the sequence's
            // prefix through the run's last target, score_from the row that predicts its first.
            // (tests/host/loss_cli.cpp's write_corpus, for the same chat file's shape.)
            for (size_t i = 1; i < seq.score.size();) {
                if (!seq.score[i]) {
                    ++i;
                    continue;
                }
                const size_t start = i;
                while (i < seq.score.size() && seq.score[i]) ++i;
                const size_t stop = i;

                std::string text;
                for (size_t j = 0; j < stop; ++j)
                    text += chat.token_text(seq.ids[j], j == 0, j + 1 == seq.ids.size());
                nlohmann::json doc = {{"prompt", text},
                                      {"score_from", int64_t(start - 1)},
                                      {"source", "chat32:" + std::to_string(d)}};
                out << doc.dump() << "\n";

                expect_tokens.insert(expect_tokens.end(), seq.ids.begin(), seq.ids.begin() + stop);
                rows += int64_t(stop - start);
                max_doc_len = std::max(max_doc_len, stop);
                ++docs;
            }
            ++d;
        }
    }
    if (rows == 0) throw std::runtime_error("device_forward_model_loss: the 32 conversations score no rows");

    rad::Config ecfg;
    ecfg.model            = container;
    ecfg.radiance_home    = home;
    ecfg.kernel_hierarchy = {"libr4d", "libavx", "libref"};
    ecfg.n_spec           = 0;
    ecfg.max_tok          = std::max<int64_t>(512, max_doc_len);
    ecfg.max_seqs         = int32_t(std::max<int64_t>(8, int64_t(docs)));
    ecfg.max_ctx          = std::max<int64_t>(512, max_doc_len);
    ecfg.kld_corpus       = corpus;
    ecfg.kld_record       = kld_dir;
    rad::Engine engine(std::move(ecfg));
    int s = engine.start();
    if (s < 0)
        throw std::runtime_error("device_forward_model_loss: kld record: the engine could not start: " +
                                 std::string(rad_strerror(s)));
    auto watch = std::make_unique<rad::StallWatch>("device_forward kld", 60000);
    s = engine.serve(std::move(watch));
    if (s < 0)
        throw std::runtime_error("device_forward_model_loss: kld record: " +
                                 std::string(rad_strerror(s)));

    // The record's token stream is what the test put in, so both sides really score the 32
    // conversations, in the same order.
    const std::vector<int32_t> got_tokens = read_tokens(kld_dir + "/tokens.i32");
    const bool token_match = got_tokens == expect_tokens;
    const RecordLoss kld = record_loss(kld_dir);

    nlohmann::json j = {
        {"name", "device_forward_model_loss"},
        {"device_loss", fwd.loss},
        {"device_scored", fwd.scored},
        {"kld_loss", kld.loss},
        {"kld_rows", kld.rows},
        {"token_match", token_match},
    };
    std::ofstream out(dir + "/device_forward_model_loss.json");
    if (!out) throw std::runtime_error("device_forward_model_loss: cannot write the results");
    out << j.dump(2) << "\n";
}

// ------------------------------------------- test 3: one prefill piece, 4096 and 2048 tokens

int64_t read_vram_used(const std::string& card) {
    return r97test::read_sys_int("/sys/class/drm/" + card + "/device/mem_info_vram_used");
}

// The child: one `R97_DF_TOKENS`-token sequence through the device forward; the card's VRAM at the
// peak (a background poller), and the step's wall time.
void test3_child() {
    r97test::GuardOptions o;
    const char* de = std::getenv("R97_DEVICE");
    o.device    = r97test::parse_device(de ? de : "0");
    o.need_vram = int64_t(20) << 30;
    o.bdf       = r97test::hip_bdf(o.device);
    wait_for_card_free(o);
    r97test::DeviceGuard guard(o);
    const std::string card = guard.card();  // the /sys/class/drm/cardN directory

    const std::string container = r97test::container_or_skip();
    const std::string dir = out_dir();

    // The one-prompt length this child was asked for (4096, the ticket's shape, when unset). A step
    // is one prefill piece (src/forward.cpp:177-181), and the driver switches radiance's
    // linear-state checkpoint interval off -- that interval (2048) is what would otherwise cap the
    // piece (core/sched/geometry.cpp:209) -- so the piece is whatever budget the caller sets. Both
    // sizes run one piece.
    const char* tk = std::getenv("R97_DF_TOKENS");
    const int64_t n = (tk && *tk) ? std::strtoll(tk, nullptr, 10) : 4096;
    TokenSeq seq;
    seq.ids.resize(size_t(n));
    for (size_t i = 0; i < seq.ids.size(); ++i) seq.ids[i] = 1 + int32_t((i * 7) % 299);
    seq.score.assign(seq.ids.size(), 1);
    seq.score[0] = 0;
    Batch batch;
    batch.push_back(seq);

    r97::ForwardConfig cfg;
    cfg.model         = container;
    cfg.radiance_home = radiance_home();
    cfg.kernels       = {"libr4d", "libavx", "libref"};  // the device tier
    cfg.max_tok       = n;
    cfg.max_ctx       = n;
    cfg.max_seqs      = 1;
    cfg.window_rows   = 256;
    std::string err;
    std::unique_ptr<r97::ForwardDriver> driver = r97::ForwardDriver::open(cfg, &err);
    if (!driver) throw std::runtime_error(err);
    r97::Model model;
    model.objective = r97::CeObjective;
    const r97::StepFn step = driver->step(model);

    // The step runs the batch as one engine run: start() (the model's load) and serve() (the
    // forward) inside it. Poll the card's VRAM at the peak while it runs.
    std::atomic<bool> running{true};
    std::atomic<long long> peak{0};
    std::thread poller([&] {
        while (running.load(std::memory_order_relaxed)) {
            try {
                const int64_t v = read_vram_used(card);
                long long cur = peak.load(std::memory_order_relaxed);
                while (v > cur && !peak.compare_exchange_weak(cur, v)) {
                }
            } catch (const std::exception&) {
            }  // one unreadable poll is noise; the step's time is what matters
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    });
    // The poller must be joined before any exception leaves this function: a joinable std::thread
    // destroyed during unwinding calls std::terminate, which would hide the step's real error.
    double step_seconds = 0.0;
    r97::StepResult result{0.0, 0};
    try {
        const auto t0 = std::chrono::steady_clock::now();
        // The ticket's number is the loss, not only the throughput: keep the step's result.
        result = step(batch, r97::ParamSet(), nullptr);
        step_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    } catch (...) {
        running.store(false);
        poller.join();
        throw;
    }
    running.store(false);
    poller.join();

    // The VRAM the model holds after it is loaded (before the step's buffers) is the steady-state
    // footprint: re-read it once the step is done, when the poller is off and the buffers are the
    // step's own. The peak is at least this.
    const int64_t vram_after_load = read_vram_used(card);

    const char* tn = std::getenv("R97_TEST_ONLY");
    const std::string name = (tn && *tn) ? tn : "device_forward_seq4096";
    nlohmann::json j = {
        {"name", name},
        {"tok_s", double(n) / step_seconds},
        {"vram_after_load", vram_after_load},
        {"vram_peak", peak.load()},
        {"tokens", n},
        // The loss over the sequence (the sum over scored tokens, r97/loss.h) and how many tokens
        // it covered: the ticket's number.
        {"loss", result.loss},
        {"scored", result.scored},
        {"mean_loss", result.scored > 0 ? result.loss / double(result.scored) : 0.0},
        {"card", card},
    };
    std::ofstream out(dir + "/" + name + ".json");
    if (!out) throw std::runtime_error(name + ": cannot write the results");
    out << j.dump(2) << "\n";
}

}  // namespace

// ------------------------------------------------------------------ the three tests

// The real container on the device tier: the forward loss over the three short conversations is
// finite, every scored token is scored, and the engine's own report names libr4d -- so the device
// tier is the one running the 27B, not a silent host fallback. (The host tier cannot be the other
// side of this comparison: it refuses the 27B's geometry -- hidden 5120, 262144 ctx -- and the
// device tier refuses the tiny fixture's geometry -- head_dim 32, head_k 16 and n 64 have no
// libr4d kernel -- so no model runs on both tiers on this host. The device forward's numbers are
// checked against radiance's own record in device_forward_model_loss.)
R97_MODEL_TEST(device_forward_device_tier) {
#ifdef R97_MODEL_RAD
    (void)r97test::container_or_skip(R97_MODEL_RAD);
#else
    (void)r97test::container_or_skip("");
#endif
    if (const char* m = std::getenv("R97_DF_ROLE")) {
        test1_device();
        return;  // the child did the work and wrote its file; run_all reports the pass
    }

    const std::string dir = temp_dir("device_forward_device_tier");
    run_in_child(dir, "device_forward_device_tier", "device", /*device=*/true);

    std::ifstream d(dir + "/device.json");
    if (!d) throw std::runtime_error("device_forward_device_tier: no device results at " + dir + "/device.json");
    nlohmann::json dj;
    d >> dj;
    const double loss = dj.at("loss").get<double>();
    const int64_t scored = dj.at("scored").get<int64_t>();
    const bool device_tier = dj.at("device_selected_libr4d").get<bool>();

    R97_CHECK(device_tier);
    R97_CHECK(scored > 0);
    R97_CHECK(std::isfinite(loss));
    std::printf("device_forward_device_tier: %lld scored tokens, loss %.6f, libr4d selected\n",
                (long long)scored, loss);
}

// The real container's held-out loss on the device tier against radiance's own KL record of the
// same 32 conversations, both on the device tier: the forward driver and the engine's KL mode agree
// to 1e-3 relative, on the same token stream.
R97_MODEL_TEST(device_forward_model_loss) {
#ifdef R97_MODEL_RAD
    (void)r97test::container_or_skip(R97_MODEL_RAD);
#else
    (void)r97test::container_or_skip("");
#endif

    if (const char* m = std::getenv("R97_DF_ROLE")) {
        test2_child();
        return;
    }

    const std::string dir = temp_dir("device_forward_model_loss");
    run_in_child(dir, "device_forward_model_loss", "device", /*device=*/true);

    std::ifstream in(dir + "/device_forward_model_loss.json");
    if (!in)
        throw std::runtime_error("device_forward_model_loss: no results at " + dir +
                                 "/device_forward_model_loss.json");
    nlohmann::json j;
    in >> j;
    const double device_loss = j.at("device_loss").get<double>();
    const int64_t device_scored = j.at("device_scored").get<int64_t>();
    const double kld_loss = j.at("kld_loss").get<double>();
    const int64_t kld_rows = j.at("kld_rows").get<int64_t>();
    const bool token_match = j.at("token_match").get<bool>();

    R97_CHECK(token_match);
    R97_CHECK(kld_rows == device_scored);
    // The driver's loss is the sum over the scored tokens; the KL record's is their mean. Compare
    // the means.
    const double device_mean = device_scored > 0 ? device_loss / double(device_scored) : 0.0;
    const double rel = std::fabs(device_mean - kld_loss) / std::max(std::fabs(kld_loss), 1e-12);
    if (rel > 1e-3)
        throw std::runtime_error("device_forward_model_loss: device mean loss " +
                                 std::to_string(device_mean) + " against the KL record's " +
                                 std::to_string(kld_loss) + " (relative " + std::to_string(rel) +
                                 ")");
}

// One device forward of `n` tokens through the driver: that it runs at all, the loss it produces
// over every scored token (the ticket's number), its throughput, and the card's VRAM at the step's
// peak. Two sizes are registered, 2048 and 4096, because the prompt length is the caller's budget
// and not a fixed chunk: both must run as one piece. No throughput floor is checked: the tok/s is
// the report's number, not a gate.
static void seq_forward_case(const char* test, int64_t n) {
#ifdef R97_MODEL_RAD
    (void)r97test::container_or_skip(R97_MODEL_RAD);
#else
    (void)r97test::container_or_skip("");
#endif

    if (std::getenv("R97_DF_ROLE")) {
        test3_child();
        return;
    }

    const std::string dir = temp_dir(test);
    run_in_child(dir, test, "device", /*device=*/true, n);

    const std::string path = dir + "/" + test + ".json";
    std::ifstream in(path);
    if (!in) throw std::runtime_error(std::string(test) + ": no results at " + path);
    nlohmann::json j;
    in >> j;
    const double tok_s = j.at("tok_s").get<double>();
    const int64_t vram_after_load = j.at("vram_after_load").get<int64_t>();
    const int64_t vram_peak = j.at("vram_peak").get<int64_t>();
    const int64_t tokens = j.at("tokens").get<int64_t>();
    const double loss = j.at("loss").get<double>();
    const int64_t scored = j.at("scored").get<int64_t>();
    const std::string card = j.at("card").get<std::string>();

    R97_CHECK(tokens == n);
    // Every token but the first is scored (its row predicts the second): the step's loss covers the
    // whole prompt, and no token is silently dropped.
    R97_CHECK(scored == n - 1);
    R97_CHECK(std::isfinite(loss));
    R97_CHECK(vram_peak >= vram_after_load);
    R97_CHECK(tok_s > 0);
    std::printf("%s: %s: %lld scored tokens, loss %.6f (mean %.6f), %.1f tok/s, VRAM %lld MiB "
                "after load, %lld MiB at the step's peak\n",
                test, card.c_str(), (long long)scored, loss, loss / double(scored), tok_s,
                (long long)(vram_after_load / r97test::kMiB), (long long)(vram_peak / r97test::kMiB));
}

// The ticket's shape: one 4096-token sequence.
R97_MODEL_TEST(device_forward_seq4096) { seq_forward_case("device_forward_seq4096", 4096); }

// And the shorter piece, which must keep working: the piece is the budget the caller sets, not a
// fixed 2048 (the driver turns radiance's checkpoint interval off; see src/forward.cpp).
R97_MODEL_TEST(device_forward_seq2048) { seq_forward_case("device_forward_seq2048", 2048); }

#endif  // R97_FORWARD
