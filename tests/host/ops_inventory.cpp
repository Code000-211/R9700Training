// tests/host/ops_inventory.cpp -- the op inventory of both tiny fixtures is the training forward.
//
// The inventory is src/op_inventory.h's reading of one recorded prefill step. Four things the
// adjoint issues depend on are checked here, for each fixture: the inventory is stable run to run
// (so a doc's table and a later issue's are comparable), it counts every issue the step made (no
// op is left out by name), the training forward carries no drafter, MTP head, vision tower or
// sampler op, and exactly one op exposes the `logits` seed an objective starts its walk from. The
// block kinds, the on-path mark and the adjoint families are checked for shape too, so a
// classifier that quietly collapses stops the test rather than the doc.
//
// The step is recorded in a child process, by `r97train ops`. The host binary runs its registered
// tests from the at-exit runner (support/test.h), and by then the statics a recorded step needs
// are gone: the fixture's own (support/fixture.h) and radiance's, whose device layer reads state
// destructed with them, so a step recorded in this process dies instead of running. Each
// fixture's container path is therefore copied into a POD while the fixtures are built at
// startup -- no destructor, so it is still readable at exit -- and the step runs in the child,
// which is what the subcommand this issue adds does for a person. A serving build has no prefill
// mode, and this file compiles to nothing there.
#include "support/fixture.h"
#include "support/fixture_w8a8.h"
#include "support/test.h"

#if R97_FIXTURE_PREFILL

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

// Filled while the fixtures are alive, read after teardown: no std::string, nothing destroys it.
struct Containers {
    char bf16[4096];
    char w8a8[4096];
    char failure[1024];
};
Containers g_containers;

void remember(char* to, size_t size, const std::string& from) {
    std::snprintf(to, size, "%s", from.c_str());
}

struct CaptureFixtures {
    CaptureFixtures() {
        try {
            remember(g_containers.bf16, sizeof g_containers.bf16, r97::fixture().container);
            remember(g_containers.w8a8, sizeof g_containers.w8a8, r97::w8a8_fixture().container);
        } catch (const std::exception& e) {
            remember(g_containers.failure, sizeof g_containers.failure, e.what());
        }
    }
} g_capture_fixtures;

std::string shell_quote(const std::string& s) {
    std::string q = "'";
    for (char c : s) {
        if (c == '\'') q += "'\\''";
        else q += c;
    }
    return q + "'";
}

// The directory this binary sits in, asked for on every call: a lookup kept in a static would
// already have been destroyed by the time this test runs.
std::string build_dir() {
    std::error_code ec;
    const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
    return ec ? std::string() : exe.parent_path().string();
}

// One run of the subcommand, as it prints for a person. The engine's own log lines go to stderr
// and are dropped: this reads the inventory, and a run that failed says so in its exit status.
std::string inventory_text(const std::string& container) {
    const std::string build = build_dir();
    const std::string tool = build.empty() ? std::string("r97train") : build + "/r97train";
    std::string cmd = shell_quote(tool) + " ops " + shell_quote(container);
    std::error_code ec;
    if (!build.empty()) {
        const std::filesystem::path home =
            std::filesystem::path(build) / "radiance-build" / "radiance_home";
        if (std::filesystem::exists(home, ec)) cmd += " --home " + shell_quote(home.string());
    }
    cmd += " 2>/dev/null";

    std::FILE* pipe = ::popen(cmd.c_str(), "r");
    if (!pipe) throw std::runtime_error("ops_inventory: cannot run " + tool);
    std::string out;
    char buf[4096];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof buf, pipe)) > 0) out.append(buf, n);
    const int status = ::pclose(pipe);
    if (status != 0)
        throw std::runtime_error("ops_inventory: '" + tool + " ops " + container +
                                 "' failed (status " + std::to_string(status) + ")");
    if (out.empty()) throw std::runtime_error("ops_inventory: '" + tool + "' printed nothing");
    return out;
}

std::string trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t");
    if (a == std::string::npos) return std::string();
    const size_t b = s.find_last_not_of(" \t");
    return s.substr(a, b - a + 1);
}

// The cells between the bars of one markdown row, header and rule included.
std::vector<std::string> cells(const std::string& line) {
    std::vector<std::string> out;
    size_t bar = line.find('|');
    while (bar != std::string::npos) {
        const size_t next = line.find('|', bar + 1);
        if (next == std::string::npos) break;
        out.push_back(trim(line.substr(bar + 1, next - bar - 1)));
        bar = next;
    }
    return out;
}

// "# <container>: one prefill step, <issues> issues, <forms> op forms".
void headline(const std::string& text, long long& issues, long long& forms, const std::string& what) {
    const size_t at = text.find("one prefill step, ");
    if (at == std::string::npos) throw std::runtime_error(what + "the output has no headline");
    if (std::sscanf(text.c_str() + at, "one prefill step, %lld issues, %lld op forms", &issues,
                    &forms) != 2)
        throw std::runtime_error(what + "the headline cannot be read");
}

struct Row {
    std::string op, block, count, seed, path, adjoint;
};

std::vector<Row> rows_of(const std::string& text, const std::string& what) {
    std::vector<Row> rows;
    size_t at = 0;
    while (at < text.size()) {
        size_t end = text.find('\n', at);
        if (end == std::string::npos) end = text.size();
        const std::string line = text.substr(at, end - at);
        at = end + 1;
        if (line.rfind("| `", 0) != 0) continue;  // the header, the rule, the rest of the output
        const std::vector<std::string> c = cells(line);
        if (c.size() != 10)
            throw std::runtime_error(what + "a row has " + std::to_string(c.size()) +
                                     " columns, not 10: " + line);
        Row row;
        row.op = c[0];
        if (row.op.size() >= 2 && row.op.front() == '`' && row.op.back() == '`')
            row.op = row.op.substr(1, row.op.size() - 2);
        row.block = c[1];
        row.count = c[3];
        row.seed = c[7];
        row.path = c[8];
        row.adjoint = c[9];
        rows.push_back(row);
    }
    return rows;
}

bool forbidden(const std::string& op) {
    // The drafter (DFlash2 and its dspark sibling), the MTP head, the vision tower and the
    // sampler: none of them belongs to the training forward of this milestone.
    static const char* const words[] = {"draft", "dflash", "dspark", "mtp", "vision", "vit_",
                                       "sample", "sampler", "top_k", "top_p", "min_p"};
    for (const char* w : words)
        if (op.find(w) != std::string::npos) return true;
    return false;
}

void check_one(const char* container, const char* which) {
    const std::string what = std::string(which) + ": ";
    if (g_containers.failure[0] != '\0')
        throw std::runtime_error(what + "the fixture was not built: " + g_containers.failure);
    if (container[0] == '\0') throw std::runtime_error(what + "no container path was captured");

    const std::string once = inventory_text(container);
    long long issues = 0, forms = 0;
    headline(once, issues, forms, what);
    const std::vector<Row> rows = rows_of(once, what);
    const std::string twice = inventory_text(container);

    // Stable across runs: the same step, recorded again, prints the same table.
    if (once != twice)
        throw std::runtime_error(what + "two runs' inventories differ:\n" + once + "\nagainst\n" +
                                 twice);
    if (issues <= 0) throw std::runtime_error(what + "the step issued no op");
    if (rows.empty()) throw std::runtime_error(what + "the inventory has no row");
    if ((long long)rows.size() != forms)
        throw std::runtime_error(what + "the headline says " + std::to_string(forms) +
                                 " op forms, the table has " + std::to_string(rows.size()));

    long long counted = 0, seeds = 0, embedding = 0, attention = 0, delta = 0, mlp = 0, head = 0;
    for (const Row& row : rows) {
        if (forbidden(row.op)) throw std::runtime_error(what + "the step issued '" + row.op + "'");
        if (row.seed == "logits") ++seeds;
        if (row.seed == "ce" || row.seed == "gate_logits")
            throw std::runtime_error(what + "the plain forward writes the rollout seed '" +
                                     row.seed + "'");
        if (row.adjoint == "none")
            throw std::runtime_error(what + "op '" + row.op + "' fits no adjoint family");
        const long long count = std::atoll(row.count.c_str());
        if (count <= 0)
            throw std::runtime_error(what + "op '" + row.op + "' has no count");
        counted += count;
        // The sampler is absent because the prefill driver asks for no token
        // (patches/training/0105), not because a filter here drops its ops: every issue the step
        // made has a row, so an op the inventory forgets fails the count above rather than going
        // unnoticed.
        if (row.path != "yes" && row.path.rfind("no: ", 0) != 0)
            throw std::runtime_error(what + "op '" + row.op + "' has no on-path mark");
        if (row.block == "embedding") ++embedding;
        else if (row.block == "full-attention") ++attention;
        else if (row.block == "gated-delta-net") ++delta;
        else if (row.block == "mlp") ++mlp;
        else if (row.block == "head") ++head;
        else
            throw std::runtime_error(what + "op '" + row.op + "' runs in an unknown block '" +
                                     row.block + "'");
    }
    if (counted != issues)
        throw std::runtime_error(what + "the table counts " + std::to_string(counted) +
                                 " issues, the step made " + std::to_string(issues));
    if (seeds != 1)
        throw std::runtime_error(what + "expected exactly one op with the 'logits' seed, found " +
                                 std::to_string(seeds));
    if (!embedding || !attention || !delta || !mlp || !head)
        throw std::runtime_error(what + "a block kind is missing from the inventory");

    std::printf("  %s: %lld ops in %zu forms, one logits seed, no drafter/vision/sampler op ok\n",
                which, issues, rows.size());
    std::printf("\n%s\n", once.c_str());
}

R97_TEST(ops_inventory_bf16_fixture) { check_one(g_containers.bf16, "bf16 fixture"); }

R97_TEST(ops_inventory_w8a8_fixture) { check_one(g_containers.w8a8, "w8a8 fixture"); }

}  // namespace

#endif  // R97_FIXTURE_PREFILL
