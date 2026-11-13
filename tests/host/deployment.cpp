// tests/host/deployment.cpp -- `docs/DEPLOYMENT.md` against the tree it describes.
//
// Deployment is the serving-only build: a radiance that carries `patches/serving/` and nothing
// else, the model container, and an adapter directory, serving on the host and on the card. The
// document is the one place that says what such a build carries and how the adapter is served from
// it, and tests/host/runbook.cpp holds its commands; this test holds the facts the document can
// drift on without anything failing:
//   - every patch in `patches/serving/` is named by it and no patch in `patches/training/` is: the
//     deployment is the serving set alone, so a new serving patch has to reach the document and a
//     trainer-only one must not be listed as something a deployment applies;
//   - its build line is the patch set's own name, taken from `cmake/ApplyPatches.cmake` where the
//     set is accepted, and both flavours are named: the card's HIP build and the host's
//     `-DR97_WITH_HIP=OFF`, because the build serves on both;
//   - the card is named the way the trainer names it (`ROCR_VISIBLE_DEVICES`, `--device`), since
//     that is what `docs/RUNBOOK.md` section 7 runs;
//   - the check it documents is `docs/EVAL.md`'s: the trainer's loss and radiance's KL record, at
//     1e-4 relative, through `tools/kld_loss.py`.
#include "../support/test.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

// The tree's top, the directory holding docs/DEPLOYMENT.md. The recipe's path is the source
// tree's own, written into this binary by CMake, so the first ancestor that holds the document is
// it.
fs::path source_root() {
    fs::path at = fs::path(R97_FIXTURE_W8A8_RECIPE);
    for (int up = 0; up < 6 && !at.empty(); ++up) {
        if (fs::exists(at / "docs" / "DEPLOYMENT.md")) return at;
        at = at.parent_path();
    }
    throw std::runtime_error("deployment: cannot find docs/DEPLOYMENT.md from " +
                             std::string(R97_FIXTURE_W8A8_RECIPE));
}

std::string read_file(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("deployment: cannot read " + path.string());
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::vector<std::string> patch_names(const fs::path& dir) {
    std::vector<std::string> names;
    for (const fs::directory_entry& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file() || e.path().extension() != ".patch") continue;
        names.push_back(e.path().filename().string());
    }
    std::sort(names.begin(), names.end());
    return names;
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

}  // namespace

R97_TEST(deployment_doc_names_the_serving_build_it_ships) {
    const fs::path root = source_root();
    const std::string doc = read_file(root / "docs" / "DEPLOYMENT.md");

    const std::vector<std::string> serving = patch_names(root / "patches" / "serving");
    R97_CHECK(!serving.empty());
    for (const std::string& name : serving) R97_CHECK(contains(doc, name));

    for (const std::string& name : patch_names(root / "patches" / "training"))
        R97_CHECK(!contains(doc, name));

    // The set's name is the one the patch mechanism accepts, not a name the document invented.
    const std::string sets = read_file(root / "cmake" / "ApplyPatches.cmake");
    R97_CHECK(contains(sets, "\"serving\""));
    R97_CHECK(contains(doc, "-DR97_PATCH_SET=serving"));
    R97_CHECK(contains(doc, "-DR97_WITH_HIP=OFF"));

    R97_CHECK(contains(doc, "--radiance-home"));
    R97_CHECK(contains(doc, "--lora"));
    R97_CHECK(contains(doc, "ROCR_VISIBLE_DEVICES"));
    R97_CHECK(contains(doc, "--device"));
    // A HIP serving home also holds libr97, the trainer's plugin, and the server loads every
    // library under kernels/ unless --kernels names them: the document has to say which three the
    // card's serve names, or the served number is not the trainer's kernels' number.
    R97_CHECK(contains(doc, "libr97"));
    R97_CHECK(contains(doc, "--kernels libr4d,libavx,libref"));

    R97_CHECK(contains(doc, "1e-4"));
    R97_CHECK(contains(doc, "tools/kld_loss.py"));
    R97_CHECK(contains(read_file(root / "docs" / "EVAL.md"), "1e-4"));
}
