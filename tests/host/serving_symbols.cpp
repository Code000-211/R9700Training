// tests/host/serving_symbols.cpp -- what a serving build must not carry from the training set.
//
// The serving set (patches/serving/) is what goes upstream and what a deployment runs; the
// training set (patches/training/) adds the issue observer and the replay hook. A serving build
// that quietly picked up either would be carrying code the deployment does not need and nobody
// upstreamed. Both names are declared in abi/rad_runtime.h, so a check for the string in the built
// rad_core's archive answers "was this compiled in?" without a card: the archive holds a symbol
// table, and a function that was never compiled leaves no symbol behind.
//
// The default build runs the same check from the other side (the symbols must be present), so a
// check that could no longer see them would fail there rather than pass here.
#include "../support/test.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// The observer (0101) and the replay entry (0102). The block brackets are the observer patch's,
// and the replay block is the entry point 0102 adds.
constexpr const char* kTrainingSymbols[] = {
    "rad_set_issue_observer",
    "rad_block_begin",
    "rad_block_end",
    "rad_replay_block",
};

std::string archive_bytes() {
#ifdef R97_RAD_CORE_ARCHIVE
    const std::filesystem::path path = R97_RAD_CORE_ARCHIVE;
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("serving_symbols: cannot read " + path.string());
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
#else
    throw std::runtime_error("serving_symbols: no rad_core archive path was compiled in");
#endif
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

}  // namespace

R97_TEST(serving_build_carries_no_observer_or_replay_symbol) {
    const std::string bytes = archive_bytes();
    R97_CHECK(!bytes.empty());
#if defined(R97_PATCH_SET_SERVING)
    for (const char* symbol : kTrainingSymbols) {
        if (contains(bytes, symbol))
            std::fprintf(stderr, "serving_symbols: serving rad_core carries %s\n", symbol);
        R97_CHECK(!contains(bytes, symbol));
    }
#else
    // The default build carries the training set; seeing the symbols here is what makes the
    // serving-only assertion above a measurement rather than a string that is never there.
    for (const char* symbol : kTrainingSymbols) R97_CHECK(contains(bytes, symbol));
#endif
}
