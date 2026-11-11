// Write the tiny bf16 Qwen3.5 checkpoint to a directory, exactly as the tests' fixture writes it.
//
// The oracle script loads this directory instead of a container: it is the same bytes every host
// test starts from, so the reference and the fixture cannot drift apart. Nothing is converted and
// no card is touched.
//
//   r97_write_fixture /tmp/tiny_qwen35_bf16
//
// The directory is created if needed; existing files the fixture writes are replaced.

#include <cstdio>
#include <exception>
#include <string>

#include "support/fixture.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <checkpoint-directory>\n", argv[0]);
        return 2;
    }
    try {
        const std::string dir = r97::write_checkpoint(argv[1]);
        std::printf("%s\n", dir.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "r97_write_fixture: %s\n", e.what());
        return 1;
    }
    return 0;
}
