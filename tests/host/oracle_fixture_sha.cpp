// tests/host/oracle_fixture_sha.cpp -- the PyTorch reference was made from today's fixture.
//
// tools/oracle/make_reference.py records the SHA-256 of the checkpoint's model.safetensors in
// tests/fixtures/oracle/manifest.json. A fixture change is caught here, by name, instead of
// surfacing later as a gradient mismatch somewhere else. Label: host.
#include "support/fixture.h"
#include "support/oracle.h"
#include "support/test.h"

#include <filesystem>
#include <string>

namespace {

std::filesystem::path oracle_dir() {
    return std::filesystem::path(__FILE__).parent_path().parent_path() / "fixtures" / "oracle";
}

}  // namespace

// The reference is compared against the fixture the host tests use, so this is the fixture's own
// write path (write_qwen35), not a copy under tests/fixtures/.
R97_TEST(oracle_fixture_sha_is_the_manifest_sha) {
    const r97::Fixture& f = r97::fixture();
    const std::string model = f.checkpoint + "/model.safetensors";
    const std::string actual = r97::oracle::sha256_file(model);
    const std::string expected =
        r97::oracle::manifest_fixture_sha256((oracle_dir() / "manifest.json").string());

    std::printf("oracle_fixture_sha: fixture %s\n", actual.c_str());
    if (actual != expected) {
        std::fprintf(stderr,
                     "oracle_fixture_sha: the fixture changed. manifest.json records %s, today's "
                     "checkpoint is %s. Regenerate the reference with tools/oracle/make_reference.py "
                     "only after the cause is known and written into docs/ORACLE.md.\n",
                     expected.c_str(), actual.c_str());
        R97_CHECK(false);
    }
}
