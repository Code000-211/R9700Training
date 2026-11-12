// tests/host/sha256.cpp -- the container hash `r97train train` puts in a run's base_id.
//
// The standard FIPS 180-4 vectors, the two-block message and the million-byte one among them, and
// the file form the command uses: the hash of a file is the hash of its bytes. Label: host.
#include "r97/sha256.h"
#include "support/test.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Vector {
    const char* name;
    std::string message;
    const char* digest;
};

}  // namespace

R97_TEST(sha256_matches_the_standard_vectors) {
    const std::vector<Vector> vectors = {
        {"empty", "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc", "abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"two blocks", "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
        {"a million a", std::string(1000000, 'a'),
         "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"},
    };
    for (const Vector& v : vectors) {
        const std::string got = r97::sha256_hex(v.message);
        std::printf("sha256: %-12s %s\n", v.name, got.c_str());
        R97_CHECK(got == v.digest);
    }
}

R97_TEST(sha256_file_is_the_hash_of_its_bytes) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "r97_sha256_test";
    std::filesystem::create_directories(dir);
    const std::filesystem::path file = dir / "bytes.bin";

    // A short message and one that pads across a block boundary, so the file reader's chunking
    // never decides the answer.
    for (const std::string& content : {std::string("abc"), std::string(200, 'x')}) {
        {
            std::ofstream out(file, std::ios::binary | std::ios::trunc);
            out.write(content.data(), static_cast<std::streamsize>(content.size()));
        }
        R97_CHECK(r97::sha256_file_hex(file.string()) == r97::sha256_hex(content));
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);

    // A file the command cannot read is an error, not the hash of nothing.
    bool threw = false;
    try {
        r97::sha256_file_hex((dir / "missing.bin").string());
    } catch (const std::runtime_error&) {
        threw = true;
    }
    R97_CHECK(threw);
}
