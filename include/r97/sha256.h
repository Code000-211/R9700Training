// r97/sha256.h -- SHA-256 (FIPS 180-4), and the hash of a file.
//
// `r97train train` names the container it trains against by the file's hash, so a resume against a
// different container is refused by name (docs/TRAINING.md, base_id). The digest is the standard
// one: sixty-four lowercase hex digits, the same string `sha256sum` prints.
#pragma once

#include <cstddef>
#include <string>

namespace r97 {

// The digest of `size` bytes at `data`.
std::string sha256_hex(const void* data, std::size_t size);

// The digest of a string's bytes.
inline std::string sha256_hex(const std::string& bytes) {
    return sha256_hex(bytes.data(), bytes.size());
}

// The digest of a file's bytes. Streamed in chunks, so a container is never held in memory twice.
// Throws std::runtime_error when the file cannot be opened or read: a missing container has no id,
// and saying so is better than hashing nothing.
std::string sha256_file_hex(const std::string& path);

}  // namespace r97
