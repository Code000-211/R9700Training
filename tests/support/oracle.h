// tests/support/oracle.h -- read the PyTorch reference under tests/fixtures/oracle/.
//
// The reference file is safetensors, f64 throughout (token ids are i64), written by
// tools/oracle/make_reference.py. Its format is the one radiance's tests/tiny_model.h reads and
// writes: eight little-endian bytes of header length, the JSON header naming each tensor's dtype,
// shape and byte range, then the tensors in that order. The reader here is deliberately small:
// f64 and i64 are all the reference uses, and every value is handed to the tests as a double, so
// a comparison never has to know about bf16.
//
// Header-only, like the rest of tests/support/. Nothing here reads Python or needs PyTorch: the
// files are checked in and the tests only parse them.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace r97::oracle {

struct Tensor {
    std::vector<int64_t> shape;
    std::vector<double>  data;

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
    double at(const std::vector<int64_t>& index) const {
        if (index.size() != shape.size())
            throw std::runtime_error("oracle: tensor has " + std::to_string(shape.size()) +
                                     " dimensions, indexed with " + std::to_string(index.size()));
        int64_t flat = 0;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (index[i] < 0 || index[i] >= shape[i])
                throw std::runtime_error("oracle: tensor index out of range");
            flat = flat * shape[i] + index[i];
        }
        return data[size_t(flat)];
    }
};

namespace detail {

inline std::string read_binary(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("oracle: cannot open " + path);
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    if (n < 8) throw std::runtime_error("oracle: " + path + " is not a safetensors file");
    in.seekg(0, std::ios::beg);
    std::string out(size_t(n), '\0');
    in.read(out.data(), n);
    if (!in) throw std::runtime_error("oracle: cannot read " + path);
    return out;
}

inline uint64_t little_u64(const char* p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | uint8_t(p[i]);
    return v;
}

inline int64_t little_i64(const char* p) {
    uint64_t v = little_u64(p);
    int64_t out;
    std::memcpy(&out, &v, sizeof out);
    return out;
}

inline uint32_t little_u32(const char* p) {
    uint32_t v = 0;
    for (int i = 3; i >= 0; --i) v = (v << 8) | uint8_t(p[i]);
    return v;
}

inline uint16_t little_u16(const char* p) {
    return uint16_t(uint8_t(p[0]) | (uint16_t(uint8_t(p[1])) << 8));
}

// One safetensors file, headers parsed once and tensors decoded on demand.
class File {
  public:
    explicit File(std::string path) : path_(std::move(path)), raw_(read_binary(path_)) {
        const uint64_t header_len = little_u64(raw_.data());
        if (header_len > raw_.size() - 8)
            throw std::runtime_error("oracle: " + path_ + " has a header longer than the file");
        base_ = size_t(8 + header_len);
        header_ = nlohmann::json::parse(raw_.substr(8, size_t(header_len)), nullptr, false);
        if (header_.is_discarded() || !header_.is_object())
            throw std::runtime_error("oracle: " + path_ + " has an unreadable header");
        header_.erase("__metadata__");
    }

    const std::string& path() const { return path_; }

    bool has(const std::string& name) const { return header_.contains(name); }

    std::vector<std::string> names() const {
        std::vector<std::string> out;
        for (auto it = header_.begin(); it != header_.end(); ++it) out.push_back(it.key());
        return out;
    }

    const Tensor& tensor(const std::string& name) const {
        const auto cached = cache_.find(name);
        if (cached != cache_.end()) return cached->second;

        const auto entry = header_.find(name);
        if (entry == header_.end())
            throw std::runtime_error("oracle: " + path_ + " has no tensor \"" + name + "\"");
        const std::string dtype = entry->at("dtype").get<std::string>();
        const auto offsets = entry->at("data_offsets");
        const int64_t begin = offsets.at(0).get<int64_t>();
        const int64_t end = offsets.at(1).get<int64_t>();
        if (begin < 0 || end < begin || uint64_t(end) > raw_.size() - base_)
            throw std::runtime_error("oracle: tensor \"" + name + "\" points outside " + path_);

        Tensor t;
        for (const auto& d : entry->at("shape")) t.shape.push_back(d.get<int64_t>());
        int64_t itemsize = 0;
        if (dtype == "F64") itemsize = 8;
        else if (dtype == "I64") itemsize = 8;
        else if (dtype == "F32") itemsize = 4;
        else if (dtype == "I32") itemsize = 4;
        else if (dtype == "BF16" || dtype == "F16") itemsize = 2;
        else if (dtype == "U8" || dtype == "BOOL") itemsize = 1;
        else
            throw std::runtime_error("oracle: tensor \"" + name + "\" has unsupported dtype " +
                                     dtype);

        if (t.numel() * itemsize != end - begin)
            throw std::runtime_error("oracle: tensor \"" + name + "\" has a shape and byte range "
                                     "that disagree");
        const char* p = raw_.data() + base_ + begin;
        t.data.reserve(size_t(t.numel()));
        for (int64_t i = 0; i < t.numel(); ++i, p += itemsize) {
            if (dtype == "F64") {
                const int64_t bits = little_i64(p);
                double v;
                std::memcpy(&v, &bits, sizeof v);
                t.data.push_back(v);
            } else if (dtype == "I64") {
                t.data.push_back(double(little_i64(p)));
            } else if (dtype == "F32") {
                const uint32_t bits = little_u32(p);
                float v;
                std::memcpy(&v, &bits, sizeof v);
                t.data.push_back(double(v));
            } else if (dtype == "I32") {
                t.data.push_back(double(int32_t(little_u32(p))));
            } else if (dtype == "U8" || dtype == "BOOL") {
                t.data.push_back(double(uint8_t(p[0])));
            } else if (dtype == "BF16") {
                const uint32_t bits = uint32_t(little_u16(p)) << 16;
                float v;
                std::memcpy(&v, &bits, sizeof v);
                t.data.push_back(double(v));
            } else {  // F16
                const uint16_t h = little_u16(p);
                const uint32_t sign = uint32_t(h & 0x8000) << 16;
                const uint32_t exponent = (h >> 10) & 0x1f;
                const uint32_t mantissa = h & 0x3ff;
                uint32_t bits;
                float f;
                if (exponent == 0) {
                    // Subnormal: 2^-14 * mantissa / 1024; the values the reference never uses.
                    f = std::ldexp(float(mantissa), -24);
                    f = (h & 0x8000) ? -f : f;
                    t.data.push_back(double(f));
                    continue;
                }
                if (exponent == 0x1f) bits = sign | 0x7f800000u | (mantissa << 13);
                else bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
                std::memcpy(&f, &bits, sizeof f);
                t.data.push_back(double(f));
            }
        }
        return cache_.emplace(name, std::move(t)).first->second;
    }

  private:
    std::string                 path_;
    std::string                 raw_;
    size_t                      base_ = 0;
    nlohmann::json              header_;
    mutable std::map<std::string, Tensor> cache_;
};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

inline std::string sha256_hex(const std::string& bytes) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

    std::string msg = bytes;
    const uint64_t bitlen = uint64_t(bytes.size()) * 8;
    msg.push_back(char(0x80));
    while (msg.size() % 64 != 56) msg.push_back('\0');
    for (int i = 7; i >= 0; --i) msg.push_back(char((bitlen >> (8 * i)) & 0xff));

    for (size_t off = 0; off < msg.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            const auto* p = reinterpret_cast<const uint8_t*>(msg.data() + off + 4 * i);
            w[i] = (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) |
                   uint32_t(p[3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = k + S1 + ch + K[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + maj;
            k = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += k;
    }

    static const char* hex = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (uint32_t v : h)
        for (int i = 7; i >= 0; --i) out.push_back(hex[(v >> (4 * i)) & 0xf]);
    return out;
}

}  // namespace detail

// The SHA-256 of a whole file, lower-case hex.
inline std::string sha256_file(const std::string& path) {
    return detail::sha256_hex(detail::read_binary(path));
}

// tests/fixtures/oracle/manifest.json's fixture.sha256: the hash of the model.safetensors the
// reference was generated from. A changed fixture fails the host test here, by name.
inline std::string manifest_fixture_sha256(const std::string& manifest_path) {
    const std::string raw = detail::read_binary(manifest_path);
    const nlohmann::json m = nlohmann::json::parse(raw, nullptr, false);
    if (m.is_discarded() || !m.contains("fixture") || !m["fixture"].contains("sha256"))
        throw std::runtime_error("oracle: " + manifest_path + " has no fixture.sha256");
    return m["fixture"]["sha256"].get<std::string>();
}

}  // namespace r97::oracle
