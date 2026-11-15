// tests/host/parallel_for.cpp -- the parallel for the host references use, held to the serial
// loop it replaces: the same values, bit for bit, at the sizes a partition must survive, and a
// worker's exception coming back to the caller instead of into std::terminate.
//
// The body every check runs is the reference output's own shape: one element at a time, each
// the serial sum of its own row of distinct inputs -- what a twin or adjoint row writes per
// token row. The parallel answer is compared against the serial one byte for byte, not in
// tolerance: the helper's promise is that the summation order never moved.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "support/parallel_for.h"
#include "support/test.h"

namespace {

constexpr int64_t kW = 37;  // the terms one output sums; odd, so no pairing hides in the input

double term(uint64_t s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return (double)(s >> 11) * (1.0 / 9007199254740992.0) - 0.5;
}

std::vector<double> make_inputs(int64_t n) {
    std::vector<double> in((size_t)(n * kW));
    uint64_t s = 0x9e3779b97f4a7c15ull;
    for (double& x : in) x = term(s);
    return in;
}

// out[i] = the serial sum of row i, exactly what the serial loop writes.
void serial_sum(const std::vector<double>& in, std::vector<double>* out) {
    const int64_t n = static_cast<int64_t>(in.size() / kW);
    out->assign((size_t)n, 0.0);
    for (int64_t i = 0; i < n; ++i) {
        double acc = 0.0;
        for (int64_t k = 0; k < kW; ++k) acc += in[static_cast<std::size_t>(i * kW + k)];
        (*out)[static_cast<std::size_t>(i)] = acc;
    }
}

void check_size(int64_t n, int64_t chunk) {
    const std::vector<double> in = make_inputs(n);
    std::vector<double> serial, parallel((size_t)n);
    serial_sum(in, &serial);
    r97::parallel_for(n, chunk, [&](int64_t i) {
        double acc = 0.0;
        for (int64_t k = 0; k < kW; ++k) acc += in[static_cast<std::size_t>(i * kW + k)];
        parallel[static_cast<std::size_t>(i)] = acc;
    });
    if (parallel.size() != serial.size() ||
        (n > 0 &&
         std::memcmp(parallel.data(), serial.data(), (size_t)n * sizeof(double)) != 0)) {
        throw std::runtime_error("parallel_for at n = " + std::to_string(n) +
                                 " (chunk " + std::to_string(chunk) +
                                 ") is not the serial loop bit for bit");
    }
    std::printf("parallel_for n %lld chunk %lld: %s\n", (long long)n, (long long)chunk,
                n < chunk ? "serial (under one chunk)" : "threaded");
}

// The sizes a partition must survive: empty, one, the chunk's edges, and the non-divisible
// ones the tail chunk cuts through.
void check_sizes(int64_t chunk) {
    const std::vector<int64_t> sizes = {0, 1, chunk - 1, chunk, chunk + 1, 25, 1000, 333,
                                        1 << 20};
    for (int64_t n : sizes) check_size(n, chunk);
}

// R97_THREADS for the duration of one scope, restored on the way out: the count is read per
// call, so a test that sets it must leave the runner's value behind.
struct ThreadsScope {
    explicit ThreadsScope(const char* v) {
        if (const char* old = std::getenv("R97_THREADS")) {
            had = true;
            saved = old;
        }
        if (v) ::setenv("R97_THREADS", v, 1);
        else ::unsetenv("R97_THREADS");
    }
    ~ThreadsScope() {
        if (had) ::setenv("R97_THREADS", saved.c_str(), 1);
        else ::unsetenv("R97_THREADS");
    }
    bool had = false;
    std::string saved;
};

}  // namespace

R97_TEST(parallel_for_bit_identity) {
    check_sizes(8);   // 0, 1, 7, 8, 9, and the non-divisible 25, 333, 1000
    check_sizes(64);  // the same set at the lora rows' chunk
    check_sizes(1);   // every size over one threads; the order still does not move
}

R97_TEST(parallel_for_worker_count) {
    // One requested worker and one parsed from an unsettable env value: the inline loop, the
    // serial answer. The value is read per call, so the scope restores it for the runner.
    {
        ThreadsScope one("1");
        check_size(1000, 1);
        R97_CHECK(r97::parallel_thread_count() == 1);
    }
    {
        ThreadsScope zero("0");  // below one: the default, not zero threads
        R97_CHECK(r97::parallel_thread_count() >= 1);
        check_size(1000, 1);
    }
    {
        ThreadsScope junk("not a number");
        R97_CHECK(r97::parallel_thread_count() >= 1);
        check_size(25, 8);
    }
}

R97_TEST(parallel_for_exception) {
    // The unit that throws is deep enough that other threads are past it when it fails: the
    // caller gets the exception back after every worker has exited, and the test never hangs
    // on a join.
    const int64_t n = 1 << 18, chunk = 64;
    bool seen = false;
    try {
        r97::parallel_for(n, chunk, [&](int64_t i) {
            if (i == n / 2) throw std::runtime_error("the unit's own failure");
        });
    } catch (const std::runtime_error& e) {
        seen = true;
        R97_CHECK(std::string(e.what()) == "the unit's own failure");
    }
    R97_CHECK(seen);
    // The inline path throws the same way, uncaught by any join.
    seen = false;
    try {
        r97::parallel_for(3, 8, [](int64_t i) {
            if (i == 1) throw std::runtime_error("the serial unit's failure");
        });
    } catch (const std::runtime_error& e) {
        seen = true;
        R97_CHECK(std::string(e.what()) == "the serial unit's failure");
    }
    R97_CHECK(seen);
}
