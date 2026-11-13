// src/parallel.h -- the parallel for the host reference loops share.
//
// The f64 reference arithmetic is a bit-exact contract: the tests compare its values against
// pinned expectations, and the joins sum its cotangents in the order they arrive. Parallelism
// may change how many cores do the work, never what any output element sums. The rules the
// helper enforces or assumes:
//
//   * Partition by output. body(i) computes the output at i from the inputs and writes only
//     that output. It must never read or write a value another i writes, and never accumulate
//     into a buffer several units share: a shared accumulator would need a cross-thread sum,
//     and the sum's order is part of the contract. A reduction over one unit's own inputs
//     stays that unit's serial inner loop, in the order the serial code had.
//   * Fixed chunks. [0, n) is cut into chunks of `chunk` units and worker t takes chunks
//     t, t + T, ... in that order. A chunk decides which thread does the work and nothing
//     else: every unit runs on exactly one thread, in the serial order of its inner loops.
//   * The worker count is R97_THREADS when it holds an integer of at least 1, otherwise the
//     smaller of 8 and the hardware concurrency (one when it is unknown). With fewer than one
//     chunk of work, or one worker, the loop runs inline on the calling thread: no join, no
//     copy, the serial loop exactly.
//
// A unit that throws does not terminate the process and does not hang the join: the workers
// record the first exception and run out their own chunks (their outputs are disjoint and the
// caller discards the buffer on the rethrow), and parallel_for rethrows it once every worker
// has exited.

#pragma once

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace r97 {

// The worker count the reference loops use: R97_THREADS when it holds an integer of at least 1,
// else min(8, hardware_concurrency), one when the concurrency is unknown.
inline int parallel_thread_count() {
    if (const char* e = std::getenv("R97_THREADS")) {
        if (*e) {
            char* end = nullptr;
            const long v = std::strtol(e, &end, 10);
            if (end != e && *end == '\0' && v >= 1) return static_cast<int>(v);
        }
    }
    const unsigned hw = std::thread::hardware_concurrency();
    return hw > 0 ? (hw < 8u ? static_cast<int>(hw) : 8) : 1;
}

// body(i) over i in [0, n), under the rules at the top of this file. `chunk` is both the unit
// of the partition and the threshold: n < chunk runs serial.
template <class F>
inline void parallel_for(int64_t n, int64_t chunk, F&& body) {
    if (n <= 0) return;
    if (chunk < 1) chunk = 1;
    const int T = parallel_thread_count();
    if (T < 2 || n < chunk) {
        for (int64_t i = 0; i < n; ++i) body(i);
        return;
    }
    const int64_t nchunks = (n + chunk - 1) / chunk;
    std::mutex mu;
    std::exception_ptr first;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(T));
    try {
        for (int t = 0; t < T; ++t) {
            workers.emplace_back([&, t]() {
                try {
                    for (int64_t c = t; c < nchunks; c += T) {
                        const int64_t lo = c * chunk;
                        const int64_t hi = lo + chunk;
                        for (int64_t i = lo; i < (hi < n ? hi : n); ++i) body(i);
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lk(mu);
                    if (!first) first = std::current_exception();
                }
            });
        }
    } catch (...) {
        // A thread that would not start: join the ones that did and give the error back.
        for (auto& w : workers)
            if (w.joinable()) w.join();
        throw;
    }
    for (auto& w : workers) w.join();
    if (first) std::rethrow_exception(first);
}

}  // namespace r97
