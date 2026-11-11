// tests/support/bench.h -- the gpu binary's --bench switch, read by the device harness.
//
// Correctness runs first either way: --bench adds a timed pass per device case, it does not
// replace the comparison. tests/gpu/main.cpp owns the argument; everything else only reads the
// flag, so a support header needs no argv.
#pragma once

namespace r97test {

inline bool& bench_flag() {
    static bool on = false;
    return on;
}

inline bool bench() { return bench_flag(); }

}  // namespace r97test
