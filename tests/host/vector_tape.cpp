// tests/host/vector_tape.cpp -- VectorTape answers reads from its arrays and records them.
//
// This file also owns the collected host binary's entry point: its main runs every test another
// host file registered with R97_TEST (support/test.h), then _Exits with the result.
#include "support/test.h"
#include "support/vector_tape.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

using r97::Batch;
using r97::TokenSeq;

static Batch two_sequences() {
    Batch b;
    b.push_back(TokenSeq{{1, 2, 3, 4}, {0, 1, 0, 1}});
    b.push_back(TokenSeq{{5, 6, 7}, {0, 0, 1}});
    return b;
}

int main() {
    // A logits buffer of 5 rows x 3 cols: rows 0..2 for the first sequence, 3..4 for the second.
    std::vector<float> logits = {
        0, 1, 2,    // row 0
        3, 4, 5,    // row 1
        6, 7, 8,    // row 2
        9, 10, 11,  // row 3
        12, 13, 14, // row 4
    };
    r97test::VectorTape tape;
    tape.set_batch(two_sequences())
        .set_scored(4)
        .set_buffer("logits", {5, 3}, logits);

    // Rows [1, 3) come back whole, in order, as f32.
    std::vector<float> got = tape.read("logits", 1, 3, true);
    assert((got == std::vector<float>{3, 4, 5, 6, 7, 8}));

    // The window is the whole buffer until a test restricts it.
    int64_t lo = -1, hi = -1;
    tape.window("logits", &lo, &hi);
    assert(lo == 0 && hi == 5);

    tape.set_window("logits", 2, 5);
    tape.window("logits", &lo, &hi);
    assert(lo == 2 && hi == 5);

    // Only rows inside the window read out of the restricted range, row counts per shape.
    assert(tape.shape("logits") == std::vector<int64_t>({5, 3}));
    assert(tape.boundaries() == std::vector<std::string>({"logits"}));
    assert(tape.total_scored() == 4);
    assert(tape.batch().size() == 2 && tape.batch()[1].ids[2] == 7);

    // The record holds every read, with its range and its differentiated flag; a read of another
    // buffer would not be folded into the first's record.
    assert(tape.reads_of("logits").size() == 1);
    assert(tape.reads_of("logits")[0].row_lo == 1 && tape.reads_of("logits")[0].row_hi == 3);
    assert(tape.reads_of("logits")[0].differentiated == true);
    tape.read("logits", 0, 1, false);
    tape.read("logits", 1, 2, true);
    assert(tape.reads().size() == 3);
    assert(!tape.reads()[1].differentiated && tape.reads()[2].differentiated);
    assert(tape.reads_of("other").empty());

    std::puts("vector_tape ok");

    // The registered tests run here, from main, rather than from the at-exit runner support/test.h
    // arms at the first registration. That runner is armed during static initialisation and runs
    // after the binary's static destructors have run, and the statics a host test needs are gone by
    // then: chatdata_mask.cpp builds the fixture (support/fixture.h) from its own static
    // initialiser, so the fixture, the temp directory behind it and radiance's device layer are
    // destroyed before an armed handler would call a test. Running the tests inside main keeps
    // every static alive for the whole run, whatever order the translation units initialise in;
    // _Exit then skips the rest of exit(), so no destructor and no second run can hide the status.
    // tests/gpu/main.cpp does the same, for the registry a plugin rebuilds as it loads.
    const int failed = r97test::run_all();
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(failed ? 1 : 0);
}
