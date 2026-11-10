// tests/host/seed_check.cpp -- the refusals of check_seeds fire and name the buffer, a result
// with two seeds on two boundaries passes, a result with no seed passes, and BoundaryNames maps a
// boundary's name to the id and rows the tape recorded (label: host).
#undef NDEBUG   // the checks must survive the default RelWithDebInfo build's -DNDEBUG
#include "r97/model.h"

#include "../support/reader_objective.h"
#include "../support/recorded_vector_tape.h"

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

using r97::Seed;

static r97test::RecordedVectorTape tapeWithLogits() {
    r97test::RecordedVectorTape tape;
    tape.set_buffer("logits", {5, 4}, std::vector<float>(20, 1.0f));   // rows [0, 5), 4 columns
    return tape;
}

static std::string refuses(const r97test::RecordedVectorTape& tape,
                           const r97::ObjectiveResult& result) {
    try {
        r97::check_seeds(tape, result);
        assert(false && "check_seeds must refuse this result");
        return "";
    } catch (const std::exception& e) {
        return e.what();
    }
}

static Seed ones(const char* buffer, int64_t lo, int64_t hi) {
    return Seed{buffer, lo, hi, std::vector<float>(
        static_cast<size_t>((hi - lo) * 4), 1.0f)};
}

int main() {
    // 1. A seed on a buffer that is not a boundary of the tape.
    {
        r97test::RecordedVectorTape tape = tapeWithLogits();
        r97::ObjectiveResult result;
        result.seeds = {ones("embeddings", 0, 1)};
        const std::string what = refuses(tape, result);
        assert(what.find("embeddings") != std::string::npos);
        assert(what.find("not a boundary") != std::string::npos);
    }

    // 2a. A seed whose row range lies outside the window.
    {
        r97test::RecordedVectorTape tape = tapeWithLogits();
        r97::ObjectiveResult result;
        result.seeds = {ones("logits", 4, 7)};
        const std::string what = refuses(tape, result);
        assert(what.find("logits") != std::string::npos);
        assert(what.find("outside the window") != std::string::npos);
    }

    // 2b. The same for a window that is narrower than the buffer.
    {
        r97test::RecordedVectorTape tape = tapeWithLogits();
        tape.set_window("logits", 1, 4);
        r97::ObjectiveResult result;
        result.seeds = {ones("logits", 0, 2)};
        const std::string what = refuses(tape, result);
        assert(what.find("logits") != std::string::npos);
        assert(what.find("outside the window") != std::string::npos);
    }

    // 2c. A seed whose cotangent does not have the size of its rows.
    {
        r97test::RecordedVectorTape tape = tapeWithLogits();
        r97::ObjectiveResult result;
        result.seeds = {Seed{"logits", 0, 2, std::vector<float>(4, 1.0f)}};   // 2 rows need 8
        const std::string what = refuses(tape, result);
        assert(what.find("logits") != std::string::npos);
        assert(what.find("cotangent") != std::string::npos);
    }

    // 3. Two seeds of one result that overlap on a buffer.
    {
        r97test::RecordedVectorTape tape = tapeWithLogits();
        r97::ObjectiveResult result;
        result.seeds = {ones("logits", 0, 3), ones("logits", 2, 4)};
        const std::string what = refuses(tape, result);
        assert(what.find("logits") != std::string::npos);
        assert(what.find("overlap") != std::string::npos);
    }

    // 4a. A differentiated read of a boundary that no seed answers.
    {
        r97test::RecordedVectorTape tape = tapeWithLogits();
        r97::ObjectiveResult result;
        result.seeds = {ones("logits", 0, 2)};
        tape.read("logits", 0, 1, true);   // answered by the seed
        tape.read("logits", 2, 4, true);   // read, but not answered
        const std::string what = refuses(tape, result);
        assert(what.find("logits") != std::string::npos);
        assert(what.find("has no seed") != std::string::npos);
    }

    // 4b. A seed on a boundary the objective never read as differentiated.
    {
        r97test::RecordedVectorTape tape = tapeWithLogits();
        r97::ObjectiveResult result;
        result.seeds = {ones("logits", 0, 2)};
        const std::string what = refuses(tape, result);
        assert(what.find("logits") != std::string::npos);
        assert(what.find("never read") != std::string::npos);
    }

    // Two seeds on two boundaries, both read, pass.
    {
        r97test::RecordedVectorTape tape = tapeWithLogits();
        tape.set_buffer("hidden", {5, 4}, std::vector<float>(20, 0.5f));
        r97::ObjectiveResult result;
        result.seeds = {ones("logits", 0, 1), ones("hidden", 0, 2)};
        tape.read("logits", 0, 1, true);
        tape.read("hidden", 0, 2, true);
        r97::check_seeds(tape, result);
    }

    // Two seeds that only touch on one buffer do not overlap: they are adjacent row ranges.
    {
        r97test::RecordedVectorTape tape = tapeWithLogits();
        r97::ObjectiveResult result;
        result.seeds = {ones("logits", 0, 2), ones("logits", 2, 4)};
        tape.read("logits", 0, 2, true);
        tape.read("logits", 2, 4, true);
        r97::check_seeds(tape, result);
    }

    // No seed at all and no differentiated read: a forward-only objective.
    {
        r97test::RecordedVectorTape tape = tapeWithLogits();
        tape.read("logits", 0, 1, false);
        r97::ObjectiveResult result;   // no seeds
        r97::check_seeds(tape, result);
    }

    // An objective that reads a boundary as differentiated and returns no seed is refused; the
    // same objective reading it detached passes.
    {
        const r97::Objective differentiated = r97test::make_differentiated_reader(r97::Config());
        r97test::RecordedVectorTape tape = tapeWithLogits();
        const r97::ObjectiveResult result = differentiated(tape, r97::ParamSet());
        assert(result.seeds.empty());
        const std::string what = refuses(tape, result);
        assert(what.find("logits") != std::string::npos);

        const r97::Objective detached = r97test::make_detached_reader(r97::Config());
        r97test::RecordedVectorTape bare = tapeWithLogits();
        r97::check_seeds(bare, detached(bare, r97::ParamSet()));
    }

    // BoundaryNames: a boundary's id is its position, its rows are what the tape recorded.
    {
        r97test::RecordedVectorTape tape = tapeWithLogits();
        tape.set_buffer("hidden", {5, 4}, std::vector<float>(20, 0.0f));
        tape.set_window("hidden", 1, 3);
        const r97::BoundaryNames names = r97::BoundaryNames::from(tape);
        assert((names.names() == std::vector<std::string>{"hidden", "logits"}));
        assert(names.has("hidden") && names.has("logits") && !names.has("embeddings"));
        assert(names.at("hidden").id == 0);
        assert(names.at("hidden").row_lo == 1 && names.at("hidden").row_hi == 3);
        assert(names.at("logits").id == 1);
        assert(names.at("logits").row_lo == 0 && names.at("logits").row_hi == 5);
        try {
            names.at("embeddings");
            assert(false && "an unknown boundary must be refused");
        } catch (const std::out_of_range&) {
        }
    }

    return 0;
}
