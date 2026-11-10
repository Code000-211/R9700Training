// tests/host/seed_check.cpp -- the refusals of check_seeds fire and name the buffer, a result
// with two seeds on two boundaries passes, a result with no seed passes, and BoundaryNames maps a
// boundary's name to the id and rows the tape recorded (label: host).
#include "r97/model.h"

#include "../support/reader_objective.h"
#include "../support/recorded_vector_tape.h"
#include "../support/test.h"

#include <exception>
#include <string>
#include <vector>

using r97::Seed;

namespace {

r97test::RecordedVectorTape tape_with_logits() {
    r97test::RecordedVectorTape tape;
    tape.set_buffer("logits", {5, 4}, std::vector<float>(20, 1.0f));   // rows [0, 5), 4 columns
    return tape;
}

std::string refuses(const r97test::RecordedVectorTape& tape,
                    const r97::ObjectiveResult& result) {
    try {
        r97::check_seeds(tape, result);
    } catch (const std::exception& e) {
        return e.what();
    }
    R97_CHECK(false && "check_seeds must refuse this result");
    return "";
}

Seed ones(const char* buffer, int64_t lo, int64_t hi) {
    return Seed{buffer, lo, hi,
                std::vector<float>(static_cast<size_t>((hi - lo) * 4), 1.0f)};
}

}  // namespace

// Called by tests/host/test_registrations.cpp, which places the test entry after every static
// registration the test binary needs.
void run_seed_check() {
    // 1. A seed on a buffer that is not a boundary of the tape.
    {
        r97test::RecordedVectorTape tape = tape_with_logits();
        r97::ObjectiveResult result;
        result.seeds = {ones("embeddings", 0, 1)};
        const std::string what = refuses(tape, result);
        R97_CHECK(what.find("embeddings") != std::string::npos);
        R97_CHECK(what.find("not a boundary") != std::string::npos);
    }

    // 2a. A seed whose row range lies outside the window.
    {
        r97test::RecordedVectorTape tape = tape_with_logits();
        r97::ObjectiveResult result;
        result.seeds = {ones("logits", 4, 7)};
        const std::string what = refuses(tape, result);
        R97_CHECK(what.find("logits") != std::string::npos);
        R97_CHECK(what.find("outside the window") != std::string::npos);
    }

    // 2b. The same for a window that is narrower than the buffer.
    {
        r97test::RecordedVectorTape tape = tape_with_logits();
        tape.set_window("logits", 1, 4);
        r97::ObjectiveResult result;
        result.seeds = {ones("logits", 0, 2)};
        const std::string what = refuses(tape, result);
        R97_CHECK(what.find("logits") != std::string::npos);
        R97_CHECK(what.find("outside the window") != std::string::npos);
    }

    // 2c. A seed whose cotangent does not have the size of its rows.
    {
        r97test::RecordedVectorTape tape = tape_with_logits();
        r97::ObjectiveResult result;
        result.seeds = {Seed{"logits", 0, 2, std::vector<float>(4, 1.0f)}};   // 2 rows need 8
        const std::string what = refuses(tape, result);
        R97_CHECK(what.find("logits") != std::string::npos);
        R97_CHECK(what.find("cotangent") != std::string::npos);
    }

    // 3. Two seeds of one result that overlap on a buffer.
    {
        r97test::RecordedVectorTape tape = tape_with_logits();
        r97::ObjectiveResult result;
        result.seeds = {ones("logits", 0, 3), ones("logits", 2, 4)};
        const std::string what = refuses(tape, result);
        R97_CHECK(what.find("logits") != std::string::npos);
        R97_CHECK(what.find("overlap") != std::string::npos);
    }

    // 4a. A differentiated read of a boundary that no seed answers.
    {
        r97test::RecordedVectorTape tape = tape_with_logits();
        r97::ObjectiveResult result;
        result.seeds = {ones("logits", 0, 2)};
        tape.read("logits", 0, 1, true);   // answered by the seed
        tape.read("logits", 2, 4, true);   // read, but not answered
        const std::string what = refuses(tape, result);
        R97_CHECK(what.find("logits") != std::string::npos);
        R97_CHECK(what.find("has no seed") != std::string::npos);
    }

    // 4b. A seed on a boundary the objective never read as differentiated.
    {
        r97test::RecordedVectorTape tape = tape_with_logits();
        r97::ObjectiveResult result;
        result.seeds = {ones("logits", 0, 2)};
        const std::string what = refuses(tape, result);
        R97_CHECK(what.find("logits") != std::string::npos);
        R97_CHECK(what.find("never read") != std::string::npos);
    }

    // Two seeds on two boundaries, both read, pass.
    {
        r97test::RecordedVectorTape tape = tape_with_logits();
        tape.set_buffer("hidden", {5, 4}, std::vector<float>(20, 0.5f));
        r97::ObjectiveResult result;
        result.seeds = {ones("logits", 0, 1), ones("hidden", 0, 2)};
        tape.read("logits", 0, 1, true);
        tape.read("hidden", 0, 2, true);
        r97::check_seeds(tape, result);
    }

    // Two seeds that only touch on one buffer do not overlap: they are adjacent row ranges.
    {
        r97test::RecordedVectorTape tape = tape_with_logits();
        r97::ObjectiveResult result;
        result.seeds = {ones("logits", 0, 2), ones("logits", 2, 4)};
        tape.read("logits", 0, 2, true);
        tape.read("logits", 2, 4, true);
        r97::check_seeds(tape, result);
    }

    // No seed at all and no differentiated read: a forward-only objective.
    {
        r97test::RecordedVectorTape tape = tape_with_logits();
        tape.read("logits", 0, 1, false);
        r97::ObjectiveResult result;   // no seeds
        r97::check_seeds(tape, result);
    }

    // An objective that reads a boundary as differentiated and returns no seed is refused; the
    // same objective reading it detached passes.
    {
        const r97::Objective differentiated = r97test::make_differentiated_reader(r97::Config());
        r97test::RecordedVectorTape tape = tape_with_logits();
        const r97::ObjectiveResult result = differentiated(tape, r97::ParamSet());
        R97_CHECK(result.seeds.empty());
        const std::string what = refuses(tape, result);
        R97_CHECK(what.find("logits") != std::string::npos);

        const r97::Objective detached = r97test::make_detached_reader(r97::Config());
        r97test::RecordedVectorTape bare = tape_with_logits();
        r97::check_seeds(bare, detached(bare, r97::ParamSet()));
    }

    // BoundaryNames: a boundary's id is its position, its rows are what the tape recorded.
    {
        r97test::RecordedVectorTape tape = tape_with_logits();
        tape.set_buffer("hidden", {5, 4}, std::vector<float>(20, 0.0f));
        tape.set_window("hidden", 1, 3);
        const r97::BoundaryNames names = r97::BoundaryNames::from(tape);
        R97_CHECK((names.names() == std::vector<std::string>{"hidden", "logits"}));
        R97_CHECK(names.has("hidden") && names.has("logits") && !names.has("embeddings"));
        R97_CHECK(names.at("hidden").id == 0);
        R97_CHECK(names.at("hidden").row_lo == 1 && names.at("hidden").row_hi == 3);
        R97_CHECK(names.at("logits").id == 1);
        R97_CHECK(names.at("logits").row_lo == 0 && names.at("logits").row_hi == 5);
        bool refused = false;
        try {
            (void)names.at("embeddings");
        } catch (const std::out_of_range&) {
            refused = true;
        }
        R97_CHECK(refused);
    }
}
