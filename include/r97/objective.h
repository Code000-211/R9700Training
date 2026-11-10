// r97/objective.h -- the loss, and where its gradient enters the tape.
//
// The objective owns the loss. It sees one recorded forward step (or a window of rows of it)
// through ForwardTape, and answers with ObjectiveResult: the loss with its seed(s) onto the
// tape's buffers, which the backward starts from.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "r97/batch.h"
#include "r97/params.h"

namespace r97 {

// A cotangent seed: the forward buffer a loss differentiates, and its cotangent.
//   buffer  a name the forward recorded on the tape (OpRecord carries buffer identity)
//   rows    the row range the seed covers (logits: the scored rows; ce: all)
struct Seed {
    std::string buffer;
    int64_t row_lo = 0, row_hi = 0;
    std::vector<float> cotangent;   // f32; the backward casts to the op's dtype
};

struct ObjectiveResult {
    double  loss   = 0.0;
    int64_t scored = 0;
    std::vector<Seed> seeds;   // one per boundary the loss touches; may be empty
};

// One recorded forward step as an objective sees it, or a window of rows of it. The forward
// driver hands it to the Objective; tests hand a stand-in (tests/support/vector_tape.h).
struct ForwardTape {
    virtual ~ForwardTape() = default;
    virtual const Batch& batch() const = 0;                      // ids and the score mask
    virtual int64_t total_scored() const = 0;                    // of the whole step, not of the window
    virtual std::vector<std::string> boundaries() const = 0;     // "logits" in the plain forward
    virtual std::vector<int64_t> shape(const std::string& buffer) const = 0;
    virtual void window(const std::string& buffer, int64_t* row_lo, int64_t* row_hi) const = 0;
    // Rows [row_lo, row_hi) as f32. `differentiated` says the loss depends on what it reads here:
    // such a read must be answered by a Seed on that buffer, a detached read must not be.
    virtual std::vector<float> read(const std::string& buffer, int64_t row_lo, int64_t row_hi,
                                    bool differentiated) const = 0;
};

// The objective owns the loss and where its gradient enters the tape. `fwd` gives read access to
// the recorded forward buffers (the tape) and to the named boundary tensors the driver exposes.
using Objective = std::function<ObjectiveResult(const ForwardTape& fwd, const ParamSet& deltas)>;

}  // namespace r97
