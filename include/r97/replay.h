// r97/replay.h -- the residual stream at every layer boundary, and one layer re-run from it.
//
// The forward of the backward keeps only the residual at the layer boundaries: boundary 0 is what
// the embeddings wrote, boundary `i` is block `i-1`'s output, and a trunk of L layers has L+1 of
// them. A layer replay copies one boundary back into the step's residual buffer, restores the
// state the first forward started from, and re-issues exactly that block's ops, so the backward
// can take one layer at a time without holding every activation.
//
// The replay is a training path: it reads the block brackets and the issue observer that
// patches/training/0101 adds to radiance, and the replay block that 0102 adds; an architecture
// must honour that block (qwen35_bf16 does). A serving build applies neither patch, so there is no
// replay there rather than one that would issue the whole model. A build that leaves 0102 out has
// the observer and the brackets but no replay, so the tests that need it skip by name.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "r97/tape.h"

namespace rad {
class Engine;
}  // namespace rad

namespace r97 {

// The residual at every layer boundary of one step, bf16 on the host. `tokens` is how many rows
// the step has (its sequences' tokens together), `hidden` the residual's width, so one boundary
// is tokens * hidden * 2 bytes. Layers replay from the same buffer the forward wrote; `source`
// keeps that pointer for each boundary.
class ResidualStore {
public:
    ResidualStore() = default;
    ResidualStore(int64_t tokens, int64_t hidden);

    // Copy one boundary out of the step's buffer at `data`. Boundaries are kept in order from 0;
    // a gap or a repeat throws std::out_of_range, and a null `data` std::invalid_argument.
    void keep(int64_t boundary, const void* data);

    bool has(int64_t boundary) const;
    int64_t boundaries() const { return (int64_t)values_.size(); }
    int64_t tokens() const { return tokens_; }
    int64_t hidden() const { return hidden_; }
    int64_t boundary_bytes() const { return tokens_ * hidden_ * 2; }
    int64_t bytes() const { return boundaries() * boundary_bytes(); }

    // The stored bytes of `boundary`, or null when it was not kept.
    const uint16_t* at(int64_t boundary) const;
    // The step buffer `boundary` was copied out of, or null when it was not kept.
    const void* source(int64_t boundary) const;

private:
    int64_t tokens_ = 0;
    int64_t hidden_ = 0;
    std::vector<std::vector<uint16_t>> values_;
    std::vector<const void*> sources_;
};

#if defined(RAD_ISSUE_OBSERVER) && defined(RAD_REPLAY_BLOCK)
#define R97_REPLAY 1

// Called after one op of a replayed block issued, before the next op can reuse its buffers. `op`
// is the forward's record of the same op and `operands` its content now; a caller that wants the
// bytes must copy them here. The residual the block leaves is the last written operand of its
// last op, exactly the next boundary.
using ReplaySnapshot =
    std::function<void(const OpRecord& op, const std::vector<OperandContent>& operands)>;

// Records the residual at each block boundary of a step as it runs, through the hooks of the tape
// that is recording the step: install it after the tape is attached, read store() after the step,
// and detach it before replaying a layer (the replay replaces those hooks). The architecture must
// bracket its blocks (rad_block_begin/end).
class BoundaryRecorder {
public:
    BoundaryRecorder(Tape& tape, int64_t tokens, int64_t hidden);
    ~BoundaryRecorder();

    BoundaryRecorder(const BoundaryRecorder&) = delete;
    BoundaryRecorder& operator=(const BoundaryRecorder&) = delete;

    // Stop observing. The store stays readable.
    void detach();

    const ResidualStore& store() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Re-runs one block of a recorded step from the stored residual of its input boundary. The tape's
// op records and block ranges are copied at construction and the tape is detached; replay()
// attaches that same tape again for the block, with this replay's hooks. The engine must outlive
// the replay and the step must be its last live prefill.
class LayerReplay {
public:
    LayerReplay(rad::Engine& engine, Tape& tape, ResidualStore store);
    ~LayerReplay();

    LayerReplay(const LayerReplay&) = delete;
    LayerReplay& operator=(const LayerReplay&) = delete;

    // How many blocks the recorded step has. A step of L layers has L blocks and L+1 boundaries.
    int64_t layers() const;

    const ResidualStore& store() const;

    // The step buffer a boundary's residual lives in, or null when the store lacks it. Restoring
    // boundary `layer` writes into this buffer, so it holds boundary `layer + 1` after the replay.
    const void* residual(int64_t boundary) const;

    // Restore boundary `layer` into the step buffer, clear the block's recurrent state, and issue
    // exactly block `layer`. `after` runs once per op of the block as it issues. Throws
    // std::out_of_range for an unknown block and std::runtime_error when the replay fails.
    void replay(int64_t layer, const ReplaySnapshot& after = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif  // RAD_ISSUE_OBSERVER && RAD_REPLAY_BLOCK

}  // namespace r97
