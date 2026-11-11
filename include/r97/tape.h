// r97/tape.h -- one step's forward graph, recorded from radiance's issue observer.
//
// The tape copies what the runtime reports for every rad_issue: the op's name, its resolved
// parameters, its operands and the block the architecture was in. It keeps a pointer to each
// operand's data, so a reader can look at what the op saw without a copy; the pointer is valid for
// as long as the buffer lives, which for the training engine is the step.
//
// Block ranges are recorded from the begin/end calls the architecture makes around a layer. Ops
// outside any block have block -1 and appear in ops() but in no range.
//
// The observer is a training patch on radiance (patches/training/0101-issue-observer.patch), and a
// serving build applies none of the training set. A runtime without the observer has no graph to
// read, so there is no tape there at all rather than one that records nothing.
#pragma once

#include "r97/op.h"
#include "rad_runtime.h"

#if defined(RAD_ISSUE_OBSERVER)

#include <cstddef>
#include <cstdint>
#include <vector>

namespace r97 {

// One operand's content at the issue: the slice the kernel saw, not the whole declared buffer.
// `bytes` is 0 when the slice is not one contiguous span.
struct OperandContent {
    const void* data = nullptr;
    size_t bytes = 0;
};

class Tape {
public:
    // A half-open range [begin, end) of ops() that one architecture block issued.
    struct Block {
        int64_t index = -1;  // the index the architecture named the block by
        size_t begin = 0;
        size_t end = 0;
    };

    Tape() = default;
    ~Tape();

    Tape(const Tape&) = delete;
    Tape& operator=(const Tape&) = delete;

    // Record `ctx`'s issues into this tape, replacing any observer already there. Attaching twice
    // detaches from the first context. The context must outlive the tape's attachment.
    void attach(RadCtx* ctx);
    void detach();

    // Drop the recorded step. The observer stays attached.
    void clear();

    const std::vector<OpRecord>& ops() const { return ops_; }
    const std::vector<Block>& blocks() const { return blocks_; }

    // The content of every operand of op `op`, in schema order. Out-of-range indices throw
    // std::out_of_range.
    const std::vector<OperandContent>& contents(size_t op) const;

    // The content of one operand. Out-of-range indices throw std::out_of_range.
    OperandContent content(size_t op, size_t operand) const;

private:
    static void observe_op(void* user, RadCtx* ctx, const RadIssueOp* issue);
    static void observe_block(void* user, RadCtx* ctx, int64_t block, int begin);

    void record(const RadIssueOp& issue);
    void open_block(int64_t block);
    void close_block(int64_t block);

    RadCtx* ctx_ = nullptr;
    int64_t block_ = -1;
    std::vector<OpRecord> ops_;
    std::vector<std::vector<OperandContent>> contents_;
    std::vector<Block> blocks_;
};

}  // namespace r97

#endif  // RAD_ISSUE_OBSERVER
