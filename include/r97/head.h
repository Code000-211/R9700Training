// r97/head.h -- the head's chunk loop: the loss over the vocabulary, a chunk of rows at a time.
//
// A logit row is `vocab` values wide -- 248320 of them for the real container, 496 KiB of bf16 --
// so neither the head's logits nor their gradient is held whole. The head product writes a chunk
// of rows, the loss turns that chunk into its rows' losses and its cotangent, and the transposed
// head product consumes the cotangent chunk before the next one is produced: `head_ce_loop` is
// that walk, and `head_ce_register` is where the card's kernel for one chunk comes from
// (docs/kernels/head.md).
//
// HeadCeObjective is the same loss as an Objective (r97/objective.h): called once per window of
// the tape, it answers with that window's loss, its scored count and its seed on "logits", and it
// agrees with the reference CeObjective (r97/loss.h, docs/losses/ce-objective.md). The cotangent
// it answers with is f32 on the host, the contract's vector -- the checked form. A device join
// needs no vector: it fills and sinks the chunk loop itself and hands the gradient buffer on while
// it is still on the card.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "r97/objective.h"

namespace r97 {

// Rows per chunk when the caller does not choose: 64 rows of the real vocabulary are 31.8 MB of
// bf16 logits, and as much again for their gradient.
inline constexpr int64_t kHeadChunkRows = 64;

// One chunk of a window's rows, on the card. Row i is the `vocab` bf16 logits at
// `logits + i * vocab`; `ids[i]` is the row's target id, or -1 where the row is not scored. The
// kernel writes the row's loss to `loss[i]`, and the row's gradient to `grad + i * vocab`:
// `(softmax(row) - onehot(ids[i])) * grad_scale`, and zeros on a row that is not scored. `grad`
// and `loss` may be null: a kernel then produces only what it was given room for.
struct HeadChunk {
    void* logits = nullptr;    // device bf16, positions * vocab: in, the producer writes it
    void* grad = nullptr;      // device bf16, positions * vocab: out
    float* loss = nullptr;     // device f32, positions: out, one per row
    int32_t* ids = nullptr;    // device i32, positions: in, -1 where the row is not scored
    int64_t positions = 0;     // rows in this chunk
    int64_t vocab = 0;         // values per row
    float grad_scale = 1.0f;   // 1 / the step's scored tokens: the gradient is the mean
    void* stream = nullptr;    // the RadStream to queue on; the kernel's own when null
};

// The chunk kernel, as libr97 registers it when the plugin loads: it queues the chunk's work on
// `chunk.stream` and returns true, or returns false and leaves `*error` on a message that outlives
// the call. It does not wait; a caller that reads `loss` or hands `grad` to another stream
// synchronises first.
using HeadCeKernel = bool (*)(const HeadChunk& chunk, const char** error);

// libr97 registers its kernel as the plugin loads; a process that links r97 exports this for it to
// find, and a process without the plugin has none -- head_ce_find() then returns null, as a
// process without the plugin has no device adjoints.
void head_ce_register(HeadCeKernel fn);
HeadCeKernel head_ce_find();

// What one pass over a window produced: its rows' losses summed in f64 from their f32 values, and
// how many of its rows were scored.
struct HeadCeSum {
    double loss = 0.0;
    int64_t scored = 0;
};

// A chunk's producer answers with the rows it wrote into `chunk.logits` and their targets on the
// host -- -1 where the row is not scored; the loop puts them on the card. 0 rows ends the window,
// and a chunk never holds more rows than `chunk.positions`.
struct HeadChunkRows {
    int64_t rows = 0;
    std::vector<int32_t> ids;
};

// The loop's two ends: `fill` produces a chunk, and `sink` takes that chunk once its gradient is
// complete on the card, before the next `fill`.
using HeadChunkFill = std::function<HeadChunkRows(HeadChunk& chunk)>;
using HeadGradSink = std::function<void(const HeadChunk& chunk)>;

// One window of the head: `rows` logit rows of `vocab`, `chunk` of them at a time on the card,
// their gradient scaled by `grad_scale` (1 / the step's scored tokens).
struct HeadCeWindow {
    int64_t rows = 0;
    int64_t vocab = 0;
    int64_t chunk = kHeadChunkRows;
    double grad_scale = 1.0;
};

// Walks a window: one chunk's buffers -- logits, gradient, ids and a row of losses each -- from
// the device scratch, filled, run, summed and handed to `sink`, and freed however the walk ends.
// Throws when the build has no device kernel or the kernel fails; `sink` may be empty, a pass that
// produces the loss and hands no gradient on.
HeadCeSum head_ce_loop(const HeadCeWindow& window, const HeadChunkFill& fill,
                       const HeadGradSink& sink = HeadGradSink());

// The device loss as an Objective: called once per window of the tape, it reads that window's
// logits and ids, runs them through the chunk loop `chunk` rows at a time, and answers with the
// window's summed loss, its scored count and its seed on "logits" -- the same result as
// CeObjective, produced on the card. The seed's cotangent is f32 on the host: the contract's
// vector, which is what a check compares, and what a device join does not spend.
Objective HeadCeObjective(int64_t chunk = kHeadChunkRows);

}  // namespace r97
