// src/head.cpp -- the head's chunk loop, on the host side of the card.
//
// include/r97/head.h is the contract; docs/kernels/head.md is the account. This file walks a
// window a chunk at a time (head_ce_loop), and states the loss as an Objective over a ForwardTape
// (HeadCeObjective) so a check can hold it to the reference. The chunk's arithmetic lives in the
// plugin (src/adjoint/device/head.hip); here a chunk's buffers are scratch, its rows are uploaded
// and its losses come back -- the cotangent only when the caller asked for it, which is what the
// reference comparison does and what a device join does not.
#include "r97/head.h"

#include "adjoint/device/head_row.h"
#include "adjoint/device/scratch.h"

#include "r97/loss.h"
#include "r97/params.h"

#include "rad_device.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace r97 {
namespace {

// ---------------------------------------------------------------- the kernel's registration

struct Registry {
    std::mutex mu;
    HeadCeKernel kernel = nullptr;
};

Registry& registry() {
    static Registry r;
    return r;
}

// ------------------------------------------------------------------------- one chunk's buffers

// One chunk's device buffers, freed when the walk leaves -- by the end of the window, by a kernel
// failure, or by a producer that throws.
struct ChunkBuffers {
    void* logits = nullptr;
    void* grad = nullptr;
    float* loss = nullptr;
    int32_t* ids = nullptr;

    ChunkBuffers(int64_t rows, int64_t vocab) {
        const int64_t row_bytes = rows * vocab * int64_t(sizeof(uint16_t));
        logits = device::scratch_alloc(row_bytes);
        grad = device::scratch_alloc(row_bytes);
        loss = static_cast<float*>(device::scratch_alloc(rows * int64_t(sizeof(float))));
        ids = static_cast<int32_t*>(device::scratch_alloc(rows * int64_t(sizeof(int32_t))));
        if (!logits || !grad || !loss || !ids) {
            release();
            throw std::runtime_error("head ce: a chunk of " + std::to_string(rows) + " x "
                                     + std::to_string(vocab) + " rows did not fit in the device's "
                                     "memory");
        }
    }
    ChunkBuffers(const ChunkBuffers&) = delete;
    ChunkBuffers& operator=(const ChunkBuffers&) = delete;
    ~ChunkBuffers() { release(); }

private:
    void release() {
        device::scratch_free(ids);
        device::scratch_free(loss);
        device::scratch_free(grad);
        device::scratch_free(logits);
        logits = grad = nullptr;
        loss = nullptr;
        ids = nullptr;
    }
};

// A stream the walk owns, so the card's work for one window is one queue, and a sink queues its
// own work behind the chunk's kernel.
struct WalkStream {
    RadStream stream = nullptr;
    WalkStream() {
        if (rad_stream_create(&stream, 0) != 0 || !stream)
            throw std::runtime_error(std::string("head ce: no device stream: ") + rad_dev_last_error());
    }
    WalkStream(const WalkStream&) = delete;
    WalkStream& operator=(const WalkStream&) = delete;
    ~WalkStream() { rad_stream_destroy(stream); }
};

void check(int rc, const char* what) {
    if (rc != 0) throw std::runtime_error(std::string("head ce: ") + what + ": " + rad_dev_last_error());
}

}  // namespace

void head_ce_register(HeadCeKernel fn) {
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.mu);
    r.kernel = fn;
}

HeadCeKernel head_ce_find() {
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.mu);
    return r.kernel;
}

HeadCeSum head_ce_loop(const HeadCeWindow& window, const HeadChunkFill& fill,
                       const HeadGradSink& sink) {
    if (!fill) throw std::invalid_argument("head ce: the loop has no fill");
    if (window.rows < 0) throw std::invalid_argument("head ce: a window has no negative rows");
    if (window.vocab <= 0) throw std::invalid_argument("head ce: a window's rows are not empty");
    if (window.chunk <= 0) throw std::invalid_argument("head ce: a chunk holds at least one row");
    if (window.rows == 0) return HeadCeSum{};

    HeadCeKernel kernel = head_ce_find();
    if (!kernel)
        throw std::runtime_error(
            "head ce: no device chunk kernel in this process; libr97 registers it when it loads "
            "(R97_DEVICE selects the card, tests/support/device_check.h loads it)");

    const int64_t capacity = std::min(window.chunk, window.rows);
    ChunkBuffers buffers(capacity, window.vocab);
    WalkStream walk;

    HeadCeSum total;
    std::vector<float> row_loss(static_cast<size_t>(capacity));
    int64_t done = 0;
    while (done < window.rows) {
        HeadChunk chunk;
        chunk.logits = buffers.logits;
        chunk.grad = buffers.grad;
        chunk.loss = buffers.loss;
        chunk.ids = buffers.ids;
        chunk.positions = capacity;
        chunk.vocab = window.vocab;
        chunk.grad_scale = static_cast<float>(window.grad_scale);
        chunk.stream = walk.stream;

        const HeadChunkRows rows = fill(chunk);
        if (rows.rows <= 0) break;
        if (rows.rows > capacity || done + rows.rows > window.rows)
            throw std::runtime_error("head ce: a fill wrote " + std::to_string(rows.rows)
                                     + " rows into a chunk of " + std::to_string(capacity)
                                     + " with " + std::to_string(window.rows - done)
                                     + " rows of the window left");
        if (static_cast<int64_t>(rows.ids.size()) != rows.rows)
            throw std::runtime_error("head ce: a fill wrote " + std::to_string(rows.rows)
                                     + " rows and " + std::to_string(rows.ids.size()) + " ids");
        chunk.positions = rows.rows;

        check(rad_memcpy_async(buffers.ids, rows.ids.data(),
                               rows.rows * int64_t(sizeof(int32_t)), walk.stream),
              "uploading a chunk's ids");
        const char* error = nullptr;
        if (!kernel(chunk, &error))
            throw std::runtime_error(std::string("head ce: the chunk's kernel refused it: ")
                                     + (error ? error : "no reason given"));
        check(rad_memcpy_async(row_loss.data(), buffers.loss,
                               rows.rows * int64_t(sizeof(float)), walk.stream),
              "downloading a chunk's losses");
        check(rad_stream_sync(walk.stream), "waiting for a chunk");

        for (int64_t i = 0; i < rows.rows; ++i) {
            if (rows.ids[static_cast<size_t>(i)] < 0) continue;
            total.loss += row_loss[static_cast<size_t>(i)];
            ++total.scored;
        }
        if (sink) sink(chunk);
        done += rows.rows;
    }
    return total;
}

Objective HeadCeObjective(int64_t chunk) {
    if (chunk <= 0) throw std::invalid_argument("head ce: a chunk holds at least one row");
    return [chunk](const ForwardTape& fwd, const ParamSet&) -> ObjectiveResult {
        ObjectiveResult out;
        const std::vector<int64_t> shape = fwd.shape("logits");
        if (shape.size() != 2)
            throw std::runtime_error("head ce: logits is a [rows, vocab] buffer, not a "
                                     + std::to_string(shape.size()) + "-d one");
        const int64_t vocab = shape[1];
        if (vocab <= 0) throw std::runtime_error("head ce: logits has an empty row");

        int64_t lo = 0, hi = 0;
        fwd.window("logits", &lo, &hi);
        if (hi <= lo) return out;

        const std::vector<ScoredRow> scored = ce_grad_rows(fwd.batch(), lo, hi);
        if (scored.empty()) return out;
        const int64_t span = hi - lo;
        const std::vector<float> logits = fwd.read("logits", lo, hi, /*differentiated=*/true);

        std::vector<int32_t> row_ids(static_cast<size_t>(span), -1);
        for (const ScoredRow& r : scored) row_ids[static_cast<size_t>(r.row - lo)] = r.id;

        // The gradient is the mean over the step's scored tokens; the loss stays the sum.
        const int64_t step_scored = fwd.total_scored();
        const double scale = step_scored > 0 ? 1.0 / static_cast<double>(step_scored) : 0.0;

        std::vector<float> cotangent(logits.size(), 0.0f);
        std::vector<uint16_t> staging;  // one chunk of the same rows, as bf16 on the host
        int64_t cursor = 0;             // the row this chunk starts at

        HeadCeWindow window;
        window.rows = span;
        window.vocab = vocab;
        window.chunk = chunk;
        window.grad_scale = scale;

        const HeadChunkFill fill = [&](HeadChunk& c) {
            HeadChunkRows rows;
            const int64_t n = std::min(c.positions, span - cursor);
            if (n <= 0) return rows;
            const int64_t first = cursor * vocab;
            staging.resize(static_cast<size_t>(n * vocab));
            for (int64_t i = 0; i < n * vocab; ++i)
                staging[static_cast<size_t>(i)] =
                    head_bf16_from_f32(logits[static_cast<size_t>(first + i)]);
            check(rad_memcpy_async(c.logits, staging.data(), n * vocab * int64_t(sizeof(uint16_t)),
                                   static_cast<RadStream>(c.stream)),
                  "uploading a chunk's logits");
            rows.rows = n;
            rows.ids.assign(row_ids.begin() + cursor, row_ids.begin() + cursor + n);
            return rows;
        };

        const HeadGradSink sink = [&](const HeadChunk& c) {
            const int64_t n = c.positions;
            staging.resize(static_cast<size_t>(n * vocab));
            check(rad_memcpy_async(staging.data(), c.grad, n * vocab * int64_t(sizeof(uint16_t)),
                                   static_cast<RadStream>(c.stream)),
                  "downloading a chunk's gradient");
            check(rad_stream_sync(static_cast<RadStream>(c.stream)),
                  "waiting for a chunk's gradient");
            const int64_t first = cursor * vocab;
            for (int64_t i = 0; i < n * vocab; ++i)
                cotangent[static_cast<size_t>(first + i)] =
                    head_bf16_to_f32(staging[static_cast<size_t>(i)]);
            cursor += n;
        };

        const HeadCeSum sum = head_ce_loop(window, fill, sink);
        if (sum.scored == 0) return out;

        out.loss = sum.loss;
        out.scored = sum.scored;

        // The seed covers the rows the window scores, from its first to its last: the rows between
        // them may be unscored (their cotangent is zero, as ce_sum leaves it), and the rows outside
        // are not this window's to answer for.
        Seed seed;
        seed.buffer = "logits";
        seed.row_lo = scored.front().row;
        seed.row_hi = scored.back().row + 1;
        const int64_t seed_rows = seed.row_hi - seed.row_lo;
        seed.cotangent.assign(cotangent.begin() + (seed.row_lo - lo) * vocab,
                              cotangent.begin() + (seed.row_lo - lo + seed_rows) * vocab);
        out.seeds.push_back(std::move(seed));
        return out;
    };
}

}  // namespace r97
