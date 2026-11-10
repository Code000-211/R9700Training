// tests/support/vector_tape.h -- a ForwardTape over named f32 arrays held in memory.
//
// Header only, so the objective tests and the seed-check need no model. Every read is recorded,
// (buffer, row range, differentiated or not), so a test can assert what the objective read and
// which reads it claimed as differentiated.
#pragma once

#include "r97/objective.h"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace r97test {

struct TapeRead {
    std::string buffer;
    int64_t row_lo = 0, row_hi = 0;
    bool differentiated = false;
};

class VectorTape : public r97::ForwardTape {
public:
    // ------------------------------------------------------------ construction
    VectorTape& set_batch(r97::Batch b) {
        batch_ = std::move(b);
        return *this;
    }
    VectorTape& set_scored(int64_t scored) {
        scored_ = scored;
        return *this;
    }
    // A named boundary buffer, all rows, shaped [rows, cols]; rows count from the shape.
    VectorTape& set_buffer(const std::string& name, std::vector<int64_t> shape,
                           std::vector<float> value) {
        buffers_[name] = std::move(value);
        shapes_[name] = std::move(shape);
        windows_[name] = {0, rows_of(name)};
        return *this;
    }
    // Restrict what a buffer exposes to [lo, hi): the objective sees a window of the step.
    VectorTape& set_window(const std::string& name, int64_t lo, int64_t hi) {
        windows_[name] = {lo, hi};
        return *this;
    }

    // ------------------------------------------------------------ ForwardTape
    const r97::Batch& batch() const override { return batch_; }
    int64_t total_scored() const override { return scored_; }

    std::vector<std::string> boundaries() const override {
        std::vector<std::string> names;
        for (const auto& [name, _] : buffers_) names.push_back(name);
        return names;
    }

    std::vector<int64_t> shape(const std::string& buffer) const override {
        auto it = shapes_.find(buffer);
        if (it == shapes_.end()) throw std::out_of_range("vector_tape: no buffer " + buffer);
        return it->second;
    }

    void window(const std::string& buffer, int64_t* row_lo, int64_t* row_hi) const override {
        auto it = windows_.find(buffer);
        if (it == windows_.end()) throw std::out_of_range("vector_tape: no buffer " + buffer);
        *row_lo = it->second.first;
        *row_hi = it->second.second;
    }

    std::vector<float> read(const std::string& buffer, int64_t row_lo, int64_t row_hi,
                            bool differentiated) const override {
        record_.push_back(TapeRead{buffer, row_lo, row_hi, differentiated});
        auto it = buffers_.find(buffer);
        if (it == buffers_.end()) throw std::out_of_range("vector_tape: no buffer " + buffer);
        const int64_t cols = cols_of(shapes_.at(buffer));
        const int64_t lo = row_lo * cols;
        return std::vector<float>(it->second.begin() + lo, it->second.begin() + row_hi * cols);
    }

    // ------------------------------------------------------------ the record
    const std::vector<TapeRead>& reads() const { return record_; }
    // All reads of one buffer, in order.
    std::vector<TapeRead> reads_of(const std::string& buffer) const {
        std::vector<TapeRead> out;
        for (const auto& r : record_)
            if (r.buffer == buffer) out.push_back(r);
        return out;
    }

private:
    static int64_t cols_of(const std::vector<int64_t>& shape) {
        int64_t n = 1;
        for (size_t i = 1; i < shape.size(); ++i) n *= shape[i];
        return n;
    }
    int64_t rows_of(const std::string& name) const { return shapes_.at(name)[0]; }

    r97::Batch batch_;
    int64_t scored_ = 0;
    std::map<std::string, std::vector<float>> buffers_;
    std::map<std::string, std::vector<int64_t>> shapes_;
    std::map<std::string, std::pair<int64_t, int64_t>> windows_;
    mutable std::vector<TapeRead> record_;
};

}  // namespace r97test
