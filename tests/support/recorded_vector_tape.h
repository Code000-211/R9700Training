// tests/support/recorded_vector_tape.h -- a VectorTape behind the RecordedTape interface.
//
// VectorTape is the plain in-memory tape the objective tests read; check_seeds audits what an
// objective read, and the audit reaches that record through r97::RecordedTape. This wrapper adds
// that interface without changing VectorTape itself: the reads are recorded here as they pass
// through.
#pragma once

#include "r97/model.h"

#include "vector_tape.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace r97test {

class RecordedVectorTape : public r97::RecordedTape {
public:
    // ------------------------------------------------------------ construction
    RecordedVectorTape& set_batch(r97::Batch b) {
        tape_.set_batch(std::move(b));
        return *this;
    }
    RecordedVectorTape& set_scored(int64_t scored) {
        tape_.set_scored(scored);
        return *this;
    }
    RecordedVectorTape& set_buffer(const std::string& name, std::vector<int64_t> shape,
                                   std::vector<float> value) {
        tape_.set_buffer(name, std::move(shape), std::move(value));
        return *this;
    }
    RecordedVectorTape& set_window(const std::string& name, int64_t lo, int64_t hi) {
        tape_.set_window(name, lo, hi);
        return *this;
    }

    // ------------------------------------------------------------ ForwardTape
    const r97::Batch& batch() const override { return tape_.batch(); }
    int64_t total_scored() const override { return tape_.total_scored(); }
    std::vector<std::string> boundaries() const override { return tape_.boundaries(); }
    std::vector<int64_t> shape(const std::string& buffer) const override {
        return tape_.shape(buffer);
    }
    void window(const std::string& buffer, int64_t* row_lo, int64_t* row_hi) const override {
        tape_.window(buffer, row_lo, row_hi);
    }
    std::vector<float> read(const std::string& buffer, int64_t row_lo, int64_t row_hi,
                            bool differentiated) const override {
        record_.push_back(r97::TapeRead{buffer, row_lo, row_hi, differentiated});
        return tape_.read(buffer, row_lo, row_hi, differentiated);
    }

    // ------------------------------------------------------------ the record
    const std::vector<r97::TapeRead>& reads() const override { return record_; }
    std::vector<r97::TapeRead> reads_of(const std::string& buffer) const {
        std::vector<r97::TapeRead> out;
        for (const r97::TapeRead& r : record_)
            if (r.buffer == buffer) out.push_back(r);
        return out;
    }

private:
    VectorTape tape_;
    mutable std::vector<r97::TapeRead> record_;
};

}  // namespace r97test
