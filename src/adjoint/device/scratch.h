// src/adjoint/device/scratch.h -- device scratch for kernels, with the peak a test reports.
//
// A device adjoint receives f64 operands from the tape and has no device buffers of its own, so
// every allocation it makes for one call is scratch: uploads, the output, a reduction's staging.
// This allocator hands those out from radiance's device layer and keeps the high-water mark, so a
// test can say how much a kernel needed and a kernel author can compare two spellings of the same
// kernel. The peak is process-wide, across every kernel and every call since the last reset.
#pragma once

#include <cstdint>

namespace r97 {
namespace device {

struct ScratchStats {
    int64_t live_bytes = 0;   // held now: zero between calls, or a kernel leaked
    int64_t peak_bytes = 0;   // the high-water mark of live_bytes since the last reset
    int64_t allocations = 0;  // calls to scratch_alloc since the last reset
};

ScratchStats scratch_stats();

// Zeroes the counters. Throws std::runtime_error when a kernel still holds scratch: folding two
// runs' peaks together would hide the leak, and the live bytes are the only trace of it.
void scratch_reset();

// Device memory for the duration of one adjoint call. Non-zero sizes are 256-byte aligned, the
// alignment radiance's device allocations give. Null when the device refuses the allocation;
// rad_dev_last_error() names why. A null request and a failed one return null.
void* scratch_alloc(int64_t bytes);

// Returns a block from scratch_alloc. An unknown pointer is ignored: a kernel that has already
// failed has nothing better to do than finish unwinding.
void scratch_free(void* p);

// One call's buffer, freed on every return path, including the early ones a failed allocation or
// copy takes. One block owns one allocation, so it neither copies nor moves.
struct ScratchBlock {
    void* p = nullptr;
    explicit ScratchBlock(int64_t bytes) : p(scratch_alloc(bytes)) {}
    ~ScratchBlock() { scratch_free(p); }
    ScratchBlock(const ScratchBlock&) = delete;
    ScratchBlock& operator=(const ScratchBlock&) = delete;
    ScratchBlock(ScratchBlock&&) = delete;
    ScratchBlock& operator=(ScratchBlock&&) = delete;
    explicit operator bool() const { return p != nullptr; }
};

}  // namespace device
}  // namespace r97
