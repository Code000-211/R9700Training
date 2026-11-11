// src/adjoint/device/scratch.cpp -- the scratch allocator's ledger.
//
// The allocations themselves come from radiance's device layer (rad_dev_alloc), so they are
// charged against the same VRAM pool as the engine's, and a test that watches that pool sees a
// device adjoint's scratch appear and go away.
#include "scratch.h"

#include "rad_device.h"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace r97 {
namespace device {
namespace {

struct Ledger {
    std::mutex mu;
    std::unordered_map<void*, int64_t> live;
    ScratchStats stats;
};

Ledger& ledger() {
    static Ledger l;
    return l;
}

}  // namespace

ScratchStats scratch_stats() {
    Ledger& l = ledger();
    std::lock_guard<std::mutex> lock(l.mu);
    return l.stats;
}

void scratch_reset() {
    Ledger& l = ledger();
    std::lock_guard<std::mutex> lock(l.mu);
    if (l.stats.live_bytes != 0)
        throw std::runtime_error("device scratch: " + std::to_string(l.stats.live_bytes)
                                 + " bytes were never freed; the kernel that held them leaked");
    l.stats = ScratchStats{};
}

void* scratch_alloc(int64_t bytes) {
    if (bytes <= 0) return nullptr;
    void* p = rad_dev_alloc(bytes, RAD_MEM_DEVICE);
    if (!p) return nullptr;

    Ledger& l = ledger();
    std::lock_guard<std::mutex> lock(l.mu);
    l.live[p] = bytes;
    l.stats.live_bytes += bytes;
    l.stats.peak_bytes = std::max(l.stats.peak_bytes, l.stats.live_bytes);
    ++l.stats.allocations;
    return p;
}

void scratch_free(void* p) {
    if (!p) return;

    int64_t bytes = 0;
    {
        Ledger& l = ledger();
        std::lock_guard<std::mutex> lock(l.mu);
        auto it = l.live.find(p);
        if (it == l.live.end()) return;
        bytes = it->second;
        l.live.erase(it);
        l.stats.live_bytes -= bytes;
    }
    rad_dev_free(p, RAD_MEM_DEVICE);
}

}  // namespace device
}  // namespace r97
