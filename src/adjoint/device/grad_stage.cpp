// src/adjoint/device/grad_stage.cpp -- the step's trainable gradients, waiting on the card.
//
// The accumulators come from radiance's device layer (rad_dev_alloc) rather than from the scratch
// allocator: scratch is a call's memory and the harness resets its ledger between calls, while this
// is the step's memory and outlives every call of the walk. A test that watches the stage therefore
// watches grad_stage_bytes() rather than the scratch ledger.
#include "grad_stage.h"

#include "rad_device.h"
#include "rad_plugin.h"

#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace r97 {
namespace device {
namespace {

// One leaf of the step: the f32 accumulator and what the ParamSet leaf of that name looks like.
struct Block {
    std::string name;
    std::vector<int64_t> shape;
    int64_t n = 0;
    float* p = nullptr;
};

struct Stage {
    std::mutex mu;
    std::vector<Block> blocks;
    int64_t bytes = 0;
};

Stage& stage() {
    static Stage s;
    return s;
}

// One stream for the stage's zeroing and for its copies out. A device row syncs its stream before
// it returns, so by the time the stage copies a block out the kernels that filled it are done and
// the two streams never race.
RadStream stage_stream() {
    static RadStream s = [] {
        RadStream created = nullptr;
        if (rad_stream_create(&created, 0) != 0) return (RadStream) nullptr;
        return created;
    }();
    return s;
}

int64_t elements(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) {
        if (d <= 0) return 0;
        n *= d;
    }
    return n;
}

int64_t f32_bytes(int64_t n) { return n * (int64_t)sizeof(float); }

Param* leaf_named(ParamSet* grads, const std::string& name) {
    for (Param& p : *grads)
        if (p.name == name) return &p;
    return nullptr;
}

// The step's f32 total, rounded once to the dtype a gradient has on the card and added to the leaf.
void add_rounded(Param* leaf, const std::vector<float>& f32) {
    for (size_t i = 0; i < f32.size(); ++i)
        leaf->value[i] += rad_bf16_to_f32(rad_f32_to_bf16(f32[i]));
}

// The step is over: every accumulator goes back to the card's pool.
void release(Stage& s) {
    for (Block& b : s.blocks) rad_dev_free(b.p, RAD_MEM_DEVICE);
    s.blocks.clear();
    s.bytes = 0;
}

}  // namespace

float* grad_stage_slot(const std::string& name, const std::vector<int64_t>& shape) {
    const int64_t n = elements(shape);
    if (n <= 0) return nullptr;

    Stage& s = stage();
    std::lock_guard<std::mutex> lock(s.mu);
    for (const Block& b : s.blocks) {
        if (b.name != name) continue;
        // A leaf whose shape moved inside one step is two different leaves; refuse it rather than
        // hand out a buffer the row would write past.
        return b.n == n && b.shape == shape ? b.p : nullptr;
    }

    RadStream stream = stage_stream();
    if (!stream) return nullptr;
    float* p = (float*)rad_dev_alloc(f32_bytes(n), RAD_MEM_DEVICE);
    if (!p) return nullptr;
    if (rad_memset_async(p, 0, f32_bytes(n), stream) != 0 || rad_stream_sync(stream) != 0) {
        rad_dev_free(p, RAD_MEM_DEVICE);
        return nullptr;
    }
    s.blocks.push_back(Block{name, shape, n, p});
    s.bytes += f32_bytes(n);
    return p;
}

int64_t grad_stage_bytes() {
    Stage& s = stage();
    std::lock_guard<std::mutex> lock(s.mu);
    return s.bytes;
}

bool grad_stage_flush(ParamSet* grads) {
    Stage& s = stage();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!grads) {
        release(s);
        return false;
    }

    RadStream stream = stage_stream();
    bool ok = stream != nullptr;
    std::vector<float> f32;
    for (const Block& b : s.blocks) {
        Param* leaf = leaf_named(grads, b.name);
        if (!leaf) {
            Param p;
            p.name = b.name;
            p.shape = b.shape;
            p.value.assign((size_t)b.n, 0.0f);
            grads->push_back(std::move(p));
            leaf = &grads->back();
        }
        if (leaf->shape != b.shape || (int64_t)leaf->value.size() != b.n) {
            ok = false;
            continue;
        }
        if (!stream) continue;
        f32.resize((size_t)b.n);
        if (rad_memcpy_async(f32.data(), b.p, f32_bytes(b.n), stream) != 0 ||
            rad_stream_sync(stream) != 0) {
            ok = false;
            continue;
        }
        add_rounded(leaf, f32);
    }
    release(s);
    return ok;
}

void grad_stage_discard() {
    Stage& s = stage();
    std::lock_guard<std::mutex> lock(s.mu);
    release(s);
}

}  // namespace device
}  // namespace r97
