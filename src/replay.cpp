// r97/replay.cpp -- the residual store, the boundary recorder and the layer replay.
//
// The store is host memory and needs no radiance; the recorder and the replay need the block
// brackets and the post-issue observer that patches/training/0101 and 0102 add, so a serving build
// compiles them away the way src/tape.cpp does.
#include "r97/replay.h"

#include <cstring>
#include <stdexcept>

namespace r97 {

ResidualStore::ResidualStore(int64_t tokens, int64_t hidden)
    : tokens_(tokens), hidden_(hidden) {
    if (tokens <= 0 || hidden <= 0)
        throw std::invalid_argument("residual store needs positive tokens and hidden");
}

void ResidualStore::keep(int64_t boundary, const void* data) {
    if (!data) throw std::invalid_argument("residual store needs bytes to keep");
    if (boundary < 0 || boundary != boundaries())
        throw std::out_of_range("residual store keeps boundaries in order from zero");
    values_.emplace_back((size_t)(tokens_ * hidden_));
    std::memcpy(values_.back().data(), data, (size_t)boundary_bytes());
    sources_.push_back(data);
}

bool ResidualStore::has(int64_t boundary) const {
    return boundary >= 0 && boundary < boundaries();
}

const uint16_t* ResidualStore::at(int64_t boundary) const {
    return has(boundary) ? values_[(size_t)boundary].data() : nullptr;
}

const void* ResidualStore::source(int64_t boundary) const {
    return has(boundary) ? sources_[(size_t)boundary] : nullptr;
}

}  // namespace r97

#if defined(RAD_ISSUE_OBSERVER) && defined(RAD_REPLAY_BLOCK)

#include <string>
#include <utility>

#include "engine.h"

namespace r97 {
namespace {

// The residual is one operand: bf16, two dimensions, `tokens * hidden * 2` bytes, written. The
// first one seen names the buffer (the embeddings' output); later operands must carry that name,
// so a norm or a tap of the same byte count is not mistaken for the residual.
bool residual_bytes(const RadIssueOperand& o, const ResidualStore& s, std::string& name) {
    if (!o.written || o.bytes != s.boundary_bytes()) return false;
    const char* n = o.name ? o.name : "";
    if (!name.empty()) return name == n;
    if (!n || !*n) return false;
    name = n;
    return true;
}

}  // namespace

struct BoundaryRecorder::Impl {
    RadCtx* ctx = nullptr;
    ResidualStore store;
    RadReplayObserver observer{};
    std::string name;
    const void* last = nullptr;
    bool attached = false;

    Impl(RadCtx* c, int64_t tokens, int64_t hidden) : ctx(c), store(tokens, hidden) {
        if (!ctx) throw std::invalid_argument("boundary recorder needs a started engine");
        observer.user = this;
        observer.issued = &issued;
        observer.block = &block;
        rad_set_replay_observer(ctx, &observer);
        attached = true;
    }

    void attach() {
        if (!attached) {
            rad_set_replay_observer(ctx, &observer);
            attached = true;
        }
    }

    void detach() {
        if (attached) {
            rad_set_replay_observer(ctx, nullptr);
            attached = false;
        }
    }

    static void issued(void* user, RadCtx*, const RadIssueOp* o) {
        Impl& s = *static_cast<Impl*>(user);
        for (int i = 0; i < o->n_operands; ++i) {
            const RadIssueOperand& op = o->operands[i];
            if (residual_bytes(op, s.store, s.name)) s.last = op.data;
        }
    }

    static void block(void* user, RadCtx*, int64_t b, int begin) {
        Impl& s = *static_cast<Impl*>(user);
        if (!s.last) return;
        if (begin) {
            // Boundary 0 is what ran before the first block: the embeddings.
            if (b == 0) s.store.keep(0, s.last);
        } else {
            // The block's last residual write is boundary b + 1.
            s.store.keep(b + 1, s.last);
        }
    }
};

BoundaryRecorder::BoundaryRecorder(RadCtx* ctx, int64_t tokens, int64_t hidden)
    : impl_(new Impl(ctx, tokens, hidden)) {}

BoundaryRecorder::~BoundaryRecorder() = default;

void BoundaryRecorder::detach() { impl_->detach(); }

const ResidualStore& BoundaryRecorder::store() const { return impl_->store; }

struct LayerReplay::Impl {
    rad::Engine* engine = nullptr;
    RadCtx* ctx = nullptr;
    ResidualStore store;
    std::vector<OpRecord> ops;
    std::vector<Tape::Block> blocks;
    RadReplayObserver observer{};
    const ReplaySnapshot* after = nullptr;
    int64_t target = -1;
    int64_t op = 0;
    bool inside = false;
    bool attached = false;

    Impl(rad::Engine& e, Tape& tape, ResidualStore s)
        : engine(&e), ctx(e.issue_ctx()), store(std::move(s)) {
        if (!ctx) throw std::runtime_error("layer replay needs a started engine");
        ops = tape.ops();
        blocks = tape.blocks();
        // One observer slot: the forward's tape gives it up so the replay can watch each op.
        tape.detach();
        observer.user = this;
        observer.issued = &issued;
        observer.block = &block;
    }

    const Tape::Block* find(int64_t layer) const {
        for (const Tape::Block& b : blocks)
            if (b.index == layer) return &b;
        return nullptr;
    }

    static void issued(void* user, RadCtx*, const RadIssueOp* o) {
        Impl& s = *static_cast<Impl*>(user);
        if (!s.inside) return;
        const Tape::Block* b = s.find(s.target);
        if (b) {
            const int64_t at = b->begin + (size_t)s.op;
            if (at < (int64_t)b->end && at < (int64_t)s.ops.size()) {
                if (s.after) {
                    std::vector<OperandContent> content((size_t)o->n_operands);
                    for (int i = 0; i < o->n_operands; ++i) {
                        content[(size_t)i].data = o->operands[i].data;
                        content[(size_t)i].bytes = (size_t)o->operands[i].bytes;
                    }
                    (*s.after)(s.ops[(size_t)at], content);
                }
            }
        }
        ++s.op;
    }

    static void block(void* user, RadCtx*, int64_t b, int begin) {
        Impl& s = *static_cast<Impl*>(user);
        if (b != s.target) return;
        s.inside = begin != 0;
        if (begin) s.op = 0;
    }

    void watch(const ReplaySnapshot* fn) {
        after = fn;
        target = -1;
        op = 0;
        inside = false;
        rad_set_replay_observer(ctx, &observer);
        attached = true;
    }

    void stop() {
        if (attached) {
            rad_set_replay_observer(ctx, nullptr);
            attached = false;
        }
        after = nullptr;
        inside = false;
    }
};

LayerReplay::LayerReplay(rad::Engine& engine, Tape& tape, ResidualStore store)
    : impl_(new Impl(engine, tape, std::move(store))) {}

LayerReplay::~LayerReplay() { impl_->stop(); }

int64_t LayerReplay::layers() const { return (int64_t)impl_->blocks.size(); }

const ResidualStore& LayerReplay::store() const { return impl_->store; }

const void* LayerReplay::residual(int64_t boundary) const {
    return impl_->store.source(boundary);
}

void LayerReplay::replay(int64_t layer, const ReplaySnapshot& after) {
    Impl& s = *impl_;
    const Tape::Block* b = s.find(layer);
    if (!b) throw std::out_of_range("no recorded block for that layer");

    const uint16_t* input = s.store.at(layer);
    const void* dst = s.store.source(layer);
    if (!input || !dst) throw std::runtime_error("the store lacks this layer's input boundary");
    // The residual buffer is the one the forward wrote and the one this block writes again.
    std::memcpy(const_cast<void*>(dst), input, (size_t)s.store.boundary_bytes());

    s.watch(&after);
    s.target = layer;
    const int rc = s.engine->replay_block(layer);
    s.stop();
    if (rc != RAD_OK) throw std::runtime_error(std::string("layer replay failed: ") +
                                               rad_strerror(rc));
}

}  // namespace r97

#endif  // RAD_ISSUE_OBSERVER && RAD_REPLAY_BLOCK
