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

// The residual is one operand: bf16, two dimensions, `tokens * hidden * 2` bytes, written, and it
// names a declared buffer. The first one seen names that buffer (the embeddings' output); later
// operands must carry that name, so a norm, a tap of the same byte count, or a KV write (whose
// tape identity is synthetic) is not mistaken for the residual.
bool residual_bytes(const Operand& o, const OperandContent& c, const ResidualStore& s,
                    std::string& name) {
    if (!o.written || !c.named || c.bytes != (size_t)s.boundary_bytes()) return false;
    if (!name.empty()) return name == o.buffer;
    if (o.buffer.empty()) return false;
    name = o.buffer;
    return true;
}

}  // namespace

struct BoundaryRecorder::Impl {
    Tape* tape = nullptr;
    ResidualStore store;
    std::string name;
    const void* last = nullptr;
    bool attached = false;

    Impl(Tape& t, int64_t tokens, int64_t hidden) : tape(&t), store(tokens, hidden) {
        if (!tape->attached())
            throw std::invalid_argument("boundary recorder needs a tape attached to a started engine");
        attach();
    }

    void attach() {
        if (attached) return;
        tape->set_issue_hook([this](const Tape&, const OpRecord& op,
                                    const std::vector<OperandContent>& content) {
            issued(op, content);
        });
        tape->set_bracket_hook([this](const Tape&, int64_t b, bool begin) { block(b, begin); });
        attached = true;
    }

    void detach() {
        if (!attached) return;
        // The recorder must let go before a replay replaces these hooks.
        tape->set_issue_hook(nullptr);
        tape->set_bracket_hook(nullptr);
        attached = false;
    }

    void issued(const OpRecord& op, const std::vector<OperandContent>& content) {
        for (size_t i = 0; i < op.operands.size() && i < content.size(); ++i)
            if (residual_bytes(op.operands[i], content[i], store, name)) last = content[i].data;
    }

    void block(int64_t b, bool begin) {
        if (!last) return;
        if (begin) {
            // Boundary 0 is what ran before the first block: the embeddings.
            if (b == 0) store.keep(0, last);
        } else {
            // The block's last residual write is boundary b + 1.
            store.keep(b + 1, last);
        }
    }
};

BoundaryRecorder::BoundaryRecorder(Tape& tape, int64_t tokens, int64_t hidden)
    : impl_(new Impl(tape, tokens, hidden)) {}

BoundaryRecorder::~BoundaryRecorder() = default;

void BoundaryRecorder::detach() { impl_->detach(); }

const ResidualStore& BoundaryRecorder::store() const { return impl_->store; }

struct LayerReplay::Impl {
    rad::Engine* engine = nullptr;
    Tape* tape = nullptr;
    ResidualStore store;
    std::vector<OpRecord> ops;
    std::vector<Tape::Block> blocks;
    const ReplaySnapshot* after = nullptr;
    int64_t target = -1;
    int64_t op = 0;
    bool inside = false;

    Impl(rad::Engine& e, Tape& t, ResidualStore s)
        : engine(&e), tape(&t), store(std::move(s)) {
        if (!e.issue_ctx()) throw std::runtime_error("layer replay needs a started engine");
        ops = t.ops();
        blocks = t.blocks();
        // The forward's tape has recorded what this replay needs. It lets go of the context so the
        // same tape can attach again for the replayed block, with this replay's hooks.
        t.detach();
    }

    const Tape::Block* find(int64_t layer) const {
        for (const Tape::Block& b : blocks)
            if (b.index == layer) return &b;
        return nullptr;
    }

    void issued(const OpRecord&, const std::vector<OperandContent>& content) {
        if (!inside) return;
        const Tape::Block* b = find(target);
        if (b) {
            const int64_t at = b->begin + op;
            if (at < b->end && at < (int64_t)ops.size() && after) (*after)(ops[(size_t)at], content);
        }
        ++op;
    }

    void block(int64_t b, bool begin) {
        if (b != target) return;
        inside = begin;
        if (begin) op = 0;
    }

    void watch(const ReplaySnapshot* fn) {
        after = fn;
        target = -1;
        op = 0;
        inside = false;
        RadCtx* ctx = engine->issue_ctx();
        if (!ctx) throw std::runtime_error("layer replay needs a started engine");
        tape->set_issue_hook([this](const Tape&, const OpRecord& o,
                                    const std::vector<OperandContent>& content) {
            issued(o, content);
        });
        tape->set_bracket_hook([this](const Tape&, int64_t b, bool begin) { block(b, begin); });
        tape->attach(ctx);
    }

    void stop() {
        tape->detach();
        tape->set_issue_hook(nullptr);
        tape->set_bracket_hook(nullptr);
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
