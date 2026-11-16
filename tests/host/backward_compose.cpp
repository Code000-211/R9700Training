// tests/host/backward_compose.cpp -- the walk's rule for an in-place write the record also reads.
//
// A fused producer freezes its residual operand: the row does not compute that slot, and the walk
// is what carries the write's cotangent to the value it replaced. So the slot must be sized and
// distributed *even though the row froze it*. On the live 27B tape (`rmsnorm_had_quant_i8g`,
// issue #145) it was neither: the freeze left the slot unsized, the passthrough skipped unsized
// slots, and the distribution skipped frozen operands -- so the residual stream's whole cotangent
// died at every fused norm and only the branch term survived, which the quantiser's 1/scale
// inflates by about 1.6e3 per block.
//
// The two cases below are that record's shape, with a stand-in row that freezes its in-place
// operand and adds nothing, so only the walk's own rule is under test.
#include "r97/backward.h"

#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "support/test.h"
#include "support/vector_tape.h"

namespace {

using r97::AdValue;
using r97::OpRecord;
using r97::Operand;
using r97::Seed;

// A row that computes nothing: the record's in-place write is frozen (index 1), exactly as
// `rmsnorm_had_quant_i8g` freezes its residual. It keeps what the walk handed it, so a case that
// fails says whether the seed reached the row or the passthrough failed to carry it on.
std::vector<double> g_seen_cot;
void frozen_inplace_adjoint(const r97::AdjointCall& call) {
    g_seen_cot.clear();
    for (const AdValue& c : call.cot)
        for (double v : c.value) g_seen_cot.push_back(v);
}
R97_REGISTER_ADJOINT_F("bwd_frozen_inplace", host, frozen_inplace_adjoint, 1)

// The same record with a row that *does* answer the separate read: `dc[read] = 2 * cot`. The
// passthrough must then leave the in-place write's slot alone -- both slots distribute to the same
// prior version, so filling both would count the cotangent twice.
void answered_inplace_adjoint(const r97::AdjointCall& call) {
    if (call.cot.empty() || call.dc->empty()) return;
    AdValue& read_dc = call.dc->at(0);
    const AdValue& y = call.cot.at(0);
    for (size_t i = 0; i < read_dc.value.size() && i < y.value.size(); ++i)
        read_dc.value[i] += 2.0 * y.value[i];
}
R97_REGISTER_ADJOINT_F("bwd_answered_inplace", host, answered_inplace_adjoint, 1)

// ------------------------------------------------------------------ a block's buffers

struct Env {
    std::map<std::string, std::vector<int64_t>> shape;
    std::map<std::string, std::vector<double>> value;
};

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

int64_t cols(const std::vector<int64_t>& shape) {
    if (shape.size() <= 1) return 1;
    return numel(shape) / shape[0];
}

Operand read(const std::string& buffer, std::vector<int64_t> shape) {
    Operand o;
    o.buffer = buffer;
    o.dtype = "f32";
    o.shape = std::move(shape);
    o.written = false;
    return o;
}

Operand written(const std::string& buffer, std::vector<int64_t> shape) {
    Operand o = read(buffer, std::move(shape));
    o.written = true;
    return o;
}

// A write that also reads the value it replaces: the engine's `RAD_OPD_INOUT`.
Operand in_out(const std::string& buffer, std::vector<int64_t> shape) {
    Operand o = written(buffer, std::move(shape));
    o.in_place = true;
    return o;
}

OpRecord op(const std::string& name, std::vector<Operand> operands, int64_t position) {
    OpRecord rec;
    rec.op = name;
    rec.operands = std::move(operands);
    rec.position = position;
    return rec;
}

std::vector<double>& buffer_of(Env& env, const Operand& o) {
    const auto it = env.value.find(o.buffer);
    if (it != env.value.end()) return it->second;
    env.shape[o.buffer] = o.shape;
    env.value[o.buffer].assign(static_cast<size_t>(numel(o.shape)), 0.0);
    return env.value[o.buffer];
}

AdValue advalue(const Env& env, const Operand& o) {
    AdValue v;
    v.dtype = o.dtype;
    v.shape = o.shape;
    v.value = env.value.at(o.buffer);
    return v;
}

// The record's forward, writing every operand's snapshot: a read's value before the write, each
// written operand's value after it. `y = h` and `x = h`: the residual copy is the incoming value.
Env run(const std::vector<OpRecord>& records, Env env, r97::StoredSnapshots* snaps) {
    for (const OpRecord& rec : records) {
        for (size_t j = 0; j < rec.operands.size(); ++j) {
            const Operand& o = rec.operands[j];
            if (o.written) continue;
            buffer_of(env, o);
            if (snaps) snaps->set(rec.position, j, advalue(env, o));
        }
        for (size_t j = 0; j < rec.operands.size(); ++j) {
            const Operand& o = rec.operands[j];
            if (!o.written) continue;
            const std::vector<double>& src = env.value.at(rec.operands[0].buffer);
            std::vector<double>& dst = buffer_of(env, o);
            dst = src;
            if (snaps) snaps->set(rec.position, j, advalue(env, o));
        }
    }
    return env;
}

r97::BoundaryNames boundaries_of(const Env& end) {
    r97test::VectorTape tape;
    for (const auto& [name, value] : end.value) {
        std::vector<float> f(value.size());
        for (size_t i = 0; i < value.size(); ++i) f[i] = static_cast<float>(value[i]);
        tape.set_buffer(name, end.shape.at(name), std::move(f));
    }
    return r97::BoundaryNames::from(tape);
}

Seed seed_of(std::string buffer, int64_t row_lo, int64_t row_hi, std::vector<float> cotangent) {
    Seed s;
    s.buffer = std::move(buffer);
    s.row_lo = row_lo;
    s.row_hi = row_hi;
    s.cotangent = std::move(cotangent);
    return s;
}

const r97::InputCotangent* input_named(const r97::BackwardResult& got, const std::string& name) {
    for (const auto& in : got.inputs)
        if (in.buffer == name) return &in;
    return nullptr;
}

// ------------------------------------------------------------------ the cases

// The record reads `h`, writes the residual `x` in place and the normalised value `y`. Its row
// freezes `x` and computes nothing, so the seed on `x` can reach the block's input only through
// the walk's own rule for an in-place write.
r97::BackwardResult walk_one(const bool in_place) {
    const std::vector<int64_t> shape{3, 2};
    Env start;
    start.shape["h"] = shape;
    start.value["h"] = {1.0, -2.0, 0.5, 3.0, -1.5, 0.25};
    std::vector<OpRecord> records{
        op("bwd_frozen_inplace",
           {read("h", shape), in_place ? in_out("x", shape) : written("x", shape),
            written("y", shape)},
           0),
    };
    r97::StoredSnapshots snaps;
    const Env end = run(records, start, &snaps);
    const std::vector<float> c{1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 0.25f};
    const std::vector<Seed> seeds{seed_of("x", 0, 3, c)};
    // The passthrough is the caller's opt-in, and both step paths set it: it is the rule under
    // test, so the test turns it on rather than inheriting a default.
    r97::BackwardOptions options;
    options.in_place_passthrough = true;
    return r97::backward_block(records, snaps, boundaries_of(end), seeds, {}, options);
}

// The write's cotangent reaches the value it replaced, exactly once.
R97_TEST(backward_compose_frozen_inplace_reaches_the_input) {
    const r97::BackwardResult got = walk_one(true);
    const r97::InputCotangent* x = input_named(got, "x");
    R97_CHECK(x != nullptr);
    const std::vector<double> want{1.0, -2.0, 0.5, 3.0, -1.5, 0.25};
    R97_CHECK(x->value.size() == want.size());
    // The whole vector, named: a wrong count is either zero (the slot never reached the input) or
    // a multiple of the seed (two rules both filled it), and the two are one line apart.
    std::string got_text, want_text;
    for (size_t i = 0; i < x->value.size(); ++i) {
        got_text += (i ? " " : "") + std::to_string(x->value[i]);
        want_text += (i ? " " : "") + std::to_string(want[i]);
    }
    if (got_text != want_text) {
        std::string seen;
        for (size_t i = 0; i < g_seen_cot.size(); ++i)
            seen += (i ? " " : "") + std::to_string(g_seen_cot[i]);
        std::fprintf(stderr,
                     "backward_compose: input x = [%s], want [%s]; the row was handed cot = [%s]\n",
                     got_text.c_str(), want_text.c_str(), seen.c_str());
    }
    for (size_t i = 0; i < want.size(); ++i) R97_CHECK(std::fabs(x->value[i] - want[i]) <= 1e-12);
}

// The same-buffer shape: a separate read of `x` and an in-place write of `x`. The row answers the
// read, so the direct path must not also fire -- `dc[read]` and the write's slot name the same
// prior version, and both would be added.
R97_TEST(backward_compose_same_buffer_counts_once) {
    const std::vector<int64_t> shape{3, 2};
    Env start;
    start.shape["x"] = shape;
    start.value["x"] = {0.5, -1.0, 2.0, 0.25, -0.75, 1.5};
    std::vector<OpRecord> records{
        op("bwd_answered_inplace",
           {read("x", shape), in_out("x", shape), written("y", shape)}, 0),
    };
    r97::StoredSnapshots snaps;
    const Env end = run(records, start, &snaps);
    const std::vector<float> c{1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 0.25f};
    const std::vector<Seed> seeds{seed_of("x", 0, 3, c)};
    r97::BackwardOptions options;
    options.in_place_passthrough = true;
    const r97::BackwardResult got =
        r97::backward_block(records, snaps, boundaries_of(end), seeds, {}, options);
    const r97::InputCotangent* x = input_named(got, "x");
    R97_CHECK(x != nullptr);
    // The row's answer, once: 2 * the seed. Three times it means both paths fired.
    for (size_t i = 0; i < x->value.size(); ++i)
        R97_CHECK(std::fabs(x->value[i] - 2.0 * c[i]) <= 1e-12);
}

// A plain write replaces its value, so the prior value is dead: nothing may reach the input. The
// rule is the `in_place` mark, not the frozen set, and this case is what keeps it that way.
R97_TEST(backward_compose_plain_write_reaches_nothing) {
    const r97::BackwardResult got = walk_one(false);
    R97_CHECK(input_named(got, "x") == nullptr);
}

}  // namespace
