// tests/host/walk_parallel.cpp -- the walk's own per-op work, partitioned over the cores.
//
// The walk copies each operand's forward value out of the snapshot store, drains the version
// store's cotangents into the op's `cot` slots, sizes the `dc` slots and hands what the adjoints
// left back to the versions each read saw. None of that is the adjoint row's arithmetic, and all of
// it is one worker's today at a real block's rows. The partition may change how many threads do the
// work, never what any output element sums: the same tape walked with one worker and with eight has
// to agree byte for byte, and the three-read case pins the order one buffer's contributions add in
// against the walk's own recomputation.
#include "r97/backward.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "support/test.h"
#include "support/vector_tape.h"

namespace {

using r97::AdValue;
using r97::BoundaryNames;
using r97::OpParam;
using r97::OpRecord;
using r97::Operand;
using r97::Seed;

// ------------------------------------------------------------------ stand-in adjoints

int64_t param_int(const OpRecord& rec, const std::string& name, int64_t fallback) {
    for (const OpParam& p : rec.params)
        if (p.first == name)
            if (const auto* i = std::get_if<int64_t>(&p.second)) return *i;
    return fallback;
}

// y = s * x, with `s` an integer parameter: the cotangent of x is s times the cotangent of y.
void scale_adjoint(const r97::AdjointCall& call) {
    const double s = static_cast<double>(param_int(call.rec, "s", 1));
    const AdValue& y = call.cot.at(0);
    AdValue& x = call.dc->at(0);
    for (size_t i = 0; i < y.value.size(); ++i) x.value[i] += s * y.value[i];
}
R97_REGISTER_ADJOINT("walkpar_scale", host, scale_adjoint)

// y = x * w with `w` frozen: its snapshot is still materialised for the row, and its cotangent
// slot is never sized.
void mul_adjoint(const r97::AdjointCall& call) {
    const AdValue& y = call.cot.at(0);
    const AdValue& w = call.fwd.at(1);
    AdValue& x = call.dc->at(0);
    for (size_t i = 0; i < y.value.size(); ++i) x.value[i] += y.value[i] * w.value[i];
}
R97_REGISTER_ADJOINT_F("walkpar_mul", host, mul_adjoint, 1)

// y = s0*v + s1*v + s2*v, v read three times: the three cotangents all add into the one version
// the three reads saw, in the record's operand order.
void triple_adjoint(const r97::AdjointCall& call) {
    const AdValue& y = call.cot.at(0);
    for (size_t j = 0; j < 3; ++j) {
        const double s = static_cast<double>(param_int(call.rec, "s" + std::to_string(j), 1));
        AdValue& v = call.dc->at(j);
        for (size_t i = 0; i < y.value.size(); ++i) v.value[i] += s * y.value[i];
    }
}
R97_REGISTER_ADJOINT("walkpar_triple", host, triple_adjoint)

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

Operand read(const std::string& buffer, const std::vector<int64_t>& shape) {
    Operand o;
    o.buffer = buffer;
    o.dtype = "f32";
    o.shape = shape;
    return o;
}

Operand written(const std::string& buffer, const std::vector<int64_t>& shape) {
    Operand o = read(buffer, shape);
    o.written = true;
    return o;
}

OpRecord op(const std::string& name, std::vector<OpParam> params, std::vector<Operand> operands,
            int64_t position) {
    OpRecord rec;
    rec.op = name;
    rec.params = std::move(params);
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

// The forward of the test's ops in f64, writing every operand's snapshot as it goes: a read's
// value before the record's write, the written operand's value after it.
Env run(const std::vector<OpRecord>& records, Env env, r97::StoredSnapshots* snaps) {
    for (const OpRecord& rec : records) {
        for (size_t j = 0; j < rec.operands.size(); ++j) {
            const Operand& o = rec.operands[j];
            if (o.written) continue;
            buffer_of(env, o);
            if (snaps) snaps->set(rec.position, j, advalue(env, o));
        }
        size_t w = 0;
        while (w < rec.operands.size() && !rec.operands[w].written) ++w;
        if (w == rec.operands.size()) throw std::runtime_error("test: a record with no write");
        const Operand& out = rec.operands[w];
        std::vector<double>& dst = buffer_of(env, out);
        const std::vector<double>& a = env.value.at(rec.operands[0].buffer);
        if (rec.op == "walkpar_scale") {
            const double s = static_cast<double>(param_int(rec, "s", 1));
            for (size_t i = 0; i < dst.size(); ++i) dst[i] = s * a[i];
        } else if (rec.op == "walkpar_mul") {
            const std::vector<double>& wv = env.value.at(rec.operands[1].buffer);
            for (size_t i = 0; i < dst.size(); ++i) dst[i] = a[i] * wv[i];
        } else if (rec.op == "walkpar_triple") {
            const double s0 = static_cast<double>(param_int(rec, "s0", 1));
            const double s1 = static_cast<double>(param_int(rec, "s1", 1));
            const double s2 = static_cast<double>(param_int(rec, "s2", 1));
            for (size_t i = 0; i < dst.size(); ++i) dst[i] = s0 * a[i] + s1 * a[i] + s2 * a[i];
        } else {
            throw std::runtime_error("test: no forward for op " + rec.op);
        }
        if (snaps) snaps->set(rec.position, w, advalue(env, out));
    }
    return env;
}

// The boundaries a block was recorded with: every buffer, its shape, the whole rows.
BoundaryNames boundaries_of(const Env& end) {
    r97test::VectorTape tape;
    for (const auto& [name, value] : end.value) {
        std::vector<float> f(value.size());
        for (size_t i = 0; i < value.size(); ++i) f[i] = static_cast<float>(value[i]);
        tape.set_buffer(name, end.shape.at(name), std::move(f));
    }
    return BoundaryNames::from(tape);
}

Seed seed_of(std::string buffer, int64_t row_lo, int64_t row_hi, std::vector<float> cotangent) {
    Seed s;
    s.buffer = std::move(buffer);
    s.row_lo = row_lo;
    s.row_hi = row_hi;
    s.cotangent = std::move(cotangent);
    return s;
}

// The tape: x --3--> a --*w--> b --(2,5,7 on three reads)--> c --11--> y. One scale at each end,
// a frozen weight in the middle, and a record whose three read operands are the same buffer -- the
// case the distribute loop's unit (the buffer) exists for.
std::vector<OpRecord> tape_of(int64_t rows, int64_t cols) {
    const std::vector<int64_t> shape{rows, cols};
    return {
        op("walkpar_scale", {{"s", int64_t(3)}}, {read("x", shape), written("a", shape)}, 0),
        op("walkpar_mul", {}, {read("a", shape), read("w", shape), written("b", shape)}, 1),
        op("walkpar_triple", {{"s0", int64_t(2)}, {"s1", int64_t(5)}, {"s2", int64_t(7)}},
           {read("b", shape), read("b", shape), read("b", shape), written("c", shape)}, 2),
        op("walkpar_scale", {{"s", int64_t(11)}}, {read("c", shape), written("y", shape)}, 3),
    };
}

uint64_t next(uint64_t h) {
    h ^= h << 13;
    h ^= h >> 7;
    h ^= h << 17;
    return h;
}

double draw(uint64_t h) {
    return static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0) - 0.5;
}

// The block's inputs: values in [-0.5, 0.5), so no sum cancels to a value the order cannot move.
Env start_of(int64_t rows, int64_t cols) {
    Env start;
    const size_t n = static_cast<size_t>(rows * cols);
    uint64_t h = 0x9e3779b97f4a7c15ull;
    for (const char* name : {"x", "w"}) {
        start.shape[name] = {rows, cols};
        std::vector<double>& v = start.value[name];
        v.resize(n);
        for (size_t i = 0; i < n; ++i) v[i] = draw(h = next(h));
    }
    return start;
}

std::vector<float> seed_values(size_t n) {
    std::vector<float> out(n);
    uint64_t h = 0x2545f4914f6cdd1dull;
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(draw(h = next(h)));
    return out;
}

// R97_THREADS for the duration of one scope, restored on the way out: the count is read per call,
// so a test that sets it must leave the runner's value behind.
struct ThreadsScope {
    explicit ThreadsScope(const char* v) {
        if (const char* old = std::getenv("R97_THREADS")) {
            had = true;
            saved = old;
        }
        if (v) ::setenv("R97_THREADS", v, 1);
        else ::unsetenv("R97_THREADS");
    }
    ~ThreadsScope() {
        if (had) ::setenv("R97_THREADS", saved.c_str(), 1);
        else ::unsetenv("R97_THREADS");
    }
    bool had = false;
    std::string saved;
};

struct Walked {
    r97::BackwardResult result;
    double seconds = 0.0;
};

Walked walk(const std::vector<OpRecord>& records, const r97::StoredSnapshots& snaps,
            const BoundaryNames& boundaries, const std::vector<Seed>& seeds) {
    Walked out;
    const auto begin = std::chrono::steady_clock::now();
    out.result = r97::backward_block(records, snaps, boundaries, seeds, {}, {});
    out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    return out;
}

void check_same_inputs(const r97::BackwardResult& serial, const r97::BackwardResult& threaded) {
    R97_CHECK(serial.inputs.size() == threaded.inputs.size());
    for (size_t i = 0; i < serial.inputs.size(); ++i) {
        const r97::InputCotangent& a = serial.inputs[i];
        const r97::InputCotangent& b = threaded.inputs[i];
        R97_CHECK(a.buffer == b.buffer);
        R97_CHECK(a.value.size() == b.value.size());
        R97_CHECK(a.value.empty() || std::memcmp(a.value.data(), b.value.data(),
                                                 a.value.size() * sizeof(double)) == 0);
    }
    // The walk's own figures are the same in both runs: the partition moved no byte and no count.
    R97_CHECK(serial.report.peak_snapshot_bytes == threaded.report.peak_snapshot_bytes);
    R97_CHECK(serial.report.peak_cotangent_bytes == threaded.report.peak_cotangent_bytes);
}

}  // namespace

// A block at the brief's own size: 4096 rows (every row chunk the partition cuts, and the rows the
// walk's own copies grow eight-fold from the 512 case), 512 wide, so one record's operand payload is
// megabytes. The same tape, walked by one worker and by eight, must leave the same input
// cotangents -- byte for byte.
R97_TEST(walk_parallel_is_the_serial_walk) {
    const int64_t rows = 4096, cols = 512;
    const std::vector<OpRecord> records = tape_of(rows, cols);
    r97::StoredSnapshots snaps;
    const Env end = run(records, start_of(rows, cols), &snaps);
    const BoundaryNames boundaries = boundaries_of(end);
    const std::vector<Seed> seeds{
        seed_of("y", 0, rows, seed_values(static_cast<size_t>(rows * cols)))};

    // Both orders, because the first walk of a tape is the cold one -- a walk that reuses the
    // allocator's pages is cheaper than one that faults them in -- and a timing that only ever
    // runs serial-then-threaded would credit the partition with the warm caches.
    Walked serial, threaded, warm, back;
    {
        ThreadsScope one("1");
        serial = walk(records, snaps, boundaries, seeds);
    }
    {
        ThreadsScope eight("8");
        threaded = walk(records, snaps, boundaries, seeds);
    }
    {
        ThreadsScope eight("8");
        warm = walk(records, snaps, boundaries, seeds);
    }
    {
        ThreadsScope one("1");
        back = walk(records, snaps, boundaries, seeds);
    }

    check_same_inputs(serial.result, threaded.result);
    check_same_inputs(serial.result, warm.result);
    check_same_inputs(serial.result, back.result);
    std::printf(
        "walk_parallel %lldx%lld: 1 worker %.3f then %.3f s, 8 workers %.3f then %.3f s\n",
        (long long)rows, (long long)cols, serial.seconds, back.seconds, threaded.seconds,
        warm.seconds);
}

// The three reads of one buffer all add into the version they saw, in the record's operand order,
// on top of a zeroed version: ((0 + 2c) + 5c) + 7c. A unit that let the three race, or reordered
// them, would move the last bits; the walk's own recomputation pins the order.
R97_TEST(walk_parallel_keeps_the_serial_order) {
    const int64_t rows = 3, cols = 2;
    const std::vector<OpRecord> records = tape_of(rows, cols);
    const Env start = start_of(rows, cols);
    r97::StoredSnapshots snaps;
    const Env end = run(records, start, &snaps);
    const BoundaryNames boundaries = boundaries_of(end);
    const std::vector<float> e = seed_values(static_cast<size_t>(rows * cols));
    const std::vector<Seed> seeds{seed_of("y", 0, rows, e)};

    Walked threaded;
    {
        ThreadsScope eight("8");
        threaded = walk(records, snaps, boundaries, seeds);
    }
    {
        ThreadsScope one("1");
        const Walked serial = walk(records, snaps, boundaries, seeds);
        check_same_inputs(serial.result, threaded.result);
    }

    const std::vector<double>& w = start.value.at("w");
    std::vector<double> want(static_cast<size_t>(rows * cols));
    for (size_t i = 0; i < want.size(); ++i) {
        double ce = 0.0;
        ce += 11.0 * static_cast<double>(e[i]);  // the seed through the last scale
        double acc = 0.0;                        // the triple read, in operand order
        acc += 2.0 * ce;
        acc += 5.0 * ce;
        acc += 7.0 * ce;
        double da = 0.0;
        da += acc * w[i];  // the frozen weight's forward value
        double dx = 0.0;
        dx += 3.0 * da;  // the first scale
        want[i] = dx;
    }
    R97_CHECK(threaded.result.inputs.size() == 1);
    R97_CHECK(threaded.result.inputs[0].buffer == "x");
    R97_CHECK(threaded.result.inputs[0].value == want);
}
