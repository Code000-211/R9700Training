// tests/host/backward_core.cpp -- the backward core on hand-built blocks.
//
// The blocks are built from `add`, which already has an adjoint, and from stand-in adjoints
// registered here, so nothing here needs a model. Every case compares the walk's input cotangent
// with a hand derivation and with a central difference of the same program; the report, the
// missing-adjoint error and the stop marker are checked where they belong.
#include "r97/backward.h"

#include <cmath>
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
using r97::OpParam;
using r97::OpRecord;
using r97::Operand;
using r97::Seed;

// ------------------------------------------------------------------ stand-in adjoints

int64_t param_int(const OpRecord& rec, const char* name, int64_t fallback) {
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
R97_REGISTER_ADJOINT("bwd_scale", host, scale_adjoint)

// A stop stands in for the ops whose path ends (docs/adjoints/gdn_prep.md section 5): its adjoint
// adds nothing, and the walk must not call it at all.
int g_stop_calls = 0;
void stop_adjoint(const r97::AdjointCall&) { ++g_stop_calls; }
R97_REGISTER_ADJOINT("bwd_stop", host, stop_adjoint)

// A probe for the dtype tag the walk puts on a cotangent: the writer's dtype, not the seed's.
std::string g_seen_dtype;
void dtype_adjoint(const r97::AdjointCall& call) { g_seen_dtype = call.cot.at(0).dtype; }
R97_REGISTER_ADJOINT("bwd_dtype", host, dtype_adjoint)

// "bwd_unserved" has a forward in this file and no adjoint anywhere: the walk must name it.

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

Operand read(const std::string& buffer, std::vector<int64_t> shape, int64_t row_lo = 0,
             int64_t rows = 0) {
    Operand o;
    o.buffer = buffer;
    o.dtype = "f32";
    o.shape = std::move(shape);
    o.row_lo = row_lo;
    o.rows = rows;
    o.written = false;
    return o;
}

Operand written(const std::string& buffer, std::vector<int64_t> shape, int64_t row_lo = 0,
                int64_t rows = 0) {
    Operand o = read(buffer, std::move(shape), row_lo, rows);
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

// The forward of the test's ops, in f64, writing every operand's snapshot as it goes: a read's
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
        const Operand& out = rec.operands[w];
        std::vector<double> produced(static_cast<size_t>(numel(out.shape)), 0.0);
        if (rec.op == "add") {
            const auto& a = env.value.at(rec.operands[0].buffer);
            const auto& b = env.value.at(rec.operands[1].buffer);
            for (size_t i = 0; i < produced.size(); ++i) produced[i] = a[i] + b[i];
        } else if (rec.op == "bwd_scale" || rec.op == "bwd_stop" || rec.op == "bwd_unserved" ||
                   rec.op == "bwd_dtype") {
            const double s =
                rec.op == "bwd_scale" ? static_cast<double>(param_int(rec, "s", 1)) : 1.0;
            const auto& x = env.value.at(rec.operands[0].buffer);
            for (size_t i = 0; i < produced.size(); ++i) produced[i] = s * x[i];
        } else {
            throw std::runtime_error("test: no forward for op " + rec.op);
        }
        // A row-ranged write leaves the buffer's other rows as they were.
        std::vector<double>& dst = buffer_of(env, out);
        const int64_t width = cols(out.shape);
        const int64_t lo = out.row_lo;
        const int64_t hi = out.row_lo + (out.rows != 0 ? out.rows : out.shape[0]);
        for (int64_t r = lo; r < hi; ++r)
            for (int64_t k = 0; k < width; ++k)
                dst[static_cast<size_t>(r * width + k)] =
                    produced[static_cast<size_t>(r * width + k)];
        if (snaps) snaps->set(rec.position, w, advalue(env, out));
    }
    return env;
}

// The boundaries a block was recorded with: every buffer, its shape, the whole rows.
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

// ------------------------------------------------------------------ comparisons

const r97::InputCotangent& input_named(const r97::BackwardResult& got, const std::string& name) {
    for (const auto& in : got.inputs)
        if (in.buffer == name) return in;
    throw std::runtime_error("test: no input cotangent for " + name);
}

void expect_values(const std::vector<double>& got, const std::vector<double>& want,
                   double tol = 1e-9) {
    R97_CHECK(got.size() == want.size());
    for (size_t i = 0; i < got.size(); ++i) {
        const double scale = std::max(1.0, std::fabs(want[i]));
        R97_CHECK(std::fabs(got[i] - want[i]) <= tol * scale);
    }
}

double loss_at(const std::vector<OpRecord>& records, Env env, const std::vector<Seed>& seeds) {
    env = run(records, std::move(env), nullptr);
    double sum = 0.0;
    for (const Seed& seed : seeds) {
        const std::vector<double>& value = env.value.at(seed.buffer);
        const int64_t width = cols(env.shape.at(seed.buffer));
        for (int64_t r = seed.row_lo; r < seed.row_hi; ++r)
            for (int64_t k = 0; k < width; ++k) {
                const size_t at = static_cast<size_t>((r - seed.row_lo) * width + k);
                sum += static_cast<double>(seed.cotangent[at]) *
                       value[static_cast<size_t>(r * width + k)];
            }
    }
    return sum;
}

// A deterministic direction over every input, in [-1, 1].
std::map<std::string, std::vector<double>> direction_of(const Env& start) {
    std::map<std::string, std::vector<double>> dir;
    uint64_t h = 0x9e3779b97f4a7c15ull;
    for (const auto& [name, value] : start.value) {
        std::vector<double> d(value.size());
        for (size_t i = 0; i < d.size(); ++i) {
            h ^= h >> 30;
            h *= 0xbf58476d1ce4e5b9ull;
            h ^= h >> 27;
            d[i] = static_cast<double>(h % 2001) / 1000.0 - 1.0;
        }
        dir[name] = std::move(d);
    }
    return dir;
}

struct Diff {
    double analytic = 0.0;
    double numeric = 0.0;
    double rel = 0.0;
};

// The directional derivative the walk reports against a central difference of the same program.
Diff compare_direction(const std::vector<OpRecord>& records, const Env& start,
                       const std::vector<Seed>& seeds, const r97::BackwardResult& got) {
    const auto dir = direction_of(start);
    Diff out;
    for (const auto& in : got.inputs) {
        const std::vector<double>& d = dir.at(in.buffer);
        const int64_t width = cols(in.shape);
        for (int64_t r = in.row_lo; r < in.row_hi; ++r)
            for (int64_t k = 0; k < width; ++k)
                out.analytic += in.value[static_cast<size_t>((r - in.row_lo) * width + k)] *
                                d[static_cast<size_t>(r * width + k)];
    }
    constexpr double eps = 1e-6;
    Env plus = start;
    Env minus = start;
    for (const auto& [name, d] : dir) {
        std::vector<double>& p = plus.value[name];
        std::vector<double>& m = minus.value[name];
        for (size_t i = 0; i < d.size(); ++i) {
            p[i] += eps * d[i];
            m[i] -= eps * d[i];
        }
    }
    out.numeric =
        (loss_at(records, std::move(plus), seeds) - loss_at(records, std::move(minus), seeds)) /
        (2.0 * eps);
    const double scale = std::max(
        1.0, std::max(std::fabs(out.analytic), std::fabs(out.numeric)));
    out.rel = std::fabs(out.analytic - out.numeric) / scale;
    return out;
}

// The block's input and its value, the shape {3, 2} most cases here use.
Env x_start() {
    Env start;
    start.shape["x"] = {3, 2};
    start.value["x"] = {0.5, -1.0, 2.0, 0.25, -0.75, 1.5};
    return start;
}

std::vector<float> seed_values() { return {1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 0.25f}; }

// ------------------------------------------------------------------ the cases

// A chain: x --3--> h --5--> y. The cotangent of x is 15 times the seed's.
R97_TEST(backward_core_chain) {
    const std::vector<int64_t> shape{3, 2};
    const Env start = x_start();
    const std::vector<OpRecord> records{
        op("bwd_scale", {{"s", int64_t(3)}}, {read("x", shape), written("h", shape)}, 0),
        op("bwd_scale", {{"s", int64_t(5)}}, {read("h", shape), written("y", shape)}, 1),
    };
    r97::StoredSnapshots snaps;
    const Env end = run(records, start, &snaps);
    const r97::BoundaryNames boundaries = boundaries_of(end);
    const std::vector<float> c = seed_values();
    const std::vector<Seed> seeds{seed_of("y", 0, 3, c)};

    const r97::BackwardResult got =
        r97::backward_block(records, snaps, boundaries, seeds, {}, {});

    R97_CHECK(got.inputs.size() == 1);
    std::vector<double> want;
    for (float v : c) want.push_back(15.0 * static_cast<double>(v));
    expect_values(input_named(got, "x").value, want);
    R97_CHECK(input_named(got, "x").shape == shape);

    const Diff diff = compare_direction(records, start, seeds, got);
    R97_CHECK(diff.rel < 1e-7);

    // Two operands of six values per op, one op at a time.
    R97_CHECK(got.report.peak_snapshot_bytes == 96);
    R97_CHECK(got.report.peak_cotangent_bytes >= 48);
}

// A diamond: x feeds a and b, y is their sum. One value read twice sums both routes.
R97_TEST(backward_core_diamond) {
    const std::vector<int64_t> shape{3, 2};
    const Env start = x_start();
    const std::vector<OpRecord> records{
        op("bwd_scale", {{"s", int64_t(2)}}, {read("x", shape), written("a", shape)}, 0),
        op("bwd_scale", {{"s", int64_t(7)}}, {read("x", shape), written("b", shape)}, 1),
        op("add", {}, {read("a", shape), read("b", shape), written("y", shape)}, 2),
    };
    r97::StoredSnapshots snaps;
    const Env end = run(records, start, &snaps);
    const r97::BoundaryNames boundaries = boundaries_of(end);
    const std::vector<float> c = seed_values();
    const std::vector<Seed> seeds{seed_of("y", 0, 3, c)};

    const r97::BackwardResult got =
        r97::backward_block(records, snaps, boundaries, seeds, {}, {});

    std::vector<double> want;
    for (float v : c) want.push_back(9.0 * static_cast<double>(v));
    expect_values(input_named(got, "x").value, want);

    const Diff diff = compare_direction(records, start, seeds, got);
    R97_CHECK(diff.rel < 1e-7);
}

// An in-place op: the same buffer is read and written. The adjoint receives what it read from,
// not what it wrote.
R97_TEST(backward_core_in_place) {
    const std::vector<int64_t> shape{3, 2};
    const Env start = x_start();
    const std::vector<OpRecord> records{
        op("bwd_scale", {{"s", int64_t(4)}},
           {read("x", shape), written("x", shape)}, 0),
        op("bwd_scale", {{"s", int64_t(2)}}, {read("x", shape), written("y", shape)}, 1),
    };
    r97::StoredSnapshots snaps;
    const Env end = run(records, start, &snaps);
    const r97::BoundaryNames boundaries = boundaries_of(end);
    const std::vector<float> c = seed_values();
    const std::vector<Seed> seeds{seed_of("y", 0, 3, c)};

    const r97::BackwardResult got =
        r97::backward_block(records, snaps, boundaries, seeds, {}, {});

    std::vector<double> want;
    for (float v : c) want.push_back(8.0 * static_cast<double>(v));
    expect_values(input_named(got, "x").value, want);

    const Diff diff = compare_direction(records, start, seeds, got);
    R97_CHECK(diff.rel < 1e-7);
}

// A transient buffer written twice: the earlier readers' cotangents belong to the first write,
// the later readers' to the second.
R97_TEST(backward_core_reused_transient) {
    const std::vector<int64_t> shape{3, 2};
    Env start = x_start();
    start.shape["u"] = shape;
    start.value["u"] = {-0.25, 0.75, 1.25, -1.75, 0.5, 2.0};
    const std::vector<OpRecord> records{
        op("bwd_scale", {{"s", int64_t(2)}}, {read("x", shape), written("t", shape)}, 0),
        op("bwd_scale", {{"s", int64_t(3)}}, {read("t", shape), written("y1", shape)}, 1),
        op("bwd_scale", {{"s", int64_t(5)}}, {read("u", shape), written("t", shape)}, 2),
        op("bwd_scale", {{"s", int64_t(7)}}, {read("t", shape), written("y2", shape)}, 3),
    };
    r97::StoredSnapshots snaps;
    const Env end = run(records, start, &snaps);
    const r97::BoundaryNames boundaries = boundaries_of(end);
    const std::vector<float> c1{1.0f, -2.0f, 3.0f, 0.5f, -0.5f, 2.0f};
    const std::vector<float> c2{-1.0f, 4.0f, 0.25f, -3.0f, 1.0f, 0.5f};
    const std::vector<Seed> seeds{seed_of("y1", 0, 3, c1), seed_of("y2", 0, 3, c2)};

    const r97::BackwardResult got =
        r97::backward_block(records, snaps, boundaries, seeds, {}, {});

    std::vector<double> want_x;
    std::vector<double> want_u;
    for (float v : c1) want_x.push_back(6.0 * static_cast<double>(v));    // 2 * 3
    for (float v : c2) want_u.push_back(35.0 * static_cast<double>(v));  // 5 * 7
    expect_values(input_named(got, "x").value, want_x);
    expect_values(input_named(got, "u").value, want_u);

    const Diff diff = compare_direction(records, start, seeds, got);
    R97_CHECK(diff.rel < 1e-7);

    // Six buffers of six values each are alive over the block; the released versions keep the
    // peak below their sum, which is what tracking versions buys.
    R97_CHECK(got.report.peak_cotangent_bytes < 6 * 48);
}

// A partial row range: an op writes rows [1, 3) only, and only those rows carry a cotangent.
R97_TEST(backward_core_partial_rows) {
    const std::vector<int64_t> shape{4, 2};
    Env start;
    start.shape["x"] = shape;
    start.value["x"] = {0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.125, -0.375};
    const std::vector<OpRecord> records{
        op("bwd_scale", {{"s", int64_t(3)}},
           {read("x", shape, 1, 2), written("t", shape, 1, 2)}, 0),
        op("bwd_scale", {{"s", int64_t(5)}},
           {read("t", shape, 1, 2), written("y", shape, 1, 2)}, 1),
    };
    r97::StoredSnapshots snaps;
    const Env end = run(records, start, &snaps);
    const r97::BoundaryNames boundaries = boundaries_of(end);
    const std::vector<float> c{1.0f, -2.0f, 3.0f, 0.5f};
    const std::vector<Seed> seeds{seed_of("y", 1, 3, c)};

    const r97::BackwardResult got =
        r97::backward_block(records, snaps, boundaries, seeds, {}, {});

    std::vector<double> want(8, 0.0);
    for (size_t i = 0; i < c.size(); ++i)
        want[2 + i] = 15.0 * static_cast<double>(c[i]);  // rows [1, 3) of a width-2 buffer
    expect_values(input_named(got, "x").value, want);

    const Diff diff = compare_direction(records, start, seeds, got);
    R97_CHECK(diff.rel < 1e-7);
}

// Two seeds on two buffers: the walk is the sum of the two single-seed walks.
R97_TEST(backward_core_two_seeds) {
    const std::vector<int64_t> shape{3, 2};
    const Env start = x_start();
    const std::vector<OpRecord> records{
        op("bwd_scale", {{"s", int64_t(2)}}, {read("x", shape), written("a", shape)}, 0),
        op("add", {}, {read("x", shape), read("a", shape), written("b", shape)}, 1),
    };
    r97::StoredSnapshots snaps;
    const Env end = run(records, start, &snaps);
    const r97::BoundaryNames boundaries = boundaries_of(end);
    const std::vector<float> sa{1.0f, -2.0f, 3.0f, 0.5f, -0.5f, 2.0f};
    const std::vector<float> sb{-1.0f, 4.0f, 0.25f, -3.0f, 1.0f, 0.5f};
    const Seed on_a = seed_of("a", 0, 3, sa);
    const Seed on_b = seed_of("b", 0, 3, sb);

    const r97::BackwardResult both =
        r97::backward_block(records, snaps, boundaries, {on_a, on_b}, {}, {});
    const r97::BackwardResult only_a =
        r97::backward_block(records, snaps, boundaries, {on_a}, {}, {});
    const r97::BackwardResult only_b =
        r97::backward_block(records, snaps, boundaries, {on_b}, {}, {});

    std::vector<double> want;
    for (size_t i = 0; i < sa.size(); ++i)
        want.push_back(2.0 * static_cast<double>(sa[i]) + 3.0 * static_cast<double>(sb[i]));
    expect_values(input_named(both, "x").value, want);

    // The sum of the single-seed walks, bit for bit with these exact values.
    const std::vector<double>& xa = input_named(only_a, "x").value;
    const std::vector<double>& xb = input_named(only_b, "x").value;
    std::vector<double> sum(xa.size());
    for (size_t i = 0; i < sum.size(); ++i) sum[i] = xa[i] + xb[i];
    R97_CHECK(sum == input_named(both, "x").value);

    const Diff diff = compare_direction(records, start, {on_a, on_b}, both);
    R97_CHECK(diff.rel < 1e-7);
}

// One seed on the output is bit-identical to the same cotangent handed down from above.
R97_TEST(backward_core_seed_matches_the_cotangent_from_above) {
    const std::vector<int64_t> shape{3, 2};
    const Env start = x_start();
    const std::vector<OpRecord> records{
        op("bwd_scale", {{"s", int64_t(3)}}, {read("x", shape), written("h", shape)}, 0),
        op("bwd_scale", {{"s", int64_t(5)}}, {read("h", shape), written("y", shape)}, 1),
    };
    r97::StoredSnapshots snaps;
    const Env end = run(records, start, &snaps);
    const r97::BoundaryNames boundaries = boundaries_of(end);
    const std::vector<float> c = seed_values();
    std::vector<double> above(c.begin(), c.end());

    const r97::BackwardResult from_seed =
        r97::backward_block(records, snaps, boundaries, {seed_of("y", 0, 3, c)}, {}, {});
    const r97::BackwardResult from_above = r97::backward_block(
        records, snaps, boundaries, {}, {r97::IncomingCotangent{"y", 0, 3, above}}, {});

    R97_CHECK(from_seed.inputs.size() == from_above.inputs.size());
    for (size_t i = 0; i < from_seed.inputs.size(); ++i) {
        R97_CHECK(from_seed.inputs[i].buffer == from_above.inputs[i].buffer);
        R97_CHECK(from_seed.inputs[i].value == from_above.inputs[i].value);
    }
    R97_CHECK(from_seed.report.peak_snapshot_bytes == from_above.report.peak_snapshot_bytes);
    R97_CHECK(from_seed.report.peak_cotangent_bytes == from_above.report.peak_cotangent_bytes);
}

// An op marked as a stop ends the walk on its inputs: the unserved op behind it is never asked.
// A seed carries f32 values; the walk tags the cotangent with the dtype of the op that wrote the
// rows, which is what that op's adjoint expects.
R97_TEST(backward_core_tags_the_cotangent_with_the_writer_dtype) {
    const std::vector<int64_t> shape{3, 2};
    const Env start = x_start();
    Operand out = written("y", shape);
    out.dtype = "bf16";
    const std::vector<OpRecord> records{
        op("bwd_dtype", {{"alpha", int64_t(2)}}, {read("x", shape), out}, 0),
    };
    r97::StoredSnapshots snaps;
    const Env end = run(records, start, &snaps);
    const r97::BoundaryNames boundaries = boundaries_of(end);
    g_seen_dtype.clear();
    r97::backward_block(records, snaps, boundaries, {seed_of("y", 0, 3, seed_values())}, {}, {});
    R97_CHECK(g_seen_dtype == "bf16");
}

R97_TEST(backward_core_stop_ends_the_path) {
    const std::vector<int64_t> shape{3, 2};
    Env start = x_start();
    start.shape["z"] = shape;
    start.value["z"] = {0.25, 0.5, -0.5, 1.0, -1.0, 2.0};
    const std::vector<OpRecord> records{
        op("bwd_unserved", {{"alpha", int64_t(3)}}, {read("z", shape), written("q", shape)}, 0),
        op("bwd_stop", {}, {read("q", shape), written("x", shape)}, 1),
        op("bwd_scale", {{"s", int64_t(2)}}, {read("x", shape), written("y", shape)}, 2),
    };
    r97::StoredSnapshots snaps;
    const Env end = run(records, start, &snaps);
    const r97::BoundaryNames boundaries = boundaries_of(end);
    const std::vector<Seed> seeds{seed_of("y", 0, 3, seed_values())};

    r97::BackwardOptions options;
    options.stops = {"bwd_stop"};
    g_stop_calls = 0;
    const r97::BackwardResult got =
        r97::backward_block(records, snaps, boundaries, seeds, {}, options);
    R97_CHECK(g_stop_calls == 0);
    R97_CHECK(got.inputs.empty());  // x and q are written; z is never reached

    // Without the marker the same block walks through the no-op adjoint and meets the op that
    // nothing serves: the stop is the marker, not the empty adjoint.
    bool threw = false;
    try {
        r97::backward_block(records, snaps, boundaries, seeds, {}, {});
    } catch (const std::exception& e) {
        threw = std::string(e.what()).find("bwd_unserved") != std::string::npos;
    }
    R97_CHECK(threw);
}

R97_TEST(backward_core_names_a_missing_adjoint) {
    const std::vector<int64_t> shape{3, 2};
    const Env start = x_start();
    const std::vector<OpRecord> records{
        op("bwd_unserved", {{"alpha", int64_t(3)}, {"layout", std::string("row")}},
           {read("x", shape), written("y", shape)}, 0),
    };
    r97::StoredSnapshots snaps;
    const Env end = run(records, start, &snaps);
    const r97::BoundaryNames boundaries = boundaries_of(end);
    bool threw = false;
    try {
        r97::backward_block(records, snaps, boundaries, {seed_of("y", 0, 3, seed_values())}, {},
                            {});
    } catch (const std::exception& e) {
        const std::string what = e.what();
        threw = what.find("bwd_unserved") != std::string::npos &&
                what.find("alpha=3") != std::string::npos &&
                what.find("layout=row") != std::string::npos;
    }
    R97_CHECK(threw);
}

R97_TEST(backward_core_names_a_seed_it_cannot_place) {
    const std::vector<int64_t> shape{3, 2};
    const Env start = x_start();
    const std::vector<OpRecord> records{
        op("bwd_scale", {{"s", int64_t(2)}}, {read("x", shape), written("y", shape)}, 0),
    };
    r97::StoredSnapshots snaps;
    const Env end = run(records, start, &snaps);
    const r97::BoundaryNames boundaries = boundaries_of(end);

    bool named = false;
    try {
        r97::backward_block(records, snaps, boundaries, {seed_of("ghost", 0, 3, seed_values())},
                            {}, {});
    } catch (const std::exception& e) {
        const std::string what = e.what();
        named = what.find("ghost") != std::string::npos &&
                what.find("not a boundary") != std::string::npos;
    }
    R97_CHECK(named);

    // A boundary the block read but never wrote is refused too, by name.
    named = false;
    try {
        r97::backward_block(records, snaps, boundaries, {seed_of("x", 0, 3, seed_values())}, {},
                            {});
    } catch (const std::exception& e) {
        const std::string what = e.what();
        named = what.find("'x'") != std::string::npos &&
                what.find("never wrote") != std::string::npos;
    }
    R97_CHECK(named);
}

}  // namespace
