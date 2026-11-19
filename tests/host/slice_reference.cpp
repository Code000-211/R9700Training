// tests/host/slice_reference.cpp -- the slice runner's two acceptance cases.
//
// The runner itself (namespace r97::slice, with the reviewed scope note: what a slice-vs-device
// comparison rules out and what it does not) is src/slice_reference.h, included below and shared
// with the `r97train` subcommand that drives it from dump files. This file holds only the two
// synthetic, deterministic, analytic acceptance cases that comment describes, with their local
// helpers.
#include "slice_reference.h"  // the runner (internal, src/), shared with the r97train subcommand
#include "support/test.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <unistd.h>

namespace {

namespace slice = r97::slice;  // the runner, src/slice_reference.h

using r97::OpRecord;
using r97::Operand;
using r97::Seed;

// ------------------------------------------------------------------ the cases

// Every element at 1e-12 relative or tighter, with an absolute floor for a zero both sides agree
// on. `want` is the exact f64 closed form; the quantities compared here -- the twins' forward
// productions and the walk's dc / InputCotangent values -- are f64 too, so the difference is
// double arithmetic alone. Where a closed form consumes the seed it consumes the values the walk
// holds, the f32 cotangent widened back to f64 (backward's enter, src/backward.cpp:396-398).
void expect_exact(const char* what, const std::vector<double>& got,
                  const std::vector<double>& want) {
    R97_CHECK(got.size() == want.size());
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i] - want[i]) > 1e-12 * std::max(1.0, std::fabs(want[i])))
            throw std::runtime_error(std::string(what) + ": element " + std::to_string(i) +
                                     " is " + std::to_string(got[i]) + ", want " +
                                     std::to_string(want[i]));
    }
}

void as_f32(const std::vector<double>& v, std::vector<float>* out) {
    out->assign(static_cast<size_t>(v.size()), 0.0f);
    for (size_t i = 0; i < v.size(); ++i) (*out)[i] = static_cast<float>(v[i]);
}

// The f32 cotangent widened back to f64: the values the walk hands its adjoints.
std::vector<double> widened(const std::vector<double>& v) {
    std::vector<double> out;
    out.reserve(v.size());
    for (double x : v) out.push_back(static_cast<double>(static_cast<float>(x)));
    return out;
}

// A leaf gradient the lora rows accumulate in f32 (ParamSet's storage, include/r97/params.h): an
// entry is a running f32 sum of f32 products in the rows' own serial order, so the check rebuilds
// the same accumulation (f32 rounding at every add) and takes the remainder as 1e-5 of the leaf's
// largest magnitude -- a hundred times f32's own noise at these sizes, still catching a term that
// is missing, doubled or transposed (tests/host/adjoint_lora.cpp's grad_tol).
void expect_leaf(const char* what, const std::vector<float>& got,
                 const std::vector<float>& want) {
    R97_CHECK(got.size() == want.size());
    double scale = 1.0;
    for (float x : want) scale = std::max(scale, std::fabs(static_cast<double>(x)));
    for (size_t i = 0; i < got.size(); ++i)
        if (std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i])) >
            1e-5 * scale + 1e-12)
            throw std::runtime_error(std::string(what) + ": element " + std::to_string(i) +
                                     " is " + std::to_string(got[i]) + ", want " +
                                     std::to_string(want[i]));
}

Operand operand(const char* buffer, const char* dtype, std::vector<int64_t> shape, bool written) {
    Operand o;
    o.buffer = buffer;
    o.dtype = dtype;
    o.shape = std::move(shape);
    o.written = written;
    return o;
}

// splitmix64, the shared checker's own deterministic randomness (tests/support/adjoint_check.h):
// uniform in [-1, 1), fixed seed, nothing the process shares.
struct Deterministic {
    uint64_t state;
    double next() {
        state += 0x9e3779b97f4a7c15ull;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return (double)(z ^ (z >> 31)) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
    }
};

// ------------------------------------------------------------------ case (a): linear + scale + adapter

// graph:
//   t[m,j]   = sum_k a[m,k] * A[0,j,k]     (lora_shrink, bf16 float trunk)
//   inc[m,n] = sum_j t[m,j] * B[0,n,j]     (lora_expand, S=1, scale 1)
//   yA[m,n]  = sum_k a[m,k] * W[n,k]       (gemm_nt, the frozen trunk linear)
//   y1[m,n]  = s * yA[m,n]                 (scale, x * s[0])
//   y[m,n]   = y1[m,n] + inc[m,n]          (add, the adapter's parallel branch into y)
//
// seeded at y with cotangent dy (read by the walk as its f32 widening, `dyw` below), the closed
// forms -- against the twins' and adjoints' own rows (src/adjoint/host/lora.cpp:246-258, 443-449;
// src/adjoint/host/linear.cpp:221-243; dW goes to nobody: the trunk linear is frozen):
//   dt[m,j]  = sum_n dyw[m,n] * B[0,n,j]
//   dA[j,k]  = (sum over m, f32 at every add) dt[m,j] * a[m,k]
//   dB[n,j]  = (sum over m, f32 at every add) dyw[m,n] * t[m,j]
//   dx[m,k]  = s * sum_n dyw[m,n] * W[n,k] + sum_j dt[m,j] * A[0,j,k]
//              (the add passes dy down BOTH branches: the trunk route picks up scale's factor s,
//              the adapter route is the shrink's own value-space cotangent, lora.cpp's g)
R97_TEST(slice_reference_linear_scale) {
    const int64_t M = 3, K = 4, N = 5, R = 2;

    r97::OpRecord shrink;
    shrink.op = "lora_shrink";
    shrink.block = 7;
    shrink.params.push_back({"dtype", std::string("bf16")});
    // The float trunk's five-wide schema: a, none:1 (the absent a_scale), slot, lora_a, t.
    shrink.operands = {
        operand("a", "f64", {M, K}, false),
        operand("none:1", "invalid", {}, false),
        operand("adapter_slot", "i32", {M}, false),
        operand("b7test_lora_a", "f64", {1, R, K}, false),
        operand("t", "f64", {M, R}, true),
    };

    r97::OpRecord expand;
    expand.op = "lora_expand";
    expand.block = 7;
    expand.params.push_back({"dtype", std::string("bf16")});
    expand.operands = {
        operand("t", "f64", {M, R}, false),
        operand("adapter_slot", "i32", {M}, false),
        operand("b7test_lora_b", "f64", {1, N, R}, false),
        operand("lora_scale", "f32", {1, N}, false),
        operand("inc", "f64", {M, N}, true),
    };

    r97::OpRecord linear;
    linear.op = "gemm_nt";
    linear.block = 7;
    linear.params = {{"M", M}, {"N", N}, {"K", K}};
    linear.operands = {
        operand("a", "f64", {M, K}, false),
        operand("b", "f64", {N, K}, false),  // the schema's own name: gemm_nt's rows find
        operand("yA", "f64", {M, N}, true),  // their weight as operand "b" (linear.cpp:196)
    };

    r97::OpRecord scale_op;
    scale_op.op = "scale";
    scale_op.block = 7;
    scale_op.operands = {
        operand("yA", "f64", {M, N}, false),
        operand("scale_factor", "f32", {1}, false),
        operand("y1", "f64", {M, N}, true),
    };

    r97::OpRecord sum_op;
    sum_op.op = "add";
    sum_op.block = 7;
    sum_op.operands = {
        operand("y1", "f64", {M, N}, false),
        operand("inc", "f64", {M, N}, false),
        operand("y", "f64", {M, N}, true),
    };

    slice::ProgramSlice program;
    program.ops = {shrink, expand, linear, scale_op, sum_op};
    for (size_t i = 0; i < program.ops.size(); ++i)
        program.ops[i].position = static_cast<int64_t>(i);  // snapshots key on (position, slot);

    Deterministic rng{0x5eed1234ull};
    std::vector<double> a(static_cast<size_t>(M * K)), A(static_cast<size_t>(R * K)),
        B(static_cast<size_t>(N * R)), W(static_cast<size_t>(N * K)),
        dy(static_cast<size_t>(M * N));
    for (double& v : a) v = rng.next();
    for (double& v : A) v = rng.next();
    for (double& v : B) v = rng.next();
    for (double& v : W) v = rng.next();
    for (double& v : dy) v = rng.next();
    const double s = 1.75;  // the scale's factor

    // ---- the forward's closed forms, f64, in the twins' own operation order
    std::vector<double> t_want(static_cast<size_t>(M * R), 0.0);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t j = 0; j < R; ++j) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; ++k)
                acc += a[static_cast<size_t>(m * K + k)] * A[static_cast<size_t>(j * K + k)];
            t_want[static_cast<size_t>(m * R + j)] = acc;
        }
    std::vector<double> inc_want(static_cast<size_t>(M * N), 0.0);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int64_t j = 0; j < R; ++j)
                acc += t_want[static_cast<size_t>(m * R + j)] * B[static_cast<size_t>(n * R + j)];
            inc_want[static_cast<size_t>(m * N + n)] = acc;
        }
    std::vector<double> yA_want(static_cast<size_t>(M * N), 0.0);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; ++k)
                acc += a[static_cast<size_t>(m * K + k)] * W[static_cast<size_t>(n * K + k)];
            yA_want[static_cast<size_t>(m * N + n)] = acc;
        }
    std::vector<double> y1_want(static_cast<size_t>(M * N), 0.0);
    std::vector<double> y_want(static_cast<size_t>(M * N), 0.0);
    for (size_t i = 0; i < y_want.size(); ++i) {
        y1_want[i] = s * yA_want[i];
        y_want[i] = y1_want[i] + inc_want[i];
    }

    // ---- the backward's closed forms, from the widened seed the walk holds
    const std::vector<double> dyw = widened(dy);
    // dt[m,j] = sum_n dyw[m,n] * B[0,n,j] -- the expand's t cotangent, ascending n as its row adds
    std::vector<double> dt(static_cast<size_t>(M * R), 0.0);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t j = 0; j < R; ++j) {
            double acc = 0.0;
            for (int64_t n = 0; n < N; ++n)
                acc += dyw[static_cast<size_t>(m * N + n)] * B[static_cast<size_t>(n * R + j)];
            dt[static_cast<size_t>(m * R + j)] = acc;
        }
    // dx, both routes into `a`: the trunk's s * (dyw @ W), plus the shrink's value-space cotangent
    // sum_j dt[m,j] * A[0,j,k] (lora.cpp's g). The walk adds the gemm's dc first then the
    // shrink's (reverse record order), f64 both; the two orders of these f64 sums agree far
    // inside the 1e-12 the check takes.
    std::vector<double> dx(static_cast<size_t>(M * K), 0.0);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t k = 0; k < K; ++k) {
            double trunk = 0.0;
            for (int64_t n = 0; n < N; ++n)
                trunk += dyw[static_cast<size_t>(m * N + n)] * W[static_cast<size_t>(n * K + k)];
            double adapter = 0.0;
            for (int64_t j = 0; j < R; ++j)
                adapter += dt[static_cast<size_t>(m * R + j)] * A[static_cast<size_t>(j * K + k)];
            dx[static_cast<size_t>(m * K + k)] = s * trunk + adapter;
        }
    // dA, the shrink row's own accumulation: f32 at every add, ascending m (lora.cpp:325)
    std::vector<float> dA(static_cast<size_t>(R * K), 0.0f);
    for (int64_t j = 0; j < R; ++j)
        for (int64_t k = 0; k < K; ++k) {
            float acc = 0.0f;
            for (int64_t m = 0; m < M; ++m)
                acc += static_cast<float>(dt[static_cast<size_t>(m * R + j)] *
                                          a[static_cast<size_t>(m * K + k)]);
            dA[static_cast<size_t>(j * K + k)] = acc;
        }
    // dB, the expand row's own accumulation: f32 at every add, ascending m (lora.cpp:517)
    std::vector<float> dB(static_cast<size_t>(N * R), 0.0f);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t j = 0; j < R; ++j) {
            float acc = 0.0f;
            for (int64_t m = 0; m < M; ++m)
                acc += static_cast<float>(dyw[static_cast<size_t>(m * N + n)] *
                                          t_want[static_cast<size_t>(m * R + j)]);
            dB[static_cast<size_t>(n * R + j)] = acc;
        }

    // ---- the injection set: one boundary forward value (a), the frozen operands as declared
    // constants -- the slot included, exercised here exactly so the acceptance proves the
    // constants path serves what the slice declared end to end.
    slice::SliceInput in;
    in.prefix = "b7test";
    in.boundary["a"] = slice::make_value("f64", {M, K}, a);
    in.constants["adapter_slot"] =
        slice::make_value("i32", {M}, std::vector<double>(static_cast<size_t>(M), 0.0));
    in.constants["b7test_lora_a"] = slice::make_value("f64", {1, R, K}, A);
    in.constants["b7test_lora_b"] = slice::make_value("f64", {1, N, R}, B);
    in.constants["lora_scale"] =
        slice::make_value("f32", {1, N}, std::vector<double>(static_cast<size_t>(N), 1.0));
    in.constants["scale_factor"] = slice::make_value("f32", {1}, {s});
    in.constants["b"] = slice::make_value("f64", {N, K}, W);  // gemm_nt's weight, read by name
    in.out_buffers = {"t", "inc", "yA", "y1", "y"};
    in.leaf_buffers = {"a"};

    Seed seed;
    seed.buffer = "y";
    seed.row_lo = 0;
    seed.row_hi = M;
    as_f32(dy, &seed.cotangent);
    in.seeds.push_back(std::move(seed));

    const slice::SliceRun got = slice::run_slice(program, in);
    slice::print_report(got, in);

    // ---- forward: the twins replay the closed forms (f64 against f64)
    R97_CHECK(got.env.count("t") == 1);
    R97_CHECK(got.env.count("inc") == 1);
    R97_CHECK(got.env.count("yA") == 1);
    R97_CHECK(got.env.count("y1") == 1);
    R97_CHECK(got.env.count("y") == 1);
    expect_exact("t (forward)", got.env.at("t").value.value, t_want);
    expect_exact("inc (forward)", got.env.at("inc").value.value, inc_want);
    expect_exact("yA (forward)", got.env.at("yA").value.value, yA_want);
    expect_exact("y1 (forward)", got.env.at("y1").value.value, y1_want);
    expect_exact("y (forward)", got.env.at("y").value.value, y_want);
    // the served constants never became buffers: the environment holds only what a slice op
    // wrote and what the boundary injected
    R97_CHECK(got.env.count("adapter_slot") == 0);
    R97_CHECK(got.env.count("b") == 0);

    // ---- backward: the leaves (b7test.lora_A, b7test.lora_B) against their closed forms
    const r97::Param* leaf_a = nullptr;
    const r97::Param* leaf_b = nullptr;
    for (const r97::Param& p : got.grads) {
        if (p.name == "b7test.lora_A") leaf_a = &p;
        if (p.name == "b7test.lora_B") leaf_b = &p;
    }
    R97_CHECK(leaf_a != nullptr);
    R97_CHECK(leaf_b != nullptr);
    if (leaf_a != nullptr)
        expect_leaf("lora_A (dA)",
                    std::vector<float>(leaf_a->value.begin(), leaf_a->value.end()), dA);
    if (leaf_b != nullptr)
        expect_leaf("lora_B (dB)",
                    std::vector<float>(leaf_b->value.begin(), leaf_b->value.end()), dB);

    // dx: the injected boundary's own cotangent, both routes summed, dense over the buffer
    const r97::InputCotangent* input_a = nullptr;
    for (const r97::InputCotangent& ic : got.walk.inputs)
        if (ic.buffer == "a") input_a = &ic;
    R97_CHECK(input_a != nullptr);
    if (input_a != nullptr) expect_exact("a (dx)", input_a->value, dx);

    // The constants carry no input cotangent of their own: they were never buffers, and the
    // weight name resolved nothing by itself. "a", the one boundary the slice read without
    // writing, is the walk's only input.
    for (const r97::InputCotangent& ic : got.walk.inputs) R97_CHECK(ic.buffer == "a");

    // ---- the refusal by name: the same slice, its boundary `a` removed. Nothing else can serve
    // the shrink's and the linear's read -- resolve_read would fall through to the container
    // weight of that name (src/reference_step.cpp:958-982); this runner refuses, naming it.
    slice::SliceInput refuse_in = in;
    refuse_in.boundary.clear();
    refuse_in.out_buffers.clear();
    refuse_in.leaf_buffers.clear();
    refuse_in.seeds.clear();
    bool threw = false;
    std::string message;
    try {
        (void)slice::run_slice(program, refuse_in);
    } catch (const std::exception& e) {
        threw = true;
        message = e.what();
    }
    R97_CHECK(threw);
    if (threw) R97_CHECK(message.find("'a'") != std::string::npos);
}

// ------------------------------------------------------------------ case (b): residual, both consumers

// graph:  y = (x + a) + (x + b), seeded at y, the slice arriving through parse_slice from an
// r97-program file written to a temp path (the parser the real block's slice will use).
//   dx = 2 dy -- x is read by two adds and both cotangents must arrive and sum
//   da = dy, db = dy
R97_TEST(slice_reference_residual_two_consumers) {
    const int64_t M = 4;
    const int64_t Wd = 3;
    const std::vector<int64_t> shape{M, Wd};

    // The program, exactly the field semantics `program_json` writes (whole-buffer operands,
    // rows 0).
    const json program_json = json::parse(R"({
        "format": "r97-program",
        "version": 1,
        "batch": [[1, 5, 9, 2]],
        "ops": [
            {"op": "add", "block": 3, "position": 0, "n": 12, "params": {},
             "operands": [
                {"buffer": "x", "dtype": "f64", "shape": [4, 3], "row_lo": 0, "rows": 0, "written": false},
                {"buffer": "a", "dtype": "f64", "shape": [4, 3], "row_lo": 0, "rows": 0, "written": false},
                {"buffer": "h", "dtype": "f64", "shape": [4, 3], "row_lo": 0, "rows": 0, "written": true}]},
            {"op": "add", "block": 3, "position": 1, "n": 12, "params": {},
             "operands": [
                {"buffer": "x", "dtype": "f64", "shape": [4, 3], "row_lo": 0, "rows": 0, "written": false},
                {"buffer": "b", "dtype": "f64", "shape": [4, 3], "row_lo": 0, "rows": 0, "written": false},
                {"buffer": "g", "dtype": "f64", "shape": [4, 3], "row_lo": 0, "rows": 0, "written": true}]},
            {"op": "add", "block": 3, "position": 2, "n": 12, "params": {},
             "operands": [
                {"buffer": "h", "dtype": "f64", "shape": [4, 3], "row_lo": 0, "rows": 0, "written": false},
                {"buffer": "g", "dtype": "f64", "shape": [4, 3], "row_lo": 0, "rows": 0, "written": false},
                {"buffer": "y", "dtype": "f64", "shape": [4, 3], "row_lo": 0, "rows": 0, "written": true}]},
            {"op": "mul", "block": 3, "position": 3, "n": 12, "params": {},
             "operands": [
                {"buffer": "x", "dtype": "f64", "shape": [4, 3], "row_lo": 0, "rows": 0, "written": false},
                {"buffer": "s", "dtype": "f64", "shape": [4, 3], "row_lo": 0, "rows": 0, "written": false},
                {"buffer": "y", "dtype": "f64", "shape": [4, 3], "row_lo": 0, "rows": 0, "written": true, "inout": true}]}
        ]
    })");
    char path[] = "/tmp/r97_slice_reference_XXXXXX";
    const int fd = mkstemp(path);
    R97_CHECK(fd >= 0);
    if (fd >= 0) {
        const std::string text = program_json.dump();
        R97_CHECK(write(fd, text.data(), text.size()) == static_cast<ssize_t>(text.size()));
        close(fd);
    }
    const slice::ProgramSlice program = slice::parse_slice(std::string(path));
    unlink(path);
    R97_CHECK(program.ops.size() == 4);

    Deterministic rng{0xb1655eedull};
    std::vector<double> xv(static_cast<size_t>(M * Wd)), av(static_cast<size_t>(M * Wd)),
        bv(static_cast<size_t>(M * Wd)), sv(static_cast<size_t>(M * Wd)),
        dy(static_cast<size_t>(M * Wd));
    for (double& v : xv) v = rng.next();
    for (double& v : av) v = rng.next();
    for (double& v : bv) v = rng.next();
    for (double& v : sv) v = rng.next();
    for (double& v : dy) v = rng.next();

    slice::SliceInput in;
    in.boundary["x"] = slice::make_value("f64", shape, xv);
    in.boundary["a"] = slice::make_value("f64", shape, av);
    in.boundary["b"] = slice::make_value("f64", shape, bv);
    in.constants["s"] = slice::make_value("f64", shape, sv);
    in.prefix = "b3resid";
    in.out_buffers = {"h", "g", "y"};
    in.leaf_buffers = {"x", "a", "b", "s"};

    Seed seed;
    seed.buffer = "y";
    seed.row_lo = 0;
    seed.row_hi = M;
    as_f32(dy, &seed.cotangent);
    in.seeds.push_back(std::move(seed));

    const slice::SliceRun got = slice::run_slice(program, in);
    slice::print_report(got, in);
    // forward closed forms: h = x + a, g = x + b, and the in-place mul then overwrites y with
    // x * s (mul_twin reads operands 0,1 and writes operand 2, elementwise.cpp:169-183).
    std::vector<double> h_want(static_cast<size_t>(M * Wd)), g_want(static_cast<size_t>(M * Wd)),
        y_want(static_cast<size_t>(M * Wd));
    for (int64_t i = 0; i < M * Wd; ++i) {
        const size_t u = static_cast<size_t>(i);
        h_want[u] = xv[u] + av[u];
        g_want[u] = xv[u] + bv[u];
        y_want[u] = xv[u] * sv[u];
    }

    // forward: h = x + a, g = x + b, and then the in-place mul: its reads are `x` (the boundary
    // value, the record's slot 0) and `s` (the constant), and it replaces `y`'s environment
    // slot with their product (mul_twin, elementwise.cpp:169-183: y = a * b of the reads).
    expect_exact("h (forward)", got.env.at("h").value.value, h_want);
    expect_exact("g (forward)", got.env.at("g").value.value, g_want);
    expect_exact("y (forward)", got.env.at("y").value.value, y_want);

    // backward, from the widened seed. The seed enters at the mul's own write (the version that
    // replaced the add's); walking back:
    //   dx += s * dyw (the mul's read of `x`)    ds += x * dyw (the mul's read of `s`;
    //   elementwise.cpp:185-206: db += dy * fwd[a] with a = x)
    // and the passthrough (options.in_place_passthrough) carries the write's cotangent, unchanged, to
    // value the mul replaced -- the add's `y`, whose adjoint routes it through BOTH of its
    // reads: the both-consumers sum at x, three routes total: dx = (2 + s) dyw, da = dyw, db = dyw.
    const std::vector<double> dyw = widened(dy);
    std::vector<double> dx(dyw.size()), da(dyw.size()), db(dyw.size()), ds(dyw.size());
    for (size_t i = 0; i < dyw.size(); ++i) {
        dx[i] = 2.0 * dyw[i] + sv[i] * dyw[i];  // add's two routes + mul's read of x
        da[i] = dyw[i];
        db[i] = dyw[i];
        ds[i] = xv[i] * dyw[i];  // mul_adjoint: db += dy * fwd[a] with a = x (elementwise.cpp:195)
    }
    const r97::InputCotangent* input_x = nullptr;
    const r97::InputCotangent* input_a = nullptr;
    const r97::InputCotangent* input_b = nullptr;
    const r97::InputCotangent* input_s = nullptr;
    for (const r97::InputCotangent& ic : got.walk.inputs) {
        if (ic.buffer == "x") input_x = &ic;
        if (ic.buffer == "a") input_a = &ic;
        if (ic.buffer == "b") input_b = &ic;
        if (ic.buffer == "s") input_s = &ic;
    }
    R97_CHECK(input_x != nullptr);
    R97_CHECK(input_a != nullptr);
    R97_CHECK(input_b != nullptr);
    R97_CHECK(input_s != nullptr);
    if (input_x != nullptr) expect_exact("x (dx)", input_x->value, dx);
    if (input_a != nullptr) expect_exact("a (da)", input_a->value, da);
    if (input_b != nullptr) expect_exact("b (db)", input_b->value, db);
    if (input_s != nullptr) expect_exact("s (ds)", input_s->value, ds);
    // no trainable leaf is in the graph: no gradient at all
    R97_CHECK(got.grads.empty());
}

// ------------------------------------------------------------------ the operand view
//
// A narrowed read is a tensor of its own: the walk's snapshot packs it over its view shape
// (src/join_common.h:264-295), and the gated-delta layer reads one fused buffer twice through
// two disjoint halves (`gdn_ab`'s A|B, gdn_in's q|k|v). The dump and the parser must carry
// view_shape/view_stride/view_off -- the real block's dump proved they do not (the constants
// hold gdn_in [512,10240] of a [512,16384] record) -- and `operand_at` must gather the read
// through them. Without either, the two halves become the same operand.
R97_TEST(slice_operand_view_gathers_the_named_half) {
    // The parser's half: the exact keys src/step_device.cpp's dump writes.
    const nlohmann::json doc = nlohmann::json::parse(R"({
      "format": "r97-program", "version": 1, "batch": [],
      "ops": [{"op": "gdn_conv_prep", "block": 0, "position": 0, "n": 1, "params": {},
               "operands": [
                 {"buffer": "ab", "dtype": "f32", "shape": [2, 4], "row_lo": 0, "rows": 2,
                  "written": false, "inout": false,
                  "view_shape": [2, 2], "view_stride": [4, 1], "view_off": 0},
                 {"buffer": "ab", "dtype": "f32", "shape": [2, 4], "row_lo": 0, "rows": 2,
                  "written": false, "inout": false,
                  "view_shape": [2, 2], "view_stride": [4, 1], "view_off": 2}]}]})");
    const slice::ProgramSlice prog = slice::parse_slice(doc);
    R97_CHECK(prog.ops.size() == 1);
    R97_CHECK(prog.ops[0].operands.size() == 2);
    const Operand& lo = prog.ops[0].operands[0];
    const Operand& hi = prog.ops[0].operands[1];
    R97_CHECK(lo.view_shape == std::vector<int64_t>({2, 2}));
    R97_CHECK(lo.view_stride == std::vector<int64_t>({4, 1}));
    R97_CHECK(lo.view_off == 0);
    R97_CHECK(hi.view_off == 2);

    // The gather's half: one [2,4] buffer, two [2,2] halves.
    slice::SliceInput in;
    const std::vector<int64_t> whole_shape{2, 4};
    std::vector<double> whole;
    for (int64_t i = 0; i < 8; ++i) whole.push_back(static_cast<double>(i));
    in.boundary["ab"] = slice::make_value("f32", whole_shape, whole);
    std::map<std::string, slice::Buffer> env;
    for (const auto& [name, value] : in.boundary) env[name] = slice::Buffer{value};

    const OpRecord& rec = prog.ops[0];
    const slice::AdValue a = slice::operand_at(rec, lo, env, in);
    const slice::AdValue b = slice::operand_at(rec, hi, env, in);
    R97_CHECK(a.shape == std::vector<int64_t>({2, 2}));
    R97_CHECK(b.shape == std::vector<int64_t>({2, 2}));
    expect_exact("ab low half", a.value, {0.0, 1.0, 4.0, 5.0});
    expect_exact("ab high half", b.value, {2.0, 3.0, 6.0, 7.0});
    // The whole buffer is not the read: without the view both halves would be {0,1,2,3,4,5,6,7}
    // and the two operands would compare equal, which is what the collapsed dump did.
    R97_CHECK(a.value != b.value);
}

// ------------------------------------------------------------------ the cold kv write
//
// The walk's cold rule is not read-only (src/step_device.cpp:1590-1594, no in-place test there):
// a written `kv:` operand this block has not written yet enters ZERO in the snapshot the adjoint
// sees, while the environment still receives the twin's production, so a later in-block read of
// the pool sees the live state. `gdn_chunk_scan` is the real block's instance: it READS `kv:2` at
// position 6 and WRITES it at position 9, and that write carries no `inout` mark -- so
// `r97::in_place_read` is false for it (include/r97/backward.h:42-50: a `kv:` write that a
// separate read in the same record names is not an in-place read), and the rule has to come from
// the plain-write path. Asserted on the snapshots themselves -- the exact values the adjoint
// reads -- not through a gradient: a plain re-write of one buffer also pulls in the join's
// passthrough, which makes a gradient proxy say things this rule is not about.
R97_TEST(slice_reference_cold_kv_write_zeroes_the_snapshot) {
    const int64_t M = 4;
    const int64_t Wd = 3;
    const std::vector<int64_t> shape{M, Wd};
    // op 0: kv:s (cold read, zeros) + b -> kv:s, a plain write of a pool nothing wrote yet.
    // op 1: the same record again, now warm: its read is b, its write is 2b.
    const std::string ops = R"([
            {"op": "add", "block": 3, "position": 0, "n": 12, "params": {},
             "operands": [
                {"buffer": "kv:s", "dtype": "f64", "shape": [4, 3], "row_lo": 0, "rows": 0, "written": false},
                {"buffer": "b", "dtype": "f64", "shape": [4, 3], "row_lo": 0, "rows": 0, "written": false},
                {"buffer": "kv:s", "dtype": "f64", "shape": [4, 3], "row_lo": 0, "rows": 0, "written": true}]},
            {"op": "add", "block": 3, "position": 1, "n": 12, "params": {},
             "operands": [
                {"buffer": "kv:s", "dtype": "f64", "shape": [4, 3], "row_lo": 0, "rows": 0, "written": false},
                {"buffer": "b", "dtype": "f64", "shape": [4, 3], "row_lo": 0, "rows": 0, "written": false},
                {"buffer": "kv:s", "dtype": "f64", "shape": [4, 3], "row_lo": 0, "rows": 0, "written": true}]}
    ])";
    const nlohmann::json doc = nlohmann::json::parse(
        R"({"format": "r97-program", "version": 1, "batch": [[1, 5, 9, 2]], "ops": )" + ops + "}");
    const slice::ProgramSlice program = slice::parse_slice(doc);
    R97_CHECK(program.ops.size() == 2);
    // The premise the rule turns on: the write is not an in-place read.
    R97_CHECK(!r97::in_place_read(program.ops[0], 2));
    R97_CHECK(!r97::in_place_read(program.ops[1], 2));

    Deterministic rng{0xc01d4b17ull};
    std::vector<double> bv(static_cast<size_t>(M * Wd));
    // `next()` is uniform over [-1, 4095] (the splitmix64 draw scaled by 2^-52); sin keeps the
    // values O(1) so a zero and a production cannot be confused by rounding.
    for (double& v : bv) v = std::sin(rng.next());

    slice::SliceInput in;
    in.boundary["b"] = slice::make_value("f64", shape, bv);
    in.prefix = "b3coldkv";
    in.out_buffers = {"kv:s"};
    in.leaf_buffers = {"b"};

    std::map<std::string, slice::Buffer> env;
    for (const auto& [name, value] : in.boundary) env[name] = slice::Buffer{value};
    r97::StoredSnapshots snaps;
    slice::run_forward(program, in, &env, &snaps);

    const std::vector<double> zeros(bv.size(), 0.0);
    std::vector<double> twice(bv.size());
    for (size_t i = 0; i < bv.size(); ++i) twice[i] = 2.0 * bv[i];
    // op 0: the read is the walk's cold zero, and so is the WRITE's snapshot ...
    const r97::AdValue* cold_read = snaps.at(0, 0);
    R97_CHECK(cold_read != nullptr);
    if (cold_read != nullptr) expect_exact("kv:s (snapshot, cold read)", cold_read->value, zeros);
    const r97::AdValue* cold_write = snaps.at(0, 2);
    R97_CHECK(cold_write != nullptr);
    if (cold_write != nullptr) {
        R97_CHECK(cold_write->shape == shape);
        expect_exact("kv:s (snapshot, cold write)", cold_write->value, zeros);
    }
    // ... op 1 reads the production and, warm, keeps its own ...
    const r97::AdValue* warm_read = snaps.at(1, 0);
    R97_CHECK(warm_read != nullptr);
    if (warm_read != nullptr) expect_exact("kv:s (snapshot, warm read)", warm_read->value, bv);
    const r97::AdValue* warm_write = snaps.at(1, 2);
    R97_CHECK(warm_write != nullptr);
    if (warm_write != nullptr)
        expect_exact("kv:s (snapshot, warm write)", warm_write->value, twice);
    // ... and the environment holds the last production either way.
    expect_exact("kv:s (forward)", env.at("kv:s").value.value, twice);
}

}  // namespace
