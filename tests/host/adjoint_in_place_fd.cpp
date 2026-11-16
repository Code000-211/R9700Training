// tests/host/adjoint_in_place_fd.cpp -- the five in-place ops of the live tape, FD-checked with
// their output aliased over their input (issue 144).
//
// An in-place operand is one schema slot holding two values -- the op reads the pre-op one and
// writes the post-op one. To express the aliased case a record marks the operand written and
// in_place, the check is handed the pre-op value among its reads and the slot's post-op value
// among the twin's outputs, and `check_adjoint` is told the slot is in place, so the finite
// differences run at the points production actually uses: the twin's output (the default) and
// the pre-op value the walk serves into the slot (R97_FD_PREOP=1).
//
// The five ops `written=1 in_place=1` on one 512-token step, none of which a test ever aliased
// before this file:
//
//   lora_expand            the engine adds the adapter's branch to the buffer the trunk's GEMM
//                          already wrote, so the write's slot is also its read.
//   rmsnorm_had_quant_i8g  writes the elementwise part's output over its own input x.
//   gdn_conv_prep          writes the convolution state over the value an earlier layer left.
//   rope                   rotates the query (or key) buffer in place.
//   kv_store               stores the keys and values over the pool's rows.
//
// For each op this file states the aliased record -- the tape's own mark, not a test's
// convenience -- and runs what can be run at it: the twin against radiance's host kernel, where
// the kernel reads the pre-op value the way the engine does (lora_expand adds to y, rope rotates
// in place, kv_store overwrites its rows, gdn_conv_prep's cold start overwrites the state), and
// the host adjoint against central differences of the twin. The twins the rows run on are the
// rows' own, so a row that differentiates at the wrong point, or forgets the value an aliased
// slot carries, fails here the way it would fail in the walk.
//
// `rmsnorm_had_quant_i8g` has no host kernel (device only), and its row reads the slot's value,
// so it is checked at the walk's point -- the pre-op value, R97_FD_PREOP -- only: at the
// twin's output the row would differentiate the norm at its own output, a point production never
// serves. The engine stores the int8 codes over the bf16 x buffer there, so the slot's "output"
// is a reinterpretation nothing differentiates; the value the backward may read the slot for is
// the part's own output, which the twin produces for it.

#include "support/adjoint_check.h"
#include "support/test.h"

#include "../../src/adjoint/host/qk.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

using namespace r97;
using namespace r97::adjoint_check;

namespace {

/* A kernel that rounds as it goes lands within about 2^-9 relative of the f64 twin; 0.02 leaves
 * the sum's rounding and the store room. The twin is exact in f64 and affine in its reads where
 * it is linearised, so central differences reproduce the adjoint far inside 1e-6 relative. */
const double bf16_tol = 0.02;
const double fd_tol = 1e-6;

/* A bf16 value, the way the container stores one: an f32 with its low significand bits cleared. */
double as_bf16(double v) { return (double)rad_bf16_to_f32(rad_f32_to_bf16((float)v)); }

AdValue values(const char* dtype, std::vector<int64_t> shape, std::vector<double> v) {
    AdValue a;
    a.dtype = dtype;
    a.shape = std::move(shape);
    a.value = std::move(v);
    return a;
}

/* The written operands' dtypes and shapes, which is all the checks want of them. */
std::vector<AdValue> specs(const std::vector<Operand>& ops) {
    std::vector<AdValue> out;
    for (const Operand& o : ops) {
        if (!o.written) continue;
        int64_t n = 1;
        for (int64_t d : o.shape) n *= d;
        out.push_back(values(o.dtype.c_str(), o.shape, std::vector<double>((size_t)n, 0.0)));
    }
    return out;
}

/* The walk's point for `check_adjoint`: the in-place slot serves the pre-op value the walk
 * serves into it, not the twin's output. Scoped, so a run that sets the variable around the
 * whole suite keeps its value. */
struct PreopPoint {
    bool was_set = false;
    std::string saved;
    PreopPoint() {
        if (const char* e = std::getenv("R97_FD_PREOP")) {
            was_set = true;
            saved = e;
        }
        setenv("R97_FD_PREOP", "1", 1);
    }
    ~PreopPoint() {
        if (was_set) setenv("R97_FD_PREOP", saved.c_str(), 1);
        else unsetenv("R97_FD_PREOP");
    }
};

void both_points(const OpRecord& rec, const std::vector<AdValue>& in,
                 const std::vector<AdValue>& out, uint64_t seed,
                 const std::vector<size_t>& in_place, const char* what) {
    const FdReport post = check_adjoint(rec, in, out, fd_tol, seed, in_place);
    std::printf("%-24s %-28s post-op point: adjoint rel err %.3g\n", what, rec.op.c_str(),
                post.rel);
    require(post);
    PreopPoint preop;
    const FdReport pre = check_adjoint(rec, in, out, fd_tol, seed ^ 0x5ca1eu, in_place);
    std::printf("%-24s %-28s pre-op point:  adjoint rel err %.3g\n", what, rec.op.c_str(),
                pre.rel);
    require(pre);
}

// ------------------------------------------------------------------ lora_expand

/* One site of the adapter pool: a row band of M over the trunk's K, N out, rank r, S pool rows.
 * Row 1 is a base row spelled -1 and row 2 one spelled past the pool, so the write's rule that
 * a base row keeps its old value unrounded is part of what is checked. */
struct Site {
    const char* name;
    int64_t M, K, N, R, S;
};

const Site sites[] = {
    {"q-proj", 6, 128, 128, 16, 2},
    {"ffn-up", 5, 128, 256, 4, 1},
    {"down-proj", 4, 256, 128, 4, 1},
};

int32_t slot_of(const Site& s, int64_t m) {
    if (m == 1) return -1;
    if (m == 2) return (int32_t)s.S;
    return (int32_t)(m % s.S);
}

/* t, slot, lora_b, scale, then the write y marked in place: the record `src/join_common.h`
 * gives every lora_expand of the live tape, whose buffer is the trunk's output the engine adds
 * to. */
OpRecord expand_rec(const Site& s, const char* op = "lora_expand") {
    OpRecord rec;
    rec.op = op;
    rec.params = {{"M", s.M}, {"N", s.N}, {"R", s.R}, {"S", s.S}, {"dtype", "bf16"}};
    rec.n = s.M;
    rec.operands = {{"t", "f32", {s.M, s.R}, 0, s.M, false},
                    {"slot", "i32", {s.M}, 0, s.M, false},
                    {"lora_b", "bf16", {s.S, s.N, s.R}, 0, s.S, false},
                    {"scale", "f32", {s.S, s.N}, 0, s.S, false},
                    {"y", "bf16", {s.M, s.N}, 0, s.M, true}};
    rec.operands[4].in_place = true;
    return rec;
}

/* The reads in schema order -- the write's old value is among them, last -- and the write's
 * spec. The values are the trunk's: bf16 exactly, t in f32, one positive f32 scale per column
 * so a column read from the wrong pool row shows. */
void expand_case(const Site& s, const OpRecord& rec, uint64_t seed, std::vector<AdValue>* in,
                 std::vector<AdValue>* out) {
    in->clear();
    for (const Operand& o : rec.operands) {
        if (o.written && o.in_place) continue;  // its old value enters the reads below
        AdValue v;
        v.dtype = o.dtype;
        v.shape = o.shape;
        int64_t n = 1;
        for (int64_t d : o.shape) n *= d;
        v.value.resize((size_t)n);
        if (o.buffer == "slot") {
            for (int64_t i = 0; i < n; ++i) v.value[(size_t)i] = (double)slot_of(s, i);
        } else if (o.buffer == "t") {
            for (int64_t i = 0; i < n; ++i) v.value[(size_t)i] = (double)(float)urand(&seed);
        } else if (o.buffer == "scale") {
            for (int64_t i = 0; i < n; ++i)
                v.value[(size_t)i] = (double)(float)(2.0 / (1 + (i / s.N) % 2));
        } else {
            for (int64_t i = 0; i < n; ++i) v.value[(size_t)i] = as_bf16(urand(&seed));
        }
        in->push_back(std::move(v));
    }
    const Operand& y = rec.operands[4];
    AdValue base;
    base.dtype = y.dtype;
    base.shape = y.shape;
    int64_t ny = 1;
    for (int64_t d : y.shape) ny *= d;
    base.value.resize((size_t)ny);
    for (int64_t i = 0; i < ny; ++i) base.value[(size_t)i] = as_bf16(urand(&seed));
    in->push_back(std::move(base));  // the read list carries the old value last
    *out = specs(rec.operands);
}

R97_TEST(in_place_lora_expand) {
    for (const Site& s : sites) {
        for (const char* op : {"lora_expand", "lora_expand_mm"}) {
            const OpRecord rec = expand_rec(s, op);
            std::vector<AdValue> in, out;
            expand_case(s, rec, 0x9e3779b9ull ^ (uint64_t)s.M, &in, &out);
            const KernelReport k = check_kernel(rec, in, out, bf16_tol, {4});
            std::printf("%-24s %-28s kernel max err %.6g\n", s.name, rec.op.c_str(), k.max_err);
            require(k);
        }
        /* The mm variant rounds t to bf16 before the sum: a step function, so its twin is
         * piecewise constant and a central difference of it is the derivative of nothing. The
         * plain op is the one the finite differences run on, at both points the slot can hold. */
        const OpRecord rec = expand_rec(s);
        std::vector<AdValue> in, out;
        expand_case(s, rec, 0x5deece66dull ^ (uint64_t)s.K, &in, &out);
        both_points(rec, in, out, 0x1000u + (uint64_t)(s.M * 31 + s.K * 7 + s.N * 3), {4}, s.name);
    }
}

// ------------------------------------------------------------------ rope

/* A rope geometry: M tokens over the fused [q|k|v] row's width, the NeoX rotation the table
 * carries. The record the live tape gives the op is the in-place write and the positions, and
 * the twin takes the buffer's prior value as its first read. */
struct RopeCase {
    const char* name;
    int64_t M, n_head, n_head_kv, head_dim, rotary_dim;
    double theta;
};

std::vector<RopeCase> rope_cases() {
    return {
        {"grouped", 5, 4, 2, 16, 16, 10000.0},
        {"partial", 7, 2, 2, 16, 8, 1e6},  // half the head rotated
    };
}

OpRecord rope_rec(const RopeCase& c) {
    OpRecord rec;
    rec.op = "rope";
    rec.params = {{"M", c.M},
                  {"head_dim", c.head_dim},
                  {"n_head", c.n_head},
                  {"n_head_kv", c.n_head_kv},
                  {"rotary_dim", c.rotary_dim},
                  {"theta", (int64_t)c.theta},
                  {"scale", std::string("1.0")},
                  {"mode", std::string("neox")}};
    rec.n = c.M;
    const int64_t row = (c.n_head + 2 * c.n_head_kv) * c.head_dim;
    rec.operands = {{"q", "bf16", {c.M, row}, 0, c.M, true},
                    {"pos", "i32", {c.M}, 0, c.M, false}};
    rec.operands[0].in_place = true;
    return rec;
}

R97_TEST(in_place_rope) {
    for (const RopeCase& c : rope_cases()) {
        const OpRecord rec = rope_rec(c);
        std::vector<AdValue> in;
        const int64_t row = (c.n_head + 2 * c.n_head_kv) * c.head_dim;
        const int64_t n = c.M * row;
        uint64_t seed = 0x9e3779b9ull + (uint64_t)c.M;
        std::vector<double> q((size_t)n);
        for (double& v : q) v = as_bf16(urand(&seed));
        std::vector<double> pos((size_t)c.M);
        for (int64_t t = 0; t < c.M; ++t) pos[(size_t)t] = (double)t;  // the table has M rows
        in.push_back(values("bf16", {c.M, row}, std::move(q)));
        in.push_back(values("i32", {c.M}, std::move(pos)));
        const std::vector<AdValue> out = specs(rec.operands);
        const KernelReport k = check_kernel(rec, in, out, bf16_tol, {0});
        std::printf("%-24s %-28s kernel max err %.6g\n", c.name, rec.op.c_str(), k.max_err);
        require(k);
        both_points(rec, in, out, 0x2a00u + (uint64_t)c.M, {0}, c.name);
    }
}

// ------------------------------------------------------------------ kv_store

/* A paged pool: tokens of k and v stored at the rows their slot mapping names, one block of
 * block_size rows per (page, head). The pool is bigger than the mapping reaches, so rows the
 * store leaves alone are part of the aliased slot -- the part of the old pool the write does not
 * overwrite, the only part whose old value the store's output carries. */
struct PoolCase {
    const char* name;
    int64_t tokens, kv_heads, head_dim, block_size;
    int64_t n_blocks() const { return (tokens + block_size - 1) / block_size + 1; }
};

std::vector<PoolCase> pool_cases() {
    return {
        {"sparse-pages", 5, 2, 16, 2},  // 3 pages in a pool of 4
        {"one-page", 4, 3, 8, 4},       // 1 page in a pool of 2
    };
}

OpRecord kv_rec(const PoolCase& c) {
    OpRecord rec;
    rec.op = "kv_store";
    rec.params = {{"head_dim", c.head_dim}, {"block_size", c.block_size},
                  {"n_head_kv", c.kv_heads}};
    rec.n = c.tokens;
    rec.operands = {
        {"k", "bf16", {c.tokens, c.kv_heads, c.head_dim}, 0, c.tokens, false},
        {"v", "bf16", {c.tokens, c.kv_heads, c.head_dim}, 0, c.tokens, false},
        {"slot", "i32", {c.tokens}, 0, c.tokens, false},
        {"pool", "bf16", {c.n_blocks(), c.kv_heads, c.block_size, 2 * c.head_dim}, 0,
         c.n_blocks(), true}};
    rec.operands[3].in_place = true;
    return rec;
}

/* The slot mapping: the pages in a shuffled block table, token t at page t/block_size, so the
 * pages the table names are distinct and the pool's last block is left to its old value. */
std::vector<double> kv_slots(const PoolCase& c) {
    const int64_t pages = (c.tokens + c.block_size - 1) / c.block_size;
    std::vector<int32_t> table((size_t)pages);
    std::iota(table.begin(), table.end(), 0);
    std::reverse(table.begin(), table.end());  // a fixed shuffle: page 0 last
    std::vector<double> slot((size_t)c.tokens);
    for (int64_t t = 0; t < c.tokens; ++t)
        slot[(size_t)t] = (double)r97::adjoint::qk::kv_slot(table.data(), 1, c.block_size, 0, t);
    return slot;
}

R97_TEST(in_place_kv_store) {
    for (const PoolCase& c : pool_cases()) {
        const OpRecord rec = kv_rec(c);
        uint64_t seed = 0x9e3779b9ull + (uint64_t)c.tokens;
        const int64_t head_values = c.kv_heads * c.head_dim;
        std::vector<double> kv_k((size_t)(c.tokens * head_values)),
            kv_v((size_t)(c.tokens * head_values));
        for (double& x : kv_k) x = as_bf16(urand(&seed));
        for (double& x : kv_v) x = as_bf16(urand(&seed));
        const std::vector<double> slot = kv_slots(c);
        const std::vector<AdValue> in = {
            values("bf16", {c.tokens, c.kv_heads, c.head_dim}, std::move(kv_k)),
            values("bf16", {c.tokens, c.kv_heads, c.head_dim}, std::move(kv_v)),
            values("i32", {c.tokens}, std::move(slot)),
        };
        const Operand& pool = rec.operands[3];
        const int64_t n_pool = pool.shape[0] * pool.shape[1] * pool.shape[2] * pool.shape[3];
        std::vector<double> old((size_t)n_pool);
        for (double& x : old) x = as_bf16(urand(&seed));
        std::vector<AdValue> reads = in;
        reads.push_back(values(pool.dtype.c_str(), pool.shape, std::move(old)));
        const std::vector<AdValue> out = specs(rec.operands);
        const KernelReport k = check_kernel(rec, reads, out, bf16_tol, {3});
        std::printf("%-24s %-28s kernel max err %.6g\n", c.name, rec.op.c_str(), k.max_err);
        require(k);
        both_points(rec, reads, out, 0x3b00u + (uint64_t)c.tokens, {3}, c.name);
    }
}

// ------------------------------------------------------------------ gdn_conv_prep

/* The convolution's geometry: k and v per head, head counts, the convolution's width and the
 * gate's chunk, and the sequence lengths of one batch. */
struct ConvShape {
    const char* name;
    int64_t k, v, hk, hv, width, chunk;
    std::vector<int64_t> lens;
    int64_t conv_dim() const { return 2 * hk * k + hv * v; }
    int64_t tokens() const {
        int64_t t = 0;
        for (int64_t n : lens) t += n;
        return t;
    }
    int64_t seqs() const { return (int64_t)lens.size(); }
};

std::vector<ConvShape> conv_cases() {
    return {
        {"tiny-fixture", 16, 16, 2, 4, 4, 4, {8}},
        {"ragged-short", 8, 8, 2, 2, 3, 2, {5, 1}},  // one sequence shorter than the window
    };
}

OpRecord conv_rec(const ConvShape& s) {
    const int64_t t = s.tokens();
    const int64_t cw = s.conv_dim();
    const int64_t seq = s.seqs();
    const int64_t keep = s.width - 1;
    OpRecord rec;
    rec.op = "gdn_conv_prep";
    rec.params = {{"M", t},
                  {"head_k", s.k},
                  {"head_v", s.v},
                  {"chunk", s.chunk},
                  {"conv_width", s.width},
                  {"dtype", std::string("bf16")}};
    rec.n = t;
    rec.operands = {
        {"x", "bf16", {t, cw}, 0, t, false},
        {"w", "bf16", {cw, s.width}, 0, cw, false},
        {"b", "bf16", {cw}, 0, 1, false},
        {"A_log", "f32", {s.hv}, 0, 1, false},
        {"dt_bias", "f32", {s.hv}, 0, 1, false},
        {"conv_state", "bf16", {seq, cw, keep}, 0, seq, true},
        {"cu", "i32", {seq + 1}, 0, 1, false},
        {"a", "bf16", {t, s.hv}, 0, t, false},
        {"b_gate", "bf16", {t, s.hv}, 0, t, false},
        {"cache_idx", "i32", {seq}, 0, seq, false},
        {"has_init", "i32", {seq}, 0, seq, false},
        {"q", "bf16", {t, s.hk, s.k}, 0, t, true},
        {"k", "bf16", {t, s.hk, s.k}, 0, t, true},
        {"v", "bf16", {t, s.hv, s.v}, 0, t, true},
        {"g", "f32", {t, s.hv}, 0, t, true},
        {"beta", "f32", {t, s.hv}, 0, t, true},
    };
    // The live tape's mark: the state's write is over the value an earlier layer left in the
    // buffer. The forward this family serves is a cold prefill (has_init zero), so the state the
    // op writes is the tail of x alone and the old value is positioned, not used.
    rec.operands[5].in_place = true;
    return rec;
}

R97_TEST(in_place_gdn_conv_prep) {
    for (const ConvShape& s : conv_cases()) {
        const OpRecord rec = conv_rec(s);
        uint64_t seed = 0x9e3779b9ull + (uint64_t)s.tokens();
        // The state's old value, generated first so the reads below can carry it at the
        // state's schema position: the read list is the schema with the plain writes out,
        // and the in-place write's read keeps its place between dt_bias and cu.
        const Operand& state = rec.operands[5];
        int64_t ns = 1;
        for (int64_t d : state.shape) ns *= d;
        std::vector<double> old((size_t)ns);
        for (double& x : old) x = as_bf16(urand(&seed));
        std::vector<AdValue> in;
        for (const Operand& o : rec.operands) {
            if (o.written && !o.in_place) continue;
            if (o.in_place) {
                in.push_back(values(o.dtype.c_str(), o.shape, old));
                continue;
            }
            AdValue v;
            v.dtype = o.dtype;
            v.shape = o.shape;
            int64_t n = 1;
            for (int64_t d : o.shape) n *= d;
            v.value.resize((size_t)n);
            if (o.buffer == "cu") {
                v.value[0] = 0.0;
                for (size_t i = 0; i < s.lens.size(); ++i)
                    v.value[i + 1] = v.value[i] + (double)s.lens[i];
            } else if (o.buffer == "cache_idx") {
                for (int64_t i = 0; i < n; ++i) v.value[(size_t)i] = (double)i;
            } else if (o.buffer == "has_init") {
                v.value.assign(v.value.size(), 0.0);  // the cold start
            } else if (o.dtype == "f32") {
                for (int64_t i = 0; i < n; ++i)
                    v.value[(size_t)i] = (double)(float)(1.0 + 0.5 * urand(&seed));
            } else {
                for (int64_t i = 0; i < n; ++i) v.value[(size_t)i] = as_bf16(urand(&seed));
            }
            in.push_back(std::move(v));
        }
        const std::vector<AdValue> out = specs(rec.operands);
        const KernelReport k = check_kernel(rec, in, out, bf16_tol, {5});
        std::printf("%-24s %-28s kernel max err %.6g\n", s.name, rec.op.c_str(), k.max_err);
        require(k);
        both_points(rec, in, out, 0x4c00u + (uint64_t)s.tokens(), {5}, s.name);
    }
}

// ------------------------------------------------------------------ rmsnorm_had_quant_i8g

/* The int8 trunk's residual norm: x, the gain, the codes and the scale, and the norm's output
 * over x -- the one of the five whose aliased slot the tape marks on the op's own input. The
 * base point of the linearised row is the part's own output, the activation the quantiser saw,
 * appended after the writes. */
struct FusedShape {
    const char* name;
    int64_t rows, n, group;
    int64_t ngrp() const {
        const int64_t g = group <= 0 || group > n ? n : group;
        return (n + g - 1) / g;
    }
};

std::vector<FusedShape> fused_cases() {
    return {
        {"trunk-128", 5, 128, 128},
        {"two-groups", 3, 64, 32},
    };
}

OpRecord fused_rec(const FusedShape& s, bool linearised) {
    OpRecord rec;
    rec.op = linearised ? "rmsnorm_had_quant_i8g_ste" : "rmsnorm_had_quant_i8g";
    rec.params = {{"M", s.rows}, {"n", s.n}, {"group", s.group}, {"dtype", std::string("bf16")},
                  {"eps", std::string("1e-6")}};
    rec.n = s.rows;
    rec.operands = {{"x", "bf16", {s.rows, s.n}, 0, s.rows, true},
                    {"w", "f32", {s.n}, 0, 1, false},
                    {"codes", "i8", {s.rows, s.n}, 0, s.rows, true},
                    {"scale", "f32", {s.rows, s.ngrp()}, 0, s.rows, true}};
    rec.operands[0].in_place = true;
    if (linearised)
        rec.operands.push_back({"base", "bf16", {s.rows, s.n}, 0, s.rows, false});
    return rec;
}

/* The part's output over the reads, the way the fused row composes it: the part's own twin on
 * the record's parameters. For the linearised record it is also the base point. */
std::vector<double> part_output(const OpRecord& rec, const std::vector<AdValue>& reads) {
    std::string err;
    TwinFn part = twin_find("rmsnorm", rec.params, &err);
    if (part == nullptr) fail(std::string("no twin for the rmsnorm part: ") + err);
    std::vector<AdValue> y;
    part(rec, reads, &y);
    if (y.empty() || y[0].value.size() != reads[0].value.size())
        fail("rmsnorm part wrote no activation");
    return y[0].value;
}

R97_TEST(in_place_rmsnorm_had_quant_i8g) {
    /* The frozen slots are schema positions, and aliasing keeps the positions: the gain at 1,
     * the base point at 4 under _ste. A row that moved a slot with the mark would mis-freeze the
     * walk, so the statement is here, on the aliased record itself. */
    const FusedShape frozen_shape = {"frozen", 5, 128, 128};
    for (bool linearised : {false, true}) {
        const OpRecord rec = fused_rec(frozen_shape, linearised);
        const std::vector<int> frozen =
            linearised ? std::vector<int>{1, 4} : std::vector<int>{1};
        R97_CHECK(adjoint_frozen(rec.op, Side::host, rec.params) == frozen);
    }
    for (const FusedShape& s : fused_cases()) {
        uint64_t seed = 0x5deece66dull ^ (uint64_t)s.n;
        std::vector<double> x((size_t)(s.rows * s.n));
        for (double& v : x) v = as_bf16(urand(&seed));
        std::vector<double> w((size_t)s.n);
        for (double& v : w) v = (double)(float)(1.0 + 0.5 * urand(&seed));
        const std::vector<AdValue> reads = {values("bf16", {s.rows, s.n}, x),
                                            values("f32", {s.n}, std::move(w))};
        const OpRecord rec = fused_rec(s, true);
        const std::vector<double> part = part_output(rec, reads);
        std::vector<AdValue> in = reads;
        in.push_back(values("bf16", {s.rows, s.n}, std::move(part)));  // the base point, last
        const std::vector<AdValue> out = specs(rec.operands);
        /* The row reads the slot's value, so it is checked at the walk's point only: the
         * pre-op value the walk serves into the slot. At the twin's output the row would
         * differentiate the norm at its own output, a point production never uses. */
        PreopPoint preop;
        const FdReport fd = check_adjoint(rec, in, out, fd_tol, seed ^ 0xc0ffeeull, {0});
        std::printf("%-24s %-28s pre-op point: adjoint rel err %.3g\n", s.name, rec.op.c_str(),
                    fd.rel);
        require(fd);
    }
}

}  // namespace
