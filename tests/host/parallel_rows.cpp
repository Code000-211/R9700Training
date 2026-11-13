// tests/host/parallel_rows.cpp -- the host rows the parallel for now serves, held to the serial
// answer: every family is run once with R97_THREADS=1 and once with R97_THREADS=8, and the twin's
// outputs and the adjoint's cotangents are compared byte for byte.
//
// The promise of src/parallel.h is that a partition changes how many cores do the work, never what
// any output element sums, so a tolerance would hide exactly the bug this test exists to catch. The
// sizes cross the rows' chunk boundaries (a row chunk of 2, a flat chunk of 4096) so the tail chunk
// is cut through, and one case per family is smaller than a chunk, where the row must run inline.
//
// Each case goes through the registry, the way a walk finds the row: twin_find and adjoint_find by
// the record's op name and parameters. The adjoint is handed the twin's own forward values, the
// cotangents of what it wrote and a zeroed cotangent for every operand it read.
#include "r97/adjoint.h"
#include "support/test.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace r97;

namespace {

/* R97_THREADS for the duration of one scope, restored on the way out: the count is read per call,
 * so a test that sets it must leave the runner's value behind. */
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

/* A deterministic stream in [-1, 1): every case's values are a function of its seed, so a failure
 * reproduces without the file recording any of them. */
double next_value(uint64_t* s) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    return (double)(*s >> 11) / (double)(1ull << 53) * 2.0 - 1.0;
}

AdValue value(const char* dtype, std::vector<int64_t> shape, double fill = 0.0) {
    AdValue v;
    v.dtype = dtype;
    v.shape = std::move(shape);
    int64_t n = 1;
    for (int64_t d : v.shape) n *= d;
    v.value.assign((size_t)n, fill);
    return v;
}

AdValue filled(const char* dtype, std::vector<int64_t> shape, uint64_t* seed) {
    AdValue v = value(dtype, std::move(shape));
    for (double& x : v.value) x = next_value(seed);
    return v;
}

std::vector<AdValue> run_twin(const OpRecord& rec, const std::vector<AdValue>& reads,
                              const char* threads) {
    std::string err;
    TwinFn twin = twin_find(rec.op, rec.params, &err);
    if (twin == nullptr) throw std::runtime_error(err);
    ThreadsScope scope(threads);
    std::vector<AdValue> out;
    twin(rec, reads, &out);
    return out;
}

std::vector<AdValue> run_adjoint(const OpRecord& rec, const std::vector<AdValue>& fwd,
                                 const std::vector<AdValue>& cot, const char* threads) {
    std::string err;
    AdjointFn adj = adjoint_find(rec.op, Side::host, rec.params, &err);
    if (adj == nullptr) throw std::runtime_error(err);
    std::vector<AdValue> dc = fwd;
    for (AdValue& v : dc) v.value.assign(v.value.size(), 0.0);
    ParamSet grads;
    AdjointCall call{rec, fwd, cot, &dc, &grads, nullptr, {}, false};
    ThreadsScope scope(threads);
    adj(call);
    return dc;
}

void same_bytes(const std::string& what, const std::vector<AdValue>& a,
                const std::vector<AdValue>& b) {
    if (a.size() != b.size())
        throw std::runtime_error(what + ": operand count differs across thread counts");
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].value.size() != b[i].value.size())
            throw std::runtime_error(what + ": operand " + std::to_string(i) + " differs in size");
        if (!a[i].value.empty() &&
            std::memcmp(a[i].value.data(), b[i].value.data(),
                        a[i].value.size() * sizeof(double)) != 0)
            throw std::runtime_error(what + ": operand " + std::to_string(i) +
                                     " differs between one thread and eight");
    }
}

/* One family's case: the record and the values of its read operands. The twin runs at one and
 * eight threads and its outputs are compared; then the adjoint does, over the twin's own forward
 * values and one cotangent per write. */
void check_family(const std::string& label, const OpRecord& rec, const std::vector<AdValue>& reads,
                  uint64_t seed) {
    const std::vector<AdValue> out_one = run_twin(rec, reads, "1");
    const std::vector<AdValue> out_eight = run_twin(rec, reads, "8");
    same_bytes(label + " twin", out_one, out_eight);

    std::vector<AdValue> fwd(rec.operands.size());
    size_t rd = 0, wr = 0;
    for (size_t i = 0; i < rec.operands.size(); ++i) {
        if (rec.operands[i].written) {
            if (wr >= out_one.size())
                throw std::runtime_error(label + ": the twin wrote fewer operands than the record");
            fwd[i] = out_one[wr++];
        } else {
            if (rd >= reads.size())
                throw std::runtime_error(label + ": the record reads more operands than were given");
            fwd[i] = reads[rd++];
        }
    }
    std::vector<AdValue> cot;
    for (const AdValue& v : out_one) {
        AdValue c = v;
        for (double& x : c.value) x = next_value(&seed);
        cot.push_back(std::move(c));
    }
    const std::vector<AdValue> dc_one = run_adjoint(rec, fwd, cot, "1");
    const std::vector<AdValue> dc_eight = run_adjoint(rec, fwd, cot, "8");
    same_bytes(label + " adjoint", dc_one, dc_eight);
}

/* ------------------------------------------------------------------ elementwise */

OpRecord unary_rec(const char* op, int64_t m, int64_t n) {
    OpRecord rec;
    rec.op = op;
    rec.params = {{"M", m}, {"n", n}, {"dtype", "bf16"}};
    rec.operands = {{"x", "bf16", {m, n}, 0, m, false},
                    {"y", "bf16", {m, n}, 0, m, true}};
    return rec;
}

OpRecord mul_rec(int64_t m, int64_t n, bool one_row) {
    OpRecord rec;
    rec.op = "mul";
    rec.params = {{"M", m}, {"n", n}, {"dtype", "bf16"}};
    rec.operands = {{"a", "bf16", {m, n}, 0, m, false},
                    {"b", "bf16", one_row ? std::vector<int64_t>{n} : std::vector<int64_t>{m, n}, 0,
                     one_row ? 1 : m, false},
                    {"y", "bf16", {m, n}, 0, m, true}};
    return rec;
}

void elementwise_cases() {
    /* Flat passes cross the 4096-element chunk; the row passes cross the two-row chunk. The small
     * shapes sit below both and must stay inline. */
    struct Shape {
        const char* name;
        int64_t m, n;
    };
    const Shape shapes[] = {{"40 x 128 (crosses the chunk)", 40, 128},
                            {"5 x 17 (below the chunk)", 5, 17}};
    for (const Shape& s : shapes) {
        uint64_t seed = 0x9e3779b97f4a7c15ull;
        for (const char* op : {"scale", "sigmoid", "silu", "cast"}) {
            const OpRecord rec = unary_rec(op, s.m, s.n);
            std::vector<AdValue> reads = {filled("bf16", {s.m, s.n}, &seed)};
            if (std::string(op) == "scale") reads.push_back(value("bf16", {1}, 0.5));
            check_family(std::string(op) + " " + s.name, rec, reads, seed);
        }
        /* mul with the whole block and with one broadcast row: the broadcast adjoint splits over
         * the row's columns, the other over the elements. */
        {
            const OpRecord rec = mul_rec(s.m, s.n, false);
            std::vector<AdValue> reads = {filled("bf16", {s.m, s.n}, &seed),
                                          filled("bf16", {s.m, s.n}, &seed)};
            check_family(std::string("mul whole ") + s.name, rec, reads, seed);
        }
        {
            const OpRecord rec = mul_rec(s.m, s.n, true);
            std::vector<AdValue> reads = {filled("bf16", {s.m, s.n}, &seed),
                                          filled("bf16", {s.n}, &seed)};
            check_family(std::string("mul broadcast ") + s.name, rec, reads, seed);
        }
        /* act_mul: the two gradients are per element. */
        {
            OpRecord rec;
            rec.op = "act_mul";
            rec.params = {{"M", s.m}, {"n", s.n}, {"act", std::string("silu")},
                          {"dtype", "bf16"}};
            rec.operands = {{"a", "bf16", {s.m, s.n}, 0, s.m, false},
                            {"b", "bf16", {s.m, s.n}, 0, s.m, false},
                            {"y", "bf16", {s.m, s.n}, 0, s.m, true}};
            std::vector<AdValue> reads = {filled("bf16", {s.m, s.n}, &seed),
                                          filled("bf16", {s.m, s.n}, &seed)};
            check_family(std::string("act_mul ") + s.name, rec, reads, seed);
        }
        /* silu_mul reads the fused [rows, 2n] gate_up row. */
        {
            OpRecord rec;
            rec.op = "silu_mul";
            rec.params = {{"M", s.m}, {"n", s.n}, {"dtype", "bf16"}};
            rec.operands = {{"gate_up", "bf16", {s.m, 2 * s.n}, 0, s.m, false},
                            {"y", "bf16", {s.m, s.n}, 0, s.m, true}};
            std::vector<AdValue> reads = {filled("bf16", {s.m, 2 * s.n}, &seed)};
            check_family(std::string("silu_mul ") + s.name, rec, reads, seed);
        }
        /* rmsnorm: one row's mean is the row's own reduction, so the partition is over rows. */
        {
            OpRecord rec;
            rec.op = "rmsnorm";
            rec.params = {{"M", s.m}, {"n", s.n}, {"eps", std::string("1e-06")},
                          {"dtype", "bf16"}, {"wadd", std::string("1")},
                          {"mult", std::string("1")}};
            rec.operands = {{"x", "bf16", {s.m, s.n}, 0, s.m, false},
                            {"w", "f32", {s.n}, 0, 1, false},
                            {"y", "bf16", {s.m, s.n}, 0, s.m, true}};
            std::vector<AdValue> reads = {filled("bf16", {s.m, s.n}, &seed),
                                          filled("f32", {s.n}, &seed)};
            check_family(std::string("rmsnorm ") + s.name, rec, reads, seed);
        }
    }
    /* mul's broadcast pass needs more than one element chunk of columns before it splits: three
     * rows over an 8192-wide row is the shape a real block's gate uses. */
    {
        uint64_t seed = 0x1234567890abcdefull;
        const OpRecord rec = mul_rec(3, 8192, true);
        std::vector<AdValue> reads = {filled("bf16", {3, 8192}, &seed),
                                      filled("bf16", {8192}, &seed)};
        check_family("mul broadcast 3 x 8192 (crosses the element chunk)", rec, reads, seed);
    }
}

/* ------------------------------------------------------------------ had_quant_act_i8g */

void quant_cases() {
    struct Shape {
        const char* name;
        int64_t rows, n, group;
    };
    const Shape shapes[] = {{"5 x 512, groups of 128", 5, 512, 128},
                            {"1 x 128 (below the row chunk)", 1, 128, 128}};
    for (const Shape& s : shapes) {
        for (const char* op : {"had_quant_act_i8g", "quant_act_i8g"}) {
            uint64_t seed = 0x51ed270b1c3d4e5full;
            OpRecord rec;
            rec.op = op;
            rec.params = {{"M", s.rows}, {"n", s.n}, {"group", s.group},
                          {"dtype", "bf16"}};
            const int64_t ngrp = (s.n + s.group - 1) / s.group;
            rec.operands = {{"x", "bf16", {s.rows, s.n}, 0, s.rows, false},
                            {"codes", "i8", {s.rows, s.n}, 0, s.rows, true},
                            {"scale", "f32", {s.rows, ngrp}, 0, s.rows, true}};
            std::vector<AdValue> reads = {filled("bf16", {s.rows, s.n}, &seed)};
            check_family(std::string(op) + " " + s.name, rec, reads, seed);
        }
    }
}

/* ------------------------------------------------------------------ gemm_nt_q_gated */

void gated_cases() {
    struct Shape {
        const char* name;
        int64_t m, k, n, group;
    };
    const Shape shapes[] = {{"5 x 64 x 64, groups of 16", 5, 64, 64, 16},
                            {"1 x 32 x 32, groups of 16 (one row, below the chunk)", 1, 32, 32,
                             16}};
    for (const Shape& s : shapes) {
        uint64_t seed = 0x2545f4914f6cdd1dull;
        const int64_t half = s.n / 2, groups_k = (s.k + s.group - 1) / s.group,
                      groups_n = half / s.group;
        OpRecord rec;
        rec.op = "gemm_nt_q_gated";
        rec.params = {{"M", s.m}, {"K", s.k}, {"N", s.n}, {"group", s.group},
                      {"act", std::string("silu")}, {"dtype", "bf16"}};
        rec.operands = {{"a", "bf16", {s.m, s.k}, 0, s.m, false},
                        {"a_scale", "f32", {s.m, groups_k}, 0, s.m, false},
                        {"b", "bf16", {s.n, s.k}, 0, s.n, false},
                        {"b_scale", "f32", {s.n, groups_k}, 0, s.n, false},
                        {"q", "i8", {s.m, half}, 0, s.m, true},
                        {"scale", "f32", {s.m, groups_n}, 0, s.m, true},
                        {"out_bf16", "bf16", {s.m, half}, 0, s.m, true}};
        std::vector<AdValue> reads = {filled("bf16", {s.m, s.k}, &seed),
                                      filled("f32", {s.m, groups_k}, &seed),
                                      filled("bf16", {s.n, s.k}, &seed),
                                      filled("f32", {s.n, groups_k}, &seed)};
        check_family(std::string("gemm_nt_q_gated ") + s.name, rec, reads, seed);
    }
}

/* ------------------------------------------------------------------ attn_paged */

void attention_cases() {
    struct Shape {
        const char* name;
        int64_t rows, q_heads, head_dim, block_size, n_blocks, ctx, max_blocks;
    };
    const Shape shapes[] = {
        {"6 rows, 2 heads of 4, 4 keys", 6, 2, 4, 4, 2, 4, 1},
        {"16 rows, 2 heads of 8, 12 keys", 16, 2, 8, 8, 2, 12, 2},
    };
    for (const Shape& s : shapes) {
        uint64_t seed = 0x6a09e667f3bcc909ull;
        OpRecord rec;
        rec.op = "attn_paged";
        rec.params = {{"head_dim", s.head_dim}, {"gqa", s.q_heads}, {"causal", 1}};
        rec.operands = {
            {"q", "bf16", {s.rows, s.q_heads, s.head_dim}, 0, s.rows, false},
            {"kv_cache", "f32", {s.n_blocks, 1, s.block_size, 2 * s.head_dim}, 0, s.n_blocks,
             false},
            {"block_table", "i32", {1, s.max_blocks}, 0, 1, false},
            {"seqused", "i32", {1}, 0, 1, false},
            {"cu_seqlens", "i32", {2}, 0, 1, false},
            {"out", "bf16", {s.rows, s.q_heads, s.head_dim}, 0, s.rows, true},
        };
        AdValue table = value("i32", {1, s.max_blocks});
        for (int64_t i = 0; i < s.max_blocks; ++i) table.value[(size_t)i] = (double)i;
        AdValue seqused = value("i32", {1});
        seqused.value[0] = (double)s.ctx;
        AdValue cu = value("i32", {2});
        cu.value[0] = 0.0;
        cu.value[1] = (double)s.rows;
        std::vector<AdValue> reads = {filled("bf16", {s.rows, s.q_heads, s.head_dim}, &seed),
                                      filled("f32", {s.n_blocks, 1, s.block_size,
                                                     2 * s.head_dim}, &seed),
                                      table, seqused, cu};
        check_family(std::string("attn_paged ") + s.name, rec, reads, seed);
    }
}

/* ------------------------------------------------------------------ rope and kv_store */

void qk_cases() {
    /* rope: one token row per unit; the table is constant and never differentiated. */
    {
        uint64_t seed = 0xbb67ae8584caa73bull;
        const int64_t tokens = 5, n_head = 2, n_head_kv = 1, head_dim = 4, rotary = 4;
        const int64_t row = (n_head + 2 * n_head_kv) * head_dim;
        OpRecord rec;
        rec.op = "rope";
        rec.params = {{"M", tokens}, {"head_dim", head_dim}, {"rotary_dim", rotary},
                      {"n_head", n_head}, {"n_head_kv", n_head_kv},
                      {"theta", std::string("10000")}, {"scale", std::string("1")}};
        rec.operands = {{"qkv", "bf16", {tokens, row}, 0, tokens, false},
                        {"positions", "i32", {tokens}, 0, 1, false},
                        {"out", "bf16", {tokens, row}, 0, tokens, true}};
        AdValue pos = value("i32", {tokens});
        for (int64_t i = 0; i < tokens; ++i) pos.value[(size_t)i] = (double)i;
        std::vector<AdValue> reads = {filled("bf16", {tokens, row}, &seed), pos};
        check_family("rope 5 tokens", rec, reads, seed);
    }
    /* kv_store: distinct slots split over tokens; a record that repeats a slot keeps the order
     * that decides which token's row survives. */
    for (const bool distinct : {true, false}) {
        uint64_t seed = 0x3c6ef372fe94f82bull;
        const int64_t tokens = 6, head_dim = 4, block_size = 4, n_blocks = 2;
        OpRecord rec;
        rec.op = "kv_store";
        rec.params = {{"n_head_kv", 1}, {"head_dim", head_dim}, {"block_size", block_size}};
        rec.operands = {
            {"k", "bf16", {tokens, 1, head_dim}, 0, tokens, false},
            {"v", "bf16", {tokens, 1, head_dim}, 0, tokens, false},
            {"slot", "i32", {tokens}, 0, 1, false},
            {"pool", "f32", {n_blocks, 1, block_size, 2 * head_dim}, 0, n_blocks, true},
        };
        AdValue slot = value("i32", {tokens});
        for (int64_t i = 0; i < tokens; ++i)
            slot.value[(size_t)i] = (double)(distinct ? i : i / 2);
        std::vector<AdValue> reads = {filled("bf16", {tokens, 1, head_dim}, &seed),
                                      filled("bf16", {tokens, 1, head_dim}, &seed), slot};
        check_family(distinct ? "kv_store distinct slots" : "kv_store repeated slots", rec, reads,
                     seed);
    }
}

/* ------------------------------------------------------------------ gdn_chunk_scan */

struct ScanShape {
    const char* name;
    int64_t k, v, hk, hv, chunk;
    std::vector<int64_t> lens;
    int64_t slots = 0;              // 0: one state slot per sequence
    std::vector<int64_t> sidx;      // empty: no `state_idx` operand
};

OpRecord scan_rec(const ScanShape& s) {
    const int64_t t = [&] {
        int64_t n = 0;
        for (int64_t x : s.lens) n += x;
        return n;
    }();
    const int64_t seq = (int64_t)s.lens.size();
    const int64_t slots = s.slots > 0 ? s.slots : seq;
    OpRecord rec;
    rec.op = "gdn_chunk_scan";
    rec.params = {{"M", t},
                  {"head_k", s.k},
                  {"head_v", s.v},
                  {"chunk", s.chunk},
                  {"scale", std::string("0.5")}};
    rec.operands = {
        {"q", "bf16", {t, s.hk, s.k}, 0, t, false},
        {"k", "bf16", {t, s.hk, s.k}, 0, t, false},
        {"v", "bf16", {t, s.hv, s.v}, 0, t, false},
        {"A", "bf16", {t, s.hv, s.chunk}, 0, t, false},
        {"g", "f32", {t, s.hv}, 0, t, false},
        {"beta", "f32", {t, s.hv}, 0, t, false},
        {"h0", "f32", {slots, s.hv, s.v, s.k}, 0, slots, false},
        {"cu", "i32", {seq + 1}, 0, 1, false},
        {"o", "bf16", {t, s.hv, s.v}, 0, t, true},
        {"ht", "f32", {slots, s.hv, s.v, s.k}, 0, slots, true},
    };
    if (!s.sidx.empty())
        rec.operands.push_back({"state_idx", "i32", {seq}, 0, 1, false});
    return rec;
}

std::vector<AdValue> scan_reads(const OpRecord& rec, const ScanShape& s, uint64_t* seed) {
    std::vector<AdValue> reads;
    for (const Operand& o : rec.operands) {
        if (o.written) continue;
        AdValue v = filled(o.dtype.c_str(), o.shape, seed);
        if (o.buffer == "h0") {
            v.value.assign(v.value.size(), 0.0);  // a prefill starts from the zero state
        } else if (o.buffer == "cu") {
            v.value[0] = 0.0;
            for (size_t n = 0; n < s.lens.size(); ++n)
                v.value[n + 1] = v.value[n] + (double)s.lens[n];
        } else if (o.buffer == "state_idx") {
            for (size_t n = 0; n < s.sidx.size(); ++n) v.value[n] = (double)s.sidx[n];
        } else if (o.buffer == "g") {
            /* A within-chunk cumsum of negative increments, so no exponent grows: the twin's own
             * arithmetic is all the test compares, but an overflow would make the comparison
             * meaningless. */
            std::vector<double> run((size_t)s.hv, 0.0);
            int64_t at = 0;
            for (size_t n = 0; n < s.lens.size(); ++n)
                for (int64_t i = 0; i < s.lens[n]; ++i, ++at) {
                    if (i % s.chunk == 0) run.assign((size_t)s.hv, 0.0);
                    for (int64_t h = 0; h < s.hv; ++h) {
                        run[(size_t)h] -= 0.05;
                        v.value[(size_t)(at * s.hv + h)] = run[(size_t)h];
                    }
                }
        } else if (o.buffer == "beta") {
            for (double& x : v.value) x = 0.25 + 0.25 * (x + 1.0) * 0.5;
        }
        reads.push_back(std::move(v));
    }
    return reads;
}

void gdn_cases() {
    /* A ragged batch (three sequences, two key heads, one value head each) and the tiny fixture's
     * four value heads over two key heads, where a key head's group of value heads must stay one
     * unit's serial work. The state-pool case names its slots; the aliased one points both
     * sequences at slot 0, where the twin must fall back to the record's order. */
    const ScanShape shapes[] = {
        {"ragged batch", 8, 8, 2, 2, 5, {2, 7, 13}, 0, {}},
        {"two key heads, four value heads", 16, 16, 2, 4, 4, {3}, 0, {}},
        {"state pool", 8, 8, 1, 2, 4, {5, 4}, 3, {2, 0}},
        {"aliased state slot", 8, 8, 1, 2, 4, {3, 4}, 1, {0, 0}},
    };
    for (const ScanShape& s : shapes) {
        uint64_t seed = 0xa54ff53a5f1d36f1ull;
        const OpRecord rec = scan_rec(s);
        std::vector<AdValue> reads = scan_reads(rec, s, &seed);
        check_family(std::string("gdn_chunk_scan ") + s.name, rec, reads, seed);
    }
}

}  // namespace

R97_TEST(parallel_rows_elementwise) { elementwise_cases(); }
R97_TEST(parallel_rows_quant) { quant_cases(); }
R97_TEST(parallel_rows_gated) { gated_cases(); }
R97_TEST(parallel_rows_attention) { attention_cases(); }
R97_TEST(parallel_rows_qk) { qk_cases(); }
R97_TEST(parallel_rows_gdn_scan) { gdn_cases(); }
