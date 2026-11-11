// src/adjoint/host/qk.h -- the qk family's f64 twins and their adjoints.
//
// The host issues the gated full-attention block as rmsnorm on queries and keys, then `rope`,
// then `kv_store`, and `sigmoid` followed by `mul` for the gate. This header is the f64 version
// of the ops that belong to the qk family: rope (partial NeoX rotary, Qwen3.5's mode), the copy
// into the paged KV pool, and the two elementwise gate ops. Every function works on plain f64
// arrays with the same row-major layouts as radiance's host kernels, so a finite difference can
// compare against the analytic adjoint without a tape.
#pragma once

#include <cstdint>
#include <vector>

namespace r97::adjoint::qk {

// ------------------------------------------------------------------ rope

// A rope operand is `tokens` rows of (n_head + 2*n_head_kv) * head_dim. The q heads come first,
// then the k heads, then the v heads; only q and k are rotated. Qwen3.5 uses partial rotary:
// `rotary_dim` lanes of each head turn and `head_dim - rotary_dim` pass through. The pairing is
// NeoX (lane i with lane i + rotary_dim/2) and the frequency ladder divides by `rotary_dim`, not
// `head_dim`, which is radiance's default "rotary" ladder.
struct RopeShape {
    int64_t tokens = 0;
    int64_t n_head = 0;      // query heads
    int64_t n_head_kv = 0;   // key heads; the same number of value heads follows them
    int64_t head_dim = 0;
    int64_t rotary_dim = 0;  // even and <= head_dim; 0 passes every head through
};

// The constant cos/sin plane rope reads: `rows` rows of `rotary_dim` values, cosines in the
// first half and sines in the second. It is position-indexed, like radiance's [max_ctx, rot]
// rope_table, and is never differentiated: rope's adjoint treats it as a constant.
struct RopeTable {
    int64_t rows = 0;
    int64_t rotary_dim = 0;
    std::vector<double> cos_sin;  // rows * rotary_dim, row-major

    const double* at(int64_t position) const {
        return cos_sin.data() + position * rotary_dim;
    }
};

RopeTable rope_table_f64(int64_t rows, int64_t rotary_dim, double theta, double scale = 1.0);

void rope_forward_f64(const double* qkv, const int32_t* positions, const RopeShape& shape,
                      const RopeTable& table, double* out);
void rope_adjoint_f64(const double* d_out, const int32_t* positions, const RopeShape& shape,
                      const RopeTable& table, double* d_qkv);

// ------------------------------------------------------------------ kv_store

// k and v are [tokens, kv_heads, head_dim]. The pool is [n_blocks, kv_heads, block_size,
// 2*head_dim], k in the first head_dim lanes of a row and v in the second. slot[t] names the
// row token t writes; a negative slot is a padded row and writes nothing.
struct KvStoreShape {
    int64_t tokens = 0;
    int64_t kv_heads = 0;
    int64_t head_dim = 0;
    int64_t block_size = 0;
    int64_t n_blocks = 0;
};

// The pool cotangent is addressed by (sequence, position): block_table[sequence][position /
// block_size] is the physical block and the position's offset inside that block is position %
// block_size. kv_store's slot mapping is that same address for this step's tokens; no cotangent
// goes to a slot the step does not write. `block_table_stride` is in int32 elements per sequence.
// A negative block (an empty sequence) returns -1.
int64_t kv_slot(const int32_t* block_table, int64_t block_table_stride, int64_t block_size,
                int64_t sequence, int64_t position);

void kv_store_forward_f64(const double* k, const double* v, const int32_t* slot,
                          const KvStoreShape& shape, double* pool);
void kv_store_adjoint_f64(const double* d_pool, const int32_t* slot, const KvStoreShape& shape,
                          double* d_k, double* d_v);

// ------------------------------------------------------------------ the gate

void sigmoid_forward_f64(const double* x, int64_t n, double* out);
void sigmoid_adjoint_f64(const double* out, const double* d_out, int64_t n, double* d_x);

// y = a * b. When `b_broadcast` is set, b holds `n` values and every one of the `rows` rows of a
// multiplies by the same vector, which is radiance's one broadcast rule for `mul`; otherwise b
// holds rows * n values. The gate multiplies the same-shaped sigmoid output, so it does not
// broadcast.
void mul_forward_f64(const double* a, const double* b, int64_t rows, int64_t n,
                     bool b_broadcast, double* out);
void mul_adjoint_f64(const double* d_out, const double* a, const double* b, int64_t rows,
                     int64_t n, bool b_broadcast, double* d_a, double* d_b);

}  // namespace r97::adjoint::qk
