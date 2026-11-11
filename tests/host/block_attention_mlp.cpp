// tests/host/block_attention_mlp.cpp -- one attention block and one MLP block, composed.
//
// The per-op tests check a family's twin against its adjoint and against radiance's host kernel
// one record at a time. This file checks the four families together: it runs the op sequence the
// fixture's qwen35 block issues, on the host, with the registered twins forward and the
// registered adjoints backward, and asks three questions of the composition.
//
//   1. The composed adjoint is the derivative of the composed forward: central differences of the
//      block along a random direction of its input residual, against the adjoint's directional
//      derivative, to 1e-6 relative. One sequence and a batch of three unequal lengths.
//   2. The twins' block output is what radiance's host kernels compute for the same block, with
//      every op's boundaries rounded to bf16 as the engine's are. That is the check that the
//      record the fixture issues -- which operands are in place, where a view starts, which
//      parameter carries which count -- is a block the host can also run.
//   3. Against the PyTorch reference (tests/fixtures/oracle, cases attention_sublayer and
//      mlp_sublayer), the twin block reproduces the reference's output and input cotangent from
//      its input and output cotangent, in f64, to 1e-8 relative. The reference's batch is padded
//      to its longest row; this file runs the packed valid rows, which is the same block, because
//      no valid query row reads a padded one. The padded rows do read the valid ones, though -- the
//      reference leaves a fully masked row attending to every key the padding leaves visible -- so
//      the reference's input cotangent for a padded sequence carries a term the block has not:
//      docs/ORACLE.md has the cause and the tolerance that follows from it, and kPaddedRowTolerance
//      is that tolerance.
//
// The pool cotangent is where the qk family and the attention family have to agree. kv_store
// writes k and v at the slot kv_slot() names for (sequence, position); attn_paged's adjoint
// scatters dk and dv back to the slot it read, and this file's kv_store adjoint hands them on.
// A batch of 17, 11 and 6 tokens crosses a 16-token block boundary in the first sequence, so a
// wrong slot lands on another sequence's tokens and the finite differences notice.
//
// rope and kv_store have f64 twins and adjoints in the qk family but no registered twin or adjoint
// row: src/adjoint/host/qk.cpp registers neither op. The walk calls their f64 functions directly,
// the same ones a registered row would call, and the kernel chain runs their records. Everything
// else goes through twin_find()/adjoint_find(). That gap is the qk family's to close; docs/BLOCKS.md
// records it, and this file adds and patches no adjoint.

#include "../../src/adjoint/host/qk.h"

#include "../support/adjoint_check.h"
#include "../support/block_check.h"
#include "../support/fixture.h"
#include "../support/oracle.h"
#include "../support/test.h"

#include "tiny_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using r97::AdValue;
using r97::OpParam;
using r97::OpRecord;
using r97::Operand;
using r97::adjoint::qk::KvStoreShape;
using r97::adjoint::qk::RopeShape;
using r97::adjoint::qk::RopeTable;
namespace bc = r97::block_check;

// The fixture's geometry (tests/support/fixture.h: qwen35_spec).
constexpr int64_t kHidden = 128;
constexpr int64_t kHeads = 4;
constexpr int64_t kKvHeads = 2;
constexpr int64_t kHeadDim = 32;
constexpr int64_t kFfn = 256;
constexpr int64_t kRotary = 16;   // 128 // 2: partial_rotary_factor 0.5
constexpr int64_t kBlockSize = 16;  // rad_arch.h's RAD_KV_BLOCK
constexpr int64_t kQDim = kHeads * kHeadDim;
constexpr int64_t kKvDim = kKvHeads * kHeadDim;
constexpr int64_t kQGateDim = 2 * kQDim;
constexpr double kRmsEps = 1e-6;
constexpr double kRopeTheta = 10000.0;
constexpr int64_t kAttentionLayer = 3;  // (l + 1) % interval == 0: the one full-attention layer
constexpr int64_t kMlpLayer = 0;        // the reference's mlp_sublayer case is layer 0's MLP


// The block cases' inputs are the reference's own: the fixture is the checkpoint the reference was
// written from, so its weights are the ones both sides see.
constexpr double kFdTolerance = 1e-6;
constexpr double kOracleTolerance = 1e-8;
constexpr double kBf16Tolerance = 1e-2;  // measured worst 4.6e-3 on this fixture: docs/BLOCKS.md
// The reference's padded query rows are fully masked, and its input row there is zero, and
// transformers' attention leaves a fully masked row attending to every key the padding leaves
// visible: such a row averages the sequence's valid values with weight 1/length. Its random output
// cotangent therefore adds a term to the valid rows' value gradient that a block masked the way
// radiance masks it has not. Measured 0.165 relative on this fixture's third sequence, 5.9e-2 on
// its second, and 3e-16 on its first, which fills the batch and has no padded row: docs/ORACLE.md
// has the cause and this bound.
constexpr double kPaddedRowTolerance = 0.2;

constexpr const char* kDtype = "bf16";

OpParam num(const char* name, int64_t v) { return {name, v}; }
// A real parameter reaches a record as its decimal spelling: the record's value is a number or a
// text, and the families read the text with atof. 17 significant digits round-trip exactly, so
// 1/sqrt(head_dim) is the scale the oracle used, not its first six digits.
OpParam real(const char* name, double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return {name, std::string(buf)};
}
OpParam text(const char* name, const char* v) { return {name, std::string(v)}; }

// ------------------------------------------------------------------ the fixture's weights

double bf16_at(const uint8_t* p) {
    const uint16_t raw = (uint16_t)(p[0] | (p[1] << 8));
    const uint32_t bits = (uint32_t)raw << 16;
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return (double)f;
}

std::vector<double> values_of(const tiny::Tensor& t) {
    std::vector<double> v;
    if (t.dtype == "BF16") {
        v.resize(t.bytes.size() / 2);
        for (size_t i = 0; i < v.size(); ++i) v[i] = bf16_at(t.bytes.data() + 2 * i);
    } else if (t.dtype == "F32") {
        v.resize(t.bytes.size() / 4);
        for (size_t i = 0; i < v.size(); ++i) std::memcpy(&v[i], t.bytes.data() + 4 * i, 4);
    } else {
        throw std::runtime_error("block: weight \"" + t.name + "\" has dtype " + t.dtype);
    }
    return v;
}

struct Layer {
    std::vector<double> input_norm, q_proj, q_norm, k_proj, k_norm, v_proj, o_proj;
    std::vector<double> post_norm, gate_up, down_proj;
};

struct Blocks {
    Layer attention;
    Layer mlp;
};

// A directory of this file's own, removed when the run ends. The checkpoint is written here
// rather than read from tests/support/fixture.h's cached one, so this file's bytes are its own;
// they are the fixture's own write (r97::write_checkpoint), so the reference and this file still
// see one checkpoint.
struct CheckpointDir {
    char path[64];

    CheckpointDir() {
        std::strcpy(path, "/tmp/r97_blocks_XXXXXX");
        if (::mkdtemp(path) == nullptr)
            throw std::runtime_error("block: cannot make a temporary directory under /tmp");
        r97::write_checkpoint(path);
    }
    ~CheckpointDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
    CheckpointDir(const CheckpointDir&)            = delete;
    CheckpointDir& operator=(const CheckpointDir&) = delete;
};

const std::map<std::string, std::vector<double>>& checkpoint_weights() {
    static const CheckpointDir dir;
    static const std::map<std::string, std::vector<double>> w = [] {
        const std::string path = std::string(dir.path) + "/model.safetensors";
        std::map<std::string, std::vector<double>> m;
        for (const tiny::Tensor& t : tiny::read_safetensors(path)) m[t.name] = values_of(t);
        return m;
    }();
    return w;
}

std::vector<double> weight(const std::string& name) {
    const auto& w = checkpoint_weights();
    const auto it = w.find(name);
    if (it == w.end()) throw std::runtime_error("block: the checkpoint has no \"" + name + "\"");
    return it->second;
}

// The attention weights of layer `l`, whose self_attn tensors only a full-attention layer has.
Layer attention_layer(int64_t l) {
    const std::string p = "model.language_model.layers." + std::to_string(l) + ".";
    Layer L;
    L.input_norm = weight(p + "input_layernorm.weight");
    L.q_proj = weight(p + "self_attn.q_proj.weight");
    L.q_norm = weight(p + "self_attn.q_norm.weight");
    L.k_proj = weight(p + "self_attn.k_proj.weight");
    L.k_norm = weight(p + "self_attn.k_norm.weight");
    L.v_proj = weight(p + "self_attn.v_proj.weight");
    L.o_proj = weight(p + "self_attn.o_proj.weight");
    return L;
}

// The MLP weights of layer `l`: a post-attention norm, the fused gate-and-up projection and the
// down projection. The checkpoint carries gate and up separately; the converter concatenates their
// rows, gate first, into the one [2 * ffn, hidden] parameter the block reads (radiance's
// rad_block_mlp.h map_concat), and this file builds the same weight the same way, so the block's
// one gemm_nt sees what the fixture's does. The reference's mlp_sublayer case is layer 0's, per the
// manifest's cases.blocks; the block reads nothing of the mixer that comes before it, so the layer
// is only which weights it uses.
Layer mlp_layer(int64_t l) {
    const std::string p = "model.language_model.layers." + std::to_string(l) + ".";
    Layer L;
    L.post_norm = weight(p + "post_attention_layernorm.weight");
    const auto gate = weight(p + "mlp.gate_proj.weight");
    const auto up = weight(p + "mlp.up_proj.weight");
    if (gate.size() != (size_t)(kFfn * kHidden) || up.size() != gate.size())
        throw std::runtime_error("block: layer " + std::to_string(l) +
                                 "'s gate and up projections are not both [ffn, hidden]");
    L.gate_up = gate;
    L.gate_up.insert(L.gate_up.end(), up.begin(), up.end());
    L.down_proj = weight(p + "mlp.down_proj.weight");
    return L;
}

const Blocks& blocks() {
    static const Blocks b = [] {
        Blocks x;
        x.attention = attention_layer(kAttentionLayer);
        x.mlp = mlp_layer(kMlpLayer);
        return x;
    }();
    return b;
}

// ------------------------------------------------------------------ a batch's addressing

struct Batch {
    std::vector<int64_t> lens;
    int64_t block_size = kBlockSize;

    int64_t n_seq() const { return (int64_t)lens.size(); }
    int64_t total() const {
        int64_t n = 0;
        for (int64_t l : lens) n += l;
        return n;
    }
    int64_t max_len() const { return *std::max_element(lens.begin(), lens.end()); }
    int64_t blocks_of(int64_t s) const {
        return (lens[(size_t)s] + block_size - 1) / block_size;
    }
    int64_t max_blocks() const {
        int64_t m = 0;
        for (int64_t s = 0; s < n_seq(); ++s) m = std::max(m, blocks_of(s));
        return m;
    }
    int64_t first_block(int64_t s) const {
        int64_t b = 0;
        for (int64_t i = 0; i < s; ++i) b += blocks_of(i);
        return b;
    }
    int64_t n_blocks() const {
        int64_t n = 0;
        for (int64_t s = 0; s < n_seq(); ++s) n += blocks_of(s);
        return n;
    }
    int64_t pool_values() const { return n_blocks() * kKvHeads * block_size * 2 * kHeadDim; }

    std::vector<int32_t> block_table() const {
        const int64_t stride = max_blocks();
        std::vector<int32_t> t((size_t)(n_seq() * stride), -1);
        for (int64_t s = 0; s < n_seq(); ++s)
            for (int64_t b = 0; b < blocks_of(s); ++b)
                t[(size_t)(s * stride + b)] = (int32_t)(first_block(s) + b);
        return t;
    }
    std::vector<int32_t> seqused() const {
        std::vector<int32_t> t;
        for (int64_t l : lens) t.push_back((int32_t)l);
        return t;
    }
    std::vector<int32_t> cu_seqlens() const {
        std::vector<int32_t> t{0};
        for (int64_t l : lens) t.push_back((int32_t)(t.back() + l));
        return t;
    }
    // The pool slot of every packed row, addressed as the qk family names it.
    std::vector<int32_t> slot_mapping() const {
        const std::vector<int32_t> table = block_table();
        const int64_t stride = max_blocks();
        std::vector<int32_t> slots;
        slots.reserve((size_t)total());
        for (int64_t s = 0; s < n_seq(); ++s)
            for (int64_t p = 0; p < lens[(size_t)s]; ++p)
                slots.push_back((int32_t)r97::adjoint::qk::kv_slot(
                    table.data(), stride, block_size, s, p));
        return slots;
    }
    std::vector<std::pair<int64_t, int64_t>> rows() const {
        std::vector<std::pair<int64_t, int64_t>> r;
        r.reserve((size_t)total());
        for (int64_t s = 0; s < n_seq(); ++s)
            for (int64_t p = 0; p < lens[(size_t)s]; ++p) r.emplace_back(s, p);
        return r;
    }
};

// ------------------------------------------------------------------ the records the fixture issues

OpRecord rmsnorm_rec(int64_t rows, int64_t n) {
    OpRecord r;
    r.op = "rmsnorm";
    r.params = {num("M", rows), num("n", n), real("eps", kRmsEps), real("wadd", 1.0),
                real("mult", 1.0), text("dtype", kDtype)};
    r.n = rows;
    r.operands = {{"x", kDtype, {rows, n}, 0, rows, false},
                  {"w", kDtype, {n}, 0, 0, false},
                  {"y", kDtype, {rows, n}, 0, rows, true}};
    return r;
}

OpRecord gemm_rec(int64_t m, int64_t n, int64_t k) {
    OpRecord r;
    r.op = "gemm_nt";
    r.params = {num("M", m), num("N", n), num("K", k), text("dtype", kDtype)};
    r.n = m;
    r.operands = {{"a", kDtype, {m, k}, 0, m, false},
                  {"b", kDtype, {n, k}, 0, n, false},
                  {"y", kDtype, {m, n}, 0, m, true}};
    return r;
}

OpRecord sigmoid_rec(int64_t rows, int64_t n) {
    OpRecord r;
    r.op = "sigmoid";
    r.params = {num("M", rows), num("n", n), text("dtype", kDtype)};
    r.n = rows;
    r.operands = {{"x", kDtype, {rows, n}, 0, rows, false},
                  {"y", kDtype, {rows, n}, 0, rows, true}};
    return r;
}

OpRecord mul_rec(int64_t rows, int64_t n) {
    OpRecord r;
    r.op = "mul";
    r.params = {num("M", rows), num("n", n), text("dtype", kDtype)};
    r.n = rows;
    r.operands = {{"a", kDtype, {rows, n}, 0, rows, false},
                  {"b", kDtype, {rows, n}, 0, rows, false},
                  {"y", kDtype, {rows, n}, 0, rows, true}};
    return r;
}

OpRecord add_rec(int64_t rows, int64_t n) {
    OpRecord r;
    r.op = "add";
    r.params = {num("M", rows), num("n", n), num("div", 1), text("dtype", kDtype)};
    r.n = rows;
    r.operands = {{"a", kDtype, {rows, n}, 0, rows, false},
                  {"b", kDtype, {rows, n}, 0, rows, false},
                  {"y", kDtype, {rows, n}, 0, rows, true}};
    return r;
}

OpRecord silu_mul_rec(int64_t rows, int64_t n) {
    OpRecord r;
    r.op = "silu_mul";
    r.params = {num("M", rows), num("n", n), text("dtype", kDtype)};
    r.n = rows;
    r.operands = {{"gate_up", kDtype, {rows, 2 * n}, 0, rows, false},
                  {"y", kDtype, {rows, n}, 0, rows, true}};
    return r;
}

OpRecord attn_rec(const Batch& b) {
    const int64_t rows = b.total();
    const int64_t stride = b.max_blocks();
    OpRecord r;
    r.op = "attn_paged";
    r.params = {num("q_len", rows), num("head_dim", kHeadDim), num("gqa", kHeads / kKvHeads),
                num("block_size", b.block_size), num("causal", 1), num("window", 0),
                text("q_dtype", kDtype), text("kv_dtype", kDtype),
                real("scale", 1.0 / std::sqrt((double)kHeadDim)), num("max_ctx", b.max_len()),
                num("n_head", kHeads), num("max_seqs", b.n_seq()), text("dtype", kDtype)};
    r.n = rows;
    // The descales are here, at one, because an operand past them keeps its schema position: a
    // record that leaves them out and still writes cu_seqlens has the twins read the cu_seqlens as
    // the k descale (attention.cpp's `listed`) and the scores go flat.
    r.operands = {
        {"q", kDtype, {rows, kHeads, kHeadDim}, 0, rows, false},
        {"kv_cache", kDtype, {b.n_blocks(), kKvHeads, b.block_size, 2 * kHeadDim}, 0, 0, false},
        {"block_table", "i32", {b.n_seq(), stride}, 0, b.n_seq(), false},
        {"seqused", "i32", {b.n_seq()}, 0, b.n_seq(), false},
        {"k_descale", "f32", {b.n_seq() * kKvHeads}, 0, b.n_seq() * kKvHeads, false},
        {"v_descale", "f32", {b.n_seq() * kKvHeads}, 0, b.n_seq() * kKvHeads, false},
        {"cu_seqlens", "i32", {b.n_seq() + 1}, 0, b.n_seq() + 1, false},
        {"out", kDtype, {rows, kHeads, kHeadDim}, 0, rows, true},
    };
    return r;
}

// rope's and kv_store's records: the kernel chain runs them as the fixture issues them, the twin
// chain calls the qk family's f64 functions. rope's schema names its angle base "theta"; a record
// that spells it "rope" reaches the kernel with that parameter missing and the kernel falls back
// to its own 10000, which this fixture's theta happens to equal, so the chain would agree by
// accident and pin no parameter. The twin and the kernel are compared on the record below, and the
// theta the table is built from is the same constant.
OpRecord rope_rec(int64_t T, int64_t heads, int64_t n_head_kv) {
    OpRecord r;
    r.op = "rope";
    r.params = {num("M", T), num("head_dim", kHeadDim), num("n_head", heads),
                num("n_head_kv", n_head_kv), num("rotary_dim", kRotary),
                real("theta", kRopeTheta), real("scale", 1.0), text("mode", "neox"),
                text("dtype", kDtype)};
    r.n = T;
    r.operands = {{"qkv", kDtype, {T, heads, kHeadDim}, 0, T, true},
                  {"positions", "i32", {T}, 0, T, false}};
    return r;
}

OpRecord kv_store_rec(int64_t T, const Batch& b) {
    OpRecord r;
    r.op = "kv_store";
    r.params = {num("M", T), num("head_dim", kHeadDim), num("n_head_kv", kKvHeads),
                num("block_size", b.block_size), text("kv_dtype", kDtype), text("dtype", kDtype)};
    r.n = T;
    r.operands = {
        {"k", kDtype, {T, kKvHeads, kHeadDim}, 0, T, false},
        {"v", kDtype, {T, kKvHeads, kHeadDim}, 0, T, false},
        {"slot_mapping", "i32", {T}, 0, T, false},
        {"kv_cache", kDtype, {b.n_blocks(), kKvHeads, b.block_size, 2 * kHeadDim}, 0, 0, true},
    };
    return r;
}

// ------------------------------------------------------------------ the views the fixture takes

// q_proj's [T, 2 * heads * head_dim] row interleaves each head's query and gate; the fixture
// reads the query at the head's first head_dim values and the gate at the next.
std::vector<double> q_view(const std::vector<double>& qg, int64_t T) {
    std::vector<double> q((size_t)(T * kQDim));
    for (int64_t t = 0; t < T; ++t)
        for (int64_t h = 0; h < kHeads; ++h)
            for (int64_t d = 0; d < kHeadDim; ++d)
                q[(size_t)((t * kHeads + h) * kHeadDim + d)] =
                    qg[(size_t)(t * kQGateDim + h * 2 * kHeadDim + d)];
    return q;
}

std::vector<double> gate_view(const std::vector<double>& qg, int64_t T) {
    std::vector<double> g((size_t)(T * kQDim));
    for (int64_t t = 0; t < T; ++t)
        for (int64_t h = 0; h < kHeads; ++h)
            for (int64_t d = 0; d < kHeadDim; ++d)
                g[(size_t)((t * kHeads + h) * kHeadDim + d)] =
                    qg[(size_t)(t * kQGateDim + h * 2 * kHeadDim + kHeadDim + d)];
    return g;
}

std::vector<double> qg_from_views(const std::vector<double>& q, const std::vector<double>& gate,
                                  int64_t T) {
    std::vector<double> qg((size_t)(T * kQGateDim), 0.0);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t h = 0; h < kHeads; ++h)
            for (int64_t d = 0; d < kHeadDim; ++d) {
                qg[(size_t)(t * kQGateDim + h * 2 * kHeadDim + d)] =
                    q[(size_t)((t * kHeads + h) * kHeadDim + d)];
                qg[(size_t)(t * kQGateDim + h * 2 * kHeadDim + kHeadDim + d)] =
                    gate[(size_t)((t * kHeads + h) * kHeadDim + d)];
            }
    return qg;
}

// ------------------------------------------------------------------ the blocks, forward and back

struct BlockResult {
    std::vector<double> y;
    std::vector<double> dx;
};

BlockResult attention_block(const Blocks& B, const Batch& b, const std::vector<double>& x,
                            const std::vector<double>* dy) {
    const Layer& L = B.attention;
    const int64_t T = b.total();

    // forward: the fixture's order
    const auto h = bc::twin_run(rmsnorm_rec(T, kHidden),
                                {bc::value({T, kHidden}, x),
                                 bc::value({kHidden}, L.input_norm)})[0].value;
    const auto qg = bc::twin_run(gemm_rec(T, kQGateDim, kHidden),
                                 {bc::value({T, kHidden}, h),
                                  bc::value({kQGateDim, kHidden}, L.q_proj)})[0].value;
    const auto q = q_view(qg, T);
    const auto gate = gate_view(qg, T);
    const auto qn = bc::twin_run(rmsnorm_rec(T * kHeads, kHeadDim),
                                 {bc::value({T * kHeads, kHeadDim}, q),
                                  bc::value({kHeadDim}, L.q_norm)})[0].value;
    const auto k = bc::twin_run(gemm_rec(T, kKvDim, kHidden),
                                {bc::value({T, kHidden}, h),
                                 bc::value({kKvDim, kHidden}, L.k_proj)})[0].value;
    const auto v = bc::twin_run(gemm_rec(T, kKvDim, kHidden),
                                {bc::value({T, kHidden}, h),
                                 bc::value({kKvDim, kHidden}, L.v_proj)})[0].value;
    const auto kn = bc::twin_run(rmsnorm_rec(T * kKvHeads, kHeadDim),
                                 {bc::value({T * kKvHeads, kHeadDim}, k),
                                  bc::value({kHeadDim}, L.k_norm)})[0].value;

    // rope: the fixture's positions are the packed row's index (make_reference.py's block case
    // hands the block arange over the packed rows), and the table is gptj halves, which is what
    // rope_table_f64 builds and what rope's record says with mode = "neox"
    const RopeTable table = r97::adjoint::qk::rope_table_f64(T, kRotary, kRopeTheta, 1.0);
    std::vector<int32_t> positions((size_t)T);
    for (int64_t t = 0; t < T; ++t) positions[(size_t)t] = (int32_t)t;
    const RopeShape q_rope{T, kHeads, 0, kHeadDim, kRotary};
    const RopeShape k_rope{T, kKvHeads, 0, kHeadDim, kRotary};
    std::vector<double> qr((size_t)(T * kQDim));
    r97::adjoint::qk::rope_forward_f64(qn.data(), positions.data(), q_rope, table, qr.data());
    std::vector<double> kr((size_t)(T * kKvDim));
    r97::adjoint::qk::rope_forward_f64(kn.data(), positions.data(), k_rope, table, kr.data());

    // kv_store into the paged pool at this batch's slots
    const std::vector<int32_t> slots = b.slot_mapping();
    const KvStoreShape pool_shape{T, kKvHeads, kHeadDim, b.block_size, b.n_blocks()};
    std::vector<double> pool((size_t)b.pool_values(), 0.0);
    r97::adjoint::qk::kv_store_forward_f64(kr.data(), v.data(), slots.data(), pool_shape,
                                           pool.data());

    const OpRecord attn = attn_rec(b);
    const auto bt = b.block_table();
    const auto seqused = b.seqused();
    const auto cu = b.cu_seqlens();
    const AdValue attn_q = bc::value({T, kHeads, kHeadDim}, qr);
    const AdValue attn_kv =
        bc::value({b.n_blocks(), kKvHeads, b.block_size, 2 * kHeadDim}, pool);
    const AdValue attn_table = bc::value({b.n_seq(), b.max_blocks()},
                                         std::vector<double>(bt.begin(), bt.end()), "i32");
    const AdValue attn_seqused =
        bc::value({b.n_seq()}, std::vector<double>(seqused.begin(), seqused.end()), "i32");
    const AdValue attn_cu = bc::value({b.n_seq() + 1},
                                      std::vector<double>(cu.begin(), cu.end()), "i32");
    const AdValue attn_kdesc = bc::value({b.n_seq() * kKvHeads},
                                         std::vector<double>((size_t)(b.n_seq() * kKvHeads), 1.0),
                                         "f32");
    const AdValue attn_vdesc = attn_kdesc;
    const auto attn_out = bc::twin_run(attn, {attn_q, attn_kv, attn_table, attn_seqused,
                                              attn_kdesc, attn_vdesc, attn_cu})[0].value;

    const auto g = bc::twin_run(sigmoid_rec(T * kHeads, kHeadDim),
                                {bc::value({T * kHeads, kHeadDim}, gate)})[0].value;
    const auto gated = bc::twin_run(mul_rec(T, kQDim),
                                    {bc::value({T, kQDim}, attn_out),
                                     bc::value({T, kQDim}, g)})[0].value;
    const auto o = bc::twin_run(gemm_rec(T, kHidden, kQDim),
                                {bc::value({T, kQDim}, gated),
                                 bc::value({kHidden, kQDim}, L.o_proj)})[0].value;
    const auto y = bc::twin_run(add_rec(T, kHidden),
                                {bc::value({T, kHidden}, x), bc::value({T, kHidden}, o)})[0].value;

    BlockResult r{y, {}};
    if (!dy) return r;

    // backward: the same records, the reverse order
    const auto& d_y = *dy;
    const auto d_add = bc::adjoint_run(add_rec(T, kHidden),
                                       {bc::value({T, kHidden}, x), bc::value({T, kHidden}, o),
                                        bc::value({T, kHidden}, y)},
                                       {bc::value({T, kHidden}, d_y)});
    std::vector<double> dx = d_add[0].value;
    const auto d_o = d_add[1].value;
    const auto d_gemm_o = bc::adjoint_run(gemm_rec(T, kHidden, kQDim),
                                          {bc::value({T, kQDim}, gated),
                                           bc::value({kHidden, kQDim}, L.o_proj),
                                           bc::value({T, kHidden}, o)},
                                          {bc::value({T, kHidden}, d_o)});
    const auto d_gated = d_gemm_o[0].value;
    const auto d_mul = bc::adjoint_run(mul_rec(T, kQDim),
                                       {bc::value({T, kQDim}, attn_out), bc::value({T, kQDim}, g),
                                        bc::value({T, kQDim}, gated)},
                                       {bc::value({T, kQDim}, d_gated)});
    const auto d_attn = d_mul[0].value;
    const auto d_g = d_mul[1].value;
    const auto d_sig = bc::adjoint_run(sigmoid_rec(T * kHeads, kHeadDim),
                                       {bc::value({T * kHeads, kHeadDim}, gate),
                                        bc::value({T * kHeads, kHeadDim}, g)},
                                       {bc::value({T * kHeads, kHeadDim}, d_g)});
    const auto d_gate = d_sig[0].value;
    const auto d_attn_out = bc::adjoint_run(attn, {attn_q, attn_kv, attn_table, attn_seqused,
                                                   attn_kdesc, attn_vdesc, attn_cu,
                                                   bc::value({T, kHeads, kHeadDim}, attn_out)},
                                            {bc::value({T, kHeads, kHeadDim}, d_attn)});
    const auto d_qr = d_attn_out[0].value;
    const auto d_pool = d_attn_out[1].value;

    // kv_store's adjoint gathers dk/dv from the pool slots the attention adjoint filled
    std::vector<double> dkr((size_t)(T * kKvDim), 0.0), dv((size_t)(T * kKvDim), 0.0);
    r97::adjoint::qk::kv_store_adjoint_f64(d_pool.data(), slots.data(), pool_shape, dkr.data(),
                                           dv.data());
    std::vector<double> dkn((size_t)(T * kKvDim), 0.0);
    r97::adjoint::qk::rope_adjoint_f64(dkr.data(), positions.data(), k_rope, table, dkn.data());
    std::vector<double> dqn((size_t)(T * kQDim), 0.0);
    r97::adjoint::qk::rope_adjoint_f64(d_qr.data(), positions.data(), q_rope, table, dqn.data());

    const auto d_qn = bc::adjoint_run(rmsnorm_rec(T * kHeads, kHeadDim),
                                      {bc::value({T * kHeads, kHeadDim}, q),
                                       bc::value({kHeadDim}, L.q_norm),
                                       bc::value({T * kHeads, kHeadDim}, qn)},
                                      {bc::value({T * kHeads, kHeadDim}, dqn)});
    const auto d_kn = bc::adjoint_run(rmsnorm_rec(T * kKvHeads, kHeadDim),
                                      {bc::value({T * kKvHeads, kHeadDim}, k),
                                       bc::value({kHeadDim}, L.k_norm),
                                       bc::value({T * kKvHeads, kHeadDim}, kn)},
                                      {bc::value({T * kKvHeads, kHeadDim}, dkn)});
    const std::vector<double> d_qg = qg_from_views(d_qn[0].value, d_gate, T);

    // the three projections all read the normed input: their input cotangents add up
    std::vector<double> d_h((size_t)(T * kHidden), 0.0);
    const auto gemm_input = [&](int64_t n, const std::vector<double>& w,
                                const std::vector<double>& out, const std::vector<double>& d_out) {
        const auto d = bc::adjoint_run(gemm_rec(T, n, kHidden),
                                       {bc::value({T, kHidden}, h),
                                        bc::value({n, kHidden}, w),
                                        bc::value({T, n}, out)},
                                       {bc::value({T, n}, d_out)});
        return d[0].value;
    };
    const auto d_qg_in = gemm_input(kQGateDim, L.q_proj, qg, d_qg);
    const auto d_k_in = gemm_input(kKvDim, L.k_proj, k, d_kn[0].value);
    const auto d_v_in = gemm_input(kKvDim, L.v_proj, v, dv);
    for (size_t i = 0; i < d_h.size(); ++i) d_h[i] = d_qg_in[i] + d_k_in[i] + d_v_in[i];

    const auto d_norm = bc::adjoint_run(rmsnorm_rec(T, kHidden),
                                        {bc::value({T, kHidden}, x),
                                         bc::value({kHidden}, L.input_norm),
                                         bc::value({T, kHidden}, h)},
                                        {bc::value({T, kHidden}, d_h)});
    for (size_t i = 0; i < dx.size(); ++i) dx[i] += d_norm[0].value[i];
    r.dx = std::move(dx);
    return r;
}

BlockResult mlp_block(const Blocks& B, const Batch& b, const std::vector<double>& x,
                      const std::vector<double>* dy) {
    const Layer& L = B.mlp;
    const int64_t T = b.total();

    const auto h = bc::twin_run(rmsnorm_rec(T, kHidden),
                                {bc::value({T, kHidden}, x),
                                 bc::value({kHidden}, L.post_norm)})[0].value;
    // one projection over the fused weight: [T, hidden] -> [T, 2 * ffn], gate in the low half of
    // each row, which is how silu_mul reads it
    const auto gu = bc::twin_run(gemm_rec(T, 2 * kFfn, kHidden),
                                 {bc::value({T, kHidden}, h),
                                  bc::value({2 * kFfn, kHidden}, L.gate_up)})[0].value;
    const auto ffn = bc::twin_run(silu_mul_rec(T, kFfn), {bc::value({T, 2 * kFfn}, gu)})[0].value;
    const auto o = bc::twin_run(gemm_rec(T, kHidden, kFfn),
                                {bc::value({T, kFfn}, ffn),
                                 bc::value({kHidden, kFfn}, L.down_proj)})[0].value;
    const auto y = bc::twin_run(add_rec(T, kHidden),
                                {bc::value({T, kHidden}, x), bc::value({T, kHidden}, o)})[0].value;

    BlockResult r{y, {}};
    if (!dy) return r;

    const auto& d_y = *dy;
    const auto d_add = bc::adjoint_run(add_rec(T, kHidden),
                                       {bc::value({T, kHidden}, x), bc::value({T, kHidden}, o),
                                        bc::value({T, kHidden}, y)},
                                       {bc::value({T, kHidden}, d_y)});
    std::vector<double> dx = d_add[0].value;
    const auto d_o = d_add[1].value;
    const auto d_down = bc::adjoint_run(gemm_rec(T, kHidden, kFfn),
                                        {bc::value({T, kFfn}, ffn),
                                         bc::value({kHidden, kFfn}, L.down_proj),
                                         bc::value({T, kHidden}, o)},
                                        {bc::value({T, kHidden}, d_o)});
    const auto d_gu = bc::adjoint_run(silu_mul_rec(T, kFfn),
                                      {bc::value({T, 2 * kFfn}, gu), bc::value({T, kFfn}, ffn)},
                                      {bc::value({T, kFfn}, d_down[0].value)})[0].value;
    const auto d_h = bc::adjoint_run(gemm_rec(T, 2 * kFfn, kHidden),
                                     {bc::value({T, kHidden}, h),
                                      bc::value({2 * kFfn, kHidden}, L.gate_up),
                                      bc::value({T, 2 * kFfn}, gu)},
                                     {bc::value({T, 2 * kFfn}, d_gu)})[0].value;
    const auto d_norm = bc::adjoint_run(rmsnorm_rec(T, kHidden),
                                        {bc::value({T, kHidden}, x),
                                         bc::value({kHidden}, L.post_norm),
                                         bc::value({T, kHidden}, h)},
                                        {bc::value({T, kHidden}, d_h)});
    for (size_t i = 0; i < dx.size(); ++i) dx[i] += d_norm[0].value[i];
    r.dx = std::move(dx);
    return r;
}

// ------------------------------------------------------------------ the same blocks on the kernels

std::vector<double> attention_kernel_chain(const Blocks& B, const Batch& b,
                                           const std::vector<double>& x) {
    const Layer& L = B.attention;
    const int64_t T = b.total();

    const auto h = bc::kernel_run(rmsnorm_rec(T, kHidden),
                                  {bc::value({T, kHidden}, x),
                                   bc::value({kHidden}, L.input_norm)})[0].value;
    const auto qg = bc::kernel_run(gemm_rec(T, kQGateDim, kHidden),
                                   {bc::value({T, kHidden}, h),
                                    bc::value({kQGateDim, kHidden}, L.q_proj)})[0].value;
    const auto q = q_view(qg, T);
    const auto gate = gate_view(qg, T);
    const auto qn = bc::kernel_run(rmsnorm_rec(T * kHeads, kHeadDim),
                                   {bc::value({T * kHeads, kHeadDim}, q),
                                    bc::value({kHeadDim}, L.q_norm)})[0].value;
    const auto k = bc::kernel_run(gemm_rec(T, kKvDim, kHidden),
                                  {bc::value({T, kHidden}, h),
                                   bc::value({kKvDim, kHidden}, L.k_proj)})[0].value;
    const auto v = bc::kernel_run(gemm_rec(T, kKvDim, kHidden),
                                  {bc::value({T, kHidden}, h),
                                   bc::value({kKvDim, kHidden}, L.v_proj)})[0].value;
    const auto kn = bc::kernel_run(rmsnorm_rec(T * kKvHeads, kHeadDim),
                                   {bc::value({T * kKvHeads, kHeadDim}, k),
                                    bc::value({kHeadDim}, L.k_norm)})[0].value;

    std::vector<int32_t> positions((size_t)T);
    for (int64_t t = 0; t < T; ++t) positions[(size_t)t] = (int32_t)t;
    const std::vector<double> pos_values(positions.begin(), positions.end());
    const auto qr = bc::kernel_run(rope_rec(T, kHeads, 0),
                                   {bc::value({T}, pos_values, "i32")},
                                   {bc::value({T, kHeads, kHeadDim}, qn)})[0].value;
    const auto kr = bc::kernel_run(rope_rec(T, kKvHeads, 0),
                                   {bc::value({T}, pos_values, "i32")},
                                   {bc::value({T, kKvHeads, kHeadDim}, kn)})[0].value;

    const std::vector<int32_t> slots = b.slot_mapping();
    const std::vector<double> slot_values(slots.begin(), slots.end());
    std::vector<double> pool((size_t)b.pool_values(), 0.0);
    const auto pooled = bc::kernel_run(
        kv_store_rec(T, b),
        {bc::value({T, kKvHeads, kHeadDim}, kr), bc::value({T, kKvHeads, kHeadDim}, v),
         bc::value({T}, slot_values, "i32")},
        {bc::value({b.n_blocks(), kKvHeads, b.block_size, 2 * kHeadDim}, pool)})[0].value;

    const auto bt = b.block_table();
    const auto seqused = b.seqused();
    const auto cu = b.cu_seqlens();
    const auto attn_out = bc::kernel_run(
        attn_rec(b),
        {bc::value({T, kHeads, kHeadDim}, qr),
         bc::value({b.n_blocks(), kKvHeads, b.block_size, 2 * kHeadDim}, pooled),
         bc::value({b.n_seq(), b.max_blocks()}, std::vector<double>(bt.begin(), bt.end()), "i32"),
         bc::value({b.n_seq()}, std::vector<double>(seqused.begin(), seqused.end()), "i32"),
         bc::value({b.n_seq() * kKvHeads},
                   std::vector<double>((size_t)(b.n_seq() * kKvHeads), 1.0), "f32"),
         bc::value({b.n_seq() * kKvHeads},
                   std::vector<double>((size_t)(b.n_seq() * kKvHeads), 1.0), "f32"),
         bc::value({b.n_seq() + 1}, std::vector<double>(cu.begin(), cu.end()), "i32")})[0].value;

    const auto g = bc::kernel_run(sigmoid_rec(T * kHeads, kHeadDim),
                                  {bc::value({T * kHeads, kHeadDim}, gate)})[0].value;
    const auto gated = bc::kernel_run(mul_rec(T, kQDim),
                                      {bc::value({T, kQDim}, attn_out),
                                       bc::value({T, kQDim}, g)})[0].value;
    const auto o = bc::kernel_run(gemm_rec(T, kHidden, kQDim),
                                  {bc::value({T, kQDim}, gated),
                                   bc::value({kHidden, kQDim}, L.o_proj)})[0].value;
    return bc::kernel_run(add_rec(T, kHidden),
                          {bc::value({T, kHidden}, x), bc::value({T, kHidden}, o)})[0].value;
}

std::vector<double> mlp_kernel_chain(const Blocks& B, const Batch& b,
                                     const std::vector<double>& x) {
    const Layer& L = B.mlp;
    const int64_t T = b.total();
    const auto h = bc::kernel_run(rmsnorm_rec(T, kHidden),
                                  {bc::value({T, kHidden}, x),
                                   bc::value({kHidden}, L.post_norm)})[0].value;
    const auto gu = bc::kernel_run(gemm_rec(T, 2 * kFfn, kHidden),
                                   {bc::value({T, kHidden}, h),
                                    bc::value({2 * kFfn, kHidden}, L.gate_up)})[0].value;
    const auto ffn = bc::kernel_run(silu_mul_rec(T, kFfn), {bc::value({T, 2 * kFfn}, gu)})[0].value;
    const auto o = bc::kernel_run(gemm_rec(T, kHidden, kFfn),
                                  {bc::value({T, kFfn}, ffn),
                                   bc::value({kHidden, kFfn}, L.down_proj)})[0].value;
    return bc::kernel_run(add_rec(T, kHidden),
                          {bc::value({T, kHidden}, x), bc::value({T, kHidden}, o)})[0].value;
}

// ------------------------------------------------------------------ checks

double max_abs(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m = std::max(m, std::fabs(x));
    return m;
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size())
        throw std::runtime_error("block: shapes differ, " + std::to_string(a.size()) + " and " +
                                 std::to_string(b.size()) + " values");
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

void require_close(const std::string& what, const std::vector<double>& got,
                   const std::vector<double>& want, double rel_tol) {
    const double diff = max_abs_diff(got, want);
    const double scale = std::max(max_abs(want), 1.0);
    if (!(diff <= rel_tol * scale)) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.3g", diff / scale);
        throw std::runtime_error(what + ": largest difference " + std::to_string(diff) +
                                 ", relative " + buf + ", tolerance " +
                                 std::to_string(rel_tol));
    }
}

struct OracleCase {
    std::vector<int64_t> lens;
    std::vector<double> input, output, cotangent, input_cotangent;  // packed valid rows
};

OracleCase oracle_case(const r97::oracle::detail::File& ref, const std::string& name,
                       const std::string& tag, const std::vector<int64_t>& lens) {
    const std::string p = name + "." + tag + ".";
    const auto& in = ref.tensor(p + "input");
    const auto& out = ref.tensor(p + "output");
    const auto& cot = ref.tensor(p + "cotangent");
    const auto& d_in = ref.tensor(p + "input_cotangent");
    const auto& want_lens = ref.tensor(p + "lengths");
    const int64_t n_seq = (int64_t)lens.size();
    if (want_lens.data.size() != (size_t)n_seq)
        throw std::runtime_error("block: " + p + "lengths has " +
                                 std::to_string(want_lens.data.size()) + " entries for " +
                                 std::to_string(n_seq) + " sequences");
    for (int64_t s = 0; s < n_seq; ++s)
        if (int64_t(want_lens.data[(size_t)s]) != lens[(size_t)s])
            throw std::runtime_error("block: " + p + "lengths says " +
                                     std::to_string((int64_t)want_lens.data[(size_t)s]) +
                                     " for sequence " + std::to_string(s));
    const int64_t rows = in.shape[1];
    if (in.shape[0] != n_seq || out.shape[0] != n_seq || cot.shape[0] != n_seq ||
        d_in.shape[0] != n_seq || in.shape[2] != kHidden)
        throw std::runtime_error("block: " + p + "input has an unexpected shape");

    OracleCase c;
    c.lens = lens;
    for (const r97::oracle::Tensor* t : {&in, &out, &cot, &d_in}) {
        std::vector<double>& v = t == &in    ? c.input
                                 : t == &out ? c.output
                                 : t == &cot ? c.cotangent
                                             : c.input_cotangent;
        v.reserve((size_t)(n_seq * rows * kHidden));
        for (int64_t s = 0; s < n_seq; ++s)
            for (int64_t i = 0; i < lens[(size_t)s]; ++i)
                for (int64_t h = 0; h < kHidden; ++h)
                    v.push_back(t->at({s, i, h}));
    }
    return c;
}

// One sequence's valid rows, cut out of a packed batch tensor.
std::vector<double> rows_of(const std::vector<double>& packed, const Batch& b, int64_t s) {
    std::vector<double> v;
    int64_t off = 0;
    for (int64_t i = 0; i < s; ++i) off += b.lens[(size_t)i];
    for (int64_t i = 0; i < b.lens[(size_t)s]; ++i)
        for (int64_t h = 0; h < kHidden; ++h)
            v.push_back(packed[(size_t)((off + i) * kHidden + h)]);
    return v;
}

const r97::oracle::detail::File& reference() {
    static const r97::oracle::detail::File f(
        (std::filesystem::path(__FILE__).parent_path().parent_path() / "fixtures" / "oracle" /
         "reference.safetensors")
            .string());
    return f;
}

}  // namespace

// One sequence: 23 tokens, which crosses the 16-token block boundary.
R97_TEST(block_attention_single) {
    const Batch b{{23}};
    const OracleCase c = oracle_case(reference(), "attention_sublayer", "single", b.lens);
    const BlockResult got = attention_block(blocks(), b, c.input, &c.cotangent);
    require_close("attention_sublayer single forward", got.y, c.output, kOracleTolerance);
    require_close("attention_sublayer single adjoint", got.dx, c.input_cotangent,
                  kOracleTolerance);

    const auto dir = bc::direction(c.input.size(), 0x51ed1u);
    const auto rep = bc::check_direction(
        "attention_sublayer single",
        [&](const std::vector<double>& x) { return attention_block(blocks(), b, x, nullptr).y; },
        [&](const std::vector<double>& x) {
            return attention_block(blocks(), b, x, &c.cotangent).dx;
        },
        c.input, c.cotangent, dir, kFdTolerance);
    if (!rep.ok) throw std::runtime_error(rep.note);

    require_close("attention_sublayer single host kernels",
                  attention_kernel_chain(blocks(), b, c.input), got.y, kBf16Tolerance);
}

// A batch of three unequal lengths: 17, 11 and 6 tokens. The first sequence crosses a pool
// block boundary; the pool cotangent has to land on the right sequence and position.
R97_TEST(block_attention_batch) {
    const Batch b{{17, 11, 6}};
    const OracleCase c = oracle_case(reference(), "attention_sublayer", "batch", b.lens);
    const BlockResult got = attention_block(blocks(), b, c.input, &c.cotangent);
    require_close("attention_sublayer batch forward", got.y, c.output, kOracleTolerance);
    // The sequence that fills the batch is compared as the single one is. The two shorter ones
    // carry the reference's padded-row term: see kPaddedRowTolerance.
    for (int64_t s = 0; s < b.n_seq(); ++s) {
        const bool padded = b.lens[(size_t)s] != b.max_len();
        require_close("attention_sublayer batch adjoint, sequence " + std::to_string(s),
                      rows_of(got.dx, b, s), rows_of(c.input_cotangent, b, s),
                      padded ? kPaddedRowTolerance : kOracleTolerance);
    }

    const auto dir = bc::direction(c.input.size(), 0xb47c4u);
    const auto rep = bc::check_direction(
        "attention_sublayer batch",
        [&](const std::vector<double>& x) { return attention_block(blocks(), b, x, nullptr).y; },
        [&](const std::vector<double>& x) {
            return attention_block(blocks(), b, x, &c.cotangent).dx;
        },
        c.input, c.cotangent, dir, kFdTolerance);
    if (!rep.ok) throw std::runtime_error(rep.note);

    require_close("attention_sublayer batch host kernels",
                  attention_kernel_chain(blocks(), b, c.input), got.y, kBf16Tolerance);
}

R97_TEST(block_mlp_single) {
    const Batch b{{23}};
    const OracleCase c = oracle_case(reference(), "mlp_sublayer", "single", b.lens);
    const BlockResult got = mlp_block(blocks(), b, c.input, &c.cotangent);
    require_close("mlp_sublayer single forward", got.y, c.output, kOracleTolerance);
    require_close("mlp_sublayer single adjoint", got.dx, c.input_cotangent, kOracleTolerance);

    const auto dir = bc::direction(c.input.size(), 0x3d9a2u);
    const auto rep = bc::check_direction(
        "mlp_sublayer single",
        [&](const std::vector<double>& x) { return mlp_block(blocks(), b, x, nullptr).y; },
        [&](const std::vector<double>& x) { return mlp_block(blocks(), b, x, &c.cotangent).dx; },
        c.input, c.cotangent, dir, kFdTolerance);
    if (!rep.ok) throw std::runtime_error(rep.note);

    require_close("mlp_sublayer single host kernels", mlp_kernel_chain(blocks(), b, c.input), got.y,
                  kBf16Tolerance);
}

R97_TEST(block_mlp_batch) {
    const Batch b{{17, 11, 6}};
    const OracleCase c = oracle_case(reference(), "mlp_sublayer", "batch", b.lens);
    const BlockResult got = mlp_block(blocks(), b, c.input, &c.cotangent);
    require_close("mlp_sublayer batch forward", got.y, c.output, kOracleTolerance);
    require_close("mlp_sublayer batch adjoint", got.dx, c.input_cotangent, kOracleTolerance);

    const auto dir = bc::direction(c.input.size(), 0x7c31bu);
    const auto rep = bc::check_direction(
        "mlp_sublayer batch",
        [&](const std::vector<double>& x) { return mlp_block(blocks(), b, x, nullptr).y; },
        [&](const std::vector<double>& x) { return mlp_block(blocks(), b, x, &c.cotangent).dx; },
        c.input, c.cotangent, dir, kFdTolerance);
    if (!rep.ok) throw std::runtime_error(rep.note);

    require_close("mlp_sublayer batch host kernels", mlp_kernel_chain(blocks(), b, c.input), got.y,
                  kBf16Tolerance);
}
