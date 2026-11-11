// tests/host/gdn_layer.cpp -- the gated delta net, one whole layer at a time.
//
// Three checks, from the cheapest to the strongest:
//
//   1. the chunked twins (gdn_kkt_solve then gdn_chunk_scan) against the per-token
//      recurrence in tests/support/gdn_reference.h, forward and reverse, with a
//      finite-difference check through the composition, for sequences shorter than,
//      equal to and several times the chunk length, and for a ragged batch;
//   2. the layer's twins against radiance's host kernels on the bf16 fixture, one op
//      at a time along the kernel's own chain, within bf16 tolerance;
//   3. the layer's twins, in f64 and from the checkpoint's weights, against the
//      PyTorch reference's gdn_mixer case: the output, and the composed adjoint's
//      input cotangent. This is the one check of the delta net that does not rest on
//      docs/adjoints/gdn-math.md.
//
// The tape follows the bf16 fixture's graph (arch/common/rad_block_gdn.h): the input
// projections, gdn_conv_prep, gdn_kkt_solve, gdn_chunk_scan, gdn_gated_rmsnorm, and the
// output gemm. One deliberate difference:
// gdn_conv_prep's twin models the unfused form (a and b arrive as their own operands,
// x holds only the convolved channels), so the in-projection is issued as four gemms
// (qkv, z, a, b) against the same checkpoint tensors instead of the plugin's packed
// [q|k|v|a|b] row. The arithmetic is the same; only the record split differs.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "support/adjoint_check.h"
#include "support/fixture.h"
#include "support/gdn_reference.h"
#include "support/oracle.h"
#include "support/test.h"

namespace {

namespace ac = r97::adjoint_check;
namespace ref = r97::gdn_reference;

const double bf16_tol = 0.02;
const double fd_tol = 1e-8;

int64_t numel_of(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

r97::AdValue value_of(const std::string& dtype, std::vector<int64_t> shape,
                      std::vector<double> values) {
    r97::AdValue v;
    v.dtype = dtype;
    v.shape = std::move(shape);
    v.value = std::move(values);
    if ((int64_t)v.value.size() != numel_of(v.shape))
        ac::fail("a test value has " + std::to_string(v.value.size()) + " values for a shape of " +
                 std::to_string(numel_of(v.shape)));
    return v;
}

r97::AdValue zeros_like(const r97::AdValue& v) {
    r97::AdValue z = v;
    z.value.assign(v.value.size(), 0.0);
    return z;
}

std::string f64_text(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
}

r97::Operand operand(const std::string& name, const std::string& dtype,
                     std::vector<int64_t> shape, bool written) {
    r97::Operand o;
    o.buffer = name;
    o.dtype = dtype;
    o.shape = std::move(shape);
    o.rows = o.shape.empty() ? 0 : o.shape[0];
    o.written = written;
    return o;
}

r97::Operand read_operand(const std::string& name, const std::string& dtype,
                          std::vector<int64_t> shape) {
    return operand(name, dtype, std::move(shape), false);
}

r97::Operand write_operand(const std::string& name, const std::string& dtype,
                           std::vector<int64_t> shape) {
    return operand(name, dtype, std::move(shape), true);
}

// One op of a tape. The record's operands carry the schema's names (`a`, `b`, `y`,
// `x`, `w`, ...), which the linear twins look up by name; a step therefore maps each
// operand to the key its value lives under in the tape's store, so several gemms can
// share the schema names without sharing a buffer.
struct Step {
    r97::OpRecord rec;
    std::map<std::string, std::string> in;   // operand name -> store key
    std::map<std::string, std::string> out;  // operand name -> store key
};

Step step_of(r97::OpRecord rec) {
    Step s;
    s.rec = std::move(rec);
    return s;
}

const std::string& in_key(const Step& s, const std::string& name) {
    const auto it = s.in.find(name);
    return it == s.in.end() ? name : it->second;
}

const std::string& out_key(const Step& s, const std::string& name) {
    const auto it = s.out.find(name);
    return it == s.out.end() ? name : it->second;
}

std::vector<r97::AdValue> reads_of(const Step& s,
                                   const std::map<std::string, r97::AdValue>& store) {
    std::vector<r97::AdValue> in;
    for (const r97::Operand& o : s.rec.operands) {
        if (o.written) continue;
        const auto it = store.find(in_key(s, o.buffer));
        if (it == store.end()) ac::fail("no forward value for operand '" + o.buffer + "'");
        in.push_back(it->second);
    }
    return in;
}

std::vector<r97::AdValue> write_specs(const r97::OpRecord& rec) {
    std::vector<r97::AdValue> out;
    for (const r97::Operand& o : rec.operands) {
        if (!o.written) continue;
        r97::AdValue spec;
        spec.dtype = o.dtype;
        spec.shape = o.shape;
        out.push_back(std::move(spec));
    }
    return out;
}

// The forward through a whole tape: every read comes from the store by key, every write
// goes back into it. Constants (weights, cu, the zero state) are pre-seeded.
void run_forward(const std::vector<Step>& tape, std::map<std::string, r97::AdValue>* store) {
    for (const Step& s : tape) {
        std::string err;
        const r97::TwinFn twin = r97::twin_find(s.rec.op, s.rec.params, &err);
        if (!twin) ac::fail(err);
        const std::vector<r97::AdValue> in = reads_of(s, *store);
        std::vector<r97::AdValue> out;
        twin(s.rec, in, &out);
        size_t w = 0;
        for (const r97::Operand& o : s.rec.operands) {
            if (!o.written) continue;
            if (w >= out.size())
                ac::fail("twin '" + s.rec.op + "' wrote " + std::to_string(out.size()) +
                         " values for " + std::to_string(w + 1) + " written operands");
            (*store)[out_key(s, o.buffer)] = std::move(out[w++]);
        }
        if (w != out.size())
            ac::fail("twin '" + s.rec.op + "' wrote " + std::to_string(out.size()) +
                     " values for " + std::to_string(w) + " written operands");
    }
}

// The reverse through the same tape: each op's adjoint adds into the cotangents of its
// reads, so a buffer read by several ops (k, g and beta are read by kkt and by scan)
// accumulates them. Frozen operands (the weights) are skipped.
void run_backward(const std::vector<Step>& tape,
                  const std::map<std::string, r97::AdValue>& store,
                  std::map<std::string, r97::AdValue>* cot) {
    for (auto it = tape.rbegin(); it != tape.rend(); ++it) {
        const Step& s = *it;
        const r97::OpRecord& rec = s.rec;
        std::string err;
        const r97::AdjointFn adj = r97::adjoint_find(rec.op, r97::Side::host, rec.params, &err);
        if (!adj) ac::fail(err);
        const std::vector<int> frozen = r97::adjoint_frozen(rec.op, r97::Side::host, rec.params);

        std::vector<r97::AdValue> fwd_vec;
        fwd_vec.reserve(rec.operands.size());
        for (const r97::Operand& o : rec.operands) {
            const std::string& key = o.written ? out_key(s, o.buffer) : in_key(s, o.buffer);
            const auto f = store.find(key);
            if (f == store.end()) ac::fail("no forward value for operand '" + o.buffer + "'");
            fwd_vec.push_back(f->second);
        }
        std::vector<r97::AdValue> cot_vec;
        for (size_t i = 0; i < rec.operands.size(); ++i) {
            if (!rec.operands[i].written) continue;
            const auto c = cot->find(out_key(s, rec.operands[i].buffer));
            cot_vec.push_back(c == cot->end() ? zeros_like(fwd_vec[i]) : c->second);
        }
        std::vector<r97::AdValue> dc;
        dc.reserve(fwd_vec.size());
        for (const r97::AdValue& a : fwd_vec) dc.push_back(zeros_like(a));

        const r97::AdjointCall call{rec, fwd_vec, cot_vec, &dc, nullptr, ""};
        adj(call);

        for (size_t i = 0; i < rec.operands.size(); ++i) {
            const r97::Operand& o = rec.operands[i];
            if (o.written) continue;
            if (std::find(frozen.begin(), frozen.end(), (int)i) != frozen.end()) continue;
            const std::string& key = in_key(s, o.buffer);
            auto pos = cot->try_emplace(key, zeros_like(fwd_vec[i]));
            r97::AdValue& acc = pos.first->second;
            R97_CHECK(acc.value.size() == dc[i].value.size());
            for (size_t e = 0; e < acc.value.size(); ++e) acc.value[e] += dc[i].value[e];
        }
    }
}

void check_close(const std::vector<double>& got, const std::vector<double>& want, double tol,
                 const std::string& what) {
    if (got.size() != want.size())
        ac::fail(what + ": " + std::to_string(got.size()) + " values against " +
                 std::to_string(want.size()));
    for (size_t i = 0; i < got.size(); ++i) {
        if (!ac::close_enough(got[i], want[i], tol)) {
            char buf[320];
            std::snprintf(buf, sizeof buf, "%s: element %zu: %.17g vs %.17g", what.c_str(), i,
                          got[i], want[i]);
            ac::fail(buf);
        }
    }
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    R97_CHECK(a.size() == b.size());
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

// ---------------------------------------------------------------- the recurrence

struct RecShape {
    int64_t k = 8;        // head_k
    int64_t v = 8;        // head_v
    int64_t hk = 2;       // key heads
    int64_t hv = 4;       // value heads
    int64_t chunk = 5;
    std::vector<int64_t> lens;
    double scale = 0.35355339059327379;  // 1/sqrt(head_k)

    int64_t tokens() const {
        int64_t t = 0;
        for (int64_t n : lens) t += n;
        return t;
    }
    int64_t seqs() const { return (int64_t)lens.size(); }
};

std::vector<int64_t> cu_of(const std::vector<int64_t>& lens) {
    std::vector<int64_t> cu{0};
    for (int64_t n : lens) cu.push_back(cu.back() + n);
    return cu;
}

r97::OpRecord kkt_record(const RecShape& s) {
    const int64_t t = s.tokens();
    r97::OpRecord rec;
    rec.op = "gdn_kkt_solve";
    rec.n = t;
    rec.params = {{"M", t}, {"head_k", s.k}, {"chunk", s.chunk}};
    rec.operands = {
        read_operand("k", "bf16", {t, s.hk, s.k}),
        read_operand("beta", "f32", {t, s.hv}),
        read_operand("g", "f32", {t, s.hv}),
        read_operand("cu", "i32", {s.seqs() + 1}),
        write_operand("A", "bf16", {t, s.hv, s.chunk}),
    };
    return rec;
}

r97::OpRecord scan_record(const RecShape& s) {
    const int64_t t = s.tokens();
    const int64_t seq = s.seqs();
    r97::OpRecord rec;
    rec.op = "gdn_chunk_scan";
    rec.n = t;
    rec.params = {{"M", t},
                  {"head_k", s.k},
                  {"head_v", s.v},
                  {"chunk", s.chunk},
                  {"scale", f64_text(s.scale)}};
    rec.operands = {
        read_operand("q", "bf16", {t, s.hk, s.k}),
        read_operand("k", "bf16", {t, s.hk, s.k}),
        read_operand("v", "bf16", {t, s.hv, s.v}),
        read_operand("A", "bf16", {t, s.hv, s.chunk}),
        read_operand("g", "f32", {t, s.hv}),
        read_operand("beta", "f32", {t, s.hv}),
        read_operand("h0", "f32", {seq, s.hv, s.v, s.k}),
        read_operand("cu", "i32", {seq + 1}),
        write_operand("o", "bf16", {t, s.hv, s.v}),
        write_operand("ht", "f32", {seq, s.hv, s.v, s.k}),
    };
    return rec;
}

std::vector<Step> rec_tape(const RecShape& s) {
    return {step_of(kkt_record(s)), step_of(scan_record(s))};
}

ref::Geometry rec_geometry(const RecShape& s) {
    ref::Geometry g;
    g.key_heads = s.hk;
    g.value_heads = s.hv;
    g.head_k = s.k;
    g.head_v = s.v;
    g.chunk = s.chunk;
    g.scale = s.scale;
    return g;
}

struct RecCase {
    RecShape shape;
    ref::Inputs in;
    std::map<std::string, r97::AdValue> fwd;  // the tape's constants and inputs
};

RecCase rec_case(const RecShape& s, uint64_t seed) {
    RecCase c;
    c.shape = s;
    const int64_t t = s.tokens();
    c.in.cu = cu_of(s.lens);
    c.in.q.resize((size_t)(t * s.hk * s.k));
    c.in.k.resize((size_t)(t * s.hk * s.k));
    c.in.v.resize((size_t)(t * s.hv * s.v));
    c.in.g.resize((size_t)(t * s.hv));
    c.in.beta.resize((size_t)(t * s.hv));

    for (double& x : c.in.q) x = ac::urand(&seed) * 2.0 - 1.0;
    for (double& x : c.in.k) x = ac::urand(&seed) * 2.0 - 1.0;
    for (double& x : c.in.v) x = ac::urand(&seed) * 2.0 - 1.0;
    for (int64_t seq = 0; seq < s.seqs(); ++seq) {
        for (int64_t h = 0; h < s.hv; ++h) {
            double acc = 0.0;
            for (int64_t t0 = c.in.cu[(size_t)seq]; t0 < c.in.cu[(size_t)seq + 1]; ++t0) {
                if ((t0 - c.in.cu[(size_t)seq]) % s.chunk == 0) acc = 0.0;
                acc -= 0.02 + 0.2 * ac::urand(&seed);
                c.in.g[(size_t)(t0 * s.hv + h)] = acc;
                c.in.beta[(size_t)(t0 * s.hv + h)] =
                    1.0 / (1.0 + std::exp(-(ac::urand(&seed) * 2.0 - 1.0)));
            }
        }
    }

    c.fwd["q"] = value_of("bf16", {t, s.hk, s.k}, c.in.q);
    c.fwd["k"] = value_of("bf16", {t, s.hk, s.k}, c.in.k);
    c.fwd["v"] = value_of("bf16", {t, s.hv, s.v}, c.in.v);
    c.fwd["g"] = value_of("f32", {t, s.hv}, c.in.g);
    c.fwd["beta"] = value_of("f32", {t, s.hv}, c.in.beta);
    c.fwd["cu"] = value_of("i32", {s.seqs() + 1},
                           std::vector<double>(c.in.cu.begin(), c.in.cu.end()));
    c.fwd["h0"] = value_of("f32", {s.seqs(), s.hv, s.v, s.k},
                           std::vector<double>((size_t)(s.seqs() * s.hv * s.v * s.k), 0.0));
    return c;
}

double recurrence_loss(const RecShape& s, const std::map<std::string, r97::AdValue>& base,
                       const std::map<std::string, std::vector<double>>& dir, double eps,
                       const std::vector<double>& dO, const std::vector<double>& dHt) {
    std::map<std::string, r97::AdValue> fwd = base;
    for (const auto& d : dir) {
        std::vector<double>& v = fwd.at(d.first).value;
        R97_CHECK(v.size() == d.second.size());
        for (size_t i = 0; i < v.size(); ++i) v[i] += eps * d.second[i];
    }
    run_forward(rec_tape(s), &fwd);
    return dot(fwd.at("o").value, dO) + dot(fwd.at("ht").value, dHt);
}

// ---------------------------------------------------------------- the whole layer

// The tiny fixture's gdn geometry, from tests/support/fixture.h and docs/FIXTURES.md.
struct LayerShape {
    int64_t hidden = 128;
    int64_t key_heads = 2;   // Hg
    int64_t value_heads = 4; // H
    int64_t head_k = 16;     // K
    int64_t head_v = 16;     // V
    int64_t conv_width = 4;
    int64_t chunk = 64;
    double eps = 1e-6;

    int64_t conv_dim() const { return 2 * key_heads * head_k + value_heads * head_v; }
    int64_t value_dim() const { return value_heads * head_v; }
    double scale() const { return 1.0 / std::sqrt((double)head_k); }
};

struct LayerWeights {
    r97::AdValue wqkv, wz, wa, wb, conv_w, conv_b, a_log, dt_bias, norm_w, w_out;
};

r97::AdValue checkpoint_tensor(const r97::oracle::detail::File& f, const std::string& name,
                               const std::string& dtype, const std::vector<int64_t>& shape) {
    const r97::oracle::Tensor& t = f.tensor(name);
    R97_CHECK(t.shape == shape);
    return value_of(dtype, shape, t.data);
}

LayerWeights read_weights(const std::string& checkpoint, const LayerShape& s) {
    const r97::oracle::detail::File f(checkpoint + "/model.safetensors");
    const std::string p = "model.language_model.layers.0.linear_attn.";
    LayerWeights w;
    w.wqkv = checkpoint_tensor(f, p + "in_proj_qkv.weight", "bf16",
                               {s.conv_dim(), s.hidden});
    w.wz = checkpoint_tensor(f, p + "in_proj_z.weight", "bf16", {s.value_dim(), s.hidden});
    w.wa = checkpoint_tensor(f, p + "in_proj_a.weight", "bf16", {s.value_heads, s.hidden});
    w.wb = checkpoint_tensor(f, p + "in_proj_b.weight", "bf16", {s.value_heads, s.hidden});
    w.conv_w = checkpoint_tensor(f, p + "conv1d.weight", "bf16",
                                 {s.conv_dim(), 1, s.conv_width});
    w.conv_w.shape = {s.conv_dim(), s.conv_width};
    w.conv_b = value_of("bf16", {s.conv_dim()}, std::vector<double>((size_t)s.conv_dim(), 0.0));
    w.a_log = checkpoint_tensor(f, p + "A_log", "f32", {s.value_heads});
    w.dt_bias = checkpoint_tensor(f, p + "dt_bias", "f32", {s.value_heads});
    w.norm_w = checkpoint_tensor(f, p + "norm.weight", "f32", {s.head_v});
    w.w_out = checkpoint_tensor(f, p + "out_proj.weight", "bf16", {s.hidden, s.value_dim()});
    return w;
}

Step gemm_step(const LayerShape& s, int64_t t, const std::string& lhs,
               const std::string& weight, const std::string& result, int64_t n, int64_t k) {
    r97::OpRecord rec;
    rec.op = "gemm_nt";
    rec.n = t;
    rec.params = {{"M", t}, {"N", n}, {"K", k}, {"dtype", "bf16"}};
    rec.operands = {
        read_operand("a", "bf16", {t, k}),
        read_operand("b", "bf16", {n, k}),
        write_operand("y", "bf16", {t, n}),
    };
    Step st = step_of(std::move(rec));
    st.in = {{"a", lhs}, {"b", weight}};
    st.out = {{"y", result}};
    return st;
}

std::vector<Step> layer_tape(const LayerShape& s, int64_t t, int64_t seq) {
    const int64_t cd = s.conv_dim(), vd = s.value_dim(), H = s.value_heads;
    std::vector<Step> tape;
    tape.push_back(gemm_step(s, t, "h", "wqkv", "xqkv", cd, s.hidden));
    tape.push_back(gemm_step(s, t, "h", "wz", "z", vd, s.hidden));
    tape.push_back(gemm_step(s, t, "h", "wa", "a_proj", H, s.hidden));
    tape.push_back(gemm_step(s, t, "h", "wb", "b_proj", H, s.hidden));

    {
        r97::OpRecord rec;
        rec.op = "gdn_conv_prep";
        rec.n = t;
        rec.params = {{"M", t},
                      {"head_k", s.head_k},
                      {"head_v", s.head_v},
                      {"chunk", s.chunk},
                      {"conv_width", s.conv_width},
                      {"dtype", "bf16"}};
        rec.operands = {
            read_operand("x", "bf16", {t, cd}),
            read_operand("w", "bf16", {cd, s.conv_width}),
            read_operand("b", "bf16", {cd}),
            read_operand("A_log", "f32", {H}),
            read_operand("dt_bias", "f32", {H}),
            write_operand("conv_state", "bf16", {seq, cd, s.conv_width - 1}),
            read_operand("cu_seqlens", "i32", {seq + 1}),
            read_operand("a", "bf16", {t, H}),
            read_operand("b_gate", "bf16", {t, H}),
            read_operand("cache_idx", "i32", {seq}),
            read_operand("has_init", "i32", {seq}),
            write_operand("q", "bf16", {t, s.key_heads, s.head_k}),
            write_operand("k", "bf16", {t, s.key_heads, s.head_k}),
            write_operand("v", "bf16", {t, H, s.head_v}),
            write_operand("g", "f32", {t, H}),
            write_operand("beta", "f32", {t, H}),
        };
        Step st = step_of(std::move(rec));
        st.in = {{"x", "xqkv"},
                 {"w", "conv_w"},
                 {"b", "conv_b"},
                 {"A_log", "a_log"},
                 {"dt_bias", "dt_bias"},
                 {"cu_seqlens", "cu"},
                 {"a", "a_proj"},
                 {"b_gate", "b_proj"},
                 {"cache_idx", "cache_idx"},
                 {"has_init", "has_init"}};
        st.out = {{"conv_state", "conv_state"}};
        tape.push_back(std::move(st));
    }
    {
        r97::OpRecord rec;
        rec.op = "gdn_kkt_solve";
        rec.n = t;
        rec.params = {{"M", t}, {"head_k", s.head_k}, {"chunk", s.chunk}};
        rec.operands = {
            read_operand("k", "bf16", {t, s.key_heads, s.head_k}),
            read_operand("beta", "f32", {t, H}),
            read_operand("g", "f32", {t, H}),
            read_operand("cu_seqlens", "i32", {seq + 1}),
            write_operand("A", "bf16", {t, H, s.chunk}),
        };
        Step st = step_of(std::move(rec));
        st.in = {{"cu_seqlens", "cu"}};
        tape.push_back(std::move(st));
    }
    {
        r97::OpRecord rec;
        rec.op = "gdn_chunk_scan";
        rec.n = t;
        rec.params = {{"M", t},
                      {"head_k", s.head_k},
                      {"head_v", s.head_v},
                      {"chunk", s.chunk},
                      {"scale", f64_text(s.scale())}};
        rec.operands = {
            read_operand("q", "bf16", {t, s.key_heads, s.head_k}),
            read_operand("k", "bf16", {t, s.key_heads, s.head_k}),
            read_operand("v", "bf16", {t, H, s.head_v}),
            read_operand("A", "bf16", {t, H, s.chunk}),
            read_operand("g", "f32", {t, H}),
            read_operand("beta", "f32", {t, H}),
            read_operand("h0", "f32", {seq, H, s.head_v, s.head_k}),
            read_operand("cu_seqlens", "i32", {seq + 1}),
            write_operand("o", "bf16", {t, H, s.head_v}),
            write_operand("ht", "f32", {seq, H, s.head_v, s.head_k}),
        };
        Step st = step_of(std::move(rec));
        st.in = {{"cu_seqlens", "cu"}};
        st.out = {{"o", "o_scan"}};
        tape.push_back(std::move(st));
    }
    {
        r97::OpRecord rec;
        rec.op = "gdn_gated_rmsnorm";
        rec.n = t;
        rec.params = {{"M", t},
                      {"channels", s.head_v},
                      {"act", "silu"},
                      {"dtype", "bf16"},
                      {"eps", f64_text(s.eps)}};
        rec.operands = {
            read_operand("x", "bf16", {t, vd}),
            read_operand("z", "bf16", {t, vd}),
            read_operand("w", "f32", {s.head_v}),
            write_operand("o", "bf16", {t, vd}),
        };
        Step st = step_of(std::move(rec));
        st.in = {{"x", "o_scan"}, {"w", "norm_w"}};
        st.out = {{"o", "o_norm"}};
        tape.push_back(std::move(st));
    }
    tape.push_back(gemm_step(s, t, "o_norm", "w_out", "out", s.hidden, vd));
    return tape;
}

std::map<std::string, r97::AdValue> layer_constants(const LayerShape& s,
                                                    const LayerWeights& w, int64_t t,
                                                    int64_t seq,
                                                    const std::vector<int64_t>& cu,
                                                    std::vector<double> h) {
    const int64_t H = s.value_heads;
    std::map<std::string, r97::AdValue> fwd;
    fwd["h"] = value_of("bf16", {t, s.hidden}, std::move(h));
    fwd["wqkv"] = w.wqkv;
    fwd["wz"] = w.wz;
    fwd["wa"] = w.wa;
    fwd["wb"] = w.wb;
    fwd["conv_w"] = w.conv_w;
    fwd["conv_b"] = w.conv_b;
    fwd["a_log"] = w.a_log;
    fwd["dt_bias"] = w.dt_bias;
    fwd["norm_w"] = w.norm_w;
    fwd["w_out"] = w.w_out;
    fwd["cu"] = value_of("i32", {seq + 1},
                         std::vector<double>(cu.begin(), cu.end()));
    std::vector<double> idx((size_t)seq), zero((size_t)seq, 0.0);
    for (int64_t i = 0; i < seq; ++i) idx[(size_t)i] = (double)i;
    fwd["cache_idx"] = value_of("i32", {seq}, std::move(idx));
    fwd["has_init"] = value_of("i32", {seq}, zero);
    fwd["h0"] = value_of("f32", {seq, H, s.head_v, s.head_k},
                         std::vector<double>((size_t)(seq * H * s.head_v * s.head_k), 0.0));
    return fwd;
}

// ---------------------------------------------------------------- libref's chain

struct Launched {
    std::vector<r97::AdValue> reads;   // exactly the values the kernel read
    std::vector<r97::AdValue> writes;  // what it wrote
};

Launched launch_host(const r97::OpRecord& rec, const std::vector<r97::AdValue>& in,
                     const std::vector<r97::AdValue>& out_spec) {
    Launched out;
    const RadKernelInfo* k = ac::find_host_kernel(rec.op.c_str());
    if (!k) ac::fail("no host kernel for op '" + rec.op + "'");
    std::vector<RadTensor> t(rec.operands.size());
    size_t r = 0, w = 0;
    for (size_t i = 0; i < rec.operands.size(); ++i) {
        const r97::AdValue& a = rec.operands[i].written ? out_spec[w++] : in[r++];
        RadTensor x{};
        x.dtype = rad_dtype_parse(a.dtype.c_str());
        if (rad_dtype_bits(x.dtype) <= 0) ac::fail("not a radiance dtype: " + a.dtype);
        x.rank = (uint32_t)a.shape.size();
        int64_t n = 1;
        for (size_t d = a.shape.size(); d-- > 0;) {
            x.shape[d] = a.shape[d];
            x.stride[d] = n;
            n *= a.shape[d];
        }
        const int64_t bits = (int64_t)rad_dtype_bits(x.dtype) * n;
        x.data = std::calloc((size_t)((bits + 7) / 8 > 0 ? (bits + 7) / 8 : 1), 1);
        if (!rec.operands[i].written)  // a write starts zeroed
            for (int64_t e = 0; e < n; ++e)
                rad_store_f32(x.data, x.dtype, e, (float)a.value[(size_t)e]);
        t[i] = x;
    }

    out.reads = in;
    r = 0;
    for (size_t i = 0; i < rec.operands.size(); ++i) {
        if (rec.operands[i].written) continue;
        const int64_t n = numel_of(rec.operands[i].shape);
        for (int64_t e = 0; e < n; ++e)
            out.reads[r].value[(size_t)e] = rad_load_f32(t[i].data, t[i].dtype, e);
        ++r;
    }

    std::vector<RadParam> ps = ac::rec_params(rec);
    RadArgs args{};
    args.t = t.data();
    args.n_t = (int)t.size();
    args.p = ps.data();
    args.n_p = (int)ps.size();
    args.instance = nullptr;
    args.rank = 0;
    args.world_size = 1;
    const int rc = k->launch(&args, nullptr);
    if (rc != RAD_OK) {
        for (RadTensor& x : t) std::free(x.data);
        ac::fail("host kernel '" + rec.op + "' failed: rc=" + std::to_string(rc));
    }

    out.writes = out_spec;
    w = 0;
    for (size_t i = 0; i < rec.operands.size(); ++i) {
        if (!rec.operands[i].written) continue;
        const int64_t n = numel_of(rec.operands[i].shape);
        out.writes[w].value.resize((size_t)n);
        for (int64_t e = 0; e < n; ++e)
            out.writes[w].value[(size_t)e] = rad_load_f32(t[i].data, t[i].dtype, e);
        ++w;
    }
    for (RadTensor& x : t) std::free(x.data);
    return out;
}

void check_twin_against_kernel(const r97::OpRecord& rec, const Launched& l,
                               const std::string& ctx) {
    std::string err;
    const r97::TwinFn twin = r97::twin_find(rec.op, rec.params, &err);
    if (!twin) ac::fail(err);
    std::vector<r97::AdValue> expect;
    twin(rec, l.reads, &expect);
    R97_CHECK(expect.size() == l.writes.size());
    for (size_t i = 0; i < expect.size(); ++i) {
        if (expect[i].value.size() != l.writes[i].value.size())
            ac::fail(ctx + " " + rec.op + ": write " + std::to_string(i) + " has " +
                     std::to_string(l.writes[i].value.size()) + " values, the twin " +
                     std::to_string(expect[i].value.size()));
        for (size_t e = 0; e < expect[i].value.size(); ++e) {
            if (!ac::close_enough(l.writes[i].value[e], expect[i].value[e], bf16_tol)) {
                char buf[400];
                std::snprintf(buf, sizeof buf,
                              "%s %s: write %zu element %zu: libref %.17g vs twin %.17g",
                              ctx.c_str(), rec.op.c_str(), i, e, l.writes[i].value[e],
                              expect[i].value[e]);
                ac::fail(buf);
            }
        }
    }
}

void run_kernel_chain(const std::vector<Step>& tape, std::map<std::string, r97::AdValue>* fwd,
                      const std::string& ctx) {
    for (const Step& st : tape) {
        const std::vector<r97::AdValue> in = reads_of(st, *fwd);
        const Launched l = launch_host(st.rec, in, write_specs(st.rec));
        check_twin_against_kernel(st.rec, l, ctx);
        size_t w = 0;
        for (const r97::Operand& o : st.rec.operands)
            if (o.written) (*fwd)[out_key(st, o.buffer)] = l.writes[w++];
    }
}

// ---------------------------------------------------------------- the oracle

std::filesystem::path oracle_dir() {
    return std::filesystem::path(__FILE__).parent_path().parent_path() / "fixtures" / "oracle";
}

std::vector<double> pack_valid(const r97::oracle::Tensor& t, const std::vector<int64_t>& lens) {
    R97_CHECK(t.shape.size() == 3);
    const int64_t width = t.shape[2];
    std::vector<double> out;
    for (size_t s = 0; s < lens.size(); ++s)
        for (int64_t b = 0; b < lens[s]; ++b)
            for (int64_t c = 0; c < width; ++c)
                out.push_back(t.data[(size_t)((s * t.shape[1] + b) * width + c)]);
    return out;
}

void check_packed(const std::vector<double>& got, const r97::oracle::Tensor& want,
                  const std::vector<int64_t>& lens, double tol, const std::string& what) {
    R97_CHECK(want.shape.size() == 3);
    const int64_t width = want.shape[2];
    size_t row = 0;
    for (size_t s = 0; s < lens.size(); ++s)
        for (int64_t b = 0; b < lens[s]; ++b, ++row)
            for (int64_t c = 0; c < width; ++c) {
                const double w = want.data[(size_t)((s * want.shape[1] + b) * width + c)];
                const double g = got[row * (size_t)width + (size_t)c];
                if (!ac::close_enough(g, w, tol)) {
                    char buf[400];
                    std::snprintf(buf, sizeof buf,
                                  "%s: sequence %zu token %lld channel %lld: %.17g vs %.17g",
                                  what.c_str(), s, (long long)b, (long long)c, g, w);
                    ac::fail(buf);
                }
            }
}

void run_oracle_case(const std::string& stem, const LayerShape& s, const LayerWeights& w,
                     const r97::oracle::detail::File& ref) {
    const r97::oracle::Tensor& input = ref.tensor("gdn_mixer." + stem + ".input");
    const r97::oracle::Tensor& output = ref.tensor("gdn_mixer." + stem + ".output");
    const r97::oracle::Tensor& cot_out = ref.tensor("gdn_mixer." + stem + ".cotangent");
    const r97::oracle::Tensor& cot_in = ref.tensor("gdn_mixer." + stem + ".input_cotangent");
    const r97::oracle::Tensor& lengths = ref.tensor("gdn_mixer." + stem + ".lengths");
    R97_CHECK(input.shape.size() == 3 && input.shape[2] == s.hidden);

    std::vector<int64_t> lens;
    for (double n : lengths.data) lens.push_back((int64_t)n);
    const int64_t seq = (int64_t)lens.size();
    int64_t t = 0;
    for (int64_t n : lens) t += n;
    const std::vector<int64_t> cu = cu_of(lens);

    const std::string ctx = "gdn_mixer." + stem;
    const std::vector<Step> tape = layer_tape(s, t, seq);

    std::map<std::string, r97::AdValue> fwd =
        layer_constants(s, w, t, seq, cu, pack_valid(input, lens));
    run_forward(tape, &fwd);
    check_packed(fwd.at("out").value, output, lens, 1e-8, ctx + " output");

    std::map<std::string, r97::AdValue> cot;
    cot["out"] = value_of("bf16", {t, s.hidden}, pack_valid(cot_out, lens));
    run_backward(tape, fwd, &cot);
    check_packed(cot.at("h").value, cot_in, lens, 1e-8, ctx + " input cotangent");
}

}  // namespace

R97_TEST(gdn_recurrence_matches_the_per_token_reference) {
    /* Shorter than, equal to and several times the chunk length, and one ragged batch
     * that holds all three. The small chunk keeps the state loops and the finite
     * differences cheap; the layer test below runs the fixture's 64. */
    std::vector<RecShape> shapes;
    {
        RecShape r;
        r.chunk = 5;
        r.lens = {3};
        shapes.push_back(r);
    }
    {
        RecShape r;
        r.chunk = 5;
        r.lens = {5};
        shapes.push_back(r);
        r.lens = {23};
        shapes.push_back(r);
    }
    {
        RecShape r;
        r.chunk = 5;
        r.lens = {3, 5, 23};
        shapes.push_back(r);
    }

    uint64_t seed = 0x6dd1e17a11ull;
    for (const RecShape& s : shapes) {
        const RecCase c = rec_case(s, seed += 0x9e3779b9ull);
        const ref::Geometry geom = rec_geometry(s);

        std::map<std::string, r97::AdValue> fwd = c.fwd;
        run_forward(rec_tape(s), &fwd);
        const ref::Forward rf = ref::forward(geom, c.in);
        check_close(fwd.at("o").value, rf.o, 1e-10, "per-token forward o");
        check_close(fwd.at("ht").value, rf.ht, 1e-10, "per-token forward ht");

        std::map<std::string, r97::AdValue> cot;
        cot["o"] = zeros_like(fwd.at("o"));
        cot["ht"] = zeros_like(fwd.at("ht"));
        for (double& x : cot["o"].value) x = ac::urand(&seed) * 2.0 - 1.0;
        for (double& x : cot["ht"].value) x = ac::urand(&seed) * 2.0 - 1.0;
        ref::Cotangent rc;
        rc.o = cot["o"].value;
        rc.ht = cot["ht"].value;
        run_backward(rec_tape(s), fwd, &cot);
        const ref::Gradients rg = ref::backward(geom, c.in, rf, rc);
        check_close(cot.at("q").value, rg.q, 1e-9, "composed adjoint dq");
        check_close(cot.at("k").value, rg.k, 1e-9, "composed adjoint dk");
        check_close(cot.at("v").value, rg.v, 1e-9, "composed adjoint dv");
        check_close(cot.at("g").value, rg.g, 1e-9, "composed adjoint dg");
        check_close(cot.at("beta").value, rg.beta, 1e-9, "composed adjoint dbeta");

        /* One random direction through the composed twin: the adjoint's directional
         * derivative against central differences of the same loss. */
        std::map<std::string, std::vector<double>> dir;
        for (const char* name : {"q", "k", "v", "g", "beta"}) {
            std::vector<double> d(fwd.at(name).value.size());
            for (double& x : d) x = ac::urand(&seed) * 2.0 - 1.0;
            dir[name] = std::move(d);
        }
        const double eps = 1e-6;
        const double numeric =
            (recurrence_loss(s, c.fwd, dir, eps, rc.o, rc.ht) -
             recurrence_loss(s, c.fwd, dir, -eps, rc.o, rc.ht)) /
            (2.0 * eps);
        double analytic = 0.0;
        for (const auto& d : dir) analytic += dot(cot.at(d.first).value, d.second);
        if (!ac::close_enough(numeric, analytic, fd_tol)) {
            char buf[256];
            std::snprintf(buf, sizeof buf,
                          "composed finite difference: numeric %.17g vs adjoint %.17g", numeric,
                          analytic);
            ac::fail(buf);
        }
    }
}

R97_TEST(gdn_layer_twins_match_libref_on_the_fixture) {
    const LayerShape s;
    const r97::Fixture& fx = r97::fixture();
    const LayerWeights w = read_weights(fx.checkpoint, s);

    /* The reference case's single sequence, cast down to the fixture's bf16 by the
     * kernels themselves: this is the shape the fixture's own op graph runs. */
    const r97::oracle::detail::File ref((oracle_dir() / "reference.safetensors").string());
    const r97::oracle::Tensor& input = ref.tensor("gdn_mixer.single.input");
    const std::vector<int64_t> lens = {input.shape[1]};
    const int64_t t = input.shape[1];
    const std::vector<int64_t> cu = cu_of(lens);

    std::vector<double> h;
    h.reserve((size_t)(t * s.hidden));
    for (double x : input.data) h.push_back(x);

    std::map<std::string, r97::AdValue> fwd = layer_constants(s, w, t, 1, cu, std::move(h));
    const std::vector<Step> tape = layer_tape(s, t, 1);
    run_kernel_chain(tape, &fwd, "layer on the bf16 fixture");
}

R97_TEST(gdn_layer_matches_the_pytorch_reference) {
    const LayerShape s;
    const r97::Fixture& fx = r97::fixture();
    const LayerWeights w = read_weights(fx.checkpoint, s);
    const r97::oracle::detail::File ref((oracle_dir() / "reference.safetensors").string());

    run_oracle_case("single", s, w, ref);
    run_oracle_case("batch", s, w, ref);
}
