// src/slice_reference.h -- the g1 slice runner: one block's recorded op slice, forward
// and backward, from the registry alone. Header-only so both the acceptance test and the
// `r97train` subcommand drive the identical code; everything stays in `namespace slice`, and
// every function is `inline` (a header shared across translation units may not emit external
// definitions).
//
// The objective-free counterpart of src/reference_step.cpp (issue #55, criterion two). A slice is
// a filtered r97-program JSON file. The writer emits the in-place mark as `inout`
// (tools/ops.cpp, the tape's own key at src/tape.cpp:170), and the runner reads `inout` first
// then `in_place` (a hand-written record convenience) — both forms parse; recorded slices keep
// the mark end to end since the writer fix. The runner injects the
// boundary forward values and the boundary cotangents by buffer name, runs the registered twins
// forward in f64, then walks r97::backward_block seeded ONLY from the injected cotangents. No
// objective, no loss, no model, no container, no card: a block slice never writes logits, so
// nothing here builds a layout or asks an objective for seeds.
//
// What a comparison on this runner carries: it clears a block's OWN rows and rules out the join's
// boundary handling at that block (no head seeds, no shared boundary_cotangents join; this runner
// is an independent replay, so R97_WALK_BLOCKS' top-n limit does not apply -- nothing truncates
// this side).
// It does NOT rule out a defect inside backward_block itself, which is the same core the device
// walk drives -- only the join it is fed from is different here.
//
// Read resolution is this runner's own, not ReferenceStep's: (1) the live buffer by name -- the
// injected boundary value until a slice op writes it, the production after -- (2) a constant the
// caller declared by name (a frozen weight IS a constant here, and never a name-fallthrough to a
// container weight), (3) nothing: a read that names a buffer no injection names, no constant
// declares and no slice op wrote throws, naming the buffer. An activation can never silently
// resolve to a weight of the same name.
//
// The two acceptance cases are synthetic, deterministic and analytic:
//
//   a) `linear + elementwise scale + adapter pair`: y = s*(x @ W^T) + B(A x), seeded at y. Closed
//      forms: dx = s*(dy @ W) + A^T(B^T dy), dA = (B^T dy)^T x (f32, the lora row's own
//      accumulation), dB = t dy^T at t = A x; dW goes to nobody (the trunk linear is frozen).
//   b) `two-consumer residual + in-place mul`: h = x + a, g = x + b, y = h + g, then the mul
//      overwrites y in place with x * s (its record marks "inout": true, parsed via the
//      tape's key). Seeded at y through parse_slice from a written r97-program file. Closed
//      forms: dx = (2 + s) dyw (the add's two consumers plus the mul's read of x), da = db =
//      dyw, ds = x dyw. The passthrough (in_place_passthrough=true) is what carries the mul's
//      write cotangent back to the add's y version -- with it off, case (b) fails on dx.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "r97/backward.h"
#include "r97/model.h"

#include "adjoint_schema.h"  // prepare_adjoint_record: the schema-renaming pass the gemm rows need

using json = nlohmann::json;

namespace r97 {
namespace slice {

using r97::AdValue;

// (the runner was lifted verbatim from tests/host/slice_reference.cpp's block; see the header
// top for the reviewed scope claim. The test aliases `namespace slice = r97::slice;`.)

// ------------------------------------------------------------------ small shared arithmetic

inline int64_t numel(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 0;
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

inline AdValue make_value(std::string dtype, std::vector<int64_t> shape, std::vector<double> value) {
    if (int64_t(value.size()) != numel(shape))
        throw std::runtime_error("slice runner: value does not fill its shape");
    AdValue v;
    v.dtype = std::move(dtype);
    v.shape = std::move(shape);
    v.value = std::move(value);
    return v;
}

// The rows [lo, hi) of a value, in that shape. Both ends are in range on entry (the read's range
// is checked against the buffer's own rows before this runs); [0, shape[0]) keeps the value whole.
inline AdValue slice_rows(const AdValue& v, int64_t lo, int64_t hi) {
    if (lo == 0 && hi == v.shape[0]) return v;
    const int64_t cols = numel(v.shape) / v.shape[0];
    std::vector<int64_t> shape = v.shape;
    shape[0] = hi - lo;
    return make_value(v.dtype, std::move(shape),
                      std::vector<double>(v.value.begin() + static_cast<std::ptrdiff_t>(lo * cols),
                                          v.value.begin() + static_cast<std::ptrdiff_t>(hi * cols)));
}

// ------------------------------------------------------------------ the slice, from a program file

// An op slice: a filtered `r97-program` file, parsed by hand here because `parse_program` is
// internal to src/reference_step.cpp (its anonymous namespace keeps it from any other unit).
// version, batch, ops{op, block, position, n, params, operands{buffer,dtype,shape,row_lo,rows,
// written,inout}. The in-place mark is the tape's own serialization's `inout`
// (src/tape.cpp:170, op.in_place = o.inout != 0); `program_json` itself writes NO in-place
// field (src/reference_step.cpp:168-173), so a slice recorded through `r97train ops --record`
// silently drops every in-place marker unless that writer is extended -- the real block slice
// must come from the tape's own recording path. The parser reads `inout` first and `in_place`
// second (this file's hand-written records carry the latter); neither present is a plain write.
struct ProgramSlice {
    std::vector<std::vector<int32_t>> batch;  // untouched: this runner builds no layout from ids
    std::vector<r97::OpRecord> ops;
};

inline r97::OpParam param_from_json(const std::string& key, const json& v) {
    if (v.is_number_integer()) return {key, v.get<int64_t>()};
    if (v.is_string()) return {key, v.get<std::string>()};
    throw std::runtime_error("slice runner: bad parameter " + key + ": not an integer or a string");
}

inline ProgramSlice parse_slice(const json& j) {
    if (!j.is_object() || j.value("format", std::string()) != "r97-program")
        throw std::runtime_error("slice runner: not an r97-program");
    if (j.value("version", 0) != 1)
        throw std::runtime_error("slice runner: unsupported program version");
    ProgramSlice p;
    if (j.contains("batch") && j["batch"].is_array())
        for (const json& seq : j["batch"]) {
            std::vector<int32_t> ids;
            for (const json& v : seq)
                if (v.is_number_integer()) ids.push_back(v.get<int32_t>());
            p.batch.push_back(std::move(ids));
        }
    if (!j.contains("ops") || !j["ops"].is_array())
        throw std::runtime_error("slice runner: program has no ops");
    for (const json& o : j["ops"]) {
        r97::OpRecord rec;
        rec.op = o.value("op", std::string());
        rec.block = o.value("block", int64_t(-1));
        rec.position = o.value("position", int64_t(0));
        rec.n = o.value("n", int64_t(0));
        if (o.contains("params") && o["params"].is_object())
            for (auto it = o["params"].begin(); it != o["params"].end(); ++it)
                rec.params.push_back(param_from_json(it.key(), it.value()));
        if (o.contains("operands") && o["operands"].is_array())
            for (const json& x : o["operands"]) {
                r97::Operand op;
                op.buffer = x.value("buffer", std::string());
                op.dtype = x.value("dtype", std::string());
                if (x.contains("shape") && x["shape"].is_array())
                    for (const json& d : x["shape"])
                        if (d.is_number_integer()) op.shape.push_back(d.get<int64_t>());
                op.row_lo = x.value("row_lo", int64_t(0));
                op.rows = x.value("rows", int64_t(0));
                op.written = x.value("written", false);
                op.in_place = x.value("inout", false) || x.value("in_place", false);
                // The view a read names over its buffer: the walk's snapshot holds the view's
                // own shape and values (join_common.h:264-295), and without these fields the
                // slice replays the declared buffer instead -- two views of one buffer (the
                // gated-delta layer's A|B halves) collapse into the same operand.
                if (x.contains("view_shape") && x["view_shape"].is_array())
                    for (const json& d : x["view_shape"])
                        if (d.is_number_integer()) op.view_shape.push_back(d.get<int64_t>());
                if (x.contains("view_stride") && x["view_stride"].is_array())
                    for (const json& d : x["view_stride"])
                        if (d.is_number_integer()) op.view_stride.push_back(d.get<int64_t>());
                op.view_off = x.value("view_off", int64_t(0));
                rec.operands.push_back(std::move(op));
            }
        if (rec.op.empty()) throw std::runtime_error("slice runner: an op with no name");
        p.ops.push_back(std::move(rec));
    }
    if (p.ops.empty()) throw std::runtime_error("slice runner: the slice holds no ops");
    // The snapshot store keys by the record's position: one per op, in append order.
    for (size_t i = 0; i < p.ops.size(); ++i) p.ops[i].position = static_cast<int64_t>(i);
    return p;
}

inline ProgramSlice parse_slice(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("slice runner: cannot open slice " + path);
    json j;
    in >> j;
    return parse_slice(j);
}

// ------------------------------------------------------------------ the environment

// One named buffer the slice's ops share, in its declared full shape: the same three facts an
// AdValue holds (r97/adjoint.h: dtype, full logical shape, numel(shape) values, row-major).
struct Buffer {
    AdValue value;
};

// The injection. No name is ever a weight here: a weight operand is a constant the caller
// declared, so a name collision between a buffer a slice op reads and a container weight cannot
// serve this read (the mirror of resolve_read's fallthrough at src/reference_step.cpp:978,
// refused instead).
struct SliceInput {
    std::map<std::string, AdValue> boundary;  // injected forward values, by buffer name
    std::map<std::string, AdValue> constants; // declared frozen weights / template values, by name
    std::vector<r97::Seed> seeds;              // injected boundary cotangents (row ranges)
    std::string prefix;                        // leaf names carry this prefix
    std::vector<std::string> out_buffers;      // buffers whose output value to print
    std::vector<std::string> leaf_buffers;     // buffers whose input cotangent to print
};

// ------------------------------------------------------------------ the forward

// The forward value of one operand of one op, in the environment as it stands when the op runs.
// The int8 fused producers: their in-place operand is the residual accumulator the tape
// snapshots after the op (quant.cpp:826-832 names the family).
inline bool fused_producer_sum(const std::string& op) {
    return op == "rmsnorm_had_quant_i8g" || op == "gated_had_quant_i8g" ||
           op == "gate_had_quant_i8g";
}

// The kernel narrows the accumulated sum to the residual's dtype before the norm reads it
// (libref/ref_quant.cpp's src(i)); a bf16 store rounds to nearest even.
inline void narrow_bf16(std::vector<double>* v) {
    for (double& d : *v) {
        float f = static_cast<float>(d);
        uint32_t u = 0;
        std::memcpy(&u, &f, sizeof u);
        u += 0x7fffu + ((u >> 16) & 1u);
        u &= 0xffff0000u;
        std::memcpy(&f, &u, sizeof f);
        d = static_cast<double>(f);
    }
}

// A raw position table the tape carries as a pointer, rebuilt the way the trainer's own
// reference step names and fills them (src/reference_step.cpp:288-302 and :438-486): the
// operand's schema position decides the table, and the fills are the batch's structure --
// positions 0..M-1, cu_seqlens [0, M] for one sequence, a sequence's state slot, the zero
// [n_seq] gdn_conv_prep passes, the [[s]] arange chunk_scan passes. The lora records carry a
// different table at position 2: the pool's row map, which a one-slot pool makes all zeros.
inline AdValue raw_table(const r97::OpRecord& rec, const r97::Operand& o) {
    int64_t M = 0;
    int64_t pool = -1;
    for (const auto& kv : rec.params) {
        const int64_t* p = std::get_if<int64_t>(&kv.second);
        if (p == nullptr) continue;
        if (kv.first == "M") M = *p;
        if (kv.first == "S") pool = *p;
    }
    // The prefill length: the record's own activation rows, not its `M` -- the params are the
    // engine's and a padded chat run leaves them at the padded count while the operands hold
    // the batch's rows. The first non-raw operand with a row axis is the activation.
    int64_t seq_rows = M;
    for (const Operand& op : rec.operands) {
        if (op.buffer.rfind("raw:", 0) == 0) continue;
        if (!op.shape.empty() && op.shape[0] > 0) { seq_rows = op.shape[0]; break; }
    }
    const int64_t n = o.shape.empty() ? 0 : numel(o.shape);
    AdValue v;
    v.dtype = o.dtype.empty() ? std::string("i32") : o.dtype;
    v.shape = o.shape;
    v.value.assign(static_cast<size_t>(n), 0.0);
    auto refuse = [&](const std::string& why) {
        throw std::runtime_error("slice runner: op '" + rec.op + "' at position " +
                                 std::to_string(rec.position) + " reads raw table '" +
                                 o.buffer + "' (" + why + "), which this runner cannot derive");
    };
    // The lora pool's row map: one slot maps every row to it.
    if (pool >= 0) {
        if (pool == 1 && n > 0) return v;
        refuse("pool S=" + std::to_string(pool));
    }
    std::string key = o.buffer.substr(4);
    if (const auto at = key.find('@'); at != std::string::npos) key.resize(at);
    // attn_paged carries q_len, not M, so the row count comes from seq_rows (the activation's
    // own rows) and M is only the fallback.
    const bool seq_ok = seq_rows > 0;
    if (key == "1" && rec.op != "gather_rows") {  // positions
        if (!seq_ok || n != seq_rows) refuse("positions of a batch this runner does not hold");
        for (int64_t i = 0; i < n; ++i) v.value[(size_t)i] = double(i);
        return v;
    }
    if (key == "2") {  // the attention pool's block table or kv_store's per-token slot
        // kv_store reads [tokens] slots (the identity mapping the block table below gives);
        // attn_paged reads [n_seq, max_blocks] of block ids.
        if (o.shape.size() == 1 && o.shape[0] == seq_rows) {
            for (int64_t i = 0; i < o.shape[0]; ++i) v.value[(size_t)i] = double(i);
            return v;
        }
        if (o.shape.size() != 2 || o.shape[0] != 1) refuse("a block table of a batch this runner does not hold");
        const int64_t blocks = (seq_rows + 15) / 16;  // the op's own block_size, a prefill from 0
        for (int64_t b = 0; b < o.shape[1]; ++b) v.value[(size_t)b] = b < blocks ? double(b) : -1.0;
        return v;
    }
    if (key == "3" && rec.op.rfind("attn_", 0) == 0) {  // seqused, [n_seq]
        if (!seq_ok || n != 1) refuse("seqused of a batch this runner does not hold");
        v.value[0] = double(seq_rows);
        return v;
    }
    if (key == "3" || key == "6" || key == "7") {  // cu_seqlens
        if (!seq_ok || n != 2) refuse("cu_seqlens of a batch this runner does not hold");
        v.value[0] = 0.0;
        v.value[1] = double(seq_rows);
        return v;
    }
    if (key == "9") {  // the state slot each sequence owns
        if (!seq_ok || n != 1) refuse("state index of a batch this runner does not hold");
        return v;
    }
    if (key == "10") {
        if (o.shape.size() == 2) {  // chunk_scan's [[0], [1], ...]
            if (!seq_ok || n != 1) refuse("arange of a batch this runner does not hold");
            return v;
        }
        return v;  // gdn_conv_prep's zero [n_seq]
    }
    refuse("unmapped position");
    return v;  // unreachable
}

// The view a read names over its buffer, gathered in the view's own row-major order -- the same
// gather the walk's snapshot runs (join_common.h:264-295). A buffer whose numel already IS the
// view's (the dump's constant for a narrowed read is the view's own packed value) is returned
// reshaped, not gathered again; a record that carries `view_shape` with no stride takes the
// buffer's row-major strides, which is what a narrowed last dimension means.
inline AdValue gather_view(const AdValue& whole, const r97::Operand& o) {
    int64_t n = 1;
    for (int64_t d : o.view_shape) n *= std::max<int64_t>(d, 0);
    AdValue v;
    v.dtype = whole.dtype;
    v.shape = o.view_shape;
    if (n <= 0) return v;
    if (numel(whole.shape) == n) {
        v.value = whole.value;
        return v;
    }
    v.value.assign(static_cast<size_t>(n), 0.0);
    if (whole.value.empty()) return v;
    std::vector<int64_t> stride = o.view_stride;
    if (stride.size() != o.view_shape.size()) {
        // The buffer's own row-major strides where the ranks agree: the narrowed dimension is
        // the buffer's row pitch, not the view's.
        const std::vector<int64_t>& from = whole.shape.size() == o.view_shape.size()
                                               ? whole.shape
                                               : o.view_shape;
        stride.assign(o.view_shape.size(), 1);
        for (size_t d = o.view_shape.size(); d-- > 1;)
            stride[d - 1] = stride[d] * std::max<int64_t>(from[d], 0);
    }
    const int64_t total = numel(whole.shape);
    int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const size_t rank = o.view_shape.size();
    for (int64_t e = 0; e < n; ++e) {
        int64_t off = o.view_off;
        for (size_t d = 0; d < rank; ++d) off += idx[d] * stride[d];
        if (off < 0 || off >= total) continue;  // the walk's own tolerance (join_common.h:285-288)
        v.value[static_cast<size_t>(e)] = whole.value[static_cast<size_t>(off)];
        for (size_t d = rank; d-- > 0;) {
            if (++idx[d] < o.view_shape[d]) break;
            idx[d] = 0;
        }
    }
    return v;
}

inline AdValue operand_at(const r97::OpRecord& rec, const r97::Operand& o,
                          const std::map<std::string, Buffer>& env, const SliceInput& in) {
    // The schema's absent slot (none:1, an empty shape or dtype "invalid"): one scalar, the
    // placeholder the walk accepts against the empty shape (src/reference_step.cpp:947-952).
    if (o.dtype == "invalid" || o.shape.empty()) {
        AdValue v;  // the placeholder the walk accepts against the empty shape, built directly:
        v.dtype = "invalid";  // make_value would reject one scalar against numel({})=0
        v.value.assign(1, 0.0);  // (mirror of src/reference_step.cpp:947-952)
        return v;
    }

    // (1) the live buffer: the injected boundary value, or a production a slice op wrote.
    const auto found = env.find(o.buffer);
    if (found != env.end()) {
        const AdValue& whole = found->second.value;
        // The view wins over the row range, as it does in the walk's snapshot: a narrowed read
        // is packed over its own shape and the declared buffer's rows are not that shape.
        if (!o.view_shape.empty()) {
            AdValue v = gather_view(whole, o);
            v.dtype = o.dtype.empty() ? v.dtype : o.dtype;
            return v;
        }
        if (o.rows > 0) {
            if (o.row_lo < 0 || o.row_lo + o.rows > whole.shape[0])
                throw std::runtime_error("slice runner: op '" + rec.op + "' at position " +
                                         std::to_string(rec.position) + " reads rows [" +
                                         std::to_string(o.row_lo) + ", " +
                                         std::to_string(o.row_lo + o.rows) + ") of '" + o.buffer +
                                         "', which holds " + std::to_string(whole.shape[0]) +
                                         " rows");
            AdValue v = slice_rows(whole, o.row_lo, o.row_lo + o.rows);
            v.dtype = o.dtype.empty() ? v.dtype : o.dtype;
            return v;
        }
        AdValue v = whole;
        if (o.shape != whole.shape) {
            if (numel(o.shape) != numel(whole.shape))
                throw std::runtime_error(
                    "slice runner: op '" + rec.op + "' at position " +
                    std::to_string(rec.position) + " reads '" + o.buffer + "' with " +
                    std::to_string(numel(o.shape)) + " values, the buffer holds " +
                    std::to_string(numel(whole.shape)));
            v.shape = o.shape;  // a reshape of the same storage (a fused op's whole buffer)
        }
        if (!o.dtype.empty()) v.dtype = o.dtype;
        return v;
    }

    // (2) the constants the caller declared: the frozen weights and template values, by name.
    const auto constant = in.constants.find(o.buffer);
    if (constant != in.constants.end()) {
        const AdValue& whole = constant->second;
        // A valueless declaration: the operand's value lives only inside the engine's joined
        // form (a plugin-private quantisation scale the stored-weight decode folds in), and no
        // f64 view exists. The twin that reads this record does not consume the slot on the
        // stored path; naming it valueless keeps the refusal honest instead of fabricating.
        // Its shape is the record's -- the walk's own plugin-private snapshot carries the shape
        // and no value (raw/enc only), and the backward core checks shapes, not contents.
        if (whole.value.empty()) {
            AdValue v = whole;
            if (v.shape.empty()) v.shape = o.shape;
            if (!o.dtype.empty()) v.dtype = o.dtype;
            return v;
        }
        if (!o.view_shape.empty()) {
            AdValue v = gather_view(whole, o);
            v.dtype = o.dtype.empty() ? v.dtype : o.dtype;
            return v;
        }
        if (o.rows > 0) {
            if (o.row_lo < 0 || o.row_lo + o.rows > whole.shape[0])
                throw std::runtime_error("slice runner: op '" + rec.op + "' at position " +
                                         std::to_string(rec.position) + " reads rows [" +
                                         std::to_string(o.row_lo) + ", " +
                                         std::to_string(o.row_lo + o.rows) + ") of constant '" +
                                         o.buffer + "', which holds " +
                                         std::to_string(whole.shape[0]) + " rows");
            AdValue v = slice_rows(whole, o.row_lo, o.row_lo + o.rows);
            v.dtype = o.dtype.empty() ? v.dtype : o.dtype;
            return v;
        }
        AdValue v = whole;
        if (o.shape != whole.shape) {
            if (numel(o.shape) != numel(whole.shape))
                throw std::runtime_error(
                    "slice runner: op '" + rec.op + "' at position " +
                    std::to_string(rec.position) + " reads constant '" + o.buffer + "' with " +
                    std::to_string(numel(o.shape)) + " values, it holds " +
                    std::to_string(numel(whole.shape)));
            v.shape = o.shape;
        }
        if (!o.dtype.empty()) v.dtype = o.dtype;
        return v;
    }

    // (2b) a raw position table: the tape carries it as a pointer, never a value. The one this
    // slice reads is the lora pool's row map, and the record declares the pool's size (params
    // "S"). A one-slot pool maps every row to slot 0 -- the map a full batch has. A larger pool
    // would need the device's own table, which this runner does not have, so it refuses instead
    // of inventing one.
    if (o.buffer.rfind("raw:", 0) == 0) return raw_table(rec, o);

    // (3) the refusal: the read names a buffer no injection names, no constant declares and no
    // slice op wrote. Nothing falls through to a weight by name here.
    throw std::runtime_error("slice runner: op '" + rec.op + "' at position " +
                             std::to_string(rec.position) + " reads '" + o.buffer +
                             "', which no slice op wrote and no injection or constant names");
}

// Puts one op's written output back into the environment, at the rows the record names: a whole-
// buffer write replaces the value; a row-ranged write lands in the declared buffer's rows and
// keeps the rest (the reference step's apply_write, src/reference_step.cpp:1004-1030).
inline void put_write(const r97::Operand& o, const AdValue& produced,
                      std::map<std::string, Buffer>* env) {
    AdValue whole;
    const auto existing = env->find(o.buffer);
    if (existing != env->end() && !existing->second.value.shape.empty()) {
        whole = existing->second.value;
    } else {
        // A first write: the buffer's declared shape, zeroed, then the rows land in it.
        std::vector<int64_t> shape = o.shape;
        if (!shape.empty() && o.rows > 0)
            shape[0] = std::max<int64_t>(shape[0], o.row_lo + o.rows);
        whole = make_value(o.dtype, shape, std::vector<double>(static_cast<size_t>(numel(shape)), 0.0));
    }
    // A write that covers the record's whole declared buffer defines it: replace it outright.
    // A stale entry under the same name (the dump's activation leak, or a value the walk left
    // behind) must not veto the twin's production, and the record's own shape is authoritative
    // -- the pre-write value, when the write is in place, is already in the snapshot.
    std::vector<int64_t> want = o.shape;
    if (!want.empty() && o.rows > 0) want[0] = std::max<int64_t>(want[0], o.row_lo + o.rows);
    if (!produced.shape.empty() && produced.shape == want) {
        (*env)[o.buffer] = Buffer{produced};
        return;
    }
    const int64_t whole_rows = whole.shape.empty() ? 0 : whole.shape[0];
    const int64_t whole_cols = whole_rows > 0 ? numel(whole.shape) / whole_rows : 0;
    const int64_t produced_rows = produced.shape.empty() ? 0 : produced.shape[0];
    const int64_t produced_cols =
        produced_rows > 0 ? numel(produced.shape) / produced_rows : 0;
    if (o.rows > 0 && whole_cols > 0 && produced_cols == whole_cols &&
        o.row_lo + produced_rows <= whole_rows) {
        for (int64_t r = 0; r < produced_rows; ++r)
            for (int64_t k = 0; k < whole_cols; ++k)
                whole.value[static_cast<size_t>((o.row_lo + r) * whole_cols + k)] =
                    produced.value[static_cast<size_t>(r * produced_cols + k)];
    } else {
        if (numel(produced.shape) != numel(whole.shape))
            throw std::runtime_error("slice runner: write to '" + o.buffer + "' produces " +
                                     std::to_string(numel(produced.shape)) +
                                     " values, the buffer holds " +
                                     std::to_string(numel(whole.shape)));
        whole.value = produced.value;  // the record's own shape keeps the buffer's identity
    }
    (*env)[o.buffer] = Buffer{std::move(whole)};
}

// Runs the registered twins over the slice, one snapshot per (position, operand) in the walk's
// own contract (src/reference_step.cpp:886-927): a read holds the value the buffer held as the
// op ran; an ordinary written operand holds the twin's production, post-write; an in-place
// written operand holds the value the write replaced (r97::in_place_read,
// include/r97/backward.h:42 -- rope's operand 0 and a kv: state update a record may carry).
inline void run_forward(const ProgramSlice& program, const SliceInput& in,
                        std::map<std::string, Buffer>* env, r97::StoredSnapshots* snaps) {
    for (const r97::OpRecord& rec : program.ops) {
        std::string error;
        const r97::TwinFn twin = r97::twin_find(rec.op, rec.params, &error);
        if (twin == nullptr)
            throw std::runtime_error("slice runner: no twin for op '" + rec.op + "'" +
                                     (error.empty() ? std::string() : " (" + error + ")"));

        // The twin's call carries EVERY operand slot in schema order -- the convention
        // src/reference_step.cpp:886-927 and every registered twin are written against
        // (the fused-norm family indexes op[0]=x, op[1]=residual, op[2]=weight by schema
        // position, and an in-place write IS an op's input whose prior value is the read).
        // A slot that never existed shrinks for nobody: written-but-not-in-place slots
        // stay in their schema positions as the empty placeholder the twins skip.
        std::vector<AdValue> snapshot(rec.operands.size());
        std::vector<AdValue> reads(rec.operands.size());
        std::vector<size_t> writes;  // the written operands' schema slots, in the writes' order
        // The walk's cold rule for a carried state no earlier slice op wrote
        // (src/step_device.cpp:1441-1443, :1582-1588): it enters zero, and the dump does not
        // override it. A record that reads and writes the same `kv:` buffer
        // (`gdn_conv_prep`'s rolling conv state, `gdn_chunk_scan`'s chunk state) is captured
        // *after* its own write, so the dump's constant is the OUTGOING state; feeding it back
        // as the prior is the wrong direction, while the walk's adjoint differentiates that
        // read at zero. The comparison is slice-vs-walk, so the slice takes the walk's value.
        const auto cold_kv = [&](const r97::Operand& o) {
            return o.buffer.rfind("kv:", 0) == 0 && env->find(o.buffer) == env->end();
        };
        const auto cold_value = [](const r97::Operand& o) {
            AdValue cold;
            cold.dtype = o.dtype.empty() ? std::string("bf16") : o.dtype;
            cold.shape = o.shape;
            cold.value.assign(static_cast<size_t>(numel(o.shape)), 0.0);
            return cold;
        };
        for (size_t i = 0; i < rec.operands.size(); ++i) {
            const r97::Operand& o = rec.operands[i];
            if (!o.written) {
                reads[i] = cold_kv(o) ? cold_value(o) : operand_at(rec, o, *env, in);
                snapshot[i] = reads[i];
                continue;
            }
            writes.push_back(i);
            // An in-place write reads its own prior value. The fused norm's residual is the one
            // case where the prior is not the operand the kernel consumed: the sum is formed
            // below (this file's residual-sum block), and the report's withdrawn trial is why
            // the sum is built there and not by overwriting this read.
            if (r97::in_place_read(rec, i)) {
                reads[i] = cold_kv(o) ? cold_value(o)
                                      : operand_at(rec, o, *env, in);  // the value the write replaces
                snapshot[i] = reads[i];
            }
        }
        // The fused norm's residual is the norm's own input, and the tape snapshots it AFTER
        // the op: operand 1, written in place, of operand 0's shape (the device's own test,
        // src/adjoint/host/quant.cpp:863-869) holds the post-update sum x + residual_old
        // (quant.cpp:859-862). The slice's environment holds the PRE-write residual, so the sum
        // is formed here, bf16-narrowed as the kernel's store rounds it. The snapshot takes the
        // same value, because the walk's own snapshot of that read is the live post-op pull.
        // Checked against the device's own codes: with the sum and the weight's `wadd`, a
        // reconstruction of op 0 matches 91.5% of the dumped `h.q8` codes exactly and its
        // `h.s8` to 2e-3 -- with the residual alone, 0.9% and 95%.
        if (fused_producer_sum(rec.op) && rec.operands.size() > 1) {
            const r97::Operand& res = rec.operands[1];
            if (res.written && res.in_place && res.shape == rec.operands[0].shape &&
                !reads[0].value.empty() && reads[0].value.size() == reads[1].value.size()) {
                std::vector<double> sum = reads[0].value;
                for (size_t k = 0; k < sum.size(); ++k) sum[k] += reads[1].value[k];
                narrow_bf16(&sum);
                reads[1].value = sum;
                snapshot[1].value = std::move(sum);
            }
        }

        // The empty un-filled slots: the walk's own convention for an absent value
        // (r97/adjoint.h's AdValue with an empty value; a twin that reads one declines).
        for (size_t i = 0; i < rec.operands.size(); ++i) {
            if (reads[i].value.empty() && reads[i].dtype.empty()) reads[i].dtype = "invalid";
        }

        // The twin finds its operands by schema name (a/b/a_scale/b_scale) on the linear ops;
        // the tape's own buffer names never carry them. Same pass the joins hand their rows
        // (prepare_adjoint_record); a twin without it reads operand_pos("b") = -1 and returns
        // no outputs.
        OpRecord trec;
        prepare_adjoint_record(rec, &trec);
        std::vector<AdValue> produced;
        twin(trec, reads, &produced);
        if (produced.size() != writes.size())
            throw std::runtime_error("slice runner: twin of '" + rec.op + "' produced " +
                                     std::to_string(produced.size()) + " outputs, the record names " +
                                     std::to_string(writes.size()) + " writes");

        size_t produced_i = 0;
        // The walk's cold rule is not read-only: `src/step_device.cpp:1590-1594` zeroes the
        // snapshot of a WRITTEN `kv:` operand this block has not written yet, with no in-place
        // test -- `gdn_chunk_scan`'s plain write of its chunk state (op 7's `kv:2` at pos 9) is
        // the block's instance, and the walk's adjoint differentiates that slot at zero while
        // this file's `put_write` below hands the twin's production to the environment (which a
        // later in-block read must see, exactly as the walk's store does). Evaluated here,
        // before any write lands: `put_write` is the only thing that grows the environment.
        std::vector<char> cold_write(rec.operands.size(), 0);
        for (size_t i = 0; i < rec.operands.size(); ++i)
            if (rec.operands[i].written) cold_write[i] = cold_kv(rec.operands[i]) ? 1 : 0;
        for (size_t i = 0; i < rec.operands.size(); ++i) {
            if (!rec.operands[i].written) continue;
            if (!r97::in_place_read(rec, i)) {
                AdValue snap = produced[produced_i];
                // The record's own shape keeps the buffer's declared identity (deep in a fused
                // slice a twin's output may carry a narrowed shape): a same-numel reshape.
                if (!rec.operands[i].shape.empty() && rec.operands[i].shape != snap.shape &&
                    numel(rec.operands[i].shape) == numel(snap.shape))
                    snap.shape = rec.operands[i].shape;
                // The walk zeroes the production in place, keeping its shape and dtype.
                if (cold_write[i]) snap.value.assign(snap.value.size(), 0.0);
                snapshot[i] = std::move(snap);
            }
            put_write(rec.operands[i], produced[produced_i], env);
            ++produced_i;
        }

        for (size_t i = 0; i < rec.operands.size(); ++i) snaps->set(rec.position, i, snapshot[i]);
    }
}

// ------------------------------------------------------------------ the run

struct SliceRun {
    r97::ParamSet grads;                    // the trainable leaves, under the prefix the input names
    r97::BackwardResult walk;               // the inputs' cotangents, backward_block's own answer
    std::map<std::string, Buffer> env;      // the buffers as the forward left them
    // The dump's own rule for a constant: a buffer is stored at its FIRST read
    // (src/step_device.cpp:1390-1415), with that read's view packed into the stored shape. The
    // comparison gathers the slice's buffer through the same view; without it a narrowed read's
    // device value and the slice's whole buffer are different tensors and the line is skipped.
    struct FirstRead {
        r97::Operand operand;
        int64_t position = -1;
        size_t slot = 0;
    };
    std::map<std::string, FirstRead> first_read;
    // The value that read consumed, straight from the snapshot store the adjoint walks: the
    // like-for-like side of the comparison for a buffer written more than once (the dump holds
    // its first read, the final environment its last write).
    std::map<std::string, r97::AdValue> first_read_value;
};

// Objective-free, container-free: the backward is seeded from the injected cotangents alone
// (SliceInput::seeds). No objective is built, no `obj.seeds` exists to seed from -- the second
// removed blocker (src/reference_step.cpp:572-579; and the scope note at the top of this file:
// the core walked is the same one the device walk drives, the join feeding it is not).
inline SliceRun run_slice(const ProgramSlice& program, const SliceInput& in) {
    SliceRun run;
    // The boundary forward values are the buffers the block's caller left in place: the
    // environment holds them before the first op, so an unwritten read of one is served.
    for (const auto& [name, value] : in.boundary) run.env[name] = Buffer{value};

    // One name per raw/absent operand: the observer names them by schema position, and two
    // ops give position 10 shapes of different ranks (gdn_conv_prep's [1], chunk_scan's [1,1]).
    // The backward core records one shape per buffer name, so each record's unnamed operands
    // take their own (src/join_common.h:530-540, the walks' own rule).
    ProgramSlice prog = program;
    for (r97::OpRecord& rec : prog.ops) rec = own_raws(rec);

    // The constants' first reads, under the dump's own rule (src/step_device.cpp:1390-1415):
    // reads in append order, the first one wins, and raw/absent slots are never dumped.
    for (const r97::OpRecord& rec : prog.ops)
        for (size_t i = 0; i < rec.operands.size(); ++i) {
            const r97::Operand& o = rec.operands[i];
            // Exactly the dump's rule (src/step_device.cpp:1398-1402): a written operand is
            // skipped unless it is a `kv:` in-place read. A non-kv in-place write -- op 3's
            // `gdn_in` -- is skipped, so the first read recorded for that buffer is op 5's
            // narrowed view, not the write's declared shape.
            if (o.written && !(o.buffer.rfind("kv:", 0) == 0 && r97::in_place_read(rec, i)))
                continue;
            if (o.buffer.rfind("raw:", 0) == 0 || o.buffer.rfind("none:", 0) == 0) continue;
            if (o.dtype == "invalid") continue;
            run.first_read.emplace(o.buffer, SliceRun::FirstRead{o, rec.position, i});
        }

    r97::StoredSnapshots snaps;
    run_forward(prog, in, &run.env, &snaps);
    for (const auto& [name, fr] : run.first_read)
        if (const r97::AdValue* v = snaps.at(fr.position, fr.slot))
            run.first_read_value[name] = *v;

    // The boundaries a seed may name: the buffers that exist at the slice's end, whole rows --
    // the row count each buffer's own shape declares and what the seeds' rows name. ReferenceStep
    // asks BoundaryNames::from(the objective's tape); this runner has no tape, so the buffers'
    // own shapes are the window (the first removed blocker, make_layout's logits demand at
    // src/reference_step.cpp:611-615, never arises: no layout is built at all).
    r97::BoundaryNames boundaries;
    for (const auto& [name, buffer] : run.env)
        boundaries.add(name, 0, buffer.value.shape.empty() ? 0 : buffer.value.shape[0]);

    r97::BackwardOptions options;
    options.side = r97::Side::host;
    options.prefix = in.prefix.c_str();
    options.grads = &run.grads;
    // The empty-slot passthrough of an in-place write whose adjoint leaves its read slot
    // empty: both real walks set it (src/step_host.cpp:122, src/step_device.cpp:1135), and
    // without it an op like lora_expand -- whose adjoint does not fill the in-place slot it
    // reads (src/adjoint/host/lora.cpp:460-468) -- produces no cotangent where the walks carry
    // the write's own through. A slice comparison must not diverge from their setting.
    options.in_place_passthrough = true;
    // The linear rows find their operands by schema name (a/b/a_scale/b_scale); the tape's
    // own buffer names never carry them. Both real walks set prepare_record
    // (src/step_host.cpp:151, src/step_device.cpp:1159); a slice walk without it sends the
    // gemm adjoints records whose operand_pos("b") is -1, and the adjoint's guard makes
    // that a SILENT no-op -- leaf lines that land with no trunk-gemm propagation behind
    // them. Same walk settings, same join: the comparison needs this set.
    options.prepare_record = prepare_adjoint_record;  // adjoint_schema.h, unguarded
    // One factor per record, not one per leaf: the engine pools a fused weight's modules into
    // one lora record, and a slot's factor without a name of its own lands under the bare
    // `<prefix>.lora_A` -- four records' grads summed positionally into one buffer. The walks
    // name it from the record's own weight (src/step_device.cpp:1145-1160); the slice has no
    // container module map, so the weight's own name is the qualifier: one leaf per weight and
    // factor, `model.layers.37.blk.37.ssm_inz.lora_A`.
    const std::string leaf_prefix = in.prefix;
    options.fused_slots = true;
    options.record_prefix = [leaf_prefix](const r97::OpRecord& rec) -> std::string {
        const std::string weight = record_weight(rec);
        if (weight.empty()) return leaf_prefix;
        return leaf_prefix + weight;
    };
    run.walk = r97::backward_block(prog.ops, snaps, boundaries, in.seeds, {}, options);
    return run;
}

// ------------------------------------------------------------------ the reports

inline void norms_of(const std::vector<double>& v, double* l1, double* l2) {
    double a = 0.0, b = 0.0;
    for (double x : v) {
        a += std::fabs(x);
        b += x * x;
    }
    *l1 = a;
    *l2 = std::sqrt(b);
}

// `BUF <buffer> l1=<value> l2=<value>` and `LEAF <name> l1=<value> l2=<value>`: greppable,
// unambiguous about the buffer each line describes. The comparison against the device walk's
// dumps is later work; this print is the artifact that comparison reads. It writes through a
// FILE* -- C stdio -- because the tool's `--out` is a file, and cout's streambuf swap does not
// capture printf.
// The run's buffers as JSON, for an arm that needs the values and not just their norms -- the
// block-local directional derivative (<g_in, J u> against <g_out, u>) is a difference of two
// forward runs, and a difference of L2 norms cannot stand in for it.
inline void print_values(const SliceRun& run, const SliceInput& in, FILE* out) {
    std::fprintf(out, "{");
    bool first = true;
    for (const std::string& name : in.out_buffers) {
        const auto it = run.env.find(name);
        if (it == run.env.end()) continue;
        const AdValue& v = it->second.value;
        std::fprintf(out, "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[", first ? "" : ",",
                     name.c_str(), v.dtype.c_str());
        first = false;
        for (size_t i = 0; i < v.shape.size(); ++i)
            std::fprintf(out, "%s%lld", i ? "," : "", (long long)v.shape[i]);
        std::fprintf(out, "],\"value\":[");
        for (size_t i = 0; i < v.value.size(); ++i)
            std::fprintf(out, "%s%.17g", i ? "," : "", v.value[i]);
        std::fprintf(out, "]}");
    }
    std::fprintf(out, "}");
}

inline void print_report(const SliceRun& run, const SliceInput& in, FILE* out = stdout) {
    for (const std::string& name : in.out_buffers) {
        const auto it = run.env.find(name);
        if (it == run.env.end()) continue;
        double l1 = 0.0, l2 = 0.0;
        norms_of(it->second.value.value, &l1, &l2);
        std::fprintf(out, "BUF %s l1=%.17g l2=%.17g\n", name.c_str(), l1, l2);
    }
    // The leaf lines carry the element the walk's own report prints (`grad=%.9g` is
    // g.value[0], tests/gpu/device_step.cpp:404-410) beside the norms, so the comparison is
    // like for like, and the set line is the walk's LEAFSET: the L2 over every leaf whose name
    // contains the prefix.
    // The slice's own forward against the device's: for every declared constant the run also
    // produced (a buffer the block wrote and the dump captured at a later read), the relative
    // L2 of the difference. A twin-computed forward that disagrees here makes the adjoint's
    // inputs differ from the walk's, and no adjoint comparison can be closer than that.
    int64_t diff_n = 0;
    std::map<std::string, const AdValue*> device;  // the dump's own values, by name
    for (const auto& [name, value] : in.boundary) device[name] = &value;
    for (const auto& [name, value] : in.constants) device[name] = &value;
    for (const auto& [name, value] : device) {
        const auto it = run.env.find(name);
        if (it == run.env.end()) continue;
        // The device's entry is what the buffer's first read consumed; a narrowed read packed
        // its view (join_common.h:264-295), so the slice's buffer is gathered the same way or
        // the two sides are different tensors -- `gdn_in`'s [512,10240] q|k|v view of a
        // [512,16384] projection, `gdn_ab`'s two [512,48] halves of one [512,96] decay buffer.
        AdValue gathered = it->second.value;
        if (const auto fr = run.first_read.find(name);
            fr != run.first_read.end() && !fr->second.operand.view_shape.empty())
            gathered = gather_view(gathered, fr->second.operand);
        const std::vector<double>& a = gathered.value;
        const std::vector<double>& b = value->value;
        if (a.empty() || a.size() != b.size()) continue;
        double dsq = 0.0, bsq = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double d = a[i] - b[i];
            dsq += d * d;
            bsq += b[i] * b[i];
        }
        ++diff_n;
        std::fprintf(out, "CONSTDIFF %s rel=%.9g n=%zu\n", name.c_str(),
                     bsq > 0.0 ? std::sqrt(dsq / bsq) : std::sqrt(dsq), a.size());
    }
    std::fprintf(out, "CONSTDIFF-COUNT n=%lld\n", static_cast<long long>(diff_n));
    // The same comparison against the value the FIRST READ consumed, taken from the snapshot
    // store the adjoint walks. For a buffer written once and read later the two agree; for a
    // buffer written again after its first read (`h.q8`/`h.s8` at op 13, `h` at every residual
    // add) this is the like-for-like side, and for a read the same record also writes
    // (`gdn_o` at op 8, `kv:` states) the dump's capture is the POST-op buffer while this is
    // the pre-op value -- the two lines together say which side is which.
    int64_t snap_n = 0;
    for (const auto& [name, value] : device) {
        const auto it = run.first_read_value.find(name);
        if (it == run.first_read_value.end()) continue;
        const std::vector<double>& a = it->second.value;
        const std::vector<double>& b = value->value;
        if (a.empty() || a.size() != b.size()) continue;
        double dsq = 0.0, bsq = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double d = a[i] - b[i];
            dsq += d * d;
            bsq += b[i] * b[i];
        }
        ++snap_n;
        std::fprintf(out, "SNAPDIFF %s rel=%.9g n=%zu\n", name.c_str(),
                     bsq > 0.0 ? std::sqrt(dsq / bsq) : std::sqrt(dsq), a.size());
    }
    std::fprintf(out, "SNAPDIFF-COUNT n=%lld\n", static_cast<long long>(snap_n));

    int64_t set_leaves = 0;
    double set_norm_sq = 0.0;
    for (const r97::Param& p : run.grads) {
        double l1 = 0.0, l2 = 0.0;
        norms_of(std::vector<double>(p.value.begin(), p.value.end()), &l1, &l2);
        const double first = p.value.empty() ? 0.0 : static_cast<double>(p.value[0]);
        std::fprintf(out, "LEAF %s l1=%.17g l2=%.17g first=%.17g\n", p.name.c_str(), l1, l2,
                     first);
        if (!p.value.empty() && p.name.find(in.prefix) != std::string::npos) {
            ++set_leaves;
            set_norm_sq += l2 * l2;
        }
    }
    std::fprintf(out, "LEAFSET %s leaves=%lld norm=%.9g\n", in.prefix.c_str(),
                 static_cast<long long>(set_leaves), std::sqrt(set_norm_sq));
    for (const std::string& name : in.leaf_buffers)
        for (const r97::InputCotangent& ic : run.walk.inputs)
            if (ic.buffer == name) {
                double l1 = 0.0, l2 = 0.0;
                norms_of(ic.value, &l1, &l2);
                std::fprintf(out, "BUF %s l1=%.17g l2=%.17g\n", name.c_str(), l1, l2);
            }
}

}  // namespace slice
}  // namespace r97
