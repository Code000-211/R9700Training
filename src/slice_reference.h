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
        for (size_t i = 0; i < rec.operands.size(); ++i) {
            const r97::Operand& o = rec.operands[i];
            if (!o.written) {
                reads[i] = operand_at(rec, o, *env, in);
                snapshot[i] = reads[i];
                continue;
            }
            writes.push_back(i);
            if (r97::in_place_read(rec, i)) {
                reads[i] = operand_at(rec, o, *env, in);  // the value the write replaces
                snapshot[i] = reads[i];
            }
        }
        // The empty un-filled slots: the walk's own convention for an absent value
        // (r97/adjoint.h's AdValue with an empty value; a twin that reads one declines).
        for (size_t i = 0; i < rec.operands.size(); ++i) {
            if (reads[i].value.empty() && reads[i].dtype.empty()) reads[i].dtype = "invalid";
        }

        std::vector<AdValue> produced;
        twin(rec, reads, &produced);
        if (produced.size() != writes.size())
            throw std::runtime_error("slice runner: twin of '" + rec.op + "' produced " +
                                     std::to_string(produced.size()) + " outputs, the record names " +
                                     std::to_string(writes.size()) + " writes");

        size_t produced_i = 0;
        for (size_t i = 0; i < rec.operands.size(); ++i) {
            if (!rec.operands[i].written) continue;
            if (!r97::in_place_read(rec, i)) {
                AdValue snap = produced[produced_i];
                // The record's own shape keeps the buffer's declared identity (deep in a fused
                // slice a twin's output may carry a narrowed shape): a same-numel reshape.
                if (!rec.operands[i].shape.empty() && rec.operands[i].shape != snap.shape &&
                    numel(rec.operands[i].shape) == numel(snap.shape))
                    snap.shape = rec.operands[i].shape;
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

    r97::StoredSnapshots snaps;
    run_forward(program, in, &run.env, &snaps);

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
    run.walk = r97::backward_block(program.ops, snaps, boundaries, in.seeds, {}, options);
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
inline void print_report(const SliceRun& run, const SliceInput& in, FILE* out = stdout) {
    for (const std::string& name : in.out_buffers) {
        const auto it = run.env.find(name);
        if (it == run.env.end()) continue;
        double l1 = 0.0, l2 = 0.0;
        norms_of(it->second.value.value, &l1, &l2);
        std::fprintf(out, "BUF %s l1=%.17g l2=%.17g\n", name.c_str(), l1, l2);
    }
    for (const r97::Param& p : run.grads) {
        double l1 = 0.0, l2 = 0.0;
        norms_of(std::vector<double>(p.value.begin(), p.value.end()), &l1, &l2);
        std::fprintf(out, "LEAF %s l1=%.17g l2=%.17g\n", p.name.c_str(), l1, l2);
    }
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
