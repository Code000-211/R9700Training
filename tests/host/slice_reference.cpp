// tests/host/slice_reference.cpp -- the g1 slice runner: one block's recorded op slice, forward
// and backward, from the registry alone.
//
// The objective-free counterpart of src/reference_step.cpp (issue #55, criterion two). A slice is
// a filtered r97-program JSON file (the format `program_json` writes); the runner injects the
// boundary forward values and the boundary cotangents by buffer name, runs the registered twins
// forward in f64, then walks r97::backward_block seeded ONLY from the injected cotangents. No
// objective, no loss, no model, no container, no card: a block slice never writes logits, so
// nothing here builds a layout or asks an objective for seeds.
//
// What a comparison on this runner carries: it clears a block's OWN rows and rules out the join's
// boundary handling at that block (no R97_WALK_BLOCKS top-n limit on this side, no head seeds).
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
//   b) `two-consumer residual`: y = (x + a) + (x + b), seeded at y, the slice arriving through
//      parse_slice from a written r97-program file: x's cotangent must arrive by both consumers
//      and sum, dx = 2 dy.
#include "r97/backward.h"
#include "r97/model.h"
#include "support/test.h"

#include <nlohmann/json.hpp>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using json = nlohmann::json;

namespace {
namespace slice {

using r97::AdValue;

// ------------------------------------------------------------------ small shared arithmetic

int64_t numel(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 0;
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

AdValue make_value(std::string dtype, std::vector<int64_t> shape, std::vector<double> value) {
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
AdValue slice_rows(const AdValue& v, int64_t lo, int64_t hi) {
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

r97::OpParam param_from_json(const std::string& key, const json& v) {
    if (v.is_number_integer()) return {key, v.get<int64_t>()};
    if (v.is_string()) return {key, v.get<std::string>()};
    throw std::runtime_error("slice runner: bad parameter " + key + ": not an integer or a string");
}

ProgramSlice parse_slice(const json& j) {
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

ProgramSlice parse_slice(const std::string& path) {
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
AdValue operand_at(const r97::OpRecord& rec, const r97::Operand& o,
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
void put_write(const r97::Operand& o, const AdValue& produced,
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
void run_forward(const ProgramSlice& program, const SliceInput& in,
                 std::map<std::string, Buffer>* env, r97::StoredSnapshots* snaps) {
    for (const r97::OpRecord& rec : program.ops) {
        std::string error;
        const r97::TwinFn twin = r97::twin_find(rec.op, rec.params, &error);
        if (twin == nullptr)
            throw std::runtime_error("slice runner: no twin for op '" + rec.op + "'" +
                                     (error.empty() ? std::string() : " (" + error + ")"));

        std::vector<AdValue> snapshot(rec.operands.size());
        std::vector<AdValue> reads;  // the reads, in schema order, the twin's call
        std::vector<size_t> writes;  // the written operands' schema slots, in the writes' order
        for (size_t i = 0; i < rec.operands.size(); ++i) {
            const r97::Operand& o = rec.operands[i];
            if (!o.written) {
                reads.push_back(operand_at(rec, o, *env, in));
                snapshot[i] = reads.back();
                continue;
            }
            writes.push_back(i);
            if (r97::in_place_read(rec, i))
                snapshot[i] = operand_at(rec, o, *env, in);  // the value the write replaces
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
SliceRun run_slice(const ProgramSlice& program, const SliceInput& in) {
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

void norms_of(const std::vector<double>& v, double* l1, double* l2) {
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
// dumps is later work; this print is the artifact that comparison reads.
void print_report(const SliceRun& run, const SliceInput& in) {
    for (const std::string& name : in.out_buffers) {
        const auto it = run.env.find(name);
        if (it == run.env.end()) continue;
        double l1 = 0.0, l2 = 0.0;
        norms_of(it->second.value.value, &l1, &l2);
        std::printf("BUF %s l1=%.17g l2=%.17g\n", name.c_str(), l1, l2);
    }
    for (const r97::Param& p : run.grads) {
        double l1 = 0.0, l2 = 0.0;
        norms_of(std::vector<double>(p.value.begin(), p.value.end()), &l1, &l2);
        std::printf("LEAF %s l1=%.17g l2=%.17g\n", p.name.c_str(), l1, l2);
    }
    for (const std::string& name : in.leaf_buffers)
        for (const r97::InputCotangent& ic : run.walk.inputs)
            if (ic.buffer == name) {
                double l1 = 0.0, l2 = 0.0;
                norms_of(ic.value, &l1, &l2);
                std::printf("BUF %s l1=%.17g l2=%.17g\n", name.c_str(), l1, l2);
            }
}

}  // namespace slice
}  // namespace

// ------------------------------------------------------------------ the cases

namespace {

using r97::OpRecord;
using r97::Operand;
using r97::Seed;

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
    //   dx += s * dyw (the mul's read of `x`)          ds += y_in * dyw (the mul's read of
    //   `s`, whose cotangent is the pre-write value's, elementwise.cpp:185-206)
    // and the passthrough (FIX 2's setting) carries the write's cotangent, unchanged, to the
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

}  // namespace
