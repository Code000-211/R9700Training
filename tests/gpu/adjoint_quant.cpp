// tests/gpu/adjoint_quant.cpp -- the device straight-through adjoint of the quant family against
// the host adjoints of the families quant and elementwise.
//
// The worked example docs/kernels/README.md points a new kernel at:
//   * on the fixture's shape (hidden 128, docs/FIXTURES.md) and on the real container's (hidden
//     5120, docs/DESIGN.md), for one sequence and for a batch of unequal lengths, the device
//     adjoint of had_quant_act_i8g and of the fused producers rmsnorm_had_quant_i8g,
//     gated_had_quant_i8g and gate_had_quant_i8g matches the harness's reference within its bf16
//     tolerance;
//   * the harness rejects a deliberately wrong kernel rather than accepting anything plausible;
//   * the scratch allocator reports its high-water mark, which the harness prints for a kernel.
//
// The reference is docs/adjoints/quant.md's: the composition of the host adjoints of the family's
// parts. A fused producer's record is the quantiser's (a had_quant_act_i8g record over the
// elementwise part's output) and the elementwise op's, so `reference()` runs those two host
// adjoints in the order the kernel undoes them, and the twin the harness builds the forward with
// is the same composition of the parts' twins. The fused producers have no host twin yet
// (docs/adjoints/quant.md: an op that turns up later belongs there, not here); this test registers
// the composition so the forward values, the device adjoint's recomputed intermediate and the
// reference all read one definition of it.
//
// Needs one R9700 (gfx1201, 32 GB) or similar; runs under the GPU test guard. No container is
// read: the real shapes are numbers from the design, so this test does not carry the `model` label.
#include "r97/adjoint.h"

#include "support/device_check.h"
#include "support/gpu_guard.h"
#include "support/test.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace r97;
using device_check::numel;

// -- the parts a fused producer is made of -----------------------------------

// The elementwise op the fused producer applies before the rotation. `none` is the plain quantiser
// of had_quant_act_i8g, which is the whole of that op.
enum class Part { none, rmsnorm, silu_mul, sigmoid_mul };

const char* part_op(Part p) {
    switch (p) {
        case Part::rmsnorm: return "rmsnorm";
        case Part::silu_mul: return "silu_mul";
        case Part::sigmoid_mul: return "act_mul";
        case Part::none: break;
    }
    return "";
}

// The params a part's host adjoint reads: rmsnorm's gain offset and its scale are the record's,
// and act_mul needs the name of its activation. The fused records carry them already, so a part's
// twin and adjoint are handed the fused record itself.
Operand read_operand(const char* buffer, int64_t rows, int64_t width) {
    return Operand{buffer, "bf16", {rows, width}, 0, rows, false};
}

int64_t int_param(const std::vector<OpParam>& params, const char* name) {
    for (const OpParam& p : params) {
        if (p.first != name) continue;
        if (const int64_t* i = std::get_if<int64_t>(&p.second)) return *i;
        if (const std::string* s = std::get_if<std::string>(&p.second)) return std::stoll(*s);
    }
    throw std::runtime_error(std::string("the record has no integer parameter '") + name + "'");
}

// The quantiser's own record: the fused producer's intermediate is the quantiser's activation, so
// the part is served by the family's existing host twin and host adjoint.
OpRecord quant_record(int64_t rows, int64_t n, int64_t group) {
    OpRecord q;
    q.op = "had_quant_act_i8g";
    q.params = {{"M", rows}, {"n", n}, {"group", group}, {"dtype", std::string("bf16")}};
    q.operands = {
        Operand{"x", "bf16", {rows, n}, 0, rows, false},
        Operand{"q", "i8", {rows, n}, 0, rows, true},
        Operand{"scale", "f32", {rows, n / group}, 0, rows, true},
    };
    return q;
}

// The forward of each fused producer, as the composition of the parts' twins: the elementwise part
// over the record's read operands, then the quantiser over its output.
void fused_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                std::vector<AdValue>* out, Part part) {
    const int64_t n = int_param(rec.params, "n");
    const int64_t group = int_param(rec.params, "group");

    std::vector<AdValue> y;
    if (part == Part::none) {
        y = {operands.at(0)};
    } else {
        TwinFn elem = twin_find(part_op(part), rec.params, nullptr);
        if (!elem) throw std::runtime_error(std::string("no twin for ") + part_op(part));
        elem(rec, operands, &y);
    }
    const int64_t rows = y.at(0).shape.at(0);
    const OpRecord q = quant_record(rows, n, group);
    TwinFn quant = twin_find(q.op, q.params, nullptr);
    if (!quant) throw std::runtime_error("no twin for had_quant_act_i8g");
    quant(q, y, out);
}

void rmsnorm_twin(const OpRecord& rec, const std::vector<AdValue>& op, std::vector<AdValue>* out) {
    fused_twin(rec, op, out, Part::rmsnorm);
}
void gated_twin(const OpRecord& rec, const std::vector<AdValue>& op, std::vector<AdValue>* out) {
    fused_twin(rec, op, out, Part::silu_mul);
}
void gate_twin(const OpRecord& rec, const std::vector<AdValue>& op, std::vector<AdValue>* out) {
    fused_twin(rec, op, out, Part::sigmoid_mul);
}

// -- the reference: the parts' host adjoints, in the order the kernel undoes them -------------

// One read operand's cotangent is the quantiser's output cotangent; the elementwise part's adjoint
// then folds it into the read operands' slots, which is where the device kernel must land too.
void reference(AdjointCall& call, Part part) {
    size_t reads = 0;
    for (const Operand& o : call.rec.operands)
        if (!o.written) ++reads;
    const int64_t n = int_param(call.rec.params, "n");
    const int64_t group = int_param(call.rec.params, "group");
    const AdValue& codes = call.fwd.at(reads);
    const AdValue& scale = call.fwd.at(reads + 1);

    std::vector<AdValue> y;  // the intermediate, when the fused part wrote one
    const AdValue* x = &call.fwd.at(0);
    if (part != Part::none) {
        TwinFn elem = twin_find(part_op(part), call.rec.params, nullptr);
        if (!elem) throw std::runtime_error(std::string("no twin for ") + part_op(part));
        elem(call.rec, call.fwd, &y);
        x = &y.at(0);
    }

    const OpRecord q = quant_record(codes.shape.at(0), n, group);
    std::vector<AdValue> qfwd = {*x, codes, scale};
    std::vector<AdValue> qcot = {call.cot.at(0), call.cot.at(1)};
    std::vector<AdValue> qdc;
    for (const Operand& o : q.operands) {
        AdValue v;
        v.dtype = o.dtype;
        v.shape = o.shape;
        v.value.assign((size_t)numel(o.shape), 0.0);
        qdc.push_back(std::move(v));
    }
    AdjointCall qcall{q, qfwd, qcot, &qdc, nullptr, ""};
    AdjointFn quant = adjoint_find(q.op, Side::host, q.params, nullptr);
    if (!quant) throw std::runtime_error("no host adjoint for had_quant_act_i8g");
    quant(qcall);

    if (part == Part::none) {
        AdValue& d = call.dc->at(0);
        for (size_t i = 0; i < d.value.size(); ++i) d.value[i] += qdc.at(0).value[i];
        return;
    }

    // The elementwise adjoint takes the parts' operands and the intermediate they wrote; it adds
    // its cotangents into the slots the parts' read operands hold, which the harness zeroed.
    std::vector<AdValue> efwd(call.fwd.begin(),
                              call.fwd.begin() + static_cast<std::ptrdiff_t>(reads));
    efwd.push_back(y.at(0));
    std::vector<AdValue> ecot = {qdc.at(0)};
    AdjointCall ecall{call.rec, efwd, ecot, call.dc, nullptr, ""};
    AdjointFn elem = adjoint_find(part_op(part), Side::host, call.rec.params, nullptr);
    if (!elem) throw std::runtime_error(std::string("no host adjoint for ") + part_op(part));
    elem(ecall);
}

// -- the cases --------------------------------------------------------------

struct Shape {
    int64_t rows;
    int64_t n;
    int64_t group;
    const char* label;
};

// The fixture's hidden and the container's (docs/FIXTURES.md, docs/DESIGN.md), one sequence and a
// batch of unequal lengths (300 + 700 + 1024 tokens). A row-wise op sees the batch as its token
// total, so the ragged case is its own row count.
const Shape kShapes[] = {
    {4, 128, 128, "4 x 128 tiny"},
    {4, 128, 64, "4 x 128 tiny, 2 groups"},
    {1024, 5120, 128, "1024 x 5120 one sequence"},
    {2024, 5120, 128, "2024 x 5120 batch 300+700+1024"},
    {4096, 5120, 128, "4096 x 5120 one sequence"},
};

device_check::Case quant_case(const Shape& s, Part part, uint64_t seed) {
    const int64_t rows = s.rows, n = s.n, group = s.group;
    const int64_t ngrp = n / group;

    device_check::Case c;
    c.rec.position = 0;
    c.rec.params = {{"M", rows}, {"n", n}, {"group", group}, {"dtype", std::string("bf16")}};

    if (part == Part::none) {
        c.rec.op = "had_quant_act_i8g";
        c.rec.operands = {read_operand("x", rows, n),
                          Operand{"q", "i8", {rows, n}, 0, rows, true},
                          Operand{"scale", "f32", {rows, ngrp}, 0, rows, true}};
        c.in = {device_check::random_value("bf16", {rows, n}, &seed)};
    } else if (part == Part::rmsnorm) {
        c.rec.op = "rmsnorm_had_quant_i8g";
        c.rec.params.push_back({"eps", std::string("1e-6")});
        c.rec.params.push_back({"wadd", std::string("0.0")});
        c.rec.operands = {read_operand("x", rows, n),
                          Operand{"w", "bf16", {n}, 0, 1, false},
                          Operand{"q", "i8", {rows, n}, 0, rows, true},
                          Operand{"scale", "f32", {rows, ngrp}, 0, rows, true}};
        c.in = {device_check::random_value("bf16", {rows, n}, &seed),
                device_check::random_value("bf16", {n}, &seed)};
    } else if (part == Part::silu_mul) {
        c.rec.op = "gated_had_quant_i8g";
        c.rec.operands = {read_operand("gate_up", rows, 2 * n),
                          Operand{"q", "i8", {rows, n}, 0, rows, true},
                          Operand{"scale", "f32", {rows, ngrp}, 0, rows, true}};
        c.in = {device_check::random_value("bf16", {rows, 2 * n}, &seed)};
    } else {
        c.rec.op = "gate_had_quant_i8g";
        c.rec.params.push_back({"act", std::string("sigmoid")});
        c.rec.operands = {read_operand("gate", rows, n), read_operand("x", rows, n),
                          Operand{"q", "i8", {rows, n}, 0, rows, true},
                          Operand{"scale", "f32", {rows, ngrp}, 0, rows, true}};
        c.in = {device_check::random_value("bf16", {rows, n}, &seed),
                device_check::random_value("bf16", {rows, n}, &seed)};
    }
    c.cot = {device_check::random_value("f32", {rows, n}, &seed),
             device_check::random_value("f32", {rows, ngrp}, &seed)};
    c.label = std::string(s.label);
    return c;
}

// Deliberately wrong: hands the cotangent back unrotated and unmasked, which the harness must
// catch. A device adjoint is an AdjointFn, so a test can hand one in place of the registered row.
void wrong_quant_adjoint(const AdjointCall& call) {
    AdValue& dx = call.dc->at(0);
    for (size_t i = 0; i < dx.value.size(); ++i) dx.value[i] += call.cot.at(0).value[i];
}

void run() {
    const auto options = device_check::guard_options(int64_t(1) << 30);
    const r97test::DeviceGuard guard(options);
    device_check::require_device(options.device);
    (void)device_check::load_plugin();

    const Part parts[] = {Part::none, Part::rmsnorm, Part::silu_mul, Part::sigmoid_mul};
    for (Part part : parts) {
        const std::string op = part == Part::none ? "had_quant_act_i8g"
                               : part == Part::rmsnorm ? "rmsnorm_had_quant_i8g"
                               : part == Part::silu_mul ? "gated_had_quant_i8g"
                                                        : "gate_had_quant_i8g";
        R97_CHECK(r97::op_rows(op).device_adjoint);
        // The RMS norm's gain is frozen (src/adjoint/host/elementwise.cpp registers it so), and the
        // fused producer inherits that: the walk asks for x's cotangent only.
        if (part == Part::rmsnorm) {
            const OpRecord tiny = quant_case(kShapes[0], part, 0).rec;
            R97_CHECK(r97::adjoint_frozen(op, Side::device, tiny.params) ==
                      std::vector<int>{1});
        }
    }

    // The row serves the activations the card issues, and refuses the ones it does not yet: a
    // gated record built for gelu has no device adjoint rather than a silently wrong one.
    {
        std::string err;
        const OpRecord gated = quant_case(kShapes[0], Part::silu_mul, 0).rec;
        std::vector<OpParam> gelu = gated.params;
        gelu.push_back({"act", std::string("gelu")});
        R97_CHECK(adjoint_find("gated_had_quant_i8g", Side::device, gelu, &err) == nullptr);
        R97_CHECK(!err.empty());
        R97_CHECK(adjoint_find("gated_had_quant_i8g", Side::device, gated.params, &err) != nullptr);
    }

    uint64_t seed = 0x9a7f0001ull;
    for (Part part : parts) {
        for (const Shape& s : kShapes) {
            const device_check::Case c = quant_case(s, part, seed++);
            device_check::HostRef ref = [part](AdjointCall& call) { reference(call, part); };
            const device_check::Report r = device_check::check(c, nullptr, ref);
            device_check::require(r);
            R97_CHECK(r.scratch_peak > 0);  // the kernel took its buffers from the allocator
            std::printf("%s %s: device == the parts' host adjoints, scratch peak %lld KiB\n",
                        c.rec.op.c_str(), c.label.c_str(), (long long)(r.scratch_peak / 1024));
            if (r97test::bench()) device_check::print_bench(c);
        }
    }

    // A kernel that skips the rotation and the mask must be rejected, in x's slot.
    {
        const device_check::Case c = quant_case(kShapes[0], Part::rmsnorm, 0xa11ceull);
        device_check::HostRef ref = [](AdjointCall& call) { reference(call, Part::rmsnorm); };
        const device_check::Report wrong = device_check::check(c, wrong_quant_adjoint, ref);
        R97_CHECK(!wrong.ok);
        R97_CHECK(wrong.first_slot == 0);
        std::printf("rejected the wrong kernel: %s\n", wrong.note.c_str());
    }
}

}  // namespace

R97_TEST(adjoint_quant_device) { run(); }

// The fused producers' twins: the parts' twins composed, registered here because the host family
// has no twin for them yet. They carry no predicate (a device test's records are the ones it
// builds), and they are what the harness and the reference build a case's forward with.
R97_REGISTER_TWIN("rmsnorm_had_quant_i8g", rmsnorm_twin)
R97_REGISTER_TWIN("gated_had_quant_i8g", gated_twin)
R97_REGISTER_TWIN("gate_had_quant_i8g", gate_twin)
