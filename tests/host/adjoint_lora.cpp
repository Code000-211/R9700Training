// tests/host/adjoint_lora.cpp -- the adapter path (src/adjoint/host/lora.cpp) through both checks
// of tests/support/adjoint_check.h, then the two things the issue asks that the shared checks do
// not: the gradients of A and B in the ParamSet against central differences of the twin, and the
// pair a fresh adapter starts with, B = 0. docs/adjoints/lora.md carries every formula.
//
// Six shapes, one of them the tiny fixture's (tests/support/fixture.h: hidden 128, ffn 256), at
// ranks 16 and 4: a q_proj 6x128 -> 128, an mlp.gate_proj 5x128 -> 256 (the fixture's ffn width),
// an mlp.down_proj 4x256 -> 128, and three of them again on the int8 trunk, its codes with a
// 128-wide group scale. Both forms radiance issues are here: the bf16 activation, whose `a_scale`
// is the absent position 1 its schema keeps, and the code stream with a real one. The code sites
// are three widths on purpose: the q_proj's K of 128 leaves one scale group per row, so every k
// reads a_scale[m, 0] whatever index the twin computes; the down_proj's K of 256 puts two groups
// in a row and a wrong group index on most of its k's; and the up_proj's K of 300 puts three groups
// in a row, the last one partial (128, 128 and 44 wide), so a width taken as ceil(K / groups) --
// 100 here -- misreads 100..127 and 200..255, and check_kernel says so.
#include "support/adjoint_check.h"
#include "support/test.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using namespace r97;
using namespace r97::adjoint_check;

namespace {

/* bf16 carries 8 significand bits, so a kernel that rounds as it goes -- the expand's y store,
 * the twin's f64 -- lands within about 2^-9 relative; 0.02 leaves the sum's rounding room without
 * hiding an op that computes something else. */
const double bf16_tol = 0.02;

/* The twin is exact in f64, so central differences reproduce its directional derivative far
 * inside 1e-6 relative (the shared check states exactly that). */
const double fd_tol = 1e-6;

/* A gradient in the ParamSet is f32 (include/r97/params.h), and an entry is a sum of f32 products
 * that can cancel, so an entry carries about 1e-7 of the largest product it summed -- which may be
 * far more than 1e-7 of the entry. A check against the twin therefore measures the difference
 * against the block's largest entry: 1e-5 of it, a hundred times the storage's own noise, still
 * catching a term that is missing, doubled or transposed. */
const double grad_tol = 1e-5;

/* ------------------------------------------------------------------ the sites */

/* One site of the adapter pool as an op call: a row band of M, the trunk's K in, N out, rank r,
 * S pool rows, and (group > 0) the int8 trunk's scale group width. */
struct Site {
    const char* name;
    int64_t M, K, N, R, S, group;
    const char* module;  // the PEFT module path the gradients are named after
};

const Site sites[] = {
    {"tiny_q_proj", 6, 128, 128, 16, 2, 0, "model.layers.0.self_attn.q_proj"},
    {"tiny_gate_proj", 5, 128, 256, 16, 1, 0, "model.layers.0.mlp.gate_proj"},
    {"tiny_down_proj", 4, 256, 128, 4, 1, 0, "model.layers.0.mlp.down_proj"},
    {"coded_q_proj", 3, 128, 128, 4, 2, 128, "model.layers.1.self_attn.q_proj"},
    {"coded_down_proj", 4, 256, 128, 4, 1, 128, "model.layers.1.mlp.down_proj"},
    {"coded_up_proj", 5, 300, 96, 4, 2, 128, "model.layers.1.mlp.up_proj"},
};

bool coded_site(const Site& s) { return s.group > 0; }

/* Row m's adapter. Row 1 is a base row spelled -1 and row 2 one spelled past the pool (S), the two
 * spellings a record uses, so both ops' base-row rule is exercised on every shape. */
int32_t slot_of(const Site& s, int64_t m) {
    if (m == 1) return -1;
    if (m == 2) return (int32_t)s.S;
    return (int32_t)(m % s.S);
}

/* ------------------------------------------------------------------ records */

/* `lora_shrink` as radiance's schema states it: a, a_scale, slot, lora_a, then the write t. The
 * schema is five operands wide in both forms (`ref_registry.cpp`: `OPD_O("a_scale")`), and a bf16
 * call passes no scale, so its position 1 is the absent operand: the tape names it `none:1` and
 * gives it dtype `invalid` (tape.cpp, rad_dtype_name(RAD_DT_INVALID)). */
const char* absent_dtype = "invalid";

OpRecord shrink_rec(const Site& s, bool coded) {
    OpRecord rec;
    rec.op = "lora_shrink";
    rec.params = {{"M", s.M},
                  {"K", s.K},
                  {"R", s.R},
                  {"S", s.S},
                  {"group", coded ? s.group : 0},
                  {"dtype", coded ? "i8" : "bf16"}};
    rec.operands = {{"a", coded ? "i8" : "bf16", {s.M, s.K}, 0, s.M, false}};
    if (coded)
        rec.operands.push_back(
            {"a_scale", "f32", {s.M, (s.K + s.group - 1) / s.group}, 0, s.M, false});
    else
        rec.operands.push_back({"a_scale", absent_dtype, {}, 0, 0, false});
    rec.operands.push_back({"slot", "i32", {s.M}, 0, s.M, false});
    rec.operands.push_back({"lora_a", "bf16", {s.S, s.R, s.K}, 0, s.S, false});
    rec.operands.push_back({"t", "f32", {s.M, s.R}, 0, s.M, true});
    return rec;
}

/* `lora_expand`: t, slot, lora_b, scale, then the inout write y. */
OpRecord expand_rec(const Site& s, const char* op = "lora_expand") {
    OpRecord rec;
    rec.op = op;
    rec.params = {{"M", s.M}, {"N", s.N}, {"R", s.R}, {"S", s.S}, {"dtype", "bf16"}};
    rec.operands = {{"t", "f32", {s.M, s.R}, 0, s.M, false},
                    {"slot", "i32", {s.M}, 0, s.M, false},
                    {"lora_b", "bf16", {s.S, s.N, s.R}, 0, s.S, false},
                    {"scale", "f32", {s.S, s.N}, 0, s.S, false},
                    {"y", "bf16", {s.M, s.N}, 0, s.M, true}};
    return rec;
}

/* ------------------------------------------------------------------ values */

double as_bf16(double v) { return (double)rad_bf16_to_f32(rad_f32_to_bf16((float)v)); }

struct Case {
    std::vector<AdValue> in, out;
};

/* The checkers' inputs: the read operands' values in schema order, the written ones' dtypes and
 * shapes. Values are the ones the trunk holds -- bf16 exactly, codes as integers, scales positive
 * f32 -- and no factor is zero unless `b_zero` asks for the adapter that has not moved off its
 * start (docs/ADAPTER.md 4: B is zero). */
Case make_case(const Site& s, const OpRecord& rec, bool b_zero, uint64_t seed) {
    Case c;
    for (const Operand& o : rec.operands) {
        AdValue v;
        v.dtype = o.dtype;
        v.shape = o.shape;
        int64_t n = 1;
        for (int64_t d : o.shape) n *= d;
        /* The absent position of an optional operand holds no tensor, so no value: the checkers
         * carry it empty and the twin must not read it. */
        if (o.dtype == absent_dtype) {
            (o.written ? c.out : c.in).push_back(std::move(v));
            continue;
        }
        v.value.assign((size_t)n, 0.0);
        if (o.buffer == "slot") {
            for (int64_t i = 0; i < n; ++i) v.value[(size_t)i] = slot_of(s, i);
        } else if (o.buffer == "a") {
            for (int64_t i = 0; i < n; ++i)
                v.value[(size_t)i] = o.dtype == "i8"
                                          ? (double)(int64_t)std::llround(urand(&seed) * 60.0)
                                          : as_bf16(urand(&seed));
        } else if (o.buffer == "a_scale") {
            /* One positive f32 scale per group, the way a rotated checkpoint's stream looks. */
            for (int64_t i = 0; i < n; ++i)
                v.value[(size_t)i] =
                    (double)(float)(0.02 + 0.01 * (double)(i % 7) + 0.004 * std::fabs(urand(&seed)));
        } else if (o.buffer == "lora_a") {
            for (int64_t i = 0; i < n; ++i) v.value[(size_t)i] = as_bf16(urand(&seed));
        } else if (o.buffer == "lora_b") {
            if (!b_zero)
                for (int64_t i = 0; i < n; ++i) v.value[(size_t)i] = as_bf16(urand(&seed));
        } else if (o.buffer == "t") {
            for (int64_t i = 0; i < n; ++i) v.value[(size_t)i] = (double)(float)urand(&seed);
        } else if (o.buffer == "scale") {
            /* alpha/r, 2.0 by default (docs/ADAPTER.md 4), one value per adapter row so a row
             * read from the wrong slot of the pool shows. */
            for (int64_t i = 0; i < n; ++i)
                v.value[(size_t)i] = (double)(float)(2.0 / (1 + (i / s.N) % 2));
        } else {  // y: the base the trunk's GEMM wrote
            for (int64_t i = 0; i < n; ++i) v.value[(size_t)i] = as_bf16(urand(&seed));
        }
        (o.written ? c.out : c.in).push_back(std::move(v));
    }
    return c;
}

uint64_t seed_of(const Site& s) {
    return 0x1000u + (uint64_t)(s.M * 31 + s.K * 7 + s.N * 3 + s.R);
}

/* One line per check, as tests/host/adjoint_add.cpp writes its numbers: the evidence a reader sees
 * in a passing run. */
void report(const std::string& what, const KernelReport& r) {
    std::printf("  %s: kernel max err %.3g\n", what.c_str(), r.max_err);
}

void report(const std::string& what, const FdReport& r) {
    std::printf("  %s: adjoint rel err %.2g\n", what.c_str(), r.rel);
}

/* A gradient block is not empty: a zero code or a zero factor zeroes a column of it honestly, so
 * the statement is that the block is filled in, not that every entry is away from zero. */
void require_filled(const Param& p) {
    size_t nonzero = 0;
    for (float v : p.value) nonzero += v != 0.0f ? 1 : 0;
    R97_CHECK(nonzero * 4 >= p.value.size() * 3);
}

/* The names a failure carries, so one line says which site and which form. */
std::string label(const Site& s, const char* what) { return std::string(s.name) + " " + what; }

void require(const std::string& what, const KernelReport& r) {
    if (!r.ok) fail(what + ": " + r.note);
}

void require(const std::string& what, const FdReport& r) {
    if (!r.ok) fail(what + ": " + r.note);
}

/* ------------------------------------------------------------------ the bf16 trunk's kernel check */

/* radiance's row for `lora_shrink` wants all five schema positions, an absent `a_scale` staying in
 * its place as a tensor with no data (abi/rad_plugin.h: "an optional operand a call does not pass
 * is a RadTensor whose data is null"). tests/support/adjoint_check.h fills every position with
 * bytes and refuses a dtype radiance does not know, and a bf16 record carries its scale position
 * absent (dtype `invalid`, no data) -- so its kernel run is launched here, by the same recipe the
 * shared check uses. */
KernelReport float_shrink_kernel(const OpRecord& rec, const std::vector<AdValue>& in,
                                 const AdValue& out_spec) {
    KernelReport rep;
    const RadKernelInfo* k = find_host_kernel("lora_shrink");
    if (!k) {
        rep.note = "no host kernel for lora_shrink in the kernel table";
        return rep;
    }
    std::vector<AdValue> expect;
    TwinFn twin = twin_find(rec.op, rec.params, &rep.note);
    if (!twin) return rep;
    twin(rec, in, &expect);

    RadTensor t[5]{};
    auto fill = [](RadTensor* x, const AdValue& a) {
        x->dtype = rad_dtype_parse(a.dtype.c_str());
        x->rank = (uint32_t)a.shape.size();
        int64_t n = 1;
        for (size_t d = a.shape.size(); d-- > 0;) {
            x->shape[d] = a.shape[d];
            x->stride[d] = n;
            n *= a.shape[d];
        }
        x->data = std::calloc((size_t)std::max<int64_t>(1, rad_dtype_bytes(x->dtype, n)), 1);
        for (int64_t e = 0; e < n; ++e) rad_store_f32(x->data, x->dtype, e, (float)a.value[(size_t)e]);
    };
    fill(&t[0], in[0]);
    t[1].dtype = RAD_DT_INVALID;  // a_scale: present, no bytes
    fill(&t[2], in[2]);
    fill(&t[3], in[3]);
    fill(&t[4], out_spec);
    for (int64_t e = 0; e < rad_tensor_numel(&t[4]); ++e) rad_store_f32(t[4].data, t[4].dtype, e, 0.0f);

    std::vector<RadParam> ps = rec_params(rec);
    RadArgs args{};
    args.t = t;
    args.n_t = 5;
    args.p = ps.data();
    args.n_p = (int)ps.size();
    args.world_size = 1;
    const int rc = k->launch(&args, nullptr);
    if (rc != RAD_OK) {
        rep.note = "host kernel launch failed: rc=" + std::to_string(rc);
    } else {
        rep.ok = true;
        for (int64_t e = 0; e < rad_tensor_numel(&t[4]); ++e) {
            const double got = rad_load_f32(t[4].data, t[4].dtype, e);
            const double want = expect[0].value[(size_t)e];
            const double err = std::fabs(got - want);
            rep.max_err = std::max(rep.max_err, err);
            if (err > bf16_tol + bf16_tol * std::fabs(want)) {
                rep.ok = false;
                if (rep.note.empty())
                    rep.note = "kernel " + std::to_string(got) + " vs twin " + std::to_string(want) +
                               " at element " + std::to_string(e);
            }
        }
    }
    for (RadTensor& x : t) std::free(x.data);
    return rep;
}

/* ------------------------------------------------------------------ the gradient set */

/* One adjoint call with a gradient set, built the way the shared FD check builds it: `fwd` spans
 * the schema, `dc` the same with every read's slot zeroed, `cot` one random entry for each write
 * (or the values handed in, for the walk that has already fixed them). */
struct Call {
    std::vector<AdValue> fwd, cot, dc;
    ParamSet grads;
};

Call run_adjoint(const OpRecord& rec, const std::vector<AdValue>& in,
                 const std::vector<AdValue>& out_spec, const char* prefix, uint64_t seed,
                 const std::vector<double>* cot_in = nullptr, ParamSet* grads_in = nullptr) {
    Call c;
    if (grads_in) c.grads = *grads_in;
    std::vector<AdValue> out;
    TwinFn twin = twin_find(rec.op, rec.params, nullptr);
    twin(rec, in, &out);
    c.fwd.resize(rec.operands.size());
    c.dc.resize(rec.operands.size());
    size_t r = 0, w = 0;
    for (size_t i = 0; i < rec.operands.size(); ++i) {
        if (rec.operands[i].written) {
            c.fwd[i] = out[w];
            AdValue cot = out_spec[w];
            int64_t n = 1;
            for (int64_t d : cot.shape) n *= d;
            cot.value.assign((size_t)n, 0.0);
            for (double& v : cot.value) v = urand(&seed);
            if (cot_in) cot.value = *cot_in;
            c.cot.push_back(std::move(cot));
            ++w;
        } else {
            c.fwd[i] = in[r];
            c.dc[i] = in[r];
            c.dc[i].value.assign(in[r].value.size(), 0.0);
            ++r;
        }
    }
    AdjointCall call{rec, c.fwd, c.cot, &c.dc, &c.grads, prefix};
    AdjointFn adj = adjoint_find(rec.op, Side::host, rec.params, nullptr);
    R97_CHECK(adj != nullptr);
    adj(call);
    if (grads_in) *grads_in = c.grads;
    return c;
}

const Param* find_grad(const ParamSet& grads, const std::string& name) {
    for (const Param& p : grads)
        if (p.name == name) return &p;
    return nullptr;
}

/* <twin(rec, in), cot> with `dir` added to read operand `leaf` -- the loss the adjoint's numbers
 * are the derivative of. */
double loss_along(const OpRecord& rec, const std::vector<AdValue>& in, size_t leaf,
                  const std::vector<double>& dir, double eps, const std::vector<double>& cot) {
    std::vector<AdValue> x = in;
    for (size_t e = 0; e < dir.size(); ++e) x[leaf].value[e] += eps * dir[e];
    std::vector<AdValue> out;
    TwinFn twin = twin_find(rec.op, rec.params, nullptr);
    twin(rec, x, &out);
    double s = 0.0;
    for (size_t e = 0; e < out[0].value.size() && e < cot.size(); ++e)
        s += out[0].value[e] * cot[e];
    return s;
}

/* The Param a leaf's gradient lands in, against central differences of the twin.
 *
 * A factor is [S, ...] and its gradient one block of that shape, added up over the pool rows the
 * record's slots name: the Param says what that sum is, so the direction that checks it is the same
 * block on every pool row. Each entry is checked on its own that way, and the whole block once
 * along a random direction, where the entries cancel and nothing tighter than f32's own noise in
 * the ParamSet can be asked for. */
void check_grad(const std::string& what, const OpRecord& rec, const Call& c,
                const std::vector<AdValue>& in, size_t leaf, const std::string& name,
                const std::vector<int64_t>& shape, uint64_t* seed) {
    const Param* p = find_grad(c.grads, name);
    R97_CHECK(p != nullptr);
    R97_CHECK(p->shape == shape);
    int64_t pn = 1;
    for (int64_t d : shape) pn *= d;
    R97_CHECK(p->value.size() == (size_t)pn);

    double worst = 0.0;
    double biggest = 0.0;
    for (float v : p->value) biggest = std::max(biggest, (double)std::fabs((double)v));
    auto grad_close = [&](double a, double n) {
        return std::fabs(a - n) <= grad_tol * biggest + 1e-12;
    };

    const double eps = 1e-3;  // the twin is linear in the leaf, so the step only moves the noise
    const int64_t n_all = (int64_t)in[leaf].value.size();
    std::vector<double> full((size_t)n_all, 0.0);
    auto derivative = [&](const std::vector<double>& dir) {
        return (loss_along(rec, in, leaf, dir, eps, c.cot[0].value) -
                loss_along(rec, in, leaf, dir, -eps, c.cot[0].value)) /
               (2.0 * eps);
    };

    for (int64_t k = 0; k < pn && k < 32; ++k) {
        const int64_t i = (int64_t)(((uint64_t)k * 2654435761u) % (uint64_t)pn);
        std::fill(full.begin(), full.end(), 0.0);
        for (int64_t e = i; e < n_all; e += pn) full[(size_t)e] = 1.0;  // the block, on every slot
        const double numeric = derivative(full);
        const double analytic = (double)p->value[(size_t)i];
        if (biggest > 0.0) worst = std::max(worst, std::fabs(analytic - numeric) / biggest);
        if (!grad_close(analytic, numeric)) {
            char note[256];
            std::snprintf(note, sizeof note,
                          "%s: %s[%lld]: adjoint says %.12g, finite differences say %.12g",
                          what.c_str(), name.c_str(), (long long)i, analytic, numeric);
            fail(note);
        }
    }

    for (double& v : full) v = urand(seed);
    for (int64_t i = 0; i < n_all; ++i) full[(size_t)i] = full[(size_t)(i % pn)];  // the block again
    const double numeric = derivative(full);
    double analytic = 0.0;
    for (int64_t i = 0; i < pn; ++i) analytic += (double)p->value[(size_t)i] * full[(size_t)i];
    if (biggest > 0.0) worst = std::max(worst, std::fabs(analytic - numeric) / biggest);
    std::printf("  %s %s: %lld entries and a direction sum, worst %.1g of the block's largest\n",
                what.c_str(), name.c_str(), (long long)std::min<int64_t>(pn, 32), worst);
    if (!grad_close(analytic, numeric)) {
        char note[256];
        std::snprintf(note, sizeof note,
                      "%s: %s: adjoint says %.12g, finite differences say %.12g", what.c_str(),
                      name.c_str(), analytic, numeric);
        fail(note);
    }
}

/* ------------------------------------------------------------------ the schema the tape records */

/* The schema a radiance plugin declares for an op: the runtime issues one operand per position and
 * the tape records them in that order, absent optional ones included, so a record the twin is
 * checked against has to be this one. libref exports the same table the checks use for kernels. */
const RadOpSchema* find_schema(const char* op) {
    Plugin& p = libref();
    if (!p.so) return nullptr;
    auto count = (int (*)())dlsym(p.so, "rad_kernel_schema_count");
    auto at = (const RadOpSchema* (*)(int))dlsym(p.so, "rad_kernel_schema_at");
    if (!count || !at) return nullptr;
    for (int i = 0; i < count(); ++i) {
        const RadOpSchema* s = at(i);
        if (s && s->op && std::string(s->op) == op) return s;
    }
    return nullptr;
}

void require_schema(const OpRecord& rec) {
    const RadOpSchema* s = find_schema(rec.op.c_str());
    R97_CHECK(s != nullptr);
    R97_CHECK(s->n_operands == (int)rec.operands.size());
    for (int i = 0; i < s->n_operands; ++i) {
        const RadOperandSpec& spec = s->operands[i];
        const Operand& o = rec.operands[i];
        R97_CHECK(spec.name != nullptr && std::string(spec.name) == o.buffer);
        const bool written = spec.role == RAD_OPD_OUT || spec.role == RAD_OPD_INOUT;
        R97_CHECK(written == o.written);
    }
}

/* The records above are the schema radiance issues, operand for operand, and the shrink's is five
 * wide in both forms: `a_scale` is the schema's optional operand at position 1, and a bf16 call
 * leaves that place absent (`none:1`, dtype `invalid`) rather than dropping it. This is the test
 * the family first lacked: a four-operand bf16 record is refused, whether it drops the position or
 * fills it with a scale its `dtype` says is not there. */
R97_TEST(records_are_the_schema_radiance_issues) {
    for (const Site& s : sites) {
        require_schema(shrink_rec(s, false));
        if (coded_site(s)) require_schema(shrink_rec(s, true));
        require_schema(expand_rec(s));
        require_schema(expand_rec(s, "lora_expand_mm"));
    }

    const RadOpSchema* sh = find_schema("lora_shrink");
    R97_CHECK(sh && sh->n_operands == 5 && sh->operands[1].name != nullptr);
    R97_CHECK(std::string(sh->operands[1].name) == "a_scale" && sh->operands[1].optional == 1);

    const Site& s = sites[0];  // tiny_q_proj, the bf16 trunk
    const OpRecord f = shrink_rec(s, false);
    R97_CHECK(f.operands.size() == 5 && f.operands[1].dtype == absent_dtype);

    OpRecord dropped = f;  // the four-operand record this family first believed
    dropped.operands.erase(dropped.operands.begin() + 1);
    Case cd = make_case(s, dropped, false, seed_of(s));
    std::vector<AdValue> od;
    twin_find("lora_shrink", dropped.params, nullptr)(dropped, cd.in, &od);
    R97_CHECK(od.empty());

    OpRecord lying = f;  // a float record with a real scale in the absent position
    lying.operands[1] = {"a_scale", "f32", {s.M, 1}, 0, s.M, false};
    Case cl = make_case(s, lying, false, seed_of(s));
    std::vector<AdValue> ol;
    twin_find("lora_shrink", lying.params, nullptr)(lying, cl.in, &ol);
    R97_CHECK(ol.empty());
}

/* ------------------------------------------------------------------ the tests */

/* The shrink's twin against radiance's host row: the bf16 activation through the launch above
 * (position 1 absent), the code stream with its group scale through the shared check (all five
 * positions filled). */
R97_TEST(lora_shrink_twin_matches_the_kernel) {
    for (const Site& s : sites) {
        const bool coded = coded_site(s);
        const OpRecord rec = shrink_rec(s, coded);
        const Case c = make_case(s, rec, false, seed_of(s));
        const KernelReport r = coded ? check_kernel(rec, c.in, c.out, bf16_tol)
                                     : float_shrink_kernel(rec, c.in, c.out[0]);
        report(label(s, coded ? "lora_shrink i8" : "lora_shrink bf16"), r);
        require(label(s, coded ? "lora_shrink i8" : "lora_shrink bf16"), r);
    }
}

/* The expand's twin against radiance's host row, and the mm variant's, whose only difference is t
 * rounded to bf16 before the sum. */
R97_TEST(lora_expand_twin_matches_the_kernel) {
    for (const Site& s : sites) {
        for (const char* op : {"lora_expand", "lora_expand_mm"}) {
            const OpRecord rec = expand_rec(s, op);
            const Case c = make_case(s, rec, false, seed_of(s));
            const KernelReport r = check_kernel(rec, c.in, c.out, bf16_tol);
            report(label(s, op), r);
            require(label(s, op), r);
        }
    }
}

/* The shrink's adjoint against central differences of its twin, on both forms. For the code stream
 * the difference is taken over the codes and over the scales, so the dequantisation's own factor is
 * part of what is checked: the value-space cotangent split the way the quantiser will read it. */
R97_TEST(lora_shrink_adjoint_matches_finite_differences) {
    for (const Site& s : sites) {
        for (bool coded : {false, true}) {
            if (coded != coded_site(s)) continue;
            const OpRecord rec = shrink_rec(s, coded);
            const Case c = make_case(s, rec, false, seed_of(s) ^ 0x51ab);
            const FdReport r = check_adjoint(rec, c.in, c.out, fd_tol, seed_of(s));
            report(label(s, coded ? "lora_shrink i8" : "lora_shrink bf16"), r);
            require(label(s, coded ? "lora_shrink i8" : "lora_shrink bf16"), r);
        }
    }
}

/* The expand's adjoint against central differences of its twin. The mm variant is not in this
 * list: rounding t to bf16 is a step function, so its twin is piecewise constant and a central
 * difference of it is the derivative of nothing. */
R97_TEST(lora_expand_adjoint_matches_finite_differences) {
    for (const Site& s : sites) {
        const OpRecord rec = expand_rec(s);
        const Case c = make_case(s, rec, false, seed_of(s));
        const FdReport r = check_adjoint(rec, c.in, c.out, fd_tol, seed_of(s) ^ 0xe8ab);
        report(label(s, "lora_expand"), r);
        require(label(s, "lora_expand"), r);
    }
}

/* The gradients are the adapter's trainable leaves: one Param per op, named
 * `<prefix>.lora_A` [r, K] and `<prefix>.lora_B` [N, r], each against central differences of the
 * twin, and a second call of the same module adds up instead of overwriting. */
R97_TEST(lora_gradients_are_named_leaves) {
    for (const Site& s : sites) {
        const bool coded = coded_site(s);
        uint64_t seed = seed_of(s) ^ 0x6e55;

        const OpRecord srec = shrink_rec(s, coded);
        const Case sc = make_case(s, srec, false, seed);
        const uint64_t cot_seed = 0x9e37u + (uint64_t)s.K;
        const Call cs = run_adjoint(srec, sc.in, sc.out, s.module, cot_seed);
        check_grad(label(s, "lora_shrink"), srec, cs, sc.in, 3,
                   std::string(s.module) + ".lora_A", {s.R, s.K}, &seed);
        /* A block that stayed zero would pass a shape check and fail the point. */
        const Param* pa = find_grad(cs.grads, std::string(s.module) + ".lora_A");
        R97_CHECK(pa != nullptr);
        require_filled(*pa);

        const OpRecord erec = expand_rec(s);
        const Case ec = make_case(s, erec, false, seed);
        const Call ce = run_adjoint(erec, ec.in, ec.out, s.module, seed);
        check_grad(label(s, "lora_expand"), erec, ce, ec.in, 2, std::string(s.module) + ".lora_B",
                   {s.N, s.R}, &seed);
        const Param* pb = find_grad(ce.grads, std::string(s.module) + ".lora_B");
        R97_CHECK(pb != nullptr);
        require_filled(*pb);

        /* The same module reached twice, the second call adding to the set the first left: one
         * entry, twice the gradient -- the set is added to by name, not rewritten. */
        ParamSet twice = cs.grads;
        run_adjoint(srec, sc.in, sc.out, s.module, cot_seed, nullptr, &twice);
        R97_CHECK(twice.size() == cs.grads.size());
        const Param* once = find_grad(cs.grads, std::string(s.module) + ".lora_A");
        const Param* doubled = find_grad(twice, std::string(s.module) + ".lora_A");
        R97_CHECK(once && doubled);
        /* Added up in f32, so twice the block is twice its largest entry to f32's noise, not
         * exactly: (((a + b) + c) + a) + b is not (a + b + c) * 2 when the entries cancel. */
        float biggest = 0.0f;
        for (float v : once->value) biggest = std::max(biggest, std::fabs(v));
        for (size_t e = 0; e < once->value.size(); ++e)
            R97_CHECK(std::fabs((double)doubled->value[e] - 2.0 * (double)once->value[e]) <=
                      grad_tol * (double)biggest + 1e-12);
    }
}

/* A fresh adapter starts with B = 0 (docs/ADAPTER.md 4). Then the expand adds nothing to y, its
 * cotangent of t is zero, and the shrink that follows gets dA = 0 while dB is not: the first step
 * moves B, not A. Both halves are stated by the twin and by the adjoint. */
R97_TEST(lora_path_with_b_zero) {
    for (const Site& s : sites) {
        const bool coded = coded_site(s);
        uint64_t seed = seed_of(s) ^ 0xb0;

        const OpRecord erec = expand_rec(s);
        const Case ec = make_case(s, erec, /*b_zero=*/true, seed);
        std::vector<AdValue> out;
        TwinFn twin = twin_find(erec.op, erec.params, nullptr);
        twin(erec, ec.in, &out);
        for (double v : out[0].value) R97_CHECK(v == 0.0);  // y's increment is exactly nothing
        require(label(s, "B = 0 lora_expand kernel"), check_kernel(erec, ec.in, ec.out, bf16_tol));

        const Call ce = run_adjoint(erec, ec.in, ec.out, s.module, seed);
        for (const AdValue& v : ce.dc)  // every read's cotangent: t's is the only one the op has
            for (double e : v.value) R97_CHECK(e == 0.0);

        const Param* pb = find_grad(ce.grads, std::string(s.module) + ".lora_B");
        R97_CHECK(pb != nullptr);
        bool nonzero = false;
        for (float v : pb->value) nonzero = nonzero || v != 0.0f;
        R97_CHECK(nonzero);
        check_grad(label(s, "B = 0 lora_expand"), erec, ce, ec.in, 2,
                   std::string(s.module) + ".lora_B", {s.N, s.R}, &seed);

        /* The shrink, handed the zero cotangent the expand gave back: dA is zero both ways. */
        const OpRecord srec = shrink_rec(s, coded);
        const Case sc = make_case(s, srec, false, seed);
        const std::vector<double> zero((size_t)(s.M * s.R), 0.0);
        const Call cs = run_adjoint(srec, sc.in, sc.out, s.module, seed, &zero);
        const Param* pa = find_grad(cs.grads, std::string(s.module) + ".lora_A");
        R97_CHECK(pa != nullptr);
        for (float v : pa->value) R97_CHECK(v == 0.0f);
        for (const AdValue& v : cs.dc)
            for (double e : v.value) R97_CHECK(e == 0.0);
        check_grad(label(s, "B = 0 lora_shrink"), srec, cs, sc.in, 3,
                   std::string(s.module) + ".lora_A", {s.R, s.K}, &seed);
    }
}

/* One row serves both forms, and it freezes the absent a_scale position with slot and lora_a: a
 * bf16 trunk's record asks no cotangent of any of the three. */
R97_TEST(shrink_frozen_positions_cover_the_absent_scale) {
    for (const Site& s : sites) {
        for (bool coded : {false, true}) {
            if (coded != coded_site(s)) continue;
            const OpRecord rec = shrink_rec(s, coded);
            R97_CHECK(adjoint_frozen(rec.op, Side::host, rec.params) ==
                      std::vector<int>({1, 2, 3}));
        }
    }
}

/* The bf16 form adds both things the issue asks of it: dA in the ParamSet under the module's name,
 * and the contribution to dx. This is the form that first added neither -- its record was read as
 * three operands and refused -- so the test states the two additions directly, not through the FD
 * check's sum over the operands. */
R97_TEST(lora_shrink_bf16_adds_da_and_dx) {
    const Site& s = sites[0];  // tiny_q_proj: the tiny fixture's q_proj, K 128, rank 16
    const OpRecord rec = shrink_rec(s, false);
    const Case c = make_case(s, rec, false, seed_of(s));
    const Call a = run_adjoint(rec, c.in, c.out, s.module, seed_of(s));

    const Param* pa = find_grad(a.grads, std::string(s.module) + ".lora_A");
    R97_CHECK(pa != nullptr);
    R97_CHECK(pa->shape == std::vector<int64_t>({s.R, s.K}));
    double da = 0.0;
    for (float v : pa->value) da += std::fabs((double)v);
    R97_CHECK(da > 0.0);

    R97_CHECK(a.dc[0].value.size() == (size_t)(s.M * s.K));
    double dx = 0.0;
    for (double v : a.dc[0].value) dx += std::fabs(v);
    R97_CHECK(dx > 0.0);

    /* The absent position stays empty: it is not a tensor, so there is no cotangent to add. */
    R97_CHECK(a.dc[1].value.empty());
}

}  // namespace

