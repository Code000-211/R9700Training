// src/adjoint/host/gemm_nt_q_gated.cpp -- the host twin and adjoint of the gated narrow fold.
//
// `gemm_nt_q_gated` at i8a8h is what the int8 trunk issues for gate_up where the row count
// fits the fold's band, in place of `gemm_nt_q` and `gated_had_quant_i8g`: one [N, K] plane,
// the gate's rows first and the up's after, and the product's 128-wide Hadamard quantiser in
// the GEMM's epilogue, so the [M, N] bf16 plane between the two is never written. This file
// is the f64 twin of that pair and the adjoint that puts the quantiser's straight-through
// epilogue (docs/adjoints/quant.md, section 3) in front of the GEMM's backward
// (docs/adjoints/linear.md):
//
//   z[m,j]  =  sum_{k in g} a[m,k] b[j,k] sa[m,g] sb[j,g]      j = 0 .. N/2 - 1, up = rows [N/2, N)
//   p[m,j]  =  act(z[m,j]) * z[m, N/2 + j]                      the record's `act`, silu
//   v       =  the unnormalised Hadamard of each `group` block of p,  s = amax(v) / 127,
//              q = clamp(round(v / s), -127, 127)
//
// The walk hands back the cotangents of what the op wrote: the codes, the scale (dropped, a
// constant) and, when bound, the unrotated product. In the dequantised units the int8 walk
// carries, the codes' cotangent is the product's own (quant.md, section 3), so the epilogue is
// the clamp's mask at the forward's codes, over the group's scale, rotated back with W^T = W,
// with no division the rotation would add: the forward never normalised. The gate then takes
// the product's cotangent to its halves as gated_quant_fp8's adjoint does, and the GEMM's
// backward gives the activation's cotangent in the dequantised basis, as gemm_nt_q's row gives
// it -- its scale not applied again -- and the weight's and its scale's, which the row answers
// because the fold's weight is not the frozen one: only a_scale is marked frozen.
//
// The live tape carries the engine's buffer names, and the join's record prep names only the
// plain linears after their schema, so this row finds its operands by the schema's positions,
// taking the names a tightened record carries when it has them.

#include "r97/adjoint.h"

#include "parallel.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace r97 {
namespace {

// The unit the parallel for cuts this row's row loops into (a one-row call stays on the calling
// thread, and an odd M leaves one row to a worker), the unit a factor pass splits over (one
// output column pair of the fold), and the unit a flat elementwise pass uses. Every unit writes
// its own output and the inner sums stay that unit's serial order, so the partition moves nothing
// a neighbour reads.
constexpr int64_t kRowChunk = 2;
constexpr int64_t kPairChunk = 2;
constexpr int64_t kElemChunk = 1024;

// The top of the int8 code grid, as quant.cpp's kTop.
constexpr double kTop = 127.0;

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

const OpParam* find_param(const OpRecord& rec, const char* name) {
    for (const OpParam& p : rec.params) {
        if (p.first == name) return &p;
    }
    return nullptr;
}

int64_t param_int(const OpRecord& rec, const char* name, int64_t fallback) {
    const OpParam* p = find_param(rec, name);
    if (p == nullptr) return fallback;
    if (const int64_t* v = std::get_if<int64_t>(&p->second)) return *v;
    return fallback;
}

std::string param_string(const OpRecord& rec, const char* name, const char* fallback) {
    const OpParam* p = find_param(rec, name);
    if (p == nullptr) return fallback;
    if (const std::string* s = std::get_if<std::string>(&p->second)) return *s;
    return fallback;
}

// The schema position of the operand a read or a write names; the position itself when the
// live tape's engine name is there instead.
int operand_at(const OpRecord& rec, const char* name, int pos) {
    for (std::size_t i = 0; i < rec.operands.size(); ++i) {
        if (rec.operands[i].buffer == name) return static_cast<int>(i);
    }
    return pos < static_cast<int>(rec.operands.size()) ? pos : -1;
}

const Operand* operand_named(const OpRecord& rec, const char* name) {
    for (const Operand& o : rec.operands) {
        if (o.buffer == name) return &o;
    }
    return nullptr;
}

// The read operand at a schema position; the twin's `operands` holds reads only, in schema order.
const AdValue* read_at(const OpRecord& rec, const std::vector<AdValue>& operands, int schema_pos) {
    if (schema_pos < 0) return nullptr;
    int read = 0;
    for (std::size_t i = 0; i < rec.operands.size(); ++i) {
        if (rec.operands[i].written) continue;
        if (static_cast<int>(i) == schema_pos) {
            return read < static_cast<int>(operands.size()) ? &operands[static_cast<std::size_t>(read)]
                                                            : nullptr;
        }
        ++read;
    }
    return nullptr;
}

AdValue make_value(const std::string& dtype, const std::vector<int64_t>& shape,
                   std::vector<double> value) {
    AdValue v;
    v.dtype = dtype;
    v.shape = shape;
    v.value = std::move(value);
    return v;
}

// A write's dtype as the record carries it, or the twin's own when the record leaves it unset.
std::string write_dtype(const OpRecord& rec, const char* name, const char* fallback) {
    const Operand* o = operand_named(rec, name);
    return o != nullptr && !o->dtype.empty() ? o->dtype : fallback;
}

// The grouped scale for row `row` and group `g` of K, as the host linear reads it: one per row
// and block, a whole-row or a whole-matrix scale accepted as well.
double scale_at(const AdValue* scale, int64_t row, int64_t group, int64_t rows, int64_t groups) {
    if (scale == nullptr) return 1.0;
    const int64_t n = static_cast<int64_t>(scale->value.size());
    if (n == 0) return 1.0;
    if (n == 1) return scale->value[0];
    if (n == rows * groups) return scale->value[static_cast<std::size_t>(row * groups + group)];
    if (n == groups) return scale->value[static_cast<std::size_t>(group)];
    if (n == rows) return scale->value[static_cast<std::size_t>(row)];
    const int64_t at = std::min<int64_t>(row * groups + group, n - 1);
    return scale->value[static_cast<std::size_t>(at)];
}

// silu and its slope, gated_quant_fp8's pair; the sigmoid form is the record's other `act`.
double act_of(double g, bool sigmoid) {
    const double s = 1.0 / (1.0 + std::exp(-g));
    return sigmoid ? s : g * s;
}

double act_slope_of(double g, bool sigmoid) {
    const double s = 1.0 / (1.0 + std::exp(-g));
    return sigmoid ? s * (1.0 - s) : s * (1.0 + g * (1.0 - s));
}

// The unnormalised Walsh-Hadamard of one group, quant.cpp's loop: W W = t I, and the row's
// backward rotates back with the same transform, so nothing divides by t.
void fwht(double* v, int64_t t) {
    for (int64_t h = 1; h < t; h <<= 1)
        for (int64_t i0 = 0; i0 < t; i0 += 2 * h)
            for (int64_t j = i0; j < i0 + h; ++j) {
                const double a = v[j];
                const double b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
}

// The call's extents, as the kernel reads them: one `group` key at once the K block and the
// quantiser's block of the fused width.
struct GatedShape {
    int64_t M = 0, K = 0, N = 0, n = 0, group = 0, groups_k = 0, groups_n = 0;
};

bool shape_of(const OpRecord& rec, const AdValue& a, const AdValue& b, GatedShape* sh) {
    sh->K = param_int(rec, "K", a.shape.size() >= 2 ? a.shape[1] : 0);
    sh->N = param_int(rec, "N", b.shape.size() >= 2 ? b.shape[0] : 0);
    // `M` is the weight's capacity; the operand's own first dimension is how many rows this call
    // actually has, and the one a tightened record carries.
    sh->M = a.shape.empty() ? param_int(rec, "M", 0) : a.shape[0];
    sh->group = param_int(rec, "group", 0);
    if (sh->M <= 0 || sh->K <= 0 || sh->N % 2) return false;
    // The rotation exists only for a power of two, and a group must not straddle the end of the
    // fused width: the kernel refuses both, and so does the row.
    if (sh->group <= 0 || (sh->group & (sh->group - 1)) != 0) return false;
    sh->n = sh->N / 2;
    if (sh->n % sh->group != 0) return false;
    sh->groups_k = (sh->K + sh->group - 1) / sh->group;
    sh->groups_n = sh->n / sh->group;
    return true;
}

// The gate and the up halves in f64, the host kernel's grouped chain: a sub-sum per K block,
// folded with the block's scales.
void halves(const AdValue& a, const AdValue& b, const AdValue* as, const AdValue* bs,
            const GatedShape& sh, std::vector<double>* zg, std::vector<double>* zu) {
    zg->assign(static_cast<std::size_t>(sh.M * sh.n), 0.0);
    zu->assign(static_cast<std::size_t>(sh.M * sh.n), 0.0);
    parallel_for(sh.M, kRowChunk, [&](int64_t m) {
        for (int64_t j = 0; j < sh.n; ++j) {
            double g_acc = 0.0, u_acc = 0.0;
            for (int64_t g = 0; g < sh.groups_k; ++g) {
                const int64_t k0 = g * sh.group;
                const int64_t k1 = std::min(sh.K, k0 + sh.group);
                double s_g = 0.0, s_u = 0.0;
                for (int64_t k = k0; k < k1; ++k) {
                    const double av = a.value[static_cast<std::size_t>(m * sh.K + k)];
                    s_g += av * b.value[static_cast<std::size_t>(j * sh.K + k)];
                    s_u += av * b.value[static_cast<std::size_t>((sh.n + j) * sh.K + k)];
                }
                const double sa = scale_at(as, m, g, sh.M, sh.groups_k);
                g_acc += s_g * sa * scale_at(bs, j, g, sh.N, sh.groups_k);
                u_acc += s_u * sa * scale_at(bs, sh.n + j, g, sh.N, sh.groups_k);
            }
            (*zg)[static_cast<std::size_t>(m * sh.n + j)] = g_acc;
            (*zu)[static_cast<std::size_t>(m * sh.n + j)] = u_acc;
        }
    });
}

// The unrotated product and the quantiser's block over one `group` of it: the rotated value,
// the scale and the codes, quant.cpp's rule.
void gated_quant_block(const std::vector<double>& p, int64_t m, int64_t gi, const GatedShape& sh,
                       std::vector<double>* v, std::vector<double>* code, double* s) {
    const int64_t k0 = gi * sh.group;
    const std::size_t base = static_cast<std::size_t>(m * sh.n + k0);
    v->assign(p.begin() + base, p.begin() + base + static_cast<std::size_t>(sh.group));
    fwht(v->data(), sh.group);
    double amax = 0.0;
    for (const double x : *v) amax = std::max(amax, std::fabs(x));
    // The zero group has no direction to scale: one scale, codes zero, nothing clamped.
    *s = amax > 0.0 ? amax * (1.0 / kTop) : 1.0;
    code->assign(static_cast<std::size_t>(sh.group), 0.0);
    for (int64_t i = 0; i < sh.group; ++i)
        (*code)[static_cast<std::size_t>(i)] =
            std::min(kTop, std::max(-kTop, std::nearbyint((*v)[static_cast<std::size_t>(i)] *
                                                           (1.0 / *s))));
}

void gemm_nt_q_gated_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                          std::vector<AdValue>* out) {
    const int ia = operand_at(rec, "a", 0);
    const int ib = operand_at(rec, "b", 2);
    if (ia < 0 || ib < 0) return;
    const AdValue* a = read_at(rec, operands, ia);
    const AdValue* b = read_at(rec, operands, ib);
    const AdValue* as = read_at(rec, operands, operand_at(rec, "a_scale", 1));
    const AdValue* bs = read_at(rec, operands, operand_at(rec, "b_scale", 3));
    if (a == nullptr || b == nullptr) return;
    GatedShape sh;
    if (!shape_of(rec, *a, *b, &sh)) return;
    const bool sigmoid = param_string(rec, "act", "silu") == "sigmoid";

    std::vector<double> zg, zu;
    halves(*a, *b, as, bs, sh, &zg, &zu);
    std::vector<double> p(static_cast<std::size_t>(sh.M * sh.n), 0.0);
    parallel_for(static_cast<int64_t>(p.size()), kElemChunk, [&](int64_t i) {
        p[static_cast<std::size_t>(i)] =
            act_of(zg[static_cast<std::size_t>(i)], sigmoid) * zu[static_cast<std::size_t>(i)];
    });

    // `out_bf16`, when bound, is the unrotated product the codes were taken over.
    const Operand* ob = operand_named(rec, "out_bf16");
    const bool has_bf16 = ob != nullptr && ob->written && !ob->shape.empty();

    std::vector<double> code(static_cast<std::size_t>(sh.M * sh.n), 0.0);
    std::vector<double> scale(static_cast<std::size_t>(sh.M * sh.groups_n), 0.0);
    parallel_for(sh.M, kRowChunk, [&](int64_t m) {
        std::vector<double> v, c;
        for (int64_t gi = 0; gi < sh.groups_n; ++gi) {
            const int64_t k0 = gi * sh.group;
            double s = 0.0;
            gated_quant_block(p, m, gi, sh, &v, &c, &s);
            const std::size_t base = static_cast<std::size_t>(m * sh.n + k0);
            std::copy(c.begin(), c.end(), code.begin() + base);
            scale[static_cast<std::size_t>(m * sh.groups_n + gi)] = s;
        }
    });
    out->push_back(make_value(write_dtype(rec, "q", "i8"), {sh.M, sh.n}, std::move(code)));
    out->push_back(make_value(write_dtype(rec, "scale", "f32"), {sh.M, sh.groups_n},
                              std::move(scale)));
    if (has_bf16) out->push_back(make_value(write_dtype(rec, "out_bf16", "bf16"), {sh.M, sh.n},
                                            std::move(p)));
}

void gemm_nt_q_gated_adjoint(const AdjointCall& call) {
    const int ia = operand_at(call.rec, "a", 0);
    const int ib = operand_at(call.rec, "b", 2);
    const int ibs = operand_at(call.rec, "b_scale", 3);
    if (ia < 0 || ib < 0 || call.cot.empty() || call.dc == nullptr ||
        static_cast<std::size_t>(ia) >= call.dc->size() ||
        static_cast<std::size_t>(ib) >= call.dc->size())
        return;
    const AdValue& a = call.fwd[static_cast<std::size_t>(ia)];
    const AdValue& b = call.fwd[static_cast<std::size_t>(ib)];
    const AdValue* as = nullptr;
    const int ias = operand_at(call.rec, "a_scale", 1);
    if (ias >= 0) as = &call.fwd[static_cast<std::size_t>(ias)];
    const AdValue* bs = ibs >= 0 ? &call.fwd[static_cast<std::size_t>(ibs)] : nullptr;
    const AdValue& q = call.fwd[static_cast<std::size_t>(operand_at(call.rec, "q", 4))];
    const AdValue& sc = call.fwd[static_cast<std::size_t>(operand_at(call.rec, "scale", 5))];
    GatedShape sh;
    if (!shape_of(call.rec, a, b, &sh)) return;
    // The operand extents and the record's M, N, K have to agree; a record that does not is one
    // this row was not written for, and an adjoint must not throw.
    if (a.value.size() != static_cast<std::size_t>(sh.M * sh.K) ||
        b.value.size() != static_cast<std::size_t>(sh.N * sh.K) ||
        q.value.size() != static_cast<std::size_t>(sh.M * sh.n) ||
        sc.value.size() != static_cast<std::size_t>(sh.M * sh.groups_n))
        return;
    const bool sigmoid = param_string(call.rec, "act", "silu") == "sigmoid";

    std::vector<double> zg, zu;
    halves(a, b, as, bs, sh, &zg, &zu);
    std::vector<double> p(static_cast<std::size_t>(sh.M * sh.n), 0.0);
    parallel_for(static_cast<int64_t>(p.size()), kElemChunk, [&](int64_t i) {
        p[static_cast<std::size_t>(i)] =
            act_of(zg[static_cast<std::size_t>(i)], sigmoid) * zu[static_cast<std::size_t>(i)];
    });

    // The product's cotangent in the unrotated basis: the codes', masked at the forward's codes
    // and over the group's scale, rotated back, plus out_bf16's own when bound. The scale's
    // cotangent is dropped: the scale is a constant.
    const std::size_t total = static_cast<std::size_t>(sh.M * sh.n);
    std::vector<double> dy(total, 0.0);
    if (call.cot[0].value.size() == total) {
        std::vector<double> d = call.cot[0].value;  // the rotated basis, the codes' units
        // A record whose scale is not positive is one this row refuses; the check runs before the
        // partition, so a bad group leaves the activation's and the weight's cotangents untouched,
        // as the serial loop's own early return did.
        for (const double s : sc.value)
            if (!(s > 0.0)) return;
        parallel_for(sh.M, kRowChunk, [&](int64_t m) {
            for (int64_t gi = 0; gi < sh.groups_n; ++gi) {
                const int64_t k0 = gi * sh.group;
                const double s = sc.value[static_cast<std::size_t>(m * sh.groups_n + gi)];
                // The rotated value the quantiser quantised, the product's block through the
                // same transform the forward used.
                std::vector<double> v(p.begin() + static_cast<std::size_t>(m * sh.n + k0),
                                      p.begin() + static_cast<std::size_t>(m * sh.n + k0 + sh.group));
                fwht(v.data(), sh.group);
                for (int64_t i = 0; i < sh.group; ++i) {
                    const std::size_t j = static_cast<std::size_t>(m * sh.n + k0 + i);
                    // The forward's code decides the mask the way the family's does: a code at
                    // the top of the grid keeps its cotangent only when the value was inside it.
                    const double code = q.value[j];
                    const bool top = code == kTop || code == -kTop;
                    const bool live = !top || std::fabs(v[static_cast<std::size_t>(i)] * (1.0 / s)) <= kTop;
                    d[j] = live ? d[j] / s : 0.0;
                }
                fwht(d.data() + static_cast<std::size_t>(m * sh.n + k0), sh.group);
            }
        });
        dy = std::move(d);
    }
    const int iob = operand_at(call.rec, "out_bf16", 6);
    if (iob >= 0 && call.cot.size() > 2 && call.cot[2].value.size() == total)
        for (std::size_t i = 0; i < total; ++i) dy[i] += call.cot[2].value[i];

    // The gate's halves, gated_quant_fp8's split: the gate's slope, the up's activation alone.
    std::vector<double> dzg(total, 0.0), dzu(total, 0.0);
    parallel_for(static_cast<int64_t>(total), kElemChunk, [&](int64_t i) {
        dzg[static_cast<std::size_t>(i)] =
            dy[static_cast<std::size_t>(i)] * zu[static_cast<std::size_t>(i)] *
            act_slope_of(zg[static_cast<std::size_t>(i)], sigmoid);
        dzu[static_cast<std::size_t>(i)] = dy[static_cast<std::size_t>(i)] *
                                           act_of(zg[static_cast<std::size_t>(i)], sigmoid);
    });
    AdValue& dx = (*call.dc)[static_cast<std::size_t>(ia)];
    AdValue& dw = (*call.dc)[static_cast<std::size_t>(ib)];
    if (dx.value.size() < static_cast<std::size_t>(sh.M * sh.K) ||
        dw.value.size() < static_cast<std::size_t>(sh.N * sh.K))
        return;

    // The activation's cotangent, gemm_nt_q's, on both halves: the dequantised weight, and the
    // dequantised activation's own basis, its scale not applied again.
    parallel_for(sh.M, kRowChunk, [&](int64_t m) {
        for (int64_t k = 0; k < sh.K; ++k) {
            const int64_t g = k / sh.group;
            double acc = 0.0;
            for (int64_t j = 0; j < sh.n; ++j) {
                const std::size_t i = static_cast<std::size_t>(m * sh.n + j);
                acc += dzg[i] * b.value[static_cast<std::size_t>(j * sh.K + k)] *
                           scale_at(bs, j, g, sh.N, sh.groups_k) +
                       dzu[i] * b.value[static_cast<std::size_t>((sh.n + j) * sh.K + k)] *
                           scale_at(bs, sh.n + j, g, sh.N, sh.groups_k);
            }
            dx.value[static_cast<std::size_t>(m * sh.K + k)] += acc;
        }
    });

    // The weight's, in its codes, and the scale's, which the forward dequantised over: dW takes
    // the dequantised activation, dS the codes alone.
    std::vector<double> ds(static_cast<std::size_t>(sh.N * sh.groups_k), 0.0);
    // The weight's and the scale's accumulate over the activation's rows, so the pass splits over
    // the fold's output columns: a column's weight row belongs to exactly one column pair, and the
    // accumulation over the rows stays serial in the record's own order.
    parallel_for(sh.n, kPairChunk, [&](int64_t j) {
        for (int64_t m = 0; m < sh.M; ++m) {
            for (int64_t g = 0; g < sh.groups_k; ++g) {
                const int64_t k0 = g * sh.group;
                const int64_t k1 = std::min(sh.K, k0 + sh.group);
                const double sa = scale_at(as, m, g, sh.M, sh.groups_k);
                const std::size_t i = static_cast<std::size_t>(m * sh.n + j);
                for (int64_t k = k0; k < k1; ++k) {
                    const double av = a.value[static_cast<std::size_t>(m * sh.K + k)];
                    dw.value[static_cast<std::size_t>(j * sh.K + k)] += dzg[i] * av * sa;
                    dw.value[static_cast<std::size_t>((sh.n + j) * sh.K + k)] += dzu[i] * av * sa;
                    ds[static_cast<std::size_t>(j * sh.groups_k + g)] += dzg[i] * av;
                    ds[static_cast<std::size_t>((sh.n + j) * sh.groups_k + g)] += dzu[i] * av;
                }
            }
        }
    });
    // The scale's cotangent lands where the forward read the scale: the row-blocked plane the
    // kernel indexes by row block, or one of the whole planes scale_at accepts.
    if (ibs >= 0 && static_cast<std::size_t>(ibs) < call.dc->size()) {
        AdValue& dsc = (*call.dc)[static_cast<std::size_t>(ibs)];
        const int64_t ns = static_cast<int64_t>(dsc.value.size());
        if (ns == sh.N * sh.groups_k) {
            for (std::size_t i = 0; i < ds.size(); ++i) dsc.value[i] += ds[i];
        } else if (ns == sh.groups_k) {
            for (int64_t g = 0; g < sh.groups_k; ++g) {
                double acc = 0.0;
                for (int64_t j = 0; j < sh.N; ++j)
                    acc += ds[static_cast<std::size_t>(j * sh.groups_k + g)];
                dsc.value[static_cast<std::size_t>(g)] += acc;
            }
        } else if (ns == sh.N) {
            for (int64_t j = 0; j < sh.N; ++j) {
                double acc = 0.0;
                for (int64_t g = 0; g < sh.groups_k; ++g)
                    acc += ds[static_cast<std::size_t>(j * sh.groups_k + g)];
                dsc.value[static_cast<std::size_t>(j)] += acc;
            }
        } else if (ns == 1) {
            double acc = 0.0;
            for (const double x : ds) acc += x;
            dsc.value[0] += acc;
        }
        // A plane none of these names writes nothing: the scale's cotangent is dropped.
    }
}

}  // namespace

R97_REGISTER_TWIN("gemm_nt_q_gated", gemm_nt_q_gated_twin)
R97_REGISTER_ADJOINT_F("gemm_nt_q_gated", host, gemm_nt_q_gated_adjoint, 1)

}  // namespace r97
