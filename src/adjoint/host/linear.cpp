// r97/adjoint/host/linear.cpp -- host twins and adjoints for the frozen linears.
//
// Every op here has a frozen weight: only the input cotangent is produced. The int8 linear's
// cotangent is in the CODES' (rotated) activation basis -- the basis the trunk's quantisers
// consume, each dividing the code cotangent by its own group scale -- so the adjoint dequantises
// the weight AND applies the activation's own group scale to the dequantised row.

#include "r97/adjoint.h"

#include "iqr_decode.h"  // r97_iqr_bf16_to_f32
#include "parallel.h"
#include "stored_weight.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace r97 {
namespace {

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}
// The pair of rows the parallel for cuts the M-row loops into: a one-row call (the fixture's
// narrow cases) stays on the calling thread, and an odd M leaves one row to a worker rather
// than a whole chunk. The row is the loops' output -- their inner sums stay the row's serial
// order -- so the partition writes nothing a neighbour reads.
constexpr int64_t kRowChunk = 2;

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

double param_double(const OpRecord& rec, const char* name, double fallback) {
    const OpParam* p = find_param(rec, name);
    if (p == nullptr) return fallback;
    if (const int64_t* v = std::get_if<int64_t>(&p->second)) return static_cast<double>(*v);
    try {
        return std::stod(std::get<std::string>(p->second));
    } catch (...) {
        return fallback;
    }
}

// The schema position of the operand a read or a write names.
int operand_pos(const OpRecord& rec, const char* name) {
    for (std::size_t i = 0; i < rec.operands.size(); ++i) {
        if (rec.operands[i].buffer == name) return static_cast<int>(i);
    }
    return -1;
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

// The shape of the first written operand, which is the shape a twin's result must have.
const Operand* first_write(const OpRecord& rec) {
    for (const Operand& o : rec.operands) {
        if (o.written) return &o;
    }
    return nullptr;
}

AdValue make_value(const std::string& dtype, const std::vector<int64_t>& shape) {
    AdValue v;
    v.dtype = dtype;
    v.shape = shape;
    v.value.assign(static_cast<std::size_t>(numel(shape)), 0.0);
    return v;
}

// A grouped scale for row `row` and group `g` of K. The int8 grid stores one scale per row and
// block of K; a whole-row and a whole-matrix scale are accepted as well.
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

// A stored operand: the tape gives its bytes behind `raw` in the engine's private layout and
// nothing in `value`. A device row decodes that layout from the pointer; a host row that indexed
// its empty value would read out of bounds, and a scale that answered 1.0 for it would be a
// silent wrong number. The row refuses it by name, as the device row declines a stored plane it
// cannot decode, so a stored weight never reaches this file's loops.
[[noreturn]] void refuse_stored(const OpRecord& rec, int pos) {
    throw std::runtime_error("r97: host adjoint: op '" + rec.op + "': operand '" +
                             rec.operands[static_cast<std::size_t>(pos)].buffer +
                             "' is stored in a plugin-private encoding the host row cannot read");
}

// The weight a frozen-linear row differentiates against: the f64 values the tape carried, or the
// stored form the join decoded for it (src/stored_weight.h) -- the int8 codes and the form's bf16
// scale, which is the pair the device row's kernel reads from its own planes, so the two sides of
// the A/B multiply the same numbers.
//
// `at` is the weight's value at (output row, input column) and `scale` its quantisation group's
// factor; a row whose weight the tape gave values for takes its scale from the operand beside it,
// exactly as before this existed.
struct LinearWeight {
    const std::vector<double>* value = nullptr;  // [rows, cols], when the tape carried values
    const StoredWeight* stored = nullptr;        // [rows, cols] codes + [rows, cols/128] scale
    const AdValue* scale_operand = nullptr;      // the values path's scale (a_scale/b_scale)
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t groups = 1;

    bool valid() const { return value != nullptr || stored != nullptr; }

    double at(int64_t n, int64_t k) const {
        if (stored != nullptr)
            return static_cast<double>(stored->codes[static_cast<std::size_t>(n * cols + k)]);
        return (*value)[static_cast<std::size_t>(n * cols + k)];
    }
    double group_scale(int64_t n, int64_t g) const {
        if (stored != nullptr)
            return static_cast<double>(
                r97_iqr_bf16_to_f32(stored->scale[static_cast<std::size_t>(n * groups + g)]));
        return scale_at(scale_operand, n, g, rows, groups);
    }
    // The weight's value: a stored code times its group's scale, or the value the tape carried.
    // The `_q` rows take the code and the scale separately -- they fold the activation's own group
    // scale into the same product -- so they use `at`/`group_scale`. A plain `gemm_nt` has no
    // activation scale to fold and must take the whole weight here: reading the code alone is a
    // silent factor of 1/scale on a stored operand.
    double value_at(int64_t n, int64_t k) const {
        if (stored == nullptr) return (*value)[static_cast<std::size_t>(n * cols + k)];
        const int64_t group = groups > 0 ? cols / groups : cols;
        return at(n, k) * group_scale(n, group > 0 ? k / group : 0);
    }
};

// Resolve one weight operand. A stored operand is looked up in the join's decode by the encoding
// the tape carried beside it; one the join never decoded is refused by name, as every stored
// operand was before the decode existed, so a stored weight never reaches this file's loops.
// `groups` is the scale grid the row reads, or 0 for the weight's own (one a 128 columns).
void stored_weight_of(const OpRecord& rec, int pos, const AdValue& v, int64_t rows, int64_t cols,
                      int64_t groups, const AdValue* scale_operand, LinearWeight* out) {
    out->rows = rows;
    out->cols = cols;
    out->groups = groups > 0 ? groups : (cols > 0 ? cols / 128 : 1);
    if (v.raw == nullptr) {
        // A values operand that does not hold the record's shape is not a weight this row can
        // index: it names itself rather than reading past the end.
        if (rows > 0 && cols > 0 && v.value.size() != static_cast<std::size_t>(rows * cols))
            throw std::runtime_error(
                "r97: host adjoint: op '" + rec.op + "': operand '" +
                rec.operands[static_cast<std::size_t>(pos)].buffer + "' holds " +
                std::to_string(v.value.size()) + " values for its " + std::to_string(rows) + " x " +
                std::to_string(cols) + " shape");
        out->value = &v.value;
        out->scale_operand = scale_operand;
        return;
    }
    const StoredWeight* w = stored_weight_find(v.enc, rows, cols);
    if (w == nullptr || w->groups != out->groups) refuse_stored(rec, pos);
    out->stored = w;
}

// y[M,N] = a[M,K] * b[N,K]^T, with the f32 accumulation of the host kernel done in f64.
void gemm_nt_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                  std::vector<AdValue>* out) {
    const AdValue* a = read_at(rec, operands, operand_pos(rec, "a"));
    const AdValue* b = read_at(rec, operands, operand_pos(rec, "b"));
    if (a == nullptr || b == nullptr) return;
    const int ia = operand_pos(rec, "a");
    if (a->raw != nullptr) refuse_stored(rec, ia);
    const int64_t K = param_int(rec, "K", a->shape.size() >= 2 ? a->shape[1] : 0);
    const int64_t N = param_int(rec, "N", b->shape.size() >= 2 ? b->shape[0] : 0);
    // `M` is the weight's capacity; the operand's own first dimension is how many rows this call
    // actually has, and the one a tightened record carries.
    const int64_t M = a->shape.empty() ? param_int(rec, "M", 0) : a->shape[0];
    LinearWeight w;
    stored_weight_of(rec, operand_pos(rec, "b"), *b, N, K, 0, nullptr, &w);
    const Operand* wo = first_write(rec);
    out->push_back(make_value(wo != nullptr ? wo->dtype : a->dtype, {M, N}));
    AdValue& y = out->back();
    parallel_for(M, kRowChunk, [&](int64_t m) {
        for (int64_t n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                acc += a->value[static_cast<std::size_t>(m * K + k)] * w.value_at(n, k);
            }
            y.value[static_cast<std::size_t>(m * N + n)] = acc;
        }
    });
}

void gemm_nt_adjoint(const AdjointCall& call) {
    const int ia = operand_pos(call.rec, "a");
    const int ib = operand_pos(call.rec, "b");
    if (ia < 0 || ib < 0 || call.cot.empty()) return;
    const AdValue& a = call.fwd[static_cast<std::size_t>(ia)];
    const AdValue& b = call.fwd[static_cast<std::size_t>(ib)];
    const AdValue& dy = call.cot[0];
    const int64_t K = param_int(call.rec, "K", a.shape.size() >= 2 ? a.shape[1] : 0);
    const int64_t N = param_int(call.rec, "N", b.shape.size() >= 2 ? b.shape[0] : 0);
    const int64_t M = param_int(call.rec, "M", a.shape.empty() ? 0 : a.shape[0]);
    LinearWeight w;
    stored_weight_of(call.rec, ib, b, N, K, 0, nullptr, &w);
    AdValue& dx = (*call.dc)[static_cast<std::size_t>(ia)];
    parallel_for(M, kRowChunk, [&](int64_t m) {
        for (int64_t k = 0; k < K; ++k) {
            double acc = 0.0;
            for (int64_t n = 0; n < N; ++n) {
                acc += dy.value[static_cast<std::size_t>(m * N + n)] * w.value_at(n, k);
            }
            dx.value[static_cast<std::size_t>(m * K + k)] += acc;
        }
    });
}

// y[M,N] = (a codes * a_scale) * (b codes * b_scale)^T, grouped as the host kernel groups.
void gemm_nt_q_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                    std::vector<AdValue>* out) {
    const AdValue* a = read_at(rec, operands, operand_pos(rec, "a"));
    const AdValue* b = read_at(rec, operands, operand_pos(rec, "b"));
    const AdValue* as = read_at(rec, operands, operand_pos(rec, "a_scale"));
    const AdValue* bs = read_at(rec, operands, operand_pos(rec, "b_scale"));
    if (a == nullptr || b == nullptr) return;
    if (a->raw != nullptr) refuse_stored(rec, operand_pos(rec, "a"));
    if (as != nullptr && as->raw != nullptr) refuse_stored(rec, operand_pos(rec, "a_scale"));
    if (bs != nullptr && bs->raw != nullptr && b->raw == nullptr)
        refuse_stored(rec, operand_pos(rec, "b_scale"));
    const int64_t K = param_int(rec, "K", a->shape.size() >= 2 ? a->shape[1] : 0);
    const int64_t N = param_int(rec, "N", b->shape.size() >= 2 ? b->shape[0] : 0);
    const int64_t M = a->shape.empty() ? param_int(rec, "M", 0) : a->shape[0];
    const int64_t group = param_int(rec, "group", K > 0 ? K : 1);
    const int64_t groups = group > 0 ? (K + group - 1) / group : 1;
    LinearWeight w;
    stored_weight_of(rec, operand_pos(rec, "b"), *b, N, K, groups, bs, &w);
    const Operand* wo = first_write(rec);
    out->push_back(make_value(wo != nullptr ? wo->dtype : a->dtype, {M, N}));
    AdValue& y = out->back();
    parallel_for(M, kRowChunk, [&](int64_t m) {
        for (int64_t n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int64_t g = 0; g < groups; ++g) {
                const int64_t k0 = g * group;
                const int64_t k1 = std::min(K, k0 + group);
                double sub = 0.0;
                for (int64_t k = k0; k < k1; ++k) {
                    sub += a->value[static_cast<std::size_t>(m * K + k)] * w.at(n, k);
                }
                acc += sub * scale_at(as, m, g, M, groups) * w.group_scale(n, g);
            }
            y.value[static_cast<std::size_t>(m * N + n)] = acc;
        }
    });
}

// dx in the codes' activation basis: the dequantised product dy * W, times the activation's group
// scale s_a -- the cotangent a downstream quantiser consumes (gemm_nt_q_twin applies s_a to `a`).
void gemm_nt_q_adjoint(const AdjointCall& call) {
    const int ia = operand_pos(call.rec, "a");
    const int ib = operand_pos(call.rec, "b");
    const int ias = operand_pos(call.rec, "a_scale");
    const int ibs = operand_pos(call.rec, "b_scale");
    if (ia < 0 || ib < 0 || call.cot.empty()) return;
    const AdValue& a = call.fwd[static_cast<std::size_t>(ia)];
    const AdValue& b = call.fwd[static_cast<std::size_t>(ib)];
    const AdValue* as = ias >= 0 ? &call.fwd[static_cast<std::size_t>(ias)] : nullptr;
    const AdValue* bs = ibs >= 0 ? &call.fwd[static_cast<std::size_t>(ibs)] : nullptr;
    if (bs != nullptr && bs->raw != nullptr && b.raw == nullptr) refuse_stored(call.rec, ibs);
    const AdValue& dy = call.cot[0];
    const int64_t K = param_int(call.rec, "K", a.shape.size() >= 2 ? a.shape[1] : 0);
    const int64_t N = param_int(call.rec, "N", b.shape.size() >= 2 ? b.shape[0] : 0);
    const int64_t M = a.shape.empty() ? param_int(call.rec, "M", 0) : a.shape[0];
    const int64_t group = param_int(call.rec, "group", K > 0 ? K : 1);
    const int64_t groups = group > 0 ? (K + group - 1) / group : 1;
    LinearWeight w;
    stored_weight_of(call.rec, ib, b, N, K, groups, bs, &w);
    AdValue& dx = (*call.dc)[static_cast<std::size_t>(ia)];
    parallel_for(M, kRowChunk, [&](int64_t m) {
        for (int64_t k = 0; k < K; ++k) {
            const int64_t g = group > 0 ? k / group : 0;
            double acc = 0.0;
            for (int64_t n = 0; n < N; ++n) {
                acc += dy.value[static_cast<std::size_t>(m * N + n)] * w.at(n, k) *
                       w.group_scale(n, g);
            }
            dx.value[static_cast<std::size_t>(m * K + k)] += acc * scale_at(as, m, g, M, groups);
        }
    });
}

// logits_gemm: the same transposed product, with the head's own parameter names.
void logits_gemm_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                      std::vector<AdValue>* out) {
    const AdValue* x = read_at(rec, operands, operand_pos(rec, "x"));
    const AdValue* w = read_at(rec, operands, operand_pos(rec, "lm_head"));
    if (x == nullptr || w == nullptr) return;
    if (x->raw != nullptr) refuse_stored(rec, operand_pos(rec, "x"));
    const int64_t K = param_int(rec, "n_embd", x->shape.size() >= 2 ? x->shape[1] : 0);
    const int64_t N = param_int(rec, "n_vocab", w->shape.size() >= 2 ? w->shape[0] : 0);
    const int64_t M = x->shape.empty() ? 0 : x->shape[0];
    LinearWeight hw;
    stored_weight_of(rec, operand_pos(rec, "lm_head"), *w, N, K, 0, nullptr, &hw);
    const Operand* y = first_write(rec);
    out->push_back(make_value(y != nullptr ? y->dtype : x->dtype, {M, N}));
    AdValue& logits = out->back();
    parallel_for(M, kRowChunk, [&](int64_t m) {
        for (int64_t n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                acc += x->value[static_cast<std::size_t>(m * K + k)] * hw.value_at(n, k);
            }
            logits.value[static_cast<std::size_t>(m * N + n)] = acc;
        }
    });
}

void logits_gemm_adjoint(const AdjointCall& call) {
    const int ix = operand_pos(call.rec, "x");
    const int iw = operand_pos(call.rec, "lm_head");
    if (ix < 0 || iw < 0 || call.cot.empty()) return;
    const AdValue& x = call.fwd[static_cast<std::size_t>(ix)];
    const AdValue& w = call.fwd[static_cast<std::size_t>(iw)];
    const AdValue& dy = call.cot[0];
    const int64_t K = param_int(call.rec, "n_embd", x.shape.size() >= 2 ? x.shape[1] : 0);
    const int64_t N = param_int(call.rec, "n_vocab", w.shape.size() >= 2 ? w.shape[0] : 0);
    const int64_t M = x.shape.empty() ? 0 : x.shape[0];
    LinearWeight hw;
    stored_weight_of(call.rec, iw, w, N, K, 0, nullptr, &hw);
    AdValue& dx = (*call.dc)[static_cast<std::size_t>(ix)];
    parallel_for(M, kRowChunk, [&](int64_t m) {
        for (int64_t k = 0; k < K; ++k) {
            double acc = 0.0;
            for (int64_t n = 0; n < N; ++n) {
                acc += dy.value[static_cast<std::size_t>(m * N + n)] * hw.value_at(n, k);
            }
            dx.value[static_cast<std::size_t>(m * K + k)] += acc;
        }
    });
}

// gather_rows: y[i,:] = x[idx[i],:] and a negative index selects a zero row.
void gather_rows_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                      std::vector<AdValue>* out) {
    const AdValue* x = read_at(rec, operands, operand_pos(rec, "x"));
    const AdValue* idx = read_at(rec, operands, operand_pos(rec, "idx"));
    if (x == nullptr || idx == nullptr) return;
    if (x->raw != nullptr) refuse_stored(rec, operand_pos(rec, "x"));
    const int64_t width = param_int(rec, "n", x->shape.size() >= 2 ? x->shape[1] : 1);
    const Operand* y = first_write(rec);
    const int64_t rows = y != nullptr && !y->shape.empty() ? y->shape[0] : static_cast<int64_t>(idx->value.size());
    out->push_back(make_value(y != nullptr ? y->dtype : x->dtype, {rows, width}));
    AdValue& dst = out->back();
    const int64_t source_rows = x->shape.empty() ? 0 : x->shape[0];
    for (int64_t i = 0; i < rows; ++i) {
        const int64_t r = static_cast<int64_t>(idx->value[static_cast<std::size_t>(i)]);
        if (r < 0 || r >= source_rows) continue;
        for (int64_t j = 0; j < width; ++j) {
            dst.value[static_cast<std::size_t>(i * width + j)] =
                x->value[static_cast<std::size_t>(r * width + j)];
        }
    }
}

void gather_rows_adjoint(const AdjointCall& call) {
    const int ix = operand_pos(call.rec, "x");
    const int ii = operand_pos(call.rec, "idx");
    if (ix < 0 || ii < 0 || call.cot.empty()) return;
    const AdValue& idx = call.fwd[static_cast<std::size_t>(ii)];
    const AdValue& dy = call.cot[0];
    const int64_t width = param_int(call.rec, "n", dy.shape.size() >= 2 ? dy.shape[1] : 1);
    const AdValue& x = call.fwd[static_cast<std::size_t>(ix)];
    const int64_t source_rows = x.shape.empty() ? 0 : x.shape[0];
    const int64_t rows = dy.shape.empty() ? static_cast<int64_t>(idx.value.size()) : dy.shape[0];
    AdValue& dx = (*call.dc)[static_cast<std::size_t>(ix)];
    for (int64_t i = 0; i < rows; ++i) {
        const int64_t r = static_cast<int64_t>(idx.value[static_cast<std::size_t>(i)]);
        if (r < 0 || r >= source_rows) continue;
        for (int64_t j = 0; j < width; ++j) {
            dx.value[static_cast<std::size_t>(r * width + j)] +=
                dy.value[static_cast<std::size_t>(i * width + j)];
        }
    }
}

// embed_lookup: tokens select rows of the (frozen) embedding table. It is a stop, so its adjoint
// writes nothing: neither the table nor the discrete token ids have a cotangent.
void embed_lookup_twin(const OpRecord& rec, const std::vector<AdValue>& operands,
                       std::vector<AdValue>* out) {
    const AdValue* tokens = read_at(rec, operands, operand_pos(rec, "tokens"));
    const AdValue* table = read_at(rec, operands, operand_pos(rec, "wte"));
    if (tokens == nullptr || table == nullptr) return;
    if (table->raw != nullptr) refuse_stored(rec, operand_pos(rec, "wte"));
    const int64_t width = param_int(rec, "n_embd", table->shape.size() >= 2 ? table->shape[1] : 1);
    const int64_t vocab = param_int(rec, "n_vocab", table->shape.size() >= 2 ? table->shape[0] : 0);
    const int64_t offset = param_int(rec, "vocab_offset", 0);
    const double mult = param_double(rec, "mult", 1.0);
    const Operand* x = first_write(rec);
    const int64_t rows = static_cast<int64_t>(tokens->value.size());
    out->push_back(make_value(x != nullptr ? x->dtype : table->dtype, {rows, width}));
    AdValue& dst = out->back();
    for (int64_t i = 0; i < rows; ++i) {
        const int64_t id = static_cast<int64_t>(tokens->value[static_cast<std::size_t>(i)]);
        const int64_t r = id - offset;
        if (r < 0 || r >= vocab) continue;
        for (int64_t j = 0; j < width; ++j) {
            dst.value[static_cast<std::size_t>(i * width + j)] =
                mult * table->value[static_cast<std::size_t>(r * width + j)];
        }
    }
}

void embed_lookup_adjoint(const AdjointCall&) {}

}  // namespace

R97_REGISTER_TWIN("gemm_nt", gemm_nt_twin)
R97_REGISTER_ADJOINT_F("gemm_nt", host, gemm_nt_adjoint, 1)

R97_REGISTER_TWIN("gemm_nt_q", gemm_nt_q_twin)
R97_REGISTER_ADJOINT_F("gemm_nt_q", host, gemm_nt_q_adjoint, 1, 2, 3)

R97_REGISTER_TWIN("logits_gemm", logits_gemm_twin)
R97_REGISTER_ADJOINT_F("logits_gemm", host, logits_gemm_adjoint, 1)

R97_REGISTER_TWIN("gather_rows", gather_rows_twin)
R97_REGISTER_ADJOINT_F("gather_rows", host, gather_rows_adjoint, 1)

R97_REGISTER_TWIN("embed_lookup", embed_lookup_twin)
R97_REGISTER_ADJOINT_F("embed_lookup", host, embed_lookup_adjoint, 0, 1)

}  // namespace r97
