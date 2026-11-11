// src/adjoint/host/gdn_kkt.cpp -- `gdn_kkt_solve`'s f64 twin and host adjoint: the delta rule's
// in-chunk solve, taken as an op on its own operands.
//
// Schema (third_party/radiance/libref/ref_registry.cpp): reads k, beta, g, cu_seqlens; writes A.
// Per sequence, per value head, per chunk of `chunk` tokens, with hk = h / grp the key head that
// value head shares (grp = heads / key heads):
//
//     kc[i][d] = k[bos + c0 + i][hk][d]      bc[i] = beta[bos + c0 + i][h]
//     gc[i]    = g[bos + c0 + i][h]          -- the log decay, already cumsummed inside the chunk
//     M[i][l]  = bc[i] * <kc[i], kc[l]> * e^{gc[i] - gc[l]}         (l < i; zero elsewhere)
//     A        = (I + M)^{-1}, unit lower triangular, by forward substitution
//
// Each token's row of the inverse is written at A[t][h][0..chunk), zero above the diagonal, so the
// column j addresses the position inside the token's own chunk and not a column of a matrix shared
// with the rest of the sequence. The last chunk of a sequence is L = len - c0 < chunk tokens long:
// the forward writes its L rows and leaves the rows past L to whoever zeroed the buffer -- they are
// the next sequence's tokens -- and the adjoint reads the L rows back, nothing else.
//
// Operand dtypes are the ones the device row carries (k and A bf16, beta and g fp32), but the twin
// computes in f64 either way. cu_seqlens addresses rows and is frozen: a different integer there is
// a different op, not a small step.
//
// Every formula, both derivations, and why the backward reads A rather than forming it again are in
// docs/adjoints/gdn_kkt.md.

#include "r97/adjoint.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace r97 {
namespace {

/* A parameter as an integer: the record carries an integer, or a string that is one. */
int64_t param_int(const OpRecord& rec, const char* name, int64_t fallback) {
    for (const OpParam& p : rec.params) {
        if (p.first != name) continue;
        if (const auto* i = std::get_if<int64_t>(&p.second)) return *i;
        const std::string& s = std::get<std::string>(p.second);
        char* end = nullptr;
        const long long v = std::strtoll(s.c_str(), &end, 10);
        if (end && end != s.c_str() && *end == '\0') return v;
        return fallback;
    }
    return fallback;
}

/* The read operands' slots, in schema order. */
enum Slot : size_t {
    kK = 0,
    kBeta = 1,
    kG = 2,
    kCu = 3,
    kA = 4,
};

size_t read_count(const OpRecord& rec) {
    size_t n = 0;
    for (const Operand& o : rec.operands)
        if (!o.written) ++n;
    return n;
}

/* The first written operand's dtype: the twin's output is what the record says the op writes. */
const char* write_dtype(const OpRecord& rec) {
    for (const Operand& o : rec.operands)
        if (o.written) return o.dtype.c_str();
    return "f32";
}

/* A's last dimension as the record declares it: `chunk` is RAD_DERIVED and a record may leave it to
 * the shape, as the reference kernel does. */
int64_t write_chunk(const OpRecord& rec) {
    for (const Operand& o : rec.operands)
        if (o.written && o.shape.size() == 3) return o.shape.back();
    return 0;
}

/* One call's geometry, resolved once and shared by the twin and the adjoint so that the two cannot
 * disagree about which rows a chunk holds or which key head a value head reads. */
struct Call {
    const AdValue* k = nullptr;
    const AdValue* beta = nullptr;
    const AdValue* g = nullptr;
    const AdValue* A = nullptr;  // the forward's output; null for the twin

    int64_t K = 0;      // lanes of a key
    int64_t chunk = 0;  // tokens per chunk, A's last dimension
    int64_t n_seq = 0;
    int64_t rows = 0;   // tokens over all sequences, cu_seqlens[n_seq]
    int64_t heads = 0;  // value heads
    int64_t k_heads = 0;
    int64_t grp = 1;    // value heads per key head

    std::vector<int64_t> cu;  // cu_seqlens[n_seq + 1]

    double k_at(int64_t tok, int64_t hk, int64_t d) const {
        return k->value[(size_t)((tok * k_heads + hk) * K + d)];
    }
    double beta_at(int64_t tok, int64_t h) const {
        return beta->value[(size_t)(tok * heads + h)];
    }
    double g_at(int64_t tok, int64_t h) const { return g->value[(size_t)(tok * heads + h)]; }
    double A_at(int64_t tok, int64_t h, int64_t j) const {
        return A->value[(size_t)((tok * heads + h) * chunk + j)];
    }

    /* The length of sequence s, and its first token. */
    int64_t seq_len(int64_t s) const { return cu[(size_t)s + 1] - cu[(size_t)s]; }
    int64_t seq_bos(int64_t s) const { return cu[(size_t)s]; }
};

/* Resolve a record's geometry. `chunk_fallback` is A's last dimension, for a record that leaves the
 * `chunk` parameter to the shape as RAD_DERIVED lets it: the adjoint has it in the forward's
 * output, the twin in the write the record declares. False when the call is not one this op can
 * compute, so the twin writes no output and the adjoint adds no cotangent -- neither throws. */
bool resolve(const OpRecord& rec, const std::vector<AdValue>& fwd, int64_t chunk_fallback,
             Call* c) {
    const size_t n_reads = read_count(rec);
    if (n_reads < 4 || fwd.size() < 4) return false;
    c->k = &fwd[kK];
    c->beta = &fwd[kBeta];
    c->g = &fwd[kG];
    if (fwd.size() > kA) c->A = &fwd[kA];

    /* k is [rows][key head][lanes]; beta and g are [rows][value head]. */
    if (c->k->shape.size() != 3 || c->beta->shape.size() != 2) return false;
    c->rows = c->k->shape[0];
    c->k_heads = c->k->shape[1];
    c->K = c->k->shape[2];
    c->heads = c->beta->shape[1];
    if (c->rows <= 0 || c->k_heads <= 0 || c->K <= 0 || c->heads <= 0) return false;
    if (c->heads % c->k_heads != 0) return false;
    c->grp = c->heads / c->k_heads;
    if (c->k->value.size() != (size_t)(c->rows * c->k_heads * c->K)) return false;
    if (c->beta->value.size() != (size_t)(c->rows * c->heads)) return false;
    if (c->g->shape != c->beta->shape) return false;
    if (c->g->value.size() != c->beta->value.size()) return false;

    /* head_k is a parameter the record spells out; when it does, it must say the same. M is a
     * parameter too, but the op addresses rows through cu_seqlens and never reads it. */
    if (param_int(rec, "head_k", c->K) != c->K) return false;

    c->chunk = param_int(rec, "chunk", chunk_fallback);
    if (c->chunk <= 0) return false;
    if (c->A && c->A->shape.size() == 3 && c->A->shape[2] != c->chunk) return false;

    /* cu_seqlens holds one more bound than there are sequences, starts at zero, ends at the row
     * count, and never goes back. A zero-length sequence is a sequence with no chunks. */
    const AdValue& cu = fwd[kCu];
    if (cu.value.size() < 2) return false;
    c->n_seq = (int64_t)cu.value.size() - 1;
    c->cu.resize((size_t)c->n_seq + 1);
    for (int64_t s = 0; s <= c->n_seq; ++s) {
        const double v = cu.value[(size_t)s];
        if (!(std::fabs(v) <= 1e15)) return false;
        c->cu[(size_t)s] = (int64_t)std::llround(v);
    }
    if (c->cu[0] != 0 || c->cu[(size_t)c->n_seq] != c->rows) return false;
    for (int64_t s = 0; s < c->n_seq; ++s)
        if (c->cu[(size_t)s + 1] < c->cu[(size_t)s]) return false;
    return true;
}

/* The per-chunk workspace: the keys and gates of one chunk, and the matrices a chunk of length
 * `chunk` can need. One allocation per call, reused by every (sequence, head, chunk). */
struct Workspace {
    std::vector<double> kc;  // [chunk][K]
    std::vector<double> bc;  // [chunk]
    std::vector<double> gc;  // [chunk]
    std::vector<double> M;   // [chunk][chunk], the strict lower block
    std::vector<double> Y;   // [chunk][chunk], the transposed solve's intermediate
    std::vector<double> A;   // [chunk][chunk], the chunk's inverse, as the forward wrote it
    std::vector<double> s;   // [chunk]

    explicit Workspace(const Call& c)
        : kc((size_t)(c.chunk * c.K)),
          bc((size_t)c.chunk),
          gc((size_t)c.chunk),
          M((size_t)(c.chunk * c.chunk)),
          Y((size_t)(c.chunk * c.chunk)),
          A((size_t)(c.chunk * c.chunk)),
          s((size_t)c.chunk) {}
};

/* One chunk's keys, gates and strict lower block into the workspace, and the chunk's length. */
int64_t load_chunk(const Call& c, int64_t s, int64_t h, int64_t c0, Workspace* w) {
    const int64_t bos = c.seq_bos(s);
    const int64_t slen = c.seq_len(s);
    const int64_t L = std::min(c.chunk, slen - c0);
    const int64_t hk = h / c.grp;
    for (int64_t j = 0; j < L; ++j) {
        const int64_t tok = bos + c0 + j;
        w->bc[(size_t)j] = c.beta_at(tok, h);
        w->gc[(size_t)j] = c.g_at(tok, h);
        for (int64_t d = 0; d < c.K; ++d) w->kc[(size_t)(j * c.K + d)] = c.k_at(tok, hk, d);
    }
    for (int64_t i = 0; i < L; ++i) {
        const double* ki = &w->kc[(size_t)(i * c.K)];
        for (int64_t l = 0; l < i; ++l) {
            const double* kl = &w->kc[(size_t)(l * c.K)];
            double dot = 0.0;
            for (int64_t d = 0; d < c.K; ++d) dot += ki[d] * kl[d];
            w->M[(size_t)(i * c.chunk + l)] =
                w->bc[(size_t)i] * dot * std::exp(w->gc[(size_t)i] - w->gc[(size_t)l]);
        }
    }
    return L;
}

}  // namespace

/* The forward in f64: per chunk, the strict lower block M, then its unit lower triangular inverse
 * by forward substitution, A[i][j] = delta_ij - sum_{l<i} M[i][l] A[l][j]. Rows past the last
 * chunk's length and columns above the diagonal stay zero, as the kernel leaves them. */
void gdn_kkt_twin(const OpRecord& rec, const std::vector<AdValue>& fwd, std::vector<AdValue>* out) {
    Call c;
    if (!resolve(rec, fwd, write_chunk(rec), &c)) return;

    AdValue A;
    A.dtype = write_dtype(rec);
    A.shape = {c.rows, c.heads, c.chunk};
    A.value.assign((size_t)(c.rows * c.heads * c.chunk), 0.0);

    Workspace w(c);
    for (int64_t s = 0; s < c.n_seq; ++s) {
        const int64_t bos = c.seq_bos(s);
        const int64_t slen = c.seq_len(s);
        for (int64_t h = 0; h < c.heads; ++h) {
            for (int64_t c0 = 0; c0 < slen; c0 += c.chunk) {
                const int64_t L = load_chunk(c, s, h, c0, &w);
                for (int64_t i = 0; i < L; ++i) {
                    double* row = &w.M[(size_t)(i * c.chunk)];
                    for (int64_t j = 0; j <= i; ++j) w.s[(size_t)j] = (i == j) ? 1.0 : 0.0;
                    for (int64_t l = 0; l < i; ++l) {
                        const double rl = row[l];
                        if (rl == 0.0) continue;
                        const double* al = &w.A[(size_t)(l * c.chunk)];
                        for (int64_t j = 0; j <= l; ++j) w.s[(size_t)j] -= rl * al[j];
                    }
                    double* ai = &w.A[(size_t)(i * c.chunk)];
                    for (int64_t j = 0; j <= i; ++j) ai[j] = w.s[(size_t)j];
                    const size_t base = (size_t)(((bos + c0 + i) * c.heads + h) * c.chunk);
                    for (int64_t j = 0; j <= i; ++j) A.value[base + (size_t)j] = w.s[(size_t)j];
                }
            }
        }
    }
    out->push_back(std::move(A));
}

/* The adjoint. With E = I + M and A = E^{-1}, dA = -A dE A, so the cotangent of the strict lower
 * block is Et = -(A^T Ac A^T) there, read off the forward's A as one transposed triangular solve
 * against another:
 *
 *     Y[p][l]  = sum_{j<=l} Ac[p][j] A[l][j]        (l <= p,  cost L^3/6)
 *     Et[i][l] = -sum_{p>=i} A[p][i] Y[p][l]        (l < i,   cost L^3/6)
 *
 * which is the same order of work as the forward's substitution. From Et, with
 * c_ij = <k_i, k_j> e^{g_i - g_j} and M[i][j] = beta_i c_ij:
 *
 *     dbeta[i]   += Et[i][j] c_ij
 *     dg[i]      += Et[i][j] phi_ij beta_i        dg[j] -= Et[i][j] phi_ij beta_i
 *     dk[i][d]   += Et[i][j] phi_ij beta_i k_j[d] dk[j][d] += Et[i][j] phi_ij beta_i k_i[d]
 *
 * where phi_ij = e^{g_i - g_j}. cu_seqlens is frozen: this adds no cotangent for it. */
void gdn_kkt_host_adjoint(const AdjointCall& call) {
    if (call.cot.empty() || !call.dc || call.dc->size() < 4) return;
    const AdValue& Ac = call.cot[0];
    Call c;
    const int64_t chunk_fallback =
        call.fwd.size() > kA && call.fwd[kA].shape.size() == 3 ? call.fwd[kA].shape.back() : 0;
    if (!resolve(call.rec, call.fwd, chunk_fallback, &c)) return;
    if (!c.A || c.A->shape.size() != 3) return;
    if (Ac.value.size() != c.A->value.size()) return;

    AdValue& dk = call.dc->at(kK);
    AdValue& dbeta = call.dc->at(kBeta);
    AdValue& dg = call.dc->at(kG);
    if (dk.value.size() != c.k->value.size()) return;
    if (dbeta.value.size() != c.beta->value.size()) return;
    if (dg.value.size() != c.g->value.size()) return;

    Workspace w(c);
    for (int64_t s = 0; s < c.n_seq; ++s) {
        const int64_t bos = c.seq_bos(s);
        const int64_t slen = c.seq_len(s);
        for (int64_t h = 0; h < c.heads; ++h) {
            const int64_t hk = h / c.grp;
            for (int64_t c0 = 0; c0 < slen; c0 += c.chunk) {
                const int64_t L = load_chunk(c, s, h, c0, &w);
                if (L <= 0) continue;
                /* Recompute the forward's A from k, beta and g. The snapshot the walk hands an
                 * adjoint for a written operand is the value the buffer held before the write,
                 * so `c.A` cannot be read as the output here; the twin's substitution is
                 * deterministic and this is the same arithmetic. */
                for (int64_t i = 0; i < L; ++i) {
                    const double* row = &w.M[(size_t)(i * c.chunk)];
                    for (int64_t j = 0; j <= i; ++j) w.s[(size_t)j] = (i == j) ? 1.0 : 0.0;
                    for (int64_t l = 0; l < i; ++l) {
                        const double rl = row[l];
                        if (rl == 0.0) continue;
                        const double* al = &w.A[(size_t)(l * c.chunk)];
                        for (int64_t j = 0; j <= l; ++j) w.s[(size_t)j] -= rl * al[j];
                    }
                    double* ai = &w.A[(size_t)(i * c.chunk)];
                    for (int64_t j = 0; j <= i; ++j) ai[j] = w.s[(size_t)j];
                }

                /* Y[p][l] = sum_{j<=l} Ac[p][j] A[l][j]: the cotangent's rows against the
                 * forward's A. */
                for (int64_t p = 0; p < L; ++p) {
                    const double* ac = &Ac.value[(size_t)(((bos + c0 + p) * c.heads + h) * c.chunk)];
                    for (int64_t l = 0; l <= p; ++l) {
                        double acc = 0.0;
                        const double* al = &w.A[(size_t)(l * c.chunk)];
                        for (int64_t j = 0; j <= l; ++j) acc += ac[j] * al[j];
                        w.Y[(size_t)(p * c.chunk + l)] = acc;
                    }
                }

                for (int64_t i = 0; i < L; ++i) {
                    const int64_t tok_i = bos + c0 + i;
                    const double bi = w.bc[(size_t)i];
                    const double gi = w.gc[(size_t)i];
                    double* dki = &dk.value[(size_t)((tok_i * c.k_heads + hk) * c.K)];
                    const double* ki = &w.kc[(size_t)(i * c.K)];
                    double beta_acc = 0.0;
                    for (int64_t l = 0; l < i; ++l) {
                        /* Et[i][l] = -sum_{p>=i} A[p][i] Y[p][l], one column of A and one row
                         * of Y at a time. */
                        double et = 0.0;
                        for (int64_t p = i; p < L; ++p)
                            et -= w.A[(size_t)(p * c.chunk + i)] * w.Y[(size_t)(p * c.chunk + l)];
                        if (et == 0.0) continue;
                        const double phi = std::exp(gi - w.gc[(size_t)l]);
                        const double* kl = &w.kc[(size_t)(l * c.K)];
                        double dot = 0.0;
                        for (int64_t d = 0; d < c.K; ++d) dot += ki[d] * kl[d];
                        const double cij = dot * phi;
                        const double m = bi * cij;  // M[i][l], the forward's own block entry
                        beta_acc += et * cij;
                        dg.value[(size_t)(tok_i * c.heads + h)] += et * m;
                        dg.value[(size_t)((bos + c0 + l) * c.heads + h)] -= et * m;
                        const double coeff = et * bi * phi;
                        double* dkl = &dk.value[(size_t)(((bos + c0 + l) * c.k_heads + hk) * c.K)];
                        for (int64_t d = 0; d < c.K; ++d) {
                            dki[d] += coeff * kl[d];
                            dkl[d] += coeff * ki[d];
                        }
                    }
                    dbeta.value[(size_t)(tok_i * c.heads + h)] += beta_acc;
                }
            }
        }
    }
}

R97_REGISTER_TWIN("gdn_kkt_solve", gdn_kkt_twin)
// Frozen, counted in schema order: cu_seqlens 3. It addresses rows, and a different integer there
// is a different op, not a small step.
R97_REGISTER_ADJOINT_F("gdn_kkt_solve", host, gdn_kkt_host_adjoint, 3)

}  // namespace r97
