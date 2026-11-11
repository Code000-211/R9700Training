// src/adjoint/device/gdn_kkt.hip -- the device adjoint of the gated delta net's in-chunk solve.
//
// The op is `gdn_kkt_solve`: per (sequence, value head, chunk) it builds the strict lower block
// M[i][j] = beta_i <k_i, k_j> e^{g_i - g_j}, inverts I + M by forward substitution, and writes
// A = (I + M)^{-1} to the tape. The host adjoint in src/adjoint/host/gdn_kkt.cpp undoes that with
// one transposed triangular solve against another, read off the forward's own A:
//
//     Y[p][l]  = sum_{j<=l} Ac[p][j] A[l][j]        (l <= p)
//     Et[i][l] = -sum_{p>=i} A[p][i] Y[p][l]        (l < i)
//
// and from Et, with phi_ij = e^{g_i - g_j} and c_ij = <k_i, k_j> phi_ij,
//
//     dbeta[i] += Et[i][j] c_ij
//     dg[i]    += Et[i][j] phi_ij beta_i          dg[j] -= Et[i][j] phi_ij beta_i
//     dk[i][d] += Et[i][j] phi_ij beta_i k_j[d]   dk[j][d] += Et[i][j] phi_ij beta_i k_i[d]
//
// This kernel runs exactly those three steps per chunk. It solves in f32: the forward's A and the
// cotangent reach the card at f32, Y and Et are formed and held in f32, and the whole sum over
// the chunk's 64 rows is f32. The one rounding is on the way out, to the dtype the record gives
// each cotangent (bf16 for dk, f32 for dbeta and dg). docs/kernels/gdn_kkt.md carries the error
// this leaves against the f64 host adjoint over a 4096-token sequence, beside the same solve
// rounding its intermediates to bf16, which does not meet the tolerance.
//
// One block owns one (sequence, key head, chunk) and loops over the value heads that share the
// key head, so a key row's cotangent is accumulated once in shared memory and stored once, with
// no atomics: a block's key rows are its own, and dk is 2048 of the 2144 gradient values a token
// carries at the model's geometry. Y/Et and the key rows live in shared memory; the forward's A
// and the cotangent are read from the card's memory, which holds one chunk of one head in cache
// for the life of the block.
#include "r97/adjoint.h"

#include "scratch.h"

#include "rad_device.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <variant>
#include <vector>

namespace r97 {
namespace {

// bf16 is the high half of an f32, rounded to nearest even on the way down; stored as unsigned
// short so the host side of this file converts the same way the kernel does.
__host__ __device__ inline float bf16_to_f32(unsigned short b) {
    const unsigned int u = (unsigned int)b << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

__host__ __device__ inline unsigned short f32_to_bf16(float f) {
    unsigned int u;
    std::memcpy(&u, &f, sizeof u);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x0040u);  // NaN
    const unsigned int lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (unsigned short)(u >> 16);
}

// Where a kernel stores its cotangent: one value per element, f32 or bf16, rounded once here.
struct OutStore {
    void* p;
    int wide;  // 1 when the slot is f32, 0 when it is bf16

    __host__ __device__ void set(int64_t i, float v) const {
        if (wide) {
            static_cast<float*>(p)[i] = v;
        } else {
            static_cast<unsigned short*>(p)[i] = f32_to_bf16(v);
        }
    }
};

/* ------------------------------------------------------------------ the record */

constexpr size_t kK = 0, kBeta = 1, kG = 2, kCu = 3, kA = 4;

int64_t param_int(const OpRecord& rec, const char* name, int64_t fallback) {
    for (const OpParam& p : rec.params) {
        if (p.first != name) continue;
        if (const int64_t* i = std::get_if<int64_t>(&p.second)) return *i;
        const std::string& s = std::get<std::string>(p.second);
        char* end = nullptr;
        const long long v = std::strtoll(s.c_str(), &end, 10);
        return (end && end != s.c_str() && *end == '\0') ? (int64_t)v : fallback;
    }
    return fallback;
}

size_t read_count(const OpRecord& rec) {
    size_t n = 0;
    for (const Operand& o : rec.operands)
        if (!o.written) ++n;
    return n;
}

// The geometry the host adjoint resolves, from the same record and the twin's f64 values.
struct Call {
    const AdValue* k = nullptr;
    const AdValue* beta = nullptr;
    const AdValue* g = nullptr;
    const AdValue* A = nullptr;
    int64_t rows = 0, k_heads = 0, K = 0, heads = 0, grp = 0, chunk = 0, n_seq = 0;
    std::vector<int32_t> cu;

    int64_t seq_bos(int64_t s) const { return cu[(size_t)s]; }
    int64_t seq_len(int64_t s) const { return cu[(size_t)s + 1] - cu[(size_t)s]; }
};

bool resolve(const OpRecord& rec, const std::vector<AdValue>& fwd, int64_t chunk_fallback,
             Call* c) {
    if (read_count(rec) < 4 || fwd.size() < 4) return false;
    c->k = &fwd[kK];
    c->beta = &fwd[kBeta];
    c->g = &fwd[kG];
    if (fwd.size() > kA) c->A = &fwd[kA];

    if (c->k->shape.size() != 3 || c->beta->shape.size() != 2) return false;
    c->rows = c->k->shape[0];
    c->k_heads = c->k->shape[1];
    c->K = c->k->shape[2];
    c->heads = c->beta->shape[1];
    if (c->rows <= 0 || c->k_heads <= 0 || c->K <= 0 || c->heads <= 0) return false;
    if (c->heads % c->k_heads != 0) return false;
    c->grp = c->heads / c->k_heads;
    if ((int64_t)c->k->value.size() != c->rows * c->k_heads * c->K) return false;
    if ((int64_t)c->beta->value.size() != c->rows * c->heads) return false;
    if (c->g->shape != c->beta->shape) return false;
    if (c->g->value.size() != c->beta->value.size()) return false;

    if (param_int(rec, "head_k", c->K) != c->K) return false;
    c->chunk = param_int(rec, "chunk", chunk_fallback);
    if (c->chunk <= 0) return false;
    if (c->A && c->A->shape.size() == 3 && c->A->shape[2] != c->chunk) return false;

    const AdValue& cu = fwd[kCu];
    if (cu.value.size() < 2) return false;
    c->n_seq = (int64_t)cu.value.size() - 1;
    c->cu.resize((size_t)c->n_seq + 1);
    for (int64_t s = 0; s <= c->n_seq; ++s) {
        const double v = cu.value[(size_t)s];
        if (!(std::fabs(v) <= 1e15)) return false;
        c->cu[(size_t)s] = (int32_t)std::llround(v);
    }
    if (c->cu[0] != 0 || c->cu[(size_t)c->n_seq] != c->rows) return false;
    for (int64_t s = 0; s < c->n_seq; ++s)
        if (c->cu[(size_t)s + 1] < c->cu[(size_t)s]) return false;
    return true;
}

/* ------------------------------------------------------------------ the card */

constexpr int64_t f32_bytes(int64_t n) { return n * (int64_t)sizeof(float); }

int64_t slot_bytes(const AdValue& slot, int64_t n) {
    return n * (int64_t)(slot.dtype == "f32" ? sizeof(float) : sizeof(unsigned short));
}

OutStore store_in(const AdValue& slot, void* p) {
    return OutStore{p, slot.dtype == "f32" ? 1 : 0};
}

RadStream adjoint_stream() {
    static RadStream s = [] {
        RadStream created = nullptr;
        if (rad_stream_create(&created, 0) != 0) return (RadStream) nullptr;
        return created;
    }();
    return s;
}

bool upload(const AdValue& v, void* d, int64_t n, RadStream s) {
    if (n <= 0) return true;
    std::vector<float> h((size_t)n);
    for (int64_t i = 0; i < n; ++i) h[(size_t)i] = (float)v.value[(size_t)i];
    return rad_memcpy_async(d, h.data(), f32_bytes(n), s) == 0;
}

bool upload_ints(const std::vector<int32_t>& h, void* d, RadStream s) {
    if (h.empty()) return true;
    return rad_memcpy_async(d, h.data(), (int64_t)(h.size() * sizeof(int32_t)), s) == 0;
}

// Adds a kernel's result back into the tape: the contract adds, never overwrites, because several
// ops share the buffer. The call's one rounding happened in the store.
bool download_add(AdValue& slot, const void* d, int64_t n, RadStream s) {
    if (n <= 0) return true;
    if (slot.dtype == "f32") {
        std::vector<float> h((size_t)n);
        if (rad_memcpy_async(h.data(), d, f32_bytes(n), s) != 0) return false;
        if (rad_stream_sync(s) != 0) return false;
        for (int64_t i = 0; i < n; ++i) slot.value[(size_t)i] += (double)h[(size_t)i];
        return true;
    }
    std::vector<unsigned short> h((size_t)n);
    const int64_t bytes = n * (int64_t)sizeof(unsigned short);
    if (rad_memcpy_async(h.data(), d, bytes, s) != 0) return false;
    if (rad_stream_sync(s) != 0) return false;
    for (int64_t i = 0; i < n; ++i) slot.value[(size_t)i] += (double)bf16_to_f32(h[(size_t)i]);
    return true;
}

/* ------------------------------------------------------------------ the kernel */

/* One block per (sequence, key head, chunk). `chunk_prefix` is the number of chunks every earlier
 * sequence contributed, so block b is chunk (b / k_heads) and key head (b % k_heads); the block
 * walks the value heads hk*grp .. hk*grp+grp-1 that share that key head.
 *
 * Shared memory, in floats:
 *   Et       chunk * chunk   the transposed solve, Y overwritten in place by Et
 *   dks      chunk * K       the chunk's key cotangent, accumulated over the value heads
 *   bc, gc   chunk           this value head's beta and gates
 *   dga, dba chunk           this value head's dg and dbeta, then stored
 *   coeff    chunk           Et[i][l] * beta_i * phi_il, the pair's weight on k
 *   cij      chunk           Et[i][l] * <k_i, k_l> phi_il, the pair's weight on beta
 *   dgm      chunk           beta_i times cij, the pair's two weights on g
 *   krow     K               k_i, read once per row for the pair loop and the dk update
 */
constexpr int64_t kThreads = 128;

__global__ void gdn_kkt_adjoint_kernel(const float* __restrict__ k, const float* __restrict__ beta,
                                       const float* __restrict__ g, const float* __restrict__ A,
                                       const float* __restrict__ Ac, const int32_t* __restrict__ cu,
                                       const int32_t* __restrict__ chunk_prefix,
                                       OutStore dk_store, float* __restrict__ dbeta_out,
                                       float* __restrict__ dg_out, int64_t heads, int64_t k_heads,
                                       int64_t K, int64_t chunk, int64_t grp, int64_t n_seq) {
    extern __shared__ float smem[];
    float* Et = smem;
    float* dks = Et + chunk * chunk;
    float* bc = dks + chunk * K;
    float* gc = bc + chunk;
    float* dga = gc + chunk;
    float* dba = dga + chunk;
    float* coeff = dba + chunk;
    float* cij = coeff + chunk;
    float* dgm = cij + chunk;
    float* krow = dgm + chunk;

    const int64_t tid = threadIdx.x;
    const int64_t hk = blockIdx.x % k_heads;
    const int64_t cix = blockIdx.x / k_heads;
    int64_t s = 0;
    while (s + 1 < n_seq && (int64_t)chunk_prefix[s + 1] <= cix) ++s;
    const int64_t bos = cu[s];
    const int64_t c0 = (cix - (int64_t)chunk_prefix[s]) * chunk;
    const int64_t rest = (int64_t)cu[s + 1] - bos - c0;
    const int64_t L = rest < chunk ? rest : chunk;
    if (L <= 0) return;

    for (int64_t j = tid; j < L * K; j += kThreads) dks[j] = 0.0f;

    const int64_t h0 = hk * grp;
    for (int64_t h = h0; h < h0 + grp; ++h) {
        __syncthreads();  // the last row of the previous head's pairs has been read
        for (int64_t j = tid; j < L; j += kThreads) {
            const int64_t tok = bos + c0 + j;
            bc[j] = beta[tok * heads + h];
            gc[j] = g[tok * heads + h];
            dga[j] = 0.0f;
            dba[j] = 0.0f;
        }
        __syncthreads();

        // Y[p][l] = sum_{j<=l} Ac[p][j] A[l][j], into Et; a row at a time, one thread per l.
        for (int64_t p = 0; p < L; ++p) {
            const float* ac = &Ac[((bos + c0 + p) * heads + h) * chunk];
            for (int64_t l = tid; l <= p; l += kThreads) {
                const float* al = &A[((bos + c0 + l) * heads + h) * chunk];
                float acc = 0.0f;
                for (int64_t j = 0; j <= l; ++j) acc += ac[j] * al[j];
                Et[p * chunk + l] = acc;
            }
        }
        __syncthreads();

        // Et[i][l] = -sum_{p>=i} A[p][i] Y[p][l], written over Y row by row; ascending i means
        // row i is never read again once its Et has been written, and a thread owns a column.
        for (int64_t i = 0; i < L; ++i) {
            for (int64_t l = tid; l < i; l += kThreads) {
                float et = 0.0f;
                for (int64_t p = i; p < L; ++p)
                    et -= A[((bos + c0 + p) * heads + h) * chunk + i] * Et[p * chunk + l];
                Et[i * chunk + l] = et;
            }
        }
        __syncthreads();

        for (int64_t i = 0; i < L; ++i) {
            __syncthreads();  // krow and the pair rows are about to be rewritten
            for (int64_t d = tid; d < K; d += kThreads)
                krow[d] = k[((bos + c0 + i) * k_heads + hk) * K + d];
            __syncthreads();

            const float bi = bc[i];
            const float gi = gc[i];
            for (int64_t l = tid; l < i; l += kThreads) {
                const float* kl = &k[((bos + c0 + l) * k_heads + hk) * K];
                float dot = 0.0f;
                for (int64_t d = 0; d < K; ++d) dot += krow[d] * kl[d];
                const float et = Et[i * chunk + l];
                const float phi = __expf(gi - gc[l]);
                coeff[l] = et * bi * phi;
                cij[l] = et * dot * phi;
                dgm[l] = et * bi * dot * phi;
            }
            __syncthreads();

            // dk[i][d] += coeff_l k_l[d] and dk[l][d] += coeff_l k_i[d]: one thread per lane,
            // its own column of the chunk's key cotangent, so no two threads write one value.
            for (int64_t d = tid; d < K; d += kThreads) {
                const float ki = krow[d];
                float acc = 0.0f;
                for (int64_t l = 0; l < i; ++l) {
                    const float cf = coeff[l];
                    if (cf == 0.0f) continue;
                    acc += cf * k[((bos + c0 + l) * k_heads + hk) * K + d];
                }
                dks[i * K + d] += acc;
                for (int64_t l = 0; l < i; ++l) {
                    const float cf = coeff[l];
                    if (cf != 0.0f) dks[l * K + d] += cf * ki;
                }
            }
            if (tid == 0) {
                for (int64_t l = 0; l < i; ++l) {
                    dba[i] += cij[l];
                    dga[i] += dgm[l];
                    dga[l] -= dgm[l];
                }
            }
            __syncthreads();
        }

        for (int64_t j = tid; j < L; j += kThreads) {
            const int64_t tok = bos + c0 + j;
            dbeta_out[tok * heads + h] = dba[j];
            dg_out[tok * heads + h] = dga[j];
        }
    }

    __syncthreads();
    for (int64_t j = tid; j < L * K; j += kThreads) {
        const int64_t i = j / K;
        const int64_t d = j - i * K;
        dk_store.set(((bos + c0 + i) * k_heads + hk) * K + d, dks[j]);
    }
}

/* The adjoint's whole call: upload the twin's f64 operands at f32, run one block per
 * (sequence, key head, chunk), add the three cotangents back. Every buffer is scratch and freed
 * on the way out. No chunk shape this solve can hold in shared memory -- one chunk squared plus
 * one chunk of keys -- means no call: the op is left to a later, differently blocked kernel. */
void gdn_kkt_device_adjoint(const AdjointCall& call) {
    if (call.cot.empty() || !call.dc || call.dc->size() < 4) return;
    const AdValue& Ac = call.cot[0];
    Call c;
    const int64_t chunk_fallback =
        call.fwd.size() > kA && call.fwd[kA].shape.size() == 3 ? call.fwd[kA].shape.back() : 0;
    if (!resolve(call.rec, call.fwd, chunk_fallback, &c)) return;
    if (!c.A || c.A->shape.size() != 3) return;
    if ((int64_t)Ac.value.size() != c.A->value.size()) return;

    AdValue& dk = call.dc->at(kK);
    AdValue& dbeta = call.dc->at(kBeta);
    AdValue& dg = call.dc->at(kG);
    if ((int64_t)dk.value.size() != c.rows * c.k_heads * c.K) return;
    if ((int64_t)dbeta.value.size() != c.rows * c.heads) return;
    if ((int64_t)dg.value.size() != c.rows * c.heads) return;

    const int64_t smem_floats = c.chunk * c.chunk + c.chunk * c.K + 7 * c.chunk + c.K;
    const int64_t smem_bytes = smem_floats * (int64_t)sizeof(float);
    if (smem_bytes > 64 * 1024) return;

    std::vector<int32_t> prefix((size_t)c.n_seq + 1, 0);
    for (int64_t s = 0; s < c.n_seq; ++s) {
        const int64_t len = c.seq_len(s);
        prefix[(size_t)s + 1] = prefix[(size_t)s] + (int32_t)((len + c.chunk - 1) / c.chunk);
    }
    const int64_t total_blocks = (int64_t)prefix[(size_t)c.n_seq] * c.k_heads;
    if (total_blocks <= 0) return;

    RadStream stream = adjoint_stream();
    if (!stream) return;

    const int64_t n_k = c.rows * c.k_heads * c.K;
    const int64_t n_bg = c.rows * c.heads;
    const int64_t n_a = c.rows * c.heads * c.chunk;
    device::ScratchBlock d_k(f32_bytes(n_k));
    device::ScratchBlock d_beta(f32_bytes(n_bg));
    device::ScratchBlock d_g(f32_bytes(n_bg));
    device::ScratchBlock d_A(f32_bytes(n_a));
    device::ScratchBlock d_Ac(f32_bytes(n_a));
    device::ScratchBlock d_cu((int64_t)prefix.size() * (int64_t)sizeof(int32_t));
    device::ScratchBlock d_prefix((int64_t)prefix.size() * (int64_t)sizeof(int32_t));
    device::ScratchBlock d_dk(slot_bytes(dk, n_k));
    device::ScratchBlock d_dbeta(slot_bytes(dbeta, n_bg));
    device::ScratchBlock d_dg(slot_bytes(dg, n_bg));
    if (!d_k || !d_beta || !d_g || !d_A || !d_Ac || !d_cu || !d_prefix || !d_dk || !d_dbeta ||
        !d_dg)
        return;

    if (!upload(*c.k, d_k.p, n_k, stream)) return;
    if (!upload(*c.beta, d_beta.p, n_bg, stream)) return;
    if (!upload(*c.g, d_g.p, n_bg, stream)) return;
    if (!upload(*c.A, d_A.p, n_a, stream)) return;
    if (!upload(Ac, d_Ac.p, n_a, stream)) return;
    if (!upload_ints(c.cu, d_cu.p, stream)) return;
    if (!upload_ints(prefix, d_prefix.p, stream)) return;

    // The solve holds a chunk's transposed matrix and its keys in shared memory, 50.25 KiB at the
    // model's chunk of 64 and 128 lanes; that is past the default opt-in, so ask for it.
    (void)hipFuncSetAttribute((const void*)gdn_kkt_adjoint_kernel,
                              hipFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);

    hipLaunchKernelGGL(gdn_kkt_adjoint_kernel, dim3((unsigned)total_blocks),
                       dim3((unsigned)kThreads), (size_t)smem_bytes, (hipStream_t)stream,
                       (const float*)d_k.p,
                       (const float*)d_beta.p, (const float*)d_g.p, (const float*)d_A.p,
                       (const float*)d_Ac.p, (const int32_t*)d_cu.p, (const int32_t*)d_prefix.p,
                       store_in(dk, d_dk.p), (float*)d_dbeta.p, (float*)d_dg.p, c.heads, c.k_heads,
                       c.K, c.chunk, c.grp, c.n_seq);
    if (hipGetLastError() != hipSuccess) return;

    if (!download_add(dk, d_dk.p, n_k, stream)) return;
    if (!download_add(dbeta, d_dbeta.p, n_bg, stream)) return;
    download_add(dg, d_dg.p, n_bg, stream);
}

}  // namespace

R97_REGISTER_ADJOINT_F("gdn_kkt_solve", device, gdn_kkt_device_adjoint, 3)

}  // namespace r97
