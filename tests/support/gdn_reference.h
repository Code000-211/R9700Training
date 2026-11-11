// tests/support/gdn_reference.h -- the gated delta net recurrence, one token at a time.
//
// The plain loop of docs/adjoints/gdn-math.md: section 2.1 for the forward recurrence and
// section 4.4 for its reverse. Nothing here touches the chunked code in src/adjoint/host/:
// the state is carried token by token inside a chunk, and `g` is the record's within-chunk
// cumsum (its per-token decay is g[t] - g[t-1], or g[t] at the chunk's first token). The
// chunk length is read only to know where that reset happens, never to regroup the tokens.
//
// Layouts are the records': q, k are [T, Hg, K], v and o are [T, H, V], g and beta are
// [T, H], ht is [S, H, V, K], and value head h reads key head h / (H / Hg).
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace r97 {
namespace gdn_reference {

struct Geometry {
    int64_t key_heads = 0;    // Hg
    int64_t value_heads = 0;  // H
    int64_t head_k = 0;       // K
    int64_t head_v = 0;       // V
    int64_t chunk = 0;        // where the g cumsum resets
    double scale = 1.0;       // the query scale the read multiplies
};

struct Inputs {
    std::vector<double> q;     // [T, Hg, K]
    std::vector<double> k;     // [T, Hg, K]
    std::vector<double> v;     // [T, H, V]
    std::vector<double> g;     // [T, H] within-chunk cumsum
    std::vector<double> beta;  // [T, H]
    std::vector<int64_t> cu;   // [S + 1], cu[0] == 0
};

struct Forward {
    std::vector<double> o;      // [T, H, V]
    std::vector<double> ht;     // [S, H, V, K]
    std::vector<double> state;  // [T, H, K, V], the state after each token
};

struct Cotangent {
    std::vector<double> o;   // [T, H, V]
    std::vector<double> ht;  // [S, H, V, K]
};

struct Gradients {
    std::vector<double> q;     // [T, Hg, K]
    std::vector<double> k;     // [T, Hg, K]
    std::vector<double> v;     // [T, H, V]
    std::vector<double> g;     // [T, H]
    std::vector<double> beta;  // [T, H]
};

inline int64_t tokens(const Inputs& in, const Geometry& gm) {
    return gm.value_heads > 0 ? (int64_t)in.beta.size() / gm.value_heads : 0;
}

inline int64_t sequences(const Inputs& in) {
    return in.cu.empty() ? 0 : (int64_t)in.cu.size() - 1;
}

// The per-step log decay at token `t` of sequence `s`: the cumsum's difference inside
// the chunk, and the cumsum's own value at the chunk's first token.
inline double step_decay(const Inputs& in, const Geometry& gm, int64_t t, int64_t local,
                         int64_t h) {
    const double here = in.g[(size_t)(t * gm.value_heads + h)];
    if (local % gm.chunk == 0) return here;
    return here - in.g[(size_t)((t - 1) * gm.value_heads + h)];
}

inline Forward forward(const Geometry& gm, const Inputs& in) {
    const int64_t H = gm.value_heads, Hg = gm.key_heads;
    const int64_t K = gm.head_k, V = gm.head_v;
    const int64_t T = tokens(in, gm);
    const int64_t S = sequences(in);
    const int64_t grp = Hg > 0 ? H / Hg : 0;

    Forward f;
    f.o.assign((size_t)(T * H * V), 0.0);
    f.ht.assign((size_t)(S * H * V * K), 0.0);
    f.state.assign((size_t)(T * H * K * V), 0.0);
    if (grp == 0 || T == 0) return f;

    for (int64_t s = 0; s < S; ++s) {
        for (int64_t h = 0; h < H; ++h) {
            const int64_t hk = h / grp;
            std::vector<double> state((size_t)(K * V), 0.0);
            std::vector<double> x((size_t)(K * V), 0.0);
            std::vector<double> sk2((size_t)V, 0.0);
            std::vector<double> u((size_t)V, 0.0);
            for (int64_t t = in.cu[(size_t)s]; t < in.cu[(size_t)s + 1]; ++t) {
                const int64_t local = t - in.cu[(size_t)s];
                const double d = std::exp(step_decay(in, gm, t, local, h));
                const double beta = in.beta[(size_t)(t * H + h)];
                for (int64_t i = 0; i < K * V; ++i) x[(size_t)i] = d * state[(size_t)i];
                for (int64_t e = 0; e < V; ++e) {
                    double acc = 0.0;
                    for (int64_t di = 0; di < K; ++di)
                        acc += x[(size_t)(di * V + e)] *
                               in.k[(size_t)((t * Hg + hk) * K + di)];
                    sk2[(size_t)e] = acc;
                }
                for (int64_t e = 0; e < V; ++e)
                    u[(size_t)e] =
                        beta * (in.v[(size_t)((t * H + h) * V + e)] - sk2[(size_t)e]);
                for (int64_t di = 0; di < K; ++di) {
                    const double kd = in.k[(size_t)((t * Hg + hk) * K + di)];
                    for (int64_t e = 0; e < V; ++e)
                        state[(size_t)(di * V + e)] =
                            x[(size_t)(di * V + e)] + kd * u[(size_t)e];
                }
                for (int64_t di = 0; di < K; ++di) {
                    const double qd = in.q[(size_t)((t * Hg + hk) * K + di)];
                    for (int64_t e = 0; e < V; ++e)
                        f.o[(size_t)((t * H + h) * V + e)] +=
                            gm.scale * qd * state[(size_t)(di * V + e)];
                }
                for (int64_t i = 0; i < K * V; ++i)
                    f.state[(size_t)((t * H + h) * K * V + i)] = state[(size_t)i];
            }
            for (int64_t e = 0; e < V; ++e)
                for (int64_t di = 0; di < K; ++di)
                    f.ht[(size_t)(((s * H + h) * V + e) * K + di)] =
                        state[(size_t)(di * V + e)];
        }
    }
    return f;
}

// The reverse of `forward`, section 4.4: walk each sequence's tokens back to front,
// carry the state cotangent across the chunk's tokens, then fold the per-step decay
// cotangents through the in-chunk cumsum. The initial state's cotangent is dropped,
// as the read is frozen.
inline Gradients backward(const Geometry& gm, const Inputs& in, const Forward& f,
                          const Cotangent& cot) {
    const int64_t H = gm.value_heads, Hg = gm.key_heads;
    const int64_t K = gm.head_k, V = gm.head_v;
    const int64_t T = tokens(in, gm);
    const int64_t S = sequences(in);
    const int64_t grp = Hg > 0 ? H / Hg : 0;

    Gradients g;
    g.q.assign((size_t)(T * Hg * K), 0.0);
    g.k.assign((size_t)(T * Hg * K), 0.0);
    g.v.assign((size_t)(T * H * V), 0.0);
    g.g.assign((size_t)(T * H), 0.0);
    g.beta.assign((size_t)(T * H), 0.0);
    if (grp == 0 || T == 0) return g;

    std::vector<double> ds((size_t)(K * V), 0.0);
    std::vector<double> st((size_t)(K * V), 0.0);
    std::vector<double> st_prev((size_t)(K * V), 0.0);
    std::vector<double> x((size_t)(K * V), 0.0);
    std::vector<double> xt((size_t)(K * V), 0.0);
    std::vector<double> sk2((size_t)V, 0.0);
    std::vector<double> utilde((size_t)V, 0.0);
    std::vector<double> step_gamma;

    for (int64_t s = 0; s < S; ++s) {
        const int64_t first = in.cu[(size_t)s];
        const int64_t last = in.cu[(size_t)s + 1];
        const int64_t len = last - first;
        for (int64_t h = 0; h < H; ++h) {
            const int64_t hk = h / grp;
            std::fill(ds.begin(), ds.end(), 0.0);
            if (cot.ht.size() >= (size_t)(S * H * V * K))
                for (int64_t e = 0; e < V; ++e)
                    for (int64_t di = 0; di < K; ++di)
                        ds[(size_t)(di * V + e)] =
                            cot.ht[(size_t)(((s * H + h) * V + e) * K + di)];
            step_gamma.assign((size_t)len, 0.0);

            for (int64_t t = last - 1; t >= first; --t) {
                const int64_t local = t - first;
                const double d = std::exp(step_decay(in, gm, t, local, h));
                const double beta = in.beta[(size_t)(t * H + h)];

                // The forward decayed state X and the write's forward u.
                const bool has_prev = local > 0;
                for (int64_t i = 0; i < K * V; ++i) {
                    st_prev[(size_t)i] = 0.0;
                    x[(size_t)i] = 0.0;
                }
                if (has_prev)
                    for (int64_t i = 0; i < K * V; ++i)
                        st_prev[(size_t)i] =
                            f.state[(size_t)((((t - 1) * H + h) * K * V) + i)];
                for (int64_t i = 0; i < K * V; ++i) x[(size_t)i] = d * st_prev[(size_t)i];
                for (int64_t e = 0; e < V; ++e) {
                    double acc = 0.0;
                    for (int64_t di = 0; di < K; ++di)
                        acc += x[(size_t)(di * V + e)] *
                               in.k[(size_t)((t * Hg + hk) * K + di)];
                    sk2[(size_t)e] = acc;
                }

                // S~_t = Sst + scale q_t (x) oc_t, and dq reads the forward S_t.
                const double oc_base = (double)((t * H + h) * V);
                for (int64_t di = 0; di < K; ++di) {
                    const double qd = in.q[(size_t)((t * Hg + hk) * K + di)];
                    for (int64_t e = 0; e < V; ++e)
                        st[(size_t)(di * V + e)] =
                            ds[(size_t)(di * V + e)] + gm.scale * qd * cot.o[(size_t)(oc_base + e)];
                }
                for (int64_t di = 0; di < K; ++di) {
                    double acc = 0.0;
                    for (int64_t e = 0; e < V; ++e)
                        acc += f.state[(size_t)((((t * H + h) * K * V)) + di * V + e)] *
                               cot.o[(size_t)(oc_base + e)];
                    g.q[(size_t)((t * Hg + hk) * K + di)] += gm.scale * acc;
                }

                // u's cotangent, and the write's own v and beta.
                for (int64_t e = 0; e < V; ++e) {
                    double acc = 0.0;
                    for (int64_t di = 0; di < K; ++di)
                        acc += st[(size_t)(di * V + e)] *
                               in.k[(size_t)((t * Hg + hk) * K + di)];
                    utilde[(size_t)e] = acc;
                }
                for (int64_t e = 0; e < V; ++e) {
                    g.v[(size_t)((t * H + h) * V + e)] = beta * utilde[(size_t)e];
                    g.beta[(size_t)(t * H + h)] +=
                        utilde[(size_t)e] *
                        (in.v[(size_t)((t * H + h) * V + e)] - sk2[(size_t)e]);
                }

                // k reads the forward u and the decayed X's cotangent.
                for (int64_t di = 0; di < K; ++di) {
                    double a = 0.0, b = 0.0;
                    for (int64_t e = 0; e < V; ++e) {
                        const double u_fwd =
                            beta * (in.v[(size_t)((t * H + h) * V + e)] - sk2[(size_t)e]);
                        a += st[(size_t)(di * V + e)] * u_fwd;
                        b += x[(size_t)(di * V + e)] * utilde[(size_t)e];
                    }
                    g.k[(size_t)((t * Hg + hk) * K + di)] += a - beta * b;
                }

                // X~ and the per-step decay cotangent (which reads S~, not X~).
                double gamma = 0.0;
                for (int64_t di = 0; di < K; ++di) {
                    const double kd = in.k[(size_t)((t * Hg + hk) * K + di)];
                    for (int64_t e = 0; e < V; ++e) {
                        xt[(size_t)(di * V + e)] =
                            st[(size_t)(di * V + e)] - beta * kd * utilde[(size_t)e];
                        gamma += st[(size_t)(di * V + e)] * x[(size_t)(di * V + e)];
                    }
                }
                for (int64_t e = 0; e < V; ++e)
                    gamma -= beta * sk2[(size_t)e] * utilde[(size_t)e];
                step_gamma[(size_t)local] = gamma;
                for (int64_t i = 0; i < K * V; ++i) ds[(size_t)i] = d * xt[(size_t)i];
            }

            // Fold the in-chunk cumsum: g_t's cotangent is the step's minus the next
            // step's while both are in the same chunk.
            for (int64_t local = 0; local < len; ++local) {
                const int64_t t = first + local;
                double v = step_gamma[(size_t)local];
                if (local + 1 < len && (local + 1) % gm.chunk != 0)
                    v -= step_gamma[(size_t)(local + 1)];
                g.g[(size_t)(t * H + h)] = v;
            }
        }
    }
    return g;
}

}  // namespace gdn_reference
}  // namespace r97
