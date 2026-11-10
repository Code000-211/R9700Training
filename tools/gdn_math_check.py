#!/usr/bin/env python3
"""gdn-math check: the gated delta net at toy size, one head.

Plain Python, no dependencies (docs/adjoints/gdn-math.md, issue #5).  For one
(key head, value head) pair it implements:

  1. the per-token recurrence,
  2. the chunked form radiance evaluates (gdn_kkt_solve + gdn_chunk_scan, per chunk),
  3. the adjoint of gdn_kkt_solve as an op on its operands (k, beta, g -> A),
  4. the adjoint of gdn_chunk_scan as an op on its operands
     (q, k, v, A, g, beta, h0 -> o, ht),
  5. the reverse per-token recursion, the reference the adjoints must equal,

and prints four checks:

  chunked == per-token
  kkt adjoint  == central finite differences
  scan adjoint == central finite differences (one chunk)
  token-reverse recursion == chunk-reverse adjoint (three chunks)

Each to 1e-8 relative (or 1e-12 absolute where finite differencing cannot
resolve the value).  float64 throughout.

State layout follows the reference: S[d][e], d over head_k, e over head_v, so
o[e] = sum_d S[d][e] q[d] and the rank-one update is S += k x u^T.
"""

import math
import random

K, V, CHUNK = 6, 4, 5
T = 12                                  # three chunks of 5, 5, 2
SCALE = 0.37                            # the query scale, the 1/sqrt(head_k) analogue

REL, ABS = 1.0e-8, 1.0e-12


# ---------------------------------------------------------------- inputs
def make_inputs():
    rnd = random.Random(7)
    q = [[rnd.uniform(-1, 1) for _ in range(K)] for _ in range(T)]
    k = [[rnd.uniform(-1, 1) for _ in range(K)] for _ in range(T)]
    v = [[rnd.uniform(-1, 1) for _ in range(V)] for _ in range(T)]
    beta = [rnd.uniform(0.2, 0.9) for _ in range(T)]
    gamma = [-abs(rnd.uniform(0.01, 0.6)) for _ in range(T)]
    gc = []                              # per-chunk cumsum, resets at each chunk
    for n in range(0, T, CHUNK):
        run = 0.0
        for t in range(n, min(n + CHUNK, T)):
            run += gamma[t]
            gc.append(run)
    h0 = [[rnd.uniform(-0.5, 0.5) for _ in range(V)] for _ in range(K)]
    return q, k, v, beta, gc, h0


def chunk_bounds():
    return [(n, min(n + CHUNK, T)) for n in range(0, T, CHUNK)]


def rnd_cot(rnd, rows, cols, lo=0.5):
    return [[rnd.uniform(-lo, lo) for _ in range(cols)] for _ in range(rows)]


# ---------------------------------------------------------------- per-token form
def forward_token(q, k, v, beta, gc, h0, want_states=False):
    """The per-token recurrence.  The prefill ops take g as the in-chunk cumsum,
    so the decay applied at token t is the per-step gate gamma_t = g_t - g_{t-1}
    within a chunk, gamma_t = g_t at a chunk start; the state carries across a
    chunk boundary with no extra decay.  (The decode op gdn_recurrent_update
    instead takes g as the per-step log decay directly.)

      S_t = e^{gamma_t} S_{t-1} + beta_t (v_t - (e^{gamma_t} S_{t-1})^T k_t) k_t^T
      o_t = scale * S_t^T q_t

    Decay first, then the rank-one correction, exactly ref_gdn_recurrent_update."""
    S = [row[:] for row in h0]
    o, states = [], []
    T_ = len(k)
    for t in range(T_):
        gam = gc[t] - (gc[t - 1] if t > 0 and t % CHUNK != 0 else 0.0)
        D = math.exp(gam)
        Sk = [[D * S[d][e] for e in range(V)] for d in range(K)]
        sk2 = [sum(Sk[d][e] * k[t][d] for d in range(K)) for e in range(V)]
        u = [beta[t] * (v[t][e] - sk2[e]) for e in range(V)]
        S = [[Sk[d][e] + k[t][d] * u[e] for e in range(V)] for d in range(K)]
        if want_states:
            states.append(S)
        o.append([SCALE * sum(S[d][e] * q[t][d] for d in range(K)) for e in range(V)])
    return (o, S, states) if want_states else (o, S)


# ---------------------------------------------------------------- gdn_kkt_solve
def kkt_forward(kc, bc, gc):
    """A = (I + strict_lower(diag(beta) K K^T e^{g_i-g_j}))^{-1}, unit lower
    triangular, per ref_gdn_kkt_solve: M[i][j] = beta_i (k_i.k_j) e^{g_i-g_j}
    (j < i), rows solved in order by forward substitution, A[i][j] = 0 for j > i."""
    L = len(kc)
    A = [[0.0] * L for _ in range(L)]
    for i in range(L):
        mrow = [0.0] * i                      # M[i][l], kept: A overwrites its slot
        for l in range(i):
            mrow[l] = sum(kc[i][d] * kc[l][d] for d in range(K)) * bc[i] \
                * math.exp(gc[i] - gc[l])
        for j in range(i + 1):              # A[i][j] = delta_ij - sum_{l<i} M[i][l] A[l][j]
            s = 1.0 if j == i else 0.0
            for l in range(i):
                s -= mrow[l] * A[l][j]
            A[i][j] = s
    return A


def kkt_adjoint(Ac, kc, bc, gc, A=None):
    """Adjoint of gdn_kkt_solve.  E = I + M, A = E^{-1}, dA = -A dE A, so the
    cotangent of E is Et = -(A^T Ac A^T) restricted to the strict lower triangle:
    dE's diagonal is zero (E's diagonal is the constant 1 and no input reaches
    it), and dA[i][i] = 0 identically.  With M[i][j] = beta_i c_ij,
    c_ij = (k_i.k_j) e^{g_i-g_j}:

      beta_i : db[i]    = sum_{j<i} Et[i][j] c_ij
      k_i    : dk[i][d] = sum_{j<i} Et[i][j] beta_i e^{g_i-g_j} k_j[d]
      k_j    : dk[j][d]+= sum_{i>j} Et[i][j] beta_i e^{g_i-g_j} k_i[d]
      g_i    : dg[i]   += sum_{j<i} Et[i][j] beta_i c_ij
      g_j    : dg[j]   -= sum_{i>j} Et[i][j] beta_i c_ij
    """
    L = len(kc)
    if A is None:
        A = kkt_forward(kc, bc, gc)
    dk = [[0.0] * K for _ in range(L)]
    db = [0.0] * L
    dg = [0.0] * L
    for i in range(L):
        for j in range(i):
            # Et[i][j] = -sum_{p>=i, m<=min(j,p)} A[p][i] Ac[p][m] A[j][m]
            s = 0.0
            for p in range(i, L):
                if A[p][i] == 0.0:
                    continue
                for m in range(0, min(j, p) + 1):
                    if Ac[p][m] != 0.0 and A[j][m] != 0.0:
                        s += A[p][i] * Ac[p][m] * A[j][m]
            Et = -s
            c = sum(kc[i][d] * kc[j][d] for d in range(K)) * math.exp(gc[i] - gc[j])
            phi = math.exp(gc[i] - gc[j])
            db[i] += Et * c
            dg[i] += Et * bc[i] * c
            dg[j] -= Et * bc[i] * c
            w = Et * bc[i] * phi
            for d in range(K):
                dk[i][d] += w * kc[j][d]
                dk[j][d] += w * kc[i][d]
    return dk, db, dg


# ---------------------------------------------------------------- one chunk of gdn_chunk_scan
def scan_pieces(S0, kc, vc, bc, gc, Ac):
    """W and V' of one chunk."""
    L = len(kc)
    W, Vp = [], []
    for j in range(L):
        egj = math.exp(gc[j])
        sk = [sum(S0[d][e] * kc[j][d] for d in range(K)) for e in range(V)]
        W.append([bc[j] * (vc[j][e] - egj * sk[e]) for e in range(V)])
    for i in range(L):
        vp = [0.0] * V
        for j in range(i + 1):
            for e in range(V):
                vp[e] += Ac[i][j] * W[j][e]
        Vp.append(vp)
    return W, Vp


def scan_chunk(S0, qc, kc, vc, bc, gc, Ac):
    """One chunk of ref_gdn_chunk_scan:
      W[j]  = beta_j (v_j - e^{g_j} (S0^T k_j))                      [V]
      V'[i] = sum_{j<=i} A[i][j] W[j]                                [V], A unit lower tri
      o[i]  = scale e^{g_i} (S0^T q_i)
            + sum_{j<=i} scale e^{g_i-g_j} (q_i.k_j) V'[j]
      S1    = e^{g_{L-1}} S0 + sum_j (e^{g_{L-1}-g_j} k_j) V'[j]^T
    Returns o (L x V) and S1 (K x V)."""
    L = len(kc)
    W, Vp = scan_pieces(S0, kc, vc, bc, gc, Ac)
    o = []
    for i in range(L):
        egi = math.exp(gc[i])
        row = []
        for e in range(V):
            acc = SCALE * egi * sum(S0[d][e] * qc[i][d] for d in range(K))
            for j in range(i + 1):
                qk = sum(qc[i][d] * kc[j][d] for d in range(K))
                acc += SCALE * math.exp(gc[i] - gc[j]) * qk * Vp[j][e]
            row.append(acc)
        o.append(row)
    gl = gc[L - 1]
    S1 = [[math.exp(gl) * S0[d][e] for e in range(V)] for d in range(K)]
    for j in range(L):
        gs = math.exp(gl - gc[j])
        for d in range(K):
            for e in range(V):
                S1[d][e] += gs * kc[j][d] * Vp[j][e]
    return o, S1


def forward_chunked(q, k, v, beta, gc, h0):
    S = [row[:] for row in h0]
    o = []
    for (n, m) in chunk_bounds():
        Ac = kkt_forward(k[n:m], beta[n:m], gc[n:m])
        oc, S = scan_chunk(S, q[n:m], k[n:m], v[n:m], beta[n:m], gc[n:m], Ac)
        o.extend(oc)
    return o, S


def state_at(q, k, v, beta, gc, h0, n):
    """The forward state at token n (chunk starts), recomputed from h0."""
    if n == 0:
        return [row[:] for row in h0]
    _, S, _ = forward_token(q[:n], k[:n], v[:n], beta[:n], gc[:n], h0,
                            want_states=True)
    return S


# ---------------------------------------------------------------- scan adjoint
def scan_adjoint(oc, St1, S0, qc, kc, vc, bc, gc, Ac, Vp=None, W=None):
    """Adjoint of one chunk of gdn_chunk_scan over its operands q, k, v, A, g,
    beta, h0, given the cotangents of o (oc) and of the output state (St1).
    Backprop in four stages against the forward's pieces:

      S1 = e^{gL} S0 + sum_j (e^{gL-g_j} k_j) V'[j]^T : dS0, dk, dg (both roles)
      o[i] = scale[e^{gi}(S0^T q_i) + sum_{j<=i} e^{gi-gj}(q_i.k_j) V'[j]] :
             dq, dk, dVp, dg (state term and sA term), dS0
      V' = A W : dA[i][j] = Vp'[i].W[j] (j<=i); dW = A^T Vp'
      W[j] = beta_j (v_j - e^{g_j} (S0^T k_j)) : dbeta, dv, dk, dg, dS0
    """
    L = len(kc)
    if Vp is None or W is None:
        W, Vp = scan_pieces(S0, kc, vc, bc, gc, Ac)
    dq = [[0.0] * K for _ in range(L)]
    dk = [[0.0] * K for _ in range(L)]
    dv = [[0.0] * V for _ in range(L)]
    db = [0.0] * L
    dg = [0.0] * L
    dAc = [[0.0] * L for _ in range(L)]
    dS0 = [[0.0] * V for _ in range(K)]

    # ---- stage 1: S1
    gl = gc[L - 1]
    el = math.exp(gl)
    for d in range(K):
        for e in range(V):
            dS0[d][e] += el * St1[d][e]
            dg[L - 1] += St1[d][e] * el * S0[d][e]
    for j in range(L):
        gs = math.exp(gl - gc[j])
        gw = 0.0                             # dg contribution of e^{gL-g_j}
        for d in range(K):
            w = gs * sum(St1[d][e] * Vp[j][e] for e in range(V))
            dk[j][d] += w
            gw += w * kc[j][d]
        dg[j] -= gw                          # d/dg_j of e^{gL-g_j} is -itself
        dg[L - 1] += gw                      # d/dgL of e^{gL-g_j} is +itself

    # ---- stage 2: o
    dVp = [[0.0] * V for _ in range(L)]
    # the S1 stage's Vp contribution, per j, e: gs * sum_d St1[d][e] kc[j][d]
    for j in range(L):
        gs = math.exp(gl - gc[j])
        for e in range(V):
            dVp[j][e] += gs * sum(St1[d][e] * kc[j][d] for d in range(K))
    for i in range(L):
        egi = math.exp(gc[i])
        S0oi = [sum(S0[d][e] * oc[i][e] for e in range(V)) for d in range(K)]
        for d in range(K):
            dq[i][d] += SCALE * egi * S0oi[d]
            for e in range(V):
                dS0[d][e] += SCALE * egi * qc[i][d] * oc[i][e]
        dg[i] += SCALE * egi * sum(qc[i][d] * S0oi[d] for d in range(K))
        for j in range(i + 1):
            phi = math.exp(gc[i] - gc[j])
            qk = sum(qc[i][d] * kc[j][d] for d in range(K))
            vo = sum(Vp[j][e] * oc[i][e] for e in range(V))
            for d in range(K):
                dq[i][d] += SCALE * phi * vo * kc[j][d]
                dk[j][d] += SCALE * phi * vo * qc[i][d]
            for e in range(V):
                dVp[j][e] += SCALE * phi * qk * oc[i][e]
            dg[j] -= SCALE * phi * qk * vo
            dg[i] += SCALE * phi * qk * vo

    # ---- stage 3: V' = A W
    for i in range(L):
        for j in range(i + 1):
            dAc[i][j] = sum(dVp[i][e] * W[j][e] for e in range(V))
    dW = [[sum(Ac[i][j] * dVp[i][e] for i in range(j, L)) for e in range(V)]
          for j in range(L)]

    # ---- stage 4: W[j] = beta_j (v_j - e^{g_j} (S0^T k_j))
    for j in range(L):
        egj = math.exp(gc[j])
        for e in range(V):
            r = dW[j][e]
            db[j] += r * W[j][e] / bc[j]
            dv[j][e] += bc[j] * r
            dg[j] -= bc[j] * egj * sum(S0[d][e] * kc[j][d] for d in range(K)) * r
            for d in range(K):
                dk[j][d] -= bc[j] * egj * S0[d][e] * r
                dS0[d][e] -= bc[j] * egj * kc[j][d] * r
    return dq, dk, dv, db, dg, dAc, dS0


# ---------------------------------------------------------------- reverse passes over a sequence
def token_reverse(q, k, v, beta, gc, h0, states, oc, Sfinal_cot):
    """The reverse of the per-token recurrence, one token at a time, given the
    cotangents of every o_t and of the final state.  Per token t, with
    X = e^{g_t} S_{t-1} the decayed pre-state:

      S~_t     = (carried cotangent) + scale q_t (o~_t)^T
      q~_t     = scale S_t o~_t
      u~       = S~_t^T k_t
      beta~_t  = u~ . (v_t - X^T k_t)
      v~_t     = beta_t u~
      X~       = S~_t - beta_t k_t u~^T
      k~_t     = S~_t u~ - beta_t X u~
      gamma~_t = (X~ . X) - beta_t (X^T k_t) . u~
      S~_{t-1} = e^{gamma_t} X~     (carried back)

    The per-step gate is gamma_t = g_t - g_{t-1} inside a chunk, gamma_t = g_t
    at a chunk start (g is the in-chunk cumsum), so the gamma cotangents fold
    back through that cumsum: gc~_t = gamma~_t - gamma~_{t+1} in the same
    chunk, gc~_t = gamma~_t at the last token of a chunk.
    """
    dq = [[0.0] * K for _ in range(len(k))]
    dk = [[0.0] * K for _ in range(len(k))]
    dv = [[0.0] * V for _ in range(len(k))]
    db = [0.0] * len(k)
    dgam = [0.0] * len(k)
    Sst = [row[:] for row in Sfinal_cot]
    for t in range(len(k) - 1, -1, -1):
        gam = gc[t] - (gc[t - 1] if t > 0 and t % CHUNK != 0 else 0.0)
        D = math.exp(gam)
        Sprev = h0 if t == 0 else states[t - 1]
        Sk = [[D * Sprev[d][e] for e in range(V)] for d in range(K)]
        sk2 = [sum(Sk[d][e] * k[t][d] for d in range(K)) for e in range(V)]
        S_t = states[t]
        St = [[Sst[d][e] + SCALE * q[t][d] * oc[t][e] for e in range(V)]
              for d in range(K)]                        # S~_t
        for d in range(K):
            dq[t][d] += SCALE * sum(S_t[d][e] * oc[t][e] for e in range(V))
        ut = [sum(St[d][e] * k[t][d] for d in range(K)) for e in range(V)]
        db[t] += sum(ut[e] * (v[t][e] - sk2[e]) for e in range(V))
        for e in range(V):
            dv[t][e] += beta[t] * ut[e]
        Skcot = [[St[d][e] - beta[t] * k[t][d] * ut[e] for e in range(V)]
                 for d in range(K)]                     # X~
        ufwd = [beta[t] * (v[t][e] - sk2[e]) for e in range(V)]   # forward u_t
        for d in range(K):
            u1 = sum(St[d][e] * ufwd[e] for e in range(V))   # from S_t = X + k u^T
            u2 = sum(Sk[d][e] * ut[e] for e in range(V))     # from u = beta(v - X^T k)
            dk[t][d] += u1 - beta[t] * u2
        dgam[t] += sum(St[d][e] * Sk[d][e] for d in range(K) for e in range(V)) \
            - beta[t] * sum(sk2[e] * ut[e] for e in range(V))
        for d in range(K):
            for e in range(V):
                Sst[d][e] = D * Skcot[d][e]             # S~_{t-1}
    # fold the gamma cotangents back through the in-chunk cumsum of g
    dg = [0.0] * len(k)
    for t in range(len(k)):
        dg[t] = dgam[t]
        if (t + 1) % CHUNK != 0 and t + 1 < len(k):     # g_t feeds gamma_{t+1}
            dg[t] -= dgam[t + 1]
    return dq, dk, dv, db, dg, Sst                      # Sst ends as h0's cotangent


def chunk_reverse(q, k, v, beta, gc, h0, oc, Sfinal_cot):
    """The same cotangents through the chunked forward: per chunk, in reverse,
    the state cotangent carried back across the chunk boundary; dA pushed
    through gdn_kkt_solve's adjoint."""
    Ttok = len(k)
    dq = [[0.0] * K for _ in range(Ttok)]
    dk = [[0.0] * K for _ in range(Ttok)]
    dv = [[0.0] * V for _ in range(Ttok)]
    db = [0.0] * Ttok
    dg = [0.0] * Ttok
    Sst = [row[:] for row in Sfinal_cot]
    for (n, m) in reversed(chunk_bounds()):
        L = m - n
        qc, kc, vc = q[n:m], k[n:m], v[n:m]
        bc, gcc = beta[n:m], gc[n:m]
        Ac = kkt_forward(kc, bc, gcc)
        S0 = state_at(q, k, v, beta, gc, h0, n)
        W, Vp = scan_pieces(S0, kc, vc, bc, gcc, Ac)
        dqc, dkc, dvc, dbc, dgc, dAc, dS0 = scan_adjoint(
            oc[n:m], Sst, S0, qc, kc, vc, bc, gcc, Ac, Vp=Vp, W=W)
        dkk, dbk, dgk = kkt_adjoint(dAc, kc, bc, gcc, A=Ac)
        for i in range(L):
            for d in range(K):
                dq[n + i][d] += dqc[i][d]
                dk[n + i][d] += dkc[i][d] + dkk[i][d]
            for e in range(V):
                dv[n + i][e] += dvc[i][e]
            db[n + i] += dbc[i] + dbk[i]
            dg[n + i] += dgc[i] + dgk[i]
        Sst = dS0
    return dq, dk, dv, db, dg, Sst


# ---------------------------------------------------------------- finite differences
def flat(t):
    """(containers, values): every 1-element list and its float."""
    if isinstance(t, list):
        if t and not isinstance(t[0], list):
            return [(t, i) for i in range(len(t))]
        out = []
        for x in t:
            out.extend(flat(x))
        return out
    raise TypeError


def fd_grads(loss, params):
    """Central finite differences of loss() over every float in params
    (dict name -> nested list), returned in the same shapes."""
    grads = {}
    for name, arr in params.items():
        g = [[0.0] * len(row) for row in arr] if isinstance(arr[0], list) \
            else [0.0] * len(arr)
        for c, i in flat(arr):
            v = c[i]
            h = 3.0e-5 * (1.0 + abs(v))
            c[i] = v + h
            fp = loss()
            c[i] = v - h
            fm = loss()
            c[i] = v
            _set(g, c, i, (fp - fm) / (2.0 * h), arr)
        grads[name] = g
    return grads


def _set(g, c, i, val, arr):
    """Write val into the grad tree at the leaf (c, i)."""
    # walk arr and g in the same order: g mirrors arr's shape
    gi = flat(g)
    ai = flat(arr)
    for (gc_, gi_), (ac_, ai_) in zip(gi, ai):
        if ac_ is c and gi_ == i:
            gc_[gi_] = val
            return
    raise RuntimeError("leaf not found")


def rel_errors(a_tree, f_tree):
    """Worst |a - f| / max(|a|, |f|) over matching trees, with an absolute floor."""
    worst, wtxt = 0.0, ""
    ai, fi = iter(flat(a_tree)), None
    fa, ff = flat(a_tree), flat(f_tree)
    for (ca, ia), (cf, if_) in zip(fa, ff):
        a, f = ca[ia], cf[if_]
        diff = abs(a - f)
        if diff <= ABS:
            continue
        den = max(abs(a), abs(f))
        r = diff / den if den > 0 else float("inf")
        if r > worst:
            worst, wtxt = r, "(%g vs %g)" % (a, f)
    return worst, wtxt


# ---------------------------------------------------------------- checks
def main():
    rnd = random.Random(11)
    q, k, v, beta, gc, h0 = make_inputs()

    ok = True

    # 1. chunked == per-token
    o_t, S_t, states = forward_token(q, k, v, beta, gc, h0, want_states=True)
    o_c, S_c = forward_chunked(q, k, v, beta, gc, h0)
    dov, dtxt = max(abs(o_t[i][e] - o_c[i][e])
                    for i in range(T) for e in range(V)), ""
    ds = max(abs(S_t[d][e] - S_c[d][e]) for d in range(K) for e in range(V))
    d = max(dov, ds)
    rel = d / max(1e-30, max(abs(o_t[i][e]) for i in range(T) for e in range(V)))
    print("chunked == per-token        : %.3e (max abs), %s" %
          (d, "PASS" if d <= ABS else "FAIL"))
    ok &= d <= ABS

    # cotangents for the reverse checks: of every o_t and of the final state
    Ro = rnd_cot(rnd, T, V)
    Rf = rnd_cot(rnd, K, V)
    tok = token_reverse(q, k, v, beta, gc, h0, states, Ro, Rf)
    chk = chunk_reverse(q, k, v, beta, gc, h0, Ro, Rf)
    names = ["dq", "dk", "dv", "dbeta", "dg", "dh0"]
    worst, wtxt, wname = 0.0, "", ""
    for name, a, f in zip(names, tok, chk):
        r, t2 = rel_errors(a, f)
        if r > worst:
            worst, wtxt, wname = r, t2, name
    print("token-reverse == chunk-rev  : %.3e (worst %s), %s" %
          (worst, wname, "PASS" if worst <= REL else "FAIL"))
    ok &= worst <= REL

    # 2. gdn_kkt_solve's adjoint against finite differences (chunk 0)
    L = CHUNK
    kc, bc, gcc = k[:L], beta[:L], gc[:L]
    Acot = [[0.0] * L for _ in range(L)]
    for i in range(L):                      # strict lower: the op's only real outputs
        for j in range(i):
            Acot[i][j] = rnd.uniform(-1, 1)
    Ad = kkt_adjoint(Acot, kc, bc, gcc)

    def loss_kkt():
        A = kkt_forward(kc, bc, gcc)
        return sum(Acot[i][j] * A[i][j] for i in range(L) for j in range(i))

    fd = fd_grads(loss_kkt, {"k": kc, "beta": bc, "g": gcc})
    worst, wname = 0.0, ""
    for name, a, f in zip(["dk", "dbeta", "dg"], Ad, [fd["k"], fd["beta"], fd["g"]]):
        r, t2 = rel_errors(a, f)
        if r > worst:
            worst, wname = r, name
    print("kkt adjoint == finite diff  : %.3e (worst %s), %s" %
          (worst, wname, "PASS" if worst <= REL else "FAIL"))
    ok &= worst <= REL

    # 3. gdn_chunk_scan's adjoint against finite differences (one chunk, h0 as S0)
    Sc0 = rnd_cot(rnd, K, V)
    Ac = kkt_forward(kc, bc, gcc)
    Ro_c = rnd_cot(rnd, L, V)
    Rh = rnd_cot(rnd, K, V)
    ana = scan_adjoint(Ro_c, Rh, Sc0, q[:L], kc, v[:L], bc, gcc, Ac)

    def loss_scan():
        o_c, S1 = scan_chunk(Sc0, q[:L], kc, v[:L], bc, gcc, Ac)
        return sum(Ro_c[i][e] * o_c[i][e] for i in range(L) for e in range(V)) \
            + sum(Rh[d][e] * S1[d][e] for d in range(K) for e in range(V))

    fd = fd_grads(loss_scan, {"q": q[:L], "k": kc, "v": v[:L], "beta": bc,
                              "g": gcc, "A": Ac, "h0": Sc0})
    # scan_adjoint returns (dq, dk, dv, dbeta, dg, dA, dh0) in that order
    worst, wname = 0.0, ""
    for name, a, f in zip(["dq", "dk", "dv", "dbeta", "dg", "dA", "dh0"], ana,
                          [fd["q"], fd["k"], fd["v"], fd["beta"], fd["g"],
                           fd["A"], fd["h0"]]):
        r, t2 = rel_errors(a, f)
        if r > worst:
            worst, wname = r, name
    print("scan adjoint == finite diff : %.3e (worst %s), %s" %
          (worst, wname, "PASS" if worst <= REL else "FAIL"))
    ok &= worst <= REL

    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
