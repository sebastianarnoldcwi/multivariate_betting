"""
timing.py -- per-round cost of the strategies on a single data stream (online use).

MV  : moment update (O(p^2)) + one quadratic programme over Lambda (bounded) or the orthant
      (Gaussian), solved by a warm-started primal active-set method (a handful of (p+1)-dim
      linear solves); state = (muhat, Mhat), p(p+3)/2 numbers.
Avg : p one-dimensional clipped ratios; state 2p numbers.
UP  : bounded case, multiplicative update of the N mixture weights and the posterior-mean bet
      (O(Np)); state = N weights (plus the N x p point set). Gaussian case, the exact mixture:
      one update of (b, A) plus Genz's orthant probability with 256 quasi-random points.
Reported: microseconds per round (median over rounds of a 500-round stream), p = 3 and p = 9.
"""
import time, sys
import numpy as np
from simlib import *
from uplib import *


def qp_single(mu, M, lam0, capped, tol=1e-10, maxit=60):
    """Primal active-set method for max lambda'mu - 1/2 lambda'M lambda over lambda >= 0
    (and sum lambda <= 1 if capped), warm-started at lam0."""
    p = len(mu)
    if np.all(mu <= 0):
        return np.zeros(p)
    free = lam0 > tol
    if not free.any():
        free = mu > 0
    budget = bool(capped and lam0.sum() >= 1 - 1e-9)
    lam = np.zeros(p)
    for _ in range(maxit):
        idx = np.flatnonzero(free)
        A = M[np.ix_(idx, idx)]
        if budget:
            k = len(idx)
            K = np.empty((k + 1, k + 1)); K[:k, :k] = A; K[:k, k] = 1.0; K[k, :k] = 1.0; K[k, k] = 0.0
            rhs = np.append(mu[idx], 1.0)
            sol = np.linalg.solve(K, rhs)
            x, theta = sol[:k], sol[k]
        else:
            x = np.linalg.solve(A, mu[idx]); theta = 0.0
        if (x < -tol).any():                      # primal infeasible: fix the most negative one
            free[idx[np.argmin(x)]] = False
            continue
        lam[:] = 0.0; lam[idx] = x
        if capped and not budget and lam.sum() > 1 + tol:
            budget = True
            continue
        if budget and theta < -tol:               # budget multiplier negative: release the budget
            budget = False
            continue
        g = mu - M @ lam - theta
        g[free] = 0.0
        j = np.argmax(g)
        if g[j] > tol:                            # dual infeasible: free the worst coordinate
            free[j] = True
            continue
        return lam
    return qp_exact_single(mu, M, capped)         # fallback (never hit in the timings)


def time_stream(p, gauss, N, n=500, seed=3):
    rng = np.random.default_rng(seed)
    if gauss:
        R = equicorr(p, 0.3); mu = np.full(p, 0.15)
        L = np.linalg.cholesky(R)
        X = mu + rng.standard_normal((n, p)) @ L.T
        pts = None                                   # Gaussian UP is timed by time_gauss_exact
    else:
        R = equicorr(p, 0.3); mu = np.full(p, 0.10)
        Lz = np.linalg.cholesky(copula_corr(R))
        X = mu + 0.8 * (2 * ndtr(rng.standard_normal((n, p)) @ Lz.T) - 1)
        pts = simplex_points(N, p, seed=1)
    out = {}
    # --- MV plug-in: moment update + QP (active-set), warm start
    S1 = np.zeros(p); S2 = np.zeros((p, p)); lam = np.zeros(p); tm = []
    for t in range(n):
        t0 = time.perf_counter()
        if t >= p + 3:
            muh = S1 / t; Mh = S2 / t
            if gauss:
                Sh = (S2 - t * np.outer(muh, muh)) / (t - 1)
                lam = qp_single(muh, Sh, lam, capped=False)
            else:
                lam = qp_single(muh, Mh, lam, capped=True)
        S1 += X[t]; S2 += np.outer(X[t], X[t])
        tm.append(time.perf_counter() - t0)
    out['MV'] = np.median(tm[p + 3:]) * 1e6
    # --- Avg
    S1 = np.zeros(p); S2d = np.zeros(p); tm = []
    for t in range(n):
        t0 = time.perf_counter()
        if t >= p + 3:
            lam = np.clip((S1 / t) / (S2d / t), 0, 1)
        S1 += X[t]; S2d += X[t] ** 2
        tm.append(time.perf_counter() - t0)
    out['Avg'] = np.median(tm[p + 3:]) * 1e6
    # --- UP (bounded): weight update + posterior-mean bet
    if pts is not None:
        w = np.ones(N); tm = []
        for t in range(n):
            t0 = time.perf_counter()
            w *= 1.0 + pts @ X[t]
            if t % 25 == 0:
                w /= w.max()
            lam_bar = (w @ pts) / w.sum()
            tm.append(time.perf_counter() - t0)
        out['UP'] = np.median(tm) * 1e6
    return out


def time_gauss_exact(p, n=500, n_q=256, seed=3):
    """per-round cost of the exact mixture (oracle penalty): update of (b, A) + Genz."""
    rng = np.random.default_rng(seed)
    R = equicorr(p, 0.3); mu = np.full(p, 0.15)
    L = np.linalg.cholesky(R)
    X = mu + rng.standard_normal((n, p)) @ L.T
    Sinv = np.linalg.inv(R); ld = np.linalg.slogdet(R)[1]
    S1 = np.zeros(p); tm = []
    for t in range(1, n + 1):
        t0 = time.perf_counter()
        S1 += X[t - 1]
        b = S1 - 1.0
        m = (Sinv @ b) / t
        lE = 0.5 * b @ m + 0.5 * p * np.log(2 * np.pi) - 0.5 * (p * np.log(t) + ld) + \
            log_orthant_prob(m[None], (Sinv / t)[None], n_q=n_q)[0]
        tm.append(time.perf_counter() - t0)
    return np.median(tm) * 1e6


if __name__ == '__main__':
    rows = []
    for p, Ns in ((3, (2 ** 12, 2 ** 14)), (9, (2 ** 14, 2 ** 16, 2 ** 18))):
        for N in Ns:
            o = time_stream(p, False, N)
            rows.append(('bounded', p, N, o['MV'], o['Avg'], o['UP']))
            print(f"bounded  p={p} N=2^{int(np.log2(N))}: MV {o['MV']:.0f} us  Avg {o['Avg']:.0f} us  "
                  f"UP {o['UP']:.0f} us per round", flush=True)
    for p in (3, 9):
        o = time_stream(p, True, 0)
        up = time_gauss_exact(p)
        rows.append(('Gaussian', p, 256, o['MV'], o['Avg'], up))
        print(f"Gaussian p={p}: MV {o['MV']:.0f} us  Avg {o['Avg']:.0f} us  "
              f"UP {up:.0f} us per round (exact mixture, Genz with 256 points)", flush=True)
    np.save('results/timing.npy', np.array(rows, dtype=object), allow_pickle=True)
