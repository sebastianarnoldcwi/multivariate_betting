"""
uplib.py -- universal-portfolio / mixture strategies for the simulation study.

Bounded case:
    E^mix_t = int_Lambda prod_{s<=t} (1 + lambda' X_s) nu(d lambda),   nu = uniform on Lambda,
    which is the wealth of the predictable posterior-mean bet
    lambdabar_t = int lambda W_{t-1}(lambda) nu(d lambda) / int W_{t-1}(lambda) nu(d lambda).
    The integral has no closed form (Cover's exact algorithm is exponential in t); we compute the
    mixture over a fixed quasi-random point set Lambda_N (scrambled Sobol points mapped to the
    simplex), i.e. the uniform mixture over N constant bets, which is itself an e-process and
    converges to E^mix as N grows.

Gaussian case: mixture over constant bets lambda >= 0 of the exponential e-processes
    prod_s exp(lambda' X_s - 1/2 lambda' Sigma lambda)
under the prior with independent Exp(rate) coordinates (rate = p in the paper), evaluated in
closed form up to a Gaussian orthant probability (Genz's algorithm): an oracle version with the
true covariance (UP-or), and a corrected version in which the mixture runs over the scaled
strategies lambda_s = q_{k(s)} lambda' with the epoch-corrected penalty (UP); see up_gauss_exact.

Data generators reproduce exactly the streams used by simlib.run_bounded / run_gauss (same
seed, same sequence of rng calls), so that UP runs on the same data as MV and Avg.
"""
import time
import numpy as np
from scipy.special import ndtr, logsumexp
from scipy.stats import qmc
from simlib import copula_corr, NOT_REJ


# ----------------------------------------------------------------------------
# data streams (identical to simlib.run_bounded / run_gauss)
# ----------------------------------------------------------------------------
def gen_bounded_data(mu, R, a, n, nreps, seed):
    mu = np.asarray(mu, float)
    p = len(mu)
    Lz = np.linalg.cholesky(copula_corr(R))
    rng = np.random.default_rng(seed)
    data = np.empty((n, nreps, p))
    for t in range(n):
        Z = rng.standard_normal((nreps, p)) @ Lz.T
        data[t] = mu + a * (2.0 * ndtr(Z) - 1.0)
    return data


def gen_gauss_data(mu, Sigma, n, nreps, seed):
    mu = np.asarray(mu, float)
    p = len(mu)
    L = np.linalg.cholesky(Sigma)
    rng = np.random.default_rng(seed)
    data = np.empty((n, nreps, p))
    for t in range(n):
        data[t] = mu + rng.standard_normal((nreps, p)) @ L.T
    return data


# ----------------------------------------------------------------------------
# point sets
# ----------------------------------------------------------------------------
def simplex_points(N, p, seed=0, a=1.0):
    """N quasi-random points on Lambda = {lambda >= 0, sum lambda <= 1} with a Dirichlet(a,...,a)
    law for (lambda_0, lambda_1, ..., lambda_p), lambda_0 = 1 - sum lambda the cash coordinate.
    a = 1: uniform on Lambda (scrambled Sobol points -> ordered spacings); a = 1/2: the
    Dirichlet(1/2) prior of Cover and Ordentlich (1996), via Gamma(a) inverse cdfs."""
    m = int(np.ceil(np.log2(N)))
    if a == 1.0:
        u = qmc.Sobol(d=p, scramble=True, seed=seed).random_base2(m)[:N]
        u.sort(axis=1)
        return np.diff(np.concatenate([np.zeros((N, 1)), u], axis=1), axis=1)
    from scipy.special import gammaincinv
    u = qmc.Sobol(d=p + 1, scramble=True, seed=seed).random_base2(m)[:N]
    G = gammaincinv(a, u)
    W = G / G.sum(1, keepdims=True)
    return W[:, :p]


# ----------------------------------------------------------------------------
# bounded: uniform mixture over Lambda_N
# ----------------------------------------------------------------------------
def up_bounded(data, pts, alpha=0.05, chunk=200, renorm=25, verbose=False):
    """data (n, nreps, p); pts (N, p). Returns dict(tau (nreps,), logE_end (nreps,), seconds).
    Wealth of the finite uniform mixture, exactly the wealth of the posterior-mean bet."""
    n, nreps, p = data.shape
    N = len(pts)
    thr = np.log(1.0 / alpha)
    tau = np.full(nreps, NOT_REJ, dtype=np.int64)
    logE_end = np.empty(nreps)
    t0 = time.time()
    for c0 in range(0, nreps, chunk):
        c1 = min(c0 + chunk, nreps)
        W = np.ones((N, c1 - c0))
        logc = np.zeros(c1 - c0)
        tau_c = np.full(c1 - c0, NOT_REJ, dtype=np.int64)
        for t in range(n):
            W *= 1.0 + pts @ data[t, c0:c1].T          # (N, chunk)
            if (t + 1) % renorm == 0:
                m = W.max(0)
                W /= m
                logc += np.log(m)
            logE = logc + np.log(W.mean(0))
            hit = (logE >= thr) & (tau_c == NOT_REJ)
            tau_c[hit] = t + 1
        tau[c0:c1] = tau_c
        logE_end[c0:c1] = logE
        if verbose:
            print(f"    UP chunk {c0}-{c1}: {time.time()-t0:.0f}s", flush=True)
    return dict(tau=tau, logE_end=logE_end, seconds=time.time() - t0, N=N)


# ----------------------------------------------------------------------------
# Gaussian: Exp(1)-prior mixture over the orthant, corrected and oracle penalties
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Gaussian: exact mixture (up to a Gaussian orthant probability, Genz's algorithm)
# ----------------------------------------------------------------------------
_QMC_CACHE = {}


def log_orthant_prob(m, C, n_q=256, seed=0):
    """log P(Y >= 0) for Y ~ N(m, C), vectorised over the leading axis: m (R, p), C (R, p, p).
    Genz's separation-of-variables estimator with n_q scrambled Sobol points."""
    from scipy.special import log_ndtr, ndtri
    R, p = m.shape
    L = np.linalg.cholesky(C)                                   # (R, p, p)
    key = (p, n_q, seed)
    if key not in _QMC_CACHE:
        _QMC_CACHE[key] = qmc.Sobol(d=max(p - 1, 1), scramble=True, seed=seed).random(n_q)
    w = _QMC_CACHE[key]                                         # (n_q, p-1)
    z = np.zeros((R, n_q, p))
    logprod = np.zeros((R, n_q))
    for i in range(p):
        # a_i = (-m_i - sum_{j<i} L_ij z_j) / L_ii ; constraint z_i >= a_i
        s = np.einsum('rj,rqj->rq', L[:, i, :i], z[:, :, :i]) if i > 0 else 0.0
        a = (-m[:, i, None] - s) / L[:, i, i, None]              # (R, n_q)
        lq = log_ndtr(-a)                                        # log P(z_i >= a_i)
        logprod += lq
        if i < p - 1:
            # z_i ~ N(0,1) conditioned on z_i >= a_i, via the upper tail: z = -ndtri((1-w) q)
            q = np.exp(lq)
            z[:, :, i] = -ndtri(np.clip((1.0 - w[None, :, i]) * q, 1e-300, 1.0))
    return logsumexp(logprod, axis=1) - np.log(n_q)


def up_gauss_exact(data, sched, Sigma, alpha=0.05, n_q=256, chunk=250, verbose=False, rate=1.0):
    """Mixtures under the prior with i.i.d. Exp(rate) coordinates (mean stake 1/rate per
    coordinate; rate = p gives total mean stake one), oracle and scaled-corrected, via the closed form
        log E_t = 1/2 b'A^{-1}b + (p/2) log(2 pi) - 1/2 log det A + log P(N(A^{-1}b, A^{-1}) >= 0) + p log(rate),
    with (b, A) = (S_t - rate 1, t Sigma) for the oracle and (sum_s q_k X_s - rate 1, sum_s q_k S_{t_k-1})
    for the corrected version (bets q_k lambda', penalty S_{t_k-1}/q_k). The closed form is exact;
    the orthant probability in it is evaluated by quasi-Monte Carlo (Genz's algorithm, n_q points).

    Validity. UP-or is an exact test supermartingale at level alpha when Sigma is the true
    covariance. UP-sc is NOT an e-process: its conditional expectation can even be infinite on the
    event that the sample covariance underestimates Sigma. Its guarantee is the time-uniform bound
    P(exists t: E_t >= 1/alpha) <= alpha + delta of the paper (proposition on the scaled mixture
    under the epoch correction): q_k is the lower epsilon_k-quantile of the Wishart pivot
    lambda_min(Sigma^{-1/2} S_{t_k-1} Sigma^{-1/2}), so S_{t_k-1} >= q_k Sigma holds on every epoch
    simultaneously except on an event of probability <= sum_k epsilon_k <= delta, and there the
    process is dominated by the oracle supermartingale of the scaled bets (sandwich argument).

    Returns dict(tau (nreps, 2) for [UP-sc, UP-or], logE_end, seconds); logE_end[r, j] is the
    log mixture of repetition r at its stopping time, or at the horizon if the test never stops
    (rows are no longer updated after rejection)."""
    n, nreps, p = data.shape
    tk = np.asarray(sched['tk'], dtype=int)
    q_mv = np.asarray(sched['q_mv'], dtype=float)
    assert 0.0 < alpha < 1.0 and rate > 0 and n_q > 0
    assert tk.ndim == 1 and len(tk) == len(q_mv) and np.all(np.diff(tk) > 0)
    assert tk[0] >= p + 2, "need t_1 >= p + 2 so that the sample covariance is a.s. nonsingular"
    assert np.all(np.isfinite(q_mv)) and np.all(q_mv > 0)
    np.linalg.cholesky(Sigma)                       # Sigma must be symmetric positive definite
    t1 = tk[0]
    thr = np.log(1.0 / alpha)
    c0_ = 0.5 * p * np.log(2 * np.pi) + p * np.log(rate)
    tau = np.full((nreps, 2), NOT_REJ, dtype=np.int64)
    logE_end = np.full((nreps, 2), np.nan)
    t0 = time.time()
    sd, ld_S = np.linalg.slogdet(Sigma)
    Sinv = np.linalg.inv(Sigma)
    for c0 in range(0, nreps, chunk):
        c1 = min(c0 + chunk, nreps); nc = c1 - c0
        S1 = np.zeros((nc, p)); S2 = np.zeros((nc, p, p))
        b_sc = np.full((nc, p), -rate); A_sc = np.zeros((nc, p, p))
        Sk = None; q = None
        active = np.ones((nc, 2), bool)
        tau_c = np.full((nc, 2), NOT_REJ, dtype=np.int64)
        lE = np.full((nc, 2), np.nan)
        for t in range(1, n + 1):
            D = data[t - 1, c0:c1]
            if t in tk:
                k = int(np.searchsorted(tk, t)); m = t - 1
                xbar = S1 / m
                Sk = (S2 - m * xbar[:, :, None] * xbar[:, None, :]) / (m - 1)
                q = q_mv[k]
            S1 += D
            S2 += D[:, :, None] * D[:, None, :]
            # --- oracle: b = S_t - 1, A = t Sigma  (closed form, orthant prob with C = Sigma^{-1}/t)
            act = active[:, 1]
            if act.any():
                b = S1[act] - rate
                mvec = (b @ Sinv) / t
                quad = 0.5 * np.einsum('ri,ri->r', b, mvec)
                lE_or = quad + c0_ - 0.5 * (p * np.log(t) + ld_S) + \
                    log_orthant_prob(mvec, np.broadcast_to(Sinv / t, (act.sum(), p, p)), n_q=n_q)
                lE[act, 1] = lE_or
                hit = lE_or >= thr
                idx = np.flatnonzero(act)[hit]
                tau_c[idx, 1] = t; active[idx, 1] = False
            # --- scaled corrected mixture
            if t >= t1:
                b_sc += q * D
                A_sc += q * Sk
                act = active[:, 0]
                if act.any():
                    A = A_sc[act]; b = b_sc[act]
                    Ainv = np.linalg.inv(A)
                    mvec = np.einsum('rij,rj->ri', Ainv, b)
                    quad = 0.5 * np.einsum('ri,ri->r', b, mvec)
                    sgn, ldA = np.linalg.slogdet(A)
                    lE_sc = quad + c0_ - 0.5 * ldA + log_orthant_prob(mvec, Ainv, n_q=n_q)
                    lE[act, 0] = lE_sc
                    hit = lE_sc >= thr
                    idx = np.flatnonzero(act)[hit]
                    tau_c[idx, 0] = t; active[idx, 0] = False
            if not active.any():
                break
        tau[c0:c1] = tau_c
        logE_end[c0:c1] = lE
        if verbose:
            print(f"    UP-exact chunk {c0}-{c1}: {time.time()-t0:.0f}s", flush=True)
    return dict(tau=tau, logE_end=logE_end, seconds=time.time() - t0)
