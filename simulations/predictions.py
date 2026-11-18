"""
predictions.py -- theoretical predictions to set against the simulated rejection times.

bounded : Wald's identity for the constant oracle bets, log(1/alpha)/ell(lam) for MV-or and
          b_alpha/ell_j(lam_j) for Avg-or (single violation only), with the exact growth rate
          ell(lam) = E log(1 + lam'X) obtained by Monte Carlo integration against the generator.
gauss   : log(1/alpha)/ell^G for MV-or with ell^G = lam'mu - 1/2 lam'Sigma lam (exact), and the
          scaling-identity prediction for the corrected MV: first tau with
          sum_{s = t1}^{tau} q_{k(s)} ell^G >= log(1/alpha).
"""
import numpy as np
from scipy.special import ndtr
from simlib import *

alpha = 0.05
L = np.log(1 / alpha)


def exact_rate_bounded(lam, mu, R, a=0.8, nmc=2_000_000, seed=11):
    rng = np.random.default_rng(seed)
    Lz = np.linalg.cholesky(copula_corr(R))
    tot = 0.0
    for _ in range(4):
        Z = rng.standard_normal((nmc // 4, len(mu))) @ Lz.T
        X = mu + a * (2 * ndtr(Z) - 1)
        tot += np.log1p(X @ lam).sum()
    return tot / nmc


def exact_rate_single_bounded(lam, mu_i, a=0.8):
    # X_i = mu_i + a U, U ~ Unif(-1,1): E log(1 + lam X_i) by quadrature
    from scipy.integrate import quad
    return quad(lambda u: np.log1p(lam * (mu_i + a * u)) / 2, -1, 1)[0]


def predictions_bounded(mu, R, lam_mv_or, lam_av_or, p):
    b = np.log(p / alpha - p + 1)
    ell_mv = exact_rate_bounded(lam_mv_or, mu, R)
    pred_mv = L / ell_mv
    viol = np.where(mu > 0)[0]
    if len(viol) == 1:
        j = viol[0]
        ell_av = exact_rate_single_bounded(lam_av_or[j], mu[j])
        pred_av = b / ell_av
    else:
        ell_av, pred_av = np.nan, np.nan
    return dict(ell_mv=ell_mv, pred_mv=pred_mv, ell_av=ell_av, pred_av=pred_av, b_alpha=b)


def predictions_gauss(mu, Sigma, lam_mv_or, lam_av_or, sched, p):
    b = np.log(p / alpha - p + 1)
    ellG = lam_mv_or @ mu - 0.5 * lam_mv_or @ Sigma @ lam_mv_or
    pred_mv_or = L / ellG if ellG > 0 else np.nan
    # corrected MV: growth q_k ellG in epoch k, from t1
    tk, q = sched['tk'], sched['q_mv']
    t1 = tk[0]
    acc, t = 0.0, t1 - 1
    pred_mv = np.nan
    while t < 100000:
        t += 1
        k = int(np.searchsorted(tk, t, side='right')) - 1
        acc += q[k] * ellG
        if acc >= L:
            pred_mv = t
            break
    viol = np.where(mu > 0)[0]
    if len(viol) == 1:
        j = viol[0]
        ell_av = mu[j] ** 2 / (2 * Sigma[j, j])
        pred_av_or = b / ell_av
        # corrected Avg with chi-square factors
        qa = sched['q_un']
        acc, t = 0.0, t1 - 1
        pred_av = np.nan
        while t < 100000:
            t += 1
            k = int(np.searchsorted(tk, t, side='right')) - 1
            acc += qa[k] * ell_av
            if acc >= b:
                pred_av = t
                break
    else:
        pred_av_or, pred_av = np.nan, np.nan
    return dict(ellG=ellG, pred_mv_or=pred_mv_or, pred_mv=pred_mv, pred_av_or=pred_av_or, pred_av=pred_av)
