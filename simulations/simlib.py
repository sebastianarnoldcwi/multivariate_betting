"""
simlib.py -- shared library for the simulation study of the paper
"Multivariate Betting: Leveraging the Covariance for Sequential Mean Testing".
All strategies are vectorised over the Monte Carlo repetitions.

Designs
-------
Both regimes are parametrised by the mean vector mu and the correlation matrix R of X.

  bounded : X_t = mu + a (2 Phi(Z_t) - 1),  Z_t ~ N(0, R_z),  (R_z)_ik = 2 sin(pi R_ik / 6)
            so that Corr(X_t) = R exactly and Var(X_{i,t}) = a^2/3   (Gaussian copula identity
            Corr(Phi(Z_i), Phi(Z_k)) = (6/pi) arcsin(rho_ik / 2)).
  gauss   : X_t ~ N(mu, R)   (unit variances).

Strategies (all predictable; the moment estimates at round t use X_1..X_{t-1})
-----------------------------------------------------------------------------
  bounded  MV   : lambda_t = argmax_{lambda >= 0, sum lambda <= 1} lambda'muhat - 1/2 lambda' Mhat lambda,
                  Mhat = uncentred second-moment matrix (surrogate plug-in);  E_t = prod (1 + lambda_s' X_s)
           Avg  : lambda_{i,t} = clip(muhat_i / Mhat_ii, 0, 1);  E_t = (1/p) sum_i prod_s (1 + lambda_{i,s} X_{i,s})
           oracles: same rules with the true moments, constant bets from t = 1.
  gauss    MV   : lambda_t = argmax_{lambda >= 0} lambda'muhat - 1/2 lambda' SigHat_t lambda,
                  SigHat_t = epoch-corrected penalty S_{t_k - 1} / q_{eps_k}(t_k - 2, p);
                  E_t = prod exp(lambda_s' X_s - 1/2 lambda_s' SigHat_s lambda_s)
           Avg  : lambda_{i,t} = (muhat_i / sighat^2_{i,t})_+ with chi-square-corrected variances;
                  symmetric average of the p exponential e-processes.
           naive: same with the plain sample covariance / variances (validity check).
           oracles: true mu and Sigma, constant bets from t = 1.
"""
import numpy as np
from scipy.special import ndtr, logsumexp
from scipy.stats import chi2

NOT_REJ = np.iinfo(np.int32).max  # code for "not rejected by the horizon"


# ----------------------------------------------------------------------------
# batched quadratic programmes:  maximise  lambda' mu - 1/2 lambda' M lambda
# ----------------------------------------------------------------------------
# The Euclidean projections below are the proximal steps of the accelerated projected-gradient
# method in qp_batch; they are not the Q-projection that characterises the solution (Remark
# "Geometry" in the paper). The iteration (gradient step + Euclidean projection) converges to the
# constrained maximiser, which qp_polish then recomputes exactly from the KKT system of its
# active set; qp_solve agrees with an exact solver to machine precision.
def proj_orthant(L):
    return np.maximum(L, 0.0)


def proj_capped_simplex(L):
    """Euclidean projection of each row onto {lambda >= 0, sum lambda <= 1}."""
    X = np.maximum(L, 0.0)
    over = X.sum(1) > 1.0
    if over.any():
        Y = X[over]
        u = -np.sort(-Y, axis=1)
        css = np.cumsum(u, axis=1)
        k = np.arange(1, Y.shape[1] + 1)
        cond = u - (css - 1.0) / k > 0
        rho = cond.sum(1)
        theta = (css[np.arange(len(Y)), rho - 1] - 1.0) / rho
        X[over] = np.maximum(Y - theta[:, None], 0.0)
    return X


def qp_batch(mu, M, lam0, capped, iters=150, lmax=None):
    """FISTA (accelerated projected gradient, with gradient restart) on each row.

    mu : (R, p), M : (R, p, p) symmetric PSD, lam0 : (R, p) warm start.
    Returns (R, p) array. Rows whose mean vector is componentwise <= 0 get the zero bet,
    which is there the exact solution (mu <= 0 is the KKT condition at lambda = 0). This is a
    whole-row statement: a single coordinate with negative mean can still receive a positive
    stake when some other coordinate has a positive mean (see the paper).
    """
    proj = proj_capped_simplex if capped else proj_orthant
    if lmax is None:
        lmax = np.linalg.eigvalsh(M)[:, -1]
    step = (1.0 / np.maximum(lmax, 1e-12))[:, None]
    x = proj(lam0.copy())
    y = x.copy()
    tk = 1.0
    for _ in range(iters):
        g = mu - np.einsum('rij,rj->ri', M, y)
        x_new = proj(y + step * g)
        tk_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
        d = x_new - x
        # gradient-based restart (O'Donoghue & Candes): drop momentum where it points uphill
        restart = np.einsum('ri,ri->r', g, d) < 0
        mom = ((tk - 1.0) / tk_new) * (~restart)[:, None]
        y = x_new + mom * d
        x, tk = x_new, tk_new
    x[(mu <= 0).all(1)] = 0.0
    return x


def qp_polish(mu, M, lam, capped, tol=1e-9):
    """Exact KKT polish: identify the active set from lam, solve the equality-constrained
    system, and keep the polished row where it satisfies the KKT conditions."""
    R, p = lam.shape
    out = lam.copy()
    free = lam > tol                                        # (R, p)
    budget_active = capped & (lam.sum(1) > 1.0 - 1e-7)      # (R,)
    # build the KKT systems in a padded (p+1) x (p+1) form
    n = p + 1
    A = np.zeros((R, n, n))
    b = np.zeros((R, n))
    A[:, :p, :p] = M
    A[:, :p, p] = 1.0
    A[:, p, :p] = 1.0
    b[:, :p] = mu
    b[:, p] = 1.0
    # fixed coordinates: row/col -> identity with zero rhs
    fixed = ~free
    idx = np.arange(p)
    for j in range(p):
        rows = fixed[:, j]
        if rows.any():
            A[rows, j, :] = 0.0
            A[rows, :, j] = 0.0
            A[rows, j, j] = 1.0
            b[rows, j] = 0.0
    # budget inactive: theta := 0
    rows = ~budget_active
    if rows.any():
        A[rows, p, :] = 0.0
        A[rows, :, p] = 0.0
        A[rows, p, p] = 1.0
        b[rows, p] = 0.0
    try:
        sol = np.linalg.solve(A, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return out
    lam_new = sol[:, :p]
    theta = sol[:, p]
    grad = mu - np.einsum('rij,rj->ri', M, lam_new) - theta[:, None]
    ok = np.isfinite(lam_new).all(1)
    ok &= (lam_new >= -1e-10).all(1)
    ok &= (np.where(free, 0.0, grad) <= 1e-8).all(1)       # multipliers of lambda_j >= 0 nonneg.
    if capped:
        ok &= theta >= -1e-10
        ok &= np.where(budget_active, True, lam_new.sum(1) <= 1.0 + 1e-8)
    out[ok] = np.maximum(lam_new[ok], 0.0)
    return out


def qp_solve(mu, M, lam0, capped, iters=150):
    lam = qp_batch(mu, M, lam0, capped, iters=iters)
    return qp_polish(mu, M, lam, capped)


def qp_exact_single(mu, M, capped):
    """Exact solution of one small QP (used for the oracle bets)."""
    from scipy.optimize import minimize
    p = len(mu)
    if np.all(mu <= 0):
        return np.zeros(p)
    cons = [{'type': 'ineq', 'fun': lambda l: 1.0 - l.sum(), 'jac': lambda l: -np.ones(p)}] if capped else []
    f = lambda l: -(l @ mu - 0.5 * l @ M @ l)
    g = lambda l: -(mu - M @ l)
    best = None
    for start in [np.zeros(p), np.full(p, 0.1 / p), np.maximum(mu, 0) * 0.5]:
        res = minimize(f, start, jac=g, bounds=[(0, None)] * p, constraints=cons,
                       method='SLSQP', options={'ftol': 1e-14, 'maxiter': 2000})
        if best is None or res.fun < best.fun:
            best = res
    lam = np.maximum(best.x, 0.0)
    if capped and lam.sum() > 1.0:
        lam /= lam.sum()
    # polish with the batched KKT solver
    lam = qp_polish(mu[None], M[None], lam[None], capped)[0]
    return lam


# ----------------------------------------------------------------------------
# designs
# ----------------------------------------------------------------------------
def corr_p9():
    """The p = 9 correlation matrix R of the paper: two-factor Gaussian copula, Spearman-type
    transform. Returns (R, R_z)."""
    B = np.array([[0.85, 0.15], [0.75, 0.25], [0.65, 0.10],
                  [-0.85, 0.20], [-0.75, -0.15], [-0.65, 0.25],
                  [0.05, 0.60], [0.00, -0.55], [0.02, 0.04]])
    Rz = B @ B.T + np.diag(1.0 - (B ** 2).sum(1))
    R = (6.0 / np.pi) * np.arcsin(Rz / 2.0)
    np.fill_diagonal(R, 1.0)
    return R, Rz


def equicorr(p, rho):
    R = np.full((p, p), rho)
    np.fill_diagonal(R, 1.0)
    return R


def copula_corr(R):
    """Gaussian-copula correlation R_z giving Corr(2 Phi(Z) - 1) = R exactly."""
    Rz = 2.0 * np.sin(np.pi * R / 6.0)
    np.fill_diagonal(Rz, 1.0)
    return Rz


# ----------------------------------------------------------------------------
# bounded regime
# ----------------------------------------------------------------------------
def run_bounded(mu, R, a=0.8, n=500, nreps=1000, warmup=6, alpha=0.05, seed=1,
                refit=1, qp_iters=120, verbose=False):
    """Returns dict with tau (nreps, 4) for [Avg, MV, Avg-or, MV-or] (NOT_REJ if no rejection
    by n), the oracle bets and the exact moments."""
    mu = np.asarray(mu, float)
    p = len(mu)
    assert np.all(np.abs(mu) <= 1 - a + 1e-12), "need |mu_i| <= 1 - a so that |X_i| <= 1"
    Rz = copula_corr(R)
    Lz = np.linalg.cholesky(Rz)
    Sigma = (a ** 2 / 3.0) * R
    M_true = Sigma + np.outer(mu, mu)
    lam_mv_or = qp_exact_single(mu, M_true, capped=True)
    lam_av_or = np.clip(mu / np.diag(M_true), 0.0, 1.0)
    thr = np.log(1.0 / alpha)
    eps = 1e-12
    rng = np.random.default_rng(seed)

    S1 = np.zeros((nreps, p))
    S2 = np.zeros((nreps, p, p))
    lgA = np.zeros((nreps, p)); lgAo = np.zeros((nreps, p))
    lM = np.zeros(nreps); lMo = np.zeros(nreps)
    tau = np.full((nreps, 4), NOT_REJ, dtype=np.int64)
    lam_mv = np.zeros((nreps, p))
    zero = np.zeros((nreps, p))
    logp = np.log(p)

    for t in range(1, n + 1):
        Z = rng.standard_normal((nreps, p)) @ Lz.T
        D = mu + a * (2.0 * ndtr(Z) - 1.0)
        if t > warmup:
            n0 = t - 1
            muh = S1 / n0
            Mh = S2 / n0
            if t == warmup + 1 or (t - warmup - 1) % refit == 0:
                lam_mv = qp_solve(muh, Mh, lam_mv, capped=True, iters=qp_iters)
            dM = np.einsum('rii->ri', Mh)
            lam_av = np.clip(muh / (dM + eps), 0.0, 1.0)
        else:
            lam_av = zero
            lam_mv = zero.copy()
        # wealth updates
        lgA += np.log(np.maximum(1.0 + lam_av * D, eps))
        lM += np.log(np.maximum(1.0 + np.einsum('ri,ri->r', lam_mv, D), eps))
        lgAo += np.log(np.maximum(1.0 + lam_av_or * D, eps))
        lMo += np.log(np.maximum(1.0 + D @ lam_mv_or, eps))
        eA = logsumexp(lgA, axis=1) - logp
        eAo = logsumexp(lgAo, axis=1) - logp
        for j, lw in enumerate((eA, lM, eAo, lMo)):
            hit = (lw >= thr) & (tau[:, j] == NOT_REJ)
            tau[hit, j] = t
        S1 += D
        S2 += D[:, :, None] * D[:, None, :]
        if verbose and t % 100 == 0:
            print(f"  t={t}: rejected so far {(tau < NOT_REJ).mean(0).round(3)}", flush=True)
    return dict(tau=tau, lam_mv_or=lam_mv_or, lam_av_or=lam_av_or, Sigma=Sigma, M_true=M_true,
                mu=mu, R=R, a=a, n=n, alpha=alpha, nreps=nreps, warmup=warmup, refit=refit, seed=seed)


# ----------------------------------------------------------------------------
# Gaussian regime with the epoch correction
# ----------------------------------------------------------------------------
def wishart_qmin(nu, p, eps, nsim=200_000, seed=7, chunk=50_000):
    """Lower eps-quantile (empirical inverse cdf) of lambda_min(W_p(I, nu) / nu),
    simulated through the Bartlett decomposition."""
    rng = np.random.default_rng(seed + nu)
    out = np.empty(nsim)
    done = 0
    while done < nsim:
        b = min(chunk, nsim - done)
        A = np.zeros((b, p, p))
        il = np.tril_indices(p, -1)
        A[:, il[0], il[1]] = rng.standard_normal((b, len(il[0])))
        for i in range(p):
            A[:, i, i] = np.sqrt(rng.chisquare(nu - i, size=b))
        W = A @ np.transpose(A, (0, 2, 1))
        out[done:done + b] = np.linalg.eigvalsh(W / nu)[:, 0]
        done += b
    return float(np.quantile(out, eps, method='inverted_cdf'))


def epoch_schedule(n, p, t1=None, delta=0.05):
    """Epoch starts t_k = t1 2^(k-1) <= n with t1 the first power of two >= p + 2 (unless given),
    budgets eps_k = (delta/2)/K for MV and (delta/2)/(K p) per coordinate for Avg."""
    if t1 is None:
        t1 = 2 ** int(np.ceil(np.log2(p + 2)))
    tk = []
    t = t1
    while t <= n:
        tk.append(t)
        t *= 2
    tk = np.array(tk)
    K = len(tk)
    eps_mv = (delta / 2) / K
    eps_un = (delta / 2) / (K * p)
    nu = tk - 2
    q_mv = np.array([wishart_qmin(int(v), p, eps_mv) for v in nu])
    q_un = chi2.ppf(eps_un, nu) / nu
    return dict(tk=tk, K=K, nu=nu, eps_mv=eps_mv, eps_un=eps_un, q_mv=q_mv, q_un=q_un, delta=delta)


def run_gauss(mu, Sigma, sched, n=1000, nreps=2000, alpha=0.05, seed=1, refit=1,
              qp_iters=120, verbose=False):
    """Returns dict with tau (nreps, 6) for [Avg, MV, Avg-or, MV-or, Avg-nv, MV-nv]."""
    mu = np.asarray(mu, float)
    p = len(mu)
    L = np.linalg.cholesky(Sigma)
    tk, q_mv, q_un = sched['tk'], sched['q_mv'], sched['q_un']
    t1 = tk[0]
    lam_mv_or = qp_exact_single(mu, Sigma, capped=False)
    lam_av_or = np.maximum(mu / np.diag(Sigma), 0.0)
    pen_mv_or = 0.5 * lam_mv_or @ Sigma @ lam_mv_or
    pen_av_or = 0.5 * lam_av_or ** 2 * np.diag(Sigma)
    thr = np.log(1.0 / alpha)
    rng = np.random.default_rng(seed)
    logp = np.log(p)

    S1 = np.zeros((nreps, p)); S2 = np.zeros((nreps, p, p))
    lgA = np.zeros((nreps, p)); lgAo = np.zeros((nreps, p)); lgAn = np.zeros((nreps, p))
    lM = np.zeros(nreps); lMo = np.zeros(nreps); lMn = np.zeros(nreps)
    tau = np.full((nreps, 6), NOT_REJ, dtype=np.int64)
    lam_mv = np.zeros((nreps, p)); lam_av = np.zeros((nreps, p))
    lam_mv_n = np.zeros((nreps, p)); lam_av_n = np.zeros((nreps, p))
    Sh = sh = Sh_n = sh_n = None
    lmax_c = lmax_n = None
    pen_mv = pen_av = pen_mv_n = pen_av_n = None
    k_cur = -1

    for t in range(1, n + 1):
        D = mu + rng.standard_normal((nreps, p)) @ L.T
        # epoch refit of the frozen penalties, from X_1 .. X_{t_k - 1}
        if t in tk:
            k_cur = int(np.searchsorted(tk, t))
            m = t - 1
            xbar = S1 / m
            Sfull = (S2 - m * xbar[:, :, None] * xbar[:, None, :]) / (m - 1)
            Sh_n = Sfull; sh_n = np.einsum('rii->ri', Sfull).copy()
            Sh = Sfull / q_mv[k_cur]; sh = sh_n / q_un[k_cur]
            lmax_n = np.linalg.eigvalsh(Sh_n)[:, -1]; lmax_c = lmax_n / q_mv[k_cur]
        if t >= t1 and ((t - t1) % refit == 0 or t in tk):
            muh = S1 / (t - 1)
            lam_mv = qp_polish(muh, Sh, qp_batch(muh, Sh, lam_mv, False, qp_iters, lmax_c), False)
            lam_mv_n = qp_polish(muh, Sh_n, qp_batch(muh, Sh_n, lam_mv_n, False, qp_iters, lmax_n), False)
            lam_av = np.maximum(muh / sh, 0.0)
            lam_av_n = np.maximum(muh / sh_n, 0.0)
            pen_mv = 0.5 * np.einsum('ri,rij,rj->r', lam_mv, Sh, lam_mv)
            pen_mv_n = 0.5 * np.einsum('ri,rij,rj->r', lam_mv_n, Sh_n, lam_mv_n)
            pen_av = 0.5 * lam_av ** 2 * sh
            pen_av_n = 0.5 * lam_av_n ** 2 * sh_n
        if t >= t1:
            lgA += lam_av * D - pen_av
            lM += np.einsum('ri,ri->r', lam_mv, D) - pen_mv
            lgAn += lam_av_n * D - pen_av_n
            lMn += np.einsum('ri,ri->r', lam_mv_n, D) - pen_mv_n
        lgAo += lam_av_or * D - pen_av_or
        lMo += D @ lam_mv_or - pen_mv_or
        eA = logsumexp(lgA, axis=1) - logp
        eAo = logsumexp(lgAo, axis=1) - logp
        eAn = logsumexp(lgAn, axis=1) - logp
        for j, lw in enumerate((eA, lM, eAo, lMo, eAn, lMn)):
            hit = (lw >= thr) & (tau[:, j] == NOT_REJ)
            tau[hit, j] = t
        S1 += D
        S2 += D[:, :, None] * D[:, None, :]
        if verbose and t % 200 == 0:
            print(f"  t={t}: rejected so far {(tau < NOT_REJ).mean(0).round(3)}", flush=True)
    return dict(tau=tau, lam_mv_or=lam_mv_or, lam_av_or=lam_av_or, mu=mu, Sigma=Sigma, n=n,
                alpha=alpha, nreps=nreps, refit=refit, seed=seed, sched=sched)


# ----------------------------------------------------------------------------
# summaries
# ----------------------------------------------------------------------------
def summarise(tau, cols=(0, 1, 2, 3)):
    """Rejection rates by the horizon and mean / median rejection times over the runs in
    which all strategies in `cols` reject."""
    T = tau[:, list(cols)]
    rej = (tau < NOT_REJ).mean(0)
    ok = (T < NOT_REJ).all(1)
    mean = T[ok].mean(0) if ok.any() else np.full(len(cols), np.nan)
    med = np.median(T[ok], axis=0) if ok.any() else np.full(len(cols), np.nan)
    return dict(rej=rej, mean=mean, median=med, ncomplete=int(ok.sum()))


def rejection_curve(tau_col, n):
    grid = np.arange(0, n + 1)
    t = tau_col[tau_col < NOT_REJ]
    return grid, np.searchsorted(np.sort(t), grid, side='right') / len(tau_col)
