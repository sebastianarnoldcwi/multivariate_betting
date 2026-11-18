"""Ordered-pairing paths of Avg, MV, UP and the eight single-coordinate e-processes for several caps;
draws figures/ohie_paths_k07.pdf (main text) and figures/ohie_paths_kappas.pdf (appendix, 2x2)."""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import qmc
from scipy.special import logsumexp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulations")); from simlib import qp_solve
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); N = 2 ** 14; THR = np.log(20)
end = pd.read_csv(os.path.join(HERE, "data", "endpoints.csv")); names = list(end.name); c = end.bound_c.values.astype(float); p = len(names)
X = pd.read_csv(os.path.join(HERE, "data", "ohie_pairs.csv"))[names].values / c
U = qmc.Sobol(d=p, scramble=True, seed=0).random(N); U.sort(axis=1); PTS = np.diff(np.concatenate([np.zeros((N, 1)), U], axis=1), axis=1)
def run(kappa):
    n = len(X); warm = p + 4; S1 = np.zeros(p); S2 = np.zeros((p, p)); lam = np.zeros(p); la = np.zeros(p); lm = 0.0; L = kappa * PTS; lw = np.zeros(N)
    out = np.empty((n, 3 + p))
    for t in range(1, n + 1):
        x = X[t - 1]
        if t > warm:
            muh = S1 / (t - 1); Mh = S2 / (t - 1)
            lam = kappa * qp_solve(kappa * muh[None], kappa ** 2 * Mh[None], (lam / kappa)[None], capped=True)[0]
            lav = np.clip(muh / (np.diag(Mh) + 1e-12), 0.0, kappa)
        else: lam = np.zeros(p); lav = np.zeros(p)
        la = la + np.log1p(lav * x); lm = lm + np.log1p(lam @ x); lw += np.log1p(L @ x)
        out[t - 1, 0] = logsumexp(la) - np.log(p); out[t - 1, 1] = lm; out[t - 1, 2] = logsumexp(lw) - np.log(N); out[t - 1, 3:] = la
        S1 += x; S2 += np.outer(x, x)
    return pd.DataFrame(out, columns=["Avg", "MV", "UP"] + ["log_" + nm for nm in names])
paths = {}
for k in [0.5, 0.6, 0.7, 0.8, 0.9]:
    f = os.path.join(HERE, "results", f"paths_all_k{k}.csv")
    if os.path.exists(f): paths[k] = pd.read_csv(f)
    else: paths[k] = run(k); paths[k].to_csv(f, index=False)
    h = lambda w: int(np.argmax(w >= THR) + 1) if (w >= THR).any() else None
    print(k, "Avg", h(paths[k].Avg.values), "MV", h(paths[k].MV.values), "UP", h(paths[k].UP.values), "final", paths[k][["Avg", "MV", "UP"]].iloc[-1].round(1).tolist(), flush=True)

def panel(ax, P, T, title=None, legend=False):
    t = np.arange(1, len(P) + 1)
    for nm in names: ax.plot(t[:T], P["log_" + nm].values[:T], color="0.6", lw=0.6)
    ax.plot(t[:T], P.Avg.values[:T], color="#2471a3", lw=1.5, label="Avg")
    ax.plot(t[:T], P.MV.values[:T], color="#c0392b", lw=1.5, label="MV")
    ax.plot(t[:T], P.UP.values[:T], color="#1e8449", lw=1.3, ls="--", label="UP")
    ax.axhline(THR, color="k", lw=0.7, ls="--"); ax.set_xlim(1, T)
    ax.set_ylim(-3, max(P.MV.values[:T].max(), P.UP.values[:T].max(), P.Avg.values[:T].max()) + 2)
    if title: ax.set_title(title, fontsize=9)
    if legend: ax.legend(loc="upper left", frameon=False)
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8})
fig, ax = plt.subplots(figsize=(6.3, 3.0)); T = 1500
panel(ax, paths[0.7], T, legend=True)
ax.set_xlabel("round $t$ (pair of respondents)"); ax.set_ylabel(r"$\log E_t$")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "figures", "ohie_paths_k07.pdf")); plt.savefig(os.path.join(HERE, "figures", "ohie_paths_k07.png"), dpi=170); plt.close()
fig, axs = plt.subplots(2, 2, figsize=(6.3, 5.2), sharex=True, sharey=True)
for ax, k in zip(axs.flat, [0.9, 0.8, 0.6, 0.5]): panel(ax, paths[k], T, title=rf"$\omega={k}$", legend=(k == 0.9))
for ax in axs[1]: ax.set_xlabel("round $t$")
for ax in axs[:, 0]: ax.set_ylabel(r"$\log E_t$")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "figures", "ohie_paths_kappas.pdf")); plt.savefig(os.path.join(HERE, "figures", "ohie_paths_kappas.png"), dpi=150); print("figures written")
