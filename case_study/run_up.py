"""
The mixture strategy UP on the OHIE case study.

Usage:  python3 run_up.py [N] [n_pairings] [n_permutations] [kappa]   (defaults 16384, 200, 100, 0.5)

UP is the mixture e-process of the paper (Appendix on the mixture strategy, bounded case): the
uniform average over N quasi-random points lambda_1..lambda_N of the bet set of the e-processes
prod_s (1 + lambda_k' X_s). Two bet sets are used: Lambda_kappa with kappa = 1/2, the set of the
plug-in strategies in the case study, and the full set Lambda (kappa = 1). Points are uniform on
Lambda_kappa (scrambled Sobol points mapped to the simplex by the spacings construction).

Outputs (results/): up_paths.csv (log UP along the ordered pairing, both kappa), up_summary.txt,
up_pairings.csv and up_permutations.csv (same random pairings and re-assignments as
run_casestudy.py: seeds 1 and 2).
"""
import sys, os, time
import numpy as np, pandas as pd
from scipy.stats import qmc
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2 ** 14
NPAIR = int(sys.argv[2]) if len(sys.argv) > 2 else 200
NPERM = int(sys.argv[3]) if len(sys.argv) > 3 else 100
HPAIR = 2500
KUP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
ALPHA = 0.05; THR = np.log(1 / ALPHA)
end = pd.read_csv(os.path.join(HERE, "data", "endpoints.csv"))
names = list(end.name); c = end.bound_c.values.astype(float); p = len(names)
pairs = pd.read_csv(os.path.join(HERE, "data", "ohie_pairs.csv"))
resp = pd.read_csv(os.path.join(HERE, "data", "ohie_respondents.csv")).dropna(subset=names)

def simplex_points(N, p, seed=0):
    """uniform points on {lambda >= 0, sum lambda <= 1}: spacings of p sorted uniforms in [0,1]
    give a uniform point on the (p+1)-simplex; drop the last coordinate."""
    U = qmc.Sobol(d=p, scramble=True, seed=seed).random(N)
    U.sort(axis=1)
    V = np.diff(np.concatenate([np.zeros((N, 1)), U], axis=1), axis=1)   # p spacings, sum <= 1
    return V

PTS = simplex_points(N, p)

def run_up(X, kappa):
    Xs = X / c; L = kappa * PTS                                 # (N, p)
    lw = np.zeros(N); out = np.empty(len(Xs))
    for t in range(len(Xs)):
        lw += np.log1p(L @ Xs[t])
        out[t] = logsumexp(lw) - np.log(N)
    return out

def hit(w): return int(np.argmax(w >= THR) + 1) if (w >= THR).any() else None

t0 = time.time(); X = pairs[names].values
res = {k: run_up(X, k) for k in (KUP, 1.0)}
lines = [f"UP on the OHIE case study: {len(X)} pairs, p = {p}, N = {N} points, alpha = {ALPHA}"]
for k, w in res.items():
    lines.append(f"kappa = {k}: rejection round {hit(w)}; log-wealth at rounds 300/500/1000/1500/6609: " +
                 " ".join(f"{w[t-1]:.2f}" for t in (300, 500, 1000, 1500, len(X))) +
                 "; rejection at alpha 0.01/1e-3/1e-5: " + "/".join(str(int(np.argmax(w >= np.log(1/a)) + 1) if (w >= np.log(1/a)).any() else None) for a in (0.01, 1e-3, 1e-5)))
pd.DataFrame({"round": pairs["round"], "logUP_k05": res[KUP], "logUP_k1": res[1.0]}).to_csv(os.path.join(HERE, "results", f"up_paths_k{KUP}.csv"), index=False)
print("\n".join(lines), f"({time.time()-t0:.0f}s)", flush=True)

rng = np.random.default_rng(1)
B = resp[resp.arm == 1][names].values; A = resp[resp.arm == 0][names].values; m = min(len(A), len(B))
rows = []
for k in range(NPAIR):
    Xk = (B[rng.permutation(len(B))[:m]] - A[rng.permutation(len(A))[:m]])[:HPAIR]
    w = run_up(Xk, KUP); rows.append(dict(pairing=k, tau_up=hit(w), final_up=w[-1]))
if NPAIR:
    P = pd.DataFrame(rows); P.to_csv(os.path.join(HERE, "results", f"up_pairings_k{KUP}.csv"), index=False)
    s = P.tau_up.astype(float)
    lines.append(f"random pairings ({NPAIR}, horizon {HPAIR}, kappa {KUP}): UP rejection round median {s.median():.0f}, IQR [{s.quantile(.25):.0f}, {s.quantile(.75):.0f}], range [{s.min():.0f}, {s.max():.0f}], not rejected {P.tau_up.isna().sum()}")
    mv = pd.read_csv(os.path.join(HERE, "results", "pairings.csv"))
    lines.append(f"   versus the same pairings: UP faster than MV in {(P.tau_up < mv.tau_mv).mean():.3f}, than Avg in {(P.tau_up < mv.tau_avg).mean():.3f}; median ratios Avg/UP {(mv.tau_avg/P.tau_up).median():.2f}, UP/MV {(P.tau_up/mv.tau_mv).median():.2f}")
    print(lines[-2], "\n", lines[-1], flush=True)

rng = np.random.default_rng(2)
Y = resp[names].values; nB = int(resp.arm.sum()); rows = []
for k in range(NPERM):
    perm = rng.permutation(len(Y)); Bk = Y[perm[:nB]]; Ak = Y[perm[nB:]]; mk = min(len(Ak), len(Bk))
    w = run_up(Bk[:mk] - Ak[:mk], KUP); rows.append(dict(perm=k, max_up=w.max(), final_up=w[-1]))
if NPERM:
    Nn = pd.DataFrame(rows); Nn.to_csv(os.path.join(HERE, "results", f"up_permutations_k{KUP}.csv"), index=False)
    lines.append(f"permutation null ({NPERM}, kappa {KUP}): fraction with max log UP >= log(1/alpha): {(Nn.max_up >= THR).mean():.3f}; median max {Nn.max_up.median():.2f}; median final {Nn.final_up.median():.2f}")
    print(lines[-1], flush=True)
open(os.path.join(HERE, "results", f"up_summary_k{KUP}.txt"), "w").write("\n".join(lines) + "\n")
print(f"done in {time.time()-t0:.0f}s")
