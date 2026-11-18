"""
MV, Avg and UP on the ordered pairing and on random pairings, for a chosen endpoint set and cap.

Usage:  python3 run_config.py <endpoints, comma-separated> <kappa> [n_pairings] [horizon] [tag]

Endpoint names as in data/endpoints.csv (visit, oop, borrow, phq2, baddays, fairpoor, unhappy, er);
'fin' = mean of oop and borrow (range [0,1]); 'ment' = mean of the four health items after
dividing each by its range (range [0,1]). Writes results/config_<tag>.txt.
"""
import sys, os, time
import numpy as np, pandas as pd
from scipy.stats import qmc
from scipy.special import logsumexp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulations"))
from simlib import qp_solve

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = sys.argv[1].split(","); KAPPA = float(sys.argv[2])
NPAIR = int(sys.argv[3]) if len(sys.argv) > 3 else 0
HPAIR = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
TAG = sys.argv[5] if len(sys.argv) > 5 else sys.argv[1].replace(",", "-") + f"_k{KAPPA}"
ALPHA = 0.05; THR = np.log(1 / ALPHA); N = 2 ** 14
end = pd.read_csv(os.path.join(HERE, "data", "endpoints.csv")).set_index("name")
resp = pd.read_csv(os.path.join(HERE, "data", "ohie_respondents.csv"))

# build endpoint columns (oriented, original units) and ranges
def build(df):
    out = pd.DataFrame(index=df.index); c = []
    for e in EPS:
        if e == "fin":
            out[e] = (df["oop"] + df["borrow"]) / 2; c.append(1.0)
        elif e == "ment":
            out[e] = sum(df[k] / end.loc[k, "bound_c"] for k in ["phq2", "baddays", "fairpoor", "unhappy"]) / 4; c.append(1.0)
        else:
            out[e] = df[e]; c.append(float(end.loc[e, "bound_c"]))
    return out, np.array(c)
R, c = build(resp); R["arm"] = resp.arm; R = R.dropna(); p = len(EPS)
B = R[R.arm == 1][EPS].values; A = R[R.arm == 0][EPS].values; m = min(len(A), len(B))
X0 = B[:m] - A[:m]                                   # ordered pairing (return order within arm)

U = qmc.Sobol(d=p, scramble=True, seed=0).random(N); U.sort(axis=1)
PTS = np.diff(np.concatenate([np.zeros((N, 1)), U], axis=1), axis=1)

def run(X):
    Xs = X / c; n = len(Xs); warm = int(os.environ.get('WARM', p + 4))
    S1 = np.zeros(p); S2 = np.zeros((p, p)); lam = np.zeros(p)
    la = np.zeros(p); lm = 0.0; L = KAPPA * PTS; lw = np.zeros(N)
    out = np.empty((n, 3))
    for t in range(1, n + 1):
        x = Xs[t - 1]
        if t > warm:
            muh = S1 / (t - 1); Mh = S2 / (t - 1)
            lam = KAPPA * qp_solve(KAPPA * muh[None], KAPPA ** 2 * Mh[None], (lam / KAPPA)[None], capped=True)[0]
            lav = np.clip(muh / (np.diag(Mh) + 1e-12), 0.0, KAPPA)
        else:
            lam = np.zeros(p); lav = np.zeros(p)
        la = la + np.log1p(lav * x); lm = lm + np.log1p(lam @ x); lw += np.log1p(L @ x)
        out[t - 1] = (logsumexp(la) - np.log(p), lm, logsumexp(lw) - np.log(N))
        S1 += x; S2 += np.outer(x, x)
    return out  # columns Avg, MV, UP

def hit(w, thr=THR): return int(np.argmax(w >= thr) + 1) if (w >= thr).any() else None

t0 = time.time(); W = run(X0); pd.DataFrame(W, columns=["Avg","MV","UP"]).to_csv(os.path.join(HERE, "results", f"paths_{TAG}.csv"), index=False)
lines = [f"endpoints {EPS} (p={p}), kappa={KAPPA}, pairs {m}"]
for a in (0.05, 1e-3, 1e-5):
    thr = np.log(1 / a); lines.append(f"  alpha={a:g}: rejection Avg {hit(W[:,0],thr)}, MV {hit(W[:,1],thr)}, UP {hit(W[:,2],thr)}")
lines.append(f"  final log-wealth Avg {W[-1,0]:.1f}, MV {W[-1,1]:.1f}, UP {W[-1,2]:.1f}; at round 1000: {W[999,0]:.1f}, {W[999,1]:.1f}, {W[999,2]:.1f}")
print("\n".join(lines), f"({time.time()-t0:.0f}s)", flush=True)
if NPAIR:
    rng = np.random.default_rng(int(os.environ.get('PSEED', 1))); rows = []
    for k in range(NPAIR):
        Xk = (B[rng.permutation(len(B))[:m]] - A[rng.permutation(len(A))[:m]])[:HPAIR]
        Wk = run(Xk); rows.append([hit(Wk[:, j]) for j in range(3)] + [Wk[-1, j] for j in range(3)])
    P = pd.DataFrame(rows, columns=["tau_avg", "tau_mv", "tau_up", "fin_avg", "fin_mv", "fin_up"])
    P.to_csv(os.path.join(HERE, "results", f"config_{TAG}_pairings.csv"), index=False)
    med = lambda s: f"{s.median():.0f} [{s.quantile(.25):.0f},{s.quantile(.75):.0f}]"
    lines.append(f"  random pairings ({NPAIR}, horizon {HPAIR}): median [IQR] rejection Avg {med(P.tau_avg.astype(float))}, MV {med(P.tau_mv.astype(float))}, UP {med(P.tau_up.astype(float))}; "
                 f"not rejected Avg {P.tau_avg.isna().sum()} MV {P.tau_mv.isna().sum()} UP {P.tau_up.isna().sum()}; MV<Avg {(P.tau_mv < P.tau_avg).mean():.2f}, UP<Avg {(P.tau_up < P.tau_avg).mean():.2f}, UP<MV {(P.tau_up < P.tau_mv).mean():.2f}; "
                 f"median final log-wealth at horizon Avg {P.fin_avg.median():.1f}, MV {P.fin_mv.median():.1f}, UP {P.fin_up.median():.1f}")
    print(lines[-1], f"({time.time()-t0:.0f}s)", flush=True)
open(os.path.join(HERE, "results", f"config_{TAG}.txt"), "w").write("\n".join(lines) + "\n")
