"""
Case study: multivariate betting versus the averaging construction on the Oregon Health
Insurance Experiment (12-month survey, eight endpoints), bounded construction.

Usage:  python3 run_casestudy.py [n_pairings] [n_permutations] [kappa] [pairing_horizon]   (defaults 200, 200, 0.5, 2500)

Strategies, exactly as in the simulation section of the paper (bounded case):
  MV   surrogate plug-in  lambda_t = argmax_{Lambda_kappa} lambda'mu_{t-1} - 1/2 lambda'M_{t-1} lambda,
       with the empirical first and second moments of the scaled differences X_1..X_{t-1},
       no bet before round p+4; e-process prod (1 + lambda_t' X_t).
  Avg  aGRAPA fraction lambda_{i,t} = 0 v mu_{i,t-1}/M_{ii,t-1} ^ kappa on every coordinate, the p
       e-processes combined by their arithmetic mean.
Coordinates are scaled by the exact ranges c_i (data/endpoints.csv), so |X_i| <= 1. The bets are
restricted to Lambda_kappa = {lambda >= 0, sum lambda <= kappa} with kappa < 1 (Appendix on regret):
with discrete coordinates a pair with X_i = -1 on every staked coordinate has positive probability,
and kappa < 1 keeps every factor 1 + lambda'X above 1 - kappa > 0.

Outputs (results/):
  paths.csv          log Avg, log MV and the p single log e-processes along the ordered pairing
  bets.csv           the MV bet and the Avg fractions at every round
  summary.txt        rejection rounds, final log-wealths, final bets, first-stake rounds
  pairings.csv       rejection rounds of MV and Avg over random pairings of the same respondents
  permutations.csv   maximal log-wealth of MV and Avg under random re-assignment of the arms
figures/ohie_paths.pdf  the figure of the paper
"""
import sys, os, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulations"))
from simlib import qp_solve

HERE = os.path.dirname(os.path.abspath(__file__))
NPAIR = int(sys.argv[1]) if len(sys.argv) > 1 else 200
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 200
KAPPA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
HPAIR = int(sys.argv[4]) if len(sys.argv) > 4 else 2500   # horizon used for the random-pairing runs   # bets restricted to Lambda_kappa = kappa * Lambda (no factor below 1 - kappa)
ALPHA = 0.05; THR = np.log(1 / ALPHA)
end = pd.read_csv(os.path.join(HERE, "data", "endpoints.csv"))
names = list(end.name); c = end.bound_c.values.astype(float); p = len(names)
pairs = pd.read_csv(os.path.join(HERE, "data", "ohie_pairs.csv"))
resp = pd.read_csv(os.path.join(HERE, "data", "ohie_respondents.csv")).dropna(subset=names)

def run(X, warmup=None, record=False):
    """X: (n, p) differences in original units. Returns dict of log-wealth paths and bets."""
    X = X / c; n = len(X); warmup = p + 4 if warmup is None else warmup
    S1 = np.zeros(p); S2 = np.zeros((p, p)); lam_mv = np.zeros(p)
    logA = np.zeros((n, p)); logM = np.zeros(n); bets = np.zeros((n, 2 * p))
    la = 0.0; lm = 0.0
    for t in range(1, n + 1):
        x = X[t - 1]
        if t > warmup:
            n0 = t - 1; muh = S1 / n0; Mh = S2 / n0
            # maximise over Lambda_kappa = kappa*Lambda: substitute lambda = kappa*v, v in Lambda
            v = qp_solve(KAPPA * muh[None], KAPPA**2 * Mh[None], (lam_mv / KAPPA)[None], capped=True)[0]
            lam_mv = KAPPA * v
            lam_av = np.clip(muh / (np.diag(Mh) + 1e-12), 0.0, KAPPA)
        else:
            lam_mv = np.zeros(p); lam_av = np.zeros(p)
        la = la + np.log1p(lam_av * x); lm = lm + np.log1p(lam_mv @ x)
        logA[t - 1] = la; logM[t - 1] = lm; bets[t - 1, :p] = lam_mv; bets[t - 1, p:] = lam_av
        S1 += x; S2 += np.outer(x, x)
    logAvg = np.logaddexp.reduce(logA, axis=1) - np.log(p)
    return dict(logAvg=logAvg, logMV=logM, logSingle=logA, bets=bets)

def hit(w): return int(np.argmax(w >= THR) + 1) if (w >= THR).any() else None

t0 = time.time()
X = pairs[names].values
res = run(X)
out = [f"OHIE case study: {len(X)} pairs, p = {p}, alpha = {ALPHA}, warm-up {p + 4} rounds, kappa = {KAPPA}"]
out.append(f"rejection round: Avg {hit(res['logAvg'])}, MV {hit(res['logMV'])}; singles " +
           ", ".join(f"{nm} {hit(res['logSingle'][:, i])}" for i, nm in enumerate(names)))
out.append(f"final log-wealth: Avg {res['logAvg'][-1]:.2f}, MV {res['logMV'][-1]:.2f}; singles " +
           ", ".join(f"{nm} {res['logSingle'][-1, i]:.2f}" for i, nm in enumerate(names)))
out.append("final MV bet (scaled coordinates): " + ", ".join(f"{nm} {res['bets'][-1, i]:.3f}" for i, nm in enumerate(names)))
out.append("final Avg fractions:               " + ", ".join(f"{nm} {res['bets'][-1, p + i]:.3f}" for i, nm in enumerate(names)))
first = [int(np.argmax(res['bets'][:, i] > 0) + 1) if (res['bets'][:, i] > 0).any() else None for i in range(p)]
out.append("first round with a positive MV stake: " + ", ".join(f"{nm} {f}" for nm, f in zip(names, first)))
# share of rounds (after warm-up) in which each coordinate carries a positive MV stake
share = (res['bets'][p + 4:, :p] > 0).mean(0)
out.append("share of rounds with positive MV stake: " + ", ".join(f"{nm} {s:.2f}" for nm, s in zip(names, share)))
# bet at a few checkpoints
for t in [50, 100, 200, 500, 1000, 2000, len(X)]:
    out.append(f"  MV bet at round {t:5d}: " + " ".join(f"{v:.3f}" for v in res['bets'][t - 1, :p]) + f"   | log Avg {res['logAvg'][t-1]:7.2f}  log MV {res['logMV'][t-1]:7.2f}")
pd.DataFrame({"round": pairs["round"], "logAvg": res["logAvg"], "logMV": res["logMV"],
              **{"log_" + nm: res["logSingle"][:, i] for i, nm in enumerate(names)}}).to_csv(os.path.join(HERE, "results", "paths.csv"), index=False)
pd.DataFrame(res["bets"], columns=["mv_" + nm for nm in names] + ["avg_" + nm for nm in names]).to_csv(os.path.join(HERE, "results", "bets.csv"), index=False)
print("\n".join(out), flush=True)

# ---- robustness: random pairings of the same respondents (return order kept within arm) ----
rng = np.random.default_rng(1)
B = resp[resp.arm == 1][names].values; A = resp[resp.arm == 0][names].values; m = min(len(A), len(B))
rows = []
for k in range(NPAIR):
    Xk = (B[rng.permutation(len(B))[:m]] - A[rng.permutation(len(A))[:m]])[:HPAIR]   # horizon HPAIR suffices: every pairing rejects long before
    rk = run(Xk)
    rows.append(dict(pairing=k, tau_avg=hit(rk["logAvg"]), tau_mv=hit(rk["logMV"]), final_avg=rk["logAvg"][-1], final_mv=rk["logMV"][-1]))
    if (k + 1) % 20 == 0: print(f"  pairings {k + 1}/{NPAIR}  ({time.time() - t0:.0f}s)", flush=True)
if NPAIR > 0:
    P = pd.DataFrame(rows); P.to_csv(os.path.join(HERE, "results", "pairings.csv"), index=False)
    q = lambda s: f"median {s.median():.0f}, IQR [{s.quantile(.25):.0f}, {s.quantile(.75):.0f}], range [{s.min():.0f}, {s.max():.0f}]"
    out.append(f"\nrandom pairings ({NPAIR}, horizon {HPAIR}): rejection round Avg {q(P.tau_avg.astype(float))}; MV {q(P.tau_mv.astype(float))}; "
               f"not rejected by the horizon: Avg {P.tau_avg.isna().sum()}, MV {P.tau_mv.isna().sum()}; "
               f"MV faster in {(P.tau_mv < P.tau_avg).mean():.3f} of pairings; ratio Avg/MV median {(P.tau_avg / P.tau_mv).median():.2f}")

# ---- null check: random re-assignment of the arms, then ordered pairs ----
rng = np.random.default_rng(int(os.environ.get('PERMSEED', 2)))
Y = resp[names].values; nB = int(resp.arm.sum())
rows = []
for k in range(NPERM):
    perm = rng.permutation(len(Y)); Bk = Y[perm[:nB]]; Ak = Y[perm[nB:]]; mk = min(len(Ak), len(Bk))
    rk = run(Bk[:mk] - Ak[:mk])
    rows.append(dict(perm=k, max_avg=rk["logAvg"].max(), max_mv=rk["logMV"].max(), final_avg=rk["logAvg"][-1], final_mv=rk["logMV"][-1]))
    if (k + 1) % 20 == 0: print(f"  permutations {k + 1}/{NPERM}  ({time.time() - t0:.0f}s)", flush=True)
if NPERM > 0:
  N = pd.DataFrame(rows); N.to_csv(os.path.join(HERE, "results", f"permutations_k{KAPPA}_s{os.environ.get('PERMSEED',2)}.csv"), index=False)
  out.append(f"permutation null ({NPERM}): fraction with max log-wealth >= log(1/alpha): Avg {(N.max_avg >= THR).mean():.3f}, MV {(N.max_mv >= THR).mean():.3f}; "
           f"median max log-wealth Avg {N.max_avg.median():.2f}, MV {N.max_mv.median():.2f}; median final Avg {N.final_avg.median():.2f}, MV {N.final_mv.median():.2f}")
open(os.path.join(HERE, "results", "summary.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out[-2:]))

# ---- figure ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8})
fig, ax = plt.subplots(figsize=(6.3, 3.0))
t = np.arange(1, len(X) + 1); T = 1500
for i in range(p): ax.plot(t[:T], res["logSingle"][:T, i], color="0.6", lw=0.7)
ax.plot(t[:T], res["logAvg"][:T], color="#2471a3", lw=1.6, label="Avg")
ax.plot(t[:T], res["logMV"][:T], color="#c0392b", lw=1.6, label="MV")
ax.axhline(THR, color="k", lw=0.7, ls="--"); ax.text(T * 0.97, THR + 0.6, r"$\log(1/\alpha)$", ha="right", fontsize=8)
ax.set_xlim(1, T); ax.set_ylim(-3, max(res["logMV"][:T].max(), res["logAvg"][:T].max()) + 2)
ax.set_xlabel("round $t$ (pair of respondents, in order of survey return)"); ax.set_ylabel("$\\log E_t$")
ax.legend(loc="upper left", frameon=False); plt.tight_layout()
plt.savefig(os.path.join(HERE, "figures", "ohie_paths.pdf")); plt.savefig(os.path.join(HERE, "figures", "ohie_paths.png"), dpi=200)
print(f"done in {time.time() - t0:.0f}s")
