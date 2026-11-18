"""
run_up.py -- universal-portfolio / mixture strategies on the same data streams as run_all.py.

    python run_up.py bounded3 | bounded9 | gauss3 | gauss9   [nreps] [log2N] [dirichlet_a] [gauss_prior_rate]

Outputs results/up_<study>[tag].npz with tau_<name> (bounded: (nreps,), gauss: (nreps, 2) for
[UP corrected, UP oracle]) and results/up_<study>[tag]_summary.txt (including run times).
Bounded prior: Dirichlet(dirichlet_a) on the simplex (default 1 = uniform on Lambda; tag _a0.5).
Gaussian prior: i.i.d. Exp(rate) coordinates with rate p by default (total mean stake one, no tag);
any other rate r gives the tag _exp<r>; rate 1 reproduces the prior-sensitivity comparison
reported in the paper (files up_gauss*_exp1.npz).
"""
import sys, time, os
import numpy as np
from simlib import *
from uplib import *

study = sys.argv[1]
nreps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
log2N = int(sys.argv[3]) if len(sys.argv) > 3 else None
prior_a = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0      # Dirichlet parameter of the bounded prior
prior_rate = sys.argv[5] if len(sys.argv) > 5 else 'p'          # Gaussian prior: i.i.d. Exp(rate) coordinates, 'p' (default) or a number
tag = '' if prior_a == 1.0 else f'_a{prior_a:g}'
alpha = 0.05
a = 0.8
R9, _ = corr_p9()

# same scenario lists and seeds as run_all.py
sys.argv = ['run_all.py', 'none']
import importlib.util
spec = importlib.util.spec_from_file_location('ra', 'run_all.py')
src = open('run_all.py').read().split("results = {}")[0]
ns = {}
exec(src, ns)
means9, scen3_bounded, scen3_gauss = ns['means9'], ns['scen3_bounded'], ns['scen3_gauss']

results = {}
lines = []
t_all = time.time()


def fmt(tau_col):
    ok = tau_col < NOT_REJ
    return f"rej {ok.mean():.3f} mean tau {tau_col[ok].mean():.1f} median {np.median(tau_col[ok]):.0f}"


if study in ('bounded3', 'bounded9'):
    p = 3 if study == 'bounded3' else 9
    N = 2 ** (log2N or (12 if p == 3 else 14))
    pts = simplex_points(N, p, seed=1, a=prior_a)
    lines.append(f"{study}: UP = Dirichlet({prior_a:g}) mixture over N={N} quasi-random points of Lambda, nreps={nreps}")
    print(lines[-1], flush=True)
    if p == 3:
        scen = [(nm, np.array(mu), equicorr(3, rho), 20260726 + 100 * s) for s, (nm, mu, rho) in enumerate(scen3_bounded)]
    else:
        scen = [(nm, mu, R9, 20260723 + 1000 * s) for s, (nm, mu) in enumerate(means9('bounded').items())]
    for nm, mu, R, seed in scen:
        t0 = time.time()
        data = gen_bounded_data(mu, R, a, 500, nreps, seed)
        r = up_bounded(data, pts, alpha=alpha, chunk=500)
        results[nm] = r
        line = f"{nm:18s} UP: {fmt(r['tau'])} | {r['seconds']:.0f}s UP ({time.time()-t0:.0f}s incl. data)"
        print(line, flush=True); lines.append(line)
    np.savez(f'results/up_{study}{tag}.npz', **{f'tau_{nm}': r['tau'] for nm, r in results.items()},
             **{f'logE_{nm}': r['logE_end'] for nm, r in results.items()},
             **{f'sec_{nm}': r['seconds'] for nm, r in results.items()},
             names=np.array(list(results.keys())), nreps=nreps, N=N, n=500)
elif study in ('gauss3', 'gauss9'):
    p = 3 if study == 'gauss3' else 9
    N = 2 ** (log2N or (12 if p == 3 else 14))
    sched = epoch_schedule(1000, p, t1=16, delta=0.05)
    N = 256   # QMC points of Genz's orthant-probability estimator
    rate = float(p) if prior_rate == 'p' else float(prior_rate)
    tag = '' if prior_rate == 'p' else f'_exp{prior_rate}'
    lines.append(f"{study}: UP = mixture over the orthant under i.i.d. Exp({rate:g}) coordinates (mean stake "
                 f"{1/rate:.3g} per coordinate), exact up to the Gaussian orthant probability (Genz, {N} QMC "
                 f"points), nreps={nreps}; columns [UP-sc (scaled bets, epoch penalty as MV), UP-or (true Sigma)]")
    print(lines[-1], flush=True)
    if p == 3:
        scen = [(nm, np.array(mu), equicorr(3, rho), 20260724 + 100 * s) for s, (nm, mu, rho) in enumerate(scen3_gauss)]
    else:
        scen = [(nm, mu, R9, 20260724 + 1000 * s) for s, (nm, mu) in enumerate(means9('gauss').items())]
    for nm, mu, Sigma, seed in scen:
        t0 = time.time()
        data = gen_gauss_data(mu, Sigma, 1000, nreps, seed)
        r = up_gauss_exact(data, sched, Sigma, alpha=alpha, n_q=N, chunk=250, rate=rate)
        results[nm] = r
        line = f"{nm:18s} UP-sc: {fmt(r['tau'][:, 0])} | UP-or: {fmt(r['tau'][:, 1])} | {r['seconds']:.0f}s UP"
        print(line, flush=True); lines.append(line)
    np.savez(f'results/up_{study}{tag}.npz', **{f'tau_{nm}': r['tau'] for nm, r in results.items()},
             **{f'logE_{nm}': r['logE_end'] for nm, r in results.items()},
             **{f'sec_{nm}': r['seconds'] for nm, r in results.items()},
             names=np.array(list(results.keys())), nreps=nreps, N=N, n=1000)
else:
    raise SystemExit("unknown study")

with open(f'results/up_{study}{tag}_summary.txt', 'w') as f:
    f.write(f"{study}: nreps={nreps}, N={N}, total {time.time()-t_all:.0f}s\n" + "\n".join(lines) + "\n")
print(f"done {study} in {time.time()-t_all:.0f}s")
