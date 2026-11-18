"""
run_all.py -- runs the four simulation studies of the paper and stores the rejection times.

    python run_all.py bounded3 | bounded9 | gauss3 | gauss9   [nreps]

Outputs results/<study>.npz with the rejection-time matrices (one row per repetition,
columns as in simlib) and results/<study>_summary.txt.
"""
import sys, time, json
import numpy as np
from simlib import *

study = sys.argv[1]
nreps = int(sys.argv[2]) if len(sys.argv) > 2 else None
alpha = 0.05
a = 0.8
R9, _ = corr_p9()

# mean configurations for p = 9 (bounded magnitude, Gaussian magnitude)
patterns9 = {
    'pair_compl':   ([1, 4],                (.15, .22)),
    'pair_redund':  ([1, 2],                (.15, .22)),
    'six_compl':    ([1, 2, 3, 4, 5, 6],    (.12, .18)),
    'three_redund': ([1, 2, 3],             (.12, .18)),
    'nine_graded':  (None,                  None),
    'four_mixed':   ([1, 4, 7, 8],          (.13, .20)),
    'single_corr':  ([1],                   (.15, .24)),
    'single_iso':   ([9],                   (.15, .24)),
    'null_tied':    ([],                    (0., 0.)),
    'null_partial': (None,                  None),
}


def means9(regime):
    j = 0 if regime == 'bounded' else 1
    out = {}
    for nm, (idx, mag) in patterns9.items():
        mu = np.zeros(9)
        if nm == 'nine_graded':
            g = (.14, .11, .08) if regime == 'bounded' else (.22, .17, .12)
            mu = np.array(g * 3)
        elif nm == 'null_partial':
            mu[3:6] = -.10 if regime == 'bounded' else -.15
        else:
            for i in idx:
                mu[i - 1] = mag[j]
        out[nm] = mu
    return outX


scen3_bounded = [  # (name, mu, rho)
    ('three_equal_neg', (.10, .10, .10), -0.4),
    ('three_equal_ind', (.10, .10, .10), 0.0),
    ('three_equal_pos', (.10, .10, .10), 0.8),
    ('two_equal_ind',   (.12, .12, 0.0), 0.0),
    ('three_graded_neg', (.15, .10, .05), -0.4),
    ('single_ind',      (.15, 0.0, 0.0), 0.0),
    ('null_tied_ind',   (0., 0., 0.), 0.0),
    ('null_tied_pos',   (0., 0., 0.), 0.8),
    ('null_partial_ind', (0., 0., -.15), 0.0),
    ('null_partial_pos', (0., 0., -.15), 0.8),
]
scen3_gauss = [
    ('three_equal_neg', (.15, .15, .15), -0.4),
    ('three_equal_ind', (.15, .15, .15), 0.0),
    ('three_equal_pos', (.15, .15, .15), 0.8),
    ('two_equal_ind',   (.18, .18, 0.0), 0.0),
    ('three_graded_neg', (.24, .15, .08), -0.4),
    ('single_ind',      (.26, 0.0, 0.0), 0.0),
    ('null_tied_ind',   (0., 0., 0.), 0.0),
    ('null_tied_pos',   (0., 0., 0.), 0.8),
    ('null_partial_ind', (0., 0., -.20), 0.0),
    ('null_partial_pos', (0., 0., -.20), 0.8),
]

results = {}
lines = []
t_all = time.time()
if study == 'bounded3':
    nreps = nreps or 2000
    for s, (nm, mu, rho) in enumerate(scen3_bounded):
        t0 = time.time()
        out = run_bounded(np.array(mu), equicorr(3, rho), a=a, n=500, nreps=nreps, warmup=6,
                          alpha=alpha, seed=20260726 + 100 * s)
        results[nm] = out
        sm = summarise(out['tau'])
        line = (f"{nm:18s} mu={mu} rho={rho:+.1f} act={int((out['lam_mv_or']>1e-8).sum())} | rej Avg/MV/Avg-or/MV-or = "
                f"{'/'.join(f'{x:.3f}' for x in sm['rej'])} | mean tau = {'/'.join(f'{x:.1f}' for x in sm['mean'])}"
                f" | median = {'/'.join(f'{x:.0f}' for x in sm['median'])} | {time.time()-t0:.0f}s")
        print(line, flush=True); lines.append(line)
elif study == 'bounded9':
    nreps = nreps or 2000
    for s, (nm, mu) in enumerate(means9('bounded').items()):
        t0 = time.time()
        out = run_bounded(mu, R9, a=a, n=500, nreps=nreps, warmup=12, alpha=alpha, seed=20260723 + 1000 * s)
        results[nm] = out
        sm = summarise(out['tau'])
        line = (f"{nm:14s} nviol={int((mu>0).sum())} act={int((out['lam_mv_or']>1e-8).sum())} | rej Avg/MV/Avg-or/MV-or = "
                f"{'/'.join(f'{x:.3f}' for x in sm['rej'])} | mean tau = {'/'.join(f'{x:.1f}' for x in sm['mean'])}"
                f" | median = {'/'.join(f'{x:.0f}' for x in sm['median'])} | {time.time()-t0:.0f}s")
        print(line, flush=True); lines.append(line)
elif study == 'gauss3':
    nreps = nreps or 2000
    sched = epoch_schedule(1000, 3, t1=16, delta=0.05)
    lines.append(f"epochs {sched['tk'].tolist()} 1/q_mv {np.round(1/sched['q_mv'],2).tolist()} 1/q_un {np.round(1/sched['q_un'],2).tolist()}")
    print(lines[-1], flush=True)
    for s, (nm, mu, rho) in enumerate(scen3_gauss):
        t0 = time.time()
        out = run_gauss(np.array(mu), equicorr(3, rho), sched, n=1000, nreps=nreps, alpha=alpha,
                        seed=20260724 + 100 * s, refit=1)
        results[nm] = out
        sm = summarise(out['tau'])
        nv = (out['tau'][:, 4:] < NOT_REJ).mean(0)
        line = (f"{nm:18s} mu={mu} rho={rho:+.1f} act={int((out['lam_mv_or']>1e-8).sum())} | rej Avg/MV/Avg-or/MV-or = "
                f"{'/'.join(f'{x:.3f}' for x in sm['rej'][:4])} naive {nv[0]:.3f}/{nv[1]:.3f} | mean tau = "
                f"{'/'.join(f'{x:.1f}' for x in sm['mean'])} | median = {'/'.join(f'{x:.0f}' for x in sm['median'])} | {time.time()-t0:.0f}s")
        print(line, flush=True); lines.append(line)
elif study == 'gauss9':
    nreps = nreps or 2000
    sched = epoch_schedule(1000, 9, t1=16, delta=0.05)
    lines.append(f"epochs {sched['tk'].tolist()} 1/q_mv {np.round(1/sched['q_mv'],2).tolist()} 1/q_un {np.round(1/sched['q_un'],2).tolist()}")
    print(lines[-1], flush=True)
    for s, (nm, mu) in enumerate(means9('gauss').items()):
        t0 = time.time()
        out = run_gauss(mu, R9, sched, n=1000, nreps=nreps, alpha=alpha, seed=20260724 + 1000 * s, refit=1)
        results[nm] = out
        sm = summarise(out['tau'])
        nv = (out['tau'][:, 4:] < NOT_REJ).mean(0)
        line = (f"{nm:14s} nviol={int((mu>0).sum())} act={int((out['lam_mv_or']>1e-8).sum())} | rej Avg/MV/Avg-or/MV-or = "
                f"{'/'.join(f'{x:.3f}' for x in sm['rej'][:4])} naive {nv[0]:.3f}/{nv[1]:.3f} | mean tau = "
                f"{'/'.join(f'{x:.1f}' for x in sm['mean'])} | median = {'/'.join(f'{x:.0f}' for x in sm['median'])} | {time.time()-t0:.0f}s")
        print(line, flush=True); lines.append(line)
else:
    raise SystemExit("unknown study")

import os
os.makedirs('results', exist_ok=True)
np.savez(f'results/{study}.npz', **{f'tau_{nm}': out['tau'] for nm, out in results.items()},
         **{f'mu_{nm}': out['mu'] for nm, out in results.items()},
         **{f'lammvor_{nm}': out['lam_mv_or'] for nm, out in results.items()},
         **{f'lamavor_{nm}': out['lam_av_or'] for nm, out in results.items()},
         names=np.array(list(results.keys())), nreps=nreps, n=next(iter(results.values()))['n'])
with open(f'results/{study}_summary.txt', 'w') as f:
    f.write(f"{study}: nreps={nreps}, total {time.time()-t_all:.0f}s\n" + "\n".join(lines) + "\n")
print(f"done {study} in {time.time()-t_all:.0f}s")
