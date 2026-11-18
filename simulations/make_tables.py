"""
make_tables.py -- LaTeX table bodies and summary numbers from results/*.npz.
"""
import numpy as np, json, sys
from simlib import *
from predictions import *

alpha = 0.05
R9, _ = corr_p9()
titles9 = {
    'pair_compl': 'two violations, complementary', 'pair_redund': 'two violations, redundant',
    'six_compl': 'six violations, complementary', 'three_redund': 'three violations, redundant',
    'nine_graded': 'nine graded violations, mixed', 'four_mixed': 'four violations, mixed',
    'single_corr': 'single violation, correlated', 'single_iso': 'single violation, isolated',
    'null_tied': 'tied null', 'null_partial': 'partial null'}
rho3 = {'three_equal_neg': -0.4, 'three_equal_ind': 0.0, 'three_equal_pos': 0.8, 'two_equal_ind': 0.0,
        'three_graded_neg': -0.4, 'single_ind': 0.0, 'null_tied_ind': 0.0, 'null_tied_pos': 0.8,
        'null_partial_ind': 0.0, 'null_partial_pos': 0.8}


def load(study):
    d = np.load(f'results/{study}.npz', allow_pickle=True)
    names = list(d['names'])
    out = {}
    for nm in names:
        out[nm] = dict(tau=d[f'tau_{nm}'], mu=d[f'mu_{nm}'], lam_mv_or=d[f'lammvor_{nm}'], lam_av_or=d[f'lamavor_{nm}'])
    return out, int(d['n']), int(d['nreps'])


def fmt_mu(mu):
    return '(' + ','.join(f'{x:g}' for x in mu) + ')'


def fmt_rho(r):
    return f'{r:g}' if r != 0 else '0'


def table_p3(study, with_naive=False):
    res, n, nreps = load(study)
    rows = []
    nulls = []
    for nm, r in res.items():
        sm = summarise(r['tau'])
        rho = rho3[nm]
        if nm.startswith('null'):
            nulls.append((nm, rho, sm['rej']))
            continue
        cells = ' & '.join(f'{x:.1f}' for x in sm['mean'])
        note = ''
        rows.append(f"${fmt_mu(r['mu'])}$ & ${fmt_rho(rho)}$ & {cells}\\\\" + note)
        print(f"  {nm:18s} rej={np.round(sm['rej'][:4],3)}  ncomplete={sm['ncomplete']}")
    print("\n".join(rows))
    print("NULLS:")
    for nm, rho, rej in nulls:
        print(f"  {nm:18s} rho={rho:+.1f} Type-I Avg/MV = {rej[0]:.3f}/{rej[1]:.3f}" + (f"  naive {rej[4]:.3f}/{rej[5]:.3f}" if with_naive else ''))
    return rows


def table_p9(study, gauss=False):
    res, n, nreps = load(study)
    rows = []
    sched = epoch_schedule(1000, 9, t1=16, delta=0.05) if gauss else None
    for nm, r in res.items():
        sm = summarise(r['tau'])
        mu = r['mu']
        if nm.startswith('null'):
            rej = sm['rej']
            line = f"{titles9[nm]}: anytime Type-I & & {rej[0]:.3f} & {rej[1]:.3f} & & \\\\"
            if gauss:
                line += f"\n{titles9[nm]}, naive plug-in & & {rej[4]:.3f} & {rej[5]:.3f} & & \\\\"
            rows.append(line)
            continue
        nact = int((r['lam_mv_or'] > 1e-8).sum())
        m = sm['mean']
        if gauss:
            pr = predictions_gauss(mu, R9, r['lam_mv_or'], r['lam_av_or'], sched, 9)
            pa = f"{m[2]:.1f}" + (f" ({pr['pred_av_or']:.1f})" if np.isfinite(pr['pred_av_or']) else '')
            line = (f"{titles9[nm]} & {nact} & {m[0]:.1f} & {m[1]:.1f} ({pr['pred_mv']:.0f}) & {pa} & {m[3]:.1f} ({pr['pred_mv_or']:.1f})\\\\")
            extra = f"ellG={pr['ellG']:.4f} pred_av(corr)={pr['pred_av']}"
        else:
            pr = predictions_bounded(mu, R9, r['lam_mv_or'], r['lam_av_or'], 9)
            pa = f"{m[2]:.1f}" + (f" ({pr['pred_av']:.1f})" if np.isfinite(pr['pred_av']) else '')
            line = f"{titles9[nm]} & {nact} & {m[0]:.1f} & {m[1]:.1f} & {pa} & {m[3]:.1f} ({pr['pred_mv']:.1f})\\\\"
            extra = f"ell_mv={pr['ell_mv']:.4f} ell_av={pr['ell_av']}"
        rows.append(line)
        print(f"  {nm:14s} rej={np.round(sm['rej'][:4],3)} ncomplete={sm['ncomplete']} ratio Avg/MV={m[0]/m[1]:.2f} or={m[2]/m[3]:.2f} | {extra} | lam_mv_or={np.round(r['lam_mv_or'],3)}")
    print("\n".join(rows))
    return rows


if __name__ == '__main__':
    for study in sys.argv[1:]:
        print(f"===== {study} =====")
        if study.endswith('3'):
            table_p3(study, with_naive=study.startswith('gauss'))
        else:
            table_p9(study, gauss=study.startswith('gauss'))
