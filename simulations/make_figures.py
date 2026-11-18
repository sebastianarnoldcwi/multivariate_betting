"""
make_figures.py -- rejection-curve figures (empirical cdf of the rejection time) for the appendix.
"""
import numpy as np, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simlib import *
from make_tables import load, titles9, rho3

plt.rcParams.update({'font.family': 'serif', 'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
                     'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42})
COL = {'Avg': '#2471a3', 'MV': '#c0392b', 'UP': '#1e8449'}
STYLE = [('Avg', 0, '-'), ('MV', 1, '-'), ('Avg', 2, ':'), ('MV', 3, ':')]
STYLE_UP_B = [('UP', 4, '-')]              # bounded: finite uniform mixture
STYLE_UP_G = [('UP', 6, '-'), ('UP', 7, ':')]   # Gaussian: UP (scaled, corrected) solid, UP-or dotted
os.makedirs('figures', exist_ok=True)


def panel(ax, tau, n, title, labels, style):
    for name, j, ls in style:
        grid, y = rejection_curve(tau[:, j], n)
        ax.plot(grid, y, color=COL[name], ls=ls, lw=1.4, label=labels[j])
    ax.set_xlim(0, n); ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.grid(alpha=0.25, lw=0.5)


def figure(study, panels, titles, labels, fname, ncol=2):
    res, n, nreps = load(study)
    panels = [nm for nm in panels if nm in res]
    gauss = study.startswith('gauss')
    up = np.load(f'results/up_{study}.npz', allow_pickle=True) if os.path.exists(f'results/up_{study}.npz') else None
    style = list(STYLE)
    if up is not None:
        for nm in panels:
            tu = up[f'tau_{nm}']
            res[nm]['tau'] = np.column_stack([res[nm]['tau'], tu if tu.ndim == 2 else tu[:, None]])
        style += STYLE_UP_G if gauss else STYLE_UP_B
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.1, 1.95 * nrow), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for k, nm in enumerate(panels):
        ax = axes[k // ncol, k % ncol]
        panel(ax, res[nm]['tau'], n, titles[nm], labels, style)
        if k // ncol == nrow - 1:
            ax.set_xlabel('$t$')
        if k % ncol == 0:
            ax.set_ylabel('rejected by $t$')
    for k in range(len(panels), nrow * ncol):
        axes[k // ncol, k % ncol].axis('off')
    axes[0, 0].legend(loc='lower right', frameon=False)
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    fig.savefig(fname)
    plt.close(fig)
    print('wrote', fname)


p3_panels = ['three_equal_neg', 'three_equal_ind', 'three_equal_pos', 'two_equal_ind', 'three_graded_neg', 'single_ind']


def p3_titles(study):
    res, n, nreps = load(study)
    out = {}
    for nm in p3_panels:
        mu = res[nm]['mu']
        out[nm] = r'$\mu=(' + ','.join(f'{x:g}' for x in mu) + r')$, $\rho=' + f'{rho3[nm]:g}' + '$'
    return out


p9_panels = ['pair_compl', 'pair_redund', 'six_compl', 'three_redund', 'nine_graded', 'four_mixed', 'single_corr', 'single_iso']
lab_b = ['Avg', 'MV', 'Avg-or', 'MV-or', 'UP']
lab_g = ['Avg (corrected)', 'MV (corrected)', 'Avg-or', 'MV-or', 'Avg-nv', 'MV-nv', 'UP (corrected)', 'UP-or']

if __name__ == '__main__':
    todo = sys.argv[1:] or ['bounded3', 'bounded9', 'gauss3', 'gauss9']
    if 'bounded3' in todo:
        figure('bounded3', p3_panels, p3_titles('bounded3'), lab_b, 'figures/rejection_bounded_p3.pdf')
    if 'gauss3' in todo:
        figure('gauss3', p3_panels, p3_titles('gauss3'), lab_g, 'figures/rejection_gauss_p3.pdf')
    if 'bounded9' in todo:
        figure('bounded9', p9_panels, titles9, lab_b, 'figures/rejection_bounded_p9.pdf')
    if 'gauss9' in todo:
        figure('gauss9', p9_panels, titles9, lab_g, 'figures/rejection_gauss_p9.pdf')
