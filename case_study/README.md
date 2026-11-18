# Case study: Oregon Health Insurance Experiment

Code and derived data for the case-study section and its appendix of *Multivariate Betting: Leveraging the
Covariance for Sequential Mean Testing*. The analysis compares three betting strategies on the
twelve-month survey of the Oregon Health Insurance Experiment (OHIE): the multivariate plug-in
bet (MV), the averaging construction (Avg) and the uniform mixture (UP), all restricted to the
capped bet set `Lambda_kappa = kappa * Lambda` with `kappa = 0.7` in the main text and
`kappa in {0.5, 0.6, 0.8, 0.9}` in the appendix.

## Data

The raw public-use files (`oregonhie_descriptive_vars.dta`, `oregonhie_survey12m_vars.dta`) are
available from the OHIE data archive (Finkelstein et al., 2013) and are **not** included. The derived
files in `data/` are:

- `endpoints.csv` — the eight endpoints: name, variable in the public-use files, orientation, range
  (`bound_c`), arm means.
- `ohie_respondents.csv` — one row per single-person-household respondent to the twelve-month survey
  (16,702 rows): `arm` (1 = selected), survey-return order, and the eight oriented endpoint values
  (missing where the item was not answered).
- `ohie_pairs.csv` — the 6,609 oriented paired differences `Y^B - Y^A` in survey-return order
  (complete cases only), in original units; divide by `bound_c` to obtain `X_t in [-1, 1]^8`.
- `correlations.csv` — correlation matrix of the paired differences.
- `*_ext.csv` — the same with 22 endpoints (exploratory; not used in the paper).

## Scripts

All scripts import `qp_solve` from `../simulations/simlib.py` (the simulation code of the paper) and need `numpy`, `scipy`, `pandas` and `matplotlib`.

- `extract_pairs.py` — builds `data/` from the raw public-use files.
- `make_paths_figs.py` — ordered-pairing e-processes of Avg, MV, UP and the eight single-coordinate
  e-processes for `kappa in {0.5, ..., 0.9}` (`results/paths_all_k*.csv`); draws
  `figures/ohie_paths_k07.pdf` (main-text figure) and `figures/ohie_paths_kappas.pdf` (appendix figure).
- `run_config.py <endpoints> <kappa> [n_pairings] [horizon] [tag]` — Avg, MV and UP on the ordered
  pairing and on random pairings for any endpoint subset and cap. Environment variables: `PSEED`
  (pairing seed, default 1), `WARM` (warm-up rounds, default `p + 4`). Writes
  `results/config_<tag>.txt`, `results/paths_<tag>.csv`, `results/config_<tag>_pairings.csv`.
- `run_casestudy.py [n_pairings] [n_perms] [kappa] [horizon]` — MV and Avg with random pairings and the
  permutation null (lottery outcome re-assigned; `PERMSEED` selects the seed).
- `run_up.py [N] [n_pairings] [n_perms] [kappa]` — the mixture UP with `N` scrambled-Sobol points on
  `Lambda_kappa`, random pairings and permutation null.

## Results used in the paper (`results/`)

- `config_p8_k{0.5,...,0.9}.txt` — ordered pairing: rejection rounds at three levels and final
  log-wealth (upper part of the cap-sensitivity table).
- `config_p8_k{kappa}_pairings100.csv` — the same 100 random pairings (seeds 1 and 2, 50 each,
  horizon 2,000) for every cap (lower part of the cap-sensitivity table; `_s1`/`_s2` are the two chunks).
- `permutations_k0.7_100.csv`, `up_permutations_k0.7.csv` — permutation null at `kappa = 0.7`
  (100 re-assignments; MV 5 %, UP 5 %, Avg 0 % exceed `log(1/alpha)` somewhere along the sequence).
- `config_p8_k0.9_w*.txt` — warm-up sensitivity (`WARM = 12, 25, 50, 100, 150, 200, 300`).
- `paths_all_k*.csv` — the paths plotted in the figures (columns `Avg`, `MV`, `UP`, `log_<endpoint>`).

Runtime: one random pairing of horizon 2,000 takes about 8 s for the three strategies; the
full ordered sequence about 20 s.

## Reproducing

```
python3 make_paths_figs.py
for k in 0.5 0.6 0.7 0.8 0.9; do
  PSEED=1 python3 run_config.py visit,oop,borrow,phq2,baddays,fairpoor,unhappy,er $k 50 2000 p8_k${k}_s1
  PSEED=2 python3 run_config.py visit,oop,borrow,phq2,baddays,fairpoor,unhappy,er $k 50 2000 p8_k${k}_s2
done
for s in 2 3 4 5; do PERMSEED=$s python3 run_casestudy.py 0 25 0.7; done
python3 run_up.py 16384 0 100 0.7
```

## References

Finkelstein, A., Taubman, S., Wright, B., Bernstein, M., Gruber, J., Newhouse, J. P., Allen, H.,
Baicker, K., and the Oregon Health Study Group (2012). The Oregon Health Insurance Experiment:
evidence from the first year. *Quarterly Journal of Economics*, 127(3), 1057–1106.
Data: Oregon Health Insurance Experiment public-use data, https://www.nber.org/programs-projects/projects-and-centers/oregon-health-insurance-experiment.
