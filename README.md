# Multivariate Betting: Leveraging the Covariance for Sequential Mean Testing

Code, stored results and figures for

> S. Arnold, P. Masini and J. Ziegel (2026). *Multivariate Betting: Leveraging the Covariance for
> Sequential Mean Testing.*

The paper studies sequential tests of a multivariate mean built from betting (e-process)
strategies that exploit the covariance between the coordinates, and compares the joint
plug-in bet (MV) with the coordinatewise averaging construction (Avg) and the uniform mixture
(UP) in simulations and in a case study on the Oregon Health Insurance Experiment.

## Contents

| Folder | What it holds |
|---|---|
| `simulations/` | NumPy implementation of all strategies (`simlib.py`, `uplib.py`), the four simulation studies (bounded and Gaussian, `p = 3` and `p = 9`), the stored results (`results/*.npz`), and the scripts that produce the tables, the rejection-time figures, the theoretical predictions and the run-time table. See `simulations/README.md`. |
| `case_study/` | The Oregon Health Insurance Experiment case study: derived data (oriented endpoints of the 16,702 single-person-household respondents and the 6,609 paired differences), the scripts for the ordered pairing, the random pairings, the permutation null and the cap sensitivity, the stored results and the figure script. See `case_study/README.md`. |
| `figures/` | Every figure of the paper as included in the manuscript: the TikZ source and PDF of the projection figure, the four rejection-time figures of the simulation appendix, and the two e-process figures of the case study. |

The raw public-use files of the Oregon Health Insurance Experiment are not included; see
`case_study/README.md` for where to obtain them and how `extract_pairs.py` turns them into the
derived files that are included.

## Requirements

Python 3.10 or newer with

    pip install -r requirements.txt

(`numpy`, `scipy`, `matplotlib`; the case study additionally uses `pandas`). The projection
figure is compiled with `pdflatex figures/plot_projection.tex`.

## Reproducing the paper

Every run is seeded, so the stored results are reproduced exactly. Approximate run times on one
core are given in the two sub-folder READMEs; the whole simulation study takes about two hours,
the case study about one and a half.

    cd simulations
    python run_all.py bounded3 && python run_all.py bounded9
    python run_all.py gauss3   && python run_all.py gauss9
    python run_up.py bounded3  && python run_up.py bounded9
    python run_up.py gauss3    && python run_up.py gauss9
    python make_tables.py bounded3 bounded9 gauss3 gauss9
    python make_figures.py && python predictions.py && python timing.py

    cd ../case_study
    python make_paths_figs.py                      # ordered pairing for every cap, both figures
    for k in 0.5 0.6 0.7 0.8 0.9; do               # random pairings (same 100 for every cap)
      PSEED=1 python run_config.py visit,oop,borrow,phq2,baddays,fairpoor,unhappy,er $k 50 2000 p8_k${k}_s1
      PSEED=2 python run_config.py visit,oop,borrow,phq2,baddays,fairpoor,unhappy,er $k 50 2000 p8_k${k}_s2
    done
    for s in 2 3 4 5; do PERMSEED=$s python run_casestudy.py 0 25 0.7; done   # permutation null, MV and Avg
    python run_up.py 16384 0 100 0.7                                           # permutation null, UP

The tables and figures only need the stored results: `make_tables.py`, `make_figures.py` and
`make_paths_figs.py` read `results/` and do not rerun anything that is already there.

## Licence

MIT, see `LICENSE`.
