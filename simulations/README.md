# Simulation code for "Multivariate Betting: Leveraging the Covariance for Sequential Mean Testing"

Code and stored results for the simulation study (Section "Simulations" and the appendices on the
mixture strategy and the additional simulation material) of

> S. Arnold, P. Masini and J. Ziegel (2026). *Multivariate Betting: Leveraging the Covariance for
> Sequential Mean Testing.*

All strategies are implemented in NumPy and vectorised over the Monte Carlo repetitions; the
batched quadratic-programme solver is exact (verified against SLSQP/NNLS to machine precision).

## Requirements

Python 3.10 or newer with `numpy`, `scipy` and `matplotlib`:

    pip install -r requirements.txt

## Files

* `simlib.py` — shared library: data generators for the bounded and the Gaussian design, batched
  quadratic programmes (projected gradient plus exact KKT polish), the plug-in strategies MV and
  Avg with their oracles, the Wishart/chi-square epoch correction, and summaries.
* `uplib.py` — the mixture (universal-portfolio) strategies: data generators reproducing the exact
  streams of `run_all.py`, quasi-random point sets on the simplex (uniform and Dirichlet(1/2)),
  the finite uniform mixture for bounded vectors, and the exact Gaussian mixture under independent
  exponential priors, evaluated via Genz's algorithm for the orthant probability.
* `run_all.py` — runs the four studies (`bounded3`, `bounded9`, `gauss3`, `gauss9`) for MV, Avg,
  their oracles and, in the Gaussian case, the naive plug-ins; writes `results/<study>.npz` and a
  text summary.
* `run_up.py` — runs the mixture strategies on the same data streams; writes
  `results/up_<study>[tag].npz`. Optional arguments select the Dirichlet(1/2) prior in the bounded
  case (tag `_a0.5`) and the rate of the exponential prior in the Gaussian case (default `p`;
  rate 1 gives the prior-sensitivity comparison reported in the paper, tag `_exp1`).
* `make_tables.py` — table bodies and summary numbers from `results/*.npz`.
* `make_figures.py` — the rejection-time figures of the appendix (empirical cdfs).
* `predictions.py` — the theoretical predictions set against the simulated rejection times
  (Wald's identity for the oracles, the scaling identity for the corrected strategies).
* `timing.py` — per-round cost of the strategies on a single data stream (the run-time table).
* `results/` — the `.npz` rejection-time files and text summaries behind the tables and figures
  in the paper.

## Reproducing the paper

All commands are run from this directory; one core suffices. Approximate run times in brackets.

    python run_all.py bounded3        # [ 5 min]  bounded,  p = 3
    python run_all.py bounded9        # [13 min]  bounded,  p = 9
    python run_all.py gauss3          # [ 6 min]  Gaussian, p = 3
    python run_all.py gauss9          # [15 min]  Gaussian, p = 9

    python run_up.py bounded3         # [ 5 min]  mixture on the same streams
    python run_up.py bounded9         # [17 min]
    python run_up.py gauss3           # [17 min]
    python run_up.py gauss9           # [50 min]
    python run_up.py bounded9 2000 0 0.5      # Dirichlet(1/2) prior (reported in the text)
    python run_up.py gauss9 2000 0 1 1        # exponential prior with rate 1 (prior sensitivity)

    python make_tables.py bounded3 bounded9 gauss3 gauss9   # table bodies from results/
    python make_figures.py            # figures/rejection_*.pdf
    python predictions.py             # Wald and scaling-identity predictions
    python timing.py                  # [2 min] run-time table

Every run is seeded, so the stored results in `results/` are reproduced exactly; `run_up.py`
regenerates the identical data streams as `run_all.py` (same seeds, same order of the random
draws), so that all strategies are compared on the same data.

## Design

Both regimes are parametrised by the mean vector `mu` and the correlation matrix `R` of `X`.
The bounded generator is `X = mu + a (2 Phi(Z) - 1)` with `a = 0.8` and the copula correlation
`R_z = 2 sin(pi R / 6)`, so that `Corr(X) = R` exactly and the coordinates lie in `[-1, 1]`;
the Gaussian design draws `X ~ N(mu, R)` with unit variances. In dimension nine, `R` is a
two-factor design containing redundant, complementary and isolated coordinates at once; its
construction and all mean configurations are given in the paper's appendix. The horizon is 500
rounds (bounded) and 1000 rounds (Gaussian), with 2000 repetitions per configuration at
`alpha = 0.05`; the Gaussian strategies use the epoch correction with `t_1 = 16`, six epochs and
total budget `delta = 0.025`.

## Licence

MIT, see `LICENSE`.
