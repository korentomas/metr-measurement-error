# Measurement-error model for METR's time horizon

This repository contains a Bayesian reanalysis of METR's [time-horizon data](https://github.com/METR/eval-analysis-public) that models the noise in human task times. It extends Jonas Moss's [IRT reanalysis](https://www.lesswrong.com/posts/sBEzomgnYJmYHki9T) and follows the direction in Alexander Barry's [note on modeling assumptions](https://metr.org/notes/2026-03-20-impact-of-modelling-assumptions-on-time-horizon-results/). The full writeup is in [`docs/writeup.md`](docs/writeup.md).

## Overview

The model builds on METR's TH work by:

1. Treating each task's human time as a latent variable instead of a constant, based on the timed baseline runs and expert estimates
2. Fitting a success/failure (IRT) layer where a task's difficulty is its latent length + a residual, so tasks harder (or easier) than their length suggest their own term
3. Fitting an ability trend over model release dates jointly with the two layers above
4. Reading the Time Horizon as exp(ability) and the doubling time from the trend slope

Annotations that weren't timed get a wider noise term (`sigma_est`) instead of being dropped.

**Key finding**: the current doubling time is 2.4 months (95% credible interval: 2.1–2.7) under the recommended model, and stays within 2.4–3.3 months across every variant tried. Correcting the timing noise moves the frontier horizon by only ~10%, not the 25–40% a difficulty-equals-length model implies (writeup, section 2).

![Horizon trend](outputs/figures/horizon_trend.png)

## Repository Structure

```
.
├── data/
│   └── load_runs.py           # Filters METR's runs.jsonl to human timing observations
├── models/
│   ├── data_prep.py           # Builds the model's input arrays
│   └── time_horizon_model.py  # The PyMC model (full spec in its docstring)
├── scripts/
│   ├── fit_model.py           # Fits any variant; flags select shape and options
│   ├── analyze_fit.py         # Horizons, doubling time, checks on a saved fit
│   ├── make_figures.py        # Regenerates outputs/figures/
│   ├── simex.py               # Barry's SIMEX ladder run on this model (writeup section 2)
│   ├── fork_discriminator.py  # Is eps absorbing length bias? (writeup section 2)
│   ├── eps_decomposition.py   # The family split that moves the headline (writeup section 3)
│   └── ...                    
├── docs/
│   └── writeup.md
└── outputs/figures/
```

## Installation

The repository expects two sibling checkouts:

```bash
some-parent-dir/
  metr-measurement-error/      # this repo
  eval-analysis-public/        # git clone https://github.com/METR/eval-analysis-public
  metr-stats/                  # git clone https://github.com/JonasMoss/metr-stats
```

```bash
uv sync
uv run python data/load_runs.py
```

## Running the Model

```bash
# Smoke test (200 tune / 200 draws, finishes in seconds):
uv run python scripts/fit_model.py

# Recommended model (kink trend, Student-t noise, length-dependent noise scale,
# family-structured residual difficulty):
uv run python scripts/fit_model.py --tune 2000 --draws 2000 --chains 4 --target-accept 0.95 \
    --shape kink --robust --heteroscedastic --eps-structure family --log-likelihood

uv run python scripts/analyze_fit.py --fit outputs/fit_kink_robust_het_fameps.nc
```

`scripts/make_figures.py` rebuilds the figures. It needs one `--robust` fit per trend shape (`linear`, `kink`, `superexp`, `logistic`), plus a plain and a `--duration-dist weibull` linear fit for the comparison plots.

## Data

Source: `reports/time-horizon-1-1/data/raw/runs.jsonl` in `eval-analysis-public` (24,008 rows). The observation set:

| Observations | Count | Role |
|---|---|---|
| Timed baseline runs (`model == "human"`, successful) | 525 | anchor each task's latent length |
| Estimate-only annotations | 67 | same, with wider noise (`sigma_est`) |
| (model, task) success/attempt counts | 4,523 | identify difficulty and ability |
| Tasks | 228 | all tasks any model attempted |
| AI models (all dated) | 20 | trend over release dates |

## Citation

If you use this code, please cite this repository

## License

See LICENSE file for details.
