# Measurement-error model for METR's time horizon

This repository contains a Bayesian extension of Jonas Moss's [IRT reanalysis](https://www.lesswrong.com/posts/sBEzomgnYJmYHki9T) of METR's [time-horizon data](https://github.com/METR/eval-analysis-public). The extension adds a measurement-error layer for the human baseline task times. Moss's model treats each task's human time as a fixed, exactly known scalar. This model treats the human time as a latent variable, and the per-run timing data informs it. A residual difficulty term ($\varepsilon_i$) absorbs the task-difficulty variance that length does not explain.

Start with [`docs/writeup.md`](docs/writeup.md). That document tells what this model takes from Moss and from Barry. It also tells where this model and Barry's SIMEX analysis do not agree. For each number and plot behind the headline results, refer to [`docs/results.md`](docs/results.md).

## What is in the model

The model keeps all 228 tasks that the AI models attempted. It does not discard the tasks that only have an estimated task length. The observation set has two parts:

- **525 timed baseline runs.** Each observation is the wall-clock time of one successful human attempt at a task.
- **67 estimate annotations.** A task with no timed baseline run contributes its `human_minutes` annotation as one observation. These observations get a wider noise term (`sigma_est`) because an expert estimate is less reliable than a timed run.

The [Data](#data) section gives the filter steps and the full counts.

## Setup

This repository needs two sibling checkouts in the same parent directory. One is the raw data source. The other is a read-only reference implementation:

```
some-parent-dir/
  metr-measurement-error/       # this repo
  eval-analysis-public/         # git clone https://github.com/METR/eval-analysis-public
  metr-stats/                   # git clone https://github.com/JonasMoss/metr-stats
```

Then, from inside this repository:

```
uv sync
uv run python data/load_runs.py
```

## Repository layout

```
data/
  load_runs.py          # filters runs.jsonl down to human timing observations
  processed/             # output of load_runs.py (gitignored)
models/
  data_prep.py           # assembles filtered human data + full runs.jsonl + release
                         # dates into the arrays the PyMC model needs
  time_horizon_model.py  # pm.Model builder: measurement layer + IRT layer + trend
                         # (4 trend shapes, optional Student-t measurement layer)
scripts/
  fit_model.py           # runnable fit script (nutpie, falls back to PyMC NUTS);
                         # --shape {linear,kink,superexp,logistic}, --robust,
                         # --log-likelihood (pointwise, for stacking),
                         # --sota-only (METR's frontier model set, see docs/results.md)
  analyze_fit.py         # PPCs, per-model 50% horizons, doubling time
  compare_robust.py      # Normal vs Student-t before/after comparison
  compare_duration_dists.py  # PSIS-LOO of lognormal vs Student-t vs Weibull on the
                         # 525 baseline runs (Jacobian-corrected to a common scale)
  sbc.py                 # simulation-based calibration (reduced scale)
  stack_shapes.py        # PSIS-LOO + Bayesian stacking across the 4 shapes
  make_figures.py        # generates outputs/figures/*.png from the fitted .nc files
  marginal_horizon.py    # marginal (METR-style) vs conditional exp(theta) horizon
  measurement_value.py   # what the measurement layer buys (uncertainty vs plug-in)
  compare_measurement.py # baseline vs heteroscedastic vs failed-run censoring
  eps_decomposition.py   # between- vs within-family split of residual difficulty
  make_measurement_figures.py  # figures for the measurement-error experiments
  simex.py               # Barry's SIMEX ladder on this model (add sqrt(lambda)*sigma
                         # noise, refit, extrapolate to lambda=-1); writeup section 3
  estimate_feedback_diagnostic.py  # Barry's circularity warning: how much the IRT
                         # layer moves log_L on estimate-only tasks
  # one-off analysis scratch (no argparse; fit paths hardcoded):
  fork_discriminator.py  # the eps-vs-length fork: regress eps on length over
                         # well-timed tasks, test if poorly-timed long tasks obey it
  difficulty_pinning.py      # posterior sd of difficulty, and corr(log_L, eps), by group
  weakspot_family_kin.py     # IRT signal available to long estimate-only tasks
  check_longtask_shrinkage.py  # posterior log_L vs raw observed length, per task
  make_eps_family_figure.py    # eps-by-family figure (needs an --eps-structure family fit)
outputs/                 # saved InferenceData (.nc), gitignored
outputs/figures/         # generated plots (committed, see docs/results.md)
docs/
  writeup.md             # the writeup: what is reused, and where this lands vs Barry
  results.md             # full results: every number and plot
  red_team_review.md     # structural critique / shortcomings
  measurement_error_improvements.md  # experiments + recommended best model
```

## Data

The data source is METR's public `eval-analysis-public` repository, file `reports/time-horizon-1-1/data/raw/runs.jsonl`. The file has 24,008 rows. Each row is one model-task run, and each row carries the task's human timing metadata.

The human timing observations are the rows with `model == "human"`. These rows come from the persons who attempted each task or, for some tasks, estimated its duration. Each other row is an AI model's own run. Those rows carry the task's human timing metadata only for reference.

`data/load_runs.py` filters to:

```
model == "human"  AND  score_binarized == 1  AND  completed_at > 0
```

On the current snapshot, this filter gives 554 rows and 164 different tasks (525 rows with `human_source == "baseline"` and 29 with `"estimate"`). `models/data_prep.py` then builds the observation set from this filtered data. Two properties of the data give the observation model its shape. The two properties are verified against the snapshot:

1. `human_minutes` is a task-level annotation. It is identical on each row of a task (0 of the 136 multi-row tasks vary). Where successful baseline wall-times exist, it equals their geometric mean (median ratio 0.998). The per-run observation is the wall-clock time, `completed_at - started_at` (run-relative millisecond clocks). The within-task standard deviation of log wall-time is approximately 0.4 to 0.6 (that is, 1.5x to 2x). An early version fed `human_minutes` per row as if independent. That made `sigma_base` collapse to 0 and froze the sampler.
2. The task universe is all 228 tasks that the AI models attempted. Only 164 of these tasks have successful human runs. The other 64 tasks are almost all estimate-source, and most of them are long (up to 30 h). To drop them biases the trend. A task with no timed baseline run contributes its annotation as a single estimate observation.

The final observation set is **525 per-run baseline wall-times plus 67 task-level estimate annotations, over 228 tasks**. The estimate-source tasks with human runs (the RE-Bench 8 h time-boxed runs) do have wall-clock times. But those times are budget-limited work times. For those tasks, the model uses only the annotation.

Run it:

```
uv run python data/load_runs.py
```

```
Loaded 24008 raw rows from .../runs.jsonl
Filtered (score_binarized==1 & completed_at>0): 554 rows / 164 tasks (136 tasks with >=2 timed attempts)
  baseline (real-timed) rows: 525, estimate-only rows: 29
Wrote filtered data to data/processed/runs_filtered.parquet
```

A note on censored observations. The v2 spec tells to right-censor a duration observation at `time_limit` when RE-Bench-style runs stack at the limit. The `pm.Censored` branch in `time_horizon_model.py` implements this. In the current `runs.jsonl` snapshot, `time_limit` is always 0 for the `model == "human"` rows (METR populates that field for agent compute budgets). Thus the model censors no row. The branch becomes active automatically if a future data pull includes time-limited human runs.

`models/data_prep.py` also pulls:
- IRT counts. For each (model, task) pair among the 228 tasks (all non-human, non-cloned models), it aggregates attempt and success counts from the full `runs.jsonl`. On this snapshot, that gives 4,523 (model, task) rows across 20 models.
- Release dates. These come from `../metr-stats/data/release_dates.json`, plus overrides in `models/data_prep.py` for the two models that are missing there (`flamingo_2` == GPT-5.3-Codex and `claude_opus_4_6_inspect`, both 2026-02-05). The override dates come from METR's TH1.1 `logistic_fits/headline.csv` through the `alias` column of `runs.jsonl`. All 20 models are dated and are part of the trend. A model without a date would contribute no trend term, and the random effect `u_m` would hold its ability.

## Model

The docstring at the top of `models/time_horizon_model.py` gives the full spec. This model extends Moss's 2PL model with three changes. It adds a residual difficulty term. It uses a different identification fix, suited to our difficulty scale. And it adds a per-model random effect on ability.

- Measurement layer: $\log(L_i) \sim \mathcal{N}(\mu_L, \sigma_L)$. Baseline observations are $\log(\text{dur}) \sim \mathcal{N}(\log(L_i), \sigma_{\text{base}})$, censored where applicable. Estimate-only observations are $\log(\text{rep}) \sim \mathcal{N}(\log(L_i), \sigma_{\text{est}})$ with a wider prior on $\sigma_{\text{est}}$ (median 1.25, calibrated to Barry's 60%-within-3x finding). The prior is wider because an expert estimate is less reliable than a real timed run.
- IRT layer: $\text{logit}\, P(\text{success}_{im}) = a_i \left(\theta_m - (\log(L_i) + \varepsilon_i)\right)$, with $a_i \sim \text{LogNormal}(0, \sigma_a)$ and the residual difficulty term $\varepsilon_i \sim \text{ZeroSumNormal}(\sigma_\varepsilon)$. The sum-to-zero constraint is the identification fix for a shift non-identifiability between $\varepsilon_i$ and $\theta_m$. A free $\sigma_\varepsilon$ cannot resolve that shift on its own.[^1] Moss prevents it differently: he hard-anchors two $\theta$ values ($\theta_{\text{low}}=-1$, $\theta_{\text{high}}=+1$). That fix does not work here. Our difficulty scale is pinned in log-minute units by the timing data, and anchors would break the log-minute interpretation of $\theta_m$ ($h_{50,m} = \exp(\theta_m)$).
- Ability trend: $\theta_m = f(t_m) + u_m$ for dated models, and $\theta_m = \beta_0 + u_m$ for undated models. The per-model random effect $u_m$ lets the doubling-time CI show model-to-model variation. $f(t_m)$ is one of four fitted shapes. Linear ($\beta_0 + \beta_1 t_m$) is the simplest case. Kink, superexponential, and logistic are the other three shapes, combined with Bayesian stacking.
- All group-level latents ($\log L$, $\varepsilon$, $u$, named `log_L`, `eps`, `u` in code) use a non-centered parameterization. Most tasks have only 1 or 2 observations. With so few observations, a centered $\log L \sim \mathcal{N}(\mu_L, \sigma_L)$ makes a funnel between $\sigma_L$ and $\log L$.

The model graph below comes from `pm.model_to_graphviz()` on the built model (Student-t duration likelihood, linear shape). Rectangles are deterministic nodes: the non-centered reparameterizations, `theta`, `a`, `eps`, and `log_L`. Circles are free random variables, and shaded circles are observed variables. The three plates on the right give the observation counts: 4,523 (model, task) success/attempt pairs, 67 estimate-only annotations, and 525 timed baseline runs. The `task (228)` and `model (20)` plates show which latents repeat per task and per model.

![Model graph](outputs/figures/model_graph.png)

## Running

```
uv run python scripts/fit_model.py                      # tiny smoke test (200 tune/200 draws/2 chains, nutpie)
uv run python scripts/fit_model.py --tune 2000 --draws 2000 --chains 4 --target-accept 0.95 \
    --shape kink --robust --log-likelihood              # production-style fit of one shape
uv run python scripts/fit_model.py --sampler pymc        # force PyMC's own NUTS

uv run python scripts/fit_model.py --tune 2000 --draws 2000 --chains 4 --target-accept 0.95 \
    --duration-dist weibull --log-likelihood         # Weibull duration layer

uv run python scripts/fit_model.py --tune 2000 --draws 2000 --chains 4 --target-accept 0.95 \
    --shape kink --robust --log-likelihood --sota-only   # METR's 14 SOTA models only

# Measurement-error improvements (see docs/measurement_error_improvements.md):
uv run python scripts/fit_model.py --tune 2000 --draws 2000 --chains 4 --target-accept 0.95 \
    --shape kink --robust --heteroscedastic --eps-structure family --log-likelihood  # recommended best model
uv run python scripts/eps_decomposition.py --family outputs/fit_linear_robust_fameps.nc \
    --flat outputs/fit_linear_robust.nc   # between/within-family residual-difficulty split
uv run python scripts/fit_model.py --tune 2000 --draws 2000 --chains 4 --target-accept 0.95 \
    --shape linear --robust --include-human-failures --log-likelihood  # survivorship sensitivity
uv run python scripts/marginal_horizon.py --fit outputs/fit_linear_robust.nc
uv run python scripts/measurement_value.py --fit outputs/fit_linear_robust.nc
uv run python scripts/compare_measurement.py --fits outputs/fit_linear_robust.nc \
    outputs/fit_linear_robust_het.nc outputs/fit_linear_robust_hf.nc
uv run python scripts/sbc.py --shape kink --robust --sigma-est-median 1.25 --n-reps 40  # headline-config SBC

# Robustness-check scripts:
uv run python scripts/compare_robust.py --normal outputs/fit_linear.nc --robust outputs/fit_linear_robust.nc
uv run python scripts/compare_duration_dists.py \
    --fits outputs/fit_linear.nc outputs/fit_linear_robust.nc outputs/fit_linear_weibull.nc \
    --duration-dists lognormal studentt weibull --sigma-est-medians 0.8 0.8 0.8
uv run python scripts/sbc.py --n-reps 50 --tune 800 --draws 500
uv run python scripts/stack_shapes.py --fits outputs/fit_linear_robust.nc outputs/fit_kink_robust.nc \
    outputs/fit_superexp_robust.nc outputs/fit_logistic_robust.nc

uv run python scripts/make_figures.py   # regenerates outputs/figures/*.png from the .nc files below
```

## Headline results

For the full detail, each plot, and the comparison against METR's and Moss's numbers, refer to [`docs/results.md`](docs/results.md).

- The current doubling time is **2.4 months [2.1, 2.7]**, from the recommended model (kink trend, Student-t duration layer, heteroscedastic `sigma_base`, family-structured `eps`). Refer to [`docs/measurement_error_improvements.md`](docs/measurement_error_improvements.md). This number is a single-shape fit, not a stack. The flat-`eps` stack across all four shapes gives 2.8 [2.3, 3.8], and the flat kink alone gives 2.7 [2.3, 3.1]. To structure `eps` is what moves the number and tightens it.
- No refinement of the measurement *layer* moves the trend. Heteroscedasticity, Student-t, and failed-run censoring all keep the linear fit at 3.3 months. Only a new structure for the *difficulty* term moves it. This asymmetry is the model's own logic, and it is the subject of [`docs/writeup.md`](docs/writeup.md) section 3.
- The residual task-difficulty spread at fixed length (`sigma_eps`) is approximately 8x. Approximately **two thirds of it is between-family structure**. That leaves a within-family residual near 5x, which agrees with Moss's approximately 4.7x.
- Barry's SIMEX ladder, run again on this model, decreases the frontier 50% horizon by approximately **11%** (flat `eps`) or **8.7%** (family `eps`). Barry found 25% to 40% on METR's difficulty-equals-length model.
- SBC (reduced scale): the Normal linear model passes with 50 of 50 reps, well calibrated. The headline configuration (kink, Student-t, `sigma_est` median 1.25) and the heteroscedastic variant each pass with 40 of 40 reps.
- The Weibull duration likelihood (Moss's suggestion) is rejected. Its fit is worse than the Student-t and log-normal fits.
- The `sigma_est` prior now matches Barry's 60%-within-3x finding. The headline is robust to this shift.

## Open work

- SBC at full data scale, and on the family-`eps` component of the recommended model (its `sigma_eps_fam` and `sigma_eps_within` still need to be added to the ranked list). The headline configuration (kink + Student-t) and the heteroscedastic variant are already SBC-backed, both 40 of 40, with `gamma_sig` calibrated. Refer to [`docs/measurement_error_improvements.md`](docs/measurement_error_improvements.md) section 6.
- A task-family (or task-type) *covariate* on `eps`. Approximately 67% of the residual difficulty is between-family. A covariate would turn that structure into an interpretable predictor instead of a variance split.
- Marginal-horizon *bands* on the trend plot, not only the flattening factor.
- A prior-sensitivity pass on the shape-specific priors (t_k, h, s). The `sigma_est` pass is done. The headline is insensitive to a prior-median shift from 0.8 to 1.25.

[^1]: Only $\theta_m - (\log(L_i) + \varepsilon_i)$ enters the logit. Thus, if you add a constant to each $\varepsilon_i$ and subtract it from each $\theta_m$, the likelihood does not change. The mean-zero prior on $\varepsilon$ only weakly penalizes this shift, which makes a near-flat ridge in the posterior. The sum-to-zero constraint removes the degree of freedom directly.
