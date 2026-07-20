# Measurement-error model for METR's time horizon

METR tracks AI progress with a simple question: how long a task (in human
time) can an AI agent complete? Their answer, the 50% time horizon, depends
on how long each task takes a person. But those human times are noisy. Some
tasks were timed once. Some were never timed, only estimated. Two people on
the same task can differ by a factor of 2 or more.

This repository contains a Bayesian model that keeps that noise inside the
model instead of pretending the times are exact. Each task's true length is
a hidden quantity. The timed runs and the estimates inform it, with the
appropriate amount of trust in each. The model then fits the ability trend
on top, so that every horizon number carries the timing uncertainty with
it.

The full story, and where this model lands against METR's own noise
analysis, is in [`docs/writeup.md`](docs/writeup.md).

## The result

The current doubling time of the 50% horizon comes out to **2.4 months
[2.1, 2.7]** under the best model. Across every variant tried, it stays
inside 2.4 to 3.3 months. Two other findings:

- Tasks of the same human length differ enormously in how hard they are
  for AI models — a spread of roughly 8x. About two thirds of that spread
  is predictable from the task's family (its type of work), not noise.
- Correcting the timing noise moves the frontier horizon by only about
  10%, not the 25% to 40% that a simpler model implies. The reason is in
  the writeup, section 3.

![Horizon trend](outputs/figures/horizon_trend.png)

## What is in the model

All 228 tasks that the AI models attempted are in the model. Nothing is
thrown away:

- **525 timed runs** — real people who completed a task with the clock
  running. These carry the most weight.
- **67 estimate-only annotations** — tasks that no one completed with a
  timer. These enter with a wider noise term, because an expert guess is
  less reliable than a stopwatch.
- **4,523 success/failure counts** — every (model, task) attempt record,
  across 20 AI models with release dates.

The model has three connected parts:

1. **Timing part.** Each task has a hidden true length. Timed runs scatter
   around it. A heavy-tailed distribution absorbs the occasional run where
   someone took a long break mid-task.
2. **Difficulty part.** How often each AI model succeeds at each task tells
   the model how hard the task really is. Difficulty is the hidden length
   plus a per-task adjustment for tasks that are harder or easier than
   their length suggests.
3. **Trend part.** Each AI model's ability follows a curve over its release
   date. The slope of that curve gives the doubling time. Four curve shapes
   are fitted and combined by predictive weight.

All three parts are fitted at the same time, so the timing evidence and the
success evidence both get a say in every number.

## Setup and run

This repository expects two sibling checkouts in the same parent directory
(the raw data source and a reference implementation):

```
some-parent-dir/
  metr-measurement-error/       # this repo
  eval-analysis-public/         # git clone https://github.com/METR/eval-analysis-public
  metr-stats/                   # git clone https://github.com/JonasMoss/metr-stats
```

Then:

```
uv sync
uv run python data/load_runs.py     # build the filtered dataset
uv run python scripts/fit_model.py  # quick smoke-test fit (seconds)

# The recommended full fit:
uv run python scripts/fit_model.py --tune 2000 --draws 2000 --chains 4 --target-accept 0.95 \
    --shape kink --robust --heteroscedastic --eps-structure family --log-likelihood

uv run python scripts/analyze_fit.py --fit outputs/fit_kink_robust_het_fameps.nc
```

`scripts/make_figures.py` rebuilds the figures. It needs one `--robust` fit
per trend shape (`--shape linear`, `kink`, `superexp`, `logistic`), plus a
plain and a `--duration-dist weibull` linear fit for the comparison plots.

## Layout

```
data/load_runs.py        # filters METR's runs.jsonl to the human timing data
models/data_prep.py      # builds the arrays the model needs
models/time_horizon_model.py  # the model itself (full spec in its docstring)
scripts/fit_model.py     # fits any variant; flags control shape and options
scripts/analyze_fit.py   # horizons, doubling time, checks on a saved fit
scripts/make_figures.py  # regenerates the figures
docs/writeup.md          # the full writeup
```

The remaining scripts reproduce specific analyses in the writeup (each has
a docstring saying which).

## Credit

This model builds directly on two pieces of work. Jonas Moss's reanalysis
of the METR data supplied the exam-style (item response) framing and the
trend structure. Alexander Barry's note on modeling assumptions supplied
the problem statement and the noise calibration. The writeup states
precisely what came from where.
