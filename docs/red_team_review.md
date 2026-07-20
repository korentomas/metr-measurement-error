# Red-team review of the Bayesian measurement-error time-horizon model

This document is a structural critique of `models/time_horizon_model.py` and
the pipeline around it (`data/`, `models/data_prep.py`, `scripts/`). The
model is well built, and it is unusually honest about its own limits. The
README's "Open work" list and `docs/results.md` already flag several of the
points below. This document collects all of the points in one place, adds
the points that are *not* flagged, and ranks them by how much they threaten
the headline. Two small analytic probes back the structural claims. (The
probe script was a scratch file and is not in the repository. The two cited
numbers are reproducible from the formulas given below.)

The model was rebuilt independently from a synthetic 228-task, 20-model
design, and its graph was regenerated with `pm.model_to_graphviz`. The
structure matches the committed `outputs/figures/model_graph.png`.

[`measurement_error_improvements.md`](measurement_error_improvements.md)
answers findings #1, #2, #3, #6, and #7 with fitted experiments. Each of
those findings has a status line below.

---

## 0. The model, stated precisely

Three coupled likelihoods over a shared set of latent per-task difficulties.

**Latents.** For each task *i* (228 of them): latent log-length `log_L_i`,
residual difficulty `eps_i`, and discrimination `a_i`. For each model *m*
(20): ability `theta_m`. All group-level latents are non-centered.

**Measurement layer** (pins `log_L` to the timing data):

- `log_L_i = mu_L + sigma_L * z_i`,  `z_i ~ N(0,1)`
- baseline runs: `log(dur_ib) ~ StudentT(nu, log_L_i, sigma_base)` (Normal/Weibull optional)
- estimate annotations: `log(rep_i) ~ N(log_L_i, sigma_est)`, `sigma_est` prior median 1.25

**IRT layer** (the horizon signal):

- `a_i ~ LogNormal(0, sigma_a)`
- `eps_i ~ ZeroSumNormal(sigma_eps)`  (sum-to-zero over all 228 tasks)
- `difficulty_i = log_L_i + eps_i`
- `logit P(success_im) = a_i * (theta_m − difficulty_i)`
- `successes_im ~ Binomial(n_im, P)`

**Trend layer:**

- `theta_m = f(t_m) + u_m` (dated), `beta0 + u_m` (undated), with `u_m ~ N(0, sigma_u)`
- `f` is one of {linear, kink, superexp, logistic}, and `slope_now = df/dt |_{t_now}`
- the headline doubling time is `ln(2) / slope_now`, and the per-model horizon is `h50_m = exp(theta_m)` minutes

The whole model turns on one identity: **the IRT layer sees only the sum
`log_L_i + eps_i`.** Everything below follows from that identity, plus the
sparsity of the timing data.

---

## Tier 1 — threatens interpretation of the headline

### 1. The reported horizon is a *conditional* horizon, and METR's is *marginal*

`h50_m = exp(theta_m)` is, by construction, the length of a task **with
`eps_i = 0`** (median residual difficulty) that model *m* solves at p = 0.5.
METR's published 50% time horizon is different. It is the length at which a
model succeeds at 50%, **averaged over the actual mix of tasks** at that
length. That is a marginal quantity over the `eps` (and `a`) distribution.
With `sigma_eps ≈ 2.2` log-minutes (an 8x difficulty spread), these are
different objects. The README's "Open work" already noted the missing
marginal-conditional comparison.

The README does **not** say *which* comparisons survive the distinction. I
checked (probe 1):

- **The 50% level is protected.** The logistic function is odd-symmetric, and
  `eps` is symmetric with mean zero. Thus, the marginal success curve still
  crosses 0.5 exactly at `ell = theta_m`, and `h50_m = exp(theta_m)` is a
  defensible 50% horizon *level*.
- **The slope is protected.** A stationary `eps` distribution shifts the
  marginal and conditional curves by the same offset at every *t*. Thus,
  `d theta/dt`, and with it the doubling-time headline, is largely invariant
  to the distinction. This is the strongest defense of the 2.8-month number.
- **Everything else is distorted.** The marginal success curve is
  approximately **1.8x flatter** than the conditional curve at the 50% point
  (measured at `sigma_eps = 2.2, sigma_a = 0.34`). Thus: (a) any non-50%
  quantile (a "p80 horizon," or a "when does p(success) get to 0.9" claim)
  diverges from METR's number. And (b) any extrapolation to a *fixed
  threshold* ("when do we get to a 1-month horizon") inherits the full
  `sigma_eps` sensitivity. It is **not** protected the way the slope is. The
  README correctly calls extrapolation "most sensitive" to `sigma_eps`. This
  is the mechanism.

**Fix:** compute the marginal `h50` (average the fitted per-task success
curves over the empirical `eps`/`a` posterior at each length) and report it
next to `exp(theta)`. State explicitly that the doubling-time headline is
slope-based and thus robust to the distinction. State also that any
horizon-*level* comparison to METR, and any threshold extrapolation, are not
protected.

**Status:** answered in
[`measurement_error_improvements.md`](measurement_error_improvements.md),
section 4.

### 2. The measurement-error layer is nearly decoupled from the horizon that it corrects

The stated motivation (module docstring, README) is that a known-value
treatment of `human_minutes` "silently injects measurement error into the
difficulty term and, via the difficulty-vs-length regression, into the
horizon estimate." But the IRT layer only ever sees
`difficulty_i = log_L_i + eps_i`. And `eps_i` is a free per-task term with
`sigma_eps ≈ 2.2`, far larger than `sigma_base ≈ 0.41`. Thus, for the
IRT/horizon inference, `eps_i` simply reabsorbs any error or uncertainty in
`log_L_i`. Probe 2: with `log_L` pinned by even a single timed run,
**96.6%** of any shift in the identified difficulty sum lands on `eps`, not
on `log_L` (98.3% at k = 2, and 98.9% at k = 3).

Consequences:

- The elaborate measurement layer buys **very little** for the horizon
  estimate. This agrees with the repository's own findings: the doubling
  time barely moves (Normal against Student-t: 3.4 against 3.3 months, and
  the cut-feedback deltas are "approximately one day"). The layer's real
  value is in the *decomposition* of difficulty variance and in the
  estimate-only handling, not in a de-biased trend. The framing oversells
  the trend correction.
- **`sigma_eps` is not clean "residual difficulty."** `log_L` is only pinned
  to `± sigma_base/sqrt(k)`, and *k* = 1 for most tasks. Thus, the length
  channel is itself noisy, and that noise flows into `eps`. The headline "8x
  residual difficulty spread at fixed length" is thus an **upper bound**. It
  contains genuine difficulty heterogeneity, *plus* leftover
  length-measurement error, *plus* the effect of a missing difficulty
  covariate (refer to #7). To present `sigma_eps` as "the residual
  difficulty spread" attributes all of it to task heterogeneity.

**Fix:** report the variance decomposition explicitly, and caveat
`sigma_eps` as an upper bound. Consider a simulation that injects known
`log_L` error and shows how much `sigma_eps` inflates.

**Status:** answered in
[`measurement_error_improvements.md`](measurement_error_improvements.md),
section 3.

---

## Tier 2 — data and selection issues that can bias levels or slope

### 3. Latent task length is estimated from success-conditioned human times

`data/load_runs.py` filters the human rows to `score_binarized == 1`. Thus,
only **successful** human attempts anchor `log_L_i`. Failed or abandoned
human attempts are dropped, and those tend to be the long, timed-out, or
given-up attempts. `log_L_i` is thus inferred from a downward-biased
(faster) sample of human times. The bias is worst exactly where it matters
most: the hard, long tasks, where humans fail more often. This bias
propagates into `difficulty` and into the length axis that the whole trend
is read against. No document mentions it as a limitation.

**Fix:** quantify how often humans failed per task. If the count is
non-trivial, model the human attempt as its own (censored or lapse) process,
or at least report the direction and the rough size of the survivorship
bias.

**Status:** answered in
[`measurement_error_improvements.md`](measurement_error_improvements.md),
section 2 (failed-run censoring).

### 4. A latent inconsistency in the duration units, with no sanity bound

`data/load_runs.py` documents `completed_at` as "a Unix-ms timestamp."
`models/data_prep.py` computes `wall_minutes = (completed_at −
started_at)/1000/60`, and its own comment says that both are "run-relative
millisecond clocks (started_at is 0 for many rows)." These two descriptions
are mutually exclusive. If `started_at = 0` and `completed_at` were a true
Unix-ms epoch, `wall_minutes` would be approximately 2.9x10^7 minutes, not a
task duration. The pipeline only guards with
`base = base[base["wall_minutes"] > 0]`. That guard silently drops
non-positive durations (clock resets where `started > completed`), but it
puts **no upper bound** on the values. If the column semantics are mixed
across rows or sources, some rows would enter as astronomically large
outliers. That is *also* the exact symptom (3 to 4 log-unit outliers) that
motivated the Student-t layer. In other words, part of the "heavy tails"
that the robustness machinery absorbs may be a units artifact, not real
break behavior.

**Fix:** assert the intended semantics in the loader (for example, bound
`wall_minutes` to a plausible range, and count and inspect anything
dropped). Resolve the contradictory docstrings. Confirm that the six cited
outlier runs are genuine long human sessions and not clock-unit mixing.

### 5. The "current doubling time" rests on two manually back-filled dates

`slope_now` is evaluated at `t_now = max(t_model)`, the latest dated models.
The two most recent and most capable models (`flamingo_2`,
`claude_opus_4_6_inspect`) are **missing** from Moss's
`release_dates.json`. They are hardcoded to `2026-02-05` in
`RELEASE_DATE_OVERRIDES`. Thus, the single most influential input to the
headline slope, the frontier points at the right edge, is a manual override.
And both models are pinned to the *same* day. Any error there, or any
cluster artifact from the co-location of two frontier points, moves
`slope_now` directly. The kink shape (which carries 0.635 of the stacked
weight) reads its post-2024 slope directly off this region.

**Fix:** check the sensitivity of the headline to a perturbation of these
two dates by plus or minus a few weeks. Source the dates from a citable METR
artifact, not a literal in code.

---

## Tier 3 — validation does not cover the reported configuration

### 6. SBC validates a different model than the one in the headline

The headline uses the **Student-t** layer, **stacked** over four shapes, at
full scale (228 tasks, 20 models), with `sigma_est` median **1.25**. The SBC
in `scripts/sbc.py` and `docs/results.md` uses the **Normal** layer, the
**linear** shape only, reduced scale (50 tasks, 8 models), `sigma_est`
median **0.8**, and 50 replications. Thus, the calibration evidence does not
touch:

- the Student-t `nu` parameter and the heavy-tail geometry in actual use;
- the kink, superexp, and logistic shapes (the kink drives the headline) —
  their extra parameters (`delta`, `t_k`, `beta2`, `h`, `t0`, `s`) are never
  rank-checked, and the logistic is *known* to be poorly identified
  (`docs/results.md`: the ceiling parameter is pinned at its prior edge);
- the `sigma_est = 1.25` prior in actual use;
- the full-scale sparsity and geometry;
- the per-task latents `log_L`, `eps` (ZeroSumNormal), and `a` — the SBC
  only ranks the nine scalar hyperparameters.

Two SBC items are already in "Open work," but the headline-configuration gap
is broader than what is listed. A minor point: at 50 replications, the
cov50 of `mu_L` (0.38) and of `sigma_eps` (0.60) are the kind of deviation
that warrants more replications before "well calibrated" is asserted. Also,
the rank statistic `(thinned < true).mean()` uses no randomized tie-break
(fine for continuous draws, but worth a statement).

**Fix:** at minimum, run reduced-scale SBC on the Student-t + kink
configuration under the 1.25 prior. Rank `nu`, `delta`, `t_k`, and some of
the `eps_i` and `log_L_i` latents.

**Status:** answered in
[`measurement_error_improvements.md`](measurement_error_improvements.md),
section 6.

### 7. `eps` is an unstructured catch-all with no difficulty covariate

`eps_i` has no predictors. It is pure per-task noise, identified almost
entirely by the IRT layer (and, for sparse tasks, by its prior plus the
single sum-to-zero constraint). Anything systematic that makes a task hard
beyond its length — the task family, the number of required tool calls, the
context length, the reasoning type — lands in `eps` and is reported as
irreducible "residual difficulty." A model that regressed `eps` on even a
few task-level covariates would (a) shrink `sigma_eps` toward its true
irreducible floor and (b) tell you *why* length is a noisy proxy. As it
stands, the 8x headline conflates "difficulty that we cannot predict" with
"difficulty that we did not try to predict."

**Fix:** add the available task metadata (`task_family`, `task_source`) as
`eps` predictors. Or at least report `sigma_eps` within family, to show how
much is between-family and how much is genuinely irreducible.

**Status:** answered in
[`measurement_error_improvements.md`](measurement_error_improvements.md),
section 5.

### 8. The stacking weights that set the headline are within their own noise

The headline 2.8 months is a stacking mixture, 0.635 kink and 0.365 linear.
But the per-shape elpd values are −2695.6 against −2696.2. That is a 0.6-nat
gap against a **±81** standard error. And `docs/results.md` says directly
that the success/failure data alone cannot strongly separate the shapes. Two
concerns compound:

- **PSIS-LOO with leave-one-(model, task)-cell-out is weak here.** Tasks and
  models share latents (`log_L`, `eps`, and the `theta` trend). Thus, the
  removal of one cell barely perturbs the fit. LOO has little power to
  discriminate a *smooth trend shape*. That is precisely why the elpd gaps
  are tiny and why approximately 2% of the pareto-k values are more than
  0.7. The weights are read off a near-flat, noisy objective.
- As a consequence, the split between the 2.7 (kink) and 3.3 (linear)
  sub-headlines is effectively **arbitrary within noise**. Yet that split is
  what moves the stacked number below the linear-only 3.3. The honest
  headline is arguably "2.7 to 3.3 months, shape-underdetermined." The point
  estimate 2.8 conveys more precision than the model comparison supports.

**Fix:** report the stacked interval with an explicit "the weights are
within the elpd SE" caveat (partially there already). Consider
leave-one-*model*-out or leave-one-*task*-out (grouped LOO) as a more honest
test of the trend shape.

---

## Tier 4 — smaller structural and robustness notes

### 9. No lapse or guess floor in the 2PL
`P = sigmoid(a*(theta − difficulty))` goes to 0 and 1 at the extremes. Real
agents fail trivial tasks (harness and format errors), and they sometimes
pass hard tasks by luck. With no lower or upper asymptote, such cells are
explained through distortion of `a_i` and `eps_i`. The PPC flags exactly
this at the trivial `<0.1 min` bin (observed 1.000, posterior predictive
[0.996, 1.000]). A 4-parameter IRT, or a small fixed lapse rate, would
absorb it.

### 10. `sigma_est` is prior-only and unfalsifiable in-sample
No task carries both a baseline time and an estimate annotation. Thus, only
its prior identifies `sigma_est` (calibrated to Barry's external
60%-within-3x figure). The repository handles this honestly (a sensitivity
run at 0.8 against 1.25). But it means that the lengths of the 67
estimate-only tasks rest on an external number that the data cannot check.
And those tasks skew long (up to 30 h). That is, they sit at the
high-leverage end of the length axis that the trend is fit against.

### 11. The Weibull censoring branch is dead code
`pm.Censored` never activates (`time_limit == 0` for all human rows on the
snapshot). This is documented and harmless. But it is untested machinery
that will fire automatically on a future data pull. It deserves a unit test
with a synthetic censored row, so that its first execution is not in
production.

### 12. Reproducibility fragility
`get_sota_models` and `RELEASE_DATE_OVERRIDES` hardcode the
model-alias-date-p50 joins, and they `raise` on any mismatch. A routine
upstream `runs.jsonl` refresh (a new model, or a renamed alias) breaks the
pipeline instead of a gradual degradation. The `sigma_est` calibration
constant (1.25) and the SOTA set (14 models) are snapshot-specific literals
in code.

### 13. The prior pushforward on the steep slope is not shown
`beta1 ~ N(0, 1)`, while the fitted kink post-2024 slope is 3.14, which is
approximately 3 prior SDs out. This is plausibly fine (the data dominate).
But a prior-predictive check on `slope_now` and the doubling time would
confirm that the `N(0,1)` slope prior and the `N(0,1.5)` intercept prior do
not mildly fight the steep-acceleration region. The "Open work" list covers
the shape-specific priors. The base `beta1` prior deserves the same check.

---

## What holds up

To keep the critique calibrated — these parts are genuinely well handled:

- The **identification fix** (ZeroSumNormal on `eps` instead of Moss's
  hard-anchored thetas) is correct, and it preserves the log-minute
  interpretation of `theta`. The footnote derivation of the ridge is right.
- The **non-centered** parameterization of `log_L`, `eps`, `u`, and `a` is
  the correct response to the funnel geometry described.
- The **Student-t and Weibull** investigation is thorough. The negative
  Weibull result is well argued (Gumbel-minimum skew against the observed
  right skew).
- The **cut-model** check of Barry's circularity warning is exactly the
  right modularized-inference tool. The diagnostic quantifies the effect
  (corr −0.53, and a headline delta of approximately one day) instead of a
  hand-wave.
- The doubling-time headline is **slope-based**, which (per #1) is the one
  summary most robust to the conditional-marginal issue.

---

## Priority order for follow-up

1. The marginal `h50`, plus an explicit statement of what the
   conditional-marginal split does and does not protect (#1).
2. A variance decomposition that makes clear how much of `sigma_eps` is
   length noise and how much is true heterogeneity (#2), ideally with `eps`
   covariates (#7).
3. The survivorship-bias check on success-conditioned human times (#3), and
   the duration-units audit (#4).
4. SBC on the actual headline configuration (#6).
5. The honest "shape-underdetermined" frame for the stacked headline (#8),
   and the sensitivity of the two hardcoded frontier dates (#5).
