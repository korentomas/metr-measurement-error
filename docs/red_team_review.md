# Red-team review of the Bayesian measurement-error time-horizon model

A structural critique of `models/time_horizon_model.py` and the surrounding
pipeline (`data/`, `models/data_prep.py`, `scripts/`). The model is
well-built and unusually honest about its own limits — several of the points
below are already flagged in the README's "Open work" or in `docs/results.md`.
This document collects all of them in one place, adds the ones that are *not*
flagged, ranks them by how much they threaten the headline, and backs the
structural claims with small analytic/numeric checks (`scratch_probe.py`
reproduces the two numbers cited).

The model was rebuilt independently from a synthetic 228-task / 20-model
design and its graph regenerated with `pm.model_to_graphviz`; the structure
matches the committed `outputs/figures/model_graph.png`.

---

## 0. The model, stated precisely

Three coupled likelihoods over a shared set of latent per-task difficulties.

**Latents.** For each task *i* (228 of them): latent log-length
`log_L_i`, residual difficulty `eps_i`, discrimination `a_i`. For each model
*m* (20): ability `theta_m`. All group-level latents are non-centered.

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

- `theta_m = f(t_m) + u_m` (dated), `beta0 + u_m` (undated); `u_m ~ N(0, sigma_u)`
- `f` is one of {linear, kink, superexp, logistic}; `slope_now = df/dt |_{t_now}`
- headline doubling time `= ln(2) / slope_now`; per-model horizon `h50_m = exp(theta_m)` minutes

The whole thing turns on one identity: **the IRT layer sees only the sum
`log_L_i + eps_i`.** Everything below follows from that plus the sparsity of
the timing data.

---

## Tier 1 — threatens interpretation of the headline

### 1. The reported horizon is a *conditional* horizon; METR's is *marginal*

`h50_m = exp(theta_m)` is, by construction, the length of a task **with
`eps_i = 0`** (median residual difficulty) that model *m* solves at p = 0.5.
METR's published 50% time horizon is the length at which a model succeeds at
50% **averaged over the actual mix of tasks** at that length — a marginal
quantity over the `eps` (and `a`) distribution. With `sigma_eps ≈ 2.2`
log-minutes (an 8× difficulty spread), these are different objects, and the
README's "Open work" already lists a marginal-vs-conditional comparison as
undone.

What the README does **not** say is *which* comparisons survive the
distinction. I checked (`scratch_probe.py`, probe 1):

- **The 50% level is protected.** Because the logistic is odd-symmetric and
  `eps` is symmetric mean-zero, the marginal success curve still crosses 0.5
  exactly at `ell = theta_m`. So `h50_m = exp(theta_m)` is a defensible 50%
  horizon *level*.
- **The slope is protected.** A stationary `eps` distribution shifts the
  marginal and conditional curves by the same offset at every *t*, so
  `d theta/dt` — hence the doubling-time headline — is largely invariant to
  the distinction. This is the strongest defence of the 2.8-month number.
- **Everything else is distorted.** The marginal success curve is **~1.8×
  flatter** than the conditional one at the 50% point (measured at
  `sigma_eps = 2.2, sigma_a = 0.34`). So: (a) any non-50% quantile (a "p80
  horizon", a "when does p(success) hit 0.9" claim) diverges from METR's;
  (b) any extrapolation to a *fixed threshold* ("when do we reach a 1-month
  horizon") inherits full `sigma_eps` sensitivity and is **not** protected
  the way the slope is. The README correctly calls extrapolation "most
  sensitive" to `sigma_eps` — this is the mechanism.

**Fix:** compute the marginal `h50` (average the fitted per-task success
curves over the empirical `eps`/`a` posterior at each length) and report it
alongside `exp(theta)`; state explicitly that the doubling-time headline is
slope-based and therefore robust to the distinction, but that any
horizon-*level* comparison to METR and any threshold extrapolation are not.

### 2. The measurement-error layer is nearly decoupled from the horizon it is meant to correct

The stated motivation (module docstring, README) is that treating
`human_minutes` as known "silently injects measurement error into the
difficulty term and, via the difficulty-vs-length regression, into the
horizon estimate." But the IRT layer only ever sees `difficulty_i = log_L_i
+ eps_i`, and `eps_i` is a free per-task term with `sigma_eps ≈ 2.2` — far
larger than `sigma_base ≈ 0.41`. So any error or uncertainty in `log_L_i` is
simply reabsorbed by `eps_i` as far as the IRT/horizon inference is
concerned. Probe 2 (`scratch_probe.py`): with `log_L` pinned by even a single
timed run, **96.6%** of any shift in the identified difficulty sum lands on
`eps`, not `log_L` (98.3% at k=2, 98.9% at k=3).

Consequences:

- The elaborate measurement layer buys **very little** for the horizon
  estimate — consistent with the repo's own finding that the doubling time
  barely moves (Normal vs Student-t: 3.4 vs 3.3 months; cut-feedback: deltas
  of "about a day"). The layer's real value is in *decomposing* difficulty
  variance and in the estimate-only handling, not in de-biasing the trend.
  The framing oversells the trend correction.
- **`sigma_eps` is not clean "residual difficulty."** Because `log_L` is only
  pinned to `± sigma_base/sqrt(k)` and *k = 1* for most tasks, the length
  channel is itself noisy, and that noise flows into `eps`. The headline "8×
  residual difficulty spread at fixed length" is therefore an **upper
  bound**: it contains genuine difficulty heterogeneity *plus* leftover
  length-measurement error *plus* the model's failure to include any
  difficulty covariate (see #7). Presenting `sigma_eps` as "the residual
  difficulty spread" attributes all of it to task heterogeneity.

**Fix:** report the variance decomposition explicitly and caveat `sigma_eps`
as an upper bound; consider a simulation that injects known `log_L` error and
shows how much `sigma_eps` inflates.

---

## Tier 2 — data and selection issues that can bias levels/slope

### 3. Latent task length is estimated from success-conditioned human times

`data/load_runs.py` filters human rows to `score_binarized == 1` — only
**successful** human attempts anchor `log_L_i`. Failed or abandoned human
attempts (which tend to be the long, timed-out, or given-up ones) are
dropped. `log_L_i` is therefore inferred from a downward-biased (faster)
sample of human times, and the bias is worst exactly where it matters most —
hard/long tasks, where humans fail more often. This propagates into
`difficulty` and into the length axis the whole trend is read against. It is
not mentioned anywhere in the docs as a limitation.

**Fix:** quantify how often humans failed per task; if non-trivial, either
model the human attempt as its own (censored/lapse) process or at least
report the direction and rough size of the survivorship bias.

### 4. A latent inconsistency in the duration units, with no sanity bound

`data/load_runs.py` documents `completed_at` as "a Unix-ms timestamp,"
while `models/data_prep.py` computes `wall_minutes = (completed_at −
started_at)/1000/60` and its own comment says both are "run-relative
millisecond clocks (started_at is 0 for many rows)." These two descriptions
are mutually exclusive: if `started_at = 0` and `completed_at` were a true
Unix-ms epoch, `wall_minutes` would be ~2.9×10⁷ minutes, not a task
duration. The pipeline only guards with `base = base[base["wall_minutes"] >
0]` — it silently drops non-positive durations (clock resets where
`started > completed`) but imposes **no upper bound**. If the column
semantics are actually mixed across rows/sources, a handful of rows would
enter as astronomically large outliers — which is *also* the exact symptom
(3–4 log-unit outliers) that motivated the Student-t layer. In other words,
part of the "heavy tails" the robustness machinery exists to absorb may be a
units artifact rather than real break-taking behaviour.

**Fix:** assert the intended semantics in the loader (e.g. bound
`wall_minutes` to a plausible range and count/inspect anything dropped),
resolve the contradictory docstrings, and confirm the six cited outlier runs
are genuine long human sessions and not clock-unit mixing.

### 5. The "current doubling time" rests on two manually back-filled dates

`slope_now` is evaluated at `t_now = max(t_model)` — the latest dated
models. The two most recent/most capable models
(`flamingo_2`, `claude_opus_4_6_inspect`) are **missing** from Moss's
`release_dates.json` and are hardcoded to `2026-02-05` in
`RELEASE_DATE_OVERRIDES`. So the single most influential input to the
headline slope — the frontier points at the right edge — is a manual
override, and both are pinned to the *same* day. Any error there, or any
clustering artifact from co-locating two frontier points, moves `slope_now`
directly. The kink shape (which carries 0.635 of the stacked weight) reads
its post-2024 slope right off this region.

**Fix:** sensitivity-check the headline to perturbing these two dates by
±a few weeks; source them from a citable METR artifact rather than a literal.

---

## Tier 3 — validation does not cover the reported configuration

### 6. SBC validates a different model than the one in the headline

The headline is: **Student-t** layer, **stacked** over four shapes, full
scale (228/20), `sigma_est` median **1.25**. The SBC in `scripts/sbc.py` and
`docs/results.md` is: **Normal** layer, **linear only**, reduced scale
(50/8), `sigma_est` median **0.8**, 50 reps. So the calibration evidence
does not touch:

- the Student-t `nu` parameter and heavy-tail geometry actually used;
- the kink/superexp/logistic shapes (the kink drives the headline) — their
  extra parameters (`delta`, `t_k`, `beta2`, `h`, `t0`, `s`) are never
  rank-checked, and the logistic is *known* to be poorly identified
  (`docs/results.md`: "ceiling pinned at prior edge");
- the `sigma_est = 1.25` prior actually used;
- full-scale sparsity/geometry;
- the per-task latents `log_L`, `eps` (ZeroSumNormal), `a` — SBC only ranks
  the nine scalar hyperparameters.

Two SBC items are already in "Open work," but the headline-config gap is
broader than what's listed. Also minor: at 50 reps, `mu_L`'s cov50 = 0.38
and `sigma_eps`'s = 0.60 are the kind of deviation that warrants more reps
before "well-calibrated" is asserted, and the rank statistic
`(thinned < true).mean()` uses no randomized tie-breaking (fine for
continuous draws, but worth stating).

**Fix:** at minimum run reduced-scale SBC on the Student-t + kink
configuration under the 1.25 prior; rank `nu`, `delta`, `t_k`, and a couple
of `eps_i`/`log_L_i`.

### 7. `eps` is an unstructured catch-all with no difficulty covariate

`eps_i` has no predictors — it is pure per-task noise identified almost
entirely by the IRT layer (and, for sparse tasks, by its prior + the single
sum-to-zero constraint). Anything systematic that makes a task hard beyond
its length — task family, number of required tool calls, context length,
reasoning type — is dumped into `eps` and reported as irreducible "residual
difficulty." A model that regressed `eps` on even a couple of task-level
covariates would (a) shrink `sigma_eps` toward its true irreducible floor and
(b) tell you *why* length is a noisy proxy. As it stands the 8× headline
conflates "difficulty we can't predict" with "difficulty we didn't try to
predict."

**Fix:** add available task metadata (`task_family`, `task_source`) as `eps`
predictors, or at least report `sigma_eps` within family to show how much is
between-family vs genuinely irreducible.

### 8. Stacking weights that set the headline are within their own noise

The headline 2.8 months is a stacking mixture, 0.635 kink / 0.365 linear.
But the per-shape elpd values are `−2695.6` vs `−2696.2` — a 0.6-nat gap
against a **±81** standard error, and `docs/results.md` says outright "the
success/failure data alone can't strongly separate the shapes." Two concerns
compound:

- **PSIS-LOO leave-one-(model,task)-cell-out is weak here.** Tasks and models
  share latents (`log_L`, `eps`, `theta` trend), so removing one cell barely
  perturbs the fit; LOO has little power to discriminate a *smooth trend
  shape*, which is precisely why the elpd gaps are tiny and ~2% of pareto-k
  exceed 0.7. The weights are being read off a near-flat, noisy objective.
- Consequently the split between the 2.7 (kink) and 3.3 (linear) sub-headline
  is effectively **arbitrary within noise**, yet it's what nudges the stacked
  number below the linear-only 3.3. The honest headline is arguably "2.7–3.3
  months, shape-underdetermined," and the point estimate 2.8 conveys more
  precision than the model comparison supports.

**Fix:** report the stacked interval with an explicit "weights are within
elpd SE" caveat (already partially there), and consider leave-one-*model*-out
or leave-one-*task*-out (grouped LOO) as a more honest test of trend shape.

---

## Tier 4 — smaller structural / robustness notes

### 9. No lapse/guessing floor in the 2PL
`P = sigmoid(a*(theta − difficulty))` goes to 0 and 1 at the extremes. Real
agents fail trivial tasks (harness/formatting errors) and occasionally pass
hard ones by luck. With no lower/upper asymptote, such cells are explained by
distorting `a_i` and `eps_i`. The PPC flags exactly this at the trivial
`<0.1 min` bin (obs 1.000 vs pp [0.996, 1.000]). A 4-parameter IRT (or a
small fixed lapse rate) would absorb it.

### 10. `sigma_est` is prior-only and unfalsifiable in-sample
No task carries both a baseline time and an estimate annotation, so
`sigma_est` is identified purely by its prior (calibrated to Barry's external
60%-within-3× figure). This is handled honestly (sensitivity run at 0.8 vs
1.25), but it means the 67 estimate-only tasks' lengths rest on an external
number that the data cannot check, and those tasks skew long (up to 30h) —
i.e. they sit at the high-leverage end of the length axis the trend is fit
against.

### 11. Weibull censoring branch is dead code
`pm.Censored` never activates (`time_limit == 0` for all human rows on the
snapshot). Documented and harmless, but it is untested machinery that will
fire automatically on a future data pull — worth a unit test with a synthetic
censored row so it doesn't first execute in production.

### 12. Reproducibility fragility
`get_sota_models` and `RELEASE_DATE_OVERRIDES` hardcode model→alias→date→p50
joins and `raise` on any mismatch; a routine upstream `runs.jsonl` refresh
(new model, renamed alias) breaks the pipeline rather than degrading
gracefully. The `sigma_est` Barry-calibration constant (1.25) and the SOTA
set (14 models) are snapshot-specific literals in code.

### 13. Prior pushforward on the steep slope not shown
`beta1 ~ N(0, 1)` while the fitted kink post-2024 slope is 3.14 — about 3
prior SDs out. Plausibly fine (the data dominate), but a prior-predictive
check on `slope_now`/doubling-time would confirm the `N(0,1)` slope prior and
the `N(0,1.5)` intercept aren't mildly fighting the steep-acceleration
region. Listed under "Open work" for the shape-specific priors; the base
`beta1` prior deserves the same check.

---

## What holds up

To keep the critique calibrated — these are genuinely well-handled:

- The **identification fix** (ZeroSumNormal on `eps` instead of Moss's
  hard-anchored thetas) is correct and preserves the log-minute
  interpretation of `theta`; the footnote derivation of the ridge is right.
- **Non-centered** parameterization of `log_L`, `eps`, `u`, `a` is the
  correct response to the funnel geometry described.
- The **Student-t vs Weibull** investigation is thorough and the negative
  Weibull result is well-argued (Gumbel-min skew vs observed right-skew).
- The **cut-model** check of Barry's circularity warning is exactly the right
  modularized-inference tool, and the diagnostic quantifies the effect
  (corr −0.53, headline delta ~1 day) rather than hand-waving it.
- The doubling-time headline is **slope-based**, which (per #1) is the one
  summary most robust to the conditional/marginal issue.

---

## Priority order for follow-up

1. Marginal `h50` + explicit statement of what the conditional/marginal split
   does and doesn't protect (#1).
2. Variance decomposition making clear how much of `sigma_eps` is length-noise
   vs true heterogeneity (#2), ideally with `eps` covariates (#7).
3. Survivorship-bias check on success-conditioned human times (#3) and the
   duration-units audit (#4).
4. SBC on the actual headline configuration (#6).
5. Honest "shape-underdetermined" framing of the stacked headline (#8) and
   the two hardcoded frontier dates' sensitivity (#5).
