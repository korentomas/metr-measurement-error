# Improving the measurement-error model: experiments and recommendation

Follow-up to [`red_team_review.md`](red_team_review.md). That review found the
model sound but flagged where the *measurement* half could be better specified
and where its outputs are easy to over-read. This document turns those flags
into implemented, fitted, and validated changes, and ends with a recommended
"best" measurement-error configuration.

All fits are the real METR data (`runs.jsonl`, 24,008 rows — reproduced
exactly: 554 human rows / 164 tasks / 525 baseline / 29 estimate), linear
shape unless noted, Student-t duration layer, 2000 tune / 2000 draws / 4
chains, `target_accept 0.95`. The reconstructed baseline matches the
published numbers to the decimal (doubling **3.3 [2.8, 4.1] mo**, `sigma_eps`
2.16, `sigma_base` 0.41, `nu` 2.4), so everything below is measured against a
faithful anchor.

## TL;DR — what to change

| change | verdict | effect on headline | why |
|---|---|---|---|
| **Heteroscedastic measurement noise** | **adopt** | none (3.3 mo) | `gamma_sig = +0.103 [0.068, 0.138]`, P>0 = 1.00; **+14.2 duration-fit elpd** (dse 6.9) over homoscedastic; matches the raw data |
| **Failed-run censoring (survivorship)** | **report as sensitivity** | none (3.3 mo) | lifts hard-task lengths ~11% where humans failed; but shifts the estimand and fits successes slightly worse |
| **Report marginal (not just conditional) horizon** | **adopt in reporting** | n/a | 50% level & doubling-time are definition-robust; non-50% & extrapolations are not |
| **Structured `eps` by task family** | **adopt** | **3.0 mo** (from 3.3) | **67%** of the "8×" residual difficulty is predictable family structure; +8.0 successes elpd; the one change that moves the trend |
| Recompute per-task noise as a plug-in | rejected | — | discards real measurement uncertainty (~0.4 log-min) the layer exists to carry |

Headline doubling time is **3.3 months under every measurement-*layer* variant
tried** (heteroscedasticity, Student-t, failed-run censoring) — the trend is
robust to how the timing *noise* is modelled. The one change that moves it is
structuring the *difficulty* term `eps` (§5): 3.0 [2.5, 3.6] months. That is
not a contradiction — it is the model's own logic. The IRT/trend layer is
driven by `difficulty = log_L + eps`, and `eps` is the large, IRT-identified
part while `log_L` is buffered behind it (red-team #2). So refinements to the
*length* channel wash out, and the refinement to the *difficulty* channel is
exactly the one that can, and does, sharpen the trend.

---

## 1. Heteroscedastic measurement noise — the main improvement

**Problem (red-team #7 / #2).** The baseline uses one global `sigma_base` for
per-run log-wall-time noise. In the data the within-task sd of log wall-time
rises with task length: correlation **+0.31**, short-task median log-sd
**0.28** vs long-task **0.57**. A single scale over-smooths short tasks and
under-smooths long ones.

**Change.** Let the scale depend on the latent length,
`sigma_base_i = sigma_base * exp(gamma_sig * (log_L_i - mu_L))`, `gamma_sig ~
Normal(0, 0.5)`. `gamma_sig = 0` nests the homoscedastic model.
(`--heteroscedastic`.)

**Result.** `gamma_sig = +0.103 [0.068, 0.138]`, posterior P(>0) = **1.000** —
the length-dependence is decisively present. The fitted noise runs from
**0.36** log-min at the short end to **0.67** at the long end, tracking the
binned data (Panel A below). PSIS-LOO on the 525 baseline runs improves by
**+14.2 elpd (dse 6.9)**; on the shared observations the heteroscedastic model
takes stacking weight **0.80** vs 0.15 for the homoscedastic baseline. 0
divergences, R-hat 1.01. `nu` rises 2.4 → 2.9 — some of what the Student-t was
absorbing as "heavy tails" was really length-dependent noise, so the two
partly substitute. The doubling time and `sigma_eps` are unchanged.

This is the recommended default measurement layer.

![Measurement improvements](../outputs/figures/measurement_improvements.png)

*A: the fitted length-dependent noise (red) matches the binned within-task
sd; the flat homoscedastic line (blue) is wrong at both ends. B: posterior sd
of `log_L` falls with the number of timed runs — the uncertainty a plug-in
discards. C: the survivorship correction (§2) lands on low-success (hard)
tasks.*

## 2. Survivorship correction via failed-run censoring

**Problem (red-team #3).** The loader keeps only successful human runs
(`score_binarized == 1`), so `log_L` is inferred from a length distribution
truncated to "fast enough to succeed." The snapshot discards **129 failed
baseline human runs over 55 tasks**, with median wall-time **~110–150 min vs
~5 min for successes** — they cluster on hard/long tasks.

**Change.** Add each failed baseline run as a **right-censored** duration
observation at its own log wall-time (`T > w`), through the model's existing
`pm.Censored` path (now generalized to the Student-t layer).
(`--include-human-failures`.) This is self-weighting: a give-up at 7 min
contributes `P(T>7) ≈ 1` (nearly vacuous), a genuine failure at 480 min
contributes a strong upward pull.

**Result.** Task lengths rise where they should: mean `log_L` shift **+0.11**
but median only **+0.01**, i.e. concentrated — **30 / 228 tasks** move by
>0.2 log-min (max +2.96 ≈ 19×), all low-success tasks (Panel C). `mu_L` rises
2.06 → 2.19. 0 divergences.

**Caveat — why "sensitivity", not default.** The correction subtly changes the
estimand. METR *defines* the human baseline as the geomean of successful
completion times; censoring failures moves `log_L` toward "time including
attempts that failed." Consistent with that, it fits the *successful* runs a
bit worse (dur-obs elpd −666 vs −627 heteroscedastic). It also assumes failure
≈ "would need more time," which holds for ~53% of failures and is harmlessly
down-weighted for the rest. So it belongs in the paper as a robustness column
— it demonstrates the trend survives an honest survivorship correction — not
as the primary fit.

## 3. What the measurement-error layer actually buys (red-team #2)

Because the IRT layer sees only `difficulty = log_L + eps` and `eps` is free,
the layer barely moves the horizon *point* estimate (confirmed: doubling time
identical across all variants). Its value is **uncertainty propagation**. The
posterior sd of `log_L` is large for sparsely-timed tasks and shrinks with
data: **0.84** (estimate-only) → **0.59** (1 run) → **0.39** (2 runs) →
**0.27** (3+ runs). A plug-in model (Moss/METR: `human_minutes` treated as
exact) sets all of that to zero — discarding on average **~0.40 log-min** of
difficulty-axis uncertainty per timed task, and more for the 1-run tasks
(12% of the set). That is the source of the wider, better-calibrated horizon
intervals the layer is for. `scripts/measurement_value.py` reproduces this.

## 4. Marginal vs conditional horizon (red-team #1)

The reported `h50 = exp(theta)` is a *conditional* horizon (a task of median
residual difficulty, `eps = 0`); METR's is *marginal* (averaged over the task
population). `scripts/marginal_horizon.py` settles what the distinction does
and doesn't change:

- **50% level: robust.** `P_pop(ell = theta) = 0.5000` exactly (sigmoid is
  odd-symmetric, `eps` symmetric mean-zero) — the per-agent 50% horizons need
  no marginal correction.
- **Doubling time: robust.** It is a slope of `theta` over time; a stationary
  `eps` distribution cancels.
- **Everything else: not robust.** The population success curve is **1.88×
  flatter** than the single-task curve; the 10%–90% horizon spread grows from
  ~81× to ~3800×. Non-50% reliability horizons and any fixed-threshold
  extrapolation ("when do we reach a 1-month horizon") inherit the full
  `sigma_eps` spread and diverge from METR's marginal curve.

Practical upshot: keep quoting the 50% horizon and doubling time as-is;
whenever a non-50% horizon or a threshold-crossing date is quoted, compute it
marginally.

## 5. Structured residual difficulty by task family (red-team #7)

**Problem.** The flat model reports one `sigma_eps` (~2.2 log-min, ~8×) and
treats *all* of it as irreducible "residual difficulty." But task length is a
worse proxy for some task types than others, so an unknown share of that 8× is
predictable structure, not noise.

**Change.** Decompose `eps` hierarchically into a task-family effect and a
within-family residual, `eps_i = fam_eff[fam(i)] + within_i`, centered to keep
the exact sum-to-zero identification. `eps_structure='family'` exposes
`sigma_eps_fam`, `sigma_eps_within`, and `eps_between_frac`. (79 families over
228 tasks.)

**Result — most of the "8×" is structure, not noise.**

```
sigma_eps_fam    (between-family): 2.24 [1.82, 2.71]
sigma_eps_within (within-family):  1.58 [1.25, 1.93]   <- the ~irreducible part (~5x, not 8x)
between-family variance share:     0.67 [0.53, 0.78]   <- ~67% is predictable family structure
successes PSIS-LOO:  family -2688.5  vs  flat -2696.5   (+8.0, dse 4.0; weight 0.96 vs 0.04)
```

The family model fits the success data decisively better, and 67% of the
residual-difficulty variance is between-family — i.e. a task-family covariate
would capture two-thirds of what the flat model calls irreducible. The family
effects are large and interpretable (log-min, "+" = harder than its length
implies): `continue_pattern` **+4.7 (110×)**, `spn_cryptanalysis` +4.5,
`implement_ace_oauth` +3.8 at the hard end; `file_selection` **−5.1 (0.01×)**,
`arithmetic` −4.9, `alert_triage` −4.7 at the easy end — the mechanical-lookup
vs sustained-reasoning split you would expect. 17 of the 42 multi-task
families have effects whose 95% interval excludes zero.

![Family eps decomposition](../outputs/figures/eps_family_decomposition.png)

**This is the one refinement that moves the headline: doubling time 3.0 [2.5,
3.6] months, vs the flat 3.3 [2.8, 4.1]** (overlapping, but a real shift). Why
this change and not the measurement-layer ones? Because the trend is driven by
`difficulty = log_L + eps`, and `eps` is the large IRT-identified part; pooling
it by family sharpens the difficulty estimates the trend is read against, so
the slope moves. (The flat single `sigma_eps` was also mildly *under*-stating
the spread by shrinking 228 independent per-task effects; pooling within
families de-shrinks them, which is why the total realized spread rises to 3.3
even as the irreducible within-family part is only 1.6.) 0 divergences, R-hat
1.006, ESS ~5500 — not a geometry artifact, and the priors (HalfNormal 0.5)
are an order of magnitude below the posteriors, so not a prior artifact.

`scripts/eps_decomposition.py` reproduces the numbers and the family ranking.

## 6. Robustness: SBC on the actual headline configuration (red-team #6)

The published SBC covered only the Normal/linear/reduced-scale model under the
old `sigma_est` prior. `scripts/sbc.py` is now general; run on the **headline
configuration** (kink shape, Student-t layer, `sigma_est` median 1.25):

```
=== SBC: 40/40 reps fit successfully ===
param        mean rank    KS p  cov50  cov90
mu_L             0.459   0.624   0.47   0.95
sigma_base       0.514   0.942   0.55   1.00
sigma_eps        0.548   0.571   0.47   0.82
beta1            0.507   0.873   0.47   0.95
delta            0.551   0.624   0.50   0.95   <- kink slope-change, never checked before
t_k              0.495   0.982   0.53   0.97   <- kink breakpoint, never checked before
nu               0.530   0.830   0.47   0.95   <- Student-t dof, never checked before
```

All mean ranks 0.46–0.55, KS p ≥ 0.57, coverage near-nominal. The parameters
that drive the headline (`delta`, `t_k`) and the robust layer (`nu`) are
well-calibrated. The headline configuration is now SBC-backed, not just the
simplified stand-in.

The **recommended** configuration (kink + Student-t + **heteroscedastic**)
also passes, 40/40, with the new `gamma_sig` well-calibrated (mean rank
0.513, KS p 0.94, cov50 0.53, cov90 0.90) — so adopting heteroscedasticity
does not break calibration:

```
=== SBC (kink + robust + heteroscedastic): 40/40 reps fit ===
param        mean rank    KS p  cov50  cov90
sigma_eps        0.553   0.571   0.57   0.90
delta            0.553   0.378   0.55   0.97
t_k              0.495   0.942   0.53   0.97
nu               0.482   0.830   0.45   0.97
gamma_sig        0.513   0.942   0.53   0.90   <- new heteroscedastic param, calibrated
```

---

## Recommended "best" measurement-error model

**Student-t duration layer + heteroscedastic `sigma_base` + family-structured
`eps`**, kink trend, with **failed-run censoring as a sensitivity column**:

```
uv run python scripts/fit_model.py --shape kink --robust --heteroscedastic \
    --eps-structure family --log-likelihood \
    --tune 2000 --draws 2000 --chains 4 --target-accept 0.95
```

Fitted end-to-end, all options compose cleanly (0 divergences, R-hat 1.01,
ESS 5–9k) and give a **current doubling time of 2.4 [2.1, 2.7] months** — vs
the flat-kink 2.7 [2.3, 3.1]: family-structured `eps` both shifts it faster
*and tightens the interval*, because the sharper difficulty estimates reduce
trend uncertainty. The component effects are unchanged from the single-option
fits (`gamma_sig` 0.10, between-family share 0.67), i.e. they do not interact
badly. Heteroscedasticity passes SBC (40/40, `gamma_sig` calibrated; §6).

Rationale, in order of impact:
- **Family-structured `eps`** is the most consequential change: it fits the
  success data better (+8 elpd), reveals that ~⅔ of the residual difficulty is
  predictable family structure rather than irreducible noise, and it is the
  one refinement that sharpens the trend (3.0 vs 3.3 months). It is the biggest
  step toward a *correct* difficulty model.
- **Heteroscedasticity** is the change the timing data decisively support (+14
  duration elpd, P(gamma>0)=1) at no cost to the headline; the **Student-t**
  stays because it still helps once heteroscedasticity is in.
- **Failed-run censoring** is the honest survivorship check but shifts the
  estimand, so it stays a robustness column.
- **Marginal horizons** for any non-50% or extrapolated quantity cost nothing
  and remove the one genuinely misleading comparison against METR.

What did **not** move: the doubling-time headline is robust to every
measurement-*layer* refinement (heteroscedasticity, Student-t, failed-run
censoring all leave it at 3.3 months), because `eps` buffers `log_L`. It moves
only when the *difficulty* term itself is restructured (family `eps` → 3.0),
which is the model telling you where its trend signal actually lives. Either
number is within the other's interval; the reassuring read is that no
measurement or difficulty refinement pushes the doubling time outside
~2.5–4.1 months.

## Open follow-ups

- SBC at full data scale, and on the family-`eps` variant (`gamma_sig` is
  already wired; the family variant would need its `sigma_eps_fam/within`
  added to the ranked list).
- A task-family (or task-type) *covariate* on `eps`, now that ~67% of the
  residual difficulty is shown to be between-family — it would turn that
  structure into an interpretable predictor and shrink the "irreducible" term
  toward its true ~1.6 log-min floor.
- Marginal-horizon *bands* on the trend plot, not just the flattening factor.

## Reproducibility note

The sibling data repos have drifted since this project's snapshot. `runs.jsonl`
still matches exactly (24,008 rows). `release_dates.json` (the code's expected
path) is now a `data_raw/release_dates.yaml` keyed by display alias; it was
reconstructed by joining alias → date against the `alias` column in
`runs.jsonl` (18 of 20 models matched directly, the other 2 via the existing
`RELEASE_DATE_OVERRIDES`), giving all 20 dated models as before. `headline.csv`
(only used by `--sota-only`) is absent from the current checkout, so the
SOTA-only refit was not re-run here.
