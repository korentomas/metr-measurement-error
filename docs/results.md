# Full results

This document gives every number and plot behind the headline claims, in the
order of production. It lives outside the README so that the README stays
focused on the codebase itself.

The Student-t measurement layer is the preferred variant (a later section
tells why), stacked across the four trend shapes. This document reports what
that combination says. The plots come from `scripts/make_figures.py` and the
fitted `.nc` files in `outputs/`. The figures deliberately use the same axis,
tick, and annotation conventions as Moss's own plots. Thus, the two sets can
sit side by side without translation.

## Doubling time and the trend shape

Stacked across the four trend shapes, the current doubling time is **2.8
months [2.3, 3.8]**. The weight goes mostly to the kink shape (the exact
weights are in the table below). The two main shapes do not fully agree on
their own. A linear fit alone gives 3.3 [2.8, 4.0] months. When the slope can
bend near 2024 (the kink shape), the number comes down to 2.7 [2.3, 3.1]. But
that difference is small next to the uncertainty in the elpd estimates. Thus,
the correct read is not "kink wins." The correct read: the success/failure
data alone cannot strongly separate the shapes. The stacked interval is
somewhat bimodal, not clean. Extrapolations must use the stacked mixture.

What that gives you, stance by stance. If you trust the fitted stacking
weights (0.635 kink and 0.365 linear, with nothing from superexponential and
logistic), you get the stacked 2.8 [2.3, 3.8] months above. Perhaps you think
instead that the kink only fits a bend in recent, noisier data, and you would
rather keep the plainer one-slope model. Then you would use the linear-only
3.3 [2.8, 4.0] months. The two numbers are not far apart, and their intervals
overlap heavily. Thus, neither stance changes the headline by much. Which one
is more convincing is a real judgment call, and the data here do not make that
call for you.

```
shape       elpd_loo        weight   doubling time now (months)
kink        -2695.6 +- 81.4  0.635    2.7 [2.3, 3.1]
linear      -2696.2 +- 81.2  0.365    3.3 [2.8, 4.0]
superexp    -2696.9 +- 81.3  0.000    2.0 [1.7, 2.5]
logistic    -2698.4 +- 81.7  0.000    4.1 [2.2, 9.9]
STACKED                               2.8 [2.3, 3.8]
```

(A caveat: approximately 2% of the PSIS pareto-k values are more than 0.7.
These are the usual high-leverage model-task cells. The weights are this
insensitive to elpd noise, so the conclusion is unlikely to move. But exact
re-fit LOO on the flagged cells would be the rigorous check.)

### Same models as METR: the SOTA-only refit

The obvious suspect for the gap against METR's 4.2 months was the model set.
METR's headline trendline does not use every model that they evaluated. It
keeps only the models that were SOTA at their release date, defined as a
running frontier. (A model stays in the set if its 50% horizon matches or
beats the best of all models released up to that time.) My fit used all 20
dated models. Thus, I refit on exactly the 14 agents that METR's published
headline trend uses (`--sota-only`). That flag reproduces their running-max
rule from their own logistic fits. It drops Claude 3 Opus, Claude 4 Opus,
Claude 4.1 Opus, GPT-4 Turbo, GPT-5.1-Codex-Max, and GPT-5.3-Codex. The run
settings are the same as everywhere above.

It is not the model set. The stacked estimate under METR's own 14 models is
**2.8 [2.3, 4.2] months**, the same headline as before. Per shape: kink 2.7
[2.3, 3.2] (weight 0.66), and linear 3.6 [3.0, 4.5] (weight 0.34). The
linear-only number does drift approximately a third of the way toward METR's
4.2. But the stacked headline does not move. The kink shape keeps most of the
weight, and the six dropped models almost do not touch its post-2024 slope.

What separates the numbers is the trend shape or, equivalently, the time
window. METR's 4.2 is a single exponential fit through the whole 2023 to 2026
SOTA series, with the slower pre-2024 stretch included. The stacked fit here
mostly backs a bend at early 2024, with a faster slope after it. When the
windows align, the disagreement disappears. METR's own post-2024 SOTA
trendline is 3.0 months [2.4, 4.0], directly on top of the kink fit's 2.7.
And the comparison with Moss's 4.3 was never about model selection. He fits
all the v1.1 models, the same as my default, and his headline is linear-only
by construction. Thus, the right pair there is my linear 3.3. The remainder
comes from the added measurement-error layer and the per-model random
effects, not from the input data.

![Horizon vs release date, all four trend shapes with credible bands, forecast to 2029](../outputs/figures/horizon_trend.png)

The per-model 50% horizons (the points, from the kink fit) go from a few
minutes in 2023 to approximately 20 hours by early 2026. That is the same
post-2024 acceleration that METR reports from their own pipeline. All four
fitted shapes are on the plot at the same time. The x-axis runs to 2029,
approximately three years past the last dated model (the dotted line). That
forecast window is the point of the overlay. Over the span that the data
cover, the shapes mostly agree with each other and with the points. That
agreement is exactly why the stacking weights above do not land decisively
on one shape. Then the shapes separate as soon as the data end. By 2029, the
linear trend reaches a horizon of approximately three years. The kink trend
is an order of magnitude past that. The superexponential is off the chart
entirely. And the logistic flattens near three days. The full breakdown:

| shape | trend f(t_m) | weight | what it found |
|---|---|---|---|
| kink | `beta0 + beta1*t + delta*softplus((t-t_k)/w)*w` | 0.635 | breakpoint at early 2024 (t_k = -0.87 +- 0.15). slope jumps from 0.41 [-0.43, 1.21] to 3.14 [2.70, 3.59]. sigma_u falls 0.70 -> 0.38, a sign that it absorbs real structure, not noise |
| linear | `beta0 + beta1*t` | 0.365 | one steady slope. no bend is necessary to fit the data almost as well |
| superexponential | `beta0 + beta1*t + beta2*t^2` | 0.000 | the same acceleration as the kink, told as curvature: positive beta2 in log-horizon over time |
| logistic | `beta0 + h*sigmoid((t-t0)/s)` | 0.000 | the ceiling parameter is pinned at its prior edge, and the inflection point is pinned at the data edge. it degenerates to "still rising." there is no saturation signal in this data |

The wide, poorly identified slope in the logistic row is also why its band is
visibly the widest of the four on the plot above.

The same point, that the shape is underdetermined, shows again if you look
directly at the doubling time that each shape implies now, instead of the
horizon curve that it draws:

![Doubling time now, by trend shape and stacked mixture](../outputs/figures/doubling_time_density.png)

Kink and linear make two narrow peaks that mostly overlap, approximately a
month apart. The superexponential sits to the left, faster. The logistic is
the flat, wide smear that goes out past a year. That smear is its poorly
identified slope again, now seen as a distribution instead of a fan of lines.
The black curve is the stacked mixture (the same weights as the table above),
and it is visibly not a single clean peak. Kink and linear disagree with each
other more than each one disagrees with itself, and that pulls the mixture a
little bimodal.

## sigma_eps: the residual difficulty term

**sigma_eps = 2.07 to 2.28 log-minutes**, dependent on the shape and the
robustness variant. That is a residual task-difficulty spread of
approximately 8x at fixed task length. This number is the empirical case for
the whole measurement-error and heterogeneity layer. It is also the thing
that any horizon extrapolation ("when do we get to a 1-month horizon") is
most sensitive to.

Think through that number. Take a task that looks like ten minutes of human
time. Under an 8x spread, that task can be as hard for the model as a task
of an hour and twenty minutes. The reverse also holds. A task that reads as
an hour-long slog can be approximately as easy for the model as a ten-minute
task. In that form, the number does not seem unreasonable to me. Task length
is only a rough proxy for what makes a task hard for a model:

- how much the model must hold in context
- how many tool calls it must get right in a row
- whether the reasoning is the kind that these models are good at.

I would expect that proxy to be noisy by more than a small factor. If
anything, my prior guess was an even wider range.

This is our version of the figure that Jonas Moss uses to make the same point
in his post, plotted in the same way. Each task's difficulty multiplier is
`exp(eps_i)`: how many times longer or shorter its equivalent difficulty time
is than its actual human time. The plot shows that multiplier against the
posterior-mean length `L_i`, with dotted +-1 sigma and +-2 sigma reference
bands. Thus, you can read the scatter directly against the fitted `sigma_eps`
above.

![Residual task difficulty vs task length, difficulty multiplier against posterior-mean task length](../outputs/figures/difficulty_residual.png)

The scatter is not perfectly flat. The shortest tasks (well under a minute,
almost always solved by every model) cluster low and tight. The longest tasks
drift upward. Both patterns are plausibly real, not artifacts. A trivial task
does not leave much room for difficulty variation. And the very longest tasks
may demand more sustained precision than their length alone implies. But
across the bulk of the range, roughly six seconds to four and a half hours,
there is no visible trend. The vertical spread stays enormous throughout.
Most tasks sit inside the +-1 sigma band (0.11x to 9.3x), and a fair number
are out past +-2 sigma (0.012x to 86x). Tasks that take persons approximately
the same time can be wildly different in how hard they are for a model.

## Posterior predictive check

All 20 models' success rates land inside their 95% posterior predictive
intervals. Every task-length bin is calibrated except the trivial
<0.1-minute bin (observed 1.000, posterior predictive [0.996, 1.000]).

![Posterior predictive check: observed vs predicted success rate by task-length bin](../outputs/figures/ppc_calibration.png)

## Student-t and Normal measurement layers (`--robust`)

The wall-clock data has genuinely heavy tails. Persons take breaks mid-task,
so a few runs are far longer than the other attempts at the same task. The
worst cases are 3 to 4 log units off the task median (a 249-minute run on a
4-minute-median task, and a 2,185-minute run on a 96-minute-median task).
Under a Normal likelihood, single runs like these drag the task's latent
`log_L` and inflate `sigma_base` for every task, not only the tasks with
outliers. The `--robust` flag changes the baseline-run likelihood to a
Student-t with an estimated degrees-of-freedom parameter (prior mean 20,
near-Normal a priori). Thus, the data teach the tails, instead of an imposed
shape.

The data decisively reject Normal tails. The fitted dof is 2.4 [1.8, 3.2],
approximately as heavy as a t-distribution gets. Thus, the Normal fit treated
outlier noise as core noise. Its `sigma_base` (0.79) is almost double the
Student-t's fitted scale (0.41) for what is supposed to be the same everyday
within-task variability. None of this moves the headline. The doubling time
is 3.4 [2.8, 4.2] months under Normal and 3.3 [2.8, 4.0] under Student-t. And
`sigma_eps` is 2.14 against 2.22. The numbers are close enough that the
outliers never drove the trend. Now that is demonstrated, not assumed.

![Outlier pull on latent task length: Normal vs Student-t, six worst-outlier tasks](../outputs/figures/outlier_pull.png)

I deleted no data for this figure. It shows the same six tasks under both
likelihoods. The Student-t fit's pull toward the task's observed median
duration shrinks in every case (for example, `questions/swift`: +1.43 log
units under Normal, down to +0.15 under Student-t).

## Weibull duration likelihood (`--duration-dist weibull`)

Moss suggested this variant in the post itself: "I would also try a Weibull
distribution instead of log-normal, since the log-normal is typically
heavier-tailed and the Weibull is easier to justify on theoretical grounds."
I tried it, median-matched, so that `log_L` keeps the same "log median wall
time" meaning under every variant.

It samples as cleanly as the other variants, but it fits the duration data
worse than both. PSIS-LOO on the 525 baseline runs (Jacobian-corrected onto a
common duration scale):

```
likelihood   elpd_loo (duration scale)   vs Student-t
studentt     -1097.5 +- 85.6
lognormal    -1164.9 +- 85.8             -67.5 (dse 14.7)
weibull      -1219.3 +- 86.6             -121.8 (dse 25.0)
```

The Weibull is the worst of the three, and by enough that this is not LOO
noise. The reason is a shape mismatch, and it is visible directly in the
data, not only in the LOO number. Our within-task residuals are right-skewed
with heavy tails. The log of a Weibull is a Gumbel-minimum, which is fixed
left-skewed, whatever its shape parameter does. In the figure below, the gray
histogram is the pooled within-task log-residuals (each baseline run's log
duration minus its own task's median, for tasks with two or more timed runs).
The three curves are the log-normal, Student-t, and Weibull-implied densities
from the fitted parameters of each variant.

![Baseline-duration residuals: observed data vs the log-normal, Student-t, and Weibull-implied densities](../outputs/figures/duration_dist_comparison.png)

The Student-t (red) holds the sharp peak and the long right tail at the same
time. The log-normal (gray) is close, but a bit too spread in the center. The
Weibull-implied curve (purple, dashed) can do neither. It is forced into
roughly the same shape on both sides, so it undershoots the peak and
mismatches the skew. That is the finding of skew +1.09 and excess kurtosis
8.2, made visible: structurally the wrong shape for this data. The headline
does not change either way (doubling time 3.3 [2.8, 4.1] months, and
`sigma_eps` 2.21 against 2.22). Thus, this is a clean negative result.
Moss's suggestion was reasonable for task-completion times in the abstract.
But our wall-clock times (with breaks and multi-hour interruptions included)
are heavy-tailed and right-tailed in a way that a Weibull cannot be.

## sigma_est recalibrated to Barry's 60%-within-3x finding

In the comments on Moss's post, Alexander Barry reports a check on tasks
where both annotation types exist. Only approximately 60% of the
estimate-only human time annotations land within a factor of 3 of the actual
baseline time. Our data cannot
check this directly (none of the 228 tasks carry both annotation types).
Thus, his finding enters as an external prior. A match to his 60% figure
implies a prior median of 1.25 log-minutes for `sigma_est`,[^2] well above
the 0.8 that we used before. A refit with the wider prior moves the posterior
for `sigma_est` up substantially (0.65 [0.27, 1.28] -> 0.85 [0.38, 1.52]).
But that is almost the only thing that moves. The doubling time and
`sigma_base` land within a few hundredths of where they started, and
`sigma_eps` shifts a bit more (2.22 -> 2.17). The headline is robust to a big
shift in this prior. Only the latent lengths of the 67 estimate-only tasks
become honestly less certain.

## Estimate-only feedback: Barry's circularity warning, checked

Barry also raised a sharper methodological point about this class of model.
When you jointly model uncertainty over the discrimination `a_i` and the task
length `log_L_i`, the success/failure outcomes can move the inferred lengths.
A Bayesian model built this way can then be more misleading than the
frequentist original that it replaces. The exposed surface here is the 67
estimate-only tasks, whose only timing datum is one annotation. The IRT layer
has room to re-date such a task to explain its outcomes. It can pull a task
shorter when models usually solve it, and pull a task longer when models
usually fail it. The trend is then partly fit to lengths that the outcomes
themselves chose.

`scripts/estimate_feedback_diagnostic.py` measures how much of this occurs in
a saved fit. For each estimate-only task, it computes the shift between the
posterior mean of `log_L_i` and the raw annotation. It also computes the
correlation of those shifts with the task's pooled success rate. On
`outputs/fit_kink_robust.nc`: the mean shift is +0.03 log-minutes (no
aggregate bias), the sd is 0.26, and corr(shift, success rate) = -0.53. Thus,
the mechanism is real and operates in the fitted posterior, in the predicted
direction. The per-task pulls stay well inside the posterior `sigma_est`
scale (approximately 0.72), a fraction of the annotation noise that the model
already assumes.

A cut-model refit answers whether the feedback moves the headline
(`--cut-estimate-feedback` in `scripts/fit_model.py`). For the 67
estimate-only tasks, the IRT layer sees the raw annotation as a fixed
constant in place of the latent `log_L_i`. That breaks the loop. With a
single annotation and no timed runs, the measurement-only posterior mean of
`log_L_i` essentially is the annotation. Thus, the cut clamps those tasks at
their measurement-only estimate. The 161 baseline-informed tasks are
untouched, and `eps_i` stays free. I refit at the same settings as the robust
fits (2000 tune / 2000 draws / 4 chains, nutpie, target_accept 0.95,
Student-t, 0 divergences, max R-hat 1.04):

```
shape    joint (feedback on)   cut (feedback off)
kink     2.7 [2.3, 3.1]        2.6 [2.3, 3.0]
linear   3.3 [2.8, 4.0]        3.3 [2.8, 4.0]
```

The kink medians are 2.65 and 2.62 months before they are rounded, a delta of
approximately one day. The linear numbers agree to the same precision. A
stack of the two cut shapes gives 2.6 [2.3, 3.0], with all weight on kink
(elpd gap 3.6 +- 2.1). The superexponential and logistic shapes already took
weight 0.000 in the four-shape joint stack, whose headline is the 2.8
[2.3, 3.8] above. Thus, the choice between feedback on and feedback off
changes the doubling time by a few hundredths of a month per shape. The
circularity is present and detectable, and its effect on the headline is
second-order. The refit demonstrates that.

One caveat survives the check. For the estimate-only tasks, `eps_i` and
`log_L_i` are only weakly separately identified (the IRT layer informs their
sum). Thus, do not read the posterior `log_L` for those tasks as a purified
human-time estimate.

## Simulation-based calibration

Reduced-scale SBC (Talts et al. 2018): 50 tasks x 8 models x 50
replications. The simulation copies the real data's sparsity (1 to 3 timed
runs per task, 20% estimate-only, and 8 IRT attempts per model-task cell).
Each replication refits at 800 tune / 500 draws / 2 chains. The result:
**pass, 50 of 50 replications fit**.

```
param        mean rank    KS p  cov50  cov90
mu_L             0.432   0.190   0.38   0.92
sigma_L          0.570   0.256   0.48   0.92
sigma_base       0.457   0.544   0.48   0.94
sigma_est        0.442   0.190   0.58   0.92
sigma_a          0.545   0.434   0.50   0.86
sigma_eps        0.479   0.256   0.60   0.94
beta0            0.505   0.881   0.46   0.86
beta1            0.466   0.190   0.50   0.88
sigma_u          0.525   0.662   0.46   0.84
```

The mean normalized ranks are all in 0.43 to 0.57 (the target is 0.5). The
KS-against-uniform p is at least 0.19 on every parameter. The
central-interval coverage is within Monte Carlo error of nominal everywhere.
There is no sign of the classic pathologies (a rank pile at 0 or 1 shows
overconfidence, a hump at 0.5 shows underconfidence, and a mean shift shows
bias). This run is for the Normal-layer linear model at reduced scale, under
the original `sigma_est` prior (median 0.8). SBC on the headline
configuration (kink, Student-t, `sigma_est` median 1.25) is now done: refer
to [`measurement_error_improvements.md`](measurement_error_improvements.md),
section 6. SBC at full data scale stays open (refer to Open work in the
README).

I also generated the rank histograms (`outputs/figures/sbc_ranks.png`), but I
do not embed them here. At only 50 replications per parameter, they are too
noisy to add anything past the table above. (A truly uniform histogram at
n = 50 can look almost as lumpy as these do, by chance.) The KS test is the
right tool for a claim of this size. The same argument applies to
`outputs/figures/stacking_weights.png`. It shows the four numbers already in
the stacking table above, and a bar chart of them does not teach anything
that the table does not.

[^2]: Barry's finding is `P(|N(0, sigma)| < ln 3) = 0.6`, which inverts to
    `sigma = ln(3) / Phi^-1(0.8) ~= 1.305` log-minutes of total estimate
    noise. The baseline geometric mean has its own noise contribution of
    approximately 0.3. Net of that, the result is the approximately 1.25
    prior median used for `sigma_est`.
