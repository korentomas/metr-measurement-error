# A measurement-error model for METR's time horizon

Based on:
- Alexander Barry's [note on modeling assumptions](https://metr.org/notes/2026-03-20-impact-of-modelling-assumptions-on-time-horizon-results/)
- Jonas Moss's [IRT reanalysis](https://www.lesswrong.com/posts/sBEzomgnYJmYHki9T).

Results:

- Doubling time: ~3.3 months on the plain linear trend, dropping to
  2.4 months (95% credible interval: 2.1–2.7) with all the refinements
  included.
- Tasks of the same human length differ ~8x in how hard they are for
  models. Roughly two thirds of that (as variance, in log space) is
  predictable task-family structure,
  leaving a within-family spread near 5x, which agrees with Moss's ~4.7x.
- Barry found that removing the timing noise cuts the frontier model's 50%
  horizon by 25–40%. My model shows only ~10% (more on that in section 2).

## 1. The model

The timing layer is Barry's noise model with the true length hidden: each
task has an unknown true length where timed runs scatter around it and
estimate annotations scatter *more*. How much more comes from Barry's
measurement: 80% of expert estimates fall within ~4x of the true length,
which I use as the prior on the estimate noise. A free fit recovers a
timing noise of 0.79 (log scale), close to the 0.78 Barry estimated by
hand, this is a number the fit never sees. A Student-t version splits that into a tighter core
(0.41) plus heavy tails, a better description of wall-clock times that
include breaks and interruptions.

The success layer is Moss's IRT setup: how often each AI model succeeds at
each task pins down ability and difficulty, with difficulty = the task's
latent length + a per-task residual `ε` (Moss's unexplained difficulty).
There is a big change here: the success data only see the gap between ability
and difficulty, so a constant could shift freely between `ε` and every
ability score. Moss pins two abilities to stop this. Doing that here would
break the interpretation of ability as a log time horizon, so I force `ε` to sum to
zero across tasks instead.

The trend layer is Moss's ability-over-release-date curve, plus a
per-model offset so the doubling-time interval reflects real
model-to-model scatter. Four curve shapes, averaged by Bayesian stacking.

Below is the plate graph for the model. (Circles are unknowns, shaded circles are
data, and the plates show what repeats per task and per model.) The three
plates at the bottom are the three data sources: success/attempt counts, estimate annotations, and timed runs.

![Model graph](../outputs/figures/model_graph.png)

## 2. The SIMEX comparison

I spent some time on Barry's SIMEX analysis, because at first glance
my model kind of contradicts it. If I removed the timing noise his frontier
50% horizon falls 25–40% and mine fell ~10%. To make sure this gap wasn't the two of us measuring different things, I ran his exact method on my model (in `scripts/simex.py` I add extra noise to the timing, refit, extrapolate back to zero noise) and got an essentially flat line, so the ~10% holds under his own procedure.

So **the gap is `ε`.** The horizon depends on difficulty, and what the success
data constrain is length + `ε` together. METR's model has no `ε`, so in that model difficulty just is length: shrink a noisy over-long task and the horizon comes down with it. Barry's 25–40% is exactly right for that model. Give each task its difficulty residual (fitted σ ≈ 2.2 in log units) and timing noise mostly moves the split between length and `ε`; their sum, and with it the horizon, stays almost unchanged.

This is what that residual looks like in the data, each point is a task, plotted by its length and by how many times harder or easier it is than that length suggests. The vertical spread is the ~8x:

![Residual difficulty vs task length](../outputs/figures/difficulty_residual.png)

**An attempt to falsify it.** The obvious worry is that `ε` absorbs
length bias it should not. If a long-human,
model-easy task is really a mis-timed short task, `ε` should be
systematically negative on the long, poorly timed tasks. The data show
the opposite (`scripts/fork_discriminator.py`): on well-timed tasks, longer
tasks are mildly harder than their length predicts, and the poorly timed long
tasks sit clearly above even that trend. They are the 8-hour
RE-Bench-style tasks, which are genuinely hard, and the model agrees.

![Fork discriminator](../outputs/figures/fork_discriminator.png)

**Why 10%.** The absorption is not perfect, for two
reasons. First, length and `ε` trade off strongly in the fit but not completely. Second, success data can only resolve a
task's difficulty to within ~2.3x, even for the long tasks that every
model attempted ~90 times. Those two gaps let some of the
timing noise through: Barry's 25–40% becomes ~10%.

## 3. The additions, briefly

**Marginal and conditional horizons.** Barry's note has the 50% horizon
dropping under noise correction while the 80% horizon rises. The same
split appears in my model from the difficulty side: the residual is
symmetric around zero so it cancels at the 50% point.
But, the success curve for the whole task population is ~1.9x flatter than
for a single typical task, so the 80% horizon rises.

**Length-dependent noise.** The spread between attempts grows with task
length (more can go wrong in a six-hour attempt than in a six-minute one).
Letting the noise scale with length clearly improves the timing fit. It
doesn't change the doubling time.

**Survivorship.** Only successful human runs anchor task length, but 129
failed timed runs sit on the hard tasks, with median times in the hours.
Adding them as "took at least this long" observations lifts those tasks'
lengths, the opposite of Barry's long-task shrinkage; probably both effects are real, on different tasks. I keep it as METR defines the human baseline only with successful completions.

**Structure in the difficulty.** Splitting `ε` into a task-family effect plus a within-family residual shows ~two thirds of the difficulty spread is between families. Pattern-continuation and cryptanalysis run ~100x harder than their length suggests, arithmetic and file-selection ~100x easier. The within-family spread is ~5x, matching Moss's ~4.7x. And because the trend is fit on difficulty, and `ε` is the biggest part of difficulty, this is the one refinement that moves the doubling time: 2.4 months (2.1–2.7).

![Family effects on difficulty](../outputs/figures/eps_family_decomposition.png)

## 4. Conclusion

![Horizon trend](../outputs/figures/horizon_trend.png)

The doubling time stays between 2.4 and 3.3 months in every version of
the model, so the trend does not depend on how the timing is handled.
Removing the timing noise shifts the horizon by only 9–11%, roughly a
third of Barry's 25–40%. That ~10% remains because success data can only
measure a task's difficulty to within ~2.3x, even on tasks every model
attempted ~90 times, and that uncertainty lets a small part of the
timing noise reach the horizon.
