# A measurement-error model for METR's time horizon

*This writeup starts where Alexander Barry's note stopped. It is a fully
Bayesian model that puts the human-time noise inside the likelihood, and it
reuses Jonas Moss's IRT trend. It tells what I reused, and it tells the one
place where I think our models genuinely do not agree.*

Barry, at the end of your note on modeling assumptions, you said that you hoped
to explore "Bayesian modelling to directly account for the noise in task length
estimates." This model is an attempt at exactly that. Thus, much of it is
downstream of your calibration numbers and of Moss's model. The most
interesting part is the one result where my model and your SIMEX analysis do
not agree. I give that result the most space (section 3).

Where it lands:

- **Doubling time**: approximately 3.3 months on the plain linear trend. It
  tightens to **2.4 [2.1, 2.7] months** under the fuller model.
- **Residual difficulty at fixed length**: approximately **8x**. Roughly **two
  thirds of it is predictable task-family structure**. That leaves a
  within-family residual near 5x, which agrees with Moss's approximately 4.7x.
- **Your 25% to 40% frontier-horizon decrease**: my model shows only
  approximately **10%**. I think that this gap is informative, not a
  disagreement. The per-task difficulty residual absorbs most (not all) of the
  length noise. A falsification test (section 3) says that this absorption is
  legitimate. And the approximately 10% that leaks through is real, not zero.

---

## 1. Two predecessors, two different things

From **Moss**, I took the IRT reframing of METR's 50%-horizon question: a 2PL
with difficulty on the log human-minutes scale. On that scale, the ability
`θ_m` reads as a log-horizon. I also took his trend on ability over release
dates. And, importantly, I took his **unexplained-difficulty** term: the
approximately 4.7x-per-SD spread that length does not explain. I will come back
to how much that term carries.

From **Barry**, I took the problem statement and the calibration. Your note
models a baseliner attempt as the true task length, multiplied by lognormal
noise that you estimated from the multi-attempt tasks. You found 80% of
attempts within approximately 3x (σ ≈ 0.78), and the noise shrinks as σ²/n
over *n* attempts. For the estimate-only tasks, you found 80% within
approximately 4x (σ ≈ 1.05). Those two numbers are the entire empirical basis
for how much a model can trust a timing datum. I use them directly.

Neither of you did the thing that your note deferred as "infeasible for this
post": to put that noise *inside* a single joint likelihood, instead of a
correction after the fit. Your SIMEX and your task-length cap are sensible
sensitivity tools on top of a frequentist fit. The Bayesian version makes the
latent length a parameter. The timing data and the success data then inform it
at the same time. Thus, this model is the missing corner of your analysis, not
a competitor to it.

## 2. The model, briefly

The model has three layers over shared per-task latents. You both know the
ingredients, so I keep this section short.

The **measurement layer** is your model. The latent is `log(L_i)`. The timed
runs are `Normal(log L_i, σ_base)`. The estimate annotations are
`Normal(log L_i, σ_est)`, with `σ_est` from your 4x-at-80% figure (prior median
near 1.25, a shade wider than your 1.05, but not enough to matter). A free fit
under a Normal likelihood gives `σ_base` = 0.79, which agrees with your 0.78.
Under a Student-t, it splits into a tighter core (0.41) plus heavy tails
(dof ≈ 2.4). I think that the Student-t is the better read of wall-clock data
that is full of mid-task breaks.

The **IRT layer** is Moss's 2PL, with difficulty `= log(L_i) + ε_i`. Here `ε`
is his unexplained difficulty as an explicit per-task effect. One mechanical
change is identification. A free `ε` on a latent `log L` makes a shift ridge,
because only `θ_m − (log L_i + ε_i)` is identified. Moss breaks the ridge with
anchors on two abilities. I cannot anchor `θ` without damage to the log-minute
reading of the horizon. Thus, I constrain `ε` to sum to zero instead.

The **trend layer** is Moss's, plus a per-model random effect. The random
effect lets the doubling-time interval show the model-to-model scatter. There
are four trend shapes, combined with Bayesian stacking.

## 3. Where I land differently from you

I spent the most time on your SIMEX analysis, because at first my model flatly
contradicts it. The comparison, side by side:

| quantity | your SIMEX (on METR's model) | this model |
|---|---|---|
| baseliner noise σ | 0.78 (80% within 3x) | 0.79 fitted (Normal), 0.41 core + heavy tails (t) |
| estimate noise σ | 1.05 (80% within 4x) | 1.25 prior median |
| frontier 50% horizon, noise removed | **−25% to −40%** (Opus 4.6 12 h → approximately 6 h 49 min to 7 h 38 min) | **−11%** (SIMEX λ = −1 extrapolation) |
| 80% horizon under noise removal | **+9% to +23%** | rises (the marginal curve flattens, section 4) |
| doubling time under noise | (not the focus) | flat, 3.23 to 3.30 months across λ ∈ [0, 2] |
| your stated uncertainty | "cannot rule out 0% to 60%" | — |

**The difference is real.** To make sure that the difference is not a
definitional artifact, I ran your SIMEX ladder directly on my model
(`scripts/simex.py`: add √λ·σ noise to the timing, refit, extrapolate to
λ = −1). The slope is almost zero. That result is the −11%.

**The gap is `ε`.** The frontier horizon `exp(θ_m)` lives on the **difficulty**
scale, `log L + ε`. The cross-model success data pin that *sum*, not `log L`
on its own. METR's model has no per-task `ε`, so difficulty *is* length in
that model. When noise correction shrinks a noisy over-long task, its
difficulty drops, and the horizon comes down. Your 25% to 40% is exactly right
for that model. Now add a per-task difficulty residual, which the data
strongly want (σ_ε ≈ 2.2). Length noise then only moves the split between
`log L` and `ε`. Their sum, and thus the horizon, is almost untouched. The
−11% that remains is the modest hierarchical shrinkage of `log L` toward the
population mean.

**An attempt to falsify it.** The worry: `ε` can absorb length bias that it
must not absorb. If a long-human, model-easy task is really a mis-timed short
task, then `ε` must be systematically *negative* on the long, poorly timed
tasks. It goes the other way (`scripts/fork_discriminator.py`). On the
well-timed tasks, longer tasks are mildly *harder* for their length
(`ε`–length slope +0.09). And the poorly timed long tasks sit *above* that
trend (`ε` ≈ +1.5, excess +1.9, t ≈ 6.9). The model reads them as genuinely
hard. They are the 8 h RE-Bench-style tasks, and they are genuinely hard. `ε`
tracks real difficulty. It does not absorb length bias.

**Why the approximately 10% is not zero.** In my first version of this
writeup, the reason I gave was wrong. My first guess was that the long
estimate-only tasks (your 30 h ones) have thin success data, so their
difficulty is barely pinned. That guess is false. Those tasks are the RE-Bench
and AI-R&D tasks that *every* model attempted (approximately 20 models, with
approximately 90 attempts for each task). What really occurs: for those tasks,
the posterior correlation between `log L` and `ε` is approximately −0.66.
That correlation is strong, but it is not −1. `ε` absorbs *most* of a
perturbation to the annotation, but approximately a third leaks through into
the difficulty. That partial absorption is the approximately 10%. And the
difficulty of those tasks is itself only moderately pinned (posterior
sd ≈ 0.85 log-minutes, approximately 2.3x). The cause is not a small number
of attempts. The cause: to identify one task's difficulty against uncertain
abilities and uncertain discriminations has its own floor. Thus, the horizon
is not *immune* to length noise. The effect is only attenuated, from your
approximately 30% to approximately a third of that. And the residue is real.

In short: your 25% to 40% is the correct answer to the question "what does
measurement noise do to a difficulty-equals-length horizon." When difficulty
is separately identified, the answer is approximately 10%. The horizon rides
on difficulty, which the success data fix. It rides much less on the length of
the longest tasks, which the noise corrupts.

## 4. The additions, and how they connect to your findings

**Marginal and conditional horizons.** In your note, the 50% horizon drops
under noise correction, while the 80% horizon *rises*. In my model,
`exp(θ_m)` is conditional on a median-difficulty task, and the same split
appears directly. The 50% level and the trend slope are protected. By
symmetry, the marginal curve still crosses 0.5 at `θ_m`, and a stationary `ε`
cancels from the slope. But the population curve is approximately 1.9x flatter
than the single-task curve. Thus, the 80% horizon moves out and away. It is
the same mechanism, but found from the difficulty side, not from the noise
side.

**Heteroscedastic noise.** Your σ ≈ 0.78 is a single number. In the data, the
within-task spread grows with length (short approximately 0.28, long
approximately 0.57). A `σ_base` that scales with length is decisive (positive
coefficient, no mass below zero), and it improves the duration fit. But it
does not change the headline.

**Survivorship.** Only successful human runs anchor length. But 129 failed
timed runs sit on hard tasks, with median wall-times of hours. When I add
these runs as right-censored observations, they lift those tasks. That is the
*opposite* direction to your long-task shrinkage. The censored runs say that
some long tasks are under-timed. Your extreme-value inflation says that they
are over-timed. Probably both occur, on different tasks. I keep the censored
variant as a sensitivity, because it moves the estimand away from METR's
successful-completion definition.

**Structure in the difficulty.** This is the addition that I defend the
hardest. I split `ε` into a task-family effect plus a within-family residual.
The split shows that approximately **two thirds of the difficulty variance is
between-family**. That part is predictable structure, not noise. The
pattern-continuation and cryptanalysis families are approximately 100x harder
than their length implies. The arithmetic and file-selection families are
approximately 100x easier. There are two consequences. First, the
within-family residual is approximately 5x (σ ≈ 1.6), which lands on Moss's
approximately 4.7x. Thus, my inflated 8x was mostly loose family structure
that his residual implicitly pooled. Second, `ε` is the part of difficulty
that the trend reads against. Thus, this is the one refinement that moves the
headline. It tightens the doubling time to 2.4 months.

## 5. Where it lands

The doubling time stays in an approximately 2.4 to 3.3 month band through
every change. Thus, the *trend* is robust to the choice of timing model.

On the horizon *level*: your SIMEX ladder on the family-structured model gives
−8.7%, compared with −10.9% under the flat model. The doubling time stays flat
at 2.9 to 3.0 months across the ladder. Better difficulty identification does
move the residual down, in the direction that the mechanism predicts. But the
move is small. Family pooling can only help the long tasks that have
well-estimated family-mates, and those are approximately a quarter of them.
Thus, the noise correction lives in an approximately 9% to 11% band across the
two model specifications. That band is an order of magnitude below the
difficulty-equals-length figure, but it is not zero.

The remaining crux is narrower than where we started, and it moved on me. The
crux is *not* "are the long tasks under-attempted." They are not. The crux:
to identify a single task's difficulty from the success pattern has a floor
(≈ 2.3x here), even with 90 attempts. Some of your length-noise effect
survives as that approximately 10%. That residual is the real finding. And it
is where I would most value your read: is this floor the model's floor, or the
data's floor?
