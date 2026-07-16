# A measurement-error model for METR's time horizon

*Picking up where Alexander Barry's note left off: a fully Bayesian model that
puts the human-time noise inside the likelihood, reusing Jonas Moss's IRT trend.
What I reused, and the one place I think our models genuinely disagree.*

Barry, you ended your note on modelling assumptions hoping to explore "Bayesian
modelling to directly account for the noise in task length estimates." This is
an attempt at exactly that, so much of it is downstream of your calibration
numbers and Moss's model. The most interesting part is the one result where my
model and your SIMEX analysis point in different directions, so I've given it
the most space (§3).

Where it lands:

- **Doubling time**: ~3.3 months on the plain linear trend, tightening to
  **2.4 [2.1, 2.7]** under the fuller model.
- **Residual difficulty at fixed length**: about **8×**, of which roughly
  **two-thirds is predictable task-family structure**, leaving a within-family
  residual near 5× — right on Moss's ~4.7×.
- **Your 25–40% frontier-horizon reduction**: my model shows only **~10%**. I
  think the gap is informative rather than a disagreement — the per-task
  difficulty residual absorbs most (not all) of the length noise, a
  falsification test (§3) says that absorption is legitimate, and the ~10% that
  leaks through is real, not zero.

---

## 1. Two predecessors, doing two different things

From **Moss** I took the IRT reframing of METR's 50%-horizon question: a 2PL
with difficulty on the log human-minutes scale, so ability `θ_m` reads as a
log-horizon, plus a trend on ability over release dates. Also, importantly, his
**unexplained-difficulty** term — the ~4.7×-per-SD spread that length doesn't
explain. I'll come back to how much that term carries.

From **Barry** I took the problem statement and the calibration. Your note
models a baseliner attempt as the true task length times lognormal noise,
estimated from multi-attempt tasks: 80% of attempts within ~3× (σ ≈ 0.78),
shrinking as σ²/n over *n* attempts; for estimate-only tasks, 80% within ~4×
(σ ≈ 1.05). Those two numbers are the entire empirical basis for how much a
model should trust a timing datum, and I use them directly.

What neither of you did is the thing your note deferred as "infeasible for this
post": put that noise *inside* a single joint likelihood rather than correcting
for it afterward. Your SIMEX and task-length capping are sensible sensitivity
tools layered on a frequentist fit. The Bayesian version makes the latent
length a parameter and lets the timing data and the success data inform it at
once. So this is the missing corner of your analysis, not a competitor to it.

## 2. The model, briefly

Three layers over shared per-task latents; you both know the ingredients, so
I'll keep it short.

The **measurement layer** is your model: latent `log(L_i)`, timed runs
`Normal(log L_i, σ_base)`, estimate annotations `Normal(log L_i, σ_est)` with
`σ_est` from your 4×-at-80% figure (prior median around 1.25, a shade wider
than your 1.05; not enough to matter). Fit freely under a Normal likelihood,
`σ_base` comes out 0.79 — right on your 0.78. Under a Student-t it splits into
a tighter core (0.41) plus heavy tails (dof ≈ 2.4), which I think is the better
read of wall-clock data full of mid-task breaks.

The **IRT layer** is Moss's 2PL, with difficulty `= log(L_i) + ε_i`; `ε` is his
unexplained difficulty as an explicit per-task effect. One mechanical change is
identification: a free `ε` on a latent `log L` creates a shift ridge (only
`θ_m − (log L_i + ε_i)` is identified). Moss breaks it by anchoring two
abilities; I can't anchor `θ` without destroying the log-minute reading of the
horizon, so I constrain `ε` to sum to zero instead.

The **trend layer** is Moss's, plus a per-model random effect so the
doubling-time interval reflects model-to-model scatter; four shapes, stacked.

## 3. Where I land differently from you

Your SIMEX analysis is the result I spent the most time on, because at first my
model flatly contradicts it. Side by side:

| quantity | your SIMEX (on METR's model) | this model |
|---|---|---|
| baseliner noise σ | 0.78 (80% within 3×) | 0.79 fitted (Normal); 0.41 core + heavy tails (t) |
| estimate noise σ | 1.05 (80% within 4×) | 1.25 prior median |
| frontier 50% horizon, noise removed | **−25% to −40%** (Opus 4.6 12h → ~6h49m–7h38m) | **−11%** (SIMEX λ=−1 extrapolation) |
| 80% horizon under noise-removal | **+9% to +23%** | rises (marginal curve flattens, §4) |
| doubling time under noise | (not the focus) | flat, 3.23–3.30 mo across λ ∈ [0, 2] |
| your stated uncertainty | "cannot rule out 0% to 60%" | — |

**The difference is real.** I ran your SIMEX ladder directly on my model
(`scripts/simex.py`: add √λ·σ noise to the timing, refit, extrapolate to
λ=−1) to rule out a definitional artifact. The slope is essentially zero;
that's the −11%.

**The gap is `ε`.** The frontier horizon `exp(θ_m)` lives on the **difficulty**
scale, `log L + ε`, and that *sum* is what the cross-model success data pin —
not `log L` on its own. METR's model has no per-task `ε`, so difficulty *is*
length: shrinking a noisy over-long task drops its difficulty and pulls the
horizon down, and your 25–40% is exactly right for that model. Add a per-task
difficulty residual — which the data strongly want (`σ_ε ≈ 2.2`) — and length
noise merely moves the split between `log L` and `ε`, leaving their sum, and so
the horizon, almost untouched. The −11% that remains is the modest hierarchical
shrinkage of `log L` toward the population mean.

**Trying to falsify it.** The worry is that `ε` is absorbing length bias it
shouldn't. If a long-human, model-easy task were really a mis-timed short task,
`ε` should run systematically *negative* on the long, poorly-timed tasks. It
runs the other way (`scripts/fork_discriminator.py`): on well-timed tasks,
longer tasks are mildly *harder* for their length (`ε`–length slope +0.09), and
the poorly-timed long tasks sit *above* that trend (`ε` ≈ +1.5, excess +1.9,
t ≈ 6.9). The model reads them as genuinely hard — they're the 8h RE-Bench-style
tasks, and they are. `ε` is tracking real difficulty, not soaking up length
bias.

**Why the ~10% isn't zero.** When I first wrote this up I got the reason wrong.
My first guess — that the long estimate-only tasks (your 30h ones) have thin
success data, so their difficulty is barely pinned — is false: those are the
RE-Bench / AI-R&D tasks that *every* model attempted (~20 models, ~90 attempts
each). What's actually going on: for those tasks `log L` and `ε` trade off in
the posterior with correlation about −0.66 — strong, but not −1. A perturbation
to the annotation is *mostly* absorbed by `ε`, but about a third leaks through
into the difficulty; that partial absorption is the ~10%. And those tasks'
difficulty is itself only moderately pinned (posterior sd ≈ 0.85 log-min,
~2.3×) — not because attempts are few, but because identifying one task's
difficulty against uncertain abilities and discriminations has its own floor.
So the horizon is not *immune* to length noise, only attenuated — from your
~30% to about a third of that — and the residue is real.

In short: your 25–40% is the correct answer to "what does measurement noise do
to a difficulty-equals-length horizon." Once difficulty is separately
identified, the answer is ~10%, because the horizon rides on difficulty, which
the success data fix, more than on the length of the longest tasks, which the
noise corrupts.

## 4. The additions, and how they connect back to your findings

**Marginal vs conditional horizon.** Your note has the 50% horizon dropping
under noise-correction while the 80% horizon *rises*. In my model `exp(θ_m)` is
conditional on a median-difficulty task, and the split falls straight out: the
50% level and the trend slope are protected (by symmetry the marginal curve
still crosses 0.5 at `θ_m`, and a stationary `ε` cancels from the slope), but
the population curve is ~1.9× flatter than the single-task one, so the 80%
horizon is pulled out and away. Same mechanism, arrived at from the difficulty
side rather than the noise side.

**Heteroscedastic noise.** Your σ ≈ 0.78 is a single number; in the data the
within-task spread grows with length (short ~0.28, long ~0.57). Letting
`σ_base` scale with length is decisive (positive coefficient, no mass below
zero) and improves the duration fit, but leaves the headline alone.

**Survivorship.** Only successful human runs anchor length, but 129 failed
timed runs sit on hard tasks with median wall-times in the hours. Added as
right-censored observations, they lift those tasks — the *opposite* direction
to your long-task shrinkage. Censoring says some long tasks are under-timed;
extreme-value inflation says they're over-timed; both are probably happening,
on different tasks. I keep censoring as a sensitivity because it shifts the
estimand off METR's successful-completion definition.

**Structuring the difficulty.** The addition I'd defend hardest. Splitting `ε`
into a task-family effect plus a within-family residual shows about
**two-thirds of the difficulty variance is between-family** — predictable
structure, not noise — with pattern-continuation and cryptanalysis families
~100× harder than their length implies and arithmetic/file-selection ~100×
easier. Two consequences. The within-family residual is ~5× (σ ≈ 1.6), landing
on Moss's ~4.7× — so my inflated 8× was mostly loose family structure his
residual was implicitly pooling. And because `ε` is the part of difficulty the
trend reads against, this is the one refinement that moves the headline,
tightening the doubling time to 2.4.

## 5. Where it lands

The doubling time holds a ~2.4–3.3 month band through every change, so the
*trend* is robust to how the timing is modelled.

On the horizon *level*, running your SIMEX ladder on the family-structured
model gives −8.7%, versus −10.9% under the flat model, with the doubling time
flat at 2.9–3.0 across the ladder. Better difficulty identification does nudge
the residual down, in the direction the mechanism predicts, but only a little:
family pooling can only help the long tasks with well-estimated family-mates —
about a quarter of them. So the noise-correction lives in a ~9–11% band across
two model specifications — an order of magnitude below the
difficulty-equals-length figure, but not zero.

The remaining crux is narrower than where we started, and it moved on me. It is
*not* "are the long tasks under-attempted" — they aren't. It's that identifying
a single task's difficulty from the success pattern has a floor (≈2.3× here)
even with 90 attempts, and some of your length-noise effect survives as that
~10%. That residual is the real finding, and it's where I'd most value your
read: is the floor I'm hitting the model's, or the data's?
