# A measurement-error model for METR's time horizon

*Picking up where Alexander Barry's note left off — a fully Bayesian model that
puts the human-time noise inside the likelihood — and reusing Jonas Moss's IRT
trend to do it. What I reused, where I land differently from Barry, and the one
place I think our models genuinely disagree.*

Barry, you ended your note on modelling assumptions by saying you hoped to
explore "Bayesian modelling to directly account for the noise in task length
estimates" in future. This is an attempt at exactly that, so a lot of it is
downstream of your calibration numbers and Moss's model rather than new. I've
written it in the order the pieces came together, and I've tried to be honest
about the one result where I think my model and your SIMEX analysis point in
different directions — because that disagreement is more interesting than the
agreements.

Where it lands, up front and hedged the way this whole topic deserves: current
doubling time **~3.3 months** on the plain linear trend, tightening to **2.4
[2.1, 2.7]** under the fuller model; residual difficulty at fixed length of
about **8×**, of which roughly **two-thirds is predictable task-family
structure**, leaving a within-family residual near 5× that sits on Moss's
~4.7×. But — and this is the part I'm least sure of — my model does *not*
reproduce your 25–40% frontier-horizon reduction, and I don't think that's
because either of us is obviously right.

---

## 1. Two predecessors, doing two different things

From **Moss** I took the IRT reframing of METR's 50%-horizon question: a 2PL
with difficulty on the log human-minutes scale, so ability `θ_m` reads as a
log-horizon, plus a trend on ability over release dates. I also, importantly,
took his **unexplained-difficulty** term — the ~4.7×-per-SD spread that length
doesn't explain. I'll come back to how much that term carries.

From **Barry** I took the whole problem statement and the calibration. Your
note models an individual baseliner attempt as the true task length times
lognormal noise, with the noise estimated from multi-attempt tasks — 80% of
attempts within ~3× (σ ≈ 0.78), the geomean over *n* attempts shrinking it as
σ²/n. And for the estimate-only tasks, 80% of estimates within ~4× (σ ≈ 1.05).
Those two numbers are the entire empirical basis for how much a model should
trust a timing datum, and I use them directly.

The one thing neither of you did is the thing your note explicitly deferred as
"infeasible for this post": put that noise *inside* a single joint likelihood
rather than correcting for it afterward. Your primary tools were SIMEX — add
noise, refit, extrapolate back to zero — and task-length capping, both layered
on a frequentist fit. That's a sensible way to get a sensitivity band without
rebuilding the model. The Bayesian version just makes the latent length a
parameter and lets the timing data and the success data inform it at once. So
this isn't a competing analysis so much as the missing corner of yours.

## 2. The model, briefly

Three layers over shared per-task latents, and since you both know the
ingredients I'll keep it short.

The **measurement layer** is your model: latent `log(L_i)`, timed runs
`Normal(log L_i, σ_base)`, estimate annotations `Normal(log L_i, σ_est)` with
`σ_est` from your 4×-at-80% figure (I used a prior median around 1.25, a shade
wider than your 1.05; not enough to matter). Reassuringly, when I fit `σ_base`
freely under a Normal likelihood it comes out 0.79 — right on your 0.78. Under
a Student-t it splits into a tighter core (0.41) plus genuinely heavy tails
(dof ≈ 2.4), which I think is the more honest read of wall-clock data full of
mid-task breaks.

The **IRT layer** is Moss's 2PL, with difficulty `= log(L_i) + ε_i`. `ε` is his
unexplained difficulty, written as an explicit per-task effect. The one real
mechanical change is identification: a free `ε` on a latent `log L` creates a
shift ridge (only `θ_m − (log L_i + ε_i)` is identified), which Moss breaks by
anchoring two abilities. I can't anchor `θ` without destroying the log-minute
reading of the horizon, so I constrain `ε` to sum to zero instead.

The **trend layer** is Moss's, plus a per-model random effect so the
doubling-time interval reflects model-to-model scatter; four shapes, stacked.

## 3. Where I land differently from you — and why it matters

Here's the result I most want you to push on. Your SIMEX analysis finds that
removing the measurement noise **cuts the frontier 50% horizon by 25–40%**
(Opus 4.6, 12h → ~7h40m), driven by your point that the longest tasks are
overestimates — extreme values in noisy data are exaggerated, so shrinking them
pulls the top of the curve down. My joint model barely does this. The longest
tasks shrink only ~10% (top-decile posterior means ~0.94× their raw
annotations; the correlation between shift and length is −0.09), and the
doubling time is essentially unmoved by anything in the measurement layer.

I don't think this is a bug in either of us — I think it's a real
identification fork, and it's `ε`. When models succeed on a task that a human
took a long time on, there are two explanations: the task's *length* was
overestimated (your reading — shrink `L`), or the task is genuinely long but
*easy for its length* (a negative `ε`). Your correction only has the first
lever, so all of that signal becomes length-shrinkage and the horizon drops.
My model has both, and the success data mostly load it onto `ε`, so `L` stays
near the annotation and the horizon doesn't drop. The measurement layer ends up
buying honest *uncertainty* on the sparsely-timed tasks (a single-run task
keeps a real ~1.8× spread a plug-in would discard) without buying your
*downward correction*.

Which of us is right is not something my fit can settle, and it's exactly the
place a wide `σ_L` and a free `ε` could be letting my model under-shrink the
inflated long tasks you flagged. If I had to guess, the truth is in between:
some of those long tasks really are overestimates and my `ε` is absorbing bias
it shouldn't. That feels like the highest-value thing to nail down, and it's
downstream of a question neither approach answers — whether a long, human-slow,
model-easy task is short or just easy.

## 4. The additions, and how they connect back to your findings

**Marginal vs conditional horizon.** Your note has the 50% horizon dropping
under noise-correction while the 80% horizon *rises*; that split is the same
object as the marginal-vs-typical point. In my model `exp(θ_m)` is conditional
on a median-difficulty task, and the reason the two reliabilities move in
opposite directions falls straight out: the 50% level and the trend slope are
protected (by symmetry the marginal curve still crosses 0.5 at `θ_m`, and a
stationary `ε` cancels from the slope), but the population curve is ~1.9×
flatter than the single-task one, so the 80% horizon is pulled out and away.
Same mechanism, arrived at from the difficulty side rather than the noise side.

**Heteroscedastic noise.** Your σ ≈ 0.78 is a single number; in the data the
within-task spread grows with length (short ~0.28, long ~0.57). Letting `σ_base`
scale with length is decisive (positive coefficient, no mass below zero) and
improves the duration fit, though it leaves the headline alone.

**Survivorship.** Only successful human runs anchor length, but 129 failed
timed runs sit on hard tasks with median wall-times in the hours. Adding them as
right-censored observations lifts those tasks — an effect in the *opposite*
direction to your long-task shrinkage, which is worth noting: failed-run
censoring says some long tasks are if anything *under*-timed, extreme-value
inflation says they're over-timed, and both are probably happening on different
tasks. I keep censoring as a sensitivity because it shifts the estimand off
METR's successful-completion definition.

**Structuring the difficulty.** The one addition I'd defend hardest. Splitting
`ε` into a task-family effect and a within-family residual shows about
**two-thirds of the difficulty variance is between-family** — predictable
structure, not noise — with pattern-continuation and cryptanalysis families
~100× harder than their length implies and arithmetic/file-selection ~100×
easier. Two consequences. The irreducible within-family residual is ~5× (σ ≈
1.6), which lands on Moss's ~4.7× — so I think my inflated 8× was mostly loose
family structure his residual was implicitly pooling. And because `ε` is the
part of difficulty the trend reads against, structuring it is the *one*
refinement that moves the headline, tightening the doubling time to 2.4.

## 5. Where it lands

The doubling time holds a ~2.4–3.3 month band through every change, so the
*trend* is robust to how the timing is modelled. But your note is mostly about
the horizon *level* at the frontier, and there my model and your SIMEX
disagree — not by a little, and for a reason (`ε` vs length-shrinkage) that I
can't resolve from inside my own fit. Given how hedged you were about the noise
impact — you couldn't rule out anywhere from 0% to 60% — I don't want to claim
the Bayesian version settles it. If anything it reframes the uncertainty as a
concrete identification question: is a human-slow, model-easy task short, or
just easy? That's the number I'd most like to pin down next, and the place your
SIMEX and this model are the two ends of the same rope.
