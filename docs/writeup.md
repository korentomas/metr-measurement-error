# A measurement-error model for METR's time horizon

*Building on Jonas Moss's IRT reanalysis and Alexander Barry's measurement-error
model: what I reused, what I did differently, and what the additions found.*

Most of this will be familiar to the two of you, because most of it is yours.
Moss supplied the IRT-on-human-time skeleton and the trend; Barry, independently,
had already built a measurement-error model of the baseliner times that is close
to the one I use. What I did was join those two threads into a single joint
model, pick a different identification fix, and then chase a handful of questions
the running model raised about itself. The point of writing it up in order is so
you can see exactly which piece answers which question, and tell me where I've
talked myself into something.

Where it lands, up front: current doubling time **~3.3 months** under the plain
linear trend, tightening to **2.4 [2.1, 2.7]** under the recommended
configuration; residual task-difficulty spread of about **8×** at fixed length,
of which — this is the part I'd most like your read on — roughly **two-thirds is
predictable task-family structure** rather than irreducible noise, leaving a
within-family residual of about 5× that sits right on Moss's ~4.7×.

---

## 1. The two starting points

From Moss I took the reframing of METR's 50%-horizon question as a 2PL IRT
model, with difficulty anchored on the log human-minutes scale so that ability
`θ_m` reads directly as a log-horizon, and a trend on ability across release
dates. I also took, though I'll come back to it, his **unexplained-difficulty**
term — the ~4.7×-per-SD spread of difficulty that log-length doesn't account
for. That term matters more than it first looks.

The one modelling choice I didn't keep is the one Barry had already flagged and
fixed in his own way: Moss treats each task's `human_minutes` as a fixed,
exactly-known scalar. It isn't. For most tasks it's a geometric mean of one or
two baseline runs, within-task wall-times routinely differ 1.5–2×, and a chunk
of tasks have no timed run at all — only an expert estimate. So `human_minutes`
is a noisy, sparsely-observed measurement of a latent length, and because
difficulty is built on it, that noise is being pushed straight into difficulty
and the trend while being hidden from the posterior.

Barry's model already treats it correctly: baseliner times as the true task
length times lognormally-distributed noise, with the noise level estimated from
the tasks that have several attempts. That's essentially the measurement layer
below. Where we diverged, as I understand your comments, is that you handled
baselined and estimate-only tasks separately and defined the p-horizon over
baselined tasks only, whereas I keep them in one model with a wider noise prior
on the estimates — which is where your second contribution comes in.

## 2. The two things I took from Barry directly

The **60%-within-3× number.** Your finding that only ~60% of the estimate-only
annotations land within a factor of three of the real baseline time is the only
thing pinning how much the model should distrust an estimate — under a log-scale
error it implies a noise SD around 1.3 log-minutes, and since no task in my set
carries both an annotation and a timed run, I can't recover it internally. It
goes in as a prior. Without it the estimate-only tasks (which skew long, up to
30h) are free to wander.

The **circularity warning.** Your sharper point — that once length and
difficulty are both latent and both fed by the success data, the outcomes can
push a task's inferred *length* around, so the trend ends up partly fit to
lengths the outcomes themselves chose — is the kind of thing that stays
invisible until named. It turned into an explicit diagnostic and a cut-model
check later, and you were right that it's real; the question was only whether it
was large.

## 3. The model

Three layers over a shared set of per-task latents.

The **measurement layer** is Barry's: each task gets a latent `log(L_i)`, timed
runs are `log(dur) ~ Normal(log L_i, σ_base)`, estimate annotations are
`Normal(log L_i, σ_est)` with `σ_est` set by the 60%-within-3× prior.

The **IRT layer** is Moss's 2PL, `logit⁻¹(a_i·(θ_m − difficulty_i))`, with

> `difficulty_i = log(L_i) + ε_i`

and `ε_i` the unexplained-difficulty term. I want to be careful here: `ε` is not
my invention — it's Moss's ~4.7× spread, just written as an explicit per-task
effect rather than a regression residual. What *is* a real change is how I
identify it. A free `ε` added to a latent `log L` creates a shift
non-identifiability — only `θ_m − (log L_i + ε_i)` enters the likelihood, so a
constant added to every `ε` and subtracted from every `θ` is invisible, and the
sampler drifts along that ridge. Moss breaks it by anchoring two ability values;
I couldn't, because anchoring `θ` would break the log-minute reading of the
horizon that the timing data works to establish. So I constrain the `ε` to sum
to zero instead. Same problem, a fix chosen to protect the property I care
about.

The **trend layer** is `θ_m = f(t_m) + u_m`: Moss's trend plus a per-model
random effect `u_m`, so the doubling-time interval reflects model-to-model
scatter and not just within-model noise. `f` is one of four shapes combined by
stacking rather than chosen. (And the usual non-centered parameterization
throughout, without which the 1–2-observation-per-task geometry funnels and the
sampler stalls.)

So: Barry's measurement layer + Moss's IRT-and-trend, unified across baselined
and estimate tasks, with a sum-to-zero identification and per-model random
effects as the structural differences.

## 4. What it said, and where it agreed with you both

The doubling time comes out around 3.3 months on the plain linear trend, in the
same neighbourhood as Moss's 4.3 and METR's numbers once the trend shapes and
time windows are lined up; my linear-vs-Moss gap is the measurement layer and
the random effects, not the data. Barry's ~2× -over-METR at 80% and the whole
"80% horizons are an order of magnitude off" point both live in the same place
mine does — the difficulty spread — which is the next section.

The circularity you warned about is present and in the predicted direction
(tasks models solve pulled shorter, ones they fail pulled longer), but a
cut-model refit that severs the loop moves the headline by about a day. Real,
measured, small.

Two stress tests worth reporting. Moss's suggestion to try a Weibull duration
likelihood: I did, and it fits worse — the within-task residuals are
right-skewed with heavy tails and a Weibull's log is fixed left-skewed, so it's
the wrong shape however you set it. A Student-t, by contrast, earns its place;
the fitted dof lands near 2.4, so the wall-clock tails are real and heavy.

## 5. The additions, and the one number I'd flag

Once it ran, each thing I looked at hard pointed to the next.

**Which horizon is it, marginally or conditionally?** This is your and Moss's
"marginal vs typical 80%" point, and I wanted to know how much it bit for mine.
`exp(θ_m)` is a horizon conditional on a median-difficulty task (`ε = 0`);
METR's is marginal over the task population. The reassuring part: the 50% level
and the doubling-time slope are safe — by symmetry the marginal success curve
still crosses 0.5 at `θ_m`, and a stationary `ε` cancels from the slope. The
unreassuring part, and the reason your 80% number moves so much: the marginal
curve is about 1.9× flatter, so every non-50% horizon and every threshold
extrapolation diverges. So the 50% horizon and doubling time transfer directly;
anything at 80% has to be computed marginally, exactly as you both argued.

**Is the measurement layer even doing work?** Honestly, not much to the point
estimate — because the IRT layer only sees `log L + ε` and `ε` is free, error in
`log L` is absorbed by `ε` as far as the trend cares. What it buys is honest
uncertainty: a task timed by a single human run has a real ~1.8× posterior
spread in its length that a plug-in treatment throws away. The layer widens the
intervals for the sparsely-timed tasks that deserve it, and leaves the trend
alone.

**The measurement noise isn't constant.** The within-task spread of log
wall-time grows with length (short ~0.28, long ~0.57), so I let `σ_base` scale
with length. The data are decisive — the coefficient is positive with
essentially no mass below zero — and it improves the duration fit cleanly
without touching the headline.

**Survivorship.** Only successful human runs anchor length, but there are 129
failed timed runs on hard tasks with median wall-times of a couple of hours
against a few minutes for successes. Adding them back as right-censored
observations lifts the hard tasks' lengths where it should; I keep it as a
sensitivity because it does shift the estimand off METR's successful-completion
definition.

**The one I'd flag.** `ε` at ~8× (σ ≈ 2.2) is bigger than Moss's ~4.7×, and I
wanted to know why, so I split `ε` into a task-family effect and a within-family
residual. About **two-thirds of the difficulty variance is between-family** —
predictable structure, not noise — and the family effects are exactly what you'd
guess: pattern-continuation and cryptanalysis families run ~100× harder than
their length implies, arithmetic and file-selection ~100× easier. The success
data clearly prefer the structured version. Two things follow. First, the
genuinely irreducible within-family residual is about **5× (σ ≈ 1.6)** — which
lands right on Moss's ~4.7×, so I think the gap between our difficulty numbers
was mostly that his residual pools structure mine was leaving loose. Second,
because `ε` is the part of difficulty the trend actually reads against,
structuring it is the *one* refinement that moves the headline — the current
doubling time tightens to 2.4 [2.1, 2.7] months.

That's the through-line. Refinements to the *length* channel wash out, because
`ε` buffers them; the refinement to the *difficulty* channel is what moves the
trend. Which is really the model saying its signal lives in the term all three
of us have some version of — Moss's unexplained difficulty, and here its
family structure.

## 6. Where it lands

The doubling time stays in a ~2.4–3.3 month band through every change, so the
trend itself is robust to how I model the timing. The interesting part is *when*
it moves: not when I refine the human-time measurement, but when I refine what
"difficulty" means — and when I do, my headline difficulty number reconciles
with Moss's. If there's an error in the chain I'd bet it's in the family
decomposition or the marginal-horizon symmetry argument, and those are the two
places I'd most value a second pair of eyes.
