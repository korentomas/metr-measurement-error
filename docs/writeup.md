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
~4.7×. And on your 25–40% frontier-horizon reduction: my model shows only
~10%, and I think that gap is genuinely informative rather than a
disagreement — it's the per-task difficulty residual absorbing the length
noise, and a falsification test (§3) says that absorption is legitimate for
all but the very longest tasks.

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

## 3. Where I land differently from you — and, I think, why

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

So your model's frontier horizon is strongly noise-sensitive and mine is nearly
flat. I ran your SIMEX ladder directly on my model to be sure it's a real
difference and not a definitional one (`scripts/simex.py`: add √λ·σ noise to the
timing, refit, extrapolate to λ=−1), and it is real — the slope is essentially
zero, giving that −11%.

I think the gap is `ε`, and I think it's resolvable rather than a standoff.
Here's the argument. The frontier horizon `exp(θ_m)` lives on the **difficulty**
scale, `log L + ε`, and that sum is what the cross-model success data pins — not
`log L` on its own. In METR's model there is no per-task `ε`, so difficulty *is*
length, and shrinking a noisy over-long task drops its difficulty and pulls the
horizon down: your 25–40% is exactly right *for that model*. Add a per-task
difficulty residual — Moss's `ε`, which the data very much want (`σ_ε ≈ 2.2`) —
and length noise moves the split between `log L` and `ε` while leaving their sum,
and so the horizon, almost untouched. The −11% that remains is just the modest
hierarchical shrinkage of `log L` toward the population mean; the other ~25
points of your effect are absorbed by `ε`.

The obvious worry is that this is too convenient: maybe `ε` is a sponge soaking
up length bias it shouldn't, and my flat horizon is wrong. So I tried to falsify
it (`scripts/fork_discriminator.py`). The test: on well-timed tasks, where the
length is pinned by data, is `ε` doing something *sensible* with length, or just
absorbing whatever the success data throw at it? If a long-human, model-easy
task is really a mis-timed short task, `ε` should be running systematically
*negative* on the long, poorly-timed tasks — the tell of absorbed
over-estimation. It runs the other way. On well-timed tasks longer tasks are
mildly *harder* for their length (`ε`–length slope +0.09), and the poorly-timed
long tasks sit *above* that trend (`ε` ≈ +1.5, excess +1.9, t ≈ 6.9), i.e. the
model reads them as genuinely hard, which they are — they're the 8h RE-Bench-
style tasks. `ε` is not a length-bias sponge; it's tracking real difficulty.

So my read, hedged appropriately: your 25–40% is the correct answer to "what
does measurement noise do to a difficulty-equals-length horizon," and the honest
answer once difficulty is separately identified is much smaller, ~10%, because
the horizon was never really riding on the length of the longest tasks — it was
riding on their difficulty, which the success data fix directly. The one place I
can't fully back myself is the very longest estimate-only tasks (the 30h ones),
where the IRT signal is thin — few models, few attempts — so `ε` there is the
least trustworthy, and that's the residue where your simpler length-shrinkage
might still be the safer bet. But it's a residue, not the whole 25–40%.

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
*trend* is robust to how the timing is modelled. On the horizon *level* — the
thing your note is really about — I think the picture is now less of a standoff
than it looked. Your 25–40% is the right answer for a difficulty-equals-length
model; put a data-supported per-task difficulty residual in, and the honest
figure is closer to ~10%, because the horizon rides on difficulty (which the
success data fix) rather than on the length of the longest tasks (which the
noise corrupts). The discriminator says the `ε` doing that absorbing is real
and not a length-bias sponge, so I'd now put more weight on the small number
than the large one — with the explicit exception of the 30h estimate-only
tasks, where the IRT signal is too thin for me to trust `ε` over your simpler
shrinkage.

If I had to name the single crux left, it's not "is the noise 0% or 60%" any
more — it's whether the cross-model success pattern on the sparsest, longest
tasks is trustworthy enough to identify their difficulty. That's a narrower and
more answerable question than where we started, and it's probably where a
next pass (more baseliner runs on the long tasks, or a stronger prior tying
their `ε` to family structure) would actually move the number.
