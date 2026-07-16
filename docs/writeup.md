# A measurement-error model for METR's time horizon

*What I took from Jonas Moss and Alexander Barry, how I turned it into a
Bayesian measurement-error model, what that model said, and the additions I
tried once it was running.*

This is written to build the idea up in the order it actually came together,
so that anyone who has read Moss's reanalysis and Barry's comments can see
exactly which pieces are theirs, which are mine, and why each addition exists.
Nothing here is a criticism of the earlier work — it's the opposite. The whole
model is a chain of "yes, and": every layer I added is answering a question
that Moss's model or Barry's comments raised but left open.

---

## 1. Where this starts: Moss's IRT reanalysis

The starting point is Jonas Moss's item-response reanalysis of METR's
time-horizon data. The move I found most useful there is the reframing: METR
asks "at what task length does a model succeed 50% of the time?", and Moss
recasts the whole thing as a **2-parameter logistic IRT model**. Each task is
an *item* with a difficulty and a discrimination; each model is a *respondent*
with an ability; a model's probability of solving a task is a logistic function
of (ability − difficulty), scaled by the task's discrimination. The 50% time
horizon then falls out of the fitted abilities instead of being read off a
per-model curve fit.

Two things about Moss's setup carry straight into what I built:

- **Difficulty lives on the human-time scale.** A task's difficulty is
  anchored to `human_minutes` — the human baseline time — so a model's ability
  `θ_m` is interpretable in log-minutes and `exp(θ_m)` is its 50% horizon.
- **Ability moves over time.** He puts a log-linear trend on ability across
  release dates, which is what turns the cross-section into a doubling-time
  estimate.

I kept both. What I did *not* keep is one modelling assumption underneath them,
and that assumption is the reason this project exists.

## 2. The itch: human time is a measurement, not a constant

In Moss's model each task's `human_minutes` is a **fixed, exactly-known
scalar**. But look at where that number comes from: for most tasks it's the
geometric mean of one or two human baseline runs, and for a chunk of tasks
there is no timed run at all — just an expert's estimate. Within a task, human
wall-times routinely differ by 1.5–2× (people take breaks, get stuck, vary),
and the estimate-only tasks could be off by much more.

So `human_minutes` is not a constant; it's a **noisy, sparsely-observed
measurement** of a latent "true" task length. And because difficulty is built
directly on it, that measurement error is being silently injected into every
task's difficulty — and then, through the difficulty-vs-ability comparison,
into the horizon and the trend. Treating the noisy thing as known truth doesn't
make the noise go away; it just hides it and makes the posterior look more
certain than it should.

That's the itch. The natural fix is Bayesian: promote `human_minutes` from a
constant to a **latent variable** with its own measurement model, and let the
data — both the timing observations and the success/failure pattern — inform
it jointly. That's the core of this whole model.

## 3. Two things Barry pointed out

Before the model, the two things from Alexander Barry's comments that shaped it:

- **Estimates are worse than I'd have guessed, and by a measurable amount.**
  Barry noted that only ~60% of the estimate-only human-time annotations land
  within a factor of 3 of the real baseline time where both exist. That's a
  concrete, usable number: under a log-scale error it pins the noise on an
  estimate annotation to about `ln(3)/Φ⁻¹(0.8) ≈ 1.3` log-minutes. My own data
  can't check this (no task carries both an annotation *and* a timed run), so
  Barry's figure enters the model as an external prior on the estimate-noise
  scale. Without it, the model has no idea how much to distrust an estimate.

- **The circularity warning.** Barry's sharper point: once you jointly model
  task difficulty *and* task length and let both be informed by the
  success/failure data, you open a feedback loop. The success outcomes can push
  a task's *inferred length* around — a task models keep solving gets re-dated
  shorter, one they keep failing gets re-dated longer — and then the trend is
  partly fit to lengths the outcomes themselves chose. A Bayesian model built
  carelessly here can end up *more* misleading than the frequentist original.
  This is exactly the kind of thing that's invisible until someone names it,
  and it turned into a diagnostic and a cut-model check later on.

So: from Moss, the IRT skeleton and the human-time difficulty scale; from
Barry, the calibration for estimate noise and a standing warning about
feedback. Now the model.

## 4. The model I built

The model has three layers stacked on a shared set of per-task latents. The
one-line version: **each task has a true log-length I don't observe directly; I
learn it from the noisy timing data and the success pattern at once, and I keep
an explicit "residual difficulty" term so that length doesn't have to explain
everything.**

**Measurement layer.** Every task `i` gets a latent log-length `log(L_i)`. The
timed human runs are noisy observations of it, `log(dur) ~ Normal(log(L_i),
σ_base)`; the estimate-only annotations are noisier observations of the same
thing, `log(rep) ~ Normal(log(L_i), σ_est)`, with `σ_est`'s prior set by
Barry's 60%-within-3× number. This is the layer that promotes `human_minutes`
from constant to latent.

**IRT layer — and the one genuinely new term.** Keeping Moss's 2PL, the
probability model `m` solves task `i` is `logit⁻¹(a_i · (θ_m − difficulty_i))`.
The difficulty is where I depart from Moss:

> `difficulty_i = log(L_i) + ε_i`

`ε_i` is a **residual difficulty** term — how much harder or easier a task is
for models than its human length alone predicts. It has to be there: task
length is only a rough proxy for what makes something hard for a model (how
much context it must hold, how many tool calls it has to chain correctly,
whether it's the kind of reasoning these models are good at). Without `ε`, all
of that mismatch would be forced back onto `log(L)` and would corrupt the
timing estimate.

**The identification problem `ε` creates, and the fix.** The moment you add a
free per-task `ε` to the difficulty, you get a non-identifiability: only
`θ_m − (log L_i + ε_i)` enters the likelihood, so adding a constant to every
`ε` and subtracting it from every `θ` leaves everything unchanged. There's a
flat ridge in the posterior and the sampler wanders along it. Moss avoids the
analogous problem by **hard-anchoring two ability values** (`θ_low = −1`,
`θ_high = +1`). I couldn't do that, because anchoring `θ` would destroy the
log-minute meaning of the horizon that the timing data works so hard to pin
down. Instead I constrain the residuals to **sum to zero** across tasks (a
`ZeroSumNormal`). That removes the ridge directly — the shift degree of freedom
is gone — while keeping `θ_m` interpretable as a log-minute horizon. Same
disease Moss treats, a different medicine chosen to protect the property I
care about.

**Trend layer.** Ability is `θ_m = f(t_m) + u_m`: a trend `f` in release date
plus a **per-model random effect** `u_m` so the doubling-time interval reflects
real model-to-model scatter, not just within-model noise. `f` is one of four
shapes (linear, a kink, super-exponential, logistic), combined by Bayesian
stacking rather than picking one.

**Geometry.** With only 1–2 timing observations per task, the naive
parameterization funnels badly, so `log L`, `ε`, and `u` are all non-centered.
This is plumbing, but it's the difference between the sampler working and not.

That's the model: Moss's IRT trend, wrapped in a measurement layer that treats
human time as latent, with a sum-to-zero residual-difficulty term as the new
structural piece.

## 5. What it said

The first-order results:

- **The residual difficulty is large.** `σ_ε ≈ 2.2` log-minutes — about an
  **8× spread** in difficulty at fixed task length. That's the empirical case
  for having the layer at all: length really is a noisy proxy. A task that
  reads as ten minutes of human work can be as hard for a model as an
  hour-and-twenty, or vice versa.
- **The doubling time is ~3.3 months** (linear), landing in the same
  neighbourhood as Moss's and METR's numbers once you line the trend shapes and
  time windows up; the differences between the three trace to trend shape and
  the added measurement/random-effect structure, not to the data going in.
- The estimate-noise and circularity concerns from Barry both showed up exactly
  where predicted: the feedback is real and detectable (easy-for-models tasks
  pulled shorter, hard ones longer), but a cut-model refit that severs the loop
  moves the headline by about a day. Named, measured, contained.

Then the stress tests. Moss suggested trying a **Weibull** duration likelihood
instead of log-normal; I did, and it fits *worse* — our within-task residuals
are right-skewed with heavy tails, and a Weibull's log is fixed left-skewed, so
it's structurally the wrong shape. A clean negative result, but a real one. A
**Student-t** duration likelihood, by contrast, earns its place: the wall-clock
data has genuine heavy tails (a few runs 3–4 log-units off their task median),
and the fitted degrees of freedom come out around 2.4 — the data decisively
want heavy tails. And reduced-scale **simulation-based calibration** passes, so
the machinery recovers what it puts in.

## 6. The additions I tried

Everything above is the model as first built. Once it was running, each thing I
looked at hard suggested a next thing to add. This is the part I most want
feedback on.

**Is the reported horizon even the same object as METR's?** `exp(θ_m)` is a
horizon *conditional* on a task of median residual difficulty (`ε = 0`); METR's
is *marginal*, averaged over the task population. With `σ_ε` this large, I
worried these differed a lot. Working it through: the 50% level and the
doubling-time slope are actually **safe** — by symmetry the marginal success
curve still crosses 0.5 at `θ_m`, and a stationary `ε` distribution cancels out
of the slope. But everything else isn't: the population curve is ~1.9× flatter,
so any non-50% reliability horizon or any extrapolation to a fixed threshold
diverges from METR's. So the rule became: quote the 50% horizon and doubling
time as-is, but compute anything else marginally.

**What does the measurement layer actually buy?** Uncomfortable observation:
because the IRT layer only sees `log L + ε` and `ε` is free and large, any
error in `log L` gets absorbed by `ε` as far as the trend is concerned — so the
measurement layer barely moves the *point* estimate. What it buys is
**honesty about uncertainty**: a task timed by a single human run has a real
posterior spread in its length (~1.8×) that a plug-in model treating
`human_minutes` as exact throws away. The layer widens the intervals for
exactly the sparsely-timed tasks that deserve it.

**The measurement noise isn't constant.** The within-task spread of log
wall-time grows with task length in the data (short tasks ~0.28, long ~0.57).
So I let the noise scale with length, `σ_base,i = σ_base · exp(γ · (log L_i −
μ))`. The data are decisive: `γ = +0.10`, essentially zero probability of being
negative, and it improves the duration fit by a solid margin. It doesn't touch
the headline, but it's a strictly better-specified measurement layer.

**Survivorship.** METR (and so the model) uses only *successful* human runs as
timing anchors. But there are 129 *failed* timed human runs on the snapshot,
sitting on hard tasks, with median wall-times of a couple of hours versus a few
minutes for successes. Dropping them truncates the length distribution to
"fast enough to succeed." I added them back as **right-censored** observations
(a failure that ran `w` minutes says the completion time exceeds `w`), through
the censoring path the model already had. It's self-weighting — a quick give-up
says almost nothing, a genuine hard-task failure pulls hard — and it lifts the
hard tasks' lengths where it should. I keep it as a sensitivity rather than the
default, because it does subtly shift the estimand away from METR's
"successful-completion" definition of human time.

**The addition I care about most: structuring the residual difficulty.** `σ_ε`
being ~8× is the headline uncertainty, and I'd been reporting all of it as
irreducible. But is it? I split `ε` into a **task-family** effect plus a
within-family residual. The answer: **about two-thirds of the residual-
difficulty variance is between-family** — i.e. predictable structure, not
noise. And the family effects are exactly what you'd expect on inspection:
pattern-continuation and cryptanalysis families are ~100× *harder* than their
human length implies, while arithmetic, file-selection and alert-triage are
~100× *easier*. The success data clearly prefer this structured version. This
matters two ways: it says the genuinely irreducible per-task difficulty is
smaller (~5×, not 8×), and — because `ε` is the part of difficulty the trend is
actually read against — it's the **one refinement that moves the headline**,
tightening the current doubling time to about 2.4 [2.1, 2.7] months.

That last point is the through-line of the whole exercise. Refinements to the
*length* channel wash out, because `ε` buffers them; the refinement to the
*difficulty* channel is what sharpens the trend. The model is telling you where
its signal actually lives — and it lives in `ε`, the term that wasn't in the
original at all.

## 7. Where it lands

Stepping back, the lineage is: Moss's IRT-on-human-time gives the skeleton and
the horizon scale; the measurement-error layer answers "but human time is
noisy"; the residual-difficulty term answers "but length isn't difficulty," and
forces the sum-to-zero identification; Barry's two points calibrate the
estimate noise and flag the feedback; and the later additions answer the
questions the running model raised about itself — which horizon it's reporting,
what the measurement layer is really for, whether the noise is constant, whether
survivorship bites, and how much of the residual difficulty is real structure.

The reassuring summary is that the doubling time stays in a ~2.4–3.3 month band
through every one of these changes. The interesting summary is *why* it moves
when it moves: not when I refine the timing, but when I refine what "difficulty"
means. That's the part I'd most like Barry's and Moss's read on.
