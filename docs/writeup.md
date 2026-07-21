# A measurement-error model for METR's time horizon

Based on:
- Alexander Barry's [note on modeling assumptions](https://metr.org/notes/2026-03-20-impact-of-modelling-assumptions-on-time-horizon-results/)
- Jonas Moss's [IRT reanalysis](https://www.lesswrong.com/posts/sBEzomgnYJmYHki9T).

Results:

- Doubling time: ~3.3 months on the plain linear trend, tightening to
  2.4 months (95% credible interval: 2.1–2.7) with every refinement in.
- Tasks of the same human length differ ~8x in how hard they are for
  models. Roughly two thirds of that is predictable task-family structure,
  leaving a within-family spread near 5x, which agrees with Moss's ~4.7x.
- Barry found that removing the timing noise cuts the frontier model's 50%
  horizon by 25–40%. My model shows only ~10%, which surprised me enough
  that section 2 is mostly me trying to prove my own model wrong.

## 1. The model

The timing layer is Barry's noise model with the true length hidden: each
task has an unknown true length, timed runs scatter around it, and
estimate annotations scatter more. How much more comes from Barry's own
measurement — 80% of expert estimates land within ~4x of the true length —
which I use as the prior on the estimate noise. A free fit recovers a timing noise of 0.79, close to the 0.78 Barry
estimated by hand, with no access to that number. A Student-t version splits that into a tighter core
(0.41) plus heavy tails, the better read of wall-clock times that include
breaks and interruptions.

The success layer is Moss's IRT setup: how often each AI model succeeds at
each task pins down ability and difficulty, with difficulty = the task's
latent length + a per-task residual `ε` (Moss's unexplained difficulty).
One mechanical change: the success data only see the gap between ability
and difficulty, so a constant could slide freely between `ε` and every
ability score. Moss pins two abilities to stop this. Doing that here would
break the read of ability as a log time horizon, so I force `ε` to sum to
zero across tasks instead.

The trend layer is Moss's ability-over-release-date curve, plus a
per-model offset so the doubling-time interval reflects real
model-to-model scatter. Four curve shapes, averaged by predictive weight.

## 2. The SIMEX comparison

I spent some time on Barry's SIMEX analysis, because at first glance
my model kind of contradicts it: remove the timing noise and his frontier
50% horizon falls 25–40%, mine falls ~11%. To make sure the gap is not
just the two of us measuring different things, I ran his exact ladder on
my model
(`scripts/simex.py`: add extra noise to the timing, refit, extrapolate
back to zero noise) and got an essentially flat line, so the ~11% holds
under his own procedure.

**The gap is `ε`.** The horizon rides on difficulty, and what the success
data pin down is length + `ε` together, not length alone. METR's model has
no `ε`, so there difficulty just is length: shrink a noisy over-long task
and the horizon comes down with it. Barry's 25–40% is exactly right for
that model. Give each task its own difficulty residual (fitted σ ≈ 2.2)
and timing noise mostly just moves the split between
length and `ε`, leaving their sum, and with it the horizon, almost
untouched.

**An attempt to falsify it.** The obvious worry is that `ε` is quietly
absorbing length bias it has no business absorbing. If a long-human,
model-easy task is really a mis-timed short task, `ε` should run
systematically negative on the long, poorly timed tasks. The data go the
other way (`scripts/fork_discriminator.py`): on well-timed tasks, longer
tasks
are mildly harder than their length suggests, and the poorly timed long
tasks sit clearly above even that trend. They are the 8-hour
RE-Bench-style tasks, which are genuinely hard, and the model reads them
that way.

**Why the 10% is not zero.** I got the reason wrong in my first draft. My
guess was that the long estimate-only tasks have thin success data — nice
story, except they're the RE-Bench and AI-R&D tasks every model attempted,
~90 times each. The actual reason is that length and `ε` trade off
strongly in the fit (correlation ~−0.66) but not perfectly, so about a
third of any annotation error still leaks into difficulty. And one task's
difficulty can only be pinned so precisely against uncertain abilities (to
within ~2.3x here), even with that many attempts. So timing noise does
reach the horizon, just heavily damped: Barry's 30% becomes ~10%.

## 3. The additions, briefly

**Marginal and conditional horizons.** Barry's note has the 50% horizon
dropping under noise correction while the 80% horizon rises. The same
split falls out of my model from the difficulty side: the residual is
symmetric around zero, so it cancels at the 50% point and in the slope,
but the success curve for the whole task population is ~1.9x flatter than
for a single typical task, so the 80% horizon moves out and away.

**Length-dependent noise.** The spread between attempts grows with task
length (more can go wrong in a six-hour attempt than in a six-minute one).
Letting the noise scale with length clearly improves the timing fit. It
doesn't change the headline.

**Survivorship.** Only successful human runs anchor task length, but 129
failed timed runs sit on the hard tasks, with median times in the hours.
Adding them as "took at least this long" observations lifts those tasks —
the opposite direction to Barry's long-task shrinkage, and probably both
are happening on different tasks. I keep it as a sensitivity check because
it changes what is being measured: METR defines the human baseline by
successful completions only.

**Structure in the difficulty.** The addition I'd defend hardest.
Splitting `ε` into a task-family effect plus a within-family leftover
shows ~two thirds of the difficulty spread is between families:
predictable structure, not noise. Pattern-continuation and cryptanalysis
run ~100x harder than their length suggests, arithmetic and file-selection
~100x easier, models being predictably good at arithmetic. The
within-family leftover is ~5x, matching Moss's ~4.7x. And because the
trend is fit on difficulty, and `ε` is the biggest part of difficulty,
this is the one refinement that moves the headline: 2.4 months (2.1–2.7).

## 4. Conclusion

The doubling time stays between 2.4 and 3.3 months through every variant,
so the trend is robust to how the timing is modeled. Removing the timing
noise moves the horizon by 9–11% whichever model variant does the
correcting, an order of magnitude below the difficulty-equals-length
figure. What remains is that
pinning down a single task's difficulty from success data has a floor
(~2.3x here, even with ~90 attempts), and some of Barry's length-noise
effect survives through it as that 10%. The question I can't settle from
here is whether that floor belongs to the model or to the data.
