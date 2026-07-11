"""PyMC implementation of the v2 Bayesian measurement-error time-horizon
model (4 ability-trend shapes, optional Student-t measurement layer).

This is an extension of Jonas Moss's 2PL time-horizon IRT model
(metr-stats/stan/2pl_time_loglinear_theta_trend.stan), which treats each
task's human_minutes as a fixed, exactly-known scalar. That is the gap this
model closes: human timing is noisy and sparsely observed (most tasks have
only 1-2 human attempts), so treating it as known truth silently injects
measurement error into the difficulty term and, via the difficulty-vs-length
regression, into the horizon estimate itself. Here, each task's "true" log
length log(L_i) is a latent variable inferred jointly from (a) the noisy
human timing observations and (b) the IRT success/failure pattern across
models, with a free residual-difficulty term (eps_i) absorbing whatever
difficulty variance length alone doesn't explain.

Model (linear theta trend):

    Measurement layer
        log(L_i)        ~ Normal(mu_L, sigma_L)                      i = 1..n_tasks
        log(dur_{i,b})   ~ Normal(log(L_i), sigma_base)               baseline (real-timed) obs
                            [right-censored at log(time_limit) where is_censored]
        log(rep_i)       ~ Normal(log(L_i), sigma_est)                estimate-only obs (no timed run)

    IRT layer (residual difficulty eps_i is the addition vs. Moss's model)
        a_i    ~ LogNormal(0, sigma_a)
        eps_i  ~ ZeroSumNormal(sigma_eps)   # sum-to-zero: see comment in build_model
        logit P(success_{i,m}) = a_i * (theta_m - (log(L_i) + eps_i))
        successes_{i,m} ~ Binomial(attempts_{i,m}, P(success_{i,m}))

    Ability / trajectory layer
        theta_m = beta0 + beta1 * t_m + u_m         (dated models)
        theta_m = beta0 + u_m                        (undated models, no trend term)
        u_m ~ Normal(0, sigma_u)                     per-model random effect

Options on top of the base spec (all actually implemented, not stubs):

- `duration_dist` selects the baseline-run likelihood: "lognormal"
  (Normal on log(dur), the default), "studentt" (heavy-tailed on log(dur),
  same as `robust="studentt"`), or "weibull" (Weibull on the raw duration
  scale, median-matched so beta_i = L_i / ln(2)^(1/alpha) keeps log(L_i)
  the log median wall time under every variant; Moss suggested trying
  Weibull over log-normal on theoretical grounds).

- `sigma_est_median` sets the prior median of the estimate-annotation
  noise `sigma_est ~ LogNormal(log(median), 0.5)`. Default 1.25,
  calibrated to Alexander Barry's finding (LW comments) that only ~60% of
  estimate annotations fall within 3x of the real baseline time:
  P(|N(0, sigma)| < ln 3) = 0.6 implies sigma = ln(3)/Phi^-1(0.8) = 1.305
  total, ~1.27 after removing the baseline geomean's own ~0.3 noise
  contribution. The pre-calibration value was 0.8.

- `robust="studentt"` swaps the baseline-run likelihood from Normal to
  Student-t with estimated dof `nu ~ Gamma(2, 0.1)` (mean 20; lets the data
  choose between near-Normal and heavy tails). Motivation: the real
  wall-clock data contains within-task deviations of 3-4 log units (a
  249-min run on a 4-min-median task; a 2185-min run on a 96-min-median
  task), and a Normal likelihood lets those single runs drag the task's
  latent log_L and inflate sigma_base for everyone. The estimate-only
  likelihood stays Normal: each estimate task contributes exactly one
  observation, so there is no within-task replication for a tail parameter
  to learn from there.

- `cut_estimate_feedback=True` breaks the IRT -> log_L feedback loop for
  estimate-only tasks (a cut-model / modularized-inference variant, per
  Alexander Barry's warning that success/failure outcomes can circularly
  contaminate inferred task lengths for tasks whose only timing datum is an
  annotation). For the estimate-only tasks, the IRT layer sees the raw
  annotation log(rep_i) as a fixed constant instead of the latent log(L_i);
  with a single annotation and no timed runs, the measurement-only
  posterior mean of log(L_i) is essentially the annotation itself, so this
  clamps those tasks at their measurement-only estimate. Baseline-informed
  tasks are untouched. eps_i stays free, so residual difficulty is still
  estimated; only the length channel is cut.

- `shape` selects the ability-trend mean function f(t_m):
    "linear":  beta0 + beta1*t
    "kink":    beta0 + beta1*t + delta*softplus((t-t_k)/w)*w   (w=0.1yr,
               a smoothed slope change of delta at breakpoint t_k)
    "superexp": beta0 + beta1*t + beta2*t^2  (log-horizon curvature; beta2>0
               means the doubling time itself shrinks over time)
    "logistic": beta0 + h*sigmoid((t-t0)/s)  (saturating ability)
  Each model exposes pm.Deterministic("slope_now") = df/dt evaluated at the
  latest dated model's t, so ln(2)/slope_now is the *current* doubling time
  under any shape; shapes are combined via Bayesian stacking (Yao et al.
  2018) in scripts/stack_shapes.py using az.compare on the pointwise
  log-likelihood of the IRT `successes` (the only likelihood term the
  shapes differ on).
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from metr_measurement_error.data_prep import ModelData

SHAPES = ("linear", "kink", "superexp", "logistic")


DURATION_DISTS = ("lognormal", "studentt", "weibull")


def build_model(
    data: ModelData,
    shape: str = "linear",
    robust: str | None = None,
    duration_dist: str | None = None,
    sigma_est_median: float = 1.25,
    cut_estimate_feedback: bool = False,
) -> pm.Model:
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}, got {shape!r}")
    if robust not in (None, "studentt"):
        raise ValueError(f"robust must be None or 'studentt', got {robust!r}")
    # `duration_dist` supersedes `robust` (kept for backwards compat):
    # "lognormal" = Normal on log(dur); "studentt" = Student-t on log(dur);
    # "weibull" = Weibull on the raw duration scale (Moss's suggestion).
    if duration_dist is None:
        duration_dist = "studentt" if robust == "studentt" else "lognormal"
    if duration_dist not in DURATION_DISTS:
        raise ValueError(f"duration_dist must be one of {DURATION_DISTS}, got {duration_dist!r}")
    is_baseline = ~data.is_estimate
    is_censored = data.is_censored

    # Split observation indices into three disjoint groups so we can hand
    # each its own PyMC likelihood (censored baseline, uncensored baseline,
    # estimate-only). None are censored in the current data snapshot, but
    # the branch is kept so the model stays correct if that changes.
    idx_base_obs = np.where(is_baseline & ~is_censored)[0]
    idx_base_cens = np.where(is_baseline & is_censored)[0]
    idx_est = np.where(data.is_estimate)[0]

    coords = {
        "task": data.task_ids,
        "model": data.model_names,
    }

    with pm.Model(coords=coords) as model:
        # ---- Measurement layer ----
        # Non-centered parameterization: with only 1-2 observations for most
        # tasks, a centered log_L ~ Normal(mu_L, sigma_L) produces a severe
        # funnel (sigma_L and log_L are highly correlated in the posterior),
        # which in practice showed up as hundreds of divergences and R-hat
        # >2 even with non-trivial tuning. Non-centering log_L (and eps, u
        # below, for the same reason) fixes the geometry.
        mu_L = pm.Normal("mu_L", mu=3.0, sigma=2.0)  # log-minutes prior center (~e^3 ~ 20 min)
        sigma_L = pm.HalfNormal("sigma_L", sigma=1.5)
        log_L_raw = pm.Normal("log_L_raw", mu=0.0, sigma=1.0, dims="task")
        log_L = pm.Deterministic("log_L", mu_L + sigma_L * log_L_raw, dims="task")

        # Measurement-noise scales. sigma_base is the noise on real timed
        # ("baseline") human wall-clock runs; empirically the within-task sd
        # of log wall-time is ~0.4-0.6 (people differ 1.5-2x, plus breaks),
        # so the prior scale is 1.0. sigma_est is the noise on task-level
        # annotations for tasks with no timed run; those are expert
        # estimates, plausibly off by several-fold, so its prior is wider.
        # sigma_base is only meaningful for the log-scale likelihoods; under
        # the Weibull the spread is carried by the shape parameter alpha_w.
        if duration_dist in ("lognormal", "studentt"):
            sigma_base = pm.HalfNormal("sigma_base", sigma=1.0)
        # sigma_est gets a prior bounded away from zero (LogNormal) rather
        # than a HalfNormal. Each estimate-only task contributes a single
        # annotation, so the data cannot rule out sigma_est ~ 0; a
        # HalfNormal leaves a funnel neck there (the one slow mixer left
        # after the other fixes: R-hat 1.02-1.05, ESS < 150 on sigma_est)
        # and sigma_est -> 0 would assert that expert guesses for 30-hour
        # tasks are near-exact, which is absurd a priori.
        #
        # The prior median (default 1.25) is calibrated to Alexander
        # Barry's empirical finding (comments on Moss's LW post) that only
        # ~60% of estimate annotations land within 3x of the real baseline
        # time where both exist. Under a Normal log-scale error,
        # P(|err| < ln 3) = 0.60 implies total log-sd ln(3)/Phi^-1(0.8) =
        # 1.305; his comparison target (the baselined human_minutes, a
        # geomean of ~2-4 runs each with ~0.4-0.6 log-sd) contributes
        # ~0.3 of that, leaving sqrt(1.305^2 - 0.3^2) ~ 1.27 for the
        # estimate itself; we round down to 1.25. Our own data cannot check
        # this (no task has both annotation types), so it enters as prior
        # evidence. The original pre-calibration value was 0.8
        # (sigma_est_median=0.8 reproduces it).
        sigma_est = pm.LogNormal("sigma_est", mu=np.log(sigma_est_median), sigma=0.5)

        if len(idx_base_obs) > 0:
            if duration_dist == "studentt":
                # Heavy-tailed likelihood for real timed runs. nu ~ Gamma(2,
                # 0.1) (mean 20, mode 10) spans "basically Normal" to "very
                # heavy"; the data decide. sigma_base is then the *scale* of
                # the t, not the sd -- compare like-for-like via the implied
                # core-noise scale, not raw sigma_base vs the Normal fit.
                nu = pm.Gamma("nu", alpha=2.0, beta=0.1)
                pm.StudentT(
                    "dur_base_obs",
                    nu=nu,
                    mu=log_L[data.task_idx_obs[idx_base_obs]],
                    sigma=sigma_base,
                    observed=data.log_dur[idx_base_obs],
                )
            elif duration_dist == "weibull":
                # Weibull on the *raw duration scale* (Moss: "I would also
                # try a Weibull distribution instead of log-normal").
                # Median-matched parameterization: median(Weibull(alpha,
                # beta)) = beta * ln(2)^(1/alpha), so beta_i = L_i /
                # ln(2)^(1/alpha) keeps log_L interpretable as the log
                # *median* wall time -- exactly the interpretation it has
                # under the log-normal layer (whose median is also L_i).
                # Mean-matching would instead make L_i's meaning depend on
                # the fitted alpha via the Gamma(1+1/alpha) factor.
                # alpha prior: the log-sd of a Weibull is (pi/sqrt(6))/alpha
                # ~ 1.283/alpha, and the observed within-task log-sd is
                # ~0.4-0.6, so alpha ~ 2-3 is plausible a priori;
                # LogNormal(log 1.5, 0.5) spans ~[0.6, 4].
                alpha_w = pm.LogNormal("alpha_w", mu=np.log(1.5), sigma=0.5)
                beta_w = pt.exp(
                    log_L[data.task_idx_obs[idx_base_obs]]
                    - np.log(np.log(2.0)) / alpha_w
                )
                pm.Weibull(
                    "dur_base_obs",
                    alpha=alpha_w,
                    beta=beta_w,
                    observed=np.exp(data.log_dur[idx_base_obs]),
                )
            else:
                pm.Normal(
                    "dur_base_obs",
                    mu=log_L[data.task_idx_obs[idx_base_obs]],
                    sigma=sigma_base,
                    observed=data.log_dur[idx_base_obs],
                )

        if len(idx_base_cens) > 0:
            if duration_dist == "weibull":
                base_dist = pm.Weibull.dist(
                    alpha=alpha_w,
                    beta=pt.exp(
                        log_L[data.task_idx_obs[idx_base_cens]]
                        - np.log(np.log(2.0)) / alpha_w
                    ),
                )
                pm.Censored(
                    "dur_base_censored",
                    base_dist,
                    lower=None,
                    upper=np.exp(data.censor_log_time[idx_base_cens]),
                    observed=np.exp(data.log_dur[idx_base_cens]),
                )
            else:
                base_dist = pm.Normal.dist(
                    mu=log_L[data.task_idx_obs[idx_base_cens]], sigma=sigma_base
                )
                pm.Censored(
                    "dur_base_censored",
                    base_dist,
                    lower=None,
                    upper=data.censor_log_time[idx_base_cens],
                    observed=data.log_dur[idx_base_cens],
                )

        if len(idx_est) > 0:
            pm.Normal(
                "dur_estimate",
                mu=log_L[data.task_idx_obs[idx_est]],
                sigma=sigma_est,
                observed=data.log_dur[idx_est],
            )

        # ---- IRT layer ----
        sigma_a = pm.HalfNormal("sigma_a", sigma=0.5)
        log_a_raw = pm.Normal("log_a_raw", mu=0.0, sigma=1.0, dims="task")
        a = pm.Deterministic("a", pt.exp(log_a_raw * sigma_a), dims="task")

        # Residual difficulty. Sum-to-zero constraint: eta depends on
        # theta_m - (log_L_i + eps_i), so a constant shift of all eps_i is
        # exactly absorbed by an opposite shift of all theta_m. With a free
        # sigma_eps, the mean-zero prior resists that shift only weakly,
        # leaving a near-flat ridge that NUTS chains park at different
        # points along (this is the identification failure Moss avoids by
        # hard-anchoring two theta values; anchoring is wrong here because
        # our difficulty scale is pinned in log-minute units by the timing
        # data, and anchors would destroy that interpretation). Constraining
        # eps to sum to zero removes the ridge while keeping theta_m
        # directly interpretable as a log-minutes horizon.
        sigma_eps = pm.HalfNormal("sigma_eps", sigma=0.5)
        eps_raw = pm.ZeroSumNormal("eps_raw", sigma=1.0, dims="task")
        eps = pm.Deterministic("eps", eps_raw * sigma_eps, dims="task")

        # Cut-model variant: for estimate-only tasks, the IRT layer uses the
        # raw annotation as a fixed constant in place of the latent log_L,
        # so success/failure outcomes cannot move those tasks' inferred
        # lengths (the circularity Barry warned about). The measurement
        # layer above is untouched, so log_L for those tasks still gets its
        # measurement-only posterior; it just can't feed the IRT layer.
        if cut_estimate_feedback and len(idx_est) > 0:
            est_task_idx = data.task_idx_obs[idx_est]
            is_est_task = np.zeros(len(data.task_ids), dtype=bool)
            is_est_task[est_task_idx] = True
            log_L_clamp = np.zeros(len(data.task_ids))
            log_L_clamp[est_task_idx] = data.log_dur[idx_est]
            log_L_irt = pt.where(
                pt.as_tensor_variable(is_est_task),
                pt.as_tensor_variable(log_L_clamp),
                log_L,
            )
        else:
            log_L_irt = log_L

        difficulty = log_L_irt + eps  # task difficulty on the log-minutes scale

        # ---- Ability trend ----
        sigma_u = pm.HalfNormal("sigma_u", sigma=1.0)
        u_raw = pm.Normal("u_raw", mu=0.0, sigma=1.0, dims="model")
        u = pm.Deterministic("u", u_raw * sigma_u, dims="model")

        t_model = pt.as_tensor_variable(data.t_model)
        has_date = pt.as_tensor_variable(data.has_date.astype(float))
        # Latest dated model's t: where the "current doubling time" is read off.
        t_now = float(data.t_model[data.has_date].max())

        if shape == "linear":
            beta0 = pm.Normal("beta0", mu=0.0, sigma=1.5)
            beta1 = pm.Normal("beta1", mu=0.0, sigma=1.0)
            f_t = beta0 + beta1 * t_model
            slope_now = beta1
        elif shape == "kink":
            # Smoothed breakpoint: slope beta1 before t_k, beta1 + delta
            # after, blended over w = 0.1 yr so the gradient exists
            # everywhere. t_k's prior keeps the breakpoint inside the span
            # of dated models (t in [-1.9, 1.0]).
            w = 0.1
            beta0 = pm.Normal("beta0", mu=0.0, sigma=1.5)
            beta1 = pm.Normal("beta1", mu=0.0, sigma=1.0)
            delta = pm.Normal("delta", mu=0.0, sigma=1.0)
            t_k = pm.Normal("t_k", mu=0.0, sigma=0.75)
            f_t = beta0 + beta1 * t_model + delta * w * pt.softplus((t_model - t_k) / w)
            slope_now = beta1 + delta * pm.math.sigmoid((t_now - t_k) / w)
        elif shape == "superexp":
            beta0 = pm.Normal("beta0", mu=0.0, sigma=1.5)
            beta1 = pm.Normal("beta1", mu=0.0, sigma=1.0)
            beta2 = pm.Normal("beta2", mu=0.0, sigma=0.5)
            f_t = beta0 + beta1 * t_model + beta2 * t_model**2
            slope_now = beta1 + 2.0 * beta2 * t_now
        else:  # logistic
            # beta0 is the early-era (lower-asymptote) log-horizon, h the
            # total rise. Observed thetas span roughly -2..5 log-minutes, so
            # h ~ HalfNormal(8) comfortably covers the plausible rise; s is
            # bounded away from 0 (LogNormal) because s -> 0 is a step
            # function with a gradient cliff.
            beta0 = pm.Normal("beta0", mu=0.0, sigma=2.0)
            h = pm.HalfNormal("h", sigma=8.0)
            t0 = pm.Normal("t0", mu=0.0, sigma=1.0)
            s = pm.LogNormal("s", mu=np.log(0.5), sigma=0.5)
            sig = pm.math.sigmoid((t_model - t0) / s)
            f_t = beta0 + h * sig
            sig_now = pm.math.sigmoid((t_now - t0) / s)
            slope_now = h / s * sig_now * (1.0 - sig_now)

        pm.Deterministic("slope_now", slope_now)
        trend_mean = beta0 + (f_t - beta0) * has_date
        theta = pm.Deterministic("theta", trend_mean + u, dims="model")

        eta = a[data.task_idx_irt] * (
            theta[data.model_idx_irt] - difficulty[data.task_idx_irt]
        )
        pm.Binomial(
            "successes",
            n=data.n_attempts,
            p=pm.math.sigmoid(eta),
            observed=data.n_successes,
        )

    return model
