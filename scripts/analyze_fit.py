"""Posterior predictive checks and horizon summaries for a fitted
time-horizon model (outputs/fit_linear_*.nc).

Reports:
  1. IRT-layer PPC: observed vs posterior-predictive success rates, per
     model and per task-length bin (does the 2PL+eps structure reproduce
     the success pattern?).
  2. Measurement-layer PPC: within-task spread of log wall-times vs
     posterior predictive.
  3. Per-model 50% horizons, h50_m = exp(theta_m) minutes (the task length
     at which the *median* task -- eps = 0 -- is solved with p = 0.5), and
     the current trend doubling time ln(2)/slope_now (works for any of the
     four trend shapes, not just linear).

Usage:
    uv run python scripts/analyze_fit.py --fit outputs/fit_linear_robust.nc
"""

from __future__ import annotations

import argparse

import arviz as az
import numpy as np

from metr_measurement_error.data_prep import load_model_data
from metr_measurement_error.paths import PROCESSED_DATA

DEFAULT_DATA = str(PROCESSED_DATA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument(
        "--sota-only",
        action="store_true",
        help="Load the SOTA-only restricted data (must match how the fit was run).",
    )
    args = parser.parse_args()

    data = load_model_data(args.data, sota_only=args.sota_only)
    idata = az.from_netcdf(args.fit)
    post = idata.posterior

    theta = post["theta"].stack(s=("chain", "draw")).values      # (model, S)
    a = post["a"].stack(s=("chain", "draw")).values              # (task, S)
    log_L = post["log_L"].stack(s=("chain", "draw")).values      # (task, S)
    eps = post["eps"].stack(s=("chain", "draw")).values          # (task, S)
    slope_now = post["slope_now"].stack(s=("chain", "draw")).values  # (S,)
    S = theta.shape[1]

    # ---------- 1. IRT-layer PPC ----------
    ti, mi = data.task_idx_irt, data.model_idx_irt
    eta = a[ti, :] * (theta[mi, :] - (log_L[ti, :] + eps[ti, :]))  # (n_irt, S)
    p = 1.0 / (1.0 + np.exp(-eta))
    rng = np.random.default_rng(1234)
    n = data.n_attempts[:, None]
    s_rep = rng.binomial(n, p)  # (n_irt, S)

    print("=== IRT posterior predictive: success rate by model ===")
    print(f"{'model':42s} {'obs':>6s} {'pp mean':>8s} {'pp 95% CI':>16s}")
    for m, name in enumerate(data.model_names):
        rows = mi == m
        obs_rate = data.n_successes[rows].sum() / data.n_attempts[rows].sum()
        rep_rate = s_rep[rows, :].sum(axis=0) / data.n_attempts[rows].sum()
        lo, hi = np.quantile(rep_rate, [0.025, 0.975])
        flag = " *" if not (lo <= obs_rate <= hi) else ""
        print(f"{name:42s} {obs_rate:6.3f} {rep_rate.mean():8.3f}   [{lo:.3f}, {hi:.3f}]{flag}")

    print("\n=== IRT posterior predictive: success rate by task-length bin ===")
    log_L_mean = log_L.mean(axis=1)
    bins = np.quantile(log_L_mean, np.linspace(0, 1, 7))
    bin_idx = np.clip(np.digitize(log_L_mean[ti], bins) - 1, 0, 5)
    for b in range(6):
        rows = bin_idx == b
        if rows.sum() == 0:
            continue
        obs_rate = data.n_successes[rows].sum() / data.n_attempts[rows].sum()
        rep_rate = s_rep[rows, :].sum(axis=0) / data.n_attempts[rows].sum()
        lo, hi = np.quantile(rep_rate, [0.025, 0.975])
        lo_min, hi_min = np.exp(bins[b]), np.exp(bins[b + 1])
        flag = " *" if not (lo <= obs_rate <= hi) else ""
        print(
            f"  {lo_min:9.1f}-{hi_min:9.1f} min  obs {obs_rate:.3f}  "
            f"pp {rep_rate.mean():.3f} [{lo:.3f}, {hi:.3f}]{flag}"
        )

    # ---------- 2. Measurement-layer PPC ----------
    base = ~data.is_estimate
    tio = data.task_idx_obs[base]
    obs_ld = data.log_dur[base]
    sigma_base = post["sigma_base"].stack(s=("chain", "draw")).values
    # within-task sd of log duration, tasks with >=2 obs
    from collections import defaultdict

    by_task = defaultdict(list)
    for t, v in zip(tio, obs_ld):
        by_task[t].append(v)
    obs_sds = [np.std(v, ddof=1) for v in by_task.values() if len(v) >= 2]
    print("\n=== Measurement-layer PPC ===")
    print(f"observed median within-task sd(log dur): {np.median(obs_sds):.3f}")
    print(f"posterior sigma_base: {sigma_base.mean():.3f} [{np.quantile(sigma_base, 0.025):.3f}, {np.quantile(sigma_base, 0.975):.3f}]")

    # ---------- 3. Horizons and doubling time ----------
    print("\n=== 50% horizons (median task, eps=0): exp(theta_m) minutes ===")
    order = np.argsort([data.t_model[m] if data.has_date[m] else -99 for m in range(data.n_models)])
    for m in order:
        h = np.exp(theta[m, :])
        lo, hi = np.quantile(h, [0.025, 0.975])
        tag = "" if data.has_date[m] else " (undated)"
        print(f"{data.model_names[m]:42s} {np.median(h):9.1f} min  [{lo:8.1f}, {hi:9.1f}]{tag}")

    dt_months = np.log(2) / slope_now * 12.0
    lo, hi = np.quantile(dt_months, [0.025, 0.975])
    print(f"\nDoubling time now (ln(2)/slope_now): median {np.median(dt_months):.1f} months  95% CI [{lo:.1f}, {hi:.1f}]")
    sigma_eps = post["sigma_eps"].stack(s=("chain", "draw")).values
    print(f"sigma_eps (residual difficulty, log-minutes): {sigma_eps.mean():.2f} [{np.quantile(sigma_eps, 0.025):.2f}, {np.quantile(sigma_eps, 0.975):.2f}]")
    sigma_est = post["sigma_est"].stack(s=("chain", "draw")).values
    print(f"sigma_est (annotation noise): {sigma_est.mean():.2f} [{np.quantile(sigma_est, 0.025):.2f}, {np.quantile(sigma_est, 0.975):.2f}]")


if __name__ == "__main__":
    main()
