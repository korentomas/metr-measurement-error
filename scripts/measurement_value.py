"""What the measurement-error layer actually buys, read off a full fit.

Red-team point #2: because the IRT layer sees only difficulty_i = log_L_i +
eps_i and eps is free with a large sigma_eps, the measurement layer barely
moves the horizon *point* estimate (documented: Normal vs Student-t doubling
time 3.4 vs 3.3 mo). Its real value is *uncertainty propagation*: a task
timed by a single human run has genuine posterior uncertainty in log_L that a
plug-in model (Moss/METR: human_minutes treated as exactly known) sets to
zero, making downstream horizons overconfident for agents whose ability is
pinned by sparsely-timed tasks.

This script quantifies that from an existing fit, no refit needed:
  1. posterior sd of log_L vs the number of timed human runs per task -- is
     measurement uncertainty real, and is it concentrated on sparse tasks?
  2. how large that uncertainty is relative to sigma_base (the per-run noise)
     and to the residual-difficulty scale sigma_eps.
  3. the plug-in gap: the measurement sd that a fixed-log_L model discards.

Usage:
    uv run python scripts/measurement_value.py --fit outputs/fit_linear_robust.nc
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import arviz as az
import numpy as np

from models.data_prep import load_model_data

DEFAULT_DATA = str(Path(__file__).parent.parent / "data" / "processed" / "runs_filtered.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--sota-only", action="store_true")
    args = parser.parse_args()

    data = load_model_data(args.data, sota_only=args.sota_only)
    idata = az.from_netcdf(args.fit)
    post = idata.posterior

    log_L = post["log_L"].stack(s=("chain", "draw")).values   # (task, S)
    eps = post["eps"].stack(s=("chain", "draw")).values       # (task, S)
    sd_logL = log_L.std(axis=1)                               # (task,)
    sd_eps = eps.std(axis=1)
    sigma_base = float(post["sigma_base"].mean())
    sigma_eps = float(post["sigma_eps"].mean())

    # timed (non-estimate, non-censored) human runs per task
    base_mask = ~data.is_estimate & ~data.is_censored
    runs_per_task = Counter(data.task_idx_obs[base_mask].tolist())
    n_runs = np.array([runs_per_task.get(t, 0) for t in range(data.n_tasks)])

    print(f"fit: {args.fit}")
    print(f"sigma_base (per-run noise): {sigma_base:.3f}   sigma_eps (residual difficulty): {sigma_eps:.3f}\n")

    print("=== 1-2. posterior sd of log_L by number of timed human runs ===")
    print(f"{'#runs':>6s} {'#tasks':>7s} {'median sd(log_L)':>17s} {'median sd(eps)':>15s}")
    for k in [0, 1, 2, 3]:
        if k < 3:
            sel = n_runs == k
            label = f"{k}"
        else:
            sel = n_runs >= 3
            label = "3+"
        if sel.sum() == 0:
            continue
        print(f"{label:>6s} {int(sel.sum()):>7d} {np.median(sd_logL[sel]):>17.3f} {np.median(sd_eps[sel]):>15.3f}")
    print(f"\n  (#runs=0 tasks are the estimate-only tasks: log_L rests on one")
    print(f"   annotation with the wide sigma_est prior, hence the largest sd.)")

    print("\n=== 3. the plug-in gap ===")
    timed = n_runs >= 1
    single = n_runs == 1
    print(f"  tasks with exactly one timed run: {int(single.sum())} "
          f"({100*single.sum()/data.n_tasks:.0f}% of the {data.n_tasks} tasks)")
    print(f"  median posterior sd(log_L) among those: {np.median(sd_logL[single]):.3f} log-min "
          f"(~{np.exp(np.median(sd_logL[single])):.2f}x)")
    print(f"  a plug-in (fixed human_minutes) model sets all of that to 0.")
    print(f"  Across timed tasks the discarded measurement sd averages "
          f"{np.mean(sd_logL[timed]):.3f} log-min; this is the difficulty-axis")
    print(f"  uncertainty the full model propagates into agent horizons and the")
    print(f"  plug-in does not -- the source of the wider, better-calibrated CIs")
    print(f"  the measurement-error layer is for (the point estimate barely moves).")


if __name__ == "__main__":
    main()
