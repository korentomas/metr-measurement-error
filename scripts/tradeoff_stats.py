"""Posterior trade-off between latent length and difficulty residual.

Prints, from the headline fit (fit_kink_robust_het_fameps.nc), the numbers the
writeup's section 2 cites:

  - per-task posterior correlation corr(log_L_i, eps_i), grouped by how well
    the task is timed (>=3 timed runs / 1-2 / estimate-only),
  - posterior sd of log_L, eps, and their sum (the difficulty) per group,
  - difficulty sd for tasks with >=90 attempts (the "factor of ~2.2" claim),
  - where the confident-cell failures (mean p > 0.95) concentrate.

Usage:
    uv run python scripts/tradeoff_stats.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import arviz as az
import numpy as np

from models.data_prep import load_model_data

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "processed" / "runs_filtered.parquet"
FIT = ROOT / "outputs" / "fit_kink_robust_het_fameps.nc"


def main() -> None:
    data = load_model_data(DATA_PATH)
    post = az.from_netcdf(FIT).posterior
    log_L = post["log_L"].stack(s=("chain", "draw")).values
    eps = post["eps"].stack(s=("chain", "draw")).values
    D = log_L + eps
    n_tasks = log_L.shape[0]

    timed = (~data.is_estimate) & (~data.is_censored)
    n_timed = np.bincount(data.task_idx_obs[timed], minlength=n_tasks)
    att = np.bincount(data.task_idx_irt, weights=data.n_attempts, minlength=n_tasks)
    corr = np.array([np.corrcoef(log_L[i], eps[i])[0, 1] for i in range(n_tasks)])

    groups = [
        (">=3 timed", n_timed >= 3),
        ("1-2 timed", (n_timed > 0) & (n_timed < 3)),
        ("estimate-only", n_timed == 0),
    ]
    for name, mask in groups:
        print(
            "%-14s n=%3d  corr(logL,eps) mean %+.2f median %+.2f | "
            "posterior sd: logL %.2f  eps %.2f  sum %.2f"
            % (
                name,
                mask.sum(),
                corr[mask].mean(),
                np.median(corr[mask]),
                log_L[mask].std(axis=1).mean(),
                eps[mask].std(axis=1).mean(),
                D[mask].std(axis=1).mean(),
            )
        )

    hi = att >= 90
    print(
        "difficulty sd (sum), attempts>=90: n=%d median %.2f (x%.1f) mean %.2f; "
        "all tasks median %.2f (x%.1f)"
        % (
            hi.sum(),
            np.median(D[hi].std(axis=1)),
            np.exp(np.median(D[hi].std(axis=1))),
            D[hi].std(axis=1).mean(),
            np.median(D.std(axis=1)),
            np.exp(np.median(D.std(axis=1))),
        )
    )

    # difficulty resolution by success-rate regime: the "factor of ~2.2" is
    # the average of a U -- sharp at the frontier, one-sided bounds outside it
    succ = np.bincount(data.task_idx_irt, weights=data.n_successes, minlength=n_tasks)
    rate = succ / np.maximum(att, 1)
    sdD = D.std(axis=1)
    regimes = [
        ("all-fail (rate=0)", rate == 0),
        ("frontier 5-95%", (rate >= 0.05) & (rate <= 0.95)),
        ("rate>95%", rate > 0.95),
    ]
    for name, mask in regimes:
        if mask.sum():
            med = np.median(sdD[mask])
            print(
                "difficulty sd, %-18s n=%3d median %.2f (x%.1f)"
                % (name, mask.sum(), med, np.exp(med))
            )

    # confident-cell failures: how concentrated?
    theta = post["theta"].stack(s=("chain", "draw")).values
    a = post["a"].stack(s=("chain", "draw")).values
    ti, mi = data.task_idx_irt, data.model_idx_irt
    p = 1.0 / (1.0 + np.exp(-(a[ti, :] * (theta[mi, :] - D[ti, :]))))
    easy = p.mean(axis=1) > 0.95
    nf = data.n_attempts - data.n_successes
    cells = np.where(easy & (nf > 0))[0]
    by_task = Counter()
    by_model = Counter()
    for c in cells:
        by_task[int(ti[c])] += int(nf[c])
        by_model[int(mi[c])] += int(nf[c])
    print(
        "confident cells (mean p>0.95) with failures: %d cells, %d failures; "
        "max per task %d, max per model %d"
        % (
            len(cells),
            int(nf[cells].sum()),
            max(by_task.values(), default=0),
            max(by_model.values(), default=0),
        )
    )


if __name__ == "__main__":
    main()
