"""Diagnostic for IRT -> log_L feedback on estimate-only tasks.

In the joint model, log_L_i for an estimate-only task is informed by two
sources: its single annotation (measurement layer) and the model
success/failure pattern (IRT layer). Alexander Barry's warning: the second
channel is circular. If success outcomes are allowed to move a task's
inferred length, the model can "explain away" surprising successes/failures
by re-dating the task, and the trend is then partly fit to lengths that the
outcomes themselves chose.

This script measures how much of that is happening in a saved fit. For each
estimate-only task it computes

    shift_i = posterior mean of log_L_i  -  log(annotation_i)

and reports the distribution of shifts and their correlation with the
task's pooled model success rate. A negative correlation means tasks that
models succeed on get pulled shorter and tasks models fail on get pulled
longer, which is the feedback direction operating. The magnitude of the
shifts relative to sigma_est says how strong the pull is; whether it
matters for the headline is answered by the cut-model refit
(--cut-estimate-feedback in scripts/fit_model.py), not by this script.

Usage:
    uv run python scripts/estimate_feedback_diagnostic.py --fit outputs/fit_kink_robust.nc
"""

from __future__ import annotations

import argparse
import sys
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
    parser.add_argument(
        "--sota-only",
        action="store_true",
        help="Load the SOTA-only restricted data (must match how the fit was run).",
    )
    args = parser.parse_args()

    data = load_model_data(args.data, sota_only=args.sota_only)
    idata = az.from_netcdf(args.fit)
    post = idata.posterior

    log_L_mean = post["log_L"].mean(dim=("chain", "draw")).values  # (n_tasks,)

    # Estimate-only observations: one annotation per task.
    est_obs = np.where(data.is_estimate)[0]
    est_task_idx = data.task_idx_obs[est_obs]
    annotation = data.log_dur[est_obs]

    shift = log_L_mean[est_task_idx] - annotation

    # Pooled success rate per estimate-only task, across all models.
    n_att = np.zeros(data.n_tasks)
    n_suc = np.zeros(data.n_tasks)
    np.add.at(n_att, data.task_idx_irt, data.n_attempts)
    np.add.at(n_suc, data.task_idx_irt, data.n_successes)
    success_rate = n_suc[est_task_idx] / n_att[est_task_idx]

    corr = np.corrcoef(shift, success_rate)[0, 1]
    sigma_est = post["sigma_est"].values.ravel()

    print(f"fit: {args.fit}")
    print(f"estimate-only tasks: {len(est_task_idx)}")
    print(f"shift = posterior-mean log_L - log(annotation), log-minutes")
    print(f"  mean shift:            {shift.mean():+.3f}")
    print(f"  sd of shifts:          {shift.std(ddof=1):.3f}")
    print(f"  min / max shift:       {shift.min():+.3f} / {shift.max():+.3f}")
    print(f"  corr(shift, pooled success rate): {corr:+.3f}")
    print(f"  sigma_est posterior mean:         {sigma_est.mean():.3f}")
    print()
    print("Reading: a near-zero mean shift means no aggregate bias; a negative")
    print("correlation with success rate is the circular-feedback direction")
    print("(easy-for-models tasks pulled shorter, hard ones pulled longer).")
    print("Shifts bounded well inside sigma_est mean the pull is weak per task;")
    print("the cut-model refit (--cut-estimate-feedback) quantifies its effect")
    print("on the headline.")


if __name__ == "__main__":
    main()
