"""Before/after comparison of the Normal vs Student-t measurement layer.

Reports, for the Normal and robust fits of the same shape:
  - sigma_base (recall: under Student-t this is the *scale*, so also report
    the implied sd sqrt(nu/(nu-2))*sigma for nu>2, and nu itself);
  - posterior log_L (latent true task length) for the tasks carrying the
    largest within-task outliers, vs the task's median observed log
    duration -- does the outlier's pull on log_L shrink under the t?
  - headline doubling time under both, to confirm the trend inference is
    not an artifact of the outliers.

Usage:
    uv run python scripts/compare_robust.py \
        --normal outputs/fit_linear.nc --robust outputs/fit_linear_robust.nc
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import arviz as az
import numpy as np

from models.data_prep import load_model_data

DEFAULT_DATA = str(Path(__file__).parent.parent / "data" / "processed" / "runs_filtered.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", required=True)
    parser.add_argument("--robust", required=True)
    parser.add_argument("--data", default=DEFAULT_DATA)
    args = parser.parse_args()

    data = load_model_data(args.data)
    id_n = az.from_netcdf(args.normal)
    id_r = az.from_netcdf(args.robust)

    def q(x, qs=(0.025, 0.5, 0.975)):
        return np.quantile(x, qs)

    sb_n = id_n.posterior["sigma_base"].values.ravel()
    sb_r = id_r.posterior["sigma_base"].values.ravel()
    nu = id_r.posterior["nu"].values.ravel()
    sd_r = np.where(nu > 2, sb_r * np.sqrt(nu / (nu - 2)), np.nan)

    print("=== Measurement noise ===")
    lo, med, hi = q(sb_n)
    print(f"Normal    sigma_base (= sd):        {med:.3f} [{lo:.3f}, {hi:.3f}]")
    lo, med, hi = q(sb_r)
    print(f"Student-t sigma_base (scale):       {med:.3f} [{lo:.3f}, {hi:.3f}]")
    lo, med, hi = q(nu)
    print(f"Student-t nu:                       {med:.1f} [{lo:.1f}, {hi:.1f}]")
    lo, med, hi = q(sd_r[np.isfinite(sd_r)])
    print(f"Student-t implied sd (nu>2 draws):  {med:.3f} [{lo:.3f}, {hi:.3f}]")

    # --- Outlier tasks: largest |obs - task median| among multi-obs tasks ---
    base = ~data.is_estimate
    ld, ti = data.log_dur[base], data.task_idx_obs[base]
    by = defaultdict(list)
    for t, v in zip(ti, ld):
        by[t].append(v)
    devs = []
    for t, vs in by.items():
        if len(vs) < 2:
            continue
        vs = np.array(vs)
        med = np.median(vs)
        devs.append((np.max(np.abs(vs - med)), t, med, len(vs)))
    devs.sort(reverse=True)

    print("\n=== log_L posterior at the 6 worst-outlier tasks ===")
    print(f"{'task':50s} {'n':>3s} {'obs med':>8s} {'Normal log_L':>16s} {'Student-t log_L':>16s}")
    logL_n = id_n.posterior["log_L"]
    logL_r = id_r.posterior["log_L"]
    for dev, t, obs_med, n in devs[:6]:
        ln = logL_n.isel(log_L_dim_0=t).values.ravel() if "log_L_dim_0" in logL_n.dims else logL_n.sel(task=data.task_ids[t]).values.ravel()
        lr = logL_r.sel(task=data.task_ids[t]).values.ravel()
        print(
            f"{data.task_ids[t]:50s} {n:3d} {obs_med:8.2f} "
            f"{np.median(ln):8.2f} ({np.median(ln)-obs_med:+.2f}) "
            f"{np.median(lr):8.2f} ({np.median(lr)-obs_med:+.2f})"
        )
    print("(numbers in parens: pull of the model away from the task's observed median)")

    print("\n=== Doubling time ln(2)/slope_now, months ===")
    for name, idata in (("Normal", id_n), ("Student-t", id_r)):
        dt = np.log(2) / idata.posterior["slope_now"].values.ravel() * 12.0
        lo, med, hi = q(dt)
        print(f"{name:10s} {med:.1f} [{lo:.1f}, {hi:.1f}]")

    print("\n=== sigma_eps / sigma_est (context) ===")
    for name, idata in (("Normal", id_n), ("Student-t", id_r)):
        se = idata.posterior["sigma_eps"].values.ravel()
        st = idata.posterior["sigma_est"].values.ravel()
        print(
            f"{name:10s} sigma_eps {np.median(se):.2f} [{np.quantile(se,0.025):.2f}, {np.quantile(se,0.975):.2f}]"
            f"   sigma_est {np.median(st):.2f} [{np.quantile(st,0.025):.2f}, {np.quantile(st,0.975):.2f}]"
        )


if __name__ == "__main__":
    main()
