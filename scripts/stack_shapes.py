"""Bayesian stacking (Yao et al. 2018) across the four ability-trend shapes.

Loads the fitted InferenceData for each shape (must have been fit with
--log-likelihood so the pointwise log-likelihood of the IRT `successes` is
stored -- the measurement layer is identical across shapes, so `successes`
is the only likelihood term the shapes can differ on), computes PSIS-LOO
per shape, and combines them with arviz's stacking implementation
(az.compare(..., method="stacking"), which solves the Yao et al. log-score
stacking optimization -- not hand-rolled).

Also reports, per shape and under the stacked mixture, the *current*
doubling time ln(2)/slope_now (slope of the trend at the latest dated
model's release date), in months. The stacked posterior is the
weight-proportional mixture of the per-shape posteriors.

Usage:
    uv run python scripts/stack_shapes.py \
        --fits outputs/fit_linear.nc outputs/fit_kink.nc \
               outputs/fit_superexp.nc outputs/fit_logistic.nc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    idatas = {}
    for f in args.fits:
        name = Path(f).stem.replace("fit_", "")
        idatas[name] = az.from_netcdf(f)
        if "log_likelihood" not in idatas[name].groups():
            raise ValueError(f"{f} has no log_likelihood group; refit with --log-likelihood")

    print("=== PSIS-LOO per shape ===")
    loos = {}
    for name, idata in idatas.items():
        loo = az.loo(idata, var_name="successes")
        loos[name] = loo
        n_bad = int((loo.pareto_k > 0.7).sum())
        print(
            f"{name:10s} elpd_loo={loo.elpd_loo:9.1f} +- {loo.se:5.1f}   "
            f"p_loo={loo.p_loo:6.1f}   pareto_k>0.7: {n_bad}/{len(loo.pareto_k)}"
        )

    cmp = az.compare(loos, method="stacking")
    print("\n=== az.compare (method='stacking') ===")
    print(cmp[["rank", "elpd_loo", "p_loo", "elpd_diff", "weight", "se", "dse"]])

    weights = cmp["weight"].to_dict()

    # --- Doubling time per shape and stacked ---
    print("\n=== Current doubling time ln(2)/slope_now, months ===")
    rng = np.random.default_rng(args.seed)
    mix_draws = []
    n_mix = 20000
    for name, idata in idatas.items():
        s = idata.posterior["slope_now"].values.ravel()
        dt = np.log(2) / s * 12.0
        lo, med, hi = np.quantile(dt, [0.025, 0.5, 0.975])
        print(f"{name:10s} weight={weights.get(name, 0):.3f}  median {med:5.1f}  [{lo:.1f}, {hi:.1f}]")
        k = int(round(weights.get(name, 0) * n_mix))
        if k > 0:
            mix_draws.append(rng.choice(dt, size=k, replace=True))
    mix = np.concatenate(mix_draws)
    lo, med, hi = np.quantile(mix, [0.025, 0.5, 0.975])
    print(f"{'STACKED':10s} {'':13s} median {med:5.1f}  [{lo:.1f}, {hi:.1f}]")


if __name__ == "__main__":
    main()
