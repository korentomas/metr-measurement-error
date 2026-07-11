"""PSIS-LOO comparison of the three baseline-duration likelihoods
(log-normal, Student-t on log scale, Weibull on the raw scale) on the 525
per-run wall-clock duration observations -- the term the variants actually
differ on (the IRT counts and estimate-annotation layers are shared).

The subtlety this script exists for: the log-normal and Student-t
likelihoods are defined on log(dur), the Weibull on dur itself. Pointwise
log-likelihoods on different scales of the same data are NOT comparable --
elpd is only invariant if every model is scored on the same measurement
scale. So the log-scale models' pointwise log-likelihoods get the change-
of-variables Jacobian applied (log p_dur(d) = log p_logdur(log d) - log d)
before az.loo / az.compare, putting all three on the duration scale.

Each fit must carry (or be able to recompute) the pointwise log-likelihood
of `dur_base_obs`; if it is missing, it is recomputed here by rebuilding
the matching model (pass --duration-dists in the same order as --fits).

Usage:
    uv run python scripts/compare_duration_dists.py \
        --fits outputs/fit_linear.nc outputs/fit_linear_robust.nc outputs/fit_linear_weibull.nc \
        --duration-dists lognormal studentt weibull \
        --sigma-est-medians 0.8 0.8 0.8
"""

from __future__ import annotations

import argparse

import arviz as az
import numpy as np
import pymc as pm
import xarray as xr

from metr_measurement_error.data_prep import load_model_data
from metr_measurement_error.model import build_model
from metr_measurement_error.paths import PROCESSED_DATA

DEFAULT_DATA = str(PROCESSED_DATA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", nargs="+", required=True)
    parser.add_argument("--duration-dists", nargs="+", required=True,
                        choices=["lognormal", "studentt", "weibull"])
    parser.add_argument("--sigma-est-medians", nargs="+", type=float, default=None,
                        help="Prior median of sigma_est used in each fit (default 0.8 each).")
    parser.add_argument("--data", default=DEFAULT_DATA)
    args = parser.parse_args()
    if len(args.fits) != len(args.duration_dists):
        raise ValueError("--fits and --duration-dists must have the same length")
    medians = args.sigma_est_medians or [0.8] * len(args.fits)

    data = load_model_data(args.data)
    base_mask = ~data.is_estimate & ~data.is_censored
    log_dur_base = data.log_dur[base_mask]

    loos = {}
    idatas = {}
    for path, dist, med in zip(args.fits, args.duration_dists, medians):
        name = dist
        idata = az.from_netcdf(path)
        if ("log_likelihood" not in idata.groups()
                or "dur_base_obs" not in idata.log_likelihood):
            print(f"[{name}] recomputing pointwise log-likelihood of dur_base_obs ...")
            model = build_model(data, shape="linear", duration_dist=dist,
                                sigma_est_median=med)
            with model:
                ll_ds = pm.compute_log_likelihood(
                    idata, var_names=["dur_base_obs"], extend_inferencedata=False
                )
            if "log_likelihood" in idata.groups():
                idata.log_likelihood["dur_base_obs"] = ll_ds["dur_base_obs"]
            else:
                idata.add_groups({"log_likelihood": ll_ds})

        ll = idata.log_likelihood["dur_base_obs"]
        obs_dim = [d for d in ll.dims if d not in ("chain", "draw")][0]
        if dist in ("lognormal", "studentt"):
            # Jacobian: score on the duration scale, not the log scale.
            jac = xr.DataArray(log_dur_base, dims=[obs_dim])
            ll = ll - jac
        ll_id = az.InferenceData(
            posterior=idata.posterior,
            log_likelihood=xr.Dataset({"dur_base_obs": ll}),
        )
        loo = az.loo(ll_id, var_name="dur_base_obs")
        loos[name] = loo
        idatas[name] = idata
        n_bad = int((loo.pareto_k > 0.7).sum())
        print(f"{name:10s} elpd_loo={loo.elpd_loo:9.1f} +- {loo.se:5.1f}   "
              f"p_loo={loo.p_loo:6.1f}   pareto_k>0.7: {n_bad}/{len(loo.pareto_k)}")

    cmp = az.compare(loos)
    print("\n=== az.compare (duration-scale PSIS-LOO, 525 baseline runs) ===")
    print(cmp[["rank", "elpd_loo", "p_loo", "elpd_diff", "weight", "se", "dse"]])

    print("\n=== Headline posteriors per duration likelihood ===")
    for name, idata in idatas.items():
        post = idata.posterior
        dt = np.log(2) / post["slope_now"].values.ravel() * 12.0
        se = post["sigma_eps"].values.ravel()
        line = (f"{name:10s} doubling {np.median(dt):.1f} "
                f"[{np.quantile(dt, 0.025):.1f}, {np.quantile(dt, 0.975):.1f}] mo   "
                f"sigma_eps {np.median(se):.2f} "
                f"[{np.quantile(se, 0.025):.2f}, {np.quantile(se, 0.975):.2f}]")
        if "alpha_w" in post:
            a = post["alpha_w"].values.ravel()
            line += (f"   alpha_w {np.median(a):.2f} "
                     f"[{np.quantile(a, 0.025):.2f}, {np.quantile(a, 0.975):.2f}]"
                     f" (implied log-sd {1.2825/np.median(a):.2f})")
        if "nu" in post:
            nu = post["nu"].values.ravel()
            line += f"   nu {np.median(nu):.1f}"
        if "sigma_base" in post:
            sb = post["sigma_base"].values.ravel()
            line += f"   sigma_base {np.median(sb):.2f}"
        print(line)


if __name__ == "__main__":
    main()
