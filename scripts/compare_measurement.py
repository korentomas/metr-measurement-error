"""Compare measurement-layer variants against the baseline robust fit.

Prints, for each fitted InferenceData passed, the headline doubling time, the
measurement/difficulty scales, sampler health, and (for the length-scale
likelihoods that share the uncensored baseline observations) PSIS-LOO on
dur_base_obs so heteroscedastic-vs-homoscedastic can be compared on the same
data. The human-failures variant adds censored observations, so its
dur_base_obs LOO is over a different obs set and is reported but flagged
not-comparable; that variant is judged on parameter/horizon shift instead.

Usage:
    uv run python scripts/compare_measurement.py \
        --fits outputs/fit_linear_robust.nc outputs/fit_linear_robust_het.nc \
               outputs/fit_linear_robust_hf.nc outputs/fit_linear_robust_het_hf.nc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import arviz as az
import numpy as np


def q(x, lo=0.025, hi=0.975):
    return np.median(x), np.quantile(x, lo), np.quantile(x, hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", nargs="+", required=True)
    args = parser.parse_args()

    print(f"{'variant':28s} {'doubling (mo)':>18s} {'sigma_eps':>16s} "
          f"{'sigma_base':>11s} {'nu':>6s} {'gamma_sig':>16s} {'div':>4s} {'rhat':>5s}")
    idatas = {}
    for f in args.fits:
        name = Path(f).stem.replace("fit_linear_robust", "base").replace("fit_", "")
        idata = az.from_netcdf(f)
        idatas[name] = idata
        post = idata.posterior
        dt = np.log(2) / post["slope_now"].values.ravel() * 12.0
        se = post["sigma_eps"].values.ravel()
        sb = post["sigma_base"].values.ravel() if "sigma_base" in post else np.array([np.nan])
        nu = post["nu"].values.ravel() if "nu" in post else np.array([np.nan])
        gs = post["gamma_sig"].values.ravel() if "gamma_sig" in post else None
        ndiv = int(idata.sample_stats["diverging"].sum()) if "sample_stats" in idata.groups() else -1
        rhat = float(az.rhat(idata).to_array().max())
        dtm, dtl, dth = q(dt); sem, sel, seh = q(se)
        gs_str = f"{np.median(gs):+.3f}[{np.quantile(gs,.025):+.2f},{np.quantile(gs,.975):+.2f}]" if gs is not None else "--"
        print(f"{name:28s} {dtm:5.1f} [{dtl:4.1f},{dth:5.1f}]   "
              f"{sem:4.2f} [{sel:4.2f},{seh:4.2f}] {np.median(sb):>11.3f} "
              f"{np.median(nu):>6.2f} {gs_str:>16s} {ndiv:>4d} {rhat:>5.2f}")

    # LOO on dur_base_obs where present (measurement-fit comparison)
    print("\n=== PSIS-LOO on dur_base_obs (log scale; comparable only across")
    print("    variants with the SAME baseline obs set: base vs het) ===")
    loos = {}
    for name, idata in idatas.items():
        if "log_likelihood" in idata.groups() and "dur_base_obs" in idata.log_likelihood:
            n_obs = idata.log_likelihood["dur_base_obs"].sizes
            obs_dim = [d for d in idata.log_likelihood["dur_base_obs"].dims if d not in ("chain", "draw")][0]
            loo = az.loo(idata, var_name="dur_base_obs")
            loos[name] = (loo, idata.log_likelihood["dur_base_obs"].sizes[obs_dim])
            nbad = int((loo.pareto_k > 0.7).sum())
            print(f"  {name:26s} elpd_loo={loo.elpd_loo:8.1f} +- {loo.se:4.1f}  "
                  f"n_obs={loos[name][1]:4d}  pareto_k>0.7: {nbad}")
    # direct base-vs-het comparison if both have 525 obs
    comparable = {n: l for n, (l, nobs) in loos.items() if nobs == 525}
    if len(comparable) >= 2:
        print("\n  az.compare on the 525-obs variants:")
        cmp = az.compare({n: idatas[n] for n in comparable}, var_name="dur_base_obs")
        print(cmp[["rank", "elpd_loo", "elpd_diff", "dse", "weight"]].to_string())


if __name__ == "__main__":
    main()
