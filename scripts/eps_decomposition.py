"""Decompose residual task difficulty (eps) into between-family and
within-family components, and compare to the flat-eps fit.

The flat model reports a single sigma_eps (~2.2 log-min, an ~8x spread of
task difficulty at fixed length) and treats it as irreducible. The
family-structured fit (eps_structure='family') splits it into
    sigma_eps_fam    -- how much task *families* differ in difficulty beyond
                        what their length predicts (predictable structure)
    sigma_eps_within -- residual per-task difficulty within a family
                        (genuinely un-modelled)
and eps_between_frac = the share of residual-difficulty variance that is
between-family. A large between share means some of the "8x" is structure a
covariate could capture, not noise.

Reports the decomposition, the headline doubling time (does structuring eps
move it?), the most systematically hard/easy families, and -- if both fits
carry pointwise successes log-likelihood -- a PSIS-LOO comparison of the IRT
fit (family vs flat).

Usage:
    uv run python scripts/eps_decomposition.py \
        --family outputs/fit_linear_robust_fameps.nc \
        --flat outputs/fit_linear_robust.nc
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


def q(x):
    return np.median(x), np.quantile(x, 0.025), np.quantile(x, 0.975)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, help="fit with eps_structure=family")
    parser.add_argument("--flat", default=None, help="flat-eps fit for comparison")
    parser.add_argument("--data", default=DEFAULT_DATA)
    args = parser.parse_args()

    data = load_model_data(args.data)
    fam = az.from_netcdf(args.family)
    post = fam.posterior

    sf = post["sigma_eps_fam"].values.ravel()
    sw = post["sigma_eps_within"].values.ravel()
    frac = post["eps_between_frac"].values.ravel()
    se = post["sigma_eps"].values.ravel()
    dt = np.log(2) / post["slope_now"].values.ravel() * 12.0

    print(f"family-structured fit: {args.family}\n")
    print("=== residual-difficulty variance decomposition ===")
    m, lo, hi = q(sf); print(f"  sigma_eps_fam    (between-family): {m:.2f} [{lo:.2f}, {hi:.2f}]")
    m, lo, hi = q(sw); print(f"  sigma_eps_within (within-family): {m:.2f} [{lo:.2f}, {hi:.2f}]")
    m, lo, hi = q(se); print(f"  sigma_eps        (total realized): {m:.2f} [{lo:.2f}, {hi:.2f}]")
    m, lo, hi = q(frac)
    print(f"  between-family variance share:    {m:.2f} [{lo:.2f}, {hi:.2f}]  "
          f"(=> ~{m*100:.0f}% of residual difficulty is predictable family structure)")
    m, lo, hi = q(dt); print(f"\n  doubling time: {m:.1f} [{lo:.1f}, {hi:.1f}] months")

    if args.flat:
        flat = az.from_netcdf(args.flat)
        se0 = flat.posterior["sigma_eps"].values.ravel()
        dt0 = np.log(2) / flat.posterior["slope_now"].values.ravel() * 12.0
        m, lo, hi = q(se0); print(f"\n  flat-eps sigma_eps:  {m:.2f} [{lo:.2f}, {hi:.2f}]")
        m, lo, hi = q(dt0); print(f"  flat-eps doubling:   {m:.1f} [{lo:.1f}, {hi:.1f}] months")

    # --- most systematically hard / easy families ---
    fam_eff = (post["fam_raw"] * post["sigma_eps_fam"]).stack(s=("chain", "draw")).values  # (family, S)
    fam_med = np.median(fam_eff, axis=1)
    names = data.family_names
    # family sizes
    sizes = np.bincount(data.task_family, minlength=len(names))
    order = np.argsort(fam_med)
    print("\n=== most systematically HARDER-than-length families (eps_fam > 0) ===")
    print(f"  {'family':28s} {'n':>3s} {'eps_fam (log-min)':>18s}  ~x-harder")
    for i in order[::-1][:8]:
        if sizes[i] < 2:
            continue
        lo, hi = np.quantile(fam_eff[i], [0.025, 0.975])
        print(f"  {names[i]:28s} {sizes[i]:>3d} {fam_med[i]:>+9.2f} [{lo:+.2f},{hi:+.2f}]  {np.exp(fam_med[i]):.1f}x")
    print("\n=== most systematically EASIER-than-length families (eps_fam < 0) ===")
    for i in order[:8]:
        if sizes[i] < 2:
            continue
        lo, hi = np.quantile(fam_eff[i], [0.025, 0.975])
        print(f"  {names[i]:28s} {sizes[i]:>3d} {fam_med[i]:>+9.2f} [{lo:+.2f},{hi:+.2f}]  {np.exp(fam_med[i]):.2f}x")

    # --- LOO on successes (IRT fit), family vs flat ---
    if args.flat:
        try:
            loos = {}
            for name, idata in (("family", fam), ("flat", flat)):
                if "log_likelihood" in idata.groups() and "successes" in idata.log_likelihood:
                    loos[name] = az.loo(idata, var_name="successes")
            if len(loos) == 2:
                print("\n=== PSIS-LOO on successes (IRT fit): family vs flat ===")
                for name, loo in loos.items():
                    print(f"  {name:8s} elpd_loo={loo.elpd_loo:9.1f} +- {loo.se:5.1f}")
                cmp = az.compare({k: (fam if k == "family" else flat) for k in loos}, var_name="successes")
                print(cmp[["rank", "elpd_loo", "elpd_diff", "dse", "weight"]].to_string())
        except Exception as exc:  # noqa: BLE001
            print(f"\n(LOO comparison skipped: {exc!r})")


if __name__ == "__main__":
    main()
