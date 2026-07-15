"""Fit the Bayesian measurement-error time-horizon model.

Runs any of the four trend shapes (linear/kink/superexp/logistic), the
Normal/Student-t/Weibull duration likelihoods, and the full vs. SOTA-only
model sets; this is what produced every number in docs/results.md. Defaults
are a tiny smoke-test config (200 tune / 200 draws / 2 chains) so a bare
invocation finishes in seconds; pass --tune/--draws/--chains for a real fit.

Usage:
    uv run python scripts/fit_model.py
    uv run python scripts/fit_model.py --tune 2000 --draws 2000 --chains 4 \
        --shape kink --robust --log-likelihood
    uv run python scripts/fit_model.py --sampler pymc   # force fallback NUTS
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as `uv run python scripts/fit_model.py` from the repo root
# without needing PYTHONPATH set manually.
sys.path.insert(0, str(Path(__file__).parent.parent))

import arviz as az
import pymc as pm

from models.data_prep import load_model_data
from models.time_horizon_model import build_model

DEFAULT_DATA = str(Path(__file__).parent.parent / "data" / "processed" / "runs_filtered.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--tune", type=int, default=200)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--target-accept", type=float, default=0.8)
    parser.add_argument(
        "--sampler",
        choices=["nutpie", "pymc"],
        default="nutpie",
        help="Sampler backend. Falls back to PyMC's default NUTS on nutpie failure.",
    )
    parser.add_argument(
        "--shape",
        choices=["linear", "kink", "superexp", "logistic"],
        default="linear",
        help="Ability-trend mean function.",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Use a Student-t (estimated nu) likelihood for baseline timed runs.",
    )
    parser.add_argument(
        "--duration-dist",
        choices=["lognormal", "studentt", "weibull"],
        default=None,
        help="Baseline-run duration likelihood. Overrides --robust; default "
        "lognormal (or studentt if --robust).",
    )
    parser.add_argument(
        "--sigma-est-median",
        type=float,
        default=1.25,
        help="Prior median of sigma_est (LogNormal, log-sd 0.5). Default 1.25 "
        "(Barry-calibrated); pass 0.8 to reproduce the original prior.",
    )
    parser.add_argument(
        "--sota-only",
        action="store_true",
        help="Restrict to models that were SOTA at their release date "
        "(METR's running-frontier definition; 14 of the 20 models on the "
        "current snapshot). Default: all models.",
    )
    parser.add_argument(
        "--cut-estimate-feedback",
        action="store_true",
        help="Cut-model variant: for the estimate-only tasks, the IRT layer "
        "uses the raw annotation as a fixed constant instead of the latent "
        "log_L, so success/failure data cannot move those tasks' inferred "
        "lengths. Baseline-informed tasks are unchanged.",
    )
    parser.add_argument(
        "--heteroscedastic",
        action="store_true",
        help="Model baseline measurement noise as length-dependent: "
        "sigma_base_i = sigma_base * exp(gamma_sig * (log_L_i - mu_L)). "
        "gamma_sig=0 nests the homoscedastic model.",
    )
    parser.add_argument(
        "--include-human-failures",
        action="store_true",
        help="Add failed baseline human runs as right-censored duration "
        "observations (survivorship correction on log_L).",
    )
    parser.add_argument(
        "--log-likelihood",
        action="store_true",
        help="Compute pointwise log-likelihood of `successes` (stacking) and "
        "`dur_base_obs` (duration-distribution comparison).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Where to save the InferenceData (NetCDF). Default: outputs/fit_<shape>[_robust].nc",
    )
    args = parser.parse_args()
    duration_dist = args.duration_dist or ("studentt" if args.robust else "lognormal")
    if args.out is None:
        suffix = {"lognormal": "", "studentt": "_robust", "weibull": "_weibull"}[duration_dist]
        if args.sota_only:
            suffix += "_sota"
        if args.cut_estimate_feedback:
            suffix += "_cut"
        if args.heteroscedastic:
            suffix += "_het"
        if args.include_human_failures:
            suffix += "_hf"
        args.out = str(
            Path(__file__).parent.parent / "outputs" / f"fit_{args.shape}{suffix}.nc"
        )

    print(f"Loading data from {args.data} ...")
    data = load_model_data(
        args.data,
        sota_only=args.sota_only,
        include_human_failures=args.include_human_failures,
    )
    print(
        f"  {data.n_tasks} tasks, {len(data.log_dur)} human-timing obs "
        f"({data.is_estimate.sum()} estimate-only), {data.n_models} models, "
        f"{len(data.n_attempts)} (model,task) IRT count rows"
    )

    model = build_model(
        data,
        shape=args.shape,
        duration_dist=duration_dist,
        sigma_est_median=args.sigma_est_median,
        cut_estimate_feedback=args.cut_estimate_feedback,
        heteroscedastic=args.heteroscedastic,
    )
    print(
        f"Model built (shape={args.shape}, duration_dist={duration_dist}, "
        f"sigma_est prior median={args.sigma_est_median}, "
        f"cut_estimate_feedback={args.cut_estimate_feedback}). "
        f"Free RVs: {[v.name for v in model.free_RVs]}"
    )

    t0 = time.time()
    used_sampler = args.sampler
    try:
        if args.sampler == "nutpie":
            import nutpie  # noqa: F401

            with model:
                idata = pm.sample(
                    tune=args.tune,
                    draws=args.draws,
                    chains=args.chains,
                    random_seed=args.seed,
                    nuts_sampler="nutpie",
                    target_accept=args.target_accept,
                )
        else:
            raise RuntimeError("forced pymc fallback")
    except Exception as exc:  # noqa: BLE001 -- intentionally broad: this is a sampler fallback
        print(f"nutpie sampling failed or was skipped ({exc!r}); falling back to PyMC NUTS.")
        used_sampler = "pymc"
        with model:
            idata = pm.sample(
                tune=args.tune,
                draws=args.draws,
                chains=args.chains,
                random_seed=args.seed,
                nuts_sampler="pymc",
            )

    elapsed = time.time() - t0
    print(f"Sampling with backend={used_sampler} took {elapsed:.1f}s")

    base_vars = ["mu_L", "sigma_L", "sigma_est", "sigma_a", "sigma_eps", "beta0", "sigma_u", "slope_now"]
    if duration_dist in ("lognormal", "studentt"):
        base_vars.insert(2, "sigma_base")
    if duration_dist == "weibull":
        base_vars.append("alpha_w")
    shape_vars = {
        "linear": ["beta1"],
        "kink": ["beta1", "delta", "t_k"],
        "superexp": ["beta1", "beta2"],
        "logistic": ["h", "t0", "s"],
    }[args.shape]
    if duration_dist == "studentt":
        base_vars.append("nu")
    summary = az.summary(idata, var_names=base_vars + shape_vars)
    print(summary)

    n_divergences = int(idata.sample_stats["diverging"].sum())
    print(f"Divergences: {n_divergences}")
    print(f"Max R-hat: {float(az.rhat(idata).to_array().max()):.4f}")

    if args.log_likelihood:
        print("Computing pointwise log-likelihood for `successes` and `dur_base_obs` ...")
        with model:
            pm.compute_log_likelihood(idata, var_names=["successes", "dur_base_obs"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(out_path)
    print(f"Saved InferenceData to {out_path}")


if __name__ == "__main__":
    main()
