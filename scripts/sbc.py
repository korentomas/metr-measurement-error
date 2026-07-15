"""Reduced-scale simulation-based calibration (SBC) for the
measurement-error time-horizon model, any shape / duration likelihood.

Defaults reproduce the original linear + Normal run; pass --shape kink
--robust --sigma-est-median 1.25 to calibrate the actual headline
configuration (the kink shape drives the stacked headline and its delta/t_k
and the Student-t nu were previously never rank-checked -- red-team #6).

Procedure (Talts et al. 2018, reduced):
  1. Build a synthetic-design ModelData at reduced scale (default 50 tasks,
     8 models -- vs 228/20 real) with an observation design that mimics the
     real data's sparsity: most tasks have 1-3 timed baseline runs, a
     fraction are estimate-only (one annotation), and every (model, task)
     cell has a small number of IRT attempts.
  2. Draw `n_reps` joint samples from the model's own prior (parameters +
     observed nodes) via pm.sample_prior_predictive, so the simulator is
     exactly the model -- no hand-rolled generative code to drift out of
     sync.
  3. Refit the model to each simulated dataset and record, for each scalar
     hyperparameter, the rank of the true (simulating) value among thinned
     posterior draws.
  4. If the model + sampler are calibrated, those ranks are uniform. Report
     per-parameter normalized-rank summaries, a KS test against U(0,1), and
     central 50%/90% interval coverage.

Usage:
    uv run python scripts/sbc.py --n-reps 25
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import arviz as az
import numpy as np
import pymc as pm
from scipy import stats

from models.data_prep import ModelData
from models.time_horizon_model import build_model

BASE_PARAMS = [
    "mu_L", "sigma_L", "sigma_base", "sigma_est",
    "sigma_a", "sigma_eps", "beta0", "beta1", "sigma_u",
]
# Shape-specific trend parameters, ranked in addition to BASE_PARAMS so SBC
# covers the actual headline configuration (the kink shape drives the stacked
# headline, and its delta/t_k were previously never rank-checked).
SHAPE_PARAMS = {
    "linear": [],
    "kink": ["delta", "t_k"],
    "superexp": ["beta2"],
    "logistic": ["h", "t0", "s"],
}


def make_design(
    n_tasks: int = 50,
    n_models: int = 8,
    frac_estimate: float = 0.2,
    n_attempts_per_cell: int = 8,
    seed: int = 0,
) -> ModelData:
    """Synthetic observation design (indices/covariates only; observed values
    are placeholders overwritten by prior-predictive draws)."""
    rng = np.random.default_rng(seed)

    n_est = int(round(frac_estimate * n_tasks))
    est_tasks = np.arange(n_tasks - n_est, n_tasks)

    # Baseline tasks get 1-3 timed runs (real data: most tasks 1-2, a few many).
    task_idx_base = []
    for t in range(n_tasks - n_est):
        k = rng.choice([1, 1, 2, 2, 3])
        task_idx_base += [t] * int(k)
    task_idx_base = np.array(task_idx_base, dtype=int)
    task_idx_obs = np.concatenate([task_idx_base, est_tasks])
    n_obs = len(task_idx_obs)
    is_estimate = np.zeros(n_obs, dtype=bool)
    is_estimate[len(task_idx_base):] = True

    # IRT: full model x task cross, fixed attempts per cell.
    task_idx_irt, model_idx_irt = np.meshgrid(
        np.arange(n_tasks), np.arange(n_models), indexing="ij"
    )
    task_idx_irt = task_idx_irt.ravel()
    model_idx_irt = model_idx_irt.ravel()
    n_cells = len(task_idx_irt)

    t_model = np.linspace(-1.9, 1.0, n_models)
    t_model -= t_model.mean()

    return ModelData(
        n_tasks=n_tasks,
        task_ids=[f"task_{i}" for i in range(n_tasks)],
        log_dur=np.zeros(n_obs),
        task_idx_obs=task_idx_obs,
        is_estimate=is_estimate,
        is_censored=np.zeros(n_obs, dtype=bool),
        censor_log_time=np.zeros(n_obs),
        n_models=n_models,
        model_names=[f"model_{i}" for i in range(n_models)],
        n_attempts=np.full(n_cells, n_attempts_per_cell, dtype=int),
        n_successes=np.zeros(n_cells, dtype=int),
        task_idx_irt=task_idx_irt,
        model_idx_irt=model_idx_irt,
        t_model=t_model,
        has_date=np.ones(n_models, dtype=bool),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-reps", type=int, default=25)
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument("--n-models", type=int, default=8)
    parser.add_argument("--shape", choices=list(SHAPE_PARAMS), default="linear")
    parser.add_argument("--robust", action="store_true",
                        help="Student-t baseline layer (adds nu to the ranked params).")
    parser.add_argument("--duration-dist", choices=["lognormal", "studentt", "weibull"], default=None)
    parser.add_argument("--sigma-est-median", type=float, default=1.25)
    parser.add_argument("--heteroscedastic", action="store_true",
                        help="Heteroscedastic measurement layer (adds gamma_sig to ranked params).")
    parser.add_argument("--tune", type=int, default=800)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--thin-to", type=int, default=100,
                        help="Thin posterior to this many draws for rank stats.")
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--out", default=str(Path(__file__).parent.parent / "outputs" / "sbc_results.npz"))
    args = parser.parse_args()

    duration_dist = args.duration_dist or ("studentt" if args.robust else "lognormal")
    params = list(BASE_PARAMS) + SHAPE_PARAMS[args.shape]
    if duration_dist == "studentt":
        params.append("nu")
    if args.heteroscedastic:
        params.append("gamma_sig")
    build_kw = dict(shape=args.shape, duration_dist=duration_dist,
                    sigma_est_median=args.sigma_est_median,
                    heteroscedastic=args.heteroscedastic)

    design = make_design(n_tasks=args.n_tasks, n_models=args.n_models, seed=args.seed)
    print(
        f"SBC design: {design.n_tasks} tasks, {design.n_models} models, "
        f"{len(design.log_dur)} timing obs ({design.is_estimate.sum()} estimate-only), "
        f"{len(design.n_attempts)} IRT cells x {design.n_attempts[0]} attempts"
    )
    print(f"SBC config: shape={args.shape}, duration_dist={duration_dist}, "
          f"sigma_est_median={args.sigma_est_median}; ranking {params}")

    # --- Prior-predictive simulation (exactly the model's own prior) ---
    sim_model = build_model(design, **build_kw)
    with sim_model:
        prior = pm.sample_prior_predictive(
            samples=args.n_reps, random_seed=args.seed
        )
    pp = prior.prior
    obs = prior.prior_predictive

    ranks = {p: [] for p in params}
    cover50 = {p: [] for p in params}
    cover90 = {p: [] for p in params}
    diags = []

    for r in range(args.n_reps):
        true = {p: float(pp[p].isel(chain=0, draw=r)) for p in params}

        rep = make_design(n_tasks=args.n_tasks, n_models=args.n_models, seed=args.seed)
        base_sim = obs["dur_base_obs"].isel(chain=0, draw=r).values
        est_sim = obs["dur_estimate"].isel(chain=0, draw=r).values
        rep.log_dur = np.concatenate([base_sim, est_sim])
        rep.n_successes = obs["successes"].isel(chain=0, draw=r).values.astype(int)

        t0 = time.time()
        model = build_model(rep, **build_kw)
        with model:
            try:
                idata = pm.sample(
                    tune=args.tune,
                    draws=args.draws,
                    chains=args.chains,
                    random_seed=args.seed + r,
                    nuts_sampler="nutpie",
                    target_accept=args.target_accept,
                    progressbar=False,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"rep {r}: sampling FAILED ({exc!r}); skipping")
                diags.append((r, np.nan, np.nan, time.time() - t0, "failed"))
                continue

        ndiv = int(idata.sample_stats["diverging"].sum())
        rhat = float(
            az.rhat(idata, var_names=params).to_array().max()
        )
        elapsed = time.time() - t0
        diags.append((r, ndiv, rhat, elapsed, "ok"))

        for p in params:
            draws = idata.posterior[p].values.ravel()
            # Thin to reduce autocorrelation in the rank statistic.
            step = max(1, len(draws) // args.thin_to)
            thinned = draws[::step]
            ranks[p].append((thinned < true[p]).mean())
            lo50, hi50 = np.quantile(draws, [0.25, 0.75])
            lo90, hi90 = np.quantile(draws, [0.05, 0.95])
            cover50[p].append(lo50 <= true[p] <= hi50)
            cover90[p].append(lo90 <= true[p] <= hi90)

        print(
            f"rep {r:2d}: {elapsed:5.1f}s, divergences={ndiv}, max R-hat={rhat:.3f}, "
            + ", ".join(f"{p}={ranks[p][-1]:.2f}" for p in ("beta1", "sigma_eps", "sigma_base"))
        )

    # --- Report ---
    n_ok = sum(1 for d in diags if d[4] == "ok")
    print(f"\n=== SBC summary: {n_ok}/{args.n_reps} reps fit successfully ===")
    print(f"{'param':12s} {'mean rank':>9s} {'KS p':>7s} {'cov50':>6s} {'cov90':>6s}")
    for p in params:
        u = np.array(ranks[p])
        if len(u) == 0:
            continue
        ks = stats.kstest(u, "uniform")
        print(
            f"{p:12s} {u.mean():9.3f} {ks.pvalue:7.3f} "
            f"{np.mean(cover50[p]):6.2f} {np.mean(cover90[p]):6.2f}"
        )
    print("\n(mean rank ~0.5, KS p not tiny, cov50 ~0.50, cov90 ~0.90 = calibrated)")
    bad = [d for d in diags if d[4] == "ok" and (d[1] > 0 or d[2] > 1.05)]
    if bad:
        print(f"reps with divergences or R-hat>1.05: {[(d[0], d[1], round(d[2],3)) for d in bad]}")

    np.savez(
        args.out,
        **{f"rank_{p}": np.array(ranks[p]) for p in params},
        diags=np.array([(d[0], d[1], d[2], d[3]) for d in diags if d[4] == "ok"]),
    )
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
