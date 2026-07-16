"""SIMEX on the Bayesian model, to compare against Barry's SIMEX directly.

Barry's note applies SIMEX (add noise, refit, extrapolate to zero noise) to
METR's frequentist model and finds the frontier 50% horizon drops 25-40% once
the measurement noise is removed. The mechanism is that in METR's model
difficulty IS task length, so shrinking noisy (over-estimated) long tasks
pulls the horizon down.

This runs the same SIMEX ladder on the Bayesian measurement-error model, which
differs in one structural way: it carries a per-task difficulty residual eps
(Moss's unexplained difficulty), so the frontier horizon exp(theta) lives on
the difficulty scale (log_L + eps) that the cross-model success data pins
directly -- not on the length scale. The prediction is therefore that this
model's horizon is nearly FLAT in added length-noise: eps absorbs the length
perturbation, leaving difficulty (hence theta) unmoved. If so, the Bayesian
model's implied noise-correction is small precisely because it identifies
difficulty separately from length, and Barry's larger number is a property of
the difficulty=length model.

For each lambda we add independent Gaussian noise of sd sqrt(lambda)*sigma_i
to every log-duration observation (sigma_i = Barry's per-attempt calibration:
0.78 baseline, 1.05 estimate), refit, and record the doubling time and the
frontier model's 50% horizon exp(theta). Then fit log(TH(lambda)/TH(0)) =
b*lambda and extrapolate to lambda=-1 (Barry's noise-removed point).

Usage:
    uv run python scripts/simex.py --lambdas 0 0.5 1 2 --tune 1000 --draws 1000 --chains 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import arviz as az
import numpy as np
import pymc as pm

from models.data_prep import load_model_data
from models.time_horizon_model import build_model

DATA = str(Path(__file__).parent.parent / "data" / "processed" / "runs_filtered.parquet")

# Barry's per-attempt lognormal noise calibration (note: 80% within 3x/4x).
SIGMA_BASELINE = 0.78
SIGMA_ESTIMATE = 1.05


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--shape", default="linear")
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    data0 = load_model_data(DATA)
    sigma_obs = np.where(data0.is_estimate, SIGMA_ESTIMATE, SIGMA_BASELINE)
    # frontier model = latest dated model
    frontier = int(np.argmax(np.where(data0.has_date, data0.t_model, -np.inf)))
    rng = np.random.default_rng(args.seed)

    rows = []
    for lam in args.lambdas:
        data = load_model_data(DATA)
        if lam > 0:
            noise = rng.normal(0.0, np.sqrt(lam) * sigma_obs)
            data.log_dur = data.log_dur + noise
        model = build_model(data, shape=args.shape, duration_dist="studentt")
        with model:
            idata = pm.sample(tune=args.tune, draws=args.draws, chains=args.chains,
                              random_seed=args.seed, nuts_sampler="nutpie",
                              target_accept=0.95, progressbar=False)
        post = idata.posterior
        dt = np.log(2) / post["slope_now"].values.ravel() * 12.0
        th = np.exp(post["theta"].values[..., frontier].ravel())  # frontier 50% horizon, minutes
        ndiv = int(idata.sample_stats["diverging"].sum())
        rows.append((lam, np.median(dt), np.median(th), ndiv))
        print(f"lambda={lam:.2f}: doubling {np.median(dt):.2f} mo, frontier h50 {np.median(th):.0f} min "
              f"({np.median(th)/60:.1f} h), div={ndiv}", flush=True)

    lams = np.array([r[0] for r in rows])
    ths = np.array([r[2] for r in rows])
    dts = np.array([r[1] for r in rows])
    th0 = ths[lams == 0][0]
    # SIMEX fit: log(TH(lambda)/TH0) = b*lambda ; extrapolate to lambda=-1
    b = np.polyfit(lams, np.log(ths / th0), 1)[0]
    th_denoised = th0 * np.exp(b * (-1.0))
    print("\n=== SIMEX summary (Bayesian model) ===")
    print(f"frontier h50 at observed noise (lambda=0): {th0:.0f} min ({th0/60:.1f} h)")
    print(f"SIMEX slope b = {b:+.4f}  (Barry's model: strongly negative; ~flat here means eps absorbs it)")
    print(f"extrapolated noise-removed (lambda=-1) frontier h50: {th_denoised:.0f} min ({th_denoised/60:.1f} h)")
    print(f"implied noise-correction: {100*(th_denoised/th0 - 1):+.1f}%  "
          f"(Barry's SIMEX on METR's model: -25% to -40%)")
    print(f"doubling time across lambda: {dts.min():.2f}-{dts.max():.2f} mo (flat => trend noise-robust)")


if __name__ == "__main__":
    main()
