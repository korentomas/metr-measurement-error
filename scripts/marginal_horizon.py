"""Marginal (METR-style) vs conditional (exp(theta)) horizon.

The model reports each agent's 50% horizon as h50_m = exp(theta_m). By
construction that is a *conditional* horizon: the length of a task with
residual difficulty eps_i = 0 (median difficulty) that model m solves at
p = 0.5. METR's published 50% time horizon is a *marginal* quantity: the
length at which the model succeeds 50% of the time averaged over the actual
population of tasks at that length -- i.e. averaged over the eps (residual
difficulty) and a (discrimination) distributions. With sigma_eps ~ 2.2 (an
8x difficulty spread) these are different objects, and any comparison to
METR or Moss, or any extrapolation, needs to know which one it is using.

This script computes both from a fitted InferenceData and reports what the
distinction does and does not change. The population success curve at length
ell for model m is

    P_pop(ell) = E_{eps ~ N(0, sigma_eps), a ~ LogNormal(0, sigma_a)}
                   [ sigmoid( a * (theta_m - ell - eps) ) ].

The eps integral is done with the standard logistic-normal (probit) approx
E[sigmoid(N(mu, s^2))] ~= sigmoid(mu / sqrt(1 + pi*s^2/8)); the a integral is
a small Monte-Carlo average per posterior draw. From P_pop we read the
marginal h50 (crossing of 0.5) and the marginal h10/h90 (the 10% and 90%
horizons), whose log-ratio measures how *flat* the aggregate curve is
compared with the single-task (conditional) logistic.

Three facts fall out (see the printed summary):
  1. Marginal h50 == conditional h50 == exp(theta_m), exactly, because
     sigmoid is odd-symmetric and eps is symmetric mean-zero: P_pop(theta)
     = 0.5 for every draw. So the reported 50% horizon *level* is robust to
     the conditional/marginal distinction.
  2. The doubling-time headline is a slope of theta over time, unchanged by a
     stationary eps distribution, so it too is robust (it is not recomputed
     here -- slope_now is identical).
  3. Everything else moves: the marginal curve is materially flatter, so the
     10%/90% horizons and any non-50% reliability target diverge from the
     conditional ones, and any extrapolation to a fixed horizon threshold
     inherits full sigma_eps sensitivity.

Usage:
    uv run python scripts/marginal_horizon.py --fit outputs/fit_linear_robust.nc
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


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def marginal_curve(theta, sigma_eps, a_samples, ell_grid):
    """P_pop on ell_grid for one posterior draw.

    theta, sigma_eps: scalars for this draw. a_samples: (K,) draws of a for
    the MC average over discrimination. Returns (len(ell_grid),).
    """
    c = theta - ell_grid[:, None]                     # (G, 1)
    mu = a_samples[None, :] * c                        # (G, K)
    s2 = (a_samples[None, :] * sigma_eps) ** 2         # (G, K)
    p = _sigmoid(mu / np.sqrt(1.0 + np.pi * s2 / 8.0))  # (G, K)
    return p.mean(axis=1)                              # (G,)


def _cross(ell_grid, curve, level):
    """Linear-interpolate the ell where curve == level (curve is decreasing)."""
    # curve decreases in ell; find bracketing points
    idx = np.searchsorted(-curve, -level)
    if idx <= 0:
        return ell_grid[0]
    if idx >= len(curve):
        return ell_grid[-1]
    x0, x1 = ell_grid[idx - 1], ell_grid[idx]
    y0, y1 = curve[idx - 1], curve[idx]
    if y1 == y0:
        return x0
    return x0 + (level - y0) * (x1 - x0) / (y1 - y0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--sota-only", action="store_true")
    parser.add_argument("--n-draws", type=int, default=1000, help="Thin posterior to this many draws.")
    parser.add_argument("--n-a", type=int, default=400, help="MC samples for the a-average per draw.")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    data = load_model_data(args.data, sota_only=args.sota_only)
    idata = az.from_netcdf(args.fit)
    post = idata.posterior
    rng = np.random.default_rng(args.seed)

    theta = post["theta"].stack(s=("chain", "draw")).values      # (M, S)
    sigma_eps = post["sigma_eps"].stack(s=("chain", "draw")).values  # (S,)
    sigma_a = post["sigma_a"].stack(s=("chain", "draw")).values      # (S,)
    M, S = theta.shape

    if S > args.n_draws:
        sel = rng.choice(S, size=args.n_draws, replace=False)
        theta, sigma_eps, sigma_a = theta[:, sel], sigma_eps[sel], sigma_a[sel]
        S = args.n_draws

    ell_grid = np.linspace(theta.min() - 6, theta.max() + 6, 241)

    # Aggregate over models: report the population-curve flattening, which is a
    # property of (sigma_eps, sigma_a), plus per-model marginal h50/h10/h90.
    cond_logwidth = []   # conditional 10-90 log width, per draw (median a)
    marg_logwidth = []   # marginal 10-90 log width, per draw
    for si in range(S):
        a_s = np.exp(rng.normal(0, sigma_a[si], size=args.n_a))
        a_med = np.exp(0.0)  # median of LogNormal(0, sigma_a) = 1
        # conditional curve: single logistic with a = median (eps=0)
        cond = _sigmoid(a_med * (theta[:, si].mean() - ell_grid))
        marg = marginal_curve(theta[:, si].mean(), sigma_eps[si], a_s, ell_grid)
        cond_logwidth.append(_cross(ell_grid, cond, 0.1) - _cross(ell_grid, cond, 0.9))
        marg_logwidth.append(_cross(ell_grid, marg, 0.1) - _cross(ell_grid, marg, 0.9))
    cond_logwidth = np.array(cond_logwidth)
    marg_logwidth = np.array(marg_logwidth)

    print(f"fit: {args.fit}")
    print(f"posterior draws used: {S}, a-MC samples: {args.n_a}\n")

    # Fact 1: marginal h50 == conditional h50, checked numerically.
    checkP = []
    for si in range(min(S, 200)):
        a_s = np.exp(rng.normal(0, sigma_a[si], size=args.n_a))
        checkP.append(marginal_curve(theta[:, si].mean(), sigma_eps[si], a_s,
                                     np.array([theta[:, si].mean()]))[0])
    print("=== Fact 1: 50% horizon LEVEL is definition-robust ===")
    print(f"  P_pop(ell = theta) across draws: mean {np.mean(checkP):.4f} "
          f"(should be 0.5 exactly up to MC noise)")
    print("  => marginal h50 = exp(theta) = conditional h50. The reported")
    print("     per-agent 50% horizons need no marginal correction.\n")

    # Fact 3: the curve flattens, so non-50% horizons diverge.
    print("=== Fact 3: the aggregate curve is FLATTER (10%-90% horizon width) ===")
    print(f"  conditional (single-task, a=median) 10-90 log-width: "
          f"{np.median(cond_logwidth):.2f}  ({np.exp(np.median(cond_logwidth)):.1f}x in minutes)")
    print(f"  marginal   (population)            10-90 log-width: "
          f"{np.median(marg_logwidth):.2f}  ({np.exp(np.median(marg_logwidth)):.1f}x in minutes)")
    print(f"  flattening factor (marginal/conditional width): "
          f"{np.median(marg_logwidth/cond_logwidth):.2f}x\n")
    print("  => the 50% level and the doubling-time slope are robust, but any")
    print("     non-50% reliability horizon (e.g. an 80%-reliable horizon) and")
    print("     any extrapolation to a fixed threshold are NOT: they inherit")
    print("     the full sigma_eps spread and differ from METR's marginal curve.")


if __name__ == "__main__":
    main()
