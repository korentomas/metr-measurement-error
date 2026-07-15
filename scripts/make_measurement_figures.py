"""Figures for the measurement-error improvement experiments.

Panel A: fitted heteroscedastic noise curve sigma_base * exp(gamma_sig *
(ell - mu_L)) vs the raw within-task log-sd of wall-time, binned by task
length -- does the model's length-dependent noise match the data it was
motivated by?

Panel B: posterior sd of log_L vs number of timed human runs per task (what
the measurement layer buys: uncertainty a plug-in model discards).

Panel C: per-task log_L shift from adding failed runs as censored obs
(survivorship correction) vs pooled model success rate -- the correction
lands on the hard, low-success tasks.

Usage:
    uv run python scripts/make_measurement_figures.py
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import arviz as az
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.data_prep import load_model_data

DATA = str(Path(__file__).parent.parent / "data" / "processed" / "runs_filtered.parquet")
OUT = Path(__file__).parent.parent / "outputs" / "figures"


def main() -> None:
    data = load_model_data(DATA)
    base = az.from_netcdf("outputs/fit_linear_robust.nc").posterior
    het = az.from_netcdf("outputs/fit_linear_robust_het.nc").posterior
    hf = az.from_netcdf("outputs/fit_linear_robust_hf.nc").posterior

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # ---- Panel A: heteroscedastic noise curve vs raw within-task log-sd ----
    ax = axes[0]
    base_mask = ~data.is_estimate & ~data.is_censored
    ld = data.log_dur[base_mask]
    ti = data.task_idx_obs[base_mask]
    by = defaultdict(list)
    for t, v in zip(ti, ld):
        by[t].append(v)
    task_mean = {t: np.mean(v) for t, v in by.items()}
    task_sd = {t: np.std(v, ddof=1) for t, v in by.items() if len(v) >= 2}
    xs = np.array([task_mean[t] for t in task_sd])
    ys = np.array([task_sd[t] for t in task_sd])
    # bin
    bins = np.quantile(xs, np.linspace(0, 1, 7))
    bc, bm = [], []
    for b in range(6):
        sel = (xs >= bins[b]) & (xs <= bins[b + 1])
        if sel.sum() >= 3:
            bc.append(xs[sel].mean()); bm.append(np.median(ys[sel]))
    ax.scatter(xs, ys, s=12, alpha=0.3, color="gray", label="per-task log-sd (>=2 runs)")
    ax.plot(bc, bm, "o-", color="black", label="binned median")
    # fitted curves
    mu_L = float(het["mu_L"].mean())
    sb = float(het["sigma_base"].mean()); gs = float(het["gamma_sig"].mean())
    grid = np.linspace(xs.min(), xs.max(), 100)
    ax.plot(grid, sb * np.exp(gs * (grid - mu_L)), color="C3", lw=2.5,
            label=f"het fit: gamma_sig={gs:+.2f}")
    ax.axhline(float(base["sigma_base"].mean()), color="C0", ls="--", lw=2,
               label=f"homoscedastic sigma_base={float(base['sigma_base'].mean()):.2f}")
    ax.set_xlabel("task mean log wall-time (log-min)")
    ax.set_ylabel("within-task sd of log wall-time")
    ax.set_title("A. Heteroscedastic noise matches the data")
    ax.legend(fontsize=8)

    # ---- Panel B: sd(log_L) vs #runs ----
    ax = axes[1]
    log_L = base["log_L"].stack(s=("chain", "draw")).values
    sd_logL = log_L.std(axis=1)
    runs_per_task = Counter(ti.tolist())
    n_runs = np.array([runs_per_task.get(t, 0) for t in range(data.n_tasks)])
    cats = [("estimate\n(0 runs)", n_runs == 0), ("1 run", n_runs == 1),
            ("2 runs", n_runs == 2), ("3+ runs", n_runs >= 3)]
    positions = range(len(cats))
    ax.boxplot([sd_logL[sel] for _, sel in cats], positions=list(positions),
               widths=0.6, showfliers=False)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([c[0] for c in cats])
    ax.set_ylabel("posterior sd of log_L (log-min)")
    ax.set_title("B. Measurement uncertainty a plug-in discards")

    # ---- Panel C: hf log_L shift vs success rate ----
    ax = axes[2]
    lb = base["log_L"].mean(("chain", "draw")).values
    lh = hf["log_L"].mean(("chain", "draw")).values
    shift = lh - lb
    n_att = np.zeros(data.n_tasks); n_suc = np.zeros(data.n_tasks)
    np.add.at(n_att, data.task_idx_irt, data.n_attempts)
    np.add.at(n_suc, data.task_idx_irt, data.n_successes)
    sr = np.where(n_att > 0, n_suc / np.maximum(n_att, 1), np.nan)
    ax.scatter(sr, shift, s=14, alpha=0.5, color="C2")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("pooled model success rate on task")
    ax.set_ylabel("log_L shift from failed-run censoring")
    ax.set_title("C. Survivorship correction lands on hard tasks")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "measurement_improvements.png", dpi=130)
    print(f"wrote {OUT / 'measurement_improvements.png'}")


if __name__ == "__main__":
    main()
