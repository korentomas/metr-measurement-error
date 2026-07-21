"""Figures for the Bayesian measurement-error time-horizon model.

Generates, from the real fitted InferenceData in outputs/, the plots the
README's Results section embeds:

  1. horizon_trend.png     -- per-model 50% horizon vs release date, plus
                               all four fitted trend shapes (linear, kink,
                               superexp, logistic) with credible bands,
                               following Moss's fig-horizon-fan.
  2. ppc_calibration.png    -- posterior-predictive vs observed success rate
                               by task-length bin (the IRT-layer PPC).
  3. outlier_pull.png       -- how far the Normal vs Student-t fit pulls the
                               latent log_L of the worst outlier tasks away
                               from their observed median duration.
  4. sbc_ranks.png          -- SBC rank histograms for 3 representative
                               hyperparameters (uniformity check).
  5. stacking_weights.png   -- PSIS-LOO stacking weights across the 4 trend
                               shapes.
  6. doubling_time_density.png -- posterior density of ln(2)/slope_now
                               ("doubling time now"), per trend shape plus
                               the stacked mixture, cf. Moss's
                               fig-doubling-time.
  7. duration_dist_comparison.png -- pooled within-task log-residuals
                               (real data) vs the log-normal, Student-t, and
                               Weibull-implied residual densities, showing
                               why the Weibull loses the PSIS-LOO comparison.
  8. difficulty_residual.png -- posterior difficulty multiplier exp(eps_i)
                               vs task length, with +-1/2 sigma bands: our
                               version of Moss's fig-difficulty-variation.
  9. fork_discriminator.png -- the falsification test from
                               scripts/fork_discriminator.py as a picture:
                               eps-vs-length trend on well-timed tasks
                               (>=3 timed runs), with the poorly timed long
                               tasks overlaid. If eps were absorbing length
                               overestimation (Barry's reading), those tasks
                               would sit below the trend; they sit above.

Style deliberately follows Jonas Moss's own figures
(metr-stats/scripts/make_figures.py): plain matplotlib defaults, shaded
credible bands instead of gradients, no gridlines beyond light dashed
horizontals, default color cycle.

Usage:
    uv run python scripts/make_figures.py
"""

from __future__ import annotations

import gc
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from models.data_prep import DEFAULT_RELEASE_DATES, RELEASE_DATE_OVERRIDES, load_model_data

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs" / "figures"
DATA_PATH = ROOT / "data" / "processed" / "runs_filtered.parquet"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

C_LINEAR = "tab:blue"
C_KINK = "tab:orange"
C_SUPEREXP = "tab:green"
C_LOGISTIC = "tab:purple"
C_NORMAL = "tab:gray"
C_ROBUST = "tab:red"

# The four fitted trend shapes, in the order the README's stacking table
# lists them, each mapped to its preferred (Student-t / --robust) .nc file
# and the color used consistently across the horizon-trend and
# doubling-time-density figures.
SHAPE_FITS = {
    "kink": ("fit_kink_robust.nc", C_KINK, "kink trend"),
    "linear": ("fit_linear_robust.nc", C_LINEAR, "linear trend"),
    "superexp": ("fit_superexp_robust.nc", C_SUPEREXP, "superexponential trend"),
    "logistic": ("fit_logistic_robust.nc", C_LOGISTIC, "logistic trend"),
}


def _shape_f_t(shape: str, post, t_grid: np.ndarray):
    """Reconstruct f_t(t_grid) (log-minutes) from a fitted posterior's own
    parameters, following the mean functions in models/time_horizon_model.py
    exactly (one branch per shape)."""
    beta0 = post["beta0"].stack(s=("chain", "draw")).values
    if shape == "linear":
        beta1 = post["beta1"].stack(s=("chain", "draw")).values
        f_t = beta0[None, :] + beta1[None, :] * t_grid[:, None]
    elif shape == "kink":
        beta1 = post["beta1"].stack(s=("chain", "draw")).values
        delta = post["delta"].stack(s=("chain", "draw")).values
        t_k = post["t_k"].stack(s=("chain", "draw")).values
        w = 0.1
        z = (t_grid[:, None] - t_k[None, :]) / w
        softplus = np.logaddexp(0.0, z) * w
        f_t = beta0[None, :] + beta1[None, :] * t_grid[:, None] + delta[None, :] * softplus
    elif shape == "superexp":
        beta1 = post["beta1"].stack(s=("chain", "draw")).values
        beta2 = post["beta2"].stack(s=("chain", "draw")).values
        f_t = (
            beta0[None, :]
            + beta1[None, :] * t_grid[:, None]
            + beta2[None, :] * t_grid[:, None] ** 2
        )
    else:  # logistic
        h = post["h"].stack(s=("chain", "draw")).values
        t0 = post["t0"].stack(s=("chain", "draw")).values
        s = post["s"].stack(s=("chain", "draw")).values
        sig = 1.0 / (1.0 + np.exp(-(t_grid[:, None] - t0[None, :]) / s[None, :]))
        f_t = beta0[None, :] + h[None, :] * sig
    return np.clip(f_t, -20.0, 20.0)  # exp(20 min) ~= 900 years; guards against overflow


def _model_dates() -> dict[str, date]:
    with open(DEFAULT_RELEASE_DATES) as f:
        raw: dict[str, str] = json.load(f)
    raw = {**raw, **RELEASE_DATE_OVERRIDES}
    out = {}
    for m, d in raw.items():
        try:
            out[m] = date.fromisoformat(d)
        except (ValueError, TypeError):
            continue
    return out


def _format_duration_minutes(minutes: float) -> str:
    """Human-readable duration label for a tick value given in minutes,
    matching the spirit of Jonas Moss's METR-style axis labels."""
    if not np.isfinite(minutes) or minutes <= 0:
        return ""
    if minutes < 60:
        v = int(round(minutes)) if minutes >= 1 else round(minutes, 1)
        return f"{v} min"
    hours = minutes / 60.0
    if hours < 24:
        v = int(round(hours))
        return f"{v} hour" if v == 1 else f"{v} hours"
    days = hours / 24.0
    if days >= 365:
        years = int(round(days / 365.0))
        return f"{years} year" if years == 1 else f"{years} years"
    if days >= 30:
        months = int(round(days / 30.0))
        return f"{months} month" if months == 1 else f"{months} months"
    v = int(round(days))
    return f"{v} day" if v == 1 else f"{v} days"


def _apply_minute_duration_ticks(ax, y_values_minutes: np.ndarray) -> None:
    """Set human-readable y-axis ticks (log scale, minutes) spanning the
    plotted data range, e.g. "6 min", "1 hour", "1 day"."""
    y = np.asarray(y_values_minutes, dtype=float)
    y = y[np.isfinite(y) & (y > 0)]
    if y.size == 0:
        return
    ymin, ymax = float(np.min(y)), float(np.max(y))

    ladder = np.array(
        [1, 2, 5, 10, 20, 30, 60, 180, 360, 720, 1440, 4320, 10080,
         43200, 259200, 525960, 1577880, 5259600, 26298000],
        dtype=float,
    )  # 1, 2, 5, 10, 20, 30 min; 1, 3, 6, 12 hours; 1, 3, 7 days;
    #    1, 6 months; 1, 3, 10, 50 years
    lo, hi = ymin / 1.15, ymax * 1.15
    keep = ladder[(ladder >= lo) & (ladder <= hi)]
    if keep.size < 3:
        below = ladder[ladder < lo]
        above = ladder[ladder > hi]
        candidates = []
        if below.size:
            candidates.append(below[-1])
        candidates.extend(keep.tolist())
        if above.size:
            candidates.append(above[0])
        keep = np.array(candidates, dtype=float)
    if keep.size == 0:
        return

    ax.set_yticks(keep)
    ax.set_yticklabels([_format_duration_minutes(float(t)) for t in keep])


# Frontier models to annotate directly on the horizon-trend plot, mapping the
# internal alias (see models/data_prep.py RELEASE_DATE_OVERRIDES) to a
# recognizable display name.
FRONTIER_LABELS = {
    "claude_opus_4_6_inspect": "Claude Opus 4.6",
    "flamingo_2": "GPT-5.3-Codex",
    "gpt_5_2": "GPT-5.2",
}


def _t_to_date_map(data, dates: dict[str, date]):
    """Least-squares affine maps between centered t_model (years) and the
    calendar ordinal, using the dated models (t_model is an exact affine
    transform of the calendar date, so this recovers it losslessly).
    Returns (t_to_ordinal, date_to_t)."""
    idx = [i for i, m in enumerate(data.model_names) if m in dates and data.has_date[i]]
    t = np.array([data.t_model[i] for i in idx])
    ord_ = np.array([dates[data.model_names[i]].toordinal() for i in idx], dtype=float)
    slope, intercept = np.polyfit(t, ord_, 1)
    return (
        lambda tt: slope * tt + intercept,
        lambda d: (d.toordinal() - intercept) / slope,
    )


# ---------------------------------------------------------------------------
# 1. Horizon vs release date, all four trend shapes (following Moss's
#    fig-horizon-fan: one ax.plot + ax.fill_between per shape, real data
#    points overlaid once since they don't depend on which shape is fit).
# ---------------------------------------------------------------------------
def fig_horizon_trend(data, dates: dict[str, date]) -> None:
    t2d, d2t = _t_to_date_map(data, dates)

    # --- per-model horizon points (from the kink fit; preferred variant;
    # shape-independent given the data, so plotted once, cf. Moss's `hpts`) ---
    id_kink = az.from_netcdf(OUT.parent / SHAPE_FITS["kink"][0])
    theta = id_kink.posterior["theta"].stack(s=("chain", "draw")).values

    fig, ax = plt.subplots(figsize=(9, 5.6))

    all_meds = []
    frontier_points = {}
    for m in range(data.n_models):
        if not data.has_date[m]:
            continue
        h = np.exp(theta[m, :])
        med = np.median(h)
        lo, hi = np.quantile(h, [0.025, 0.975])
        d = date.fromordinal(int(round(t2d(data.t_model[m]))))
        ax.errorbar(
            d, med, yerr=[[med - lo], [hi - med]],
            fmt="o", color="black", ecolor="0.7", elinewidth=1, capsize=0,
            markersize=4, zorder=5,
        )
        all_meds.append(med)
        name = data.model_names[m]
        if name in FRONTIER_LABELS:
            frontier_points[name] = (d, med)
    del id_kink
    gc.collect()

    # --- trend lines + 95% credible bands, one per shape ---
    # The x-axis runs well past the last dated model (2029, following Moss's
    # fig-horizon-fan convention of x_max = 2029-01-01): over the observed
    # span the four shapes mostly agree, so the forecast window is where
    # their disagreement -- the whole point of overlaying all four -- becomes
    # visible.
    t_lo, t_hi = data.t_model[data.has_date].min(), data.t_model[data.has_date].max()
    forecast_end = date(2029, 1, 1)
    t_grid = np.linspace(t_lo, d2t(forecast_end), 300)
    d_grid = [date.fromordinal(int(round(t2d(t)))) for t in t_grid]
    last_model_date = date.fromordinal(int(round(t2d(t_hi))))

    # Hard sanity clip: the logistic shape is expected to be degenerate/poorly
    # identified (README: "no saturation signal ... degenerates toward still
    # rising with a wide, poorly-identified slope_now"), so its band can blow
    # up to absurd horizons. Clip the underlying data at 50 years rather than
    # hide the shape -- the point is to show that shape genuinely isn't
    # pinned down by the data, the same way Moss's fan chart does. This is
    # just a numerical safety net; the actual axis limits below are fit to
    # whatever of this clipped data is actually visible, so a shape blowing
    # up to the clip doesn't force 50 years of blank space onto the plot.
    y_clip_hi = 500 * 365.25 * 24 * 60  # 500 years, in minutes (numerical guard)
    y_clip_lo = 0.05  # 3 sec
    # Separate *display* cap: anything past this runs off the top of the axes
    # (the standard "leaves the chart" look) instead of riding along the top
    # border at the clip value, which would falsely read as saturation.
    y_axis_cap = 50 * 365.25 * 24 * 60  # 50 years

    all_hi = []
    all_lo = []
    for shape, (fname, color, label) in SHAPE_FITS.items():
        idata = az.from_netcdf(OUT.parent / fname)
        f_t = _shape_f_t(shape, idata.posterior, t_grid)
        h_grid = np.exp(f_t)  # (grid, draws), minutes
        med = np.median(h_grid, axis=1)
        lo, hi = np.quantile(h_grid, [0.025, 0.975], axis=1)
        lo = np.clip(lo, y_clip_lo, y_clip_hi)
        hi = np.clip(hi, y_clip_lo, y_clip_hi)
        med = np.clip(med, y_clip_lo, y_clip_hi)
        all_hi.append(hi)
        all_lo.append(lo)
        # Degenerate shapes get a thinner line and lighter band so the two
        # well-identified shapes (which the stacking weights favor almost
        # entirely -- kink 0.64, linear 0.36, see README) stay legible.
        degenerate = shape in ("superexp", "logistic")
        ax.plot(
            d_grid, med, color=color, lw=1.6 if degenerate else 2.2,
            ls="--" if degenerate else "-", label=label, zorder=2,
            alpha=0.85 if degenerate else 1.0,
        )
        ax.fill_between(
            d_grid, lo, hi, color=color, alpha=0.10 if degenerate else 0.18,
            lw=0, zorder=1,
        )
        del idata
        gc.collect()

    # Mark where the data end and the forecast begins.
    ax.axvline(last_model_date, color="0.4", lw=0.9, ls=":", zorder=1)

    # --- direct annotations for a few frontier models ---
    # Claude Opus 4.6 and GPT-5.3-Codex share a release date (2026-02-05), so
    # they're offset in opposite directions to avoid overlapping each other.
    annotation_offsets = {
        "claude_opus_4_6_inspect": (10, 10),
        "flamingo_2": (10, -16),
        "gpt_5_2": (-70, 8),
    }
    for name, (d, med) in frontier_points.items():
        dx, dy = annotation_offsets.get(name, (10, 10))
        ax.annotate(
            FRONTIER_LABELS[name],
            xy=(d, med),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            color="0.15",
            zorder=6,
        )

    # Fit the axis to what's actually visible (per-model points + all four
    # bands), with a log-scale margin, instead of the fixed 50-year clip --
    # otherwise the degenerate-shape safety net above wastes most of the
    # plot's vertical space on blank area no line or band reaches.
    visible = np.concatenate([np.asarray(all_meds)] + all_hi + all_lo)
    visible = visible[np.isfinite(visible) & (visible > 0)]
    y_lo = max(y_clip_lo, float(np.min(visible)) / 1.6)
    y_hi = min(y_axis_cap, float(np.max(visible)) * 1.6)

    ax.set_yscale("log")
    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("50% horizon, median task (log scale)")
    ax.set_xlabel("Release date")
    ax.set_title("Horizon vs release date: all four trend shapes")
    _apply_minute_duration_ticks(ax, visible)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "horizon_trend.png")
    plt.close(fig)
    print("wrote", OUT / "horizon_trend.png")


# ---------------------------------------------------------------------------
# 2. PPC calibration by task-length bin
# ---------------------------------------------------------------------------
def fig_ppc(data) -> None:
    idata = az.from_netcdf(OUT.parent / "fit_linear_robust.nc")
    post = idata.posterior
    theta = post["theta"].stack(s=("chain", "draw")).values
    a = post["a"].stack(s=("chain", "draw")).values
    log_L = post["log_L"].stack(s=("chain", "draw")).values
    eps = post["eps"].stack(s=("chain", "draw")).values

    ti, mi = data.task_idx_irt, data.model_idx_irt
    eta = a[ti, :] * (theta[mi, :] - (log_L[ti, :] + eps[ti, :]))
    p = 1.0 / (1.0 + np.exp(-eta))
    rng = np.random.default_rng(1234)
    n = data.n_attempts[:, None]
    s_rep = rng.binomial(n, p)

    log_L_mean = log_L.mean(axis=1)
    bins = np.quantile(log_L_mean, np.linspace(0, 1, 7))
    bin_idx = np.clip(np.digitize(log_L_mean[ti], bins) - 1, 0, 5)

    centers, obs_rates, pp_means, pp_los, pp_his = [], [], [], [], []
    for b in range(6):
        rows = bin_idx == b
        if rows.sum() == 0:
            continue
        obs_rate = data.n_successes[rows].sum() / data.n_attempts[rows].sum()
        rep_rate = s_rep[rows, :].sum(axis=0) / data.n_attempts[rows].sum()
        centers.append(np.sqrt(np.exp(bins[b]) * np.exp(bins[b + 1])))
        obs_rates.append(obs_rate)
        pp_means.append(rep_rate.mean())
        lo, hi = np.quantile(rep_rate, [0.025, 0.975])
        pp_los.append(lo)
        pp_his.append(hi)

    centers = np.array(centers)
    fig, ax = plt.subplots(figsize=(7, 5))
    order = np.argsort(centers)
    centers, obs_rates = centers[order], np.array(obs_rates)[order]
    pp_means = np.array(pp_means)[order]
    pp_los, pp_his = np.array(pp_los)[order], np.array(pp_his)[order]

    ax.plot(centers, pp_means, color="tab:blue", lw=2, label="posterior predictive mean")
    ax.fill_between(centers, pp_los, pp_his, color="tab:blue", alpha=0.2, lw=0, label="95% posterior predictive")
    ax.plot(centers, obs_rates, "o", color="black", markersize=7, label="observed", zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("Task length bin, geometric-mean minutes (log scale)")
    ax.set_ylabel("Success rate")
    ax.set_title("Posterior predictive check: success rate by task-length bin")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "ppc_calibration.png")
    plt.close(fig)
    print("wrote", OUT / "ppc_calibration.png")

    del idata
    gc.collect()


# ---------------------------------------------------------------------------
# 3. Outlier pull: Normal vs Student-t
# ---------------------------------------------------------------------------
def fig_outlier_pull(data) -> None:
    id_n = az.from_netcdf(OUT.parent / "fit_linear.nc")
    id_r = az.from_netcdf(OUT.parent / "fit_linear_robust.nc")

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
    top = devs[:6]

    logL_n = id_n.posterior["log_L"]
    logL_r = id_r.posterior["log_L"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ys = np.arange(len(top))[::-1]
    labels = []
    for y, (dev, t, obs_med, n) in zip(ys, top):
        task = data.task_ids[t]
        labels.append(task[:34])
        ln = logL_n.sel(task=task).values.ravel()
        lr = logL_r.sel(task=task).values.ravel()
        pull_n = np.median(ln) - obs_med
        pull_r = np.median(lr) - obs_med
        lo_n, hi_n = np.quantile(ln, [0.025, 0.975]) - obs_med
        lo_r, hi_r = np.quantile(lr, [0.025, 0.975]) - obs_med
        ax.plot([lo_n, hi_n], [y + 0.12, y + 0.12], color=C_NORMAL, lw=2, solid_capstyle="round")
        ax.plot([lo_r, hi_r], [y - 0.12, y - 0.12], color=C_ROBUST, lw=2, solid_capstyle="round")
        ax.plot(pull_n, y + 0.12, "o", color=C_NORMAL, markersize=6)
        ax.plot(pull_r, y - 0.12, "o", color=C_ROBUST, markersize=6)

    ax.axvline(0, color="black", lw=1, ls="--", alpha=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Posterior log_L minus task's observed median log duration (log-minutes)")
    ax.set_title("Outlier pull on latent task length: Normal vs Student-t")
    handles = [
        plt.Line2D([0], [0], color=C_NORMAL, marker="o", lw=2, label="Normal"),
        plt.Line2D([0], [0], color=C_ROBUST, marker="o", lw=2, label="Student-t"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "outlier_pull.png")
    plt.close(fig)
    print("wrote", OUT / "outlier_pull.png")

    del id_n, id_r
    gc.collect()


# ---------------------------------------------------------------------------
# 4. SBC rank histograms (representative sample)
# ---------------------------------------------------------------------------
def fig_sbc_ranks() -> None:
    d = np.load(ROOT / "outputs" / "sbc_results.npz")
    params = ["beta1", "sigma_eps", "sigma_base"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    n_reps = len(d["rank_beta1"])
    n_bins = 10
    expected = n_reps / n_bins
    for ax, p in zip(axes, params):
        ranks = d[f"rank_{p}"]
        ax.hist(ranks, bins=np.linspace(0, 1, n_bins + 1), color="tab:blue", edgecolor="white")
        ax.axhline(expected, color="black", ls="--", lw=1)
        ax.set_title(p)
        ax.set_xlabel("normalized rank")
    axes[0].set_ylabel("count")
    fig.suptitle("SBC rank histograms (50 reps; flat = calibrated)")
    fig.tight_layout()
    fig.savefig(OUT / "sbc_ranks.png")
    plt.close(fig)
    print("wrote", OUT / "sbc_ranks.png")


# ---------------------------------------------------------------------------
# 5. Stacking weights
# ---------------------------------------------------------------------------
def _compute_stacking_weights() -> dict[str, float]:
    fits = {name: OUT.parent / fname for name, (fname, _, _) in SHAPE_FITS.items()}
    loos = {}
    for name, f in fits.items():
        idata = az.from_netcdf(f)
        loos[name] = az.loo(idata, var_name="successes")
        del idata
        gc.collect()
    cmp = az.compare(loos, method="stacking")
    return cmp["weight"].to_dict()


def fig_stacking_weights(weights: dict[str, float]) -> None:
    order = ["kink", "linear", "superexp", "logistic"]
    w = [weights[o] for o in order]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(order, w, color="tab:blue", width=0.6)
    for i, wi in enumerate(w):
        ax.text(i, wi + 0.01, f"{wi:.2f}", ha="center")
    ax.set_ylabel("Stacking weight")
    ax.set_title("Bayesian stacking weights across trend shapes")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(OUT / "stacking_weights.png")
    plt.close(fig)
    print("wrote", OUT / "stacking_weights.png")


# ---------------------------------------------------------------------------
# 6. Duration-likelihood comparison: real residuals vs the three fitted dists
# ---------------------------------------------------------------------------
def fig_duration_dist_comparison(data) -> None:
    from scipy import stats

    base = ~data.is_estimate & ~data.is_censored
    ld, ti = data.log_dur[base], data.task_idx_obs[base]
    by = defaultdict(list)
    for t, v in zip(ti, ld):
        by[t].append(v)
    resid = []
    for vs in by.values():
        if len(vs) < 2:
            continue
        vs = np.array(vs)
        resid.extend((vs - np.median(vs)).tolist())
    resid = np.array(resid)

    id_lognorm = az.from_netcdf(OUT.parent / "fit_linear.nc")
    id_studentt = az.from_netcdf(OUT.parent / "fit_linear_robust.nc")
    id_weibull = az.from_netcdf(OUT.parent / "fit_linear_weibull.nc")
    sigma_lognorm = float(id_lognorm.posterior["sigma_base"].values.mean())
    sigma_studentt = float(id_studentt.posterior["sigma_base"].values.mean())
    nu_studentt = float(id_studentt.posterior["nu"].values.mean())
    alpha_w = float(id_weibull.posterior["alpha_w"].values.mean())
    del id_lognorm, id_studentt, id_weibull
    gc.collect()

    xg = np.linspace(-3.2, 3.2, 800)
    d_lognorm = stats.norm(loc=0.0, scale=sigma_lognorm).pdf(xg)
    d_studentt = stats.t(df=nu_studentt, loc=0.0, scale=sigma_studentt).pdf(xg)
    # Weibull, median-matched: log(D) - log(L) is a Gumbel-min with scale
    # 1/alpha_w, median exactly 0 by construction (see scripts/make_figures.py
    # docstring / README "Weibull duration likelihood" section for the
    # derivation from beta_i = L_i / ln(2)^(1/alpha)).
    d_weibull = stats.gumbel_l(loc=-np.log(np.log(2.0)) / alpha_w, scale=1.0 / alpha_w).pdf(xg)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(resid, bins=40, density=True, color="0.75", edgecolor="white",
            label=f"observed within-task log-residuals (n={len(resid)})", zorder=1)
    ax.plot(xg, d_lognorm, color="tab:gray", lw=2, label="log-normal fit", zorder=2)
    ax.plot(xg, d_studentt, color="tab:red", lw=2.2, label="Student-t fit (preferred)", zorder=3)
    ax.plot(xg, d_weibull, color="tab:purple", lw=2, ls="--", label="Weibull-implied (Gumbel-min)", zorder=2)
    ax.set_xlabel("log(wall time) minus task's median log(wall time)")
    ax.set_ylabel("density")
    ax.set_title("Baseline-duration residuals: real data vs three candidate likelihoods")
    ax.set_xlim(-3.2, 3.2)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "duration_dist_comparison.png")
    plt.close(fig)
    print("wrote", OUT / "duration_dist_comparison.png")


# ---------------------------------------------------------------------------
# 7. Difficulty residual (eps_i) vs task length -- our version of Moss's fig 2
# ---------------------------------------------------------------------------
def fig_difficulty_residual(data) -> None:
    idata = az.from_netcdf(OUT.parent / "fit_linear_robust.nc")
    post = idata.posterior
    log_L = post["log_L"].stack(s=("chain", "draw")).values.mean(axis=1)
    eps = post["eps"].stack(s=("chain", "draw")).values.mean(axis=1)
    sigma_eps_med = float(post["sigma_eps"].values.mean())
    del idata
    gc.collect()

    L_minutes = np.exp(log_L)
    # eps_i is already on the same log-minutes scale as log_L (no kappa
    # rescaling needed, unlike Moss's b_j ~ alpha + kappa*log(t)), so
    # exp(eps_i) is directly the "equivalent difficulty time" multiplier:
    # how many times longer/shorter a task effectively is for the models
    # than its actual human time, cf. Moss's fig-difficulty-variation.
    multiplier = np.exp(eps)
    mult_1sig = np.exp(sigma_eps_med)
    mult_2sig = np.exp(2 * sigma_eps_med)

    fig, ax = plt.subplots(figsize=(9.8, 4.8))

    # +-1 sigma / +-2 sigma reference bands, following Moss's style exactly.
    ax.axhspan(1 / mult_1sig, mult_1sig, color="tab:blue", alpha=0.06, zorder=0)
    ax.axhline(mult_1sig, color="tab:blue", lw=0.7, alpha=0.35, ls=":")
    ax.axhline(1 / mult_1sig, color="tab:blue", lw=0.7, alpha=0.35, ls=":")
    ax.axhspan(1 / mult_2sig, mult_2sig, color="tab:blue", alpha=0.03, zorder=0)
    ax.axhline(mult_2sig, color="tab:blue", lw=0.5, alpha=0.25, ls=":")
    ax.axhline(1 / mult_2sig, color="tab:blue", lw=0.5, alpha=0.25, ls=":")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.6)

    ax.scatter(L_minutes, multiplier, s=16, alpha=0.5, color="tab:blue", edgecolor="none", zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Task length, posterior mean L_i")
    ax.set_ylabel("Difficulty multiplier\n(1x = average for this task length)")
    ax.set_title("Residual task difficulty vs task length")

    # Fewer, more widely spaced ticks than a dense duration ladder would give
    # -- crowding them (e.g. "10 min" beside "30 min") makes the labels
    # unreadable at this figure width.
    tick_minutes = [1 / 60, 1, 10, 60, 4 * 60, 24 * 60]
    tick_labels = ["1 sec", "1 min", "10 min", "1 hour", "4 hours", "1 day"]
    ax.set_xticks(tick_minutes)
    ax.set_xticklabels(tick_labels, fontsize=8.5)
    ax.set_xlim(L_minutes.min() * 0.6, L_minutes.max() * 1.6)

    yticks = [0.01, 0.1, 1.0, 10.0, 100.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:g}x" for y in yticks], fontsize=8)
    ax.grid(True, which="major", axis="y", ls="--", lw=0.5, alpha=0.2)

    for val, sign in (
        (mult_1sig, "+1"), (1 / mult_1sig, "-1"),
        (mult_2sig, "+2"), (1 / mult_2sig, "-2"),
    ):
        ax.annotate(
            f"  {sign}σ = {val:.2g}x", xy=(L_minutes.max() * 1.6, val),
            fontsize=8, color="tab:blue", alpha=0.7, va="center",
        )

    fig.tight_layout()
    fig.savefig(OUT / "difficulty_residual.png")
    plt.close(fig)


def fig_fork_discriminator(data) -> None:
    idata = az.from_netcdf(OUT.parent / "fit_linear_robust.nc")
    post = idata.posterior
    log_L = post["log_L"].stack(s=("chain", "draw")).values.mean(axis=1)
    eps = post["eps"].stack(s=("chain", "draw")).values.mean(axis=1)
    del idata
    gc.collect()

    # Same split as scripts/fork_discriminator.py: tasks with >=3 timed runs
    # have their length pinned by data and carry the trustworthy eps-vs-length
    # trend; the poorly timed long tasks (<=1 timed run, top length quartile)
    # are where Barry's length-overestimation reading and the eps reading
    # disagree about which side of that trend they should fall on.
    base_mask = ~data.is_estimate & ~data.is_censored
    nruns = np.bincount(data.task_idx_obs[base_mask], minlength=data.n_tasks)
    well = nruns >= 3
    poor_long = (nruns <= 1) & (log_L >= np.quantile(log_L, 0.75))
    rest = ~well & ~poor_long

    slope, intercept = np.polyfit(log_L[well], eps[well], 1)

    L_minutes = np.exp(log_L)
    multiplier = np.exp(eps)

    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.6)

    ax.scatter(L_minutes[rest], multiplier[rest], s=14, alpha=0.25,
               color="gray", edgecolor="none", zorder=1, label="other tasks")
    ax.scatter(L_minutes[well], multiplier[well], s=16, alpha=0.55,
               color="tab:blue", edgecolor="none", zorder=2,
               label="well-timed (>=3 timed runs)")
    ax.scatter(L_minutes[poor_long], multiplier[poor_long], s=30, alpha=0.85,
               color="tab:orange", edgecolor="none", zorder=3,
               label="poorly timed, long (<=1 timed run)")

    xs = np.linspace(log_L.min(), log_L.max(), 100)
    ax.plot(np.exp(xs), np.exp(intercept + slope * xs), color="tab:blue",
            lw=1.2, alpha=0.8, zorder=2,
            label="eps-vs-length trend on well-timed tasks")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Task length, posterior mean L_i")
    ax.set_ylabel("Difficulty multiplier\n(1x = average for this task length)")
    ax.set_title("Do the poorly timed long tasks obey the well-timed trend?")

    tick_minutes = [1 / 60, 1, 10, 60, 4 * 60, 24 * 60]
    tick_labels = ["1 sec", "1 min", "10 min", "1 hour", "4 hours", "1 day"]
    ax.set_xticks(tick_minutes)
    ax.set_xticklabels(tick_labels, fontsize=8.5)
    ax.set_xlim(L_minutes.min() * 0.6, L_minutes.max() * 1.6)

    yticks = [0.01, 0.1, 1.0, 10.0, 100.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:g}x" for y in yticks], fontsize=8)
    ax.grid(True, which="major", axis="y", ls="--", lw=0.5, alpha=0.2)

    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT / "fork_discriminator.png")
    plt.close(fig)
    print(f"wrote {OUT / 'fork_discriminator.png'}")
    print("wrote", OUT / "difficulty_residual.png")


# ---------------------------------------------------------------------------
# 8. Doubling-time posterior density, per shape + stacked mixture
#    (cf. Moss's fig-doubling-time, extended to all four shapes plus the
#    stacked combination, since here doubling time isn't only defined for
#    the linear shape -- slope_now gives a "doubling time now" for any of
#    them, see models/time_horizon_model.py).
# ---------------------------------------------------------------------------
def fig_doubling_time_density(weights: dict[str, float]) -> None:
    from scipy.stats import gaussian_kde

    rng = np.random.default_rng(20260706)
    shape_draws = {}
    for shape, (fname, color, _label) in SHAPE_FITS.items():
        idata = az.from_netcdf(OUT.parent / fname)
        slope_now = idata.posterior["slope_now"].values.ravel()
        shape_draws[shape] = np.log(2.0) / slope_now * 12.0  # months
        del idata
        gc.collect()

    # Stacked mixture: resample from each shape's draws proportional to its
    # stacking weight (same weights as the README's stacking table / the
    # stacking_weights.png bar chart), giving a single mixture posterior.
    n_mix = 20000
    mix_parts = []
    for shape, dt in shape_draws.items():
        k = int(round(weights.get(shape, 0.0) * n_mix))
        if k > 0:
            mix_parts.append(rng.choice(dt, size=k, replace=True))
    mix = np.concatenate(mix_parts) if mix_parts else np.array([])

    # Display range: clip to something readable. Logistic in particular has
    # a long right tail (near-zero slope_now => doubling time -> infinity)
    # per the README's note that it "degenerates toward still rising with a
    # wide, poorly-identified current slope" -- show that as a fat tail
    # rather than let it stretch the whole axis unreadable.
    x_max = 14.0
    xg = np.linspace(0.0, x_max, 500)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for shape, (fname, color, label) in SHAPE_FITS.items():
        dt = shape_draws[shape]
        dt_clipped = dt[(dt > 0) & (dt < 100)]  # drop the most extreme tail draws for the KDE fit
        if dt_clipped.size < 10:
            continue
        kde = gaussian_kde(dt_clipped, bw_method="silverman")
        dens = kde(xg)
        degenerate = shape in ("superexp", "logistic")
        ax.plot(
            xg, dens, color=color, lw=1.6 if degenerate else 2.0,
            ls="--" if degenerate else "-", alpha=0.85 if degenerate else 1.0,
            label=f"{label} (w={weights.get(shape, 0.0):.2f})", zorder=2,
        )
        ax.fill_between(xg, dens, color=color, alpha=0.08, lw=0, zorder=1)

    if mix.size:
        kde_mix = gaussian_kde(mix[(mix > 0) & (mix < 100)], bw_method="silverman")
        dens_mix = kde_mix(xg)
        ax.plot(xg, dens_mix, color="black", lw=2.4, label="stacked mixture", zorder=3)
        med = np.median(mix)
        ax.axvline(med, color="black", lw=1, ls=":", alpha=0.6, zorder=3)

    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Doubling time now (months)")
    ax.set_ylabel("Posterior density")
    ax.set_title("Doubling time now, by trend shape and stacked")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "doubling_time_density.png")
    plt.close(fig)
    print("wrote", OUT / "doubling_time_density.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_model_data(DATA_PATH)
    dates = _model_dates()

    fig_horizon_trend(data, dates)
    fig_ppc(data)
    fig_outlier_pull(data)
    fig_sbc_ranks()
    weights = _compute_stacking_weights()
    fig_stacking_weights(weights)
    fig_doubling_time_density(weights)
    fig_duration_dist_comparison(data)
    fig_difficulty_residual(data)
    fig_fork_discriminator(data)


if __name__ == "__main__":
    main()
