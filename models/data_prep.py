"""Assemble model-ready arrays for the Bayesian measurement-error time-horizon
model from the filtered human-timing data (data/load_runs.py output) plus the
raw runs.jsonl (for model-vs-task success/failure counts) and METR's
release-dates table (for the linear ability trend in theta_m).

This module produces a single `ModelData` container consumed by
`models/time_horizon_model.py::build_model`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Sibling checkouts expected next to this repo (see README "Setup"):
# METR/eval-analysis-public and JonasMoss/metr-stats.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SIBLINGS_ROOT = _REPO_ROOT.parent

DEFAULT_RUNS_JSONL = (
    _SIBLINGS_ROOT / "eval-analysis-public"
    / "reports/time-horizon-1-1/data/raw/runs.jsonl"
)
DEFAULT_RELEASE_DATES = _SIBLINGS_ROOT / "metr-stats/data/release_dates.json"

# Models missing from Moss's release_dates.json. Dates taken from METR's own
# TH1.1 logistic-fit table (eval-analysis-public/reports/time-horizon-1-1/
# data/wrangled/logistic_fits/headline.csv), matched via the runs.jsonl
# `alias` column: flamingo_2 == "GPT-5.3-Codex", claude_opus_4_6_inspect ==
# "Claude Opus 4.6 (Inspect)". Without these, the two most capable models
# fall out of the ability trend entirely and the doubling-time posterior is
# estimated without the most recent frontier points.
RELEASE_DATE_OVERRIDES = {
    "claude_opus_4_6_inspect": "2026-02-05",
    "flamingo_2": "2026-02-05",
}

# METR's per-agent logistic-fit summaries (p50 horizon per agent), used only
# for the optional SOTA-only restriction below.
DEFAULT_HEADLINE_CSV = (
    _SIBLINGS_ROOT / "eval-analysis-public"
    / "reports/time-horizon-1-1/data/wrangled/logistic_fits/headline.csv"
)


@dataclass
class ModelData:
    # --- Measurement layer (human timing observations) ---
    n_tasks: int
    task_ids: list[str]                 # length n_tasks
    log_dur: np.ndarray                 # (n_obs,) log(human_minutes)
    task_idx_obs: np.ndarray            # (n_obs,) int, 0..n_tasks-1
    is_estimate: np.ndarray             # (n_obs,) bool; True = "estimate" (no real timed run)
    is_censored: np.ndarray             # (n_obs,) bool; True = right-censored at log(time_limit)
    censor_log_time: np.ndarray         # (n_obs,) log(time_limit) where censored, else unused

    # --- IRT layer (model success/failure counts per task) ---
    n_models: int
    model_names: list[str]              # length n_models
    n_attempts: np.ndarray              # (n_irt,) int
    n_successes: np.ndarray             # (n_irt,) int
    task_idx_irt: np.ndarray            # (n_irt,) int, 0..n_tasks-1
    model_idx_irt: np.ndarray           # (n_irt,) int, 0..n_models-1

    # --- Ability trend (theta_m = beta0 + beta1 * t_m + u_m) ---
    t_model: np.ndarray                 # (n_models,) centered/scaled release date (years)
    has_date: np.ndarray                # (n_models,) bool

    # --- Optional task grouping (for structured residual difficulty eps) ---
    # task_family[i] is the integer family code of task i (0..n_families-1),
    # aligned with task_ids; family_names lists the family labels. None when
    # the grouping isn't loaded (e.g. the synthetic SBC design), in which case
    # the model falls back to the flat ZeroSumNormal eps.
    task_family: np.ndarray | None = None      # (n_tasks,) int
    family_names: list[str] | None = None      # length n_families


def _log_minutes(minutes: pd.Series) -> np.ndarray:
    vals = minutes.to_numpy(dtype=float)
    if np.any(vals <= 0):
        raise ValueError("Found non-positive human_minutes; cannot take log().")
    return np.log(vals)


def build_measurement_data(
    runs_filtered: pd.DataFrame, runs_raw: pd.DataFrame,
    include_human_failures: bool = False,
) -> tuple[list[str], dict]:
    """Build the per-observation duration arrays and the task_id -> index map.

    Two data facts drive the structure here (both verified against the
    2026-07 runs.jsonl snapshot and both fatal to the model if ignored):

    1. `human_minutes` is a *task-level annotation* (METR's canonical human
       time, equal to the geometric mean of successful baseline wall-times
       where those exist -- median ratio 0.998). It is identical on every row
       of a task. Feeding it per-row as if it were independent observations
       makes sigma_base collapse to 0 (a degenerate likelihood spike that
       freezes NUTS entirely; this was the root cause of the R-hat ~4 runs).
       The actual per-run observation is wall-clock time,
       (completed_at - started_at), for successful baseline runs. Within-task
       sd of log wall-time is ~0.4-0.6, i.e. real 1.5-2x measurement noise.

    2. The task universe must be the tasks models attempted (228), not the
       tasks with successful human runs (164). The 64 dropped tasks are
       almost all `human_source == "estimate"` and skew long (up to 30h), so
       dropping them biases the horizon trend. Tasks with no timed baseline
       run contribute their annotation as a single noisy "estimate"
       observation (one per task -- never repeated, for the same
       degenerate-likelihood reason as above).

    Estimate-source tasks with human runs (e.g. RE-Bench 8h time-boxed runs)
    do have wall-clock times, but those are budget-limited working times, not
    completion times; we deliberately use only the annotation for them.
    """
    nonhuman = runs_raw[
        (runs_raw["model"] != "human") & (runs_raw["cloned"].fillna(0) == 0)
    ]
    task_ids = sorted(nonhuman["task_id"].unique())
    task_index = {t: i for i, t in enumerate(task_ids)}
    task_set = set(task_ids)

    # --- Baseline observations: per-run wall-clock minutes ---
    base = runs_filtered[
        (runs_filtered["human_source"] == "baseline")
        & runs_filtered["task_id"].isin(task_set)
    ].copy()
    # started_at/completed_at are run-relative millisecond clocks (started_at
    # is 0 for many rows, not a Unix epoch), so the difference is the run's
    # wall-clock duration regardless of the absolute values.
    started = pd.to_numeric(base["started_at"], errors="coerce").fillna(0.0)
    completed = pd.to_numeric(base["completed_at"], errors="coerce")
    base["wall_minutes"] = (completed - started) / 1000.0 / 60.0
    base = base[base["wall_minutes"] > 0]

    log_dur_base = _log_minutes(base["wall_minutes"])
    task_idx_base = base["task_id"].map(task_index).to_numpy(dtype=int)

    # Censoring: a baseline row is right-censored if it has a real (>0)
    # time_limit and the recorded duration reaches it. In the current
    # runs.jsonl snapshot, human rows always carry time_limit == 0 (that
    # field is populated for *agent* compute budgets, not human timing), so
    # no row is actually censored here -- but the machinery is kept so the
    # model stays correct if a future data pull includes time-limited runs.
    time_limit = pd.to_numeric(base["time_limit"], errors="coerce").fillna(0.0).to_numpy()
    wall = base["wall_minutes"].to_numpy(dtype=float)
    is_censored_base = (time_limit > 0) & (wall >= time_limit * 0.999)
    censor_base = np.where(time_limit > 0, np.log(np.clip(time_limit, 1e-6, None)), 0.0)

    # --- Estimate observations: one annotation per task without a timed run ---
    ann = nonhuman.groupby("task_id")["human_minutes"].first()
    baseline_tasks = set(base["task_id"].unique())
    est_tasks = [t for t in task_ids if t not in baseline_tasks]
    est_minutes = ann.reindex(est_tasks)
    if est_minutes.isna().any():
        missing = est_minutes[est_minutes.isna()].index.tolist()
        raise ValueError(f"No human_minutes annotation for tasks: {missing[:5]}")
    log_dur_est = _log_minutes(est_minutes)
    task_idx_est = np.array([task_index[t] for t in est_tasks], dtype=int)

    # --- Failed baseline human runs as right-censored observations ---
    # Survivorship: the score_binarized==1 filter drops every FAILED human
    # attempt, so log_L is inferred from successful humans only. Failures
    # cluster on hard/long tasks (failed baseline median wall-time ~110-150
    # min vs ~5 min for successes), biasing the length scale downward exactly
    # where it matters. A failed attempt that spent wall-time w is evidence
    # that a successful completion would have taken *longer* than w, i.e. a
    # right-censored observation of the completion time at log(w). This is
    # self-weighting: a give-up at 7 min contributes P(T>7) ~ 1 (nearly
    # vacuous), while a genuine hard-task failure at 480 min contributes a
    # strong upward pull on log_L. (Assumption: failure ~ "would need more
    # time"; the ~47% of failures shorter than the task's own success median
    # are the give-ups that the survival function correctly down-weights.)
    if include_human_failures:
        hf = runs_raw[
            (runs_raw["model"] == "human")
            & (runs_raw["score_binarized"] == 0)
            & (runs_raw["human_source"] == "baseline")
            & (runs_raw["cloned"].fillna(0) == 0)
            & runs_raw["task_id"].isin(task_set)
        ].copy()
        hf_started = pd.to_numeric(hf["started_at"], errors="coerce").fillna(0.0)
        hf_completed = pd.to_numeric(hf["completed_at"], errors="coerce")
        hf["wall_minutes"] = (hf_completed - hf_started) / 1000.0 / 60.0
        hf = hf[hf["wall_minutes"] > 0]
        log_dur_fail = _log_minutes(hf["wall_minutes"])
        task_idx_fail = hf["task_id"].map(task_index).to_numpy(dtype=int)
        n_fail = len(log_dur_fail)
    else:
        log_dur_fail = np.array([], dtype=float)
        task_idx_fail = np.array([], dtype=int)
        n_fail = 0

    # --- Stack: uncensored baseline, then censored failures, then estimate ---
    log_dur = np.concatenate([log_dur_base, log_dur_fail, log_dur_est])
    task_idx_obs = np.concatenate([task_idx_base, task_idx_fail, task_idx_est])
    is_estimate = np.concatenate([
        np.zeros(len(log_dur_base), dtype=bool),
        np.zeros(n_fail, dtype=bool),
        np.ones(len(log_dur_est), dtype=bool),
    ])
    is_censored = np.concatenate([
        is_censored_base,
        np.ones(n_fail, dtype=bool),           # failures are right-censored
        np.zeros(len(log_dur_est), dtype=bool),
    ])
    # right-censor each failure at its own log wall-time
    censor_log_time = np.concatenate([censor_base, log_dur_fail, np.zeros(len(log_dur_est))])

    return task_ids, {
        "log_dur": log_dur,
        "task_idx_obs": task_idx_obs,
        "is_estimate": is_estimate,
        "is_censored": is_censored,
        "censor_log_time": censor_log_time,
    }


def build_irt_counts(
    runs_raw: pd.DataFrame, task_ids: list[str], task_index: dict[str, int]
) -> tuple[list[str], dict]:
    """Aggregate (model, task) attempt/success counts for the IRT layer,
    restricted to the task set defined by the measurement layer and to
    non-human models (human "ability" isn't estimated here -- human timing
    already enters through log(L_i))."""
    sub = runs_raw[
        (runs_raw["model"] != "human")
        & (runs_raw["task_id"].isin(task_ids))
        & (runs_raw["cloned"].fillna(0) == 0)
    ].copy()

    grouped = sub.groupby(["model", "task_id"])["score_binarized"].agg(["count", "sum"])
    grouped = grouped.reset_index().rename(columns={"count": "n", "sum": "s"})

    model_names = sorted(grouped["model"].unique())
    model_index = {m: i for i, m in enumerate(model_names)}

    n_attempts = grouped["n"].to_numpy(dtype=int)
    n_successes = grouped["s"].to_numpy(dtype=int)
    task_idx_irt = grouped["task_id"].map(task_index).to_numpy(dtype=int)
    model_idx_irt = grouped["model"].map(model_index).to_numpy(dtype=int)

    return model_names, {
        "n_attempts": n_attempts,
        "n_successes": n_successes,
        "task_idx_irt": task_idx_irt,
        "model_idx_irt": model_idx_irt,
    }


def build_theta_trend(
    model_names: list[str], release_dates_path: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    """Map each model to a centered/scaled release date (years since the
    mean release date across dated models). Models missing from the
    release-dates table (e.g. very recent/unlabeled entries) get has_date=0
    and t_model=0; the model treats their theta as a free random effect
    with no trend contribution, matching Moss's fallback for undated
    models."""
    with open(release_dates_path) as f:
        dates_raw: dict[str, str] = json.load(f)
    dates_raw = {**dates_raw, **RELEASE_DATE_OVERRIDES}

    parsed: dict[str, date] = {}
    for m, d in dates_raw.items():
        try:
            parsed[m] = date.fromisoformat(d)
        except (ValueError, TypeError):
            continue

    dated = [m for m in model_names if m in parsed]
    if not dated:
        raise ValueError("No models in the IRT set have a parseable release date.")
    ref_date = min(parsed[m] for m in dated)

    t_model = np.zeros(len(model_names), dtype=float)
    has_date = np.zeros(len(model_names), dtype=bool)
    for i, m in enumerate(model_names):
        if m in parsed:
            t_model[i] = (parsed[m] - ref_date).days / 365.25
            has_date[i] = True

    # Center the trend covariate among dated models only, for interpretability.
    t_model[has_date] -= t_model[has_date].mean()
    return t_model, has_date


def get_sota_models(
    runs_raw: pd.DataFrame,
    release_dates_path: str | Path = DEFAULT_RELEASE_DATES,
    headline_csv_path: str | Path = DEFAULT_HEADLINE_CSV,
) -> list[str]:
    """Return the runs.jsonl model keys that were SOTA at their release date,
    reproducing METR's own definition (eval-analysis-public
    `horizon/plot/bootstrap_ci.py::get_sota_agents`): a model is SOTA if its
    p50 horizon is >= the highest p50 among all models released on or before
    the same date. p50s come from METR's own logistic-fit table
    (`headline.csv`), matched via the runs.jsonl `alias` column; release
    dates come from the same table used for the trend
    (Moss's release_dates.json + RELEASE_DATE_OVERRIDES).

    On the current snapshot this yields exactly the 14 agents METR's
    published headline trendline uses
    (reports/time-horizon-1-1/metrics/trendline_ci/headline_from_2023.yaml),
    dropping the 6 non-frontier models: Claude 3 Opus, Claude 4 Opus,
    Claude 4.1 Opus, GPT-4 Turbo, GPT-5.1-Codex-Max, GPT-5.3-Codex.
    """
    nonhuman = runs_raw[
        (runs_raw["model"] != "human") & (runs_raw["cloned"].fillna(0) == 0)
    ]
    alias_by_model = nonhuman.groupby("model")["alias"].first().to_dict()

    headline = pd.read_csv(headline_csv_path)
    p50_by_alias = headline.set_index("agent")["p50"].to_dict()

    with open(release_dates_path) as f:
        dates_raw: dict[str, str] = json.load(f)
    dates_raw = {**dates_raw, **RELEASE_DATE_OVERRIDES}

    rows = []
    for m, alias in alias_by_model.items():
        if m not in dates_raw or alias not in p50_by_alias:
            raise ValueError(f"Model {m} (alias {alias!r}) missing a release date or p50.")
        p50 = float(p50_by_alias[alias])
        if not np.isfinite(p50):
            raise ValueError(f"Model {m} has non-finite p50.")
        rows.append({"model": m, "date": date.fromisoformat(dates_raw[m]), "p50": p50})

    df = pd.DataFrame(rows).sort_values("date")
    sota: list[str] = []
    highest = float("-inf")
    for d in df["date"].unique():
        on_date = df[df["date"] == d]
        highest = max(highest, on_date["p50"].max())
        sota.extend(on_date.loc[on_date["p50"] >= highest, "model"])
    return sorted(sota)


def load_model_data(
    runs_filtered_path: str | Path,
    runs_raw_path: str | Path = DEFAULT_RUNS_JSONL,
    release_dates_path: str | Path = DEFAULT_RELEASE_DATES,
    sota_only: bool = False,
    include_human_failures: bool = False,
) -> ModelData:
    """Assemble the ModelData container.

    With sota_only=True, the run set is first restricted to the models that
    were SOTA at their release date (METR's running-frontier definition, see
    get_sota_models). The task universe, IRT counts, and measurement
    observations are all rebuilt from the restricted runs, so the whole
    pipeline sees exactly the model set METR's headline trendline uses.
    The default (False) keeps the original all-models behavior.
    """
    runs_filtered = pd.read_parquet(runs_filtered_path) if str(runs_filtered_path).endswith(
        ".parquet"
    ) else pd.read_csv(runs_filtered_path)
    runs_raw = pd.read_json(runs_raw_path, lines=True)

    if sota_only:
        sota = set(get_sota_models(runs_raw, release_dates_path))
        runs_raw = runs_raw[
            (runs_raw["model"] == "human") | runs_raw["model"].isin(sota)
        ]

    task_ids, meas = build_measurement_data(
        runs_filtered, runs_raw, include_human_failures=include_human_failures
    )
    task_index = {t: i for i, t in enumerate(task_ids)}

    model_names, irt = build_irt_counts(runs_raw, task_ids, task_index)
    t_model, has_date = build_theta_trend(model_names, release_dates_path)

    # Task-family grouping (for structured eps): one family label per task,
    # aligned with task_ids, from the runs.jsonl `task_family` column.
    nonhuman = runs_raw[
        (runs_raw["model"] != "human") & (runs_raw["cloned"].fillna(0) == 0)
    ]
    fam_by_task = nonhuman.groupby("task_id")["task_family"].first()
    fam_labels = [str(fam_by_task.get(t, "unknown")) for t in task_ids]
    family_names = sorted(set(fam_labels))
    fam_index = {f: i for i, f in enumerate(family_names)}
    task_family = np.array([fam_index[f] for f in fam_labels], dtype=int)

    return ModelData(
        n_tasks=len(task_ids),
        task_ids=task_ids,
        log_dur=meas["log_dur"],
        task_idx_obs=meas["task_idx_obs"],
        is_estimate=meas["is_estimate"],
        is_censored=meas["is_censored"],
        censor_log_time=meas["censor_log_time"],
        n_models=len(model_names),
        model_names=model_names,
        n_attempts=irt["n_attempts"],
        n_successes=irt["n_successes"],
        task_idx_irt=irt["task_idx_irt"],
        model_idx_irt=irt["model_idx_irt"],
        t_model=t_model,
        has_date=has_date,
        task_family=task_family,
        family_names=family_names,
    )
