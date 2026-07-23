"""Load and filter METR human-baseline run data for the Bayesian
measurement-error time-horizon model.

Source data: METR's public eval-analysis-public repo, `runs.jsonl` from the
time-horizon-1-1 report. Each row is one (model, task) run, annotated with
`human_minutes` (the human baseline time for that task) and `human_source`
(`"baseline"` for a real timed human run, `"estimate"` for an
expert-estimated duration when no timed run exists).

`runs.jsonl` is model-run-centric: every model's attempt at a task carries a
copy of that task's human timing info (`human_minutes`, `human_source`,
etc.), duplicated once per model. The actual human timing *observations* --
one row per person who actually attempted (or, for a handful of tasks,
estimated) the task -- live under `model == "human"`. That subset is what
this loader keeps; everything else is a model's own run and is irrelevant
to the measurement-error layer, which only cares about how long humans took.

Filter applied (matches the population used to fit METR's headline
time-horizon model):
    model == "human"  AND  score_binarized == 1  AND  completed_at > 0

`score_binarized == 1` keeps only human attempts that succeeded (this
model's IRT layer only uses successes as timing anchors); `completed_at > 0`
keeps only rows with a real wall-clock completion timestamp (completed_at
is a Unix-ms timestamp; 0/None means the row has no completion record,
e.g. an abandoned attempt).

Expected result on the reference dataset (2026-07 snapshot of
time-horizon-1-1/data/raw/runs.jsonl): 554 human rows across 164 distinct
tasks. Of those 164 tasks, 136 have >= 2 rows in the filtered set (i.e. at
least two independent human observations -- baseline timings and/or
estimates combined) and so support estimating both a task length L_i and a
measurement-noise sigma from repeat observations; the other 28 tasks have
exactly one observation.

Usage:
    uv run python data/load_runs.py \
        --input /path/to/runs.jsonl \
        --output data/processed/runs_filtered.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Sibling checkout expected next to this repo (see README "Setup"):
# METR/eval-analysis-public.
DEFAULT_INPUT = (
    Path(__file__).resolve().parent.parent.parent / "eval-analysis-public"
    / "reports/time-horizon-1-1/data/raw/runs.jsonl"
)

REQUIRED_COLUMNS = [
    "task_id",
    "task_family",
    "run_id",
    "alias",
    "model",
    "score_cont",
    "score_binarized",
    "fatal_error_from",
    "human_minutes",
    "human_score",
    "human_source",
    "task_source",
    "generation_cost",
    "human_cost",
    "time_limit",
    "started_at",
    "completed_at",
    "task_version",
    "tokens_count",
    "cloned",
    "scaffold",
    "equal_task_weight",
    "invsqrt_task_weight",
]


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"runs.jsonl missing expected columns: {sorted(missing)}")
    return df


def filter_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only human timing observations: model=='human', successful,
    with a real completion timestamp.

    `completed_at` is stored as a float Unix-ms timestamp (or NaN/None/0 when
    the row has no real completion record). We treat NaN as not-completed.
    """
    completed_at = pd.to_numeric(df["completed_at"], errors="coerce").fillna(0)
    mask = (df["model"] == "human") & (df["score_binarized"] == 1) & (completed_at > 0)
    out = df.loc[mask].copy()
    out["completed_at"] = completed_at.loc[mask]
    # run_id is a mixed int/str column upstream; normalize to string so
    # downstream parquet writes don't choke on a mixed-type object column.
    out["run_id"] = out["run_id"].astype(str)
    return out


def summarize(df: pd.DataFrame) -> dict:
    n_rows = len(df)
    n_tasks = df["task_id"].nunique()

    # "Multi-attempt" = tasks with >=2 human observations in the filtered
    # set, counting baseline (real-timed) and estimate rows together.
    row_counts = df.groupby("task_id").size()
    n_multi_attempt = int((row_counts >= 2).sum())

    is_baseline = df["human_source"] == "baseline"

    return {
        "n_rows": n_rows,
        "n_tasks": n_tasks,
        "n_multi_attempt_tasks": n_multi_attempt,
        "n_baseline_rows": int(is_baseline.sum()),
        "n_estimate_rows": int((~is_baseline).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to runs.jsonl")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "processed" / "runs_filtered.parquet"),
        help="Output path (.parquet or .csv)",
    )
    args = parser.parse_args()

    raw = load_raw(args.input)
    filtered = filter_runs(raw)
    stats = summarize(filtered)

    print(f"Loaded {len(raw)} raw rows from {args.input}")
    print(
        f"Filtered (score_binarized==1 & completed_at>0): "
        f"{stats['n_rows']} rows / {stats['n_tasks']} tasks "
        f"({stats['n_multi_attempt_tasks']} tasks with >=2 timed attempts)"
    )
    print(
        f"  baseline (real-timed) rows: {stats['n_baseline_rows']}, "
        f"estimate-only rows: {stats['n_estimate_rows']}"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".csv":
        filtered.to_csv(out_path, index=False)
    else:
        filtered.to_parquet(out_path, index=False)
    print(f"Wrote filtered data to {out_path}")


if __name__ == "__main__":
    main()
