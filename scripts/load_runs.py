"""Filter METR's runs.jsonl down to the human timing observations and write
data/processed/runs_filtered.parquet. Full documentation of the source data
and filter lives in metr_measurement_error.load_runs.

Usage:
    uv run python scripts/load_runs.py \
        --input /path/to/runs.jsonl \
        --output data/processed/runs_filtered.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

from metr_measurement_error.load_runs import filter_runs, load_raw, summarize
from metr_measurement_error.paths import DEFAULT_RUNS_JSONL, PROCESSED_DATA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_RUNS_JSONL, help="Path to runs.jsonl")
    parser.add_argument(
        "--output",
        default=str(PROCESSED_DATA),
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
