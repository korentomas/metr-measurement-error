"""Repo-root-relative paths shared by the package and the scripts/ CLIs."""

from __future__ import annotations

from pathlib import Path


def _find_repo_root() -> Path:
    """Walk up from this file to the directory containing pyproject.toml.

    Valid for the editable install `uv sync` produces (the package lives in
    src/ inside the repo). A non-editable install has no repo to find; every
    consumer takes explicit path arguments as the escape hatch.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(
        f"No pyproject.toml above {here}; pass explicit paths instead of "
        "relying on the repo-relative defaults."
    )


REPO_ROOT = _find_repo_root()
_SIBLINGS_ROOT = REPO_ROOT.parent

# In-repo locations.
PROCESSED_DATA = REPO_ROOT / "data" / "processed" / "runs_filtered.parquet"
OUTPUTS_DIR = REPO_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Sibling checkouts expected next to this repo (see README "Setup"):
# METR/eval-analysis-public and JonasMoss/metr-stats.
DEFAULT_RUNS_JSONL = (
    _SIBLINGS_ROOT / "eval-analysis-public"
    / "reports/time-horizon-1-1/data/raw/runs.jsonl"
)
DEFAULT_RELEASE_DATES = _SIBLINGS_ROOT / "metr-stats/data/release_dates.json"

# METR's per-agent logistic-fit summaries (p50 horizon per agent), used only
# for the optional SOTA-only restriction in data_prep.get_sota_models.
DEFAULT_HEADLINE_CSV = (
    _SIBLINGS_ROOT / "eval-analysis-public"
    / "reports/time-horizon-1-1/data/wrangled/logistic_fits/headline.csv"
)
