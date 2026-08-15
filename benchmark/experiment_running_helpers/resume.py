"""Resume support for aborted/failed experiments.

A *fold instance* — one ``(fold, num_trajs, gt_rate)`` triple, mapped to a single
directory under ``testing/`` — is the atomic unit of work. When a fold finishes,
:func:`save_fold_result` writes a ``fold_result.json`` marker holding that fold's
per-algorithm result rows. On resume, completed folds are skipped and their rows
reloaded. These per-cell markers are the single source of truth for reports,
which are generated on demand (see ``collect_results`` / ``experiment_report``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

FOLD_RESULT_FILENAME = "fold_result.json"

# run_params.json keys that may legitimately differ between the original run and
# a resume without invalidating already-computed folds (the grid may be extended;
# the timestamp always changes; fold-level cluster jobs each run a different
# subset of folds of the same experiment).
RESUME_IGNORED_PARAMS = frozenset({
    "timestamp",
    "num_trajectories_list",
    "gt_rate_percentages",
    "folds",
})

# run_params.json keys renamed in place; old cells hold the old spelling, so both
# sides are compared under the current one.
_RENAMED_PARAMS = {
    "run_cdps_milp": "run_pisam_milp",
    "run_cdps_milp_loop": "run_pisam_milp_loop",
}


def fold_instance_dir(testing_dir: Path, fold: int, num_trajs: int, gt_rate: int) -> Path:
    """Return the canonical directory for one ``(fold, num_trajs, gt_rate)`` instance."""
    return testing_dir / f"fold{fold}_numtrajs{num_trajs}_gtrate{gt_rate}"


def save_fold_result(fold_dir: Path, rows: List[dict]) -> None:
    """Persist a completed fold's result rows as its completion marker."""
    with open(fold_dir / FOLD_RESULT_FILENAME, "w") as f:
        json.dump(rows, f, indent=2)


def try_load_fold_result(fold_dir: Path) -> Optional[List[dict]]:
    """Return a completed fold's cached result rows, or ``None`` if not complete."""
    marker = fold_dir / FOLD_RESULT_FILENAME
    if not marker.exists():
        return None
    with open(marker) as f:
        return json.load(f)


def _canonical_params(params: dict) -> dict:
    """A run_params dict with renamed keys mapped to their current spelling."""
    return {_RENAMED_PARAMS.get(key, key): value for key, value in params.items()}


def run_params_conflicts(existing: dict, current: dict) -> List[str]:
    """Return hyperparameter keys whose values differ between two run_params dicts.

    Keys in :data:`RESUME_IGNORED_PARAMS` are excluded, and keys in
    :data:`_RENAMED_PARAMS` are compared under their current spelling. Used to
    strictly abort a resume when the current config would mix incompatible
    results into an existing experiment directory.
    """
    existing = _canonical_params(existing)
    current = _canonical_params(current)
    keys = (set(existing) | set(current)) - RESUME_IGNORED_PARAMS
    return sorted(k for k in keys if existing.get(k) != current.get(k))
