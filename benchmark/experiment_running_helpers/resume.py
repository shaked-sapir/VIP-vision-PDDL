"""Resume support for aborted/failed experiments.

A *fold instance* — one ``(fold, num_trajs, gt_rate)`` triple, mapped to a single
directory under ``testing/`` — is the atomic unit of work. When a fold finishes,
:func:`save_fold_result` writes a ``fold_result.json`` marker holding that fold's
result rows. On resume, completed folds are skipped and their rows reloaded, so
the aggregated CSV/Excel reports still include every fold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

FOLD_RESULT_FILENAME = "fold_result.json"

# run_params.json keys that may legitimately differ between the original run and
# a resume without invalidating already-computed folds (the grid may be extended;
# the timestamp always changes).
RESUME_IGNORED_PARAMS = frozenset({
    "timestamp",
    "num_trajectories_list",
    "gt_rate_percentages",
})


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


def run_params_conflicts(existing: dict, current: dict) -> List[str]:
    """Return hyperparameter keys whose values differ between two run_params dicts.

    Keys in :data:`RESUME_IGNORED_PARAMS` are excluded. Used to strictly abort a
    resume when the current config would mix incompatible results into an
    existing experiment directory.
    """
    keys = (set(existing) | set(current)) - RESUME_IGNORED_PARAMS
    return sorted(k for k in keys if existing.get(k) != current.get(k))
