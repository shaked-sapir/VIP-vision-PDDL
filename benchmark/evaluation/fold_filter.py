"""Filter fold directories by CFM solving ratio.

Given a parent experiment directory, finds all foldX_numtrajsY_gtrateZ
subdirectories that have at least one model (CFM or fallback) with
solving_ratio > 0, and reports whether the solving model is a legitimate
conflict-free model or the partial fallback.

Usage:
    python -m benchmark.evaluation.fold_filter <experiment_dir>

Example:
    python -m benchmark.evaluation.fold_filter \\
        benchmark/data/new_experiments/blocksworld/TO=300__largest__cv5
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FoldSolvingResult:
    """Outcome for a single fold directory.

    Attributes:
        fold_dir: Absolute path to the foldX_numtrajsY_gtrateZ directory.
        best_solving_ratio: Highest solving_ratio across all models in this fold.
        best_solution_index: solution_index of the model that achieved it.
            >= 0 → legitimate CFM.  -1 → fallback / partial model.
        solving_count_under_threshold: Number of CFMs with solving_ratio > 0
            discovered within ``time_threshold_seconds`` of conflict search.
    """
    fold_dir: Path
    best_solving_ratio: float
    best_solution_index: int
    solving_count_under_threshold: int = 0
    time_threshold_seconds: float = 300.0

    @property
    def is_legitimate_cfm(self) -> bool:
        return self.best_solution_index >= 0

    @property
    def model_type(self) -> str:
        return "CFM" if self.is_legitimate_cfm else "fallback (partial model)"

    def __str__(self) -> str:
        threshold = int(self.time_threshold_seconds)
        return (
            f"{self.fold_dir.name}: "
            f"solving_ratio={self.best_solving_ratio:.2f} "
            f"[{self.model_type}, solution_index={self.best_solution_index}] "
            f"solving_under_{threshold}s={self.solving_count_under_threshold}"
        )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _load_all_solutions(fold_dir: Path) -> Optional[List[dict]]:
    """Load all_solutions_metrics.json from a fold directory.

    Returns None if the file is missing or malformed.
    """
    metrics_path = fold_dir / "all_solutions_metrics.json"
    if not metrics_path.exists():
        logger.debug("No all_solutions_metrics.json in %s — skipping", fold_dir.name)
        return None
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read %s: %s — skipping", metrics_path, e)
        return None


def _best_solving_entry(entries: List[dict]) -> Optional[dict]:
    """Return the entry with the highest solving_ratio > 0, or None."""
    candidates = [e for e in entries if e.get("solving_ratio", 0) > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e["solving_ratio"])


def _load_solutions_log(fold_dir: Path) -> Optional[List[dict]]:
    """Load conflict_free_solutions_log.json from a fold directory."""
    log_path = fold_dir / "conflict_free_solutions_log.json"
    if not log_path.exists():
        return None
    try:
        with open(log_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read %s: %s — skipping timing", log_path, e)
        return None


def _count_solving_cfms_under_threshold(
    entries: List[dict],
    solutions_log: Optional[List[dict]],
    threshold_seconds: float,
) -> int:
    """Count CFMs with solving_ratio > 0 found within threshold_seconds."""
    if not solutions_log:
        return 0

    metrics_by_idx = {e["solution_index"]: e for e in entries if "solution_index" in e}
    count = 0
    for sol in solutions_log:
        idx = sol.get("index")
        if idx is None or idx < 0:
            continue
        metrics = metrics_by_idx.get(idx)
        if metrics is None or metrics.get("solving_ratio", 0) <= 0:
            continue
        wall_time = sol.get("wall_time_so_far")
        if wall_time is not None and wall_time <= threshold_seconds:
            count += 1
    return count


def get_folds_with_solving_cfm(
    experiment_dir: Path,
    time_threshold_seconds: float = 300.0,
) -> List[FoldSolvingResult]:
    """Return fold results where at least one model has solving_ratio > 0.

    Scans ``<experiment_dir>/testing/fold*`` directories.  Both legitimate
    CFMs (solution_index >= 0) and the fallback partial model
    (solution_index == -1) are considered; the result explicitly marks which
    kind achieved the best solving ratio.

    Args:
        experiment_dir: Parent experiment config directory (the one that
            contains a ``testing/`` subdirectory).

    Returns:
        Sorted list of ``FoldSolvingResult``, one per qualifying fold,
        ordered by fold directory name.
    """
    testing_dir = experiment_dir / "testing"
    if not testing_dir.exists():
        raise FileNotFoundError(
            f"Expected a 'testing/' subdirectory under {experiment_dir}, "
            f"but none was found."
        )

    fold_dirs = sorted(
        [d for d in testing_dir.iterdir() if d.is_dir() and d.name.startswith("fold")],
        key=lambda d: d.name,
    )

    results: List[FoldSolvingResult] = []
    for fold_dir in fold_dirs:
        entries = _load_all_solutions(fold_dir)
        if entries is None:
            continue

        best = _best_solving_entry(entries)
        if best is None:
            continue

        solutions_log = _load_solutions_log(fold_dir)
        solving_under_threshold = _count_solving_cfms_under_threshold(
            entries, solutions_log, time_threshold_seconds,
        )

        results.append(FoldSolvingResult(
            fold_dir=fold_dir,
            best_solving_ratio=best["solving_ratio"],
            best_solution_index=best.get("solution_index", -1),
            solving_count_under_threshold=solving_under_threshold,
            time_threshold_seconds=time_threshold_seconds,
        ))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List fold directories with at least one solving model (CFM or fallback)."
    )
    parser.add_argument(
        "experiment_dir", type=Path,
        help="Parent experiment config directory (contains a testing/ subdirectory).",
    )
    parser.add_argument(
        "--time-threshold", type=float, default=300.0,
        help="Count solving CFMs discovered within this many seconds (default: 300).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    args = _parse_args()

    results = get_folds_with_solving_cfm(
        args.experiment_dir, time_threshold_seconds=args.time_threshold,
    )

    if not results:
        print("No folds with solving_ratio > 0 found.")
    else:
        cfm_count = sum(1 for r in results if r.is_legitimate_cfm)
        fallback_count = len(results) - cfm_count
        print(f"Found {len(results)} fold(s) with solving_ratio > 0 "
              f"({cfm_count} via CFM, {fallback_count} via fallback):\n")
        for r in results:
            print(f"  {r}")
