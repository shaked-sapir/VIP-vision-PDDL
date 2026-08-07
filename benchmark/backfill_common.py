"""Shared helpers for backfilling algorithm rows into existing experiment cells.

These utilities are source-agnostic (used by both ``backfill_baseline`` for
BaselineRunners and ``backfill_cdps`` for the anchored CDPS variant). They cover
cell identity parsing, problem-PDDL resolution, ``fold_result.json`` merging, and
``data_dir`` resolution from an experiment's ``run_params.json``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

_CELL_RE = re.compile(r"^fold(\d+)_numtrajs(\d+)_gtrate(\d+)$")

# Same metric keys run_fold uses for its null record.
NULL_METRIC_KEYS = [
    "precision_precs_pos", "precision_precs_neg",
    "precision_eff_pos", "precision_eff_neg", "precision_overall",
    "recall_precs_pos", "recall_precs_neg", "recall_eff_pos",
    "recall_eff_neg", "recall_overall", "solving_ratio",
    "false_plans_ratio", "unsolvable_ratio", "planning_timed_out_ratio",
    "pred_app_precision", "pred_app_recall",
    "pred_eff_precision", "pred_eff_recall",
]


def parse_cell_name(cell_name: str) -> Optional[Tuple[int, int, int]]:
    """Parse a ``fold{f}_numtrajs{n}_gtrate{g}`` cell name → (fold, num_trajs, gt_rate)."""
    m = _CELL_RE.match(cell_name)
    if not m:
        return None
    return tuple(int(g) for g in m.groups())  # type: ignore[return-value]


def is_cell_dir(path: Path) -> bool:
    """True if ``path`` is a ``fold*_numtrajs*_gtrate*`` cell directory."""
    return path.is_dir() and _CELL_RE.match(path.name) is not None


def find_problem_pddl(data_dir: Path, problem_name: str) -> Optional[Path]:
    """Locate a problem's PDDL file under the experiment's data directory.

    Covers the known data_dir layouts:
      <data_dir>/training/trajectories/<problem>/<problem>.pddl  (standard)
      <data_dir>/<problem>/<problem>.pddl
      <data_dir>/problems/<problem>/<problem>.pddl
      <data_dir>/<problem>.pddl
    """
    candidates = [
        data_dir / "training" / "trajectories" / problem_name / f"{problem_name}.pddl",
        data_dir / problem_name / f"{problem_name}.pddl",
        data_dir / "problems" / problem_name / f"{problem_name}.pddl",
        data_dir / f"{problem_name}.pddl",
    ]
    for c in candidates:
        if c.exists():
            return c
    for parent in (data_dir / "training" / "trajectories", data_dir, data_dir / "problems"):
        problem_dir = parent / problem_name
        if problem_dir.is_dir():
            pddl_files = sorted(problem_dir.glob("*.pddl"))
            if pddl_files:
                return pddl_files[0]
    return None


def merge_row(fold_result_path: Path, row: dict) -> None:
    """Merge one algorithm row into fold_result.json (replace same-algorithm)."""
    if fold_result_path.exists():
        backup = fold_result_path.with_suffix(".json.bak")
        if not backup.exists():
            shutil.copy2(fold_result_path, backup)
        rows = json.loads(fold_result_path.read_text())
    else:
        rows = []

    rows = [r for r in rows if r.get("algorithm") != row["algorithm"]]
    rows.append(row)
    fold_result_path.write_text(json.dumps(rows, indent=2))


def existing_algorithms(fold_result_path: Path) -> List[str]:
    """Return the algorithm names already present in a cell's fold_result.json."""
    if not fold_result_path.exists():
        return []
    try:
        return [r.get("algorithm") for r in json.loads(fold_result_path.read_text())]
    except (json.JSONDecodeError, AttributeError):
        return []


def read_data_dir_from_run_params(exp_dir: Path) -> Optional[Path]:
    """Return the ``data_dir`` recorded in an experiment's run_params.json.

    The experiment runner writes ``evaluation_results/run_params.json`` with the
    exact ``data_dir`` the experiment used. Reading it here keeps backfill pinned
    to the same data the cells were frozen against, instead of relying on the
    operator to re-supply the right (and easily-confused, e.g. ``*_fixed``)
    directory.

    Returns:
        The resolved ``data_dir`` path, or ``None`` if run_params.json is missing,
        unreadable, or has no ``data_dir`` field.
    """
    run_params = exp_dir / "evaluation_results" / "run_params.json"
    if not run_params.is_file():
        return None
    try:
        params = json.loads(run_params.read_text())
    except (json.JSONDecodeError, OSError) as err:
        print(f"[WARN] could not read {run_params}: {err}")
        return None
    data_dir = params.get("data_dir")
    if not data_dir:
        return None
    return Path(data_dir).resolve()


def read_run_params(exp_dir: Path) -> Optional[dict]:
    """Return the full parsed ``run_params.json`` for an experiment, or None."""
    run_params = exp_dir / "evaluation_results" / "run_params.json"
    if not run_params.is_file():
        return None
    try:
        return json.loads(run_params.read_text())
    except (json.JSONDecodeError, OSError) as err:
        print(f"[WARN] could not read {run_params}: {err}")
        return None


def resolve_data_dir(
    exp_dir: Path, override: Optional[Path]
) -> Tuple[Optional[Path], str]:
    """Resolve the data_dir for one experiment from an authoritative source.

    Resolution order (so the operator can just point backfill at the
    dashboard_config experiment dirs and get the right data with no confusion):
      1. ``--data-dir`` override, if given.
      2. run_params.json's ``data_dir`` — the exact folder the experiment used.

    No name-based guessing: experiment folders and data folders don't share a
    naming convention, and sibling data folders can differ, so a guess could
    silently resolve to the wrong-but-existing dir. If run_params.json is
    stale/missing, the caller skips loudly and the operator passes ``--data-dir``.

    Returns:
        ``(data_dir, source_label)``; ``data_dir`` is ``None`` if unresolved.
    """
    if override is not None:
        return override, "override"

    recorded = read_data_dir_from_run_params(exp_dir)
    if recorded is not None:
        return recorded, "run_params.json"

    return None, "unresolved"


def worker_init() -> None:
    """Pool-worker initializer: cap BLAS/torch thread pools so parallelism
    comes from the process count, not from N workers x M torch threads."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
