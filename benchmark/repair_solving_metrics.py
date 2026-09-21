"""Re-score the solving metrics of rows the planner could not classify.

A learned model with a no-effect operator used to make Fast Downward reject the
whole domain, which AMLGym's ``problem_solving`` counts in no bucket, so the row
holds 0 for solved, false plans, unsolvable and timed out alike. Evaluation now
plans on a copy without such operators
(``experiment_running_helpers/planning_copy.py``); this tool applies the same
rule to rows written before that, from the models already on disk. No learning
is repeated and no other field changes.

Two kinds of record are repaired, per fold instance:

  - rows of ``fold_result.json`` — the model is ``learned_domain_<algorithm>.pddl``
    somewhere under the cell;
  - entries of every ``all_solutions_metrics.json`` under the cell (the per-model
    scores of a CDPS-family arm) — the model is
    ``conflict_free_models/conflict_free_model_<i>/model.pddl`` beside it, or
    ``final_model/model.pddl`` for index -1.

Usage:
    python -m benchmark.repair_solving_metrics \
        --experiment-dir benchmark/running_results/*/simulation-final-run__mask=*__noise=* --dry-run
    python -m benchmark.repair_solving_metrics --experiment-dir ... --workers 5

Do not run it while a backfill is writing into the same experiments: a row merged
into a cell after this tool has passed that cell is not repaired.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from benchmark.backfill_common import (
    find_problem_pddl,
    is_cell_dir,
    merge_row,
    resolve_data_dir,
    resolve_problem_dir,
    worker_init,
)
from benchmark.experiment_running_helpers.planning_copy import solving_metrics
from benchmark.experiment_running_helpers.resume import FOLD_RESULT_FILENAME
from src.utils.atomic_json import write_json_atomic

ALL_SOLUTIONS_FILENAME = "all_solutions_metrics.json"
_OUTCOME_FIELDS = (
    "solving_ratio", "false_plans_ratio", "unsolvable_ratio", "planning_timed_out_ratio",
)


def is_unclassified(record: dict) -> bool:
    """True when every recorded planning outcome is exactly zero."""
    return all(record.get(field) == 0.0 for field in _OUTCOME_FIELDS)


def _test_problems(cell: Path, problem_dir: Path) -> List[str]:
    fold_info = json.loads((cell / "fold_info.json").read_text())
    paths = (find_problem_pddl(problem_dir, name) for name in fold_info.get("test_problems", []))
    return [str(path) for path in paths if path is not None]


def _row_model(cell: Path, algorithm: str) -> Optional[Path]:
    """The saved model of one ``fold_result.json`` row."""
    return next(iter(sorted(cell.rglob(f"learned_domain_{algorithm}.pddl"))), None)


def _solution_model(metrics_file: Path, solution_index: int) -> Optional[Path]:
    """The saved model of one ``all_solutions_metrics.json`` entry."""
    work_dir = metrics_file.parent
    if solution_index == -1:
        candidate = work_dir / "conflict_free_models" / "final_model" / "model.pddl"
    else:
        candidate = (
            work_dir / "conflict_free_models"
            / f"conflict_free_model_{solution_index}" / "model.pddl"
        )
    return candidate if candidate.exists() else None


def _rescore(model: Path, cell: Path, problems: List[str], timeout: int) -> Dict[str, object]:
    return solving_metrics(model, cell / "domain_reference.pddl", problems, timeout=timeout)


def repair_cell(cell: Path, problem_dir: Path, timeout: int, dry_run: bool) -> Dict[str, int]:
    """Repair one fold instance. Returns counts of what was found and fixed."""
    counts = {"rows": 0, "rows_fixed": 0, "solutions": 0, "solutions_fixed": 0, "no_model": 0}
    fold_result = cell / FOLD_RESULT_FILENAME
    if not fold_result.exists():
        return counts
    problems = _test_problems(cell, problem_dir)
    if not problems:
        return counts

    # AMLGym's problem_solving writes ./tmp into the cwd.
    original_cwd = os.getcwd()
    os.chdir(cell)
    try:
        for row in json.loads(fold_result.read_text()):
            if not is_unclassified(row):
                continue
            counts["rows"] += 1
            model = _row_model(cell, row["algorithm"])
            if model is None:
                counts["no_model"] += 1
                continue
            if dry_run:
                continue
            merge_row(fold_result, {**row, **_rescore(model, cell, problems, timeout)})
            counts["rows_fixed"] += 1

        for metrics_file in sorted(cell.rglob(ALL_SOLUTIONS_FILENAME)):
            entries = json.loads(metrics_file.read_text())
            changed = False
            for position, entry in enumerate(entries):
                if not is_unclassified(entry):
                    continue
                counts["solutions"] += 1
                model = _solution_model(metrics_file, entry.get("solution_index", 0))
                if model is None:
                    counts["no_model"] += 1
                    continue
                if dry_run:
                    continue
                entries[position] = {**entry, **_rescore(model, cell, problems, timeout)}
                counts["solutions_fixed"] += 1
                changed = True
            if changed:
                write_json_atomic(metrics_file, entries)
    finally:
        os.chdir(original_cwd)
    return counts


def _worker(cell: str, problem_dir: str, timeout: int, dry_run: bool) -> Tuple[str, Dict[str, int]]:
    return cell, repair_cell(Path(cell), Path(problem_dir), timeout, dry_run)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--experiment-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="Override the data_dir each experiment's run_params.json records.")
    ap.add_argument("--planning-timeout", type=int, default=60)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true",
                    help="Count what would be re-scored; write nothing.")
    args = ap.parse_args()

    tasks: List[Tuple[Path, Path]] = []
    for exp_dir in args.experiment_dir:
        exp_dir = exp_dir.resolve()
        testing = exp_dir / "testing"
        if not testing.is_dir():
            continue
        data_dir, _ = resolve_data_dir(exp_dir, args.data_dir.resolve() if args.data_dir else None)
        if data_dir is None or not data_dir.is_dir():
            print(f"[SKIP] {exp_dir.name}: data_dir unresolved")
            continue
        problem_dir = resolve_problem_dir(exp_dir, data_dir)
        tasks.extend((cell, problem_dir) for cell in sorted(testing.iterdir()) if is_cell_dir(cell))

    totals = {"rows": 0, "rows_fixed": 0, "solutions": 0, "solutions_fixed": 0, "no_model": 0}
    workers = max(1, min(args.workers, len(tasks) or 1))
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        futures = [
            pool.submit(_worker, str(cell), str(problem_dir), args.planning_timeout, args.dry_run)
            for cell, problem_dir in tasks
        ]
        for future in as_completed(futures):
            cell, counts = future.result()
            if counts["rows"] or counts["solutions"]:
                print(f"  {Path(cell).parent.parent.name}/{Path(cell).name}: {counts}", flush=True)
            for key, value in counts.items():
                totals[key] += value

    verb = "would re-score" if args.dry_run else "re-scored"
    print(f"\n{len(tasks)} fold instances scanned; {verb}: {totals}")


if __name__ == "__main__":
    main()
