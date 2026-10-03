"""Run the anchored PI-SAM+MILP loop with a per-solve time limit on existing cells.

For each already-executed ``testing/fold*_numtrajs*_gtrate*`` cell this runs
``pisam_milp_loop`` with ``gt_anchoring: init_only`` on the cell's frozen
observations, unchanged, with one MILP solve capped at ``--solve-time-limit``
seconds, and evaluates it on the cell's own test problems and test states.

Nothing is written into the source experiment. Rows and artifacts go under
``--out-root``, in a tree that mirrors the source one::

    <out-root>/<domain>/<experiment>/testing/<cell>/
        <arm work dir>/             models and round logs
        fold_result.json            the new row only
        fold_info.json, domain_reference.pddl, provenance.json

Usage:
    python -m benchmark.backfill_anchored_solve_cap \
        --experiment-dir "benchmark/running_results/hanoi/large-corpora__L-sweep__v4__*" \
        --milp-config benchmark/run_config_large.yaml --solve-time-limit 60
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

from benchmark.algorithms import (
    PISAM_MILP_LOOP,
    milp_config_for,
    milp_work_subdir,
    pisam_milp_algorithm_name,
)
from benchmark.backfill_cdps import AlgorithmSpec, ExperimentSettings, resolve_experiment
from benchmark.backfill_common import (
    NULL_METRIC_KEYS,
    existing_algorithms,
    find_problem_pddl,
    find_test_states,
    is_cell_dir,
    merge_row,
    parse_cell_name,
    worker_init,
)
from benchmark.backfill_nogt_s0 import (
    PROVENANCE_FILENAME,
    frozen_observations_dir,
    output_cell,
    row_recorded_error,
)
from benchmark.experiment_running_helpers.resume import FOLD_RESULT_FILENAME
from benchmark.experiment_running_helpers.run_fold import run_cdps_phase
from benchmark.experiment_running_helpers.statistics import count_total_transitions_and_gt
from src.milp.converter import GtAnchoring
from src.plan_denoising.milp_denoiser.config import PisamMilpConfig, select_milp_block

DEFAULT_OUT_ROOT = Path("benchmark/running_results_anchored_solve60")
GT_STATE_INDICES = frozenset({0})

PreparedTrajectory = Tuple[Path, Path, Path, Set[int]]
CellTask = Tuple[Path, ExperimentSettings]


@dataclass(frozen=True)
class RunOptions:
    """Where results go and which existing rows are redone."""

    out_root: Path
    force: bool
    retry_errors: bool


def read_anchored_config(path: Optional[Path]) -> PisamMilpConfig:
    """The ``pisam_milp:`` block of a YAML file with ``gt_anchoring`` set to ``init_only``.

    Accepts a run_config.yaml (``shared.pisam_milp``), a file with a top-level
    ``pisam_milp:`` block, or the bare block. An ``ablations:`` sub-block is ignored.
    """
    if path is None:
        return replace(PisamMilpConfig(), gt_anchoring=GtAnchoring.INIT_ONLY)
    raw = yaml.safe_load(path.read_text()) or {}
    shared = raw.get("shared")
    block = select_milp_block(shared) if isinstance(shared, dict) else None
    if block is None:
        block = select_milp_block(raw)
    if block is None:
        block = raw
    base = {key: value for key, value in block.items() if key != "ablations"}
    return replace(PisamMilpConfig.from_dict(base), gt_anchoring=GtAnchoring.INIT_ONLY)


def resolve_arm(milp_config_path: Optional[Path], solve_time_limit: int) -> AlgorithmSpec:
    """The anchored loop arm with one solve capped, labelled ``...__solve=<n>``."""
    config = milp_config_for(PISAM_MILP_LOOP, read_anchored_config(milp_config_path))
    config = replace(config, time_limit_seconds=solve_time_limit)
    suffix = f"__solve={solve_time_limit}"
    return AlgorithmSpec(
        PISAM_MILP_LOOP,
        pisam_milp_algorithm_name(config) + suffix,
        milp_work_subdir(PISAM_MILP_LOOP, config) + suffix,
        config,
    )


def frozen_fold_trajectories(
    frozen_dir: Path, problem_dir: Path, fold_info: dict, cell_name: str,
) -> List[PreparedTrajectory]:
    """The cell's frozen files as fold tuples whose state 0 is ground truth.

    Raises:
        FileNotFoundError: If a trajectory's frozen files or problem file are missing.
    """
    prepared: List[PreparedTrajectory] = []
    for entry in fold_info.get("trajectories", []):
        problem = entry["problem"]
        trajectory = frozen_dir / entry["trajectory_file"]
        masking = frozen_dir / entry["masking_file"]
        problem_pddl = find_problem_pddl(problem_dir, problem)
        absent = [str(p) for p in (trajectory, masking) if not p.exists()]
        if problem_pddl is None:
            absent.append(f"problem PDDL of {problem} under {problem_dir}")
        if absent:
            raise FileNotFoundError(f"{cell_name}: missing {', '.join(absent)}")
        prepared.append((trajectory, masking, problem_pddl.resolve(), set(GT_STATE_INDICES)))
    return prepared


def test_problem_paths(problem_dir: Path, fold_info: dict) -> List[str]:
    """The cell's held-out test problems as PDDL paths."""
    paths: List[str] = []
    for problem in fold_info.get("test_problems", []):
        path = find_problem_pddl(problem_dir, problem)
        if path is None:
            print(f"    Warning: test problem PDDL not found for {problem}")
            continue
        paths.append(str(path.resolve()))
    return paths


def write_cell_scaffold(cell: Path, out_cell: Path, arm: AlgorithmSpec, n_trajectories: int) -> None:
    """Copy the cell's fold description and record what was run on it."""
    out_cell.mkdir(parents=True, exist_ok=True)
    for name in ("fold_info.json", "domain_reference.pddl"):
        shutil.copy2(cell / name, out_cell / name)
    provenance = {
        "source_cell": str(cell.resolve()),
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "observations": "frozen original_observations, unchanged",
        "gt_state_indices": sorted(GT_STATE_INDICES),
        "gt_anchoring": arm.milp_config.gt_anchoring.value,
        "solve_time_limit_seconds": arm.milp_config.time_limit_seconds,
        "n_trajectories": n_trajectories,
    }
    (out_cell / PROVENANCE_FILENAME).write_text(json.dumps(provenance, indent=2))


def backfill_cell(
    cell: Path, settings: ExperimentSettings, arm: AlgorithmSpec,
    options: RunOptions, dry_run: bool,
) -> str:
    """Run ``arm`` on one cell's frozen observations. Returns done | dry | skip | invalid."""
    parsed = parse_cell_name(cell.name)
    if parsed is None:
        return "invalid"
    fold, num_trajs, gt_rate = parsed
    if gt_rate != 0:
        print(f"  [SKIP] {cell.name}: gt_rate={gt_rate} is not supported")
        return "skip"

    fold_info_path = cell / "fold_info.json"
    if not fold_info_path.exists() or not (cell / "domain_reference.pddl").exists():
        print(f"  [SKIP] {cell.name}: missing fold_info.json or domain_reference.pddl")
        return "skip"
    fold_info = json.loads(fold_info_path.read_text())

    out_cell = output_cell(options.out_root, cell).resolve()
    fold_result_path = out_cell / FOLD_RESULT_FILENAME
    retry = options.retry_errors and row_recorded_error(fold_result_path, arm.row_name)
    if (not options.force and not retry
            and arm.row_name in existing_algorithms(fold_result_path)):
        print(f"  [SKIP] {cell.name}: {arm.row_name} row already present")
        return "skip"

    problems = test_problem_paths(settings.problem_dir, fold_info)
    if not problems:
        print(f"  [SKIP] {cell.name}: no test problem PDDLs found")
        return "skip"
    test_states = find_test_states(cell)

    if dry_run:
        print(f"  [DRY] {cell.name}: would run {arm.row_name} on "
              f"{len(fold_info.get('trajectories', []))} frozen trajectories into {out_cell} "
              f"(learn_timeout={settings.learn_timeout}s)"
              f"{'' if test_states else ' (no test states!)'}")
        return "dry"

    with frozen_observations_dir(cell) as frozen_dir:
        if frozen_dir is None:
            raise FileNotFoundError(f"{cell} has no original_observations/ and no tarball of it")
        trajectories = frozen_fold_trajectories(
            frozen_dir.resolve(), settings.problem_dir, fold_info, cell.name,
        )
        if not trajectories:
            print(f"  [SKIP] {cell.name}: no training trajectories")
            return "skip"
        write_cell_scaffold(cell, out_cell, arm, len(trajectories))
        row = _run_arm(
            out_cell, settings, arm, trajectories, problems, test_states,
            fold, num_trajs, gt_rate,
        )

    if row is None:
        print(f"  [SKIP] {cell.name}: {arm.row_name} produced no row")
        return "skip"
    merge_row(fold_result_path, row)
    print(f"  [{arm.row_name}] {cell.name}: row written to {fold_result_path}")
    return "done"


def _run_arm(
    out_cell: Path, settings: ExperimentSettings, arm: AlgorithmSpec,
    trajectories: List[PreparedTrajectory], problems: List[str],
    test_states: Optional[Path], fold: int, num_trajs: int, gt_rate: int,
) -> Optional[dict]:
    """Learn and evaluate inside the output cell; returns the result row."""
    total_transitions, total_gt = count_total_transitions_and_gt(trajectories)
    original_cwd = os.getcwd()
    os.chdir(out_cell)
    try:
        print(f"  [{arm.row_name}] {out_cell.name}: running {arm.key}...")
        return run_cdps_phase(
            anchor_endpoints=False,
            algo_name=arm.row_name,
            cdps_work_dir=out_cell / arm.work_subdir,
            trajectories=trajectories,
            gt_source_indices=None,
            pre_built_observations=None,
            domain_ref_path=out_cell / "domain_reference.pddl",
            testing_dir=out_cell.parent,
            bench_name=settings.bench_name,
            fold=fold,
            num_trajectories=num_trajs,
            gt_rate=gt_rate,
            test_problem_paths=problems,
            null_metrics={k: None for k in NULL_METRIC_KEYS},
            total_transitions=total_transitions,
            total_gt_transitions=total_gt,
            conflict_search_timeout=settings.learn_timeout,
            planning_timeout=settings.planning_timeout,
            events_tracing=False,
            test_states_path=str(test_states.resolve()) if test_states is not None else None,
            milp_config=arm.milp_config,
            **settings.cdps_search_params,
        )
    finally:
        os.chdir(original_cwd)


def _backfill_cell_worker(
    task: CellTask, arm: AlgorithmSpec, options: RunOptions,
) -> Tuple[str, str]:
    """Process-pool entry point: one cell, never raising."""
    cell, settings = task
    try:
        status = backfill_cell(cell, settings, arm, options, dry_run=False)
    except Exception as error:
        return str(cell), f"error: {error}"
    return str(cell), status


def gather_tasks(args: argparse.Namespace, override_data_dir: Optional[Path]) -> List[CellTask]:
    """Expand the experiment dirs into per-cell work items."""
    tasks: List[CellTask] = []
    for exp_dir in args.experiment_dir:
        exp_dir = exp_dir.resolve()
        testing = exp_dir / "testing"
        if not testing.is_dir():
            print(f"[SKIP] {exp_dir}: no testing/ directory")
            continue
        settings = resolve_experiment(exp_dir, override_data_dir, args)
        if settings is None:
            continue
        cells = sorted(d for d in testing.iterdir() if is_cell_dir(d))
        if args.cells:
            cells = [c for c in cells if args.cells in c.name]
        print(f"  → {len(cells)} cells")
        tasks.extend((cell, settings) for cell in cells)
    return tasks


def _summarize(statuses: Dict[str, str]) -> int:
    """Print the run's summary; returns the number of cells that raised."""
    finished = sum(1 for s in statuses.values() if s == "done")
    skipped = sum(1 for s in statuses.values() if s in ("skip", "invalid"))
    errors = {c: s for c, s in statuses.items() if s.startswith("error")}
    print(f"\nSummary: {finished} finished, {skipped} skipped, {len(errors)} errors")
    for cell_str, error in errors.items():
        print(f"  ERROR {cell_str}: {error}")
    return len(errors)


def run_tasks(
    tasks: List[CellTask], arm: AlgorithmSpec, options: RunOptions, workers: int,
) -> int:
    """Run every cell, in a process pool when ``workers`` > 1; returns the error count."""
    statuses: Dict[str, str] = {}
    if workers == 1:
        for i, task in enumerate(tasks, start=1):
            cell_str, status = _backfill_cell_worker(task, arm, options)
            statuses[cell_str] = status
            print(f"[{i}/{len(tasks)}] {Path(cell_str).name}: {status}", flush=True)
        return _summarize(statuses)
    print(f"\nRunning {len(tasks)} cells with {workers} workers...")
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as executor:
        futures = {
            executor.submit(_backfill_cell_worker, task, arm, options): task[0] for task in tasks
        }
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                cell_str, status = future.result()
            except Exception as error:
                cell_str, status = str(futures[future]), f"error: {error}"
            statuses[cell_str] = status
            print(f"[{i}/{len(futures)}] {Path(cell_str).name}: {status}", flush=True)
    return _summarize(statuses)


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface."""
    ap = argparse.ArgumentParser(
        description="Run the anchored pisam_milp_loop arm with a per-solve time limit on "
                    "existing cells' frozen observations, writing under a separate results root.")
    ap.add_argument("--milp-config", type=Path, default=None,
                    help="YAML holding the pisam_milp settings (a run_config.yaml, a file "
                         "with a pisam_milp: block, or the bare block). gt_anchoring is "
                         "forced to init_only and an ablations: sub-block is ignored.")
    ap.add_argument("--experiment-dir", type=Path, nargs="+", required=True,
                    help="Source experiment director(y/ies) containing testing/.")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                    help=f"Root of the mirrored results tree. Default {DEFAULT_OUT_ROOT}.")
    ap.add_argument("--solve-time-limit", type=int, required=True,
                    help="Cap, in seconds, on one MILP solve; a solve that reaches it "
                         "returns its best solution so far. Adds __solve=<n> to the row label.")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="Override the data_dir read from each experiment's run_params.json.")
    ap.add_argument("--domain", default=None,
                    help="Domain name for the result rows (default: from the experiment path).")
    ap.add_argument("--learn-timeout", type=int, default=None,
                    help="Override run_params.json's learning_timeout_seconds.")
    ap.add_argument("--planning-timeout", type=int, default=None,
                    help="Override run_params.json's planning_timeout_seconds.")
    ap.add_argument("--cells", default=None,
                    help="Only process cell dirs whose name contains this substring.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run and replace the arm's row even if present.")
    ap.add_argument("--retry-errors", action="store_true",
                    help="Re-run cells whose row for this arm recorded an error; cells "
                         "with a clean row are still skipped.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="Cells to process in parallel. Dry runs are sequential.")
    ap.set_defaults(frame_axiom_mode=None)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    override_data_dir = args.data_dir.resolve() if args.data_dir else None
    if override_data_dir and not override_data_dir.is_dir():
        raise SystemExit(f"--data-dir does not exist: {override_data_dir}")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.solve_time_limit < 1:
        raise SystemExit("--solve-time-limit must be >= 1")

    arm = resolve_arm(args.milp_config, args.solve_time_limit)
    options = RunOptions(
        out_root=args.out_root.resolve(), force=args.force, retry_errors=args.retry_errors,
    )
    print(f"Arm {arm.key} as row '{arm.row_name}' (work dir: {arm.work_subdir}/) "
          f"→ {options.out_root}")

    tasks = gather_tasks(args, override_data_dir)
    if not tasks:
        print("Nothing to do.")
        return
    if args.dry_run:
        for cell, settings in tasks:
            backfill_cell(cell, settings, arm, options, dry_run=True)
        return
    if run_tasks(tasks, arm, options, min(args.workers, len(tasks))):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
