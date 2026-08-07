"""Backfill the anchored CDPS variant (``CDPS_ANCHORED``) into existing cells.

This is the CDPS counterpart of ``backfill_baseline``. It reconstructs the
init+final-anchored trajectories for each already-executed
``testing/fold*_numtrajs*_gtrate*`` cell, runs the *same* file-based anchored
conflict-directed patch search that a live image run uses, and merges the
resulting ``CDPS_ANCHORED`` row into the cell's ``fold_result.json`` — paired
with the existing CDPS/PISAM/baseline rows.

Why a separate script (and why simulated cells work here even though the live
runner excludes them): the live run path guards ``cdps_anchored`` for the
simulated data source because its observations are built in-memory and never
land in the on-disk problem-dir layout the anchored prep needs. For backfill,
every input already exists on disk:

  - the frozen degraded observation the cell learned from
    (``cell/original_observations/original_observation_<problem>.trajectory``);
  - the clean ground-truth trajectory that produced it
    (``<data_dir>/gt_trajectories/<problem>/<problem>_trajectory.json``), the
    source of the anchored (unmasked) final state.

We stage those into the problem-dir layout ``prepare_anchored_fold_trajectories``
expects, then reuse ``run_cdps_phase(anchor_endpoints=True,
pre_built_observations=None)`` — presenting the data as file-based, which
sidesteps the in-memory guard entirely.

All CDPS search hyperparameters, the frame-axiom mode, and the timeouts are read
from each experiment's ``evaluation_results/run_params.json`` so the anchored run
matches the original (timeouts overridable via CLI).

Usage:
    python -m benchmark.backfill_cdps \
        --experiment-dir "benchmark/running_results/hanoi/simulation-checkup-after-gtfixed__*" \
        --workers 4

Options mirror backfill_baseline: --data-dir (override), --domain, --cells,
--force, --dry-run, --workers, plus --learn-timeout / --planning-timeout /
--frame-axiom-mode to override the run_params values.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from benchmark.algorithms import CDPS_ANCHORED_ALGORITHM_NAME
from benchmark.backfill_common import (
    NULL_METRIC_KEYS,
    existing_algorithms,
    find_problem_pddl,
    is_cell_dir,
    merge_row,
    parse_cell_name,
    read_run_params,
    resolve_data_dir,
    worker_init,
)
from benchmark.experiment_running_helpers.resume import FOLD_RESULT_FILENAME
from benchmark.experiment_running_helpers.run_fold import run_cdps_phase
from benchmark.experiment_running_helpers.statistics import count_total_transitions_and_gt
from benchmark.experiment_running_helpers.trajectory_utils import (
    prepare_anchored_fold_trajectories,
)

# CDPS search-shape params persisted in run_params.json (fallbacks mirror
# CDPSConfig's defaults for the rare cell that predates a given field).
_CDPS_SEARCH_DEFAULTS: Dict[str, object] = {
    "fluent_patch_cost": 1.0,
    "fluent_patch_weight": 1.0,
    "model_patch_cost": 1.0,
    "model_constraint_weight": 0.0,
    "max_search_nodes": None,
    "search_mode": "dfs",
    "node_choosing_strategy": "model_patch_first",
    "conflict_group_strategy": "most_observations",
    "fluent_branch_mode": "group",
}


@dataclass(frozen=True)
class ExperimentSettings:
    """Everything one experiment contributes to its cells' anchored backfill.

    Resolved once per experiment from its ``run_params.json`` (with CLI
    overrides) and then shared by all of that experiment's cells, so cell-level
    code takes a single object instead of a long positional argument list.
    """

    data_dir: Path
    bench_name: str
    cdps_search_params: Dict[str, object]
    learn_timeout: int
    planning_timeout: int
    frame_axiom_mode: str


def _stage_anchored_inputs(
    cell: Path, data_dir: Path, fold_info: dict, stage_dir: Path,
) -> List[Tuple[Path, Path, Path, set]]:
    """Stage each training problem's degraded obs + clean GT JSON for anchoring.

    ``prepare_anchored_fold_trajectories`` expects, per problem, a dir holding
    ``<problem>.trajectory`` (the degraded original), ``<problem>.masking_info``,
    and an in-dir ``<problem>_trajectory.json`` (the clean GT). The cell stores
    the degraded data under a different name (``original_observation_<problem>.*``)
    and the clean GT lives in the data_dir, so we copy both into ``stage_dir``
    under the expected names.

    Returns:
        Plain (pre-anchor) fold tuples pointing at the staged files. The 4th
        element (GT indices) is unused by the anchored prep — which recomputes
        it — so it is an empty set, matching backfill_baseline's convention.
    """
    obs_dir = cell / "original_observations"
    staged: List[Tuple[Path, Path, Path, set]] = []

    for entry in fold_info.get("trajectories", []):
        problem = entry["problem"]

        degraded_traj = obs_dir / f"original_observation_{problem}.trajectory"
        if not degraded_traj.exists():  # older naming fallback
            degraded_traj = obs_dir / f"{problem}.trajectory"
        degraded_mask = degraded_traj.with_suffix(".masking_info")
        if not degraded_traj.exists() or not degraded_mask.exists():
            print(f"    Warning: no degraded trajectory/masking for {problem}, skipping")
            continue

        gt_json = data_dir / "gt_trajectories" / problem / f"{problem}_trajectory.json"
        if not gt_json.exists():
            print(f"    Warning: no clean GT JSON for {problem} at {gt_json}, skipping")
            continue

        problem_pddl = find_problem_pddl(data_dir, problem)
        if problem_pddl is None:
            print(f"    Warning: no problem PDDL for {problem} under {data_dir}, skipping")
            continue

        prob_stage = stage_dir / problem
        prob_stage.mkdir(parents=True, exist_ok=True)
        staged_traj = prob_stage / f"{problem}.trajectory"
        staged_mask = prob_stage / f"{problem}.masking_info"
        shutil.copy2(degraded_traj, staged_traj)
        shutil.copy2(degraded_mask, staged_mask)
        shutil.copy2(gt_json, prob_stage / f"{problem}_trajectory.json")

        staged.append((staged_traj, staged_mask, problem_pddl, set()))

    return staged


def _resolve_test_problem_paths(data_dir: Path, fold_info: dict) -> List[str]:
    """Resolve the cell's held-out test problems to on-disk PDDL paths."""
    paths: List[str] = []
    for problem in fold_info.get("test_problems", []):
        p = find_problem_pddl(data_dir, problem)
        if p is not None:
            paths.append(str(p))
        else:
            print(f"    Warning: test problem PDDL not found for {problem}")
    return paths


def backfill_cell(
    cell: Path, settings: ExperimentSettings, force: bool, dry_run: bool,
) -> str:
    """Backfill one cell's CDPS_ANCHORED row. Returns done | dry | skip | invalid."""
    parsed = parse_cell_name(cell.name)
    if parsed is None:
        return "invalid"
    fold, num_trajs, gt_rate = parsed

    fold_info_path = cell / "fold_info.json"
    domain_ref = cell / "domain_reference.pddl"
    if not fold_info_path.exists() or not domain_ref.exists():
        print(f"  [SKIP] {cell.name}: missing fold_info.json or domain_reference.pddl")
        return "skip"
    fold_info = json.loads(fold_info_path.read_text())

    fold_result_path = cell / FOLD_RESULT_FILENAME
    if not force and CDPS_ANCHORED_ALGORITHM_NAME in existing_algorithms(fold_result_path):
        print(f"  [SKIP] {cell.name}: {CDPS_ANCHORED_ALGORITHM_NAME} row already present")
        return "skip"

    test_problem_paths = _resolve_test_problem_paths(settings.data_dir, fold_info)
    if not test_problem_paths:
        print(f"  [SKIP] {cell.name}: no test problem PDDLs found")
        return "skip"

    test_states = cell / "predictive_power_test_states" / "test_states.json"
    test_states_str = str(test_states) if test_states.exists() else None

    # Stage degraded obs + clean GT, then build the anchored (init+final GT)
    # trajectories under the isolated cdps_anchored/ subdir.
    anchored_work_dir = cell / "cdps_anchored"
    with tempfile.TemporaryDirectory(prefix="cdps_anchored_stage_") as tmp:
        staged = _stage_anchored_inputs(cell, settings.data_dir, fold_info, Path(tmp))
        if not staged:
            print(f"  [SKIP] {cell.name}: no usable anchored inputs")
            return "skip"

        anchored = prepare_anchored_fold_trajectories(
            staged, domain_ref, gt_rate, anchored_work_dir,
            frame_axiom_mode=settings.frame_axiom_mode,
        )
        if not anchored:
            print(f"  [SKIP] {cell.name}: anchored preparation produced nothing")
            return "skip"

        total_transitions, total_gt = count_total_transitions_and_gt(anchored)

        if dry_run:
            print(f"  [DRY] {cell.name}: would run {CDPS_ANCHORED_ALGORITHM_NAME} on "
                  f"{len(anchored)} anchored trajectories, {len(test_problem_paths)} "
                  f"test problems (learn_timeout={settings.learn_timeout}s, "
                  f"planning_timeout={settings.planning_timeout}s)"
                  f"{'' if test_states_str else ' (no test states!)'}")
            return "dry"

        # AMLGym's problem_solving writes ./tmp to the cwd — work inside the cell.
        original_cwd = os.getcwd()
        os.chdir(cell)
        try:
            print(f"  [{CDPS_ANCHORED_ALGORITHM_NAME}] {cell.name}: running anchored CDPS...")
            row = run_cdps_phase(
                anchor_endpoints=True,
                algo_name=CDPS_ANCHORED_ALGORITHM_NAME,
                cdps_work_dir=anchored_work_dir,
                trajectories=anchored,
                gt_source_indices=None,
                pre_built_observations=None,
                domain_ref_path=domain_ref,
                testing_dir=cell.parent,
                bench_name=settings.bench_name,
                fold=fold,
                num_trajectories=num_trajs,
                gt_rate=gt_rate,
                test_problem_paths=test_problem_paths,
                null_metrics={k: None for k in NULL_METRIC_KEYS},
                total_transitions=total_transitions,
                total_gt_transitions=total_gt,
                conflict_search_timeout=settings.learn_timeout,
                planning_timeout=settings.planning_timeout,
                events_tracing=False,
                test_states_path=test_states_str,
                **settings.cdps_search_params,
            )
        finally:
            os.chdir(original_cwd)

    if row is None:
        print(f"  [SKIP] {cell.name}: anchored CDPS produced no row")
        return "skip"
    merge_row(fold_result_path, row)
    print(f"  [{CDPS_ANCHORED_ALGORITHM_NAME}] {cell.name}: row merged into {FOLD_RESULT_FILENAME}")
    return "done"


def _backfill_cell_worker(
    cell: Path, settings: ExperimentSettings, force: bool,
) -> Tuple[str, str]:
    """Process-pool entry point: backfill one cell, never raising.

    Returns:
        (cell_path, status) where status is backfill_cell's status or
        ``error: <message>`` (so one bad cell can't kill the whole run).
    """
    try:
        return str(cell), backfill_cell(cell, settings, force=force, dry_run=False)
    except Exception as e:
        return str(cell), f"error: {e}"


def resolve_experiment(
    exp_dir: Path, override_data_dir: Optional[Path], args: argparse.Namespace,
) -> Optional[ExperimentSettings]:
    """Resolve one experiment's backfill settings, or None if unusable.

    data_dir + CDPS hyperparameters + frame-axiom mode come from the
    experiment's own run_params.json (so the anchored run matches the original);
    ``--data-dir`` and the ``--*-timeout`` / ``--frame-axiom-mode`` flags override.
    """
    data_dir, src = resolve_data_dir(exp_dir, override_data_dir)
    if data_dir is None:
        print(f"[SKIP] {exp_dir}: no data_dir in run_params.json; pass --data-dir explicitly")
        return None
    if not data_dir.is_dir():
        print(f"[SKIP] {exp_dir}: resolved data_dir does not exist: {data_dir}")
        return None

    run_params = read_run_params(exp_dir)
    if run_params is None:
        print(f"[SKIP] {exp_dir}: no run_params.json — CDPS hyperparameters unavailable")
        return None

    settings = ExperimentSettings(
        data_dir=data_dir,
        bench_name=args.domain or exp_dir.parent.name,
        cdps_search_params={
            k: run_params.get(k, default) for k, default in _CDPS_SEARCH_DEFAULTS.items()
        },
        learn_timeout=args.learn_timeout or run_params.get("learning_timeout_seconds") or 300,
        planning_timeout=args.planning_timeout or run_params.get("planning_timeout_seconds") or 60,
        frame_axiom_mode=args.frame_axiom_mode or run_params.get("frame_axiom_mode", "after_gt_only"),
    )
    print(f"[{settings.bench_name}] {exp_dir.name}: data_dir from {src}: {data_dir} | "
          f"learn_timeout={settings.learn_timeout}s "
          f"planning_timeout={settings.planning_timeout}s "
          f"frame_axiom_mode={settings.frame_axiom_mode}")
    return settings


def _gather_tasks(
    args: argparse.Namespace, override_data_dir: Optional[Path],
) -> List[Tuple[Path, ExperimentSettings]]:
    """Expand the experiment dirs into (cell, settings) work items."""
    tasks: List[Tuple[Path, ExperimentSettings]] = []
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


def _run_parallel(
    tasks: List[Tuple[Path, ExperimentSettings]], workers: int, force: bool,
) -> None:
    """Backfill cells across a process pool, printing a per-cell + final summary."""
    print(f"\nRunning {len(tasks)} cells with {workers} workers...")
    statuses: Dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as executor:
        futures = {
            executor.submit(_backfill_cell_worker, cell, settings, force): cell
            for cell, settings in tasks
        }
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                cell_str, status = future.result()
            except Exception as e:  # worker process died before/outside the task
                cell_str, status = str(futures[future]), f"error: {e}"
            statuses[cell_str] = status
            print(f"[{i}/{len(futures)}] {Path(cell_str).name}: {status}")

    done = sum(1 for s in statuses.values() if s == "done")
    skipped = sum(1 for s in statuses.values() if s in ("skip", "invalid"))
    errors = {c: s for c, s in statuses.items() if s.startswith("error")}
    print(f"\nSummary: {done} done, {skipped} skipped, {len(errors)} errors")
    for cell_str, err in errors.items():
        print(f"  ERROR {cell_str}: {err}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Backfill anchored CDPS (CDPS_ANCHORED) results into existing "
                    "experiment cells (paired with the frozen degraded data).")
    ap.add_argument("--experiment-dir", type=Path, nargs="+", required=True,
                    help="Experiment director(y/ies) containing testing/ "
                         "(shell globs expand to multiple dirs).")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="Override the data_dir problem PDDLs / gt_trajectories "
                         "are resolved against. By default each experiment's "
                         "data_dir is read from its evaluation_results/run_params.json.")
    ap.add_argument("--domain", default=None,
                    help="Domain/bench name for the result rows "
                         "(default: inferred from the experiment path).")
    ap.add_argument("--learn-timeout", type=int, default=None,
                    help="Override the conflict-search timeout (seconds). "
                         "Default: run_params.json's learning_timeout_seconds.")
    ap.add_argument("--planning-timeout", type=int, default=None,
                    help="Override the planning timeout (seconds). "
                         "Default: run_params.json's planning_timeout_seconds.")
    ap.add_argument("--frame-axiom-mode", default=None,
                    choices=["after_gt_only", "all_states"],
                    help="Override the frame-axiom propagation mode. "
                         "Default: run_params.json's frame_axiom_mode.")
    ap.add_argument("--cells", default=None,
                    help="Only process cell dirs whose name contains this substring.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run and replace the CDPS_ANCHORED row even if present.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of cells to backfill in parallel (one process "
                         "per cell). Default 1 = sequential. Dry runs are always "
                         "sequential.")
    args = ap.parse_args()

    override_data_dir = args.data_dir.resolve() if args.data_dir else None
    if override_data_dir and not override_data_dir.is_dir():
        raise SystemExit(f"--data-dir does not exist: {override_data_dir}")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    tasks = _gather_tasks(args, override_data_dir)
    if not tasks:
        print("Nothing to do.")
        return

    workers = 1 if args.dry_run else min(args.workers, len(tasks))
    if workers == 1:
        for cell, settings in tasks:
            backfill_cell(cell, settings, force=args.force, dry_run=args.dry_run)
        return

    _run_parallel(tasks, workers, args.force)


if __name__ == "__main__":
    main()
