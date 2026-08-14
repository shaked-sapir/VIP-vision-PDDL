"""Backfill a CDPS-family arm into existing cells without regenerating data.

This is the CDPS counterpart of ``backfill_baseline``. For each already-executed
``testing/fold*_numtrajs*_gtrate*`` cell it re-runs one CDPS-family denoiser over
the data that cell already learned from, and merges the resulting row into the
cell's ``fold_result.json`` — paired with the existing CDPS/PISAM/baseline rows.

``--algorithm`` picks the arm, and the arm decides where its input comes from:

  - ``cdps_anchored`` reconstructs init+final-anchored trajectories. Why this
    works here even though the live runner refuses it for simulated data: the
    guard exists because in-memory observations never land in the on-disk
    problem-dir layout the anchored prep needs. After a run, every input does
    exist on disk — the frozen degraded observation in
    ``cell/original_observations/`` and the clean GT trajectory in
    ``<data_dir>/gt_trajectories/`` — so we stage them into that layout and the
    prep sees an ordinary file-based fold.

  - ``cdps_milp_single_round`` / ``cdps_milp_loop`` consume the frozen degraded
    files unchanged, which is what makes their rows comparable to the CDPS row
    beside them: same observations, same GT map, only the denoiser differs.

All CDPS search hyperparameters, the frame-axiom mode, and the timeouts are read
from each experiment's ``evaluation_results/run_params.json`` so the backfilled
run matches the original (timeouts overridable via CLI). The MILP arms take their
own knobs from ``--milp-config``, which reads the same ``cdps_milp:`` block a
live sweep uses — so a backfilled arm can be pinned to an existing sweep's
settings rather than retyped as flags.

Usage:
    python -m benchmark.backfill_cdps \
        --experiment-dir "benchmark/running_results/hanoi/simulation-checkup-after-gtfixed__*" \
        --workers 4

    python -m benchmark.backfill_cdps \
        --experiment-dir "benchmark/running_results/blocksworld/sim_run__*" \
        --algorithm cdps_milp_loop --milp-config benchmark/run_config.yaml

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
from typing import Dict, List, Optional, Set, Tuple

import yaml

from benchmark.algorithms import (
    CDPS_ANCHORED,
    CDPS_ANCHORED_ALGORITHM_NAME,
    CDPS_MILP_LOOP,
    CDPS_MILP_SINGLE_ROUND,
    cdps_milp_algorithm_name,
    milp_config_for,
    milp_work_subdir,
)
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
from src.plan_denoising.milp_denoiser.config import CdpsMilpConfig

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


# Each arm keeps its artifacts in its own cell subdir, matching the layout a live
# run produces, so a backfilled arm's models land exactly where the dashboard and
# the offline scorers already look for them.
_WORK_SUBDIRS = {
    CDPS_ANCHORED: "cdps_anchored",
    CDPS_MILP_SINGLE_ROUND: "cdps_milp_single_round",
    CDPS_MILP_LOOP: "cdps_milp_loop",
}


@dataclass(frozen=True)
class AlgorithmSpec:
    """Which CDPS-family arm to backfill, resolved once from the CLI.

    ``milp_config`` doubles as the branch discriminator: ``None`` means the
    anchored conflict search, anything else means a MILP denoiser. That is not a
    shortcut — the two differ in where their input comes from as well as in how
    they denoise, and the config is what carries that distinction.
    """

    key: str
    row_name: str
    work_subdir: str
    milp_config: Optional[CdpsMilpConfig]


def _read_milp_config(path: Optional[Path]) -> CdpsMilpConfig:
    """The ``cdps_milp:`` block from a YAML file, or all defaults.

    A run_config.yaml (``shared.cdps_milp``), a file holding just the block under
    ``cdps_milp:``, and a file that *is* the block are all accepted, because the
    point of the flag is to pin a backfilled arm to the settings some live sweep
    already used — and that sweep's config is the run_config itself.

    Unknown keys are rejected by ``CdpsMilpConfig.from_dict``, so pointing this
    at the wrong file fails loudly instead of silently backfilling defaults.
    """
    if path is None:
        return CdpsMilpConfig()
    raw = yaml.safe_load(path.read_text()) or {}
    shared = raw.get("shared")
    if isinstance(shared, dict) and "cdps_milp" in shared:
        return CdpsMilpConfig.from_dict(shared["cdps_milp"])
    return CdpsMilpConfig.from_dict(raw.get("cdps_milp", raw))


def resolve_algorithm_spec(key: str, milp_config_path: Optional[Path]) -> AlgorithmSpec:
    """Turn ``--algorithm`` (+ ``--milp-config``) into the arm's row name and layout."""
    if key == CDPS_ANCHORED:
        if milp_config_path is not None:
            raise SystemExit(
                "--milp-config applies to the cdps_milp_* algorithms only; "
                f"--algorithm {key} takes its settings from run_params.json"
            )
        return AlgorithmSpec(key, CDPS_ANCHORED_ALGORITHM_NAME, _WORK_SUBDIRS[key], None)

    # The selected key pins the variant, so one shared cdps_milp block can back
    # both MILP arms — the same rule the live runner follows.
    config = milp_config_for(key, _read_milp_config(milp_config_path))
    # Take the directory from the same place a live run does, rather than from
    # _WORK_SUBDIRS. Those are keyed on the algorithm *key*, so backfilling an
    # ablation (eq16 on/off, say) twice into one cell would put both arms in one
    # directory and let the second overwrite the first's models and extraction
    # artifacts — silently, since the rows stay distinct. milp_work_subdir keys
    # on the row label instead, so directory and row cannot drift apart.
    return AlgorithmSpec(
        key, cdps_milp_algorithm_name(config), milp_work_subdir(key, config), config
    )


@dataclass(frozen=True)
class ExperimentSettings:
    """Everything one experiment contributes to its cells' backfill.

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


def _degraded_files(cell: Path, problem: str) -> Optional[Tuple[Path, Path]]:
    """The frozen degraded ``.trajectory`` + ``.masking_info`` a cell learned from.

    Both arms start here, which is the whole premise of backfilling: for
    simulated cells the degradation was drawn in memory from a fold-offset seed,
    so re-deriving it from the data dir would produce different noise and the new
    row would not be comparable to the rows already in the cell.
    """
    obs_dir = cell / "original_observations"
    trajectory = obs_dir / f"original_observation_{problem}.trajectory"
    if not trajectory.exists():  # older naming fallback
        trajectory = obs_dir / f"{problem}.trajectory"
    masking = trajectory.with_suffix(".masking_info")
    if not trajectory.exists() or not masking.exists():
        return None
    return trajectory, masking


def _frozen_fold_trajectories(
    cell: Path, data_dir: Path, fold_info: dict, gt_rate: int,
) -> List[Tuple[Path, Path, Path, Set[int]]]:
    """The cell's frozen degraded files as plain (un-anchored) fold tuples.

    This is the input the MILP arms need: byte-identical to what plain CDPS
    consumed, so ``cost(MILP) <= cost(CDPS)`` is checkable against the row
    already sitting in the same ``fold_result.json``.

    Ground truth is the initial state only, matching ``SimulatedDataSource``.
    A non-zero ``gt_rate`` is refused rather than guessed: the injected GT state
    indices are recorded nowhere in the cell, and picking a different set than
    the original run would protect different states while claiming to be the
    same experiment. (No such cell exists today — the simulated source rejects
    ``gt_rate > 0`` outright.)
    """
    if gt_rate != 0:
        print(f"    Warning: gt_rate={gt_rate} — the frozen cells record no GT "
              f"state indices, so the MILP arms cannot reproduce the original "
              f"GT map; skipping")
        return []

    prepared: List[Tuple[Path, Path, Path, Set[int]]] = []
    for entry in fold_info.get("trajectories", []):
        problem = entry["problem"]

        degraded = _degraded_files(cell, problem)
        if degraded is None:
            print(f"    Warning: no degraded trajectory/masking for {problem}, skipping")
            continue

        problem_pddl = find_problem_pddl(data_dir, problem)
        if problem_pddl is None:
            print(f"    Warning: no problem PDDL for {problem} under {data_dir}, skipping")
            continue

        prepared.append((*degraded, problem_pddl, {0}))

    return prepared


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
    staged: List[Tuple[Path, Path, Path, set]] = []

    for entry in fold_info.get("trajectories", []):
        problem = entry["problem"]

        degraded = _degraded_files(cell, problem)
        if degraded is None:
            print(f"    Warning: no degraded trajectory/masking for {problem}, skipping")
            continue
        degraded_traj, degraded_mask = degraded

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


def _anchored_fold_trajectories(
    cell: Path, settings: ExperimentSettings, fold_info: dict, gt_rate: int,
) -> List[Tuple[Path, Path, Path, Set[int]]]:
    """Rebuild the cell's init+final-anchored trajectories under ``cdps_anchored/``.

    The staging dir is temporary but the output is not: the prep writes its
    anchored files into the cell, so only the intermediate problem-dir layout
    needs to be thrown away afterwards.
    """
    with tempfile.TemporaryDirectory(prefix="cdps_anchored_stage_") as tmp:
        staged = _stage_anchored_inputs(cell, settings.data_dir, fold_info, Path(tmp))
        if not staged:
            return []
        return prepare_anchored_fold_trajectories(
            staged,
            cell / "domain_reference.pddl",
            gt_rate,
            cell / _WORK_SUBDIRS[CDPS_ANCHORED],
            frame_axiom_mode=settings.frame_axiom_mode,
        )


def _fold_inputs(
    cell: Path, settings: ExperimentSettings, spec: AlgorithmSpec,
    fold_info: dict, gt_rate: int,
) -> List[Tuple[Path, Path, Path, Set[int]]]:
    """The trajectories this arm learns from — anchored, or the frozen originals."""
    if spec.milp_config is None:
        return _anchored_fold_trajectories(cell, settings, fold_info, gt_rate)
    return _frozen_fold_trajectories(cell, settings.data_dir, fold_info, gt_rate)


def backfill_cell(
    cell: Path, settings: ExperimentSettings, spec: AlgorithmSpec,
    force: bool, dry_run: bool,
) -> str:
    """Backfill one cell's row for ``spec``. Returns done | dry | skip | invalid."""
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
    if not force and spec.row_name in existing_algorithms(fold_result_path):
        print(f"  [SKIP] {cell.name}: {spec.row_name} row already present")
        return "skip"

    test_problem_paths = _resolve_test_problem_paths(settings.data_dir, fold_info)
    if not test_problem_paths:
        print(f"  [SKIP] {cell.name}: no test problem PDDLs found")
        return "skip"

    test_states = cell / "predictive_power_test_states" / "test_states.json"
    test_states_str = str(test_states) if test_states.exists() else None

    trajectories = _fold_inputs(cell, settings, spec, fold_info, gt_rate)
    if not trajectories:
        print(f"  [SKIP] {cell.name}: no usable inputs for {spec.key}")
        return "skip"

    total_transitions, total_gt = count_total_transitions_and_gt(trajectories)

    if dry_run:
        print(f"  [DRY] {cell.name}: would run {spec.row_name} on "
              f"{len(trajectories)} trajectories, {len(test_problem_paths)} "
              f"test problems (learn_timeout={settings.learn_timeout}s, "
              f"planning_timeout={settings.planning_timeout}s)"
              f"{'' if test_states_str else ' (no test states!)'}")
        return "dry"

    # AMLGym's problem_solving writes ./tmp to the cwd — work inside the cell.
    original_cwd = os.getcwd()
    os.chdir(cell)
    try:
        print(f"  [{spec.row_name}] {cell.name}: running {spec.key}...")
        row = run_cdps_phase(
            # Anchoring is a property of the trajectories, not of the denoiser;
            # the MILP arms read the frozen originals, so they are un-anchored
            # exactly like the plain-CDPS row they are compared against.
            anchor_endpoints=spec.milp_config is None,
            algo_name=spec.row_name,
            cdps_work_dir=cell / spec.work_subdir,
            trajectories=trajectories,
            # None -> the GT map is derived from the tuples' own indices, which
            # both prep paths fill in.
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
            milp_config=spec.milp_config,
            **settings.cdps_search_params,
        )
    finally:
        os.chdir(original_cwd)

    if row is None:
        print(f"  [SKIP] {cell.name}: {spec.row_name} produced no row")
        return "skip"
    merge_row(fold_result_path, row)
    print(f"  [{spec.row_name}] {cell.name}: row merged into {FOLD_RESULT_FILENAME}")
    return "done"


def _backfill_cell_worker(
    cell: Path, settings: ExperimentSettings, spec: AlgorithmSpec, force: bool,
) -> Tuple[str, str]:
    """Process-pool entry point: backfill one cell, never raising.

    Returns:
        (cell_path, status) where status is backfill_cell's status or
        ``error: <message>`` (so one bad cell can't kill the whole run).
    """
    try:
        return str(cell), backfill_cell(cell, settings, spec, force=force, dry_run=False)
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
    tasks: List[Tuple[Path, ExperimentSettings]],
    spec: AlgorithmSpec,
    workers: int,
    force: bool,
) -> None:
    """Backfill cells across a process pool, printing a per-cell + final summary."""
    print(f"\nRunning {len(tasks)} cells with {workers} workers...")
    statuses: Dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as executor:
        futures = {
            executor.submit(_backfill_cell_worker, cell, settings, spec, force): cell
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
        description="Backfill a CDPS-family arm into existing experiment cells, "
                    "learning from the frozen degraded data the cell already holds.")
    ap.add_argument("--algorithm", default=CDPS_ANCHORED,
                    choices=sorted(_WORK_SUBDIRS),
                    help="Which CDPS-family arm to backfill. Default cdps_anchored "
                         "(this script's original behaviour). The cdps_milp_* arms "
                         "consume the cell's frozen observations unchanged, so their "
                         "rows are directly comparable to the CDPS row beside them.")
    ap.add_argument("--milp-config", type=Path, default=None,
                    help="YAML holding the cdps_milp settings for the cdps_milp_* "
                         "arms — a run_config.yaml (shared.cdps_milp), a file with a "
                         "top-level cdps_milp: block, or the bare block. Default: "
                         "CdpsMilpConfig defaults. The --algorithm choice pins the "
                         "variant, so any 'variant:' key here is overridden.")
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
                    help="Re-run and replace the selected arm's row even if present.")
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

    # Resolve the arm before touching any cell: a bad --milp-config should fail
    # here, not after an hour of learning.
    spec = resolve_algorithm_spec(args.algorithm, args.milp_config)
    print(f"Backfilling {spec.key} as row '{spec.row_name}' (work dir: {spec.work_subdir}/)")

    tasks = _gather_tasks(args, override_data_dir)
    if not tasks:
        print("Nothing to do.")
        return

    workers = 1 if args.dry_run else min(args.workers, len(tasks))
    if workers == 1:
        for cell, settings in tasks:
            backfill_cell(cell, settings, spec, force=args.force, dry_run=args.dry_run)
        return

    _run_parallel(tasks, spec, workers, args.force)


if __name__ == "__main__":
    main()
