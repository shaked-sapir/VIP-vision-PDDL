"""Run the unanchored PI-SAM+MILP arms on observations that hold no clean state.

For each already-executed ``testing/fold*_numtrajs*_gtrate*`` cell this stages the
cell's frozen observations with states 0 and 1 redrawn from ground truth
(``benchmark/simulated_version/leading_state_corruption.py``), runs one
``pisam_milp_*`` arm with ``gt_anchoring: none`` on them, and evaluates it on the
cell's own test problems and test states.

Nothing is written into the source experiment. Rows and artifacts go under
``--out-root``, in a tree that mirrors the source one::

    <out-root>/<domain>/<experiment>/testing/<cell>/
        observations_noisy_s0/      the staged observations
        <arm work dir>/             models and round logs
        fold_result.json            the new row(s) only
        fold_info.json, domain_reference.pddl, provenance.json

Usage:
    python -m benchmark.backfill_nogt_s0 \
        --experiment-dir "benchmark/running_results/hanoi/simulation-final-run__*" \
        --algorithm pisam_milp_loop --milp-config benchmark/run_config.yaml --workers 4

``--prepare-only`` stages the observations and stops; ``--dry-run`` lists the work.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import yaml
from pddl_plus_parser.lisp_parsers import DomainParser

from benchmark.algorithms import (
    PISAM_MILP_LOOP,
    PISAM_MILP_SINGLE_ROUND,
    milp_config_for,
    milp_work_subdir,
    pisam_milp_algorithm_name,
)
from benchmark.backfill_cdps import (
    AlgorithmSpec,
    ExperimentSettings,
    resolve_experiment,
)
from benchmark.backfill_common import (
    NULL_METRIC_KEYS,
    existing_algorithms,
    find_problem_pddl,
    find_test_states,
    is_cell_dir,
    merge_row,
    parse_cell_name,
    read_run_params,
    worker_init,
)
from benchmark.experiment_running_helpers.resume import FOLD_RESULT_FILENAME
from benchmark.experiment_running_helpers.run_fold import run_cdps_phase
from benchmark.experiment_running_helpers.statistics import count_total_transitions_and_gt
from benchmark.simulated_version.leading_state_corruption import (
    LEADING_STATE_INDICES,
    DegradationSpec,
    stage_noisy_initial_observation,
)
from src.milp.converter import GtAnchoring
from src.observation_degradation.masking import MaskingType
from src.observation_degradation.noising import NoisingType
from src.plan_denoising.milp_denoiser.config import PisamMilpConfig, select_milp_block

DEFAULT_OUT_ROOT = Path("benchmark/running_results_nogt_s0")
OBSERVATIONS_DIRNAME = "observations_noisy_s0"
PROVENANCE_FILENAME = "provenance.json"
ROW_TAG = "s0=noisy"

_FROZEN_DIRNAME = "original_observations"
_UNANCHORED_PART = f"gt={GtAnchoring.NONE.value}"
_ARMS = (PISAM_MILP_LOOP, PISAM_MILP_SINGLE_ROUND)

PreparedTrajectory = Tuple[Path, Path, Path, Set[int]]


@dataclass(frozen=True)
class StagingOptions:
    """How a cell's observations are staged and where its results go."""

    out_root: Path
    hide_masked: bool
    prepare_only: bool


def noisy_initial_label(label: str) -> str:
    """An unanchored arm's label, tagged as run on a degraded initial state."""
    if _UNANCHORED_PART not in label:
        raise ValueError(f"'{label}' is not an unanchored arm ({_UNANCHORED_PART})")
    return label.replace(_UNANCHORED_PART, f"{_UNANCHORED_PART}__{ROW_TAG}", 1)


def read_unanchored_config(path: Optional[Path]) -> PisamMilpConfig:
    """The ``pisam_milp:`` block of a YAML file with ``gt_anchoring`` set to ``none``.

    Accepts a run_config.yaml (``shared.pisam_milp``), a file with a top-level
    ``pisam_milp:`` block, or the bare block. An ``ablations:`` sub-block is ignored.
    """
    if path is None:
        return replace(PisamMilpConfig(), gt_anchoring=GtAnchoring.NONE)
    raw = yaml.safe_load(path.read_text()) or {}
    shared = raw.get("shared")
    block = select_milp_block(shared) if isinstance(shared, dict) else None
    if block is None:
        block = select_milp_block(raw)
    if block is None:
        block = raw
    base = {key: value for key, value in block.items() if key != "ablations"}
    return replace(PisamMilpConfig.from_dict(base), gt_anchoring=GtAnchoring.NONE)


def resolve_arm(key: str, milp_config_path: Optional[Path]) -> AlgorithmSpec:
    """The unanchored arm ``key`` names, with its tagged row label and work dir."""
    config = milp_config_for(key, read_unanchored_config(milp_config_path))
    return AlgorithmSpec(
        key,
        noisy_initial_label(pisam_milp_algorithm_name(config)),
        noisy_initial_label(milp_work_subdir(key, config)),
        config,
    )


def degradation_spec(run_params: dict, seed: int) -> DegradationSpec:
    """A cell's masking and noising parameters, as its experiment recorded them."""
    missing = [k for k in ("p_mask", "p_noise") if k not in run_params]
    if missing:
        raise ValueError(f"run_params.json has no {missing}; not a simulated experiment")
    return DegradationSpec(
        masking_strategy=MaskingType(run_params.get("masking_strategy", "percentage")),
        masking_p=float(run_params["p_mask"]),
        noising_strategy=NoisingType(run_params.get("noising_strategy", "percentage")),
        noising_p=float(run_params["p_noise"]),
        seed=seed,
    )


def output_cell(out_root: Path, cell: Path) -> Path:
    """Where a source cell's results go under ``out_root``."""
    experiment = cell.parent.parent
    return out_root / experiment.parent.name / experiment.name / "testing" / cell.name


@contextmanager
def frozen_observations_dir(cell: Path) -> Iterator[Optional[Path]]:
    """The cell's frozen observations, unpacked to a temp dir if they are packed."""
    loose = cell / _FROZEN_DIRNAME
    packed = cell / f"{_FROZEN_DIRNAME}.tar.gz"
    if loose.is_dir():
        yield loose
    elif packed.is_file():
        with tempfile.TemporaryDirectory(prefix="frozen_obs_") as tmp:
            with tarfile.open(packed) as archive:
                archive.extractall(tmp, filter="data")
            yield Path(tmp) / _FROZEN_DIRNAME
    else:
        yield None


def stage_cell_observations(
    cell: Path,
    out_cell: Path,
    settings: ExperimentSettings,
    spec: DegradationSpec,
    fold: int,
    fold_info: dict,
    hide_masked: bool,
) -> List[PreparedTrajectory]:
    """Stage every training trajectory of a cell; fold tuples carry no GT index.

    Raises:
        FileNotFoundError: If a trajectory's frozen files, ground-truth
            trajectory or problem file cannot be found.
    """
    staged_dir = out_cell / OBSERVATIONS_DIRNAME
    domain = DomainParser(cell / "domain_reference.pddl", partial_parsing=True).parse_domain()
    prepared: List[PreparedTrajectory] = []
    with frozen_observations_dir(cell) as frozen_dir:
        if frozen_dir is None:
            raise FileNotFoundError(f"{cell} has no {_FROZEN_DIRNAME}/ and no tarball of it")
        for entry in fold_info.get("trajectories", []):
            problem = entry["problem"]
            frozen_trajectory = frozen_dir / entry["trajectory_file"]
            frozen_masking = frozen_dir / entry["masking_file"]
            gt_trajectory = settings.data_dir / "gt_trajectories" / problem / f"{problem}.trajectory"
            problem_pddl = find_problem_pddl(settings.problem_dir, problem)
            absent = [
                str(p) for p in (frozen_trajectory, frozen_masking, gt_trajectory)
                if not p.exists()
            ]
            if problem_pddl is None:
                absent.append(f"problem PDDL of {problem} under {settings.problem_dir}")
            if absent:
                raise FileNotFoundError(f"{cell.name}: missing {', '.join(absent)}")
            trajectory, masking = stage_noisy_initial_observation(
                frozen_trajectory, frozen_masking, gt_trajectory, problem_pddl,
                domain, spec, fold, staged_dir, hide_masked=hide_masked,
            )
            prepared.append((trajectory, masking, problem_pddl.resolve(), set()))
    return prepared


def _staging_record(spec: DegradationSpec, hide_masked: bool) -> dict:
    """The provenance fields that decide what the staged files contain."""
    return {
        "redrawn_state_indices": list(LEADING_STATE_INDICES),
        "masked_values_hidden": hide_masked,
        "seed": spec.seed,
        "masking_strategy": spec.masking_strategy.value,
        "masking_p": spec.masking_p,
        "noising_strategy": spec.noising_strategy.value,
        "noising_p": spec.noising_p,
    }


def staged_trajectories(
    out_cell: Path,
    settings: ExperimentSettings,
    spec: DegradationSpec,
    fold_info: dict,
    hide_masked: bool,
) -> Optional[List[PreparedTrajectory]]:
    """The cell's already-staged fold tuples, or None if staging must run.

    Staging is reused only when ``provenance.json`` records the same parameters
    and every trajectory's two files are present.
    """
    provenance_path = out_cell / PROVENANCE_FILENAME
    if not provenance_path.is_file():
        return None
    try:
        provenance = json.loads(provenance_path.read_text())
    except json.JSONDecodeError as error:
        print(f"[WARN] unreadable {provenance_path}: {error}")
        return None
    record = _staging_record(spec, hide_masked)
    if any(provenance.get(key) != value for key, value in record.items()):
        return None

    staged_dir = out_cell / OBSERVATIONS_DIRNAME
    prepared: List[PreparedTrajectory] = []
    for entry in fold_info.get("trajectories", []):
        trajectory = staged_dir / entry["trajectory_file"]
        masking = staged_dir / entry["masking_file"]
        problem_pddl = find_problem_pddl(settings.problem_dir, entry["problem"])
        if not trajectory.exists() or not masking.exists() or problem_pddl is None:
            return None
        prepared.append((trajectory, masking, problem_pddl.resolve(), set()))
    return prepared or None


def _git_commit() -> Optional[str]:
    """The checked-out commit, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"[WARN] could not read the git commit: {error}")
        return None
    return result.stdout.strip()


def write_cell_scaffold(
    cell: Path, out_cell: Path, spec: DegradationSpec, hide_masked: bool, n_trajectories: int,
) -> None:
    """Copy the cell's fold description and record where the staged data came from."""
    out_cell.mkdir(parents=True, exist_ok=True)
    for name in ("fold_info.json", "domain_reference.pddl"):
        shutil.copy2(cell / name, out_cell / name)
    provenance = {
        "source_cell": str(cell.resolve()),
        "staged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        **_staging_record(spec, hide_masked),
        "n_trajectories": n_trajectories,
    }
    (out_cell / PROVENANCE_FILENAME).write_text(json.dumps(provenance, indent=2))


def _test_problem_paths(problem_dir: Path, fold_info: dict) -> List[str]:
    """The cell's held-out test problems as PDDL paths."""
    paths: List[str] = []
    for problem in fold_info.get("test_problems", []):
        path = find_problem_pddl(problem_dir, problem)
        if path is None:
            print(f"    Warning: test problem PDDL not found for {problem}")
            continue
        paths.append(str(path.resolve()))
    return paths


def backfill_cell(
    cell: Path,
    settings: ExperimentSettings,
    degradation: DegradationSpec,
    arm: AlgorithmSpec,
    options: StagingOptions,
    force: bool,
    dry_run: bool,
) -> str:
    """Stage one cell and run ``arm`` on it. Returns done | staged | dry | skip | invalid."""
    parsed = parse_cell_name(cell.name)
    if parsed is None:
        return "invalid"
    fold, num_trajs, gt_rate = parsed

    fold_info_path = cell / "fold_info.json"
    if not fold_info_path.exists() or not (cell / "domain_reference.pddl").exists():
        print(f"  [SKIP] {cell.name}: missing fold_info.json or domain_reference.pddl")
        return "skip"
    fold_info = json.loads(fold_info_path.read_text())

    out_cell = output_cell(options.out_root, cell).resolve()
    fold_result_path = out_cell / FOLD_RESULT_FILENAME
    if (not force and not options.prepare_only
            and arm.row_name in existing_algorithms(fold_result_path)):
        print(f"  [SKIP] {cell.name}: {arm.row_name} row already present")
        return "skip"

    test_problem_paths = _test_problem_paths(settings.problem_dir, fold_info)
    if not test_problem_paths and not options.prepare_only:
        print(f"  [SKIP] {cell.name}: no test problem PDDLs found")
        return "skip"
    test_states = find_test_states(cell)

    if dry_run:
        print(f"  [DRY] {cell.name}: would stage {len(fold_info.get('trajectories', []))} "
              f"trajectories into {out_cell} and run {arm.row_name} "
              f"(learn_timeout={settings.learn_timeout}s)"
              f"{'' if test_states else ' (no test states!)'}")
        return "dry"

    trajectories = None if force else staged_trajectories(
        out_cell, settings, degradation, fold_info, options.hide_masked,
    )
    if trajectories is None:
        trajectories = stage_cell_observations(
            cell, out_cell, settings, degradation, fold, fold_info, options.hide_masked,
        )
        if not trajectories:
            print(f"  [SKIP] {cell.name}: no training trajectories")
            return "skip"
        write_cell_scaffold(
            cell, out_cell, degradation, options.hide_masked, len(trajectories),
        )
    if options.prepare_only:
        return "staged"

    total_transitions, total_gt = count_total_transitions_and_gt(trajectories)
    original_cwd = os.getcwd()
    os.chdir(out_cell)
    try:
        print(f"  [{arm.row_name}] {cell.name}: running {arm.key}...")
        row = run_cdps_phase(
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
            test_problem_paths=test_problem_paths,
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

    if row is None:
        print(f"  [SKIP] {cell.name}: {arm.row_name} produced no row")
        return "skip"
    merge_row(fold_result_path, row)
    print(f"  [{arm.row_name}] {cell.name}: row written to {fold_result_path}")
    return "done"


CellTask = Tuple[Path, ExperimentSettings, DegradationSpec]


def _backfill_cell_worker(
    task: CellTask, arm: AlgorithmSpec, options: StagingOptions, force: bool,
) -> Tuple[str, str]:
    """Process-pool entry point: one cell, never raising."""
    cell, settings, degradation = task
    try:
        status = backfill_cell(
            cell, settings, degradation, arm, options, force=force, dry_run=False,
        )
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
        try:
            degradation = degradation_spec(read_run_params(exp_dir) or {}, args.seed)
        except ValueError as error:
            print(f"[SKIP] {exp_dir}: {error}")
            continue
        cells = sorted(d for d in testing.iterdir() if is_cell_dir(d))
        if args.cells:
            cells = [c for c in cells if args.cells in c.name]
        print(f"  → {len(cells)} cells (mask={degradation.masking_p}, noise={degradation.noising_p})")
        tasks.extend((cell, settings, degradation) for cell in cells)
    return tasks


def _summarize(statuses: Dict[str, str]) -> int:
    """Print the run's summary; returns the number of cells that raised."""
    finished = sum(1 for s in statuses.values() if s in ("done", "staged"))
    skipped = sum(1 for s in statuses.values() if s in ("skip", "invalid"))
    errors = {c: s for c, s in statuses.items() if s.startswith("error")}
    print(f"\nSummary: {finished} finished, {skipped} skipped, {len(errors)} errors")
    for cell_str, error in errors.items():
        print(f"  ERROR {cell_str}: {error}")
    return len(errors)


def _run_sequential(
    tasks: List[CellTask], arm: AlgorithmSpec, options: StagingOptions, force: bool,
) -> int:
    """Run cells one after another; a failing cell does not stop the rest."""
    statuses: Dict[str, str] = {}
    for i, task in enumerate(tasks, start=1):
        cell_str, status = _backfill_cell_worker(task, arm, options, force)
        statuses[cell_str] = status
        print(f"[{i}/{len(tasks)}] {Path(cell_str).name}: {status}", flush=True)
    return _summarize(statuses)


def _run_parallel(
    tasks: List[CellTask], arm: AlgorithmSpec, options: StagingOptions,
    workers: int, force: bool,
) -> int:
    """Run cells across a process pool; returns the number of cells that raised."""
    print(f"\nRunning {len(tasks)} cells with {workers} workers...")
    statuses: Dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as executor:
        futures = {
            executor.submit(_backfill_cell_worker, task, arm, options, force): task[0]
            for task in tasks
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
        description="Run an unanchored pisam_milp_* arm on existing cells' observations "
                    "with states 0 and 1 degraded, writing under a separate results root.")
    ap.add_argument("--algorithm", default=PISAM_MILP_LOOP, choices=sorted(_ARMS))
    ap.add_argument("--milp-config", type=Path, default=None,
                    help="YAML holding the pisam_milp settings (a run_config.yaml, a file "
                         "with a pisam_milp: block, or the bare block). gt_anchoring is "
                         "forced to none and an ablations: sub-block is ignored.")
    ap.add_argument("--experiment-dir", type=Path, nargs="+", required=True,
                    help="Source experiment director(y/ies) containing testing/.")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                    help=f"Root of the mirrored results tree. Default {DEFAULT_OUT_ROOT}.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Base seed of the redrawn states. Default 42.")
    ap.add_argument("--keep-masked-values", action="store_true",
                    help="Keep each masked fluent's true polarity in the staged "
                         ".masking_info files instead of writing all of them positive.")
    ap.add_argument("--prepare-only", action="store_true",
                    help="Stage the observations and stop before learning.")
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

    arm = resolve_arm(args.algorithm, args.milp_config)
    options = StagingOptions(
        out_root=args.out_root.resolve(),
        hide_masked=not args.keep_masked_values,
        prepare_only=args.prepare_only,
    )
    print(f"Arm {arm.key} as row '{arm.row_name}' (work dir: {arm.work_subdir}/) "
          f"→ {options.out_root}")

    tasks = gather_tasks(args, override_data_dir)
    if not tasks:
        print("Nothing to do.")
        return

    if args.dry_run:
        for cell, settings, degradation in tasks:
            backfill_cell(cell, settings, degradation, arm, options, force=args.force, dry_run=True)
        return
    workers = min(args.workers, len(tasks))
    if workers == 1:
        failed = _run_sequential(tasks, arm, options, args.force)
    else:
        failed = _run_parallel(tasks, arm, options, workers, args.force)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
