"""Backfill the realised-corruption fields into existing image cells.

Folds that ran before ``fold_corruption.py`` existed have no
``realised_mask_rate`` / ``realised_flip_rate`` / ``flip_persistence`` /
``n_atoms`` in their ``fold_info.json``. This measures each cell's frozen
``original_observations/`` against the corpus's ``gt_trajectories/`` and
writes those fields, learning nothing and touching no other file.

The corpus comes from the experiment's ``run_params.json`` (``data_dir``) or
from ``--data-dir``; there is no name-based guessing (see
``backfill_common.resolve_data_dir``). Problem PDDLs are read from the pool
whose dialect matches the cells (``resolve_problem_dir``), and each cell is
measured with its own ``domain_reference.pddl``.

Usage:
    python -m benchmark.backfill_realised_corruption <experiment_dir> [<experiment_dir> ...]
    python -m benchmark.backfill_realised_corruption <experiment_dir> --data-dir <corpus>
    python -m benchmark.backfill_realised_corruption --from-dashboard-config [--data-size small]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from benchmark.algorithm_adapters.lamanna.realised_rates import gt_trajectory_lookup  # noqa: E402
from benchmark.backfill_common import (  # noqa: E402
    find_problem_pddl,
    is_cell_dir,
    resolve_data_dir,
    resolve_problem_dir,
)
from benchmark.experiment_running_helpers.fold_corruption import record_fold_corruption  # noqa: E402
from src.utils.corpus_layout import GT_SUBPATH  # noqa: E402

ORIGINAL_OBS_DIR = "original_observations"
ORIGINAL_OBS_PREFIX = "original_observation_"
DOMAIN_REFERENCE = "domain_reference.pddl"
DASHBOARD_CONFIG = project_root / "benchmark" / "evaluation" / "cfm" / "dashboard_config.yaml"

FrozenTrajectory = Tuple[Path, Path, Path, Set[int]]


def frozen_fold_trajectories(cell: Path, problem_dir: Path) -> List[FrozenTrajectory]:
    """The cell's frozen degraded files as fold tuples, one per training problem.

    Reads ``original_observations/original_observation_<problem>.trajectory``
    (and its ``.masking_info``), the byte-identical input every arm in the
    cell learned from; a problem whose PDDL is not under ``problem_dir`` is
    reported and skipped.
    """
    prepared: List[FrozenTrajectory] = []
    for trajectory in sorted((cell / ORIGINAL_OBS_DIR).glob(f"{ORIGINAL_OBS_PREFIX}*.trajectory")):
        problem = trajectory.stem[len(ORIGINAL_OBS_PREFIX):]
        problem_pddl = find_problem_pddl(problem_dir, problem)
        if problem_pddl is None:
            print(f"    [{cell.name}] no problem PDDL for {problem} under {problem_dir}; skipped")
            continue
        prepared.append((trajectory, trajectory.with_suffix(".masking_info"), problem_pddl, {0}))
    return prepared


def measurable_cells(experiment_dir: Path) -> List[Path]:
    """The fold instances of an experiment that hold frozen observations."""
    testing = experiment_dir / "testing"
    if not testing.is_dir():
        return []
    return sorted(
        cell for cell in testing.iterdir()
        if is_cell_dir(cell) and (cell / ORIGINAL_OBS_DIR).is_dir()
    )


def backfill_experiment(experiment_dir: Path, override_data_dir: Optional[Path]) -> Tuple[int, int]:
    """Record the realised corruption of every measurable cell; ``(recorded, cells)``."""
    experiment_dir = experiment_dir.resolve()
    data_dir, source = resolve_data_dir(experiment_dir, override_data_dir)
    if data_dir is None:
        print(f"[SKIP] {experiment_dir}: no data_dir in run_params.json; pass --data-dir")
        return 0, 0
    if not data_dir.is_dir():
        print(f"[SKIP] {experiment_dir}: data_dir from {source} does not exist: {data_dir}\n"
              f"       pass --data-dir with the corpus's current location")
        return 0, 0
    gt_lookup = gt_trajectory_lookup(data_dir)
    if not gt_lookup:
        print(f"[SKIP] {experiment_dir}: no {GT_SUBPATH}/ under {data_dir}")
        return 0, 0
    problem_dir = resolve_problem_dir(experiment_dir, data_dir)

    cells = measurable_cells(experiment_dir)
    print(f"[{experiment_dir.parent.name}] {experiment_dir.name}: {len(cells)} cells, "
          f"GT from {data_dir / GT_SUBPATH} ({source}), problems from {problem_dir}")
    recorded = 0
    for cell in cells:
        domain = cell / DOMAIN_REFERENCE
        if not domain.exists():
            print(f"  [{cell.name}] no {DOMAIN_REFERENCE}; skipped")
            continue
        prepared = frozen_fold_trajectories(cell, problem_dir)
        if not prepared:
            print(f"  [{cell.name}] no frozen observations; skipped")
            continue
        if record_fold_corruption(
            cell, prepared, domain, gt_lookup, gt_root=data_dir / GT_SUBPATH, tag=cell.name
        ) is not None:
            recorded += 1
    return recorded, len(cells)


def dashboard_image_experiments(config_path: Path, data_size: str) -> Dict[str, Path]:
    """``{domain: experiment dir}`` of the dashboard's image section for one data size."""
    cfg = yaml.safe_load(config_path.read_text())["results_dashboard"]
    block = (cfg.get("image") or {}).get(data_size) or {}
    return {
        domain: project_root / str(rel).strip()
        for domain, rel in (block.get("experiment_dir") or {}).items()
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record each image cell's realised corruption in its fold_info.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[0],
    )
    parser.add_argument("experiment_dirs", nargs="*", type=Path,
                        help="Experiment dirs (…/running_results/<domain>/<experiment>).")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Corpus root holding gt_trajectories/; overrides run_params.json. "
                             "Only with a single experiment dir.")
    parser.add_argument("--from-dashboard-config", action="store_true",
                        help="Take the experiment dirs from dashboard_config.yaml's image section.")
    parser.add_argument("--data-size", default="small", choices=("small", "large"),
                        help="Which image block of the dashboard config to read (default small).")
    parser.add_argument("--config", type=Path, default=DASHBOARD_CONFIG,
                        help="Dashboard config to read with --from-dashboard-config.")
    args = parser.parse_args(argv)

    experiments = list(args.experiment_dirs)
    if args.from_dashboard_config:
        experiments.extend(dashboard_image_experiments(args.config, args.data_size).values())
    if not experiments:
        parser.error("pass experiment dirs or --from-dashboard-config")
    if args.data_dir is not None and len(experiments) != 1:
        parser.error("--data-dir applies to exactly one experiment dir")

    total_recorded = total_cells = 0
    for experiment in experiments:
        recorded, cells = backfill_experiment(experiment, args.data_dir)
        total_recorded += recorded
        total_cells += cells
    print(f"\nRecorded realised corruption in {total_recorded} of {total_cells} cells.")
    return 0 if total_recorded == total_cells else 1


if __name__ == "__main__":
    sys.exit(main())
