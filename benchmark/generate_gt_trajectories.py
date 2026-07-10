"""
Ground-truth trajectory export.

For an experiment output directory produced by ``data_generator``, this builds a
``gt_trajectories/`` subdir holding one GT ``.trajectory`` file per problem,
derived from each problem's retained ``_trajectory.json`` (the true, un-noised
states).

Layout (nested, matching build_gt_trajectory_lookup in the simulated data
source, which maps ``path.parent.name`` → problem name):
    <experiment>/gt_trajectories/problemN/problemN.trajectory

Usable in two ways:
    - Imported by ``data_generator`` and called after inference.
    - Run standalone on an already-created experiment dir:
        python -m benchmark.generate_gt_trajectories --experiment-dir <path>
"""

import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmark.data_generator import _collect_problem_dirs
from src.utils.pddl_trajectory import build_trajectory_file


def generate_gt_trajectories(trajectories_dir: Path, experiment_dir: Path) -> Path:
    """Export one GT .trajectory per problem into experiment_dir/gt_trajectories/.

    Each problem subdir under trajectories_dir keeps its GT ``_trajectory.json``
    (the true, un-noised states). This converts that JSON back into a GT
    ``.trajectory`` file using the shared ``build_trajectory_file`` helper.

    Args:
        trajectories_dir: Directory containing the per-problem subdirs.
        experiment_dir: Experiment root; gt_trajectories/ is created here.

    Returns:
        Path to the created gt_trajectories/ directory.
    """
    gt_root = experiment_dir / "gt_trajectories"
    gt_root.mkdir(parents=True, exist_ok=True)

    for problem_dir in _collect_problem_dirs(trajectories_dir):
        problem_name = problem_dir.name
        json_paths = list(problem_dir.glob("*_trajectory.json"))
        if not json_paths:
            print(f"  ! Skipping {problem_name}: no _trajectory.json found.")
            continue

        with open(json_paths[0]) as f:
            trajectory_data = json.load(f)

        problem_gt_dir = gt_root / problem_name
        problem_gt_dir.mkdir(parents=True, exist_ok=True)
        build_trajectory_file(trajectory_data, problem_name, problem_gt_dir)

    return gt_root


def _resolve_trajectories_dir(experiment_dir: Path) -> Path:
    """Locate the per-problem trajectories dir inside an experiment dir."""
    trajectories_dir = experiment_dir / "training" / "trajectories"
    if not trajectories_dir.is_dir():
        raise FileNotFoundError(
            f"No training/trajectories dir found under {experiment_dir}. "
            f"Expected: {trajectories_dir}"
        )
    return trajectories_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export GT .trajectory files for an existing experiment dir."
    )
    parser.add_argument(
        "--experiment-dir", type=str, required=True,
        help="Path to an experiment dir (containing training/trajectories/). "
             "gt_trajectories/ is written alongside training/.",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    trajectories_dir = _resolve_trajectories_dir(experiment_dir)

    gt_root = generate_gt_trajectories(trajectories_dir, experiment_dir)
    print(f"\nGT trajectories written to: {gt_root}")
