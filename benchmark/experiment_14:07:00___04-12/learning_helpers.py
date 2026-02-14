"""
Learning algorithm wrapper functions for AMLGym experiments.
"""

from pathlib import Path
from typing import List, Tuple

from benchmark.amlgym_models.NOISY_PISAM import NOISY_PISAM
from benchmark.amlgym_models.NOISY_SAM import NOISY_SAM
from benchmark.amlgym_models.PISAM import PISAM
from benchmark.amlgym_models.PO_ROSAME import PO_ROSAME
from benchmark.amlgym_models.ROSAME import ROSAME
from benchmark.amlgym_models.SAM import SAM
from benchmark.trajectory_utils import setup_algorithm_workspace


def learn_sam_pisam(
    mode: str,
    domain_ref_path: Path,
    prepared_trajectories: List[Tuple[Path, Path, Path]],
    testing_dir: Path,
    is_denoising: bool = False
) -> Tuple[str, dict, str, any]:
    """
    Learn SAM/PISAM model.

    Args:
        mode: 'masked' or 'fullyobs'
        domain_ref_path: Path to reference domain PDDL
        prepared_trajectories: List of (trajectory_path, masking_path, problem_pddl_path)
        testing_dir: Working directory
        is_denoising: If True, use NOISY_PISAM/NOISY_SAM (returns learning report and patched observations)

    Returns:
        Tuple of (model, learning_report, algorithm_name, patched_observations)
        - model: PDDL model string or None on failure
        - learning_report: Dict with denoising metrics (empty {} if not denoising)
        - algorithm_name: "SAM" or "PISAM"
        - patched_observations: Cleaned trajectories from denoiser (None if not denoising)
    """
    if mode == 'masked':
        traj_paths = [str(t[0]) for t in prepared_trajectories]
        learner = NOISY_PISAM() if is_denoising else PISAM()
        algo_name = 'PISAM'
    else:  # fullyobs
        workspace_name = "noisy_sam" if is_denoising else "sam_unclean"
        print(f"  [DEBUG] Setting up workspace for {workspace_name}...")
        traj_paths = setup_algorithm_workspace(prepared_trajectories, workspace_name, testing_dir, mode)
        print(f"  [DEBUG] Workspace setup complete, {len(traj_paths)} trajectories")
        learner = NOISY_SAM() if is_denoising else SAM()
        algo_name = 'SAM'

    print(f"  [DEBUG] Starting {algo_name} learning (denoising={is_denoising})...")
    learning_output = learner.learn(str(domain_ref_path), traj_paths, use_problems=False)
    print(f"  [DEBUG] {algo_name} learning complete")

    # Denoising algorithms return (model, patched_observations, report)
    # Regular learners return just model
    if is_denoising and isinstance(learning_output, tuple) and len(learning_output) == 3:
        model, patched_observations, report = learning_output
    else:
        model = learning_output
        patched_observations = None
        report = {}

    return model, report, algo_name, patched_observations


def learn_rosame(
    mode: str,
    domain_ref_path: Path,
    prepared_trajectories: List[Tuple[Path, Path, Path]],
    testing_dir: Path,
    workspace_name: str
) -> Tuple[str, dict, str]:
    """
    Learn ROSAME/PO_ROSAME model.

    Args:
        mode: 'masked' (uses PO_ROSAME) or 'fullyobs' (uses ROSAME)
        domain_ref_path: Path to reference domain PDDL
        prepared_trajectories: List of (trajectory_path, masking_path, problem_pddl_path)
        testing_dir: Working directory
        workspace_name: Name for workspace directory

    Returns:
        Tuple of (model, learning_report, algorithm_name)
        - model: PDDL model string or None on failure
        - learning_report: Always {} (ROSAME has no learning report)
        - algorithm_name: "PO_ROSAME" or "ROSAME"
    """
    algo_name = 'PO_ROSAME' if mode == 'masked' else 'ROSAME'
    print(f"  [DEBUG] Setting up workspace for {workspace_name}...")
    traj_paths = setup_algorithm_workspace(prepared_trajectories, workspace_name, testing_dir, mode)
    print(f"  [DEBUG] Workspace setup complete, {len(traj_paths)} trajectories")

    if not traj_paths:
        print(f"  [DEBUG] No trajectories for ROSAME, skipping")
        return None, {}, algo_name

    try:
        learner = PO_ROSAME() if mode == 'masked' else ROSAME()
        print(f"  [DEBUG] Starting {algo_name} learning...")
        model = learner.learn(str(domain_ref_path), traj_paths, use_problems=False)
        print(f"  [DEBUG] {algo_name} learning complete")

        if model and ":action" in model:
            return model, {}, algo_name
        else:
            raise ValueError("Invalid ROSAME model")
    except Exception as e:
        print(f"  Warning: ROSAME learning failed: {e}")
        print(f"  Domain ref: {domain_ref_path}")
        print(f"  Num trajectories: {len(traj_paths)}")
        print(f"  Workspace: {workspace_name}")

        # Check if any files are empty or malformed
        if domain_ref_path.exists():
            size = domain_ref_path.stat().st_size
            print(f"  Domain file size: {size} bytes")
            if size == 0:
                print(f"  ERROR: Domain file is EMPTY!")
        else:
            print(f"  ERROR: Domain file does not exist!")

        for i, traj_path in enumerate(traj_paths):
            traj_file = Path(traj_path)
            if traj_file.exists():
                size = traj_file.stat().st_size
                if size == 0:
                    print(f"  ERROR: Trajectory {i} is EMPTY: {traj_path}")
            else:
                print(f"  ERROR: Trajectory {i} does not exist: {traj_path}")

        return None, {}, algo_name

