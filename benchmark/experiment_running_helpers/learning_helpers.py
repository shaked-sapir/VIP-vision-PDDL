"""
Learning algorithm wrapper functions for AMLGym experiments.
"""

from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import time

from pddl_plus_parser.lisp_parsers import DomainParser
from utilities import NegativePreconditionPolicy

from benchmark.amlgym_models.NOISY_PISAM import NOISY_PISAM
from benchmark.amlgym_models.NOISY_SAM import NOISY_SAM
from benchmark.amlgym_models.PISAM import PISAM
from benchmark.amlgym_models.PO_ROSAME import PO_ROSAME
from benchmark.amlgym_models.ROSAME import ROSAME
from benchmark.amlgym_models.SAM import SAM
from benchmark.experiment_running_helpers.trajectory_utils import setup_algorithm_workspace
from src.pi_sam import PISAMLearner
from src.pi_sam.plan_denoising.conflict_search import ConflictDrivenPatchSearch
from src.utils.masking import load_masked_observation


def _parse_learning_output(learning_output, is_denoising):
    """Extract model, patched_observations, and report from learning output."""
    if is_denoising and isinstance(learning_output, tuple) and len(learning_output) == 3:
        return learning_output[0], learning_output[1], learning_output[2]
    return learning_output, None, {}


def _learn_pisam_with_profiling(domain_ref_path, traj_paths, is_denoising, learner, phase, algo_name, profiler, fold_work_dir=None):
    """Learn PISAM with detailed profiling."""
    partial_domain = DomainParser(Path(str(domain_ref_path)), partial_parsing=True).parse_domain()
    masked_observations = []
    
    for traj_idx, traj_path_str in enumerate(traj_paths):
        traj_path = Path(traj_path_str)
        masking_info_path = traj_path.parent / f"{traj_path.stem}.masking_info"
        
        if not masking_info_path.exists():
            continue
        
        def timing_callback(step_name, elapsed):
            profiler.add_detailed_timing(
                f"sam_pisam_trajectory_processing_{phase}",
                step_name, elapsed,
                {'trajectory_index': traj_idx, 'problem_name': traj_path.stem}
            )
        
        # Measure total time for load_masked_observation for each trajectory
        start_load = time.perf_counter()
        masked_obs = load_masked_observation(traj_path, masking_info_path, partial_domain, timing_callback=timing_callback)
        load_elapsed = time.perf_counter() - start_load
        
        # Record total time for loading this trajectory
        profiler.add_detailed_timing(
            f"sam_pisam_trajectory_loading_{phase}",
            'load_masked_observation_total',
            load_elapsed,
            {'trajectory_index': traj_idx, 'problem_name': traj_path.stem}
        )
        
        masked_observations.append(masked_obs)
    
    start_learn = time.perf_counter()
    
    if is_denoising:
        # Create conflict_free_models directory if fold_work_dir is provided
        conflict_free_models_dir = None
        if fold_work_dir is not None:
            conflict_free_models_dir = fold_work_dir / "conflict_free_models"
        
        conflict_search = ConflictDrivenPatchSearch(
            partial_domain_template=deepcopy(partial_domain),
            negative_preconditions_policy=learner.negative_precondition_policy,
            seed=learner.seed,
            logger=None,
            conflict_free_models_dir=conflict_free_models_dir
        )
        learned_model, _, _, _, _, report, patched_observations = conflict_search.run(
            observations=masked_observations,
            max_nodes=learner.max_search_nodes,
            timeout_seconds=learner.timeout_seconds
        )
        model = learned_model.to_pddl()
    else:
        pi_sam = PISAMLearner(partial_domain=partial_domain, negative_preconditions_policy=NegativePreconditionPolicy.hard)
        learned_model, _ = pi_sam.learn_action_model(masked_observations)
        model = learned_model.to_pddl()
        patched_observations = None
        report = {}
    
    profiler.add_timing(f"learning_process_{algo_name}_{phase}", time.perf_counter() - start_learn)
    return model, report, patched_observations


def learn_sam_pisam(
    mode: str,
    domain_ref_path: Path,
    prepared_trajectories: List[Tuple[Path, Path, Path]],
    testing_dir: Path,
    is_denoising: bool = False,
    conflict_search_timeout: int = None,
    profiler=None,
    fold_work_dir: Path = None
) -> Tuple[str, dict, str, any]:
    """
    Learn SAM/PISAM model.

    Args:
        mode: 'masked' or 'fullyobs'
        domain_ref_path: Path to reference domain PDDL
        prepared_trajectories: List of (trajectory_path, masking_path, problem_pddl_path)
        testing_dir: Working directory
        is_denoising: If True, use NOISY_PISAM/NOISY_SAM (returns learning report and patched observations)
        conflict_search_timeout: Optional timeout in seconds for conflict search (cleaning phase)
        profiler: Optional TimingProfiler instance for detailed timing
        fold_work_dir: Optional fold working directory for saving conflict-free models

    Returns:
        Tuple of (model, learning_report, algorithm_name, patched_observations)
    """
    phase = "cleaned" if is_denoising else "unclean"
    algo_name = 'PISAM' if mode == 'masked' else 'SAM'
    
    # Track the actual timeout value used (for cleaned phase only)
    actual_learning_timeout = None
    
    if mode == 'masked':
        traj_paths = [str(t[0]) for t in prepared_trajectories]
        learner = NOISY_PISAM() if is_denoising else PISAM()
        if is_denoising:
            if conflict_search_timeout is not None:
                learner.timeout_seconds = conflict_search_timeout
            # Capture actual timeout used (either explicit or default)
            actual_learning_timeout = learner.timeout_seconds
        
        if profiler:
            model, report, patched_observations = _learn_pisam_with_profiling(
                domain_ref_path, traj_paths, is_denoising, learner, phase, algo_name, profiler, fold_work_dir
            )
        else:
            learning_output = learner.learn(str(domain_ref_path), traj_paths, use_problems=False)
            model, patched_observations, report = _parse_learning_output(learning_output, is_denoising)
    else:  # fullyobs
        workspace_name = "noisy_sam" if is_denoising else "sam_unclean"
        traj_paths = setup_algorithm_workspace(prepared_trajectories, workspace_name, testing_dir, mode)
        learner = NOISY_SAM() if is_denoising else SAM()
        if is_denoising:
            if conflict_search_timeout is not None:
                learner.timeout_seconds = conflict_search_timeout
            # Capture actual timeout used (either explicit or default)
            actual_learning_timeout = learner.timeout_seconds
        
        learning_output = learner.learn(str(domain_ref_path), traj_paths, use_problems=False)
        model, patched_observations, report = _parse_learning_output(learning_output, is_denoising)
    
    # Add actual timeout to report if denoising (cleaned phase)
    if is_denoising and actual_learning_timeout is not None:
        if report is None:
            report = {}
        report['actual_timeout_seconds'] = actual_learning_timeout
    
    return model, report, algo_name, patched_observations


def learn_rosame(
    mode: str,
    domain_ref_path: Path,
    prepared_trajectories: List[Tuple[Path, Path, Path]],
    testing_dir: Path,
    workspace_name: str,
    profiler=None
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
        print(f"  [DEBUG] Starting {algo_name} learning...")
        # Call static method directly on the class
        if mode == 'masked':
            model = PO_ROSAME.learn(str(domain_ref_path), traj_paths, use_problems=False, profiler=profiler)
        else:
            model = ROSAME.learn(str(domain_ref_path), traj_paths, use_problems=False, profiler=profiler)
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

