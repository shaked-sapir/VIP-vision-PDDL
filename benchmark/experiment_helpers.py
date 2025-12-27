"""
Helper functions for running AMLGym experiments.

This module contains all the trajectory preparation and fold execution logic
for the amlgym_testing experiments.
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Set

from benchmark.amlgym_models.NOISY_PISAM import NOISY_PISAM
from benchmark.amlgym_models.NOISY_SAM import NOISY_SAM
from benchmark.amlgym_models.PISAM import PISAM
from benchmark.amlgym_models.PO_ROSAME import PO_ROSAME
from benchmark.amlgym_models.ROSAME import ROSAME
from benchmark.amlgym_models.SAM import SAM
from src.utils.pddl import inject_gt_states_by_percentage, propagate_frame_axioms_selective


def pregenerate_all_gt_frame_axiom_files(
    problem_dirs: List[Path],
    domain_path: Path,
    gt_rate_percentages: List[int],
    frame_axiom_mode: str
) -> None:
    """
    Pre-generate all GT-injected and frame-axiom-applied trajectory files.
    This avoids race conditions and redundant computation during parallel fold execution.

    Args:
        problem_dirs: List of all problem directories
        domain_path: Path to domain PDDL file
        gt_rate_percentages: List of GT rates to pre-generate (e.g., [0, 10, 25, 50])
        frame_axiom_mode: "after_gt_only" or "all_states"
    """
    print(f"\n{'='*80}")
    print(f"PRE-GENERATING GT and Frame Axiom Files")
    print(f"Problems: {len(problem_dirs)}")
    print(f"GT rates: {gt_rate_percentages}")
    print(f"Frame axiom mode: {frame_axiom_mode}")
    print(f"{'='*80}\n")

    for prob_dir in problem_dirs:
        print(f"  Processing {prob_dir.name}...")

        # Find original trajectory files
        traj_files = [f for f in prob_dir.glob("*.trajectory")
                     if 'truncated' not in f.stem and 'final' not in f.stem
                     and 'frame_axioms' not in f.stem and 'gt' not in f.stem and 'gtrate' not in f.stem]

        if not traj_files:
            print(f"    ⚠ No trajectory file found, skipping")
            continue

        traj_path = traj_files[0]
        problem_name = traj_path.stem
        masking_path = prob_dir / f"{problem_name}.masking_info"
        json_trajectory_path = prob_dir / f"{problem_name}_trajectory.json"

        if not masking_path.exists() or not json_trajectory_path.exists():
            print(f"    ⚠ Missing masking or JSON file, skipping")
            continue

        # Pre-generate for each GT rate
        for gt_rate in gt_rate_percentages:
            output_suffix = f"gtrate{gt_rate}_frame_axioms"
            output_traj = prob_dir / f"{problem_name}_{output_suffix}.trajectory"
            output_masking = prob_dir / f"{problem_name}_{output_suffix}.masking_info"

            # Skip if already exists
            if output_traj.exists() and output_masking.exists():
                print(f"    ✓ GT rate {gt_rate}% already exists")
                continue

            try:
                # Step 1: GT injection
                if gt_rate > 0:
                    gt_traj_path, gt_masking_path, gt_state_indices = inject_gt_states_by_percentage(
                        traj_path, masking_path, json_trajectory_path, domain_path, gt_rate
                    )
                else:
                    gt_traj_path = traj_path
                    gt_masking_path = masking_path
                    gt_state_indices = {0}

                # Step 2: Frame axioms
                final_traj_path, final_masking_path = propagate_frame_axioms_selective(
                    gt_traj_path, gt_masking_path, domain_path, gt_state_indices, mode=frame_axiom_mode
                )

                # Rename to standard naming convention
                shutil.move(final_traj_path, output_traj)
                shutil.move(final_masking_path, output_masking)

                print(f"    ✓ Generated GT rate {gt_rate}%")

            except Exception as e:
                print(f"    ✗ Failed GT rate {gt_rate}%: {e}")
                continue

    print(f"\n{'='*80}")
    print(f"PRE-GENERATION COMPLETE")
    print(f"{'='*80}\n")




def prepare_fold_trajectories(
    selected_problem_dirs: List[Path],
    num_trajectories: int,
    gt_rate: int
) -> List[Tuple[Path, Path, Path]]:
    """
    Load pre-generated GT+frame-axiom trajectory files for a fold.
    Files must have been created by pregenerate_all_gt_frame_axiom_files().

    Args:
        selected_problem_dirs: List of problem directories (pool of 5)
        num_trajectories: Number of trajectories to use (1-5)
        gt_rate: Percentage of states to inject as GT

    Returns:
        List of tuples: (trajectory_path, masking_path, problem_pddl_path)
    """
    # Select only the first num_trajectories from the pool
    selected_dirs = selected_problem_dirs[:num_trajectories]

    prepared_trajectories = []

    for prob_dir in selected_dirs:
        # Find original trajectory to get problem name
        traj_files = [f for f in prob_dir.glob("*.trajectory")
                     if 'truncated' not in f.stem and 'final' not in f.stem
                     and 'frame_axioms' not in f.stem and 'gt' not in f.stem and 'gtrate' not in f.stem]

        if not traj_files:
            print(f"  Warning: No trajectory file found in {prob_dir}")
            continue

        problem_name = traj_files[0].stem

        # Load pre-generated files for this GT rate
        pregenerated_suffix = f"gtrate{gt_rate}_frame_axioms"
        final_traj_path = prob_dir / f"{problem_name}_{pregenerated_suffix}.trajectory"
        final_masking_path = prob_dir / f"{problem_name}_{pregenerated_suffix}.masking_info"
        problem_pddl_path = prob_dir / f"{prob_dir.name}.pddl"

        # Validate all required files exist
        if not final_traj_path.exists():
            print(f"  Warning: Pre-generated trajectory not found: {final_traj_path.name}")
            continue

        if not final_masking_path.exists():
            print(f"  Warning: Pre-generated masking not found: {final_masking_path.name}")
            continue

        if not problem_pddl_path.exists():
            print(f"  Warning: Problem PDDL not found for {prob_dir.name}")
            continue

        prepared_trajectories.append((final_traj_path, final_masking_path, problem_pddl_path))

    return prepared_trajectories


def setup_algorithm_workspace(
    prepared_trajectories: List[Tuple[Path, Path, Path]],
    algorithm_type: str,
    working_directory: Path,
    mode: str
) -> List[str]:
    """
    Set up workspace for an algorithm by copying trajectories and problem files.

    Args:
        prepared_trajectories: List of (trajectory_path, masking_path, problem_pddl_path)
        algorithm_type: "sam" or "rosame"
        working_directory: Directory to create workspace in
        mode: "masked" or "fullyobs"

    Returns:
        List of trajectory paths ready for the algorithm
    """
    workspace_dir = working_directory / f"temp_{algorithm_type}_workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    algorithm_traj_paths = []

    for traj_path, masking_path, problem_pddl_path in prepared_trajectories:
        problem_name = problem_pddl_path.stem

        # Validate input files before copying
        if not traj_path.exists():
            print(f"  Warning: Trajectory file does not exist: {traj_path}")
            continue
        if traj_path.stat().st_size == 0:
            print(f"  Warning: Trajectory file is EMPTY: {traj_path}")
            continue
        if not problem_pddl_path.exists():
            print(f"  Warning: Problem PDDL does not exist: {problem_pddl_path}")
            continue
        if problem_pddl_path.stat().st_size == 0:
            print(f"  Warning: Problem PDDL is EMPTY: {problem_pddl_path}")
            continue

        problem_dir = workspace_dir / problem_name
        problem_dir.mkdir(parents=True, exist_ok=True)

        # Copy trajectory file
        dest_traj_path = problem_dir / f"{problem_name}.trajectory"
        shutil.copy(traj_path, dest_traj_path)

        # Copy problem PDDL file
        shutil.copy(problem_pddl_path, problem_dir / f"{problem_name}.pddl")

        # For masked mode, also copy masking_info
        if mode == "masked" and masking_path.exists():
            shutil.copy(masking_path, problem_dir / f"{problem_name}.masking_info")

        algorithm_traj_paths.append(str(dest_traj_path))

    return algorithm_traj_paths


def save_fold_metadata(
    fold_dir: Path,
    prepared_trajectories: List[Tuple[Path, Path, Path]],
    fold: int,
    num_trajectories: int,
    gt_rate: int
) -> None:
    """
    Save metadata about the fold to a JSON file.

    Args:
        fold_dir: Directory where fold results are saved
        prepared_trajectories: List of (trajectory_path, masking_path, problem_pddl_path)
        fold: Fold number
        num_trajectories: Number of trajectories used
        gt_rate: GT injection rate used
    """
    import json

    metadata = {
        "fold": fold,
        "num_trajectories": num_trajectories,
        "gt_rate_percentage": gt_rate,
        "trajectories": []
    }

    for traj_path, masking_path, problem_pddl_path in prepared_trajectories:
        metadata["trajectories"].append({
            "problem": problem_pddl_path.stem,
            "trajectory_file": traj_path.name,
            "masking_file": masking_path.name if masking_path.exists() else None,
            "problem_file": problem_pddl_path.name
        })

    metadata_path = fold_dir / "fold_info.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


# ==============================================================================
# LEARNING HELPER FUNCTIONS (for cleaner code organization)
# ==============================================================================

def _learn_sam_pisam(
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


def _learn_rosame(
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


def _evaluate_and_build_result(
    model: str,
    model_name: str,
    bench_name: str,
    fold: int,
    num_trajectories: int,
    gt_rate: int,
    test_problem_paths: List[str],
    phase: str,
    domain_ref_path: Path,
    testing_dir: Path,
    evaluate_model_func,
    null_metrics: dict,
    fold_work_dir: Path = None
) -> dict:
    """Evaluate model and build result dictionary."""
    if model:
        print(f"  [DEBUG] Evaluating {model_name} ({phase})...")
        temp_path = testing_dir / f'{model_name}_{phase}_{bench_name}_fold{fold}.pddl'
        temp_path.write_text(model)

        # Save learned domain to fold directory
        if fold_work_dir:
            domain_save_path = fold_work_dir / f'learned_domain_{model_name}_{phase}.pddl'
            domain_save_path.write_text(model)
            print(f"  [DEBUG] Saved learned domain to {domain_save_path.name}")

        metrics = evaluate_model_func(str(temp_path), domain_ref_path, test_problem_paths)
        temp_path.unlink()
        print(f"  [DEBUG] Evaluation complete for {model_name}")
    else:
        print(f"  [DEBUG] No model for {model_name}, using null metrics")
        metrics = null_metrics

    return {
        'domain': bench_name,
        'algorithm': model_name,
        'fold': fold,
        'num_trajectories': num_trajectories,
        'gt_rate': gt_rate,
        'problems_count': len(test_problem_paths),
        '_internal_phase': phase,
        'fold_data_creation_timedout': 0,
        **metrics
    }


def _save_patched_observations(
    patched_observations,
    prepared_trajectories: List[Tuple[Path, Path, Path]],
    output_dir: Path,
    domain_path: Path
) -> None:
    """
    Save patched observations from denoiser to trajectory files.

    Args:
        patched_observations: List of Observation objects from NOISY_SAM/NOISY_PISAM
        prepared_trajectories: Original trajectories (to match observations to problems and get masking)
        output_dir: Directory to save final observations (e.g., fold_work_dir / "final_observations")
        domain_path: Path to domain PDDL file
    """
    from src.utils.pddl import observation_to_trajectory_file
    from src.utils.masking import load_masking_info, save_masking_info
    from pddl_plus_parser.models import Domain
    from pddl_plus_parser.lisp_parsers import DomainParser

    output_dir.mkdir(parents=True, exist_ok=True)

    # For each observation, save it as a trajectory file
    for idx, obs in enumerate(patched_observations):
        if idx >= len(prepared_trajectories):
            print(f"  Warning: More patched observations ({len(patched_observations)}) than prepared trajectories ({len(prepared_trajectories)})")
            break

        orig_traj_path, orig_masking_path, problem_pddl_path = prepared_trajectories[idx]
        problem_name = problem_pddl_path.stem

        # Save trajectory file
        out_name = f"final_observation_{problem_name}"
        traj_file = output_dir / f"{out_name}.trajectory"
        observation_to_trajectory_file(obs, traj_file)

        # Validate trajectory file was written correctly
        if not traj_file.exists():
            print(f"  ERROR: Failed to write trajectory file: {traj_file}")
            continue
        if traj_file.stat().st_size == 0:
            print(f"  ERROR: Trajectory file is EMPTY: {traj_file}")
            continue

        # Load and save masking info
        # The patched observation should maintain the same masking structure
        if orig_masking_path.exists():
            try:
                domain: Domain = DomainParser(domain_path).parse_domain()
                masking_info = load_masking_info(orig_masking_path, domain)

                # Adjust masking info to match patched observation length if needed
                obs_length = len(obs.components) + 1  # +1 for initial state
                if len(masking_info) != obs_length:
                    # Extend or trim masking info to match observation
                    if len(masking_info) < obs_length:
                        masking_info.extend([set()] * (obs_length - len(masking_info)))
                    else:
                        masking_info = masking_info[:obs_length]

                save_masking_info(output_dir, out_name, masking_info)
            except Exception as e:
                print(f"  Warning: Could not process masking info for {problem_name}: {e}")
                # Create empty masking (fully observable)
                masking_info = [set()] * (len(obs.components) + 1)
                save_masking_info(output_dir, out_name, masking_info)
        else:
            # Create empty masking (fully observable)
            masking_info = [set()] * (len(obs.components) + 1)
            save_masking_info(output_dir, out_name, masking_info)


def _convert_cleaned_dir_to_trajectory_list(
    final_observations_dir: Path,
    prepared_trajectories: List[Tuple[Path, Path, Path]]
) -> List[Tuple[Path, Path, Path]]:
    """
    Convert cleaned trajectories from final_observations directory to prepared_trajectories format.

    Args:
        final_observations_dir: Directory containing final_observation_*.trajectory files
        prepared_trajectories: Original trajectories list (to find matching problem PDDLs)

    Returns:
        List of tuples: (cleaned_trajectory_path, masking_path, problem_pddl_path)
    """
    cleaned_list = []

    for final_traj in final_observations_dir.glob("*.trajectory"):
        # Extract problem name from final_observation_{problem_name}.trajectory
        problem_name = final_traj.stem.replace("final_observation_", "")

        # Find corresponding problem PDDL from original prepared trajectories
        problem_pddl = None
        for _, _, prob_path in prepared_trajectories:
            if prob_path.stem == problem_name or problem_name in str(prob_path):
                problem_pddl = prob_path
                break

        if not problem_pddl:
            print(f"  Warning: Could not find problem PDDL for {final_traj.stem}")
            continue

        # Validate problem PDDL file
        if not problem_pddl.exists():
            print(f"  Warning: Problem PDDL does not exist: {problem_pddl}")
            continue
        if problem_pddl.stat().st_size == 0:
            print(f"  Warning: Problem PDDL is EMPTY: {problem_pddl}")
            continue

        # Validate trajectory file
        if final_traj.stat().st_size == 0:
            print(f"  Warning: Cleaned trajectory is EMPTY: {final_traj}")
            continue

        # Find corresponding masking file
        masking_path = final_observations_dir / f"{final_traj.stem}.masking_info"
        if not masking_path.exists():
            # If no masking file exists, use a dummy path (will be ignored in fullyobs mode)
            masking_path = final_traj.parent / "dummy.masking_info"

        cleaned_list.append((final_traj, masking_path, problem_pddl))

    return cleaned_list


def run_single_fold(
    fold: int,
    problem_dirs: List[Path],
    n_problems: int,
    num_trajectories: int,
    gt_rate: int,
    domain_ref_path: Path,
    testing_dir: Path,
    bench_name: str,
    mode: str,
    evaluate_model_func,
    save_learning_metrics_func
) -> List[dict]:
    """
    Run a single fold experiment with specified number of trajectories and GT rate.

    Args:
        fold: Fold number
        problem_dirs: List of all problem directories
        n_problems: Total number of problems
        num_trajectories: Number of trajectories to use for learning (1-5)
        gt_rate: Percentage of states to inject as GT (0, 10, 25, 50)
        domain_ref_path: Path to reference domain PDDL file
        testing_dir: Directory for test results
        bench_name: Benchmark domain name
        mode: 'masked' (PISAM/PO_ROSAME) or 'fullyobs' (SAM/ROSAME)
        frame_axiom_mode: "after_gt_only" or "all_states"
        evaluate_model_func: Function to evaluate a learned model
        save_learning_metrics_func: Function to save learning metrics

    Returns:
        List of 4 dicts with results for: unclean SAM/PISAM, unclean ROSAME, cleaned SAM/PISAM, cleaned ROSAME
    """
    import random

    print(f"[PID {os.getpid()}] Fold {fold}, num_trajs={num_trajectories}, gt_rate={gt_rate}%, mode={mode}")

    # Setup
    fold_work_dir = testing_dir / f"fold{fold}_numtrajs{num_trajectories}_gtrate{gt_rate}"
    fold_work_dir.mkdir(parents=True, exist_ok=True)
    original_cwd = os.getcwd()
    os.chdir(fold_work_dir)

    # Copy domain reference file to fold directory to avoid race conditions
    # Each fold gets its own copy so SimpleDomainReader doesn't conflict
    local_domain_ref = fold_work_dir / "domain_reference.pddl"
    shutil.copy2(domain_ref_path, local_domain_ref)
    domain_ref_path = local_domain_ref  # Use local copy for all evaluations

    null_metrics = {k: None for k in ['precision_precs_pos', 'precision_precs_neg',
                    'precision_eff_pos', 'precision_eff_neg', 'precision_overall',
                    'recall_precs_pos', 'recall_precs_neg', 'recall_eff_pos',
                    'recall_eff_neg', 'recall_overall', 'solving_ratio',
                    'false_plans_ratio', 'unsolvable_ratio', 'timed_out']}

    try:
        # ==================================================
        # Setup: CV split and trajectory preparation
        # ==================================================
        indices = list(range(n_problems))
        random.seed(42 + fold)
        random.shuffle(indices)
        n_train = max(1, min(int(0.8 * n_problems), n_problems - 1))
        train_idx, test_idx = indices[:n_train], indices[n_train:]

        train_problem_dirs = [problem_dirs[i] for i in train_idx]
        test_problem_dirs = [problem_dirs[i] for i in test_idx]

        # Select pool of 5 trajectories, then use first num_trajectories
        random.seed(42 + fold)
        NUM_TRAJECTORIES_POOL = 5
        selected_pool = random.sample(train_problem_dirs, min(NUM_TRAJECTORIES_POOL, len(train_problem_dirs)))

        # Load pre-generated trajectories
        print(f"  Loading {num_trajectories} pre-generated trajectories with gt_rate={gt_rate}%...")
        prepared_trajectories = prepare_fold_trajectories(
            selected_pool, num_trajectories, gt_rate
        )

        if not prepared_trajectories:
            print(f"  ERROR: No trajectories prepared for fold {fold}")
            return []

        print(f"  ✓ Prepared {len(prepared_trajectories)} trajectories")
        save_fold_metadata(fold_work_dir, prepared_trajectories, fold, num_trajectories, gt_rate)

        # Build test problem paths - use same naming convention as prepare_fold_trajectories
        test_problem_paths = []
        for d in test_problem_dirs:
            # Use consistent naming: {problem_dir_name}.pddl
            problem_pddl_path = d / f"{d.name}.pddl"
            if problem_pddl_path.exists():
                test_problem_paths.append(str(problem_pddl_path))
            else:
                # Fallback: try glob if standard naming not found
                pddl_files = list(d.glob("*.pddl"))
                if pddl_files:
                    test_problem_paths.append(str(pddl_files[0]))
                    print(f"  Warning: Used glob fallback for {d.name}, found {pddl_files[0].name}")
                else:
                    print(f"  Warning: No PDDL file found in test directory {d.name}")

        if not test_problem_paths:
            print(f"  ERROR: No test problems found for fold {fold}")
            print(f"  Test directories: {[d.name for d in test_problem_dirs]}")
            return []

        if len(test_problem_paths) < len(test_problem_dirs):
            print(f"  Warning: Only found {len(test_problem_paths)} test problems out of {len(test_problem_dirs)} directories")

        # ==================================================
        # PHASE 1: UNCLEAN (learning on prepared trajectories)
        # ==================================================
        print(f"  [PHASE 1] Learning on unclean trajectories...")

        # Learn SAM/PISAM
        print(f"  [PHASE 1] Starting SAM/PISAM learning...")
        sam_unclean_model, sam_report, sam_algo_name, _ = _learn_sam_pisam(
            mode, domain_ref_path, prepared_trajectories, testing_dir, is_denoising=False
        )
        print(f"  [PHASE 1] SAM/PISAM learning done, saving metrics...")
        save_learning_metrics_func(fold_work_dir, sam_report)
        print(f"  [PHASE 1] Evaluating SAM/PISAM model...")
        unclean_sam_result = _evaluate_and_build_result(
            sam_unclean_model, sam_algo_name, bench_name, fold, num_trajectories, gt_rate,
            test_problem_paths, 'unclean', domain_ref_path, testing_dir,
            evaluate_model_func, null_metrics, fold_work_dir
        )

        # Learn ROSAME
        print(f"  [PHASE 1] Starting ROSAME learning...")
        rosame_unclean_model, rosame_report, rosame_algo_name = _learn_rosame(
            mode, domain_ref_path, prepared_trajectories, testing_dir, "rosame_unclean"
        )
        print(f"  [PHASE 1] ROSAME learning done, saving metrics...")
        save_learning_metrics_func(fold_work_dir, rosame_report)
        print(f"  [PHASE 1] Evaluating ROSAME model...")
        unclean_rosame_result = _evaluate_and_build_result(
            rosame_unclean_model, rosame_algo_name, bench_name, fold, num_trajectories, gt_rate,
            test_problem_paths, 'unclean', domain_ref_path, testing_dir,
            evaluate_model_func, null_metrics, fold_work_dir
        )

        # ==================================================
        # PHASE 2: CLEANED (denoising with NOISY_PISAM/NOISY_SAM)
        # ==================================================
        print(f"  [PHASE 2] Denoising and re-learning...")

        try:
            # Learn with denoiser (NOISY_PISAM/NOISY_SAM) - returns patched observations!
            print(f"  [PHASE 2] Starting denoising (NOISY_SAM/NOISY_PISAM)...")
            cleaned_model, denoising_report, denoiser_algo_name, patched_observations = _learn_sam_pisam(
                mode, domain_ref_path, prepared_trajectories, testing_dir, is_denoising=True
            )
            print(f"  [PHASE 2] Denoising complete, saving metrics...")
            save_learning_metrics_func(fold_work_dir, denoising_report)

            # Evaluate cleaned SAM/PISAM model
            print(f"  [PHASE 2] Evaluating denoised model...")
            cleaned_sam_result = _evaluate_and_build_result(
                cleaned_model, denoiser_algo_name, bench_name, fold, num_trajectories, gt_rate,
                test_problem_paths, 'cleaned', domain_ref_path, testing_dir,
                evaluate_model_func, null_metrics, fold_work_dir
            )

            # SAVE patched observations to disk for ROSAME to use
            if patched_observations is not None:
                print(f"  [PHASE 2] Saving {len(patched_observations)} patched observations to disk...")
                final_observations_dir = fold_work_dir / "final_observations"
                _save_patched_observations(
                    patched_observations, prepared_trajectories, final_observations_dir, domain_ref_path
                )
                print(f"  [PHASE 2] Patched observations saved")

            # Re-learn ROSAME on PATCHED OBSERVATIONS from denoiser
            if patched_observations is not None:
                print(f"  [PHASE 2] Learning ROSAME on patched observations from denoiser...")
                final_observations_dir = fold_work_dir / "final_observations"

                if final_observations_dir.exists():
                    print(f"  [PHASE 2] Converting patched observations to trajectory list...")
                    # Convert patched observations (saved to disk) to trajectory list format
                    # Returns: List[Tuple[traj_path, masking_path, problem_pddl_path]]
                    cleaned_trajectories = _convert_cleaned_dir_to_trajectory_list(
                        final_observations_dir, prepared_trajectories
                    )
                    print(f"  [PHASE 2] Converted {len(cleaned_trajectories)} cleaned trajectories")

                    if cleaned_trajectories:
                        # Learn ROSAME on cleaned trajectories from denoiser
                        print(f"  [PHASE 2] Starting ROSAME learning on cleaned trajectories...")
                        rosame_cleaned_model, rosame_cleaned_report, rosame_cleaned_algo_name = _learn_rosame(
                            mode, domain_ref_path, cleaned_trajectories, testing_dir, "rosame_cleaned"
                        )
                        print(f"  [PHASE 2] Cleaned ROSAME learning done...")

                        print(f"  [PHASE 2] Evaluating cleaned ROSAME model...")
                        cleaned_rosame_result = _evaluate_and_build_result(
                            rosame_cleaned_model, rosame_cleaned_algo_name, bench_name, fold,
                            num_trajectories, gt_rate, test_problem_paths, 'cleaned',
                            domain_ref_path, testing_dir, evaluate_model_func, null_metrics, fold_work_dir
                        )
                    else:
                        print(f"  Warning: No cleaned trajectories found in {final_observations_dir}")
                        cleaned_rosame_result = _evaluate_and_build_result(
                            None, rosame_algo_name, bench_name, fold,
                            num_trajectories, gt_rate, test_problem_paths, 'cleaned',
                            domain_ref_path, testing_dir, evaluate_model_func, null_metrics, fold_work_dir
                        )
                else:
                    print(f"  Warning: final_observations directory does not exist")
                    cleaned_rosame_result = _evaluate_and_build_result(
                        None, rosame_algo_name, bench_name, fold,
                        num_trajectories, gt_rate, test_problem_paths, 'cleaned',
                        domain_ref_path, testing_dir, evaluate_model_func, null_metrics, fold_work_dir
                    )
            else:
                print(f"  Warning: No patched observations returned from denoiser")
                cleaned_rosame_result = _evaluate_and_build_result(
                    None, rosame_algo_name, bench_name, fold,
                    num_trajectories, gt_rate, test_problem_paths, 'cleaned',
                    domain_ref_path, testing_dir, evaluate_model_func, null_metrics, fold_work_dir
                )

        except Exception as e:
            print(f"  ERROR in denoising phase: {e}")
            import traceback
            traceback.print_exc()

            # Return null results on failure - using uniform interface
            cleaned_sam_result = _evaluate_and_build_result(
                None, sam_algo_name, bench_name, fold, num_trajectories, gt_rate,
                test_problem_paths, 'cleaned', domain_ref_path, testing_dir,
                evaluate_model_func, null_metrics, fold_work_dir
            )
            cleaned_rosame_result = _evaluate_and_build_result(
                None, rosame_algo_name, bench_name, fold, num_trajectories, gt_rate,
                test_problem_paths, 'cleaned', domain_ref_path, testing_dir,
                evaluate_model_func, null_metrics, fold_work_dir
            )

        print(f"  [FOLD COMPLETE] Returning 4 results for fold {fold}")
        return [unclean_sam_result, unclean_rosame_result, cleaned_sam_result, cleaned_rosame_result]

    finally:
        os.chdir(original_cwd)
