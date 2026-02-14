"""
Trajectory preparation and management utilities for AMLGym experiments.
"""

import json
import shutil
from pathlib import Path
from typing import List, Tuple

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
    gt_rate: int,
    test_problem_paths: List[str] = None
) -> None:
    """
    Save metadata about the fold to a JSON file.

    Args:
        fold_dir: Directory where fold results are saved
        prepared_trajectories: List of (trajectory_path, masking_path, problem_pddl_path)
        fold: Fold number
        num_trajectories: Number of trajectories used
        gt_rate: GT injection rate used
        test_problem_paths: Optional list of test problem file paths
    """
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

    # Add test problem names if provided
    if test_problem_paths:
        metadata["test_problems"] = [Path(p).stem for p in test_problem_paths]

    metadata_path = fold_dir / "fold_info.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def update_fold_metadata(
    fold_dir: Path,
    cleaned_equals_unclean_pisam: bool = None,
    cleaned_equals_unclean_rosame: bool = None
) -> None:
    """
    Update fold metadata JSON with additional information about trajectory comparisons.
    
    Args:
        fold_dir: Directory where fold results are saved
        cleaned_equals_unclean_pisam: Whether cleaned and unclean trajectories are equal for PISAM/SAM
        cleaned_equals_unclean_rosame: Whether cleaned and unclean trajectories are equal for ROSAME
    """
    metadata_path = fold_dir / "fold_info.json"
    
    # Load existing metadata if it exists
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Add comparison information
    if cleaned_equals_unclean_pisam is not None:
        metadata["cleaned_equals_unclean_pisam"] = cleaned_equals_unclean_pisam
        if cleaned_equals_unclean_pisam:
            metadata["note_pisam"] = "Cleaned and unclean trajectories are EQUAL - denoiser did not modify trajectories"
    
    if cleaned_equals_unclean_rosame is not None:
        metadata["cleaned_equals_unclean_rosame"] = cleaned_equals_unclean_rosame
        if cleaned_equals_unclean_rosame:
            metadata["note_rosame"] = "Cleaned and unclean trajectories are EQUAL - denoiser did not modify trajectories"
    
    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

