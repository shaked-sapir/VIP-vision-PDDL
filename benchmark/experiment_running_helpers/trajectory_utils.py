"""
Trajectory preparation and management utilities for AMLGym experiments.
"""

import json
import shutil
from pathlib import Path
from typing import List, Set, Tuple

from src.utils.pddl import inject_gt_states_by_percentage, propagate_frame_axioms_selective


def _resolve_gt_json_path(prob_dir: Path, problem_name: str) -> Path:
    """Resolve the ground-truth trajectory JSON for GT-rate injection.

    Prefers the clean ``gt_trajectories/{problem}/{problem}_trajectory.json``
    (true states) over the in-dir ``{problem}_trajectory.json`` (which, for
    generate-mode datasets, holds the classifier's noisy states). Falls back to
    the in-dir JSON for legacy datasets that predate the gt_trajectories/ split.

    ``prob_dir`` is ``<data_dir>/training/trajectories/<problem>``, so the
    experiment/data root is ``prob_dir.parents[2]``.
    """
    in_dir = prob_dir / f"{problem_name}_trajectory.json"
    try:
        gt_json = prob_dir.parents[2] / "gt_trajectories" / prob_dir.name / f"{problem_name}_trajectory.json"
    except IndexError:
        return in_dir
    if gt_json.exists():
        return gt_json
    print(f"    ⚠ No gt_trajectories JSON for {prob_dir.name}; "
          f"falling back to in-dir JSON (may be noisy for generate-mode data)")
    return in_dir


def _save_gt_indices(path: Path, indices: Set[int]) -> None:
    path.write_text(json.dumps(sorted(indices)))


def load_gt_indices(path: Path) -> Set[int]:
    """Load GT state indices from a .gt_indices sidecar file."""
    if not path.exists():
        return {0}
    return set(json.loads(path.read_text()))


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
        json_trajectory_path = _resolve_gt_json_path(prob_dir, problem_name)

        if not masking_path.exists() or not json_trajectory_path.exists():
            print(f"    ⚠ Missing masking or JSON file, skipping")
            continue

        # Pre-generate for each GT rate
        for gt_rate in gt_rate_percentages:
            output_suffix = f"gtrate{gt_rate}_frame_axioms"
            output_traj = prob_dir / f"{problem_name}_{output_suffix}.trajectory"
            output_masking = prob_dir / f"{problem_name}_{output_suffix}.masking_info"
            output_gt_indices = prob_dir / f"{problem_name}_{output_suffix}.gt_indices"

            # Skip if already exists
            if output_traj.exists() and output_masking.exists():
                if not output_gt_indices.exists():
                    _save_gt_indices(output_gt_indices, {0} if gt_rate == 0 else set())
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

                # Persist GT state indices for downstream conflict search
                _save_gt_indices(output_gt_indices, gt_state_indices)

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
) -> List[Tuple[Path, Path, Path, Set[int]]]:
    """
    Load pre-generated GT+frame-axiom trajectory files for a fold.
    Files must have been created by pregenerate_all_gt_frame_axiom_files().

    Args:
        selected_problem_dirs: List of problem directories (pool of 5)
        num_trajectories: Number of trajectories to use (1-5)
        gt_rate: Percentage of states to inject as GT

    Returns:
        List of tuples: (trajectory_path, masking_path, problem_pddl_path, gt_state_indices)
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
        gt_indices_path = prob_dir / f"{problem_name}_{pregenerated_suffix}.gt_indices"

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

        gt_indices = load_gt_indices(gt_indices_path)
        prepared_trajectories.append((final_traj_path, final_masking_path, problem_pddl_path, gt_indices))

    return prepared_trajectories


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

    for traj_path, masking_path, problem_pddl_path, *_ in prepared_trajectories:
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
) -> None:
    """Update fold_info.json with whether CDPS's patched trajectories == the input.

    Args:
        fold_dir: Directory where fold results are saved.
        cleaned_equals_unclean_pisam: Whether the patched (denoised) trajectories
            equal the input degraded trajectories (i.e. CDPS made no change).
    """
    metadata_path = fold_dir / "fold_info.json"

    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    if cleaned_equals_unclean_pisam is not None:
        metadata["cleaned_equals_unclean_pisam"] = cleaned_equals_unclean_pisam
        if cleaned_equals_unclean_pisam:
            metadata["note_pisam"] = "Patched and input trajectories are EQUAL - denoiser did not modify trajectories"

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

