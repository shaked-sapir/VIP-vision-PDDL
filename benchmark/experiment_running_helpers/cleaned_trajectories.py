"""
Cleaned trajectory handling utilities for AMLGym experiments.
"""

from pathlib import Path
from typing import List, Tuple

from pddl_plus_parser.lisp_parsers import DomainParser
from src.utils.masking import load_masking_info, save_masking_info
from src.utils.pddl import observation_to_trajectory_file


def save_patched_observations(
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
                domain = DomainParser(domain_path).parse_domain()
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


def convert_cleaned_dir_to_trajectory_list(
    final_observations_dir: Path,
    prepared_trajectories: List[Tuple[Path, Path, Path]]
) -> List[Tuple[Path, Path, Path]]:
    """
    Convert cleaned trajectories from final_observations directory to prepared_trajectories format.
    
    Maintains the same order as prepared_trajectories to ensure consistency.

    Args:
        final_observations_dir: Directory containing final_observation_*.trajectory files
        prepared_trajectories: Original trajectories list (to find matching problem PDDLs and maintain order)

    Returns:
        List of tuples: (cleaned_trajectory_path, masking_path, problem_pddl_path)
        Ordered to match prepared_trajectories
    """
    cleaned_list = []

    # Iterate through prepared_trajectories in order to maintain ordering
    for _, _, problem_pddl_path in prepared_trajectories:
        problem_name = problem_pddl_path.stem
        
        # Find corresponding cleaned trajectory file
        final_traj = final_observations_dir / f"final_observation_{problem_name}.trajectory"
        
        if not final_traj.exists():
            print(f"  Warning: Cleaned trajectory not found for {problem_name}: {final_traj.name}")
            continue

        # Validate trajectory file
        if final_traj.stat().st_size == 0:
            print(f"  Warning: Cleaned trajectory is EMPTY: {final_traj}")
            continue

        # Validate problem PDDL file
        if not problem_pddl_path.exists():
            print(f"  Warning: Problem PDDL does not exist: {problem_pddl_path}")
            continue
        if problem_pddl_path.stat().st_size == 0:
            print(f"  Warning: Problem PDDL is EMPTY: {problem_pddl_path}")
            continue

        # Find corresponding masking file
        masking_path = final_observations_dir / f"{final_traj.stem}.masking_info"
        if not masking_path.exists():
            # If no masking file exists, use a dummy path (will be ignored in fullyobs mode)
            masking_path = final_traj.parent / "dummy.masking_info"

        cleaned_list.append((final_traj, masking_path, problem_pddl_path))

    return cleaned_list

