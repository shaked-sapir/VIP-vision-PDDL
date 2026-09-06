"""
Observation serialization utilities for AMLGym experiments.

Serialize in-memory Observation objects to on-disk .trajectory + .masking_info
files (degraded "original_observations" and CDPS-patched "final_observations").
"""

from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

from pddl_plus_parser.models import Observation

from src.utils.masking import save_masking_info
from src.utils.pddl import observation_to_trajectory_file
from src.utils.pddl_state import get_state_masked_predicates


def extract_masking_info_from_observation(observation: Observation) -> List[Set]:
    """Extract per-state masked predicate sets from an in-memory observation."""
    states = [observation.components[0].previous_state] + [
        comp.next_state for comp in observation.components
    ]
    return [get_state_masked_predicates(state) for state in states]


def _problem_names(
    n_observations: int,
    prepared_trajectories: Sequence[Tuple[Path, Path, Path]],
    observation_indices: Optional[Sequence[int]],
) -> List[str]:
    """Problem name for each observation position.

    ``observation_indices`` maps position -> index into ``prepared_trajectories``
    for callers that hand back a subset of the fold (the MILP loop); ``None``
    means positional, i.e. observation ``i`` is the fold's ``i``-th trajectory.
    """
    if observation_indices is None:
        observation_indices = range(n_observations)
    elif len(observation_indices) != n_observations:
        raise ValueError(
            f"{len(observation_indices)} observation indices for "
            f"{n_observations} observations"
        )
    names = []
    for index in observation_indices:
        if index >= len(prepared_trajectories):
            print(
                f"  Warning: observation index {index} beyond the "
                f"{len(prepared_trajectories)} prepared trajectories"
            )
            break
        names.append(prepared_trajectories[index][2].stem)
    return names


def save_fold_observations(
    observations: List[Observation],
    prepared_trajectories: List[Tuple[Path, Path, Path]],
    output_dir: Path,
    observation_prefix: str,
    observation_indices: Optional[Sequence[int]] = None,
) -> None:
    """Save observations as trajectory + masking_info files under *output_dir*.

    Masking is extracted from each observation's ``is_masked`` flags so the
    saved files reflect exactly what was passed to conflict search (file-based
    or simulated). ``observation_indices`` names each observation by the fold
    trajectory at that index rather than by its position (see
    :func:`_problem_names`).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    problem_names = _problem_names(
        len(observations), prepared_trajectories, observation_indices
    )

    for obs, problem_name in zip(observations, problem_names):
        out_name = f"{observation_prefix}_{problem_name}"
        traj_file = output_dir / f"{out_name}.trajectory"
        observation_to_trajectory_file(obs, traj_file)

        if not traj_file.exists():
            print(f"  ERROR: Failed to write trajectory file: {traj_file}")
            continue
        if traj_file.stat().st_size == 0:
            print(f"  ERROR: Trajectory file is EMPTY: {traj_file}")
            continue

        masking_info = extract_masking_info_from_observation(obs)
        save_masking_info(output_dir, out_name, masking_info)


def save_observations_to_dir(
    patched_observations,
    prepared_trajectories: List[Tuple[Path, Path, Path]],
    output_dir: Path,
    observation_indices: Optional[Sequence[int]] = None,
) -> None:
    """Save observation list to trajectory files only (no masking). Names by problem from prepared_trajectories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    problem_names = _problem_names(
        len(patched_observations), prepared_trajectories, observation_indices
    )
    for obs, problem_name in zip(patched_observations, problem_names):
        traj_file = output_dir / f"final_observation_{problem_name}.trajectory"
        observation_to_trajectory_file(obs, traj_file)


def save_patched_observations(
    patched_observations,
    prepared_trajectories: List[Tuple[Path, Path, Path]],
    output_dir: Path,
    domain_path: Path,
    observation_indices: Optional[Sequence[int]] = None,
) -> None:
    """
    Save patched observations from denoiser to trajectory files.

    Args:
        patched_observations: List of Observation objects from CDPS (NOISY_PISAM conflict search)
        prepared_trajectories: Original trajectories (to match observations to problems)
        output_dir: Directory to save final observations (e.g., fold_work_dir / "final_observations")
        domain_path: Unused; kept for call-site compatibility.
        observation_indices: Index into ``prepared_trajectories`` per observation,
            for denoisers that return a subset of the fold; ``None`` = positional.
    """
    save_fold_observations(
        patched_observations,
        prepared_trajectories,
        output_dir,
        observation_prefix="final_observation",
        observation_indices=observation_indices,
    )

