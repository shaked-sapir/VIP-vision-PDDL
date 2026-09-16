"""The corruption a fold's observations actually carry, measured against GT.

The ``percentage`` strategies hide or flip ``max(1, round(N * p))`` atoms per
state and frame-axiom propagation restores some of them, so the rate in the
data is not the nominal ``p``. These are the numbers a row records, and the
flip rate is what a learner that takes a noise level as input is handed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import GroundedPredicate, Observation, State

from benchmark.experiment_running_helpers.simulated_data_utils import (
    build_gt_trajectory_lookup,
    load_gt_observation,
)
from src.utils.masking import load_masked_observation
from src.utils.pddl_state import get_state_grounded_predicates

GT_TRAJECTORIES_DIR = "gt_trajectories"


@dataclass(frozen=True)
class RealisedRates:
    """Counts over every state of a fold's observations."""

    n_states: int
    n_atoms: int
    n_masked: int
    n_flipped: int

    @property
    def mask_rate(self) -> float:
        """Hidden atoms over all atoms."""
        return self.n_masked / self.n_atoms if self.n_atoms else 0.0

    @property
    def noise_rate(self) -> float:
        """Flipped atoms over the atoms a learner can see (unmasked)."""
        visible = self.n_atoms - self.n_masked
        return self.n_flipped / visible if visible else 0.0

    def as_dict(self) -> Dict[str, object]:
        return {**asdict(self), "mask_rate": self.mask_rate, "noise_rate": self.noise_rate}


def gt_trajectory_lookup(data_dir: Path) -> Dict[str, Path]:
    """``{problem name: GT .trajectory}`` from ``<data_dir>/gt_trajectories/<problem>/``."""
    root = Path(data_dir) / GT_TRAJECTORIES_DIR
    return build_gt_trajectory_lookup(sorted(root.glob("*/*.trajectory")))


def _atom_key(predicate: GroundedPredicate) -> Tuple[str, Tuple[str, ...]]:
    return predicate.name, tuple(predicate.object_mapping.values())


def _states(observation: Observation) -> List[State]:
    return [observation.components[0].previous_state] + [
        component.next_state for component in observation.components
    ]


def state_counts(observed: State, ground_truth: State) -> Tuple[int, int, int]:
    """``(atoms, masked, flipped)`` for one state against its GT counterpart."""
    truth = {_atom_key(p): p.is_positive for p in get_state_grounded_predicates(ground_truth)}
    atoms = masked = flipped = 0
    for predicate in get_state_grounded_predicates(observed):
        atoms += 1
        if predicate.is_masked:
            masked += 1
            continue
        key = _atom_key(predicate)
        if key not in truth:
            raise ValueError(f"{predicate.untyped_representation} has no GT counterpart")
        if predicate.is_positive != truth[key]:
            flipped += 1
    return atoms, masked, flipped


def _load_observed(
    trajectory_path: Path, masking_path: Optional[Path], domain, problem_path: Path
) -> Observation:
    if masking_path is not None and Path(masking_path).exists():
        return load_masked_observation(trajectory_path, masking_path, domain, problem_path)
    return load_gt_observation(trajectory_path, domain, problem_path)


def realised_rates(
    prepared_trajectories: Sequence[Tuple],
    domain_path: Path,
    gt_lookup: Mapping[str, Path],
) -> RealisedRates:
    """Measure a fold's observations against their GT trajectories.

    Args:
        prepared_trajectories: ``(trajectory, masking_info | None, problem_pddl, ...)``
            tuples, the fold's prepared inputs.
        domain_path: The reference domain.
        gt_lookup: ``{problem name: GT .trajectory}``; see :func:`gt_trajectory_lookup`.

    Raises:
        KeyError: If a trajectory's problem has no GT entry.
        ValueError: If an observation and its GT differ in length.
    """
    domain = DomainParser(Path(domain_path), partial_parsing=True).parse_domain()
    n_states = n_atoms = n_masked = n_flipped = 0
    for trajectory_path, masking_path, problem_path, *_ in prepared_trajectories:
        problem = Path(problem_path).stem
        if problem not in gt_lookup:
            raise KeyError(f"no GT trajectory for {problem}")
        observed = _load_observed(Path(trajectory_path), masking_path, domain, Path(problem_path))
        truth = load_gt_observation(gt_lookup[problem], domain, Path(problem_path))
        observed_states, truth_states = _states(observed), _states(truth)
        if len(observed_states) != len(truth_states):
            raise ValueError(
                f"{problem}: {len(observed_states)} observed states vs "
                f"{len(truth_states)} GT states"
            )
        for observed_state, truth_state in zip(observed_states, truth_states):
            atoms, masked, flipped = state_counts(observed_state, truth_state)
            n_states += 1
            n_atoms += atoms
            n_masked += masked
            n_flipped += flipped
    return RealisedRates(n_states, n_atoms, n_masked, n_flipped)
