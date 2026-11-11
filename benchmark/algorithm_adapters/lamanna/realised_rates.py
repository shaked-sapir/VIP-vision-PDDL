"""The corruption a fold's observations actually carry, measured against GT.

The ``percentage`` strategies hide or flip ``max(1, round(N * p))`` atoms per
state and frame-axiom propagation restores some of them, so the rate in the
data is not the nominal ``p``. These are the numbers a row records, and the
flip rate is what a learner that takes a noise level as input is handed.

An image fold has no nominal rate at all, so :func:`realised_corruption` also
splits the counts by frame position (frame 0 against frames 1..T) and measures
how often a flipped atom stays flipped in the next frame; that is what places
an image cell on the simulated grid (``docs/image-experiments-week1-plan.md``,
track C1).
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import Dict, FrozenSet, Iterator, List, Mapping, Optional, Sequence, Tuple

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import GroundedPredicate, Observation, State

from benchmark.experiment_running_helpers.simulated_data_utils import (
    build_gt_trajectory_lookup,
    load_gt_observation,
)
from src.utils.masking import load_masked_observation
from src.utils.pddl_naming import rewrite_symbols, spelling_map
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


@dataclass(frozen=True)
class RealisedCorruption:
    """A fold's realised corruption, split by frame position.

    ``frame0`` covers each trajectory's first state and ``later_frames`` its
    states 1..T; ``total`` is their sum, the figure :func:`realised_rates`
    returns. Frame 0 is kept apart because the image pipeline treats it
    differently from every other frame (either taken from the GT as the
    one-shot example, or classified like the rest under ``__fs=shared``).

    Flip persistence: of the atoms flipped in a frame that has a successor,
    the fraction flipped again in that successor. An atom hidden in the
    successor is not flipped there, so it does not persist.
    """

    total: RealisedRates
    frame0: RealisedRates
    later_frames: RealisedRates
    n_flipped_with_next: int
    n_flip_persisted: int

    @property
    def flip_persistence(self) -> Optional[float]:
        """Persisting flips over flips with a next frame; ``None`` without any."""
        if not self.n_flipped_with_next:
            return None
        return self.n_flip_persisted / self.n_flipped_with_next

    def as_dict(self) -> Dict[str, object]:
        return {
            **self.total.as_dict(),
            "frame0": self.frame0.as_dict(),
            "later_frames": self.later_frames.as_dict(),
            "n_flipped_with_next": self.n_flipped_with_next,
            "n_flip_persisted": self.n_flip_persisted,
            "flip_persistence": self.flip_persistence,
        }


@dataclass
class _Tally:
    """Running counts over states, frozen into a :class:`RealisedRates`."""

    n_states: int = 0
    n_atoms: int = 0
    n_masked: int = 0
    n_flipped: int = 0

    def add(self, atoms: int, masked: int, flipped: int) -> None:
        self.n_states += 1
        self.n_atoms += atoms
        self.n_masked += masked
        self.n_flipped += flipped

    def freeze(self) -> RealisedRates:
        return RealisedRates(self.n_states, self.n_atoms, self.n_masked, self.n_flipped)


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


AtomKey = Tuple[str, Tuple[str, ...]]


def state_flips(observed: State, ground_truth: State) -> Tuple[int, int, FrozenSet[AtomKey]]:
    """``(atoms, masked, flipped atom keys)`` for one state against its GT counterpart."""
    truth = {_atom_key(p): p.is_positive for p in get_state_grounded_predicates(ground_truth)}
    atoms = masked = 0
    flipped = set()
    for predicate in get_state_grounded_predicates(observed):
        atoms += 1
        if predicate.is_masked:
            masked += 1
            continue
        key = _atom_key(predicate)
        if key not in truth:
            raise ValueError(f"{predicate.untyped_representation} has no GT counterpart")
        if predicate.is_positive != truth[key]:
            flipped.add(key)
    return atoms, masked, frozenset(flipped)


def state_counts(observed: State, ground_truth: State) -> Tuple[int, int, int]:
    """``(atoms, masked, flipped)`` for one state against its GT counterpart."""
    atoms, masked, flipped = state_flips(observed, ground_truth)
    return atoms, masked, len(flipped)


def _load_observed(
    trajectory_path: Path, masking_path: Optional[Path], domain, problem_path: Path
) -> Observation:
    if masking_path is not None and Path(masking_path).exists():
        return load_masked_observation(trajectory_path, masking_path, domain, problem_path)
    return load_gt_observation(trajectory_path, domain, problem_path)


def _aligned_states(
    prepared_trajectories: Sequence[Tuple],
    domain_path: Path,
    gt_lookup: Mapping[str, Path],
) -> Iterator[Tuple[str, List[State], List[State]]]:
    """``(problem, observed states, GT states)`` per prepared trajectory.

    Raises:
        KeyError: If a trajectory's problem has no GT entry.
        ValueError: If an observation and its GT differ in length.
    """
    domain = DomainParser(Path(domain_path), partial_parsing=True).parse_domain()
    for trajectory_path, masking_path, problem_path, *_ in prepared_trajectories:
        problem = Path(problem_path).stem
        if problem not in gt_lookup:
            raise KeyError(f"no GT trajectory for {problem}")
        observed = _load_observed(Path(trajectory_path), masking_path, domain, Path(problem_path))
        truth = _load_truth(gt_lookup[problem], domain, Path(problem_path))
        observed_states, truth_states = _states(observed), _states(truth)
        if len(observed_states) != len(truth_states):
            raise ValueError(
                f"{problem}: {len(observed_states)} observed states vs "
                f"{len(truth_states)} GT states"
            )
        yield problem, observed_states, truth_states


def _load_truth(gt_path: Path, domain, problem_path: Path) -> Observation:
    """The GT observation, spelled in ``domain``'s dialect.

    ``gt_trajectories/`` keeps the corpus's raw spelling (``at-robby``) while a
    normalized experiment's cells carry an underscored domain (``at_robby``);
    parsing one against the other raises ``illegal state component``. Each
    head symbol is rewritten to the name the domain declares, which also
    covers a domain that mixes the two spellings.
    """
    spellings = spelling_map(chain(domain.predicates, domain.actions))
    text = Path(gt_path).read_text()
    conformed = rewrite_symbols(text, spellings)
    if conformed == text:
        return load_gt_observation(Path(gt_path), domain, problem_path)
    with tempfile.NamedTemporaryFile("w", suffix=".trajectory", delete=False) as tmp:
        tmp.write(conformed)
        tmp_path = Path(tmp.name)
    try:
        return load_gt_observation(tmp_path, domain, problem_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def realised_corruption(
    prepared_trajectories: Sequence[Tuple],
    domain_path: Path,
    gt_lookup: Mapping[str, Path],
) -> RealisedCorruption:
    """Measure a fold's observations against their GT trajectories, frame by frame.

    Args:
        prepared_trajectories: ``(trajectory, masking_info | None, problem_pddl, ...)``
            tuples, the fold's prepared inputs.
        domain_path: The reference domain.
        gt_lookup: ``{problem name: GT .trajectory}``; see :func:`gt_trajectory_lookup`.

    Raises:
        KeyError: If a trajectory's problem has no GT entry.
        ValueError: If an observation and its GT differ in length.
    """
    total, frame0, later = _Tally(), _Tally(), _Tally()
    n_flipped_with_next = n_flip_persisted = 0
    for _problem, observed_states, truth_states in _aligned_states(
        prepared_trajectories, domain_path, gt_lookup
    ):
        previous: Optional[FrozenSet[AtomKey]] = None
        for index, (observed_state, truth_state) in enumerate(zip(observed_states, truth_states)):
            atoms, masked, flipped = state_flips(observed_state, truth_state)
            total.add(atoms, masked, len(flipped))
            (frame0 if index == 0 else later).add(atoms, masked, len(flipped))
            if previous is not None:
                n_flipped_with_next += len(previous)
                n_flip_persisted += len(previous & flipped)
            previous = flipped
    return RealisedCorruption(
        total.freeze(), frame0.freeze(), later.freeze(), n_flipped_with_next, n_flip_persisted
    )


def realised_rates(
    prepared_trajectories: Sequence[Tuple],
    domain_path: Path,
    gt_lookup: Mapping[str, Path],
) -> RealisedRates:
    """The fold-wide totals of :func:`realised_corruption`; same arguments, same errors."""
    return realised_corruption(prepared_trajectories, domain_path, gt_lookup).total
