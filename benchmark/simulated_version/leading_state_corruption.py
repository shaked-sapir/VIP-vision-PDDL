"""Simulated observations whose first states are degraded like every later one.

``create_bounded_noisy_observation`` leaves state 0 untouched and repairs state 1
from it. This module rebuilds those two states from the ground-truth trajectory
with the same per-state masking and flipping, splices them into an observation
that is already frozen on disk, and leaves every other line of that observation
as it was.
"""

import hashlib
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from pddl_plus_parser.models import Domain, Observation, State

from benchmark.experiment_running_helpers.cleaned_trajectories import (
    extract_masking_info_from_observation,
)
from benchmark.experiment_running_helpers.simulated_data_utils import (
    _masking_kwargs,
    _noising_kwargs,
    load_gt_observation,
)
from src.observation_degradation.masking import MaskingType
from src.observation_degradation.noising import NoisingType
from src.observation_degradation.predicate_masking import PredicateMasker
from src.observation_degradation.predicate_noising import PredicateNoiser
from src.utils.masking import save_masking_info
from src.utils.pddl_state import copy_observation_linked, flip_fluent_in_state
from src.utils.pddl_trajectory import observation_to_trajectory_file

LEADING_STATE_INDICES: Tuple[int, ...] = (0, 1)

_NEGATED_ENTRY = re.compile(r"\(not (\([^()]*\))\)")
_FLUENT = re.compile(r"\([^()]*\)")
_MASKING_SEPARATOR = ", "


@dataclass(frozen=True)
class DegradationSpec:
    """The masking and noising a cell applies to each state."""

    masking_strategy: MaskingType
    masking_p: float
    noising_strategy: NoisingType
    noising_p: float
    seed: int


def leading_state_seed(base_seed: int, fold: int, problem_name: str, state_index: int) -> int:
    """Seed for one leading state, stable across processes and training sizes."""
    key = f"{base_seed}|{fold}|{problem_name}|{state_index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def observation_states(observation: Observation) -> List[State]:
    """The observation's states in order, initial state first."""
    return [observation.components[0].previous_state] + [
        component.next_state for component in observation.components
    ]


def _degrade_state(state: State, spec: DegradationSpec, seed: int) -> None:
    """Mask, then flip among what stayed visible, in place."""
    masker = PredicateMasker(
        seed=seed,
        masking_strategy=spec.masking_strategy,
        masking_kwargs=_masking_kwargs(spec.masking_strategy, spec.masking_p),
    )
    noiser = PredicateNoiser(
        seed=seed,
        noising_strategy=spec.noising_strategy,
        noising_kwargs=_noising_kwargs(spec.noising_strategy, spec.noising_p),
    )
    _masked, unmasked = masker.mask_state(state)
    for predicate in noiser.noise(unmasked):
        flip_fluent_in_state(state, predicate.untyped_representation)


def degrade_leading_states(
    gt_observation: Observation, spec: DegradationSpec, fold: int, problem_name: str,
) -> Observation:
    """Copy of a ground-truth observation with its leading states degraded.

    Args:
        gt_observation: A fully grounded, unmasked ground-truth observation.
        spec: The cell's masking and noising parameters.
        fold: The fold the observation is prepared for.
        problem_name: The observation's problem, e.g. ``problem12``.

    Returns:
        A linked copy in which the states at ``LEADING_STATE_INDICES`` are masked
        and flipped and all later states are still ground truth.
    """
    degraded = copy_observation_linked(gt_observation)
    states = observation_states(degraded)
    for state_index in LEADING_STATE_INDICES:
        if state_index >= len(states):
            break
        seed = leading_state_seed(spec.seed, fold, problem_name, state_index)
        _degrade_state(states[state_index], spec, seed)
    return degraded


def hide_masked_values(masking_line: str) -> str:
    """One ``.masking_info`` line with every entry in its positive form."""
    return _NEGATED_ENTRY.sub(r"\1", masking_line)


def _sorted_state_line(line: str) -> str:
    """A ``(:init ...)`` / ``(:state ...)`` line with its fluents in sorted order."""
    head, _, body = line.partition(" ")
    return f"{head} {' '.join(sorted(_FLUENT.findall(body)))})"


def _sorted_masking_line(line: str) -> str:
    """A ``.masking_info`` line with its entries in sorted order."""
    if not line:
        return line
    return _MASKING_SEPARATOR.join(sorted(line.split(_MASKING_SEPARATOR)))


def _file_lines(path: Path) -> List[str]:
    """A file's lines without the terminator of the last one."""
    text = path.read_text()
    lines = text.split("\n")
    return lines[:-1] if text.endswith("\n") else lines


def _state_line_index(state_index: int) -> int:
    """Line of state ``state_index`` in a ``.trajectory`` file."""
    return 1 + 2 * state_index


def _splice_trajectory(frozen: List[str], fresh: List[str]) -> List[str]:
    """``frozen`` with the leading state lines taken from ``fresh``."""
    if len(frozen) != len(fresh):
        raise ValueError(
            f"frozen and ground-truth trajectories differ in length: "
            f"{len(frozen)} vs {len(fresh)} lines"
        )
    operators = range(2, len(frozen) - 1, 2)
    mismatched = [i for i in operators if frozen[i] != fresh[i]]
    if mismatched:
        raise ValueError(
            f"frozen and ground-truth trajectories execute different actions "
            f"(first difference at line {mismatched[0]})"
        )
    spliced = list(frozen)
    for state_index in LEADING_STATE_INDICES:
        line = _state_line_index(state_index)
        if line < len(frozen) - 1:
            spliced[line] = _sorted_state_line(fresh[line])
    return spliced


def _splice_masking(frozen: List[str], fresh: List[str]) -> List[str]:
    """``frozen`` with the leading state lines taken from ``fresh``."""
    if len(frozen) != len(fresh):
        raise ValueError(
            f"frozen and fresh masking files differ in length: "
            f"{len(frozen)} vs {len(fresh)} states"
        )
    spliced = list(frozen)
    for state_index in LEADING_STATE_INDICES:
        if state_index < len(frozen):
            spliced[state_index] = fresh[state_index]
    return spliced


def stage_noisy_initial_observation(
    frozen_trajectory: Path,
    frozen_masking: Path,
    gt_trajectory: Path,
    problem_pddl: Path,
    domain: Domain,
    spec: DegradationSpec,
    fold: int,
    output_dir: Path,
    hide_masked: bool = True,
) -> Tuple[Path, Path]:
    """Write a frozen observation with its leading states redrawn from ground truth.

    Args:
        frozen_trajectory: The cell's ``original_observation_<problem>.trajectory``.
        frozen_masking: Its ``.masking_info`` sibling.
        gt_trajectory: The problem's ground-truth ``.trajectory``.
        problem_pddl: The problem file declaring the trajectory's objects.
        domain: The parsed (partial) domain.
        spec: The cell's masking and noising parameters.
        fold: The fold the cell belongs to.
        output_dir: Directory the two files are written to, under the frozen names.
        hide_masked: Write every masked entry in its positive form.

    Returns:
        ``(trajectory_path, masking_path)`` of the staged files.

    Raises:
        ValueError: If the frozen and ground-truth trajectories do not describe
            the same action sequence.
    """
    problem_name = problem_pddl.stem
    gt_observation = load_gt_observation(gt_trajectory, domain, problem_pddl)
    degraded = degrade_leading_states(gt_observation, spec, fold, problem_name)

    with tempfile.TemporaryDirectory(prefix="leading_states_") as tmp:
        tmp_dir = Path(tmp)
        fresh_trajectory = observation_to_trajectory_file(
            degraded, tmp_dir / f"{problem_name}.trajectory"
        )
        save_masking_info(
            tmp_dir, problem_name, extract_masking_info_from_observation(degraded)
        )
        trajectory_lines = _splice_trajectory(
            _file_lines(frozen_trajectory), _file_lines(fresh_trajectory)
        )
        masking_lines = _splice_masking(
            _file_lines(frozen_masking),
            _file_lines(tmp_dir / f"{problem_name}.masking_info"),
        )

    if hide_masked:
        masking_lines = [hide_masked_values(line) for line in masking_lines]
    for state_index in LEADING_STATE_INDICES:
        if state_index < len(masking_lines):
            masking_lines[state_index] = _sorted_masking_line(masking_lines[state_index])

    output_dir.mkdir(parents=True, exist_ok=True)
    staged_trajectory = output_dir / frozen_trajectory.name
    staged_masking = output_dir / frozen_masking.name
    staged_trajectory.write_text("\n".join(trajectory_lines))
    staged_masking.write_text("".join(f"{line}\n" for line in masking_lines))
    return staged_trajectory, staged_masking
