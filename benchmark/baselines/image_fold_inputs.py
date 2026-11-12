"""Resolve one fold's prepared trajectories into inputs for an imaged ROSAME arm.

Every image-mode ROSAME arm needs the same four things per trajectory — the
parsed problem, its ``state_*.png`` frames, the observed action sequence, and
the ground-truth endpoint states — and none of them may come from the degraded
trajectory's own states. This module is that walk, shared by the ICAPS-24 arms
(``rosame_i_24``, ``rosame_i_milp_24``) and the ICAPS-26 ones.

The entry point is :func:`resolve_fold_inputs`; the helpers below it are
exported because the arms and their tests reach for them individually.

Both GT anchors are read from a single parse of
``<data_dir>/gt_trajectories/<problem>/<problem>.trajectory``. The 24 arms use
only :attr:`ResolvedTrace.gt_final_predicates`; the 26 arms also need
:attr:`ResolvedTrace.gt_init_predicates`, which the vendored network takes as a
hard anchor (``docs/rosame-i-milp-26-implementation-plan.md`` §4.3).

On request the walk also returns :attr:`ResolvedTrace.observed_final`: the last
state of the prepared trajectory itself, with its masked atoms as unknown. It is
the final-state anchor of a 24 arm in a setup that does not give the final state.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from pddl_plus_parser.lisp_parsers import ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Domain, GroundedPredicate, Observation, Problem, State

from benchmark.experiment_running_helpers.normalize import _normalize_hyphens
from src.utils.corpus_layout import FRAME_GLOB, IMAGE_POOL_DIR_NAMES, frames_dir
from src.utils.masking import load_masking_info

#: PDDL domain name / path part -> bench key, for domains whose names differ.
DOMAIN_ALIASES: Dict[str, str] = {
    "blocks": "blocksworld",
    "blocksworld": "blocksworld",
    "hanoi": "hanoi",
    "n_puzzle": "npuzzle",
    "n_puzzle_typed": "npuzzle",
    "npuzzle": "npuzzle",
    "gripper": "gripper",
    "depot": "depot",
}

#: The bench keys :data:`DOMAIN_ALIASES` can resolve to.
BENCH_KEYS = frozenset(DOMAIN_ALIASES.values())


@dataclass(frozen=True)
class StateLabelling:
    """A degraded state as an arm may read it.

    Attributes:
        positive: The atoms labelled true, as ``"on a b"`` strings.
        unknown: The masked atoms; every atom in neither list is labelled false.
    """

    positive: List[str]
    unknown: List[str]


@dataclass(frozen=True)
class ResolvedTrace:
    """One trajectory's inputs to an imaged ROSAME arm.

    ``gt_init_predicates`` / ``gt_final_predicates`` are untyped predicate
    strings without parentheses, e.g. ``"on a b"``. ``observed_final`` is the
    prepared trajectory's own last state, present only when asked for.
    """

    problem: Problem
    problem_name: str
    problem_dir: Path
    image_paths: List[Path]
    action_strings: List[str]
    gt_init_predicates: List[str]
    gt_final_predicates: List[str]
    gt_trajectory_path: Path
    observed_final: Optional[StateLabelling] = None


def resolve_fold_inputs(
    partial_domain: Domain,
    prepared_trajectories: Sequence[Tuple[Path, Path, Path]],
    bench: str,
    with_observed_final: bool = False,
) -> List[ResolvedTrace]:
    """Resolve every trajectory in a fold that has images on disk.

    Trajectories with no frames are dropped; an empty result means a
    simulation-mode cell, which the caller is expected to report and skip.

    Args:
        with_observed_final: Also read each prepared trajectory's last state
            and its masking info into :attr:`ResolvedTrace.observed_final`.

    Raises:
        FileNotFoundError: if a kept trajectory has no GT counterpart.
        ValueError: if a GT trajectory parses to zero transitions, or the
            masking info asked for does not cover the trajectory's states.
    """
    resolved: List[ResolvedTrace] = []

    for traj_path, masking_path, problem_pddl_path, *_ in prepared_trajectories:
        problem_dir = problem_pddl_path.parent
        problem_name = problem_pddl_path.stem

        image_paths = resolve_images(problem_dir)
        if not image_paths:
            continue

        problem = parse_problem_normalized(partial_domain, problem_pddl_path)

        # The states of this file are degraded; only ``observed_final`` reads one.
        observation = parse_trajectory_normalized(partial_domain, problem, traj_path)
        action_strings = [
            str(component.grounded_action_call)[1:-1]
            for component in observation.components
        ]

        gt_path = resolve_gt_trajectory_path(problem_dir, problem_name)
        gt_init, gt_final = resolve_gt_anchors(
            partial_domain, problem, gt_path, problem_name
        )
        resolved.append(
            ResolvedTrace(
                problem=problem,
                problem_name=problem_name,
                problem_dir=problem_dir,
                image_paths=image_paths,
                action_strings=action_strings,
                gt_init_predicates=gt_init,
                gt_final_predicates=gt_final,
                gt_trajectory_path=gt_path,
                observed_final=(
                    observed_final_state(partial_domain, observation, masking_path)
                    if with_observed_final else None
                ),
            )
        )

    return resolved


def observed_final_state(
    partial_domain: Domain, observation: Observation, masking_path: Optional[Path]
) -> StateLabelling:
    """The last state of a prepared trajectory, its masked atoms as unknown.

    Args:
        observation: The parsed prepared trajectory.
        masking_path: Its ``.masking_info`` file; ``None`` or absent means no
            atom is masked.

    Raises:
        ValueError: if the trajectory holds no transition, or the masking info
            does not have one line per state.
    """
    if not observation.components:
        raise ValueError("a trajectory with no transition has no final state to read")
    masked = last_state_masked_atoms(
        partial_domain, masking_path, num_states=len(observation.components) + 1
    )
    return labelling_of(observation.components[-1].next_state, masked)


def labelling_of(state: State, masked_atoms: Set[str]) -> StateLabelling:
    """``state`` as a labelling: its unmasked positive atoms, and the masked ones."""
    positive = sorted(
        atom
        for grounded_set in state.state_predicates.values()
        for pred in grounded_set
        if pred.is_positive and (atom := atom_string(pred)) not in masked_atoms
    )
    return StateLabelling(positive=positive, unknown=sorted(masked_atoms))


def last_state_masked_atoms(
    partial_domain: Domain, masking_path: Optional[Path], num_states: int
) -> Set[str]:
    """The atoms a ``.masking_info`` file masks in the last of ``num_states`` states.

    Raises:
        ValueError: if the file does not have one line per state.
    """
    if masking_path is None or not Path(masking_path).exists():
        return set()
    tmp_path = conformed_tempfile(Path(masking_path), ".masking_info", partial_domain)
    try:
        masking_info = load_masking_info(tmp_path, partial_domain)
    finally:
        tmp_path.unlink(missing_ok=True)
    if len(masking_info) != num_states:
        raise ValueError(
            f"masking info '{masking_path}' covers {len(masking_info)} states, "
            f"the trajectory has {num_states}"
        )
    return {atom_string(pred) for pred in masking_info[-1]}


def atom_string(pred: GroundedPredicate) -> str:
    """A grounded predicate as ``"name arg1 arg2"``, whatever its polarity."""
    return " ".join([pred.name, *pred.object_mapping.values()])


def resolve_images(problem_dir: Path) -> List[Path]:
    """Numerically-sorted ``state_*.png`` frames for one image problem folder.

    Frames live in the image pool; a folder under the normalized pool reads its
    twin's (``corpus_layout.frames_dir``).
    """
    images = list(frames_dir(problem_dir).glob(FRAME_GLOB))
    return sorted(images, key=lambda p: int(re.search(r"\d+", p.stem).group()))


def resolve_gt_trajectory_path(problem_dir: Path, problem_name: str) -> Path:
    """Path of the GT trajectory both state anchors are read from.

    The anchors must come from
    ``<data_dir>/gt_trajectories/<problem>/<problem>.trajectory``; the degraded
    trajectory next to the problem PDDL is never substituted.

    Raises:
        FileNotFoundError: if ``problem_dir`` is not in the
            ``<data_dir>/training/<image pool>/<problem>/`` layout, or the GT
            trajectory is absent.
    """
    if problem_dir.parent.name not in IMAGE_POOL_DIR_NAMES:
        raise FileNotFoundError(
            f"cannot locate gt_trajectories/ for '{problem_name}': expected "
            f"'{problem_dir}' to sit under one of {sorted(IMAGE_POOL_DIR_NAMES)}"
        )
    data_dir = problem_dir.parents[2]
    gt_path = data_dir / "gt_trajectories" / problem_name / f"{problem_name}.trajectory"
    if not gt_path.exists():
        raise FileNotFoundError(
            f"no GT trajectory for '{problem_name}': expected '{gt_path}'. "
            f"ROSAME-I anchors its last predicted state on the GT final state, "
            f"so run benchmark/generate_gt_trajectories.py for '{data_dir}' "
            f"before using an imaged ROSAME-I arm on it."
        )
    return gt_path


def resolve_gt_anchors(
    partial_domain: Domain,
    problem: Problem,
    gt_trajectory_path: Path,
    problem_name: str,
) -> Tuple[List[str], List[str]]:
    """``(init, final)`` GT positive predicate strings from one GT trajectory.

    Raises:
        ValueError: if the GT trajectory holds no transition, so it carries
            neither endpoint.
    """
    observation = parse_trajectory_normalized(
        partial_domain, problem, gt_trajectory_path
    )
    return gt_anchors_from_observation(observation, gt_trajectory_path, problem_name)


def gt_anchors_from_observation(
    observation: Observation, source: Path, problem_name: str
) -> Tuple[List[str], List[str]]:
    """``(init, final)`` positive predicate strings of an already-parsed GT trace."""
    if not observation.components:
        raise ValueError(
            f"GT trajectory '{source}' for '{problem_name}' parsed to "
            f"zero transitions, so it carries no final state to anchor on"
        )
    return (
        state_positive_predicates(observation.components[0].previous_state),
        state_positive_predicates(observation.components[-1].next_state),
    )


# Problem PDDLs and gt_trajectories on disk keep whichever dialect they were
# written in, which need not match the reference domain: an image experiment
# that ran normalize_experiment_data has an underscored domain, one that skipped
# it keeps the raw domain. Parsing an input against a domain in the other
# dialect raises ``illegal state component``, so inputs are conformed to the
# dialect the domain itself declares.


def domain_uses_hyphens(partial_domain: Domain) -> bool:
    """True if any identifier the domain declares contains a hyphen."""
    declared = (
        list(partial_domain.predicates)
        + list(partial_domain.actions)
        + list(getattr(partial_domain, "types", None) or {})
        + list(getattr(partial_domain, "constants", None) or {})
    )
    return any("-" in name for name in declared)


def conformed_tempfile(source: Path, suffix: str, partial_domain: Domain) -> Path:
    """Write a copy of ``source`` in ``partial_domain``'s dialect to a temp file.

    Hyphens are rewritten to underscores unless the domain declares hyphenated
    identifiers, in which case ``source`` is copied verbatim.
    """
    text = source.read_text()
    if not domain_uses_hyphens(partial_domain):
        text = _normalize_hyphens(text)
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as tmp:
        tmp.write(text)
        return Path(tmp.name)


def parse_problem_normalized(partial_domain: Domain, problem_pddl_path: Path) -> Problem:
    """Parse a problem PDDL, conforming its dialect to the domain's."""
    tmp_path = conformed_tempfile(problem_pddl_path, ".pddl", partial_domain)
    try:
        return ProblemParser(tmp_path, partial_domain).parse_problem()
    finally:
        tmp_path.unlink(missing_ok=True)


def parse_trajectory_normalized(
    partial_domain: Domain, problem: Problem, trajectory_path: Path
) -> Observation:
    """Parse a trajectory, conforming its dialect to the domain's."""
    tmp_path = conformed_tempfile(trajectory_path, ".trajectory", partial_domain)
    try:
        return TrajectoryParser(partial_domain, problem).parse_trajectory(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def state_positive_predicates(state: State) -> List[str]:
    """Untyped predicate strings (no parens) of a state's positive fluents."""
    predicates: List[str] = []
    for grounded_set in state.state_predicates.values():
        for pred in grounded_set:
            predicates.append(pred.untyped_representation[1:-1])
    return predicates


def infer_bench_key(domain_path: Path, partial_domain: Domain) -> str:
    """Map a PDDL domain name / experiment path to a bench key."""
    domain_name = (partial_domain.name or "").lower()
    if domain_name in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[domain_name]
    for part in Path(domain_path).resolve().parts:
        key = part.lower()
        if key in DOMAIN_ALIASES:
            return DOMAIN_ALIASES[key]
    return domain_name
