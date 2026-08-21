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
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from pddl_plus_parser.lisp_parsers import ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Domain, Observation, Problem, State

from benchmark.experiment_running_helpers.normalize import _normalize_hyphens

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

#: Staging directory names that sit one level above a problem's trajectory dir.
TRAJECTORY_DIR_NAMES = frozenset({"trajectories", "trajectories_normalized"})


@dataclass(frozen=True)
class ResolvedTrace:
    """One trajectory's inputs to an imaged ROSAME arm.

    ``gt_init_predicates`` / ``gt_final_predicates`` are untyped predicate
    strings without parentheses, e.g. ``"on a b"``.
    """

    problem: Problem
    problem_name: str
    image_paths: List[Path]
    action_strings: List[str]
    gt_init_predicates: List[str]
    gt_final_predicates: List[str]
    gt_trajectory_path: Path


def resolve_fold_inputs(
    partial_domain: Domain,
    prepared_trajectories: Sequence[Tuple[Path, Path, Path]],
    bench: str,
) -> List[ResolvedTrace]:
    """Resolve every trajectory in a fold that has images on disk.

    Trajectories with no frames are dropped; an empty result means a
    simulation-mode cell, which the caller is expected to report and skip.

    Raises:
        FileNotFoundError: if a kept trajectory has no GT counterpart.
        ValueError: if a GT trajectory parses to zero transitions.
    """
    resolved: List[ResolvedTrace] = []

    for traj_path, _masking_path, problem_pddl_path, *_ in prepared_trajectories:
        problem_dir = problem_pddl_path.parent
        problem_name = problem_pddl_path.stem

        image_paths = resolve_images(problem_dir, bench, problem_name)
        if not image_paths:
            continue

        problem = parse_problem_normalized(partial_domain, problem_pddl_path)

        # Actions only — the states in this file are degraded and never read.
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
                image_paths=image_paths,
                action_strings=action_strings,
                gt_init_predicates=gt_init,
                gt_final_predicates=gt_final,
                gt_trajectory_path=gt_path,
            )
        )

    return resolved


def resolve_images(problem_dir: Path, bench: str, problem_name: str) -> List[Path]:
    """Numerically-sorted ``state_*.png`` frames for one trajectory.

    Looks next to the problem PDDL first (PDDLGym-generated domains keep frames
    in the data dir), then falls back to the external-image domain layout
    ``src/domains/<bench>/problems/<problem>/`` (e.g. depot, gripper), whose
    frames never get copied into ``benchmark/data``.
    """
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        problem_dir,
        repo_root / "src" / "domains" / bench / "problems" / problem_name,
    ]
    for cand in candidates:
        images = list(cand.glob("state_*.png"))
        if images:
            return sorted(images, key=lambda p: int(re.search(r"\d+", p.stem).group()))
    return []


def resolve_gt_trajectory_path(problem_dir: Path, problem_name: str) -> Path:
    """Path of the GT trajectory both state anchors are read from.

    The anchors must come from
    ``<data_dir>/gt_trajectories/<problem>/<problem>.trajectory``; the degraded
    trajectory next to the problem PDDL is never substituted.

    Raises:
        FileNotFoundError: if ``problem_dir`` is not in the
            ``<data_dir>/training/<staging_dir>/<problem>/`` layout, or the GT
            trajectory is absent.
    """
    if problem_dir.parent.name not in TRAJECTORY_DIR_NAMES:
        raise FileNotFoundError(
            f"cannot locate gt_trajectories/ for '{problem_name}': expected "
            f"'{problem_dir}' to sit under one of {sorted(TRAJECTORY_DIR_NAMES)}"
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
