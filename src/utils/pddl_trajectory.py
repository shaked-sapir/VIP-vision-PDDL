"""Trajectory file I/O, frame-axiom propagation, and ground-truth injection."""

import json
import math
import os
from pathlib import Path
from typing import List, Set, Union, Tuple, Optional

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser
from pddl_plus_parser.models import Observation, GroundedPredicate, State, Domain

from src.utils.pddl_gym import parse_gym_to_pddl_literal


# ============================================================================
# Trajectory file I/O
# ============================================================================

def _format_literals(literals) -> str:
    return ' '.join(parse_gym_to_pddl_literal(lit) for lit in literals)


def build_trajectory_file(trajectory_data: List[dict], problem_name: str, output_path: Path) -> None:
    from src.utils.pddl_gym import parse_gym_to_pddl_ground_action

    lines = ["(", f"(:init {_format_literals(trajectory_data[0]['current_state']['literals'])})"]

    for i, step in enumerate(trajectory_data):
        if i > 0:
            lines.append(f"(:state {_format_literals(step['current_state']['literals'])})")
        lines.append(f"(operator: {parse_gym_to_pddl_ground_action(step['ground_action'])})")

    lines.append(f"(:state {_format_literals(trajectory_data[-1]['next_state']['literals'])})")
    lines.append(")")

    dest = os.path.join(output_path, f"{problem_name}.trajectory")
    with open(dest, "w") as f:
        f.write('\n'.join(lines))

    print(f"Trajectory saved to {dest}")


def export_gt_trajectory(
    gt_trajectory_json: List[dict],
    problem_name: str,
    gt_out_dir: Path,
) -> Path:
    """Write eval-schema GT ``.trajectory`` + ``_trajectory.json`` for one problem.

    Args:
        gt_trajectory_json: GT trajectory as a list of step dicts already in the
            eval schema (each step has ``current_state.literals``,
            ``ground_action``, ``next_state.literals``).
        problem_name: e.g. ``"problem3"``.
        gt_out_dir: Destination dir (``gt_trajectories/{problem}/``); created if
            missing.

    Returns:
        Path to the written ``gt_trajectories/{problem}/`` directory.
    """
    gt_out_dir = Path(gt_out_dir)
    gt_out_dir.mkdir(parents=True, exist_ok=True)

    json_path = gt_out_dir / f"{problem_name}_trajectory.json"
    with open(json_path, "w") as f:
        json.dump(gt_trajectory_json, f, indent=4)

    build_trajectory_file(gt_trajectory_json, problem_name, gt_out_dir)
    return gt_out_dir


def observation_to_trajectory_file(observation: Observation, output_path: Path) -> Path:
    """Build a .trajectory file from a single-agent Observation object."""
    def serialize_state_positive_only(state: State, state_type: str) -> str:
        positive_predicates = []
        for pred_name, grounded_preds in state.state_predicates.items():
            for grounded_pred in grounded_preds:
                if grounded_pred.is_positive and not grounded_pred.is_masked:
                    positive_predicates.append(grounded_pred.untyped_representation)
        predicates_str = ' '.join(positive_predicates)
        return f"({state_type} {predicates_str})"

    trajectory_lines = ["("]
    init_state = observation.components[0].previous_state
    trajectory_lines.append(serialize_state_positive_only(init_state, ":init"))

    for component in observation.components:
        action_str = str(component.grounded_action_call).strip()
        trajectory_lines.append(f"(operator: {action_str})")
        next_state = component.next_state
        trajectory_lines.append(serialize_state_positive_only(next_state, ":state"))

    trajectory_lines.append(")")

    with open(output_path, "w") as file:
        file.write('\n'.join(trajectory_lines))

    print(f"Trajectory saved to {output_path}")
    return output_path


def extract_actions_from_trajectory_json(images_dir: Path) -> List[str]:
    """Extract ground action strings from a _trajectory.json file."""
    json_paths = list(images_dir.glob("*_trajectory.json"))
    if not json_paths:
        raise FileNotFoundError(f"No _trajectory.json found in {images_dir}")
    with open(json_paths[0]) as f:
        trajectory = json.load(f)
    return [step["ground_action"] for step in trajectory]


def ensure_trajectory_json(images_dir: Path) -> None:
    """If _trajectory.json is missing, convert from .trajectory + problem.pddl."""
    from src.utils.trajectory_json_converter import convert_trajectory_to_json

    if list(images_dir.glob("*_trajectory.json")):
        return
    trajectory_files = list(images_dir.glob("*.trajectory"))
    problem_files = list(images_dir.glob("*.pddl"))
    if trajectory_files and problem_files:
        convert_trajectory_to_json(
            trajectory_path=trajectory_files[0],
            problem_path=problem_files[0],
        )



# ============================================================================
# Frame-axiom propagation helpers
# ============================================================================

def _pddl_objs(pddl_lit: str) -> Set[str]:
    s = pddl_lit.strip()
    if s.startswith("(not"):
        s = s[len("(not"):].strip()
    s = s.strip("() ").split()
    return set(s[1:]) if len(s) > 1 else set()


def _gp_to_gym(gp: GroundedPredicate) -> str:
    if not gp.object_mapping:
        return f"{gp.name}()"
    args = ",".join(f"{v}:{gp.signature[k].name}" for k, v in gp.object_mapping.items())
    return f"{gp.name}({args})"


def _positive_gym_literals(state) -> Set[str]:
    return {
        _gp_to_gym(p)
        for preds in state.state_predicates.values()
        for p in preds
        if p.is_positive
    }


def _positive_unmasked_gym_literals(state) -> Set[str]:
    """Like ``_positive_gym_literals`` but excludes masked predicates.

    In file-based trajectories masked predicates are absent from the file,
    so ``_positive_gym_literals`` implicitly ignores them.  For in-memory
    observations where ``is_masked=True`` predicates are still present in
    the State object, use this variant to get equivalent behaviour.
    """
    return {
        _gp_to_gym(p)
        for preds in state.state_predicates.values()
        for p in preds
        if p.is_positive and not p.is_masked
    }


def _is_frame_literal(gym_lit: str, action_objs: Set[str]) -> bool:
    pddl_pred_str = parse_gym_to_pddl_literal(gym_lit)
    objs = _pddl_objs(pddl_pred_str)
    return bool(objs) and not objs.issubset(action_objs)


def _compute_frame_diff(
    curr: Set[str],
    nxt: Set[str],
    action_objs: Set[str],
    masking: List[Set[GroundedPredicate]],
    state_idx: int,
    consider_masking: bool,
) -> Tuple[Set[str], Set[str]]:
    """Return (to_add, to_remove) frame-closure corrections for a single transition."""
    curr_frame = {p for p in curr if _is_frame_literal(p, action_objs)}
    nxt_frame = {p for p in nxt if _is_frame_literal(p, action_objs)}

    if consider_masking:
        masked: Set[str] = set()
        if state_idx < len(masking):
            masked = {_gp_to_gym(gp) for gp in masking[state_idx]}
        curr_frame -= masked
    return curr_frame - nxt_frame, nxt_frame - curr_frame


def _update_masking(
    masking: List[Set[GroundedPredicate]],
    state_idx: int,
    changed: Set[str],
) -> None:
    """Remove literals that were fixed by frame-closure from the masking set."""
    if state_idx < len(masking) and changed:
        masking[state_idx] = {gp for gp in masking[state_idx] if _gp_to_gym(gp) not in changed}


# ============================================================================
# Frame-axiom propagation — core loop
# ============================================================================

def _propagate_frame_axioms_core(
    obs: Observation,
    masking: List[Set[GroundedPredicate]],
    apply_at: Optional[Set[int]],
    consider_masking: bool,
) -> Tuple[List[dict], List[Set[GroundedPredicate]]]:
    """Shared frame-closure loop.

    Args:
        obs: Parsed observation.
        masking: Per-state masking sets (mutated in place and returned).
        apply_at: Set of transition indices to apply closure at.
                  ``None`` means apply to ALL transitions.
        consider_masking: If True, only propagate unmasked frame literals.

    Returns:
        (out_steps, masking) where out_steps is a list of step dicts suitable
        for ``build_trajectory_file``.
    """
    curr = _positive_gym_literals(obs.components[0].previous_state)
    out_steps: List[dict] = []

    for i, comp in enumerate(obs.components):
        nxt = _positive_gym_literals(comp.next_state)
        to_add: Set[str] = set()
        to_remove: Set[str] = set()

        if apply_at is None or i in apply_at:
            action_objs = _pddl_objs(str(comp.grounded_action_call))
            next_idx = i + 1
            to_add, to_remove = _compute_frame_diff(curr, nxt, action_objs, masking, next_idx, consider_masking)
            nxt = (nxt | to_add) - to_remove
            _update_masking(masking, next_idx, to_add | to_remove)

        out_steps.append({
            "step": i + 1,
            "current_state": {"literals": sorted(curr)},
            "ground_action": str(comp.grounded_action_call).strip("()"),
            "next_state": {"literals": sorted(nxt)},
            "frame_closure": {"added": sorted(to_add), "removed": sorted(to_remove)},
        })
        curr = nxt

    return out_steps, masking


# ============================================================================
# Frame-axiom propagation — shared I/O helpers
# ============================================================================

def _load_trajectory_context(
    trajectory_path: Path,
    masking_info_path: Path,
    domain_path: Path,
) -> Tuple[Observation, List[Set[GroundedPredicate]]]:
    from src.utils.masking import load_masking_info

    domain: Domain = DomainParser(domain_path).parse_domain()
    obs: Observation = TrajectoryParser(domain).parse_trajectory(trajectory_path)
    masking = [set(m) for m in load_masking_info(masking_info_path, domain)]
    return obs, masking


def _save_trajectory_and_return(
    out_steps: List[dict],
    masking: List[Set[GroundedPredicate]],
    out_name: str,
    traj_dir: Path,
    masking_dir: Path,
) -> Tuple[Path, Path]:
    from src.utils.masking import save_masking_info

    build_trajectory_file(out_steps, out_name, traj_dir)
    save_masking_info(masking_dir, out_name, masking)
    return (traj_dir / f"{out_name}.trajectory", masking_dir / f"{out_name}.masking_info")


# ============================================================================
# Frame-axiom propagation — public API
# ============================================================================

def propagate_frame_axioms_in_trajectory(
    trajectory_path: Union[str, Path],
    masking_info_path: Union[str, Path],
    domain_path: Union[str, Path],
    mode: str = "consider_masking",
) -> Tuple[Path, Path]:
    """Frame-closure propagation applied to every transition.

    Args:
        mode: "ignore_masking" or "consider_masking"
    """
    if mode not in ["ignore_masking", "consider_masking"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'ignore_masking' or 'consider_masking'")

    trajectory_path, masking_info_path, domain_path = Path(trajectory_path), Path(masking_info_path), Path(domain_path)
    obs, masking = _load_trajectory_context(trajectory_path, masking_info_path, domain_path)

    out_steps, masking = _propagate_frame_axioms_core(
        obs, masking, apply_at=None, consider_masking=(mode == "consider_masking"),
    )

    out_name = trajectory_path.stem + "_frame_closed"
    return _save_trajectory_and_return(out_steps, masking, out_name, trajectory_path.parent, masking_info_path.parent)


def propagate_frame_axioms_in_memory(
    observation: Observation,
    masking_info: List[Set[GroundedPredicate]],
    gt_state_indices: Set[int],
) -> int:
    """Apply frame-axiom propagation in-place on an in-memory observation.

    Mirrors the logic of ``propagate_frame_axioms_selective`` but operates
    directly on Observation and masking_info objects instead of files.
    Only transitions whose source state index is in *gt_state_indices* are
    processed (matching the ``after_gt_only`` mode).

    Args:
        observation: The observation to modify in-place.
        masking_info: Per-state masking sets (mutated in-place).
        gt_state_indices: Set of state indices considered ground truth.
            Propagation is applied at transitions whose **source** state
            index is in this set (e.g., ``{0}`` propagates only at
            transition 0 → 1).

    Returns:
        The number of fluents corrected by frame-axiom propagation.
    """
    from src.utils.pddl_state import flip_fluent_in_state

    total_corrections = 0
    # Init state is GT and unmasked — use full positive literals.
    curr = _positive_gym_literals(observation.components[0].previous_state)

    for i, comp in enumerate(observation.components):
        # Use unmasked variant: in-memory states still contain masked
        # predicates (with is_masked=True), but the file-based pipeline
        # implicitly drops them during serialization/re-parse.  Using
        # the unmasked variant keeps the two pipelines equivalent.
        nxt = _positive_unmasked_gym_literals(comp.next_state)

        if i in gt_state_indices:
            action_objs = _pddl_objs(str(comp.grounded_action_call))
            next_idx = i + 1
            to_add, to_remove = _compute_frame_diff(
                curr, nxt, action_objs, masking_info, next_idx, consider_masking=True,
            )

            # Apply corrections in-place on the actual State object
            for gym_lit in to_add | to_remove:
                pddl_lit = parse_gym_to_pddl_literal(gym_lit)
                flip_fluent_in_state(comp.next_state, pddl_lit)

            _update_masking(masking_info, next_idx, to_add | to_remove)
            total_corrections += len(to_add) + len(to_remove)

            # Update nxt for the next iteration's curr
            nxt = _positive_unmasked_gym_literals(comp.next_state)

        curr = nxt

    return total_corrections


def propagate_frame_axioms_selective(
    trajectory_path: Union[str, Path],
    masking_info_path: Union[str, Path],
    domain_path: Union[str, Path],
    gt_state_indices: Set[int],
    mode: str = "after_gt_only",
) -> Tuple[Path, Path]:
    """Frame-closure propagation only at transitions whose source state is GT.

    Args:
        gt_state_indices: Set of state indices that are ground truth.
        mode: "after_gt_only" or "all_states"
    """
    if mode not in ["after_gt_only", "all_states"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'after_gt_only' or 'all_states'")

    if mode == "all_states":
        return propagate_frame_axioms_in_trajectory(
            trajectory_path, masking_info_path, domain_path, mode="consider_masking"
        )

    trajectory_path, masking_info_path, domain_path = Path(trajectory_path), Path(masking_info_path), Path(domain_path)
    obs, masking = _load_trajectory_context(trajectory_path, masking_info_path, domain_path)

    out_steps, masking = _propagate_frame_axioms_core(
        obs, masking, apply_at=gt_state_indices, consider_masking=True,
    )

    problem_name = trajectory_path.stem.split('_frame_axioms')[0]
    out_name = f"{problem_name}_frame_axioms"
    return _save_trajectory_and_return(out_steps, masking, out_name, trajectory_path.parent, masking_info_path.parent)


# ============================================================================
# Ground-truth state injection
# ============================================================================

def _compute_gt_indices(num_states: int, gt_rate: int, anchor_final: bool = False) -> Set[int]:
    """Pick evenly-spaced state indices to replace with ground truth.

    Args:
        num_states: Total number of states in the trajectory (steps + 1).
        gt_rate: Percentage of states to inject as GT (0 = init only).
        anchor_final: When True, always include the final state (``num_states - 1``)
            as GT — used by the ``cdps_anchored`` variant to anchor both endpoints.
    """
    indices = {0}
    if gt_rate > 0:
        num_gt = max(1, math.ceil(num_states * gt_rate / 100.0))
        if num_gt > 1:
            interval = num_states / num_gt
            indices |= {int(i * interval) for i in range(num_gt) if int(i * interval) < num_states}
    if anchor_final and num_states > 0:
        indices.add(num_states - 1)
    return indices


def _gt_state_literals(gt_trajectory: List[dict], num_steps: int) -> List[Set[str]]:
    """Per-state positive-literal sets (indices 0..num_steps) from clean GT JSON."""
    states = [set(gt_trajectory[0]['current_state']['literals'])]
    for i in range(num_steps):
        states.append(set(gt_trajectory[i]['next_state']['literals']))
    return states


def _degraded_state_literals(obs: Observation, num_steps: int) -> List[Set[str]]:
    """Per-state positive-literal sets (indices 0..num_steps) from the degraded obs.

    Masked predicates are absent from file-based trajectories, so
    ``_positive_gym_literals`` already yields only the noisy *unmasked* positives.
    """
    states = [_positive_gym_literals(obs.components[0].previous_state)]
    for i in range(num_steps):
        states.append(_positive_gym_literals(obs.components[i].next_state))
    return states


def inject_gt_states_by_percentage(
    trajectory_path: Union[str, Path],
    masking_info_path: Union[str, Path],
    json_trajectory_path: Union[str, Path],
    domain_path: Union[str, Path],
    gt_rate: int,
    anchor_final: bool = False,
) -> Tuple[Path, Path, Set[int]]:
    """Inject ground truth states at percentage-based intervals throughout the trajectory.

    Selective injection (single code path for plain AND anchored CDPS): keep
    the degraded (noisy) states everywhere and inject clean GT literals ONLY at
    ``gt_state_indices`` (init + any gt_rate picks; + final when
    ``anchor_final``), unmasking exactly those states. This preserves
    noise/partial observability in the interior so CDPS still sees conflicts.

    NOTE: an earlier version rewrote EVERY state from the JSON in the plain
    path. That was a silent no-op while the in-dir JSON equaled the degraded
    trajectory, but after the gt_builder split repointed injection at the
    CLEAN gt_trajectories JSON it destroyed all interior noise for gt_rate>0.
    Injection is now selective for both variants; ``anchor_final`` only adds
    the final state index and the ``_anchored`` filename suffix (so anchored
    files never collide with the plain-CDPS files).
    """
    trajectory_path, masking_info_path = Path(trajectory_path), Path(masking_info_path)
    json_trajectory_path, domain_path = Path(json_trajectory_path), Path(domain_path)

    obs, masking = _load_trajectory_context(trajectory_path, masking_info_path, domain_path)
    num_steps = len(obs.components)

    with open(json_trajectory_path, 'r') as f:
        gt_trajectory = json.load(f)

    gt_state_indices = _compute_gt_indices(num_steps + 1, gt_rate, anchor_final=anchor_final)
    n = min(num_steps, len(gt_trajectory))

    gt_states = _gt_state_literals(gt_trajectory, n)
    degraded_states = _degraded_state_literals(obs, n)
    state_literals = [
        gt_states[idx] if idx in gt_state_indices else degraded_states[idx]
        for idx in range(n + 1)
    ]
    if 0 in gt_state_indices and len(masking) > 0:
        masking[0] = set()

    new_trajectory_data = []
    for i in range(n):
        state_idx = i + 1
        if state_idx in gt_state_indices and state_idx < len(masking):
            masking[state_idx] = set()
        new_trajectory_data.append({
            'step': state_idx,
            'current_state': {'literals': sorted(state_literals[i])},
            'ground_action': gt_trajectory[i]['ground_action'],
            'next_state': {'literals': sorted(state_literals[i + 1])},
        })

    problem_name = trajectory_path.stem.split('_gtrate')[0]
    out_name = f"{problem_name}_gtrate{gt_rate}" if gt_rate > 0 else problem_name
    if anchor_final:
        out_name = f"{out_name}_anchored"
    traj_path, mask_path = _save_trajectory_and_return(new_trajectory_data, masking, out_name, trajectory_path.parent, masking_info_path.parent)
    return traj_path, mask_path, gt_state_indices
