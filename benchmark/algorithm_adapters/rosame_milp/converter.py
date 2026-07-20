"""Converters: our pddl_plus_parser world -> vendored planning_structs inputs.

Parameter-order convention: the planning_structs Domain is built with predicate
and action-schema parameter types in **PDDL signature order**. Consequently the
lifted binding tuples ``x`` map predicate positions to *PDDL* action-parameter
positions (1-based), grounded Actions carry their args in PDDL order (exactly
as they appear in trajectories), and no ROSAME-style type-grouped
canonicalization is needed anywhere in the encoder.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from planning_structs.domain import Domain as PSDomain
from planning_structs.instance import Instance as PSInstance
from planning_structs.traces import ObservationP, ObservationT

_EPS = 1e-5

_STATE_BLOCK_RE = re.compile(r"\((?::init|:state)((?:\s*\([^()]*\))*)\s*\)")
_FLUENT_RE = re.compile(r"\(([^()]*)\)")


def build_ps_domain(pddl_domain) -> PSDomain:
    """Vendored Domain from a pddl_plus_parser Domain (signature order preserved)."""
    types_spec = []
    for name, ptype in pddl_domain.types.items():
        parent = getattr(ptype, "parent", None)
        types_spec.append((name, parent.name if parent is not None else None))

    predicates_spec = [
        (name, [str(t) for t in pred.signature.values()])
        for name, pred in pddl_domain.predicates.items()
    ]
    schemas_spec = [
        (name, [str(t) for t in action.signature.values()])
        for name, action in pddl_domain.actions.items()
    ]
    return PSDomain(types_spec, predicates_spec, schemas_spec, name=pddl_domain.name)


def build_ps_instance(ps_domain: PSDomain, pddl_domain, problem) -> PSInstance:
    """Vendored Instance grounded on a problem's objects (+ domain constants)."""
    objects: List[Tuple[str, str]] = [
        (name, str(obj.type)) for name, obj in problem.objects.items()
    ]
    for name, const in getattr(pddl_domain, "constants", {}).items():
        if name not in problem.objects:
            objects.append((name, str(const.type)))
    return PSInstance(ps_domain, objects)


def _proposition_of(instance: PSInstance, name: str, arg_names: Sequence[str]):
    pred = instance.domain.get_predicate(name)
    if pred is None:
        return None
    args = tuple(instance.get_object(a) for a in arg_names)
    if any(a is None for a in args):
        return None
    return instance.get_proposition(pred, args)


def _action_of(instance: PSInstance, name: str, arg_names: Sequence[str]):
    schema = instance.domain.get_action_schema(name)
    if schema is None:
        return None
    args = tuple(instance.get_object(a) for a in arg_names)
    if any(a is None for a in args):
        return None
    return instance.get_action(schema, args)


def _state_prob(gp) -> float:
    """Ternary encoding of one grounded predicate: masked=0.5, else 1-eps/eps."""
    if getattr(gp, "is_masked", False):
        return 0.5
    return 1.0 - _EPS if gp.is_positive else _EPS


def observation_to_trace(
    instance: PSInstance,
    observation,
    goal_fluents: Optional[Set[Tuple[str, Tuple[str, ...]]]] = None,
) -> Optional[ObservationT]:
    """One (grounded, masked) pddl_plus Observation -> vendored ObservationT.

    The returned ObservationT carries two extra attributes used by our encoder:
    ``instance`` (the trace's own grounding) and ``actions`` (t -> observed
    grounded Action).

    Args:
        goal_fluents: positive fluents of the (GT) final state as
            ``(pred_name, (obj, ...))`` tuples; ``None`` leaves the final state
            soft (no hard goal constraints — encoder skips paper eqs. 21–22).
    """
    components = observation.components
    if not components:
        return None
    states = [components[0].previous_state] + [c.next_state for c in components]
    step = len(components)

    # obs_p: ternary probabilities for every grounded predicate of every state
    obs_p: Dict[int, List[ObservationP]] = {}
    for t, state in enumerate(states, start=1):
        entries: List[ObservationP] = []
        for preds in state.state_predicates.values():
            for gp in preds:
                prop = _proposition_of(
                    instance, gp.name, [gp.object_mapping[k] for k in gp.signature.keys()]
                )
                if prop is not None:
                    entries.append(ObservationP(prop, _state_prob(gp)))
        obs_p[t] = entries

    # init: positive unmasked fluents of state 1 (GT by assumption)
    init = [op.proposition for op in obs_p[1] if op.prob > 0.5]

    # goal: GT final state when provided, else None (soft)
    goal = None
    if goal_fluents is not None:
        goal = []
        for name, arg_names in goal_fluents:
            prop = _proposition_of(instance, name, arg_names)
            if prop is not None:
                goal.append(prop)

    # observed actions per step
    actions = {}
    for t, comp in enumerate(components, start=1):
        call = comp.grounded_action_call
        action = _action_of(instance, call.name, list(call.parameters))
        if action is None:
            print(f"  [MILP] Warning: unmatched observed action {call.name} "
                  f"{call.parameters} — trace skipped")
            return None
        actions[t] = action

    trace = ObservationT(step, init, obs_p, goal, obs_a={})
    trace.instance = instance
    trace.actions = actions
    return trace


def gt_final_state_fluents(gt_trajectory_path: Path) -> Optional[Set[Tuple[str, Tuple[str, ...]]]]:
    """Positive fluents of the last state in a GT ``.trajectory`` file."""
    if not gt_trajectory_path.exists():
        return None
    blocks = _STATE_BLOCK_RE.findall(gt_trajectory_path.read_text())
    if not blocks:
        return None
    fluents: Set[Tuple[str, Tuple[str, ...]]] = set()
    for raw in _FLUENT_RE.findall(blocks[-1]):
        parts = raw.split()
        if parts:
            fluents.add((parts[0], tuple(parts[1:])))
    return fluents


def find_gt_trajectory(problem_pddl_path: Path) -> Optional[Path]:
    """Locate a problem's GT trajectory relative to the standard data_dir layout.

    Standard layout: ``<data_dir>/training/trajectories/<prob>/<prob>.pddl`` and
    ``<data_dir>/gt_trajectories/<prob>/<prob>.trajectory``.
    """
    problem = problem_pddl_path.stem
    parents = list(problem_pddl_path.parents)
    candidates = []
    for base in parents[:4]:
        candidates.append(base / "gt_trajectories" / problem / f"{problem}.trajectory")
        candidates.append(base / "gt_trajectories" / f"{problem}.trajectory")
    for c in candidates:
        if c.exists():
            return c
    return None
