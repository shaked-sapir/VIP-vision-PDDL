"""Converters: our pddl_plus_parser world -> vendored planning_structs inputs.

Parameter-order convention: the planning_structs Domain is built with predicate
and action-schema parameter types in **PDDL signature order**. Consequently the
lifted binding tuples ``x`` map predicate positions to *PDDL* action-parameter
positions (1-based), grounded Actions carry their args in PDDL order (exactly
as they appear in trajectories), and no ROSAME-style type-grouped
canonicalization is needed anywhere in the encoder.

Grounding width: see :class:`RepeatedArgsInstance` and the
``include_repeated_args`` flag on the instance builders.
"""

from __future__ import annotations

import re
from collections import Counter
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from planning_structs.domain import Domain as PSDomain
from planning_structs.instance import Action as PSAction
from planning_structs.instance import Instance as PSInstance
from planning_structs.instance import Proposition as PSProposition
from planning_structs.traces import ObservationP, ObservationT

_EPS = 1e-5


class GtAnchoring(Enum):
    """Which known-ground-truth states the MILP hard-fixes.

    CDPS receives its ground truth as ``gt_states_by_obs`` (obs_idx -> 0-based
    STATE indices that were injected clean and left unmasked). The MILP can use
    the same information as *hard* constraints — a hard-fixed state is
    unrepairable, which shrinks the search space and rules out repairs that
    contradict known truth.

    - ``INIT_ONLY``: only the initial state (GT by assumption everywhere in this
      project). Matches plain ``cdps``.
    - ``ALL_GT_STATES``: every state in ``gt_state_indices``. Matches
      ``cdps_anchored``, whose data prep injects the final state as GT too.
    """

    INIT_ONLY = "init_only"
    ALL_GT_STATES = "all_gt_states"


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


class RepeatedArgsInstance(PSInstance):
    """Vendored Instance widened to ground *repeated*-object argument tuples.

    Upstream grounds with ``itertools.permutations`` (instance.py:96,104), so an
    n-ary predicate or schema is only grounded on tuples of **distinct** objects:
    there is no ``(on a a)`` proposition and no ``stack(a,a)`` action. That is a
    safe assumption for ROSAME, whose network shares the same vocabulary, but it
    breaks two ways in our pipeline:

    1. Our observations are completed with ``get_all_possible_groundings``, which
       uses ``itertools.product`` — so states genuinely contain ``(on a a)``. With
       upstream grounding those fluents get no ``hol`` variable, pass through the
       repair untouched, and reach PI-SAM entirely unconstrained by the MILP.
       That is exactly the hole design §4.1 forbids: a "feasible" T' on which
       PI-SAM may still raise a conflict.
    2. A trajectory step whose action has repeated objects (``stack(a,a)``) has no
       grounded Action to match, so ``observation_to_trace`` drops the whole trace.

    The *lifted* vocabulary needs no widening: PI-SAM's matcher enumerates
    distinct action-parameter slots (``create_signature_permutations``), the same
    as ``Domain.build_predicate_arguments``. When the action itself repeats an
    object, PI-SAM matches ``(on a a)`` against ``stack(a,a)`` to *both*
    ``(on ?x ?y)`` and ``(on ?y ?x)``; the encoder reproduces that by OR-ing all
    unifying bindings (``encoder.CPSATObservedActions._bindings`` + ``cp.any``).

    Cost is small — the tuple count grows from n!/(n-k)! to n^k (~8% on a 10-block
    blocksworld instance).
    """

    def _build_actions(self) -> List[PSAction]:
        return [
            PSAction(schema, tup)
            for schema in self.domain.action_schemas
            for tup in product(self.objects, repeat=schema.arity)
            if self._type_match(tup, schema)
        ]

    def _build_propositions(self) -> List[PSProposition]:
        return [
            PSProposition(predicate, tup)
            for predicate in self.domain.predicates
            for tup in product(self.objects, repeat=predicate.arity)
            if self._type_match(tup, predicate)
        ]


def build_ps_instance_from_objects(
    ps_domain: PSDomain,
    pddl_domain,
    pddl_objects: Mapping[str, Any],
    include_repeated_args: bool = False,
) -> PSInstance:
    """Vendored Instance grounded on a name -> PDDLObject map (+ domain constants).

    Works from any object map: a parsed problem's ``objects`` or an
    ``Observation.grounded_objects`` (which the trajectory parser fills from the
    problem), so the MILP can be built without re-parsing the problem file.

    Args:
        include_repeated_args: ground repeated-object tuples too. ``True`` for
            the ``cdps_milp_*`` learners, whose observations contain them and
            whose output is consumed by PI-SAM (see :class:`RepeatedArgsInstance`);
            ``False`` for the ``rosame_milp*`` baselines, which must keep the
            upstream vocabulary their network shares.
    """
    objects: List[Tuple[str, str]] = [
        (name, str(obj.type)) for name, obj in pddl_objects.items()
    ]
    for name, const in getattr(pddl_domain, "constants", {}).items():
        if name not in pddl_objects:
            objects.append((name, str(const.type)))
    instance_cls = RepeatedArgsInstance if include_repeated_args else PSInstance
    return instance_cls(ps_domain, objects)


def build_ps_instance(
    ps_domain: PSDomain, pddl_domain, problem, include_repeated_args: bool = False
) -> PSInstance:
    """Vendored Instance grounded on a problem's objects (+ domain constants)."""
    return build_ps_instance_from_objects(
        ps_domain, pddl_domain, problem.objects, include_repeated_args
    )


def proposition_of(instance: PSInstance, name: str, arg_names: Sequence[str]):
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


def _positive_propositions(entries: List[ObservationP]) -> List:
    """Propositions observed TRUE and unmasked (masked entries sit at prob 0.5)."""
    return [op.proposition for op in entries if op.prob > 0.5]


def observation_to_trace(
    instance: PSInstance,
    observation,
    goal_fluents: Optional[Set[Tuple[str, Tuple[str, ...]]]] = None,
    gt_state_indices: Optional[Set[int]] = None,
    gt_anchoring: GtAnchoring = GtAnchoring.INIT_ONLY,
) -> Optional[ObservationT]:
    """One (grounded, masked) pddl_plus Observation -> vendored ObservationT.

    The returned ObservationT carries three extra attributes used by our encoder:
    ``instance`` (the trace's own grounding), ``actions`` (t -> observed grounded
    Action) and ``hard_states`` (t -> set of true Propositions to hard-fix).

    Args:
        goal_fluents: positive fluents of the (GT) final state as
            ``(pred_name, (obj, ...))`` tuples; ``None`` leaves the final state
            soft (no hard goal constraints — encoder skips paper eqs. 21–22).
        gt_state_indices: 0-based STATE indices known to be ground truth (the
            CDPS ``gt_states_by_obs`` convention). Read only when
            ``gt_anchoring`` is ``ALL_GT_STATES``.
        gt_anchoring: which of those states to hard-fix (see :class:`GtAnchoring`).
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
                prop = proposition_of(
                    instance, gp.name, [gp.object_mapping[k] for k in gp.signature.keys()]
                )
                if prop is not None:
                    entries.append(ObservationP(prop, _state_prob(gp)))
        seen = Counter(op.proposition for op in entries)
        duplicated = [str(p) for p, n in seen.items() if n > 1]
        if duplicated:
            raise ValueError(
                f"State {t} maps multiple grounded predicates to the same "
                f"proposition(s) {duplicated} — the observation is "
                f"contradictory (e.g. both polarities of one fluent). "
                f"See src/depot-polarity-test/README.md."
            )
        obs_p[t] = entries

    # init: positive unmasked fluents of state 1 (GT by assumption)
    init = _positive_propositions(obs_p[1])

    # extra hard-fixed states: the injected-GT ones are clean and unmasked, so
    # their positive fluents read straight off obs_p (state idx s -> time s+1).
    hard_states: Dict[int, Set] = {}
    if gt_anchoring is GtAnchoring.ALL_GT_STATES and gt_state_indices:
        for state_idx in gt_state_indices:
            if 0 <= state_idx <= step:
                hard_states[state_idx + 1] = set(_positive_propositions(obs_p[state_idx + 1]))

    # goal: GT final state when provided, else None (soft)
    goal = None
    if goal_fluents is not None:
        goal = []
        for name, arg_names in goal_fluents:
            prop = proposition_of(instance, name, arg_names)
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
    trace.hard_states = hard_states
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
