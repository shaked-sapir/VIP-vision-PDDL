"""Core state, observation, predicate, and grounding utilities."""

import itertools
from typing import List, Dict, Optional, Set, Tuple

from pddl_plus_parser.models import (
    Observation, ObservedComponent, Predicate, PDDLObject,
    GroundedPredicate, State, Domain,
)

from src.action_model.pddl2gym_parser import negate_str_predicate


# ============================================================================
# State predicate accessors
# ============================================================================

def get_state_grounded_predicates(state: State) -> Set[GroundedPredicate]:
    return set().union(*state.state_predicates.values())


def get_state_unmasked_predicates(state: State) -> Set[GroundedPredicate]:
    return {pred for pred in get_state_grounded_predicates(state) if not pred.is_masked}


def get_state_masked_predicates(state: State) -> Set[GroundedPredicate]:
    return {pred for pred in get_state_grounded_predicates(state) if pred.is_masked}


def flip_fluent_in_state(state: State, fluent_str: str) -> None:
    """Flip the polarity of a grounded fluent in-place.

    Searches for the predicate whose untyped_representation matches fluent_str
    (or its negation) and toggles its is_positive flag.

    Args:
        state: The state whose predicate to flip.
        fluent_str: Grounded fluent string, e.g. "(holding a)" or "(not (holding a))".

    Raises:
        ValueError: If the fluent cannot be found in the state.
    """
    candidates = {fluent_str, negate_str_predicate(fluent_str)}

    for gp in get_state_grounded_predicates(state):
        if gp.untyped_representation not in candidates:
            continue

        base_key = (
            gp.lifted_untyped_representation
            if gp.is_positive
            else negate_str_predicate(gp.lifted_untyped_representation)
        )

        for p in state.state_predicates[base_key]:
            if p.untyped_representation in candidates:
                p.is_positive = not p.is_positive
                return

    raise ValueError(
        f"Could not find fluent '{fluent_str}' or its negation to flip in state"
    )


def find_predicate_negation(
    predicate_set: Set[GroundedPredicate], predicate: GroundedPredicate,
) -> "Optional[GroundedPredicate]":
    """Find the negated version of a predicate in a set, or None.

    Matches by name and object_mapping with opposite is_positive polarity.

    Args:
        predicate_set: The set of predicates to search.
        predicate: The predicate whose negation to find.

    Returns:
        The negated counterpart if found, else None.
    """
    for p in predicate_set:
        if (p.name == predicate.name
                and p.object_mapping == predicate.object_mapping
                and p.is_positive != predicate.is_positive):
            return p
    return None


def state_positive_set(state: State, unmasked_only: bool = False) -> Set[str]:
    """Set of untyped_representation of positive fluents in state."""
    preds = get_state_unmasked_predicates(state) if unmasked_only else get_state_grounded_predicates(state)
    return {p.untyped_representation for p in preds if p.is_positive}


# ============================================================================
# State comparison
# ============================================================================

def compare_states(pred_state: State, gt_state: State, unmasked_only: bool = True) -> Dict:
    """Compare predicted state to GT. Returns tp, fp, fn, precision, recall, masked_count."""
    pred_set = state_positive_set(pred_state, unmasked_only=unmasked_only)
    gt_set = state_positive_set(gt_state, unmasked_only=False)
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    masked_count = len(get_state_masked_predicates(pred_state))
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "masked_count": masked_count}


def compare_observations(obs_pred: Observation, obs_gt: Observation, unmasked_only: bool = True) -> Dict:
    """Compare observation states to GT. Returns per_state list and overall aggregated metrics."""
    pred_states = [obs_pred.components[0].previous_state] + [c.next_state for c in obs_pred.components]
    gt_states = [obs_gt.components[0].previous_state] + [c.next_state for c in obs_gt.components]
    if len(pred_states) != len(gt_states):
        return {"per_state": [], "overall": {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "masked_count": 0}}
    per_state = [compare_states(p, g, unmasked_only=unmasked_only) for p, g in zip(pred_states, gt_states)]
    total = {"tp": sum(s["tp"] for s in per_state), "fp": sum(s["fp"] for s in per_state),
             "fn": sum(s["fn"] for s in per_state), "masked_count": sum(s["masked_count"] for s in per_state)}
    total["precision"] = total["tp"] / (total["tp"] + total["fp"]) if (total["tp"] + total["fp"]) > 0 else 0.0
    total["recall"] = total["tp"] / (total["tp"] + total["fn"]) if (total["tp"] + total["fn"]) > 0 else 0.0
    return {"per_state": per_state, "overall": total}


# ============================================================================
# Observation utilities
# ============================================================================

def copy_state(state: State) -> State:
    """Copy a state, preserving the ``is_masked`` flag on every predicate.

    ``State.copy()`` from pddl_plus_parser drops ``is_masked``; this
    function creates new ``GroundedPredicate`` objects that carry the flag
    over.  The ``signature`` and ``object_mapping`` dicts are shared (they
    are never mutated in learner paths).

    Args:
        state: The state to copy.

    Returns:
        A new ``State`` with independent predicate objects.
    """
    copied_predicates = {
        key: {
            GroundedPredicate(p.name, p.signature, p.object_mapping, p.is_positive, p.is_masked)
            for p in preds
        }
        for key, preds in state.state_predicates.items()
    }
    return State(copied_predicates, state.state_fluents, is_init=state.is_init)


def copy_observation(observation: Observation) -> Observation:
    """Creates a deep copy of the given Observation object."""
    copied_observation = Observation()
    for component in observation.components:
        copied_component = ObservedComponent(
            previous_state=component.previous_state.copy(),
            call=component.grounded_action_call,
            next_state=component.next_state.copy(),
            is_successful=component.is_successful
        )
        copied_observation.components.append(copied_component)

    copied_observation.grounded_objects = {
        name: obj.copy() for name, obj in observation.grounded_objects.items()
    }
    return copied_observation


def copy_observation_linked(observation: Observation) -> Observation:
    """Copy an observation while maintaining the shared-state invariant.

    The standard ``copy_observation`` copies each state independently,
    breaking the invariant ``comp[i].next_state is comp[i+1].previous_state``.
    This function produces N+1 state copies and re-links them so that
    consecutive components share the same boundary state object — required
    by ``apply_fluent_patches`` in the noisy learner.

    Args:
        observation: The source observation.

    Returns:
        A new ``Observation`` with N+1 linked state copies.
    """
    if not observation.components:
        linked = Observation()
        linked.grounded_objects = dict(observation.grounded_objects)
        return linked

    # Build N+1 independent state copies.
    state_copies = [copy_state(observation.components[0].previous_state)]
    for component in observation.components:
        state_copies.append(copy_state(component.next_state))

    linked = Observation()
    linked.grounded_objects = dict(observation.grounded_objects)
    for i, component in enumerate(observation.components):
        linked.components.append(ObservedComponent(
            previous_state=state_copies[i],
            call=component.grounded_action_call,
            next_state=state_copies[i + 1],
            is_successful=component.is_successful,
        ))
    return linked


def observations_equal(obs1: Observation, obs2: Observation) -> bool:
    """Check if two Observation objects are equal."""
    if len(obs1.components) != len(obs2.components):
        return False
    if set(obs1.grounded_objects.keys()) != set(obs2.grounded_objects.keys()):
        return False
    for obj_name in obs1.grounded_objects.keys():
        obj1 = obs1.grounded_objects[obj_name]
        obj2 = obs2.grounded_objects[obj_name]
        if obj1.name != obj2.name or obj1.type.name != obj2.type.name:
            return False
    for comp1, comp2 in zip(obs1.components, obs2.components):
        if str(comp1.grounded_action_call) != str(comp2.grounded_action_call):
            return False
        if comp1.is_successful != comp2.is_successful:
            return False
        if comp1.previous_state != comp2.previous_state:
            return False
        if comp1.next_state != comp2.next_state:
            return False
    return True


# ============================================================================
# Predicate grounding
# ============================================================================

def get_all_possible_groundings(
    predicate: Predicate, grounded_objects: Dict[str, PDDLObject],
) -> Set[GroundedPredicate]:
    param_names = list(predicate.signature.keys())
    param_types = list(predicate.signature.values())

    object_domains = []
    for t in param_types:
        matches = [obj.name for obj in grounded_objects.values() if obj.type.is_sub_type(t)]
        object_domains.append(matches)

    grounded_predicates = set()
    for values in itertools.product(*object_domains):
        mapping = dict(zip(param_names, values))
        grounded_predicates.add(GroundedPredicate(
            name=predicate.name,
            signature=predicate.signature,
            object_mapping=mapping,
            is_positive=predicate.is_positive,
        ))
    return grounded_predicates


def get_all_possible_groundings_for_domain(
    domain: Domain, observation: Observation,
) -> Dict[str, Set[GroundedPredicate]]:
    """For each lifted predicate in the domain, compute all possible groundings."""
    grounded_objects = observation.grounded_objects
    all_grounded_predicates = {}
    for lifted_predicate in domain.predicates.values():
        all_grounded_predicates[lifted_predicate.untyped_representation] = get_all_possible_groundings(
            lifted_predicate, grounded_objects)
    return all_grounded_predicates


def ground_all_predicates_in_state(
    state: State, all_domain_grounded_predicates: Dict[str, Set[GroundedPredicate]],
) -> State:
    """Add missing predicate groundings to the state as negative literals."""
    new_state = state.copy()
    for predicate_name, grounded_predicates in state.state_predicates.items():
        new_state.state_predicates[predicate_name] = set(grounded_predicates)

    for predicate_name, grounded_predicates in all_domain_grounded_predicates.items():
        for grounded_predicate in grounded_predicates:
            if grounded_predicate not in new_state.state_predicates.get(predicate_name, set()):
                (new_state.state_predicates.setdefault(predicate_name, set())
                 .add(grounded_predicate.copy(is_negated=True)))
    return new_state


def ground_all_states_in_observation(
    observation: Observation, all_domain_grounded_predicates: Dict[str, Set[GroundedPredicate]],
) -> Observation:
    """Ground all predicates in each state of the observation."""
    new_observation = copy_observation(observation)
    for component in new_observation.components:
        component.previous_state = ground_all_predicates_in_state(
            component.previous_state, all_domain_grounded_predicates)
        component.next_state = ground_all_predicates_in_state(
            component.next_state, all_domain_grounded_predicates)
    return new_observation


def ground_observation_completely(domain: Domain, observation: Observation) -> Observation:
    """Ground all predicates in the observation — every possible grounding is explicit."""
    all_domain_grounded_predicates = get_all_possible_groundings_for_domain(domain, observation)
    return ground_all_states_in_observation(observation, all_domain_grounded_predicates)
