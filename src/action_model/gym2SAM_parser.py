import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Set, Tuple, Any, Union

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import (
    State,
    GroundedPredicate,
    Observation,
    ActionCall,
    Domain, Problem, PDDLObject, SignatureType
)

from src.action_model.pddl2gym_parser import NEGATION_PREFIX, UNKNOWN_PREFIX, get_predicate_base_form
from src.typings import TrajectoryStep

ObjectMappingType = Dict[str, PDDLObject]


def lift_predicate(grounded_predicate_str: str, pddl_domain: Domain) -> Tuple[
    str, SignatureType, Dict[str, str]]:
    """
    parses a grounded predicate string in pddlgym format into pddl-plus-parser format ingredients, lifted form.
    :param grounded_predicate_str: string representing the predicate in pddlgym format.
    it is of the form: <pred_name>(<param_name>:<param_type>), e.g. "on(a:block,b:block)".
    :param pddl_domain: domain representation by pddl-plus-parser, needed for mapping the right lifted representations
    of the grounded predicates.
    :return:
    - @ret predicate_name(str): the name of the predicate
    - @ret lifted_signature(SignatureType): the signature of the lifted predicate
    - @ret object_mapping(Dict[str, str]): mapping a lifted predicate object name to its corresponding grounded object
                                           in the grounded instance parameter.
    """
    predicate_base_form = get_predicate_base_form(grounded_predicate_str)  # remove predicate tags if any
    predicate_name, grounded_arguments_part = predicate_base_form.split("(", 1)
    grounded_arguments_part = grounded_arguments_part.rstrip(')')

    lifted_signature: SignatureType = pddl_domain.predicates[predicate_name].signature

    grounded_arguments: List[str] = grounded_arguments_part.split(",")
    grounded_arguments_names: List[str] = [arg.split(":")[0] for arg in grounded_arguments]
    object_mapping: Dict[str, str] = {lifted_param_name: grounded_arg for lifted_param_name, grounded_arg in
                                      zip(lifted_signature.keys(), grounded_arguments_names)}

    return predicate_name, lifted_signature, object_mapping


def parse_grounded_predicates(grounded_predicate_strs: List[str], pddl_domain: Domain) -> Set[GroundedPredicate]:
    """Parse a list of grounded literals into a set of GroundedPredicate objects."""
    grounded_predicates = set()
    for predicate_str in grounded_predicate_strs:
        predicate_name, lifted_predicate_signature, predicate_object_mapping = lift_predicate(predicate_str,
                                                                                              pddl_domain)
        grounded_predicates.add(
            GroundedPredicate(name=predicate_name,
                              signature=lifted_predicate_signature,
                              object_mapping=predicate_object_mapping,
                              is_positive=NEGATION_PREFIX not in predicate_str,
                              is_masked=UNKNOWN_PREFIX in predicate_str)
        )
    return grounded_predicates


def group_predicates_by_name(predicates: Set[GroundedPredicate]) -> Dict[str, Set[GroundedPredicate]]:
    grouped_predicates = defaultdict(set)
    # [grouped_predicates[predicate.name].add(predicate) for predicate in predicates]
    [grouped_predicates[predicate.lifted_untyped_representation].add(predicate) for predicate in predicates]

    return grouped_predicates


def parse_gym_state(state: Dict[str, List], is_initial: bool, pddl_domain: Domain) -> State:
    """Parse a state dict into a State object."""
    state_literals: List[str] = state["literals"]
    unknown_literals: List[str] = state.get("unknown", [])
    all_literals = state_literals + unknown_literals

    grounded_predicates: Set[GroundedPredicate] = parse_grounded_predicates(all_literals, pddl_domain)
    grouped_predicates: Dict[str, Set[GroundedPredicate]] = group_predicates_by_name(grounded_predicates)

    return State(
        is_init=is_initial,
        predicates=grouped_predicates,
        fluents={}
    )


def parse_action_call(action_string: str) -> ActionCall:
    action_name, params = action_string.split("(", 1)
    params = params.rstrip(")").split(", ")
    params_names = [param.split(":")[0] for param in params]

    return ActionCall(
        name=action_name,
        grounded_parameters=params_names
    )


# TODO: make all calls to this accept trajectory only as a list of TrajectoryStep instead of regular dict
def create_observation_from_trajectory(trajectory: List[Dict | TrajectoryStep], pddl_domain: Domain,
                                       pddl_problem: Problem) -> Observation:
    object_mapping = pddl_problem.objects
    observation = Observation()
    observation.add_problem_objects(object_mapping)

    for step in trajectory:
        current_state = parse_gym_state(step["current_state"], is_initial=(step["step"] == 1),
                                        pddl_domain=pddl_domain)  # TODO: fix this after the general form of the steps
        resulted_state = parse_gym_state(step["next_state"], is_initial=False, pddl_domain=pddl_domain)
        action_call = parse_action_call(step["ground_action"])
        observation.add_component(
            previous_state=current_state,
            call=action_call,
            next_state=resulted_state
        )

    return observation
