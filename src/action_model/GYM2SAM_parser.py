 from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Set, Tuple

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import (
    State,
    GroundedPredicate,
    Observation,
    ActionCall,
    Domain, Problem, PDDLObject, SignatureType
)

ObjectMappingType = Dict[str, PDDLObject]


def convert_grounded_to_lifted(grounded_predicate_str: str, pddl_domain: Domain) -> Tuple[
    str, SignatureType, Dict[str, str]]:
    """
    parses a grounded predicate string in pddlgym format into pddl-plus-parser format ingredients.
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
    predicate_name, grounded_arguments_part = grounded_predicate_str.split("(", 1)
    grounded_arguments_part = grounded_arguments_part.rstrip(')')

    lifted_signature: SignatureType = pddl_domain.predicates[predicate_name].signature

    grounded_arguments: List[str] = grounded_arguments_part.split(",")
    grounded_arguments_names: List[str] = [arg.split(":")[0] for arg in grounded_arguments]
    object_mapping: Dict[str, str] = {lifted_param_name: grounded_arg for lifted_param_name, grounded_arg in
                                      zip(lifted_signature.keys(), grounded_arguments_names)}

    return predicate_name, lifted_signature, object_mapping


# TODO: fix this properly tomorrow.
def parse_grounded_predicates(grounded_predicate_strs: List[str], pddl_domain: Domain) -> Set[GroundedPredicate]:
    """Parse a list of grounded literals into a set of GroundedPredicate objects."""
    grounded_predicates = set()
    for predicate_str in grounded_predicate_strs:
        predicate_name, lifted_predicate_signature, predicate_object_mapping = convert_grounded_to_lifted(predicate_str,
                                                                                                          pddl_domain)
        grounded_predicates.add(
            GroundedPredicate(name=predicate_name,
                              signature=lifted_predicate_signature,
                              object_mapping=predicate_object_mapping)
        )
    return grounded_predicates


def group_predicates_by_name(predicates: Set[GroundedPredicate]) -> Dict[str, Set[GroundedPredicate]]:
    grouped_predicates = defaultdict(set)
    [grouped_predicates[obj.name].add(obj) for obj in predicates]

    return grouped_predicates


def parse_gym_state(gym_state_str: str, is_initial: bool, pddl_domain: Domain) -> State:
    """Parse a state string into a State object."""
    # Extract literals
    literals_part: str = gym_state_str.split("literals=frozenset({")[1].split("}),")[0]
    literals: List[str] = literals_part.split(", ")

    # Parse grounded predicates
    grounded_predicates: Set[GroundedPredicate] = parse_grounded_predicates(literals, pddl_domain)
    grouped_predicates: Dict[str, Set[GroundedPredicate]] = group_predicates_by_name(grounded_predicates)

    # Create and return the State object
    return State(
        is_init=is_initial,
        predicates=grouped_predicates,
        fluents={}
    )


def parse_action_call(action_string: str) -> ActionCall:
    """Parse an action string into an ActionCall object."""
    action_name, params = action_string.split("(", 1)
    params = params.rstrip(")").split(", ")
    params_names = [param.split(":")[0] for param in params]

    return ActionCall(
        name=action_name,
        grounded_parameters=params_names
    )


def create_observation_from_trajectory(trajectory: List[Dict], pddl_domain: Domain,
                                       pddl_problem: Problem) -> Observation:
    """Create an Observation object from a trajectory."""

    object_mapping = pddl_problem.objects
    observation = Observation()
    observation.add_problem_objects(object_mapping)

    for step in trajectory:
        # Parse current and resulted states
        current_state = parse_gym_state(step["current_state"], is_initial=(step["step"] == 0), pddl_domain=pddl_domain)
        resulted_state = parse_gym_state(step["ground_resulted_state"], is_initial=False, pddl_domain=pddl_domain)

        # Parse action call
        action_call = parse_action_call(step["ground_action"])

        # Add ObservedComponent
        observation.add_component(
            previous_state=current_state,
            call=action_call,
            next_state=resulted_state
        )

    # Create and return the Observation object
    return observation


# Example trajectory data
example_trajectory = [
    {
        "step": 0,
        "current_state": "State(literals=frozenset({ontable(f:block), on(b:block,c:block), clear(b:block), holding(a:block), on(c:block,d:block), on(e:block,f:block), on(d:block,e:block), handfull(robot:robot)}), objects=frozenset({e:block, robot:robot, b:block, a:block, c:block, d:block, f:block}), goal=AND[on(b:block,c:block), on(c:block,d:block), on(d:block,e:block), on(e:block,f:block), on(f:block,a:block)])",
        "ground_action": "putdown(a:block)",
        "operator_object_assignment": {
            "?x": "a",
            "?robot": "robot"
        },
        "lifted_preconds": "[pickup(?x:block), clear(?x:block), ontable(?x:block), handempty(?robot:robot)]",
        "ground_resulted_state": "State(literals=frozenset({on(c:block,d:block), ontable(f:block), handempty(robot:robot), ontable(a:block), on(b:block,c:block), on(e:block,f:block), clear(b:block), on(d:block,e:block), clear(a:block)}), objects=frozenset({f:block, c:block, b:block, robot:robot, e:block, a:block, d:block}), goal=AND[on(b:block,c:block), on(c:block,d:block), on(d:block,e:block), on(e:block,f:block), on(f:block,a:block)])"
    }
]

# TODO: this part is only an example, should be generalized later
# TODO: establish a connection between the trajectory and the problem it was created from (needed for proper object mapping)
# Create Observation object from the trajectory
pddl_plus_domain: Domain = DomainParser(Path("../domains/blocks/blocks.pddl")).parse_domain()
pddl_plus_problem: Problem = ProblemParser(Path("../domains/blocks/problem9.pddl"), pddl_plus_domain).parse_problem()
observation: Observation = create_observation_from_trajectory(example_trajectory, pddl_plus_domain, pddl_plus_problem)

# Output the resulting Observation object
for component in observation.components:
    print(str(component))
