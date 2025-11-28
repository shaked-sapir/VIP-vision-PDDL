import itertools
import os
import re
from pathlib import Path
from typing import List, Dict, Set

from pddl_plus_parser.models import Observation, ObservedComponent, Predicate, PDDLObject, GroundedPredicate, State, \
    Domain
from pddlgym.core import PDDLEnv
from pddlgym.parser import Operator

from src.utils.containers import shrink_whitespaces


# TODO: refactor this file, especially after exporting the pddl-plus-parser potential methods to this package.

def set_problem_by_name(pddl_env: PDDLEnv, problem_name: str):
    """
    Fixes a problem for the environment using the specific problem name.
    This is used for ease, as it is not implemented in pddlgym's PDDLEnv class,
    yet necessary for easy setting problems.
    :param pddl_env: PDDLEnv instance
    :param problem_name: specific problem name, should match a filename containing the problem definition.
    :return: (None) fixes the problem within the environment instance.
    """
    # Ensure the problem name ends with '.pddl'
    if not problem_name.endswith('.pddl'):
        problem_name += '.pddl'
    # Find the index of the problem with the given name
    for idx, prob in enumerate(pddl_env.problems):
        if prob.problem_fname.endswith(problem_name):
            pddl_env.fix_problem_index(idx)
            return
    raise ValueError(f"Problem '{problem_name}' not found in the problem list of the environment you provided.")


def ground_action(operator: Operator, assignment: dict) -> str:
    action_name: str = operator.name
    grounded_params = [str(assignment[lifted_param]) for lifted_param in operator.params]
    return f"{action_name}({', '.join(grounded_params)})"


def parse_gym_to_pddl_literal(literal: str) -> str:
    """Convert 'on(a:block,b:block)' -> '(on a b)'"""
    name, args = literal.split('(')
    args = args.rstrip(')')
    args_clean = [arg.split(':')[0] for arg in args.split(',')]
    parsed_literal = f"({name} {' '.join(args_clean)})"
    return shrink_whitespaces(parsed_literal)


def parse_gym_to_pddl_ground_action(ground_action: str) -> str:
    """Convert 'put-down(a:block, robot:robot)' -> '(putdown a robot)'"""
    if '(' in ground_action:
        name, args = ground_action.split('(')
        args = args.rstrip(')')
        args_clean = [arg.split(':')[0] for arg in args.split(',')]
        parsed_action = f"({name} {' '.join(args_clean)})"
    else:
        # If no parentheses, just remove dashes
        parsed_action = f"({ground_action})"
    return shrink_whitespaces(parsed_action)


def build_trajectory_file(trajectory_data: List[dict], problem_name: str, output_path: Path) -> None:
    output_path = os.path.join(output_path, f"{problem_name}.trajectory")

    trajectory_lines = ["("]  # the opener of the trajectory file

    # Step 0: Write the initial state from current_state of the first entry
    init_literals = trajectory_data[0]['current_state']['literals']
    init_literals_parsed = ' '.join(parse_gym_to_pddl_literal(lit) for lit in init_literals)
    trajectory_lines.append(f"(:init {init_literals_parsed})")

    # Step 1: Write the first operator (from first entry)
    ground_action = trajectory_data[0]['ground_action']
    ground_action_parsed = parse_gym_to_pddl_ground_action(ground_action)
    trajectory_lines.append(f"(operator: {ground_action_parsed})")

    # Then continue: For each NEXT state and NEXT action
    for i in range(1, len(trajectory_data)):
        step_info = trajectory_data[i]

        # Write the state
        current_literals = step_info['current_state']['literals']
        current_literals_parsed = ' '.join(parse_gym_to_pddl_literal(lit) for lit in current_literals)
        trajectory_lines.append(f"(:state {current_literals_parsed})")

        # Write the operator
        ground_action = step_info['ground_action']
        ground_action_parsed = parse_gym_to_pddl_ground_action(ground_action)
        trajectory_lines.append(f"(operator: {ground_action_parsed})")

    # Finally write the last :state after the last action
    final_state_literals = trajectory_data[-1]['next_state']['literals']
    final_state_literals_parsed = ' '.join(parse_gym_to_pddl_literal(lit) for lit in final_state_literals)
    trajectory_lines.append(f"(:state {final_state_literals_parsed})")

    trajectory_lines.append(")")  # the closer of the trajectory file

    # Save to file
    with open(output_path, "w") as f:
        f.write('\n'.join(trajectory_lines))

    print(f"Trajectory saved to {output_path}")


def observation_to_trajectory_file(observation: Observation, output_path: Path) -> Path:
    """
    Builds a .trajectory file from a single-agent Observation object.

    :param observation: the Observation object containing components with previous/next states and actions.
    :param output_path: the directory to save the trajectory file into.
    """
    trajectory_lines = ["("]  # Start of trajectory file

    # Step 0: Initial state from the first component's previous_state
    init_state = observation.components[0].previous_state
    trajectory_lines.append(init_state.serialize().strip())

    # Step 1+: For each component, append operator and next state
    for component in observation.components:
        action_str = str(component.grounded_action_call).strip()
        trajectory_lines.append(f"(operator: {action_str})")

        next_state = component.next_state
        trajectory_lines.append(next_state.serialize().strip())

    trajectory_lines.append(")")  # Close the trajectory

    # Save to file
    with open(output_path, "w") as file:
        file.write('\n'.join(trajectory_lines))

    print(f"Trajectory saved to {output_path}")

    return output_path


# TODO: suggest this to pddl_plus_parser as a new class method - I could use it - passed CR, not merged yet
def copy_observation(observation: Observation) -> Observation:
    """
    Creates a deep copy of the given Observation object.

    :param observation: the Observation object to copy.
    :return: a new Observation object with copied components and grounded objects.
    """
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


# TODO: suggest this to pddl_plus_parser as a new class method - I could use it => need to fix CR for supprting constants + add test
def get_all_possible_groundings(predicate: Predicate,
                                grounded_objects: Dict[str, PDDLObject]) -> Set[GroundedPredicate]:
    param_names = list(predicate.signature.keys())
    param_types = list(predicate.signature.values())  # handles signature with multiple objects of the same type

    # Get all objects compatible with each parameter type
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
            is_positive=predicate.is_positive
        ))

    return grounded_predicates


# TODO: suggest this to pddl_plus_parser as a new class method - I could use it
def get_state_grounded_predicates(state: State) -> Set[GroundedPredicate]:
    return set.union(*state.state_predicates.values())


def get_state_unmasked_predicates(state: State) -> Set[GroundedPredicate]:
    """Get all grounded predicates in the state, ignoring their negation."""
    return {pred for pred in get_state_grounded_predicates(state) if not pred.is_masked}


def get_state_masked_predicates(state: State) -> Set[GroundedPredicate]:
    """Get all grounded predicates in the state, ignoring their negation."""
    return {pred for pred in get_state_grounded_predicates(state) if pred.is_masked}


def multi_replace_predicate(p: str, mapping: dict[str, str]) -> str:
    # Sort by length to avoid partial overlaps (just in case)
    keys = sorted(mapping.keys(), key=len, reverse=True)
    # Match an exact token like "red:block" bounded by word boundaries
    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, keys)) + r')\b')
    return pattern.sub(lambda m: mapping[m.group(0)], p)


# ============================================================================
# Observation Grounding Utilities
# ============================================================================

def get_all_possible_groundings_for_domain(domain: Domain, observation: Observation) -> Dict[str, Set[GroundedPredicate]]:
    """
    For each lifted predicate in the domain, compute all possible groundings for the given observation.

    Note: This returns all groundings as positive literals, regardless of the actual state of the observation -
    so negativity can be handled as needed in states creation.

    :param domain: The domain containing lifted predicates.
    :param observation: The observation containing grounded objects.
    :return: A dictionary mapping lifted predicate names to their possible grounded predicates.
    """
    grounded_objects = observation.grounded_objects
    all_grounded_predicates = {}

    for lifted_predicate_name, lifted_predicate in domain.predicates.items():
        # keys are the untyped representations of the predicates, matching the predicate dicts of states
        all_grounded_predicates[lifted_predicate.untyped_representation] = get_all_possible_groundings(
            lifted_predicate, grounded_objects)

    return all_grounded_predicates


def ground_all_predicates_in_state(state: State,
                                   all_domain_grounded_predicates: Dict[str, Set[GroundedPredicate]]) -> State:
    """
    For each predicate in domain predicates, check all its possible groundings against the state's grounded
    predicates: if a grounding does not exist in the state then add it to the state as a negative literal.

    :param state: The state to ground all predicates in.
    :param all_domain_grounded_predicates: A dictionary mapping each predicate name to its possible grounded
        predicates in the domain.
    :return: A state with all predicates grounded, either positive or negative.
    """
    new_state = state.copy()

    # Add all grounded predicates from the state
    for predicate_name, grounded_predicates in state.state_predicates.items():
        new_state.state_predicates[predicate_name] = set(grounded_predicates)

    # For each predicate in the domain, check if it exists in the state, if not - add it as a negative literal
    for predicate_name, grounded_predicates in all_domain_grounded_predicates.items():
        for grounded_predicate in grounded_predicates:
            # We have to check if the there are any predicates with the same name in the state, and handle properly
            if grounded_predicate not in new_state.state_predicates.get(predicate_name, set()):
                (new_state.state_predicates.setdefault(predicate_name, set())
                 .add(grounded_predicate.copy(is_negated=True)))

    return new_state


def ground_all_states_in_observation(observation: Observation,
                                     all_domain_grounded_predicates: Dict[str, Set[GroundedPredicate]]
                                     ) -> Observation:
    """
    For a given observation, ground all predicates in each state of the observation.

    :param observation: An observation (trajectory) to handle.
    :param all_domain_grounded_predicates: A dictionary mapping each predicate name to its possible groundings in the domain,
           using the objects of the observation.
    :return: A full observation, with all possible literals for each state.
    """
    new_observation = copy_observation(observation)
    for component in new_observation.components:
        component.previous_state = ground_all_predicates_in_state(
            component.previous_state, all_domain_grounded_predicates)
        component.next_state = ground_all_predicates_in_state(
            component.next_state, all_domain_grounded_predicates)

    return new_observation


def ground_observation_completely(domain: Domain, observation: Observation) -> Observation:
    """
    Ground all predicates in the states of the observation.

    This function creates a "complete" observation where every possible predicate grounding
    is explicitly represented as either positive or negative in each state.

    :param domain: The domain containing lifted predicates.
    :param observation: The observation to ground.
    :return: A new observation with all predicates grounded.
    """
    all_domain_grounded_predicates = get_all_possible_groundings_for_domain(domain, observation)
    return ground_all_states_in_observation(observation, all_domain_grounded_predicates)
