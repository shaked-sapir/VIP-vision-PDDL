import itertools
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Set, Union

from pddl_plus_parser.lisp_parsers import DomainParser
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
    if not args:
        return f"({name})"
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
    def serialize_state_positive_only(state: State, state_type: str) -> str:
        """Serialize a state with only positive predicates."""
        positive_predicates = []
        for pred_name, grounded_preds in state.state_predicates.items():
            for grounded_pred in grounded_preds:
                if grounded_pred.is_positive:
                    positive_predicates.append(grounded_pred.untyped_representation)

        predicates_str = ' '.join(positive_predicates)
        return f"({state_type} {predicates_str})"

    trajectory_lines = ["("]  # Start of trajectory file

    # Step 0: Initial state from the first component's previous_state
    init_state = observation.components[0].previous_state
    trajectory_lines.append(serialize_state_positive_only(init_state, ":init"))

    # Step 1+: For each component, append operator and next state
    for component in observation.components:
        action_str = str(component.grounded_action_call).strip()
        trajectory_lines.append(f"(operator: {action_str})")

        next_state = component.next_state
        trajectory_lines.append(serialize_state_positive_only(next_state, ":state"))

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


# ============================================================================
# Visual Facts Utilities
# ============================================================================
def translate_pddlgym_state_to_image_predicates(pddlgym_state_literals: list[str],
                                                imaged_obj_to_gym_obj_name: dict[str, str]) -> list[str]:
    """
    Translates predicates from PDDLGym format to image object format.

    This function takes state literals from a PDDLGym trajectory (e.g., from _trajectory.json)
    and translates the object names from gym format to image format using the reverse
    mapping of imaged_obj_to_gym_obj_name.

    Args:
        pddlgym_state_literals: List of literal strings from PDDLGym state
                               (e.g., ["on(a:block,b:block)", "clear(c:block)"])

    Returns:
        List of predicates with image object names
        (e.g., ["on(red_block:block,blue_block:block)", "clear(green_block:block)"])

    Example:
        If imaged_obj_to_gym_obj_name = {"red_block": "a", "blue_block": "b"}
        And pddlgym_state_literals = ["on(a:block,b:block)"]
        Returns: ["on(red_block:block,blue_block:block)"]
    """

    # Create reverse mapping: gym_obj_name -> image_obj_name
    gym_to_image = {gym: img for img, gym in imaged_obj_to_gym_obj_name.items()}
    translated = []

    for literal in pddlgym_state_literals:
        m = re.match(r'([a-zA-Z0-9_-]+)\((.*)\)', literal)
        if not m:
            continue  # skip malformed literals

        pred, args_str = m.groups()
        if not args_str.strip():  # 0-arity predicate
            translated.append(literal)
            continue

        args = []
        for arg in map(str.strip, args_str.split(',')):
            if ':' not in arg:
                args.append(arg)
                continue

            name, typ = (p.strip() for p in arg.split(':', 1))
            name = gym_to_image.get(name, name)  # replace if we have a mapping
            args.append(f"{name}:{typ}")

        translated.append(f"{pred}({','.join(args)})")

    return translated


def extract_objects_from_pddlgym_state(
        pddlgym_state_objects: list[str],
        imaged_obj_to_gym_obj_name: dict[str, str]
) -> Set[str]:
    """
    Extracts objects from a ground truth trajectory state and translates them to image object names.

    This method:
    1. Loads the ground truth trajectory JSON file
    2. Extracts objects from the "objects" key in the specified state
    3. Back-translates using the reverse of imaged_obj_to_gym_obj_name
    4. Returns a list of object:type pairs

    Args:
        gt_trajectory_path: Path to the ground truth trajectory JSON file.
                           If None, uses self.gt_json_trajectory_path
        state_index: Index of the state to extract objects from (default: 0 for first state)

    Returns:
        List of strings in format "object_name:type"
        (e.g., ["red_block:block", "blue_block:block", "robot:robot"])

    Example:
        If imaged_obj_to_gym_obj_name = {"red_block": "a", "blue_block": "b"}
        And ground truth has objects = ["a:block", "b:block", "robot:robot"]
        Returns: ["red_block:block", "blue_block:block", "robot:robot"]
    """
    # Use provided path or fall back to instance variable
    gym2img = {gym: img for img, gym in imaged_obj_to_gym_obj_name.items()}

    def translate(obj_str: str) -> str:
        if ":" not in obj_str:
            return obj_str
        name, typ = (s.strip() for s in obj_str.split(":", 1))
        return f"{gym2img.get(name, name)}:{typ}"

    return {translate(o) for o in pddlgym_state_objects}


def propagate_frame_axioms_in_trajectory(
    trajectory_path: Union[str, Path],
    masking_info_path: Union[str, Path],
    domain_path: Union[str, Path]
) -> Path:
    """
    Propagate predicates that should persist according to frame axioms.

    For each state transition, if a predicate:
    - Exists in current state
    - Doesn't exist in next state
    - Doesn't exist in next state's masking info
    - Has no objects involved in the action
    Then add it to next state.

    Args:
        trajectory_path: Path to .trajectory file (e.g., problem8.trajectory)
        masking_info_path: Path to .masking_info file (e.g., problem8.masking_info)
        domain_path: Path to domain PDDL file

    Returns:
        Path to new .trajectory file with frame axioms propagated
    """
    from src.utils.masking import load_masking_info
    from pddl_plus_parser.lisp_parsers import TrajectoryParser

    trajectory_path, masking_info_path, domain_path = Path(trajectory_path), Path(masking_info_path), Path(domain_path)

    # Parse files
    domain: Domain = DomainParser(domain_path).parse_domain()
    parser = TrajectoryParser(domain)
    observation = parser.parse_trajectory(trajectory_path)
    masking_info = load_masking_info(masking_info_path, domain)

    def extract_objs(s: str) -> Set[str]:
        """Extract object names from '(on a b)' -> {'a', 'b'}"""
        return set(s.strip('()').split()[1:]) if '(' in s else set()

    def pred_str(p) -> str:
        """Convert GroundedPredicate to 'on(a:block,b:block)'"""
        if not p.object_mapping:
            return f"{p.name}()"
        args = ','.join(f"{v}:{p.signature[k].name}" for k, v in p.object_mapping.items())
        return f"{p.name}({args})"

    # Apply frame axioms - start with initial state and propagate forward
    curr = {pred_str(p) for preds in observation.components[0].previous_state.state_predicates.values()
            for p in preds if p.is_positive}

    trajectory = []
    for i, comp in enumerate(observation.components):
        next = {pred_str(p) for preds in comp.next_state.state_predicates.values() for p in preds if p.is_positive}
        curr_mask = {pred_str(p) for p in masking_info[i]} if i < len(masking_info) else set()
        next_mask = {pred_str(p) for p in masking_info[i + 1]} if i + 1 < len(masking_info) else set()
        action_objs = extract_objs(str(comp.grounded_action_call))

        # Remove predicates from next that are not in curr, not in curr_mask, and don't share objects with action
        next -= {p for p in next if p not in curr and p not in curr_mask and
                 extract_objs(parse_gym_to_pddl_literal(p)).isdisjoint(action_objs)}

        # Propagate from curr to next
        propagated = [p for p in curr if p not in next and p not in next_mask
                      and ((pred_objs := extract_objs(parse_gym_to_pddl_literal(p))) != set())
                      and pred_objs.isdisjoint(action_objs)]

        trajectory.append({
            'step': i + 1,
            'current_state': {'literals': list(curr)},
            'ground_action': str(comp.grounded_action_call).replace('(', '').replace(')', ''),
            'next_state': {'literals': list(next) + propagated}
        })

        # Update curr for next iteration with propagated predicates
        curr = next | set(propagated)

    # Write output
    problem_name = trajectory_path.stem + '_frame_axioms'
    build_trajectory_file(trajectory, problem_name, trajectory_path.parent)
    return trajectory_path.parent / f"{problem_name}.trajectory"


def json_to_trajectory_file(json_trajectory_path: Union[str, Path]) -> Path:
    """
    Converts a _trajectory.json file to a .trajectory file in PDDL format.

    Args:
        json_trajectory_path: Path to the JSON trajectory file (e.g., problem1_trajectory.json)

    Returns:
        Path to the generated .trajectory file
    """
    json_trajectory_path = Path(json_trajectory_path)

    # Load JSON data
    with open(json_trajectory_path, 'r') as f:
        trajectory_data = json.load(f)

    # Extract problem name from filename (remove _trajectory.json suffix)
    problem_name = json_trajectory_path.stem.replace('_trajectory', '')

    # Build trajectory file using existing function
    build_trajectory_file(trajectory_data, problem_name, json_trajectory_path.parent)

    return json_trajectory_path.parent / f"{problem_name}.trajectory"


if __name__ == "__main__":
    propagate_frame_axioms_in_trajectory(
        Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/data/blocksworld/multi_problem_04-12-2025T12:00:44__model=gpt-5.1__steps=50__planner/training/trajectories/problem7/problem7.trajectory"),
        Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/data/blocksworld/multi_problem_04-12-2025T12:00:44__model=gpt-5.1__steps=50__planner/training/trajectories/problem7/problem7.masking_info"),
        Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/domains/blocksworld/blocksworld.pddl")
    )