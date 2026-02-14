import itertools
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Set, Union, Tuple

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser
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
                if grounded_pred.is_positive and not grounded_pred.is_masked:
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


def observations_equal(obs1: Observation, obs2: Observation) -> bool:
    """
    Check if two Observation objects are equal.
    
    Two observations are considered equal if:
    - They have the same number of components
    - Each component has the same previous_state, grounded_action_call, next_state, and is_successful
    - They have the same grounded_objects
    
    :param obs1: First Observation object
    :param obs2: Second Observation object
    :return: True if observations are equal, False otherwise
    """
    # Check number of components
    if len(obs1.components) != len(obs2.components):
        return False
    
    # Check grounded objects
    if set(obs1.grounded_objects.keys()) != set(obs2.grounded_objects.keys()):
        return False
    
    for obj_name in obs1.grounded_objects.keys():
        obj1 = obs1.grounded_objects[obj_name]
        obj2 = obs2.grounded_objects[obj_name]
        if obj1.name != obj2.name or obj1.type.name != obj2.type.name:
            return False
    
    # Check each component
    for comp1, comp2 in zip(obs1.components, obs2.components):
        # Check action call
        if str(comp1.grounded_action_call) != str(comp2.grounded_action_call):
            return False
        
        # Check is_successful
        if comp1.is_successful != comp2.is_successful:
            return False
        
        # Check previous_state
        if comp1.previous_state != comp2.previous_state:
            return False
        
        # Check next_state
        if comp1.next_state != comp2.next_state:
            return False
    
    return True


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
    domain_path: Union[str, Path],
    mode: str = "consider_masking"
) -> Tuple[Path, Path]:
    """
    Frame-closure propagation (forward only):

    For each transition (s, a, s'):
      - For any *grounded* positive literal p with objects which are NOT subset of action objects:
          enforce persistence between s and s'
          => add missing p into s', remove spurious p from s'

    Args:
        trajectory_path: Path to .trajectory file
        masking_info_path: Path to .masking_info file
        domain_path: Path to domain PDDL file
        mode: Propagation mode:
            - "ignore_masking": Ignore masking when deciding persistence, but update masking to remove fixed predicates
            - "consider_masking": Only propagate predicates that are NOT in the masking of the next state

    Returns:
      (new_trajectory_path, new_masking_info_path)
    """
    from src.utils.masking import load_masking_info, save_masking_info

    if mode not in ["ignore_masking", "consider_masking"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'ignore_masking' or 'consider_masking'")

    trajectory_path = Path(trajectory_path)
    masking_info_path = Path(masking_info_path)
    domain_path = Path(domain_path)

    domain: Domain = DomainParser(domain_path).parse_domain()
    obs: Observation = TrajectoryParser(domain).parse_trajectory(trajectory_path)
    masking: List[Set[GroundedPredicate]] = load_masking_info(masking_info_path, domain)
    masking = [set(m) for m in masking]  # copy

    def pddl_objs(pddl_lit: str) -> Set[str]:
        s = pddl_lit.strip()
        if s.startswith("(not"):
            s = s[len("(not"):].strip()
        s = s.strip("() ").split()
        return set(s[1:]) if len(s) > 1 else set()

    def gp_to_gym(gp: GroundedPredicate) -> str:
        if not gp.object_mapping:
            return f"{gp.name}()"
        args = ",".join(f"{v}:{gp.signature[k].name}" for k, v in gp.object_mapping.items())
        return f"{gp.name}({args})"

    def positive_gym_literals(state) -> Set[str]:
        return {
            gp_to_gym(p)
            for preds in state.state_predicates.values()
            for p in preds
            if p.is_positive
        }

    def is_frame_literal(gym_lit: str, action_objs: Set[str]) -> bool:
        pddl_pred_str = parse_gym_to_pddl_literal(gym_lit)
        objs = pddl_objs(pddl_pred_str)
        return bool(objs) and not objs.issubset(action_objs)

    # initial state is trusted
    curr = positive_gym_literals(obs.components[0].previous_state)
    out_steps = []

    for i, comp in enumerate(obs.components):
        nxt = positive_gym_literals(comp.next_state)
        action_objs = pddl_objs(str(comp.grounded_action_call))

        curr_frame = {p for p in curr if is_frame_literal(p, action_objs)}
        nxt_frame = {p for p in nxt if is_frame_literal(p, action_objs)}

        # Mode-dependent logic
        if mode == "consider_masking":
            # Only propagate predicates that are NOT in the masking of the next state
            next_state_idx = i + 1
            masked_literals = set()
            if next_state_idx < len(masking):
                masked_literals = {gp_to_gym(gp) for gp in masking[next_state_idx]}

            # Filter out masked literals from propagation candidates
            curr_frame_unmasked = {p for p in curr_frame if p not in masked_literals}
            to_add = curr_frame_unmasked - nxt_frame
            to_remove = nxt_frame - curr_frame_unmasked
        else:  # ignore_masking
            # Propagate regardless of masking
            to_add = curr_frame - nxt_frame
            to_remove = nxt_frame - curr_frame

        if to_add:
            nxt |= to_add
        if to_remove:
            nxt -= to_remove

        # Update mask of s' (index i+1): remove literals whose truth we fixed
        next_state_idx = i + 1
        if next_state_idx < len(masking) and (to_add or to_remove):
            fixed = to_add | to_remove
            masking[next_state_idx] = {gp for gp in masking[next_state_idx] if gp_to_gym(gp) not in fixed}

        out_steps.append({
            "step": i + 1,
            "current_state": {"literals": sorted(curr)},
            "ground_action": str(comp.grounded_action_call).strip("()"),
            "next_state": {"literals": sorted(nxt)},
            "frame_closure": {"added": sorted(to_add), "removed": sorted(to_remove)},
        })

        curr = nxt

    out_name = trajectory_path.stem + "_frame_closed"
    build_trajectory_file(out_steps, out_name, trajectory_path.parent)
    save_masking_info(masking_info_path.parent, out_name, masking)

    return (
        trajectory_path.parent / f"{out_name}.trajectory",
        masking_info_path.parent / f"{out_name}.masking_info",
    )


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


def replace_every_nth_state_with_ground_truth(
    trajectory_path: Union[str, Path],
    masking_info_path: Union[str, Path],
    json_trajectory_path: Union[str, Path],
    domain_path: Union[str, Path],
    n: int
) -> Tuple[Path, Path]:
    """
    Replace every n-th state in a trajectory with ground truth from JSON.

    Starting from the first state after initial (state 1), replaces every n-th state
    (states 1, n+1, 2n+1, ...) with the ground truth state from the _trajectory.json file.
    Also removes masking info for these states since they now have ground truth.

    The function respects the length of the input trajectory - if the trajectory is already
    truncated, only that many steps will be processed from the JSON file.

    Args:
        trajectory_path: Path to .trajectory file (possibly already truncated)
        masking_info_path: Path to .masking_info file
        json_trajectory_path: Path to _trajectory.json file with ground truth
        domain_path: Path to domain PDDL file
        n: Interval for replacement (e.g., n=3 means replace states 1, 4, 7, ...)

    Returns:
        Tuple of (new_trajectory_path, new_masking_info_path)
    """
    from src.utils.masking import load_masking_info, save_masking_info

    trajectory_path = Path(trajectory_path)
    masking_info_path = Path(masking_info_path)
    json_trajectory_path = Path(json_trajectory_path)
    domain_path = Path(domain_path)

    # Load current trajectory to determine its length
    domain: Domain = DomainParser(domain_path).parse_domain()
    parser = TrajectoryParser(domain)
    current_observation = parser.parse_trajectory(trajectory_path)
    num_steps = len(current_observation.components)

    # Load JSON ground truth data
    with open(json_trajectory_path, 'r') as f:
        gt_trajectory = json.load(f)

    # Load masking info
    # Note: masking_info has length = len(states) = len(steps) + 1
    # Index 0 is for initial state, index i is for state after step i-1
    masking_info = load_masking_info(masking_info_path, domain)
    masking_info = [set(m) for m in masking_info]  # Make a copy

    # Build new trajectory data from ground truth, but only for num_steps
    # gt_trajectory[i] has step i+1, current_state is state before action, next_state is state after
    new_trajectory_data = []

    for i in range(num_steps):
        if i >= len(gt_trajectory):
            break  # Safety check

        step_data = gt_trajectory[i]
        step_index = i + 1  # Steps are 1-indexed in the JSON

        # Determine which state to use for next_state of this step
        # State indices: init=0, after step 1 = 1, after step 2 = 2, ...
        # We want to replace states at indices 1, n+1, 2n+1, ...
        # That corresponds to steps 1, n+1, 2n+1, ...
        state_after_action_index = step_index

        # Check if this state should be replaced with ground truth
        # States to replace: 1, n+1, 2n+1, ... which is (k*n + 1) for k=0,1,2,...
        if (state_after_action_index - 1) % n == 0:
            # Use ground truth for next_state
            next_state_literals = step_data['next_state']['literals']
            # Clear masking for this state
            if state_after_action_index < len(masking_info):
                masking_info[state_after_action_index] = set()
        else:
            # Keep using JSON data (which might already have errors)
            next_state_literals = step_data['next_state']['literals']

        new_trajectory_data.append({
            'step': step_index,
            'current_state': step_data['current_state'],
            'ground_action': step_data['ground_action'],
            'next_state': {'literals': next_state_literals}
        })

    # Save new trajectory file
    problem_name = trajectory_path.stem
    out_name = f"{problem_name}_gt_every_{n}"
    build_trajectory_file(new_trajectory_data, out_name, trajectory_path.parent)

    # Save new masking info (only keep entries up to num_steps + 1)
    truncated_masking_info = masking_info[:num_steps + 1]
    save_masking_info(masking_info_path.parent, out_name, truncated_masking_info)

    return (
        trajectory_path.parent / f"{out_name}.trajectory",
        masking_info_path.parent / f"{out_name}.masking_info"
    )


def propagate_frame_axioms_selective(
    trajectory_path: Union[str, Path],
    masking_info_path: Union[str, Path],
    domain_path: Union[str, Path],
    gt_state_indices: Set[int],
    mode: str = "after_gt_only"
) -> Tuple[Path, Path]:
    """
    Apply frame axiom propagation selectively based on GT state locations.

    Args:
        trajectory_path: Path to .trajectory file
        masking_info_path: Path to .masking_info file
        domain_path: Path to domain PDDL file
        gt_state_indices: Set of state indices that are ground truth
        mode: "after_gt_only" - only apply frame axioms after GT states
              "all_states" - apply frame axioms after all states

    Returns:
        Tuple of (new_trajectory_path, new_masking_info_path)
    """
    from src.utils.masking import load_masking_info, save_masking_info

    if mode not in ["after_gt_only", "all_states"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'after_gt_only' or 'all_states'")

    if mode == "all_states":
        # Apply frame axioms to all states - use existing function
        return propagate_frame_axioms_in_trajectory(
            trajectory_path, masking_info_path, domain_path, mode="consider_masking"
        )

    # For "after_gt_only" mode, we need to selectively apply frame axioms
    trajectory_path = Path(trajectory_path)
    masking_info_path = Path(masking_info_path)
    domain_path = Path(domain_path)

    print(f"    [FA DEBUG] Parsing domain...")
    domain: Domain = DomainParser(domain_path).parse_domain()
    print(f"    [FA DEBUG] Parsing trajectory...")
    obs: Observation = TrajectoryParser(domain).parse_trajectory(trajectory_path)
    print(f"    [FA DEBUG] Loading masking info...")
    masking: List[Set[GroundedPredicate]] = load_masking_info(masking_info_path, domain)
    masking = [set(m) for m in masking]  # copy
    print(f"    [FA DEBUG] Trajectory has {len(obs.components)} components")

    def pddl_objs(pddl_lit: str) -> Set[str]:
        s = pddl_lit.strip()
        if s.startswith("(not"):
            s = s[len("(not"):].strip()
        s = s.strip("() ").split()
        return set(s[1:]) if len(s) > 1 else set()

    def gp_to_gym(gp: GroundedPredicate) -> str:
        if not gp.object_mapping:
            return f"{gp.name}()"
        args = ",".join(f"{v}:{gp.signature[k].name}" for k, v in gp.object_mapping.items())
        return f"{gp.name}({args})"

    def positive_gym_literals(state) -> Set[str]:
        return {
            gp_to_gym(p)
            for preds in state.state_predicates.values()
            for p in preds
            if p.is_positive
        }

    def is_frame_literal(gym_lit: str, action_objs: Set[str]) -> bool:
        pddl_pred_str = parse_gym_to_pddl_literal(gym_lit)
        objs = pddl_objs(pddl_pred_str)
        return bool(objs) and not objs.issubset(action_objs)

    # initial state is trusted
    curr = positive_gym_literals(obs.components[0].previous_state)
    out_steps = []

    for i, comp in enumerate(obs.components):
        nxt = positive_gym_literals(comp.next_state)
        action_objs = pddl_objs(str(comp.grounded_action_call))

        # State index of current state (before this transition)
        current_state_idx = i

        # Only apply frame axioms if current state is GT
        if current_state_idx in gt_state_indices:
            curr_frame = {p for p in curr if is_frame_literal(p, action_objs)}
            nxt_frame = {p for p in nxt if is_frame_literal(p, action_objs)}

            # Only propagate predicates that are NOT in the masking of the next state
            next_state_idx = i + 1
            masked_literals = set()
            if next_state_idx < len(masking):
                masked_literals = {gp_to_gym(gp) for gp in masking[next_state_idx]}

            # Filter out masked literals from propagation candidates
            curr_frame_unmasked = {p for p in curr_frame if p not in masked_literals}
            to_add = curr_frame_unmasked - nxt_frame
            to_remove = nxt_frame - curr_frame_unmasked

            if to_add:
                nxt |= to_add
            if to_remove:
                nxt -= to_remove

            # Update mask of s' (index i+1): remove literals whose truth we fixed
            if next_state_idx < len(masking):
                for fixed_lit in (to_add | to_remove):
                    masking[next_state_idx] -= {gp for gp in masking[next_state_idx]
                                                if gp_to_gym(gp) == fixed_lit}

        # Build output step with updated next_state
        nxt_grounded = [parse_gym_to_pddl_literal(p) for p in nxt]
        out_steps.append((comp.grounded_action_call, nxt_grounded))
        curr = nxt

    # Save modified trajectory
    problem_name = trajectory_path.stem.split('_frame_axioms')[0]  # Remove previous suffix
    out_name = f"{problem_name}_frame_axioms"
    observation_to_trajectory_file(obs, trajectory_path.parent / f"{out_name}.trajectory")

    # Save modified masking
    save_masking_info(masking_info_path.parent, out_name, masking)

    return (
        trajectory_path.parent / f"{out_name}.trajectory",
        masking_info_path.parent / f"{out_name}.masking_info"
    )


def inject_gt_states_by_percentage(
    trajectory_path: Union[str, Path],
    masking_info_path: Union[str, Path],
    json_trajectory_path: Union[str, Path],
    domain_path: Union[str, Path],
    gt_rate: int
) -> Tuple[Path, Path, Set[int]]:
    """
    Inject ground truth states at percentage-based intervals throughout the trajectory.

    Starting from the initial state (which is always GT), injects GT states evenly spread
    throughout the trajectory based on the specified percentage.

    Args:
        trajectory_path: Path to .trajectory file
        masking_info_path: Path to .masking_info file
        json_trajectory_path: Path to _trajectory.json file with ground truth
        domain_path: Path to domain PDDL file
        gt_rate: Percentage of states to inject as GT (0-100)
                gt_rate=0 means only initial state is GT
                gt_rate=10 means 10% of states are GT (evenly spread)

    Returns:
        Tuple of (new_trajectory_path, new_masking_info_path)

    Example:
        For a 100-state trajectory with gt_rate=10:
        - Total GT states: 10
        - Interval: 100 / 10 = 10
        - GT state indices: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90
    """
    import math
    from src.utils.masking import load_masking_info, save_masking_info

    trajectory_path = Path(trajectory_path)
    masking_info_path = Path(masking_info_path)
    json_trajectory_path = Path(json_trajectory_path)
    domain_path = Path(domain_path)

    # Load current trajectory to determine its length
    domain: Domain = DomainParser(domain_path).parse_domain()
    parser = TrajectoryParser(domain)
    current_observation = parser.parse_trajectory(trajectory_path)
    num_steps = len(current_observation.components)
    num_states = num_steps + 1  # Including initial state

    # Load JSON ground truth data
    with open(json_trajectory_path, 'r') as f:
        gt_trajectory = json.load(f)

    # Load masking info
    masking_info = load_masking_info(masking_info_path, domain)
    masking_info = [set(m) for m in masking_info]  # Make a copy

    # Calculate which states should be GT
    gt_state_indices = set()
    gt_state_indices.add(0)  # Initial state is always GT

    if gt_rate > 0:
        # Calculate total number of GT states needed
        num_gt_states = max(1, math.ceil(num_states * gt_rate / 100.0))

        if num_gt_states > 1:
            # Calculate interval for even spacing
            interval = num_states / num_gt_states

            # Generate evenly spaced GT state indices
            for i in range(num_gt_states):
                idx = int(i * interval)
                if idx < num_states:
                    gt_state_indices.add(idx)

    # Build new trajectory data
    new_trajectory_data = []

    for i in range(num_steps):
        if i >= len(gt_trajectory):
            break  # Safety check

        step_data = gt_trajectory[i]
        step_index = i + 1  # Steps are 1-indexed in the JSON
        state_after_action_index = step_index

        # Check if this state should be replaced with ground truth
        if state_after_action_index in gt_state_indices:
            # Use ground truth for next_state
            next_state_literals = step_data['next_state']['literals']
            # Clear masking for this state
            if state_after_action_index < len(masking_info):
                masking_info[state_after_action_index] = set()
        else:
            # Keep using JSON data (which might already have errors)
            next_state_literals = step_data['next_state']['literals']

        new_trajectory_data.append({
            'step': step_index,
            'current_state': step_data['current_state'],
            'ground_action': step_data['ground_action'],
            'next_state': {'literals': next_state_literals}
        })

    # Save new trajectory file
    problem_name = trajectory_path.stem.split('_gtrate')[0]  # Remove any previous gtrate suffix
    out_name = f"{problem_name}_gtrate{gt_rate}" if gt_rate > 0 else problem_name
    build_trajectory_file(new_trajectory_data, out_name, trajectory_path.parent)

    # Save new masking info
    save_masking_info(masking_info_path.parent, out_name, masking_info)

    return (
        trajectory_path.parent / f"{out_name}.trajectory",
        masking_info_path.parent / f"{out_name}.masking_info",
        gt_state_indices  # Return the set of GT state indices for frame axiom application
    )


if __name__ == "__main__":
    path_to_change = "/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/data/hanoi/multi_problem_06-12-2025T13:58:24__model=gpt-5.1__steps=100__planner/training/trajectories/problem6/problem6"
    domain_path = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/domains/hanoi/hanoi.pddl")

    propagate_frame_axioms_in_trajectory(
        Path(f"{path_to_change}.trajectory"),
        Path(f"{path_to_change}.masking_info"),
        domain_path
    )