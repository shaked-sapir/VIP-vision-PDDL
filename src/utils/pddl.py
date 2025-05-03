import os
from pathlib import Path

from pddlgym.core import PDDLEnv
from pddlgym.parser import Operator
from typing import List

from src.utils.containers import shrink_whitespaces


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


def build_trajectory_file(trajectory: List[dict], output_path: Path) -> None:
    output_path = os.path.join(output_path, "trajectory.trajectory")

    trajectory_lines = ["("]  # the opener of the trajectory file

    # Step 0: Write the initial state from current_state of the first entry
    init_literals = trajectory[0]['current_state']['literals']
    init_literals_parsed = ' '.join(parse_gym_to_pddl_literal(lit) for lit in init_literals)
    trajectory_lines.append(f"(:init {init_literals_parsed})")

    # Step 1: Write the first operator (from first entry)
    ground_action = trajectory[0]['ground_action']
    ground_action_parsed = parse_gym_to_pddl_ground_action(ground_action)
    trajectory_lines.append(f"(operator: {ground_action_parsed})")

    # Then continue: For each NEXT state and NEXT action
    for i in range(1, len(trajectory)):
        step_info = trajectory[i]

        # Write the state
        current_literals = step_info['current_state']['literals']
        current_literals_parsed = ' '.join(parse_gym_to_pddl_literal(lit) for lit in current_literals)
        trajectory_lines.append(f"(:state {current_literals_parsed})")

        # Write the operator
        ground_action = step_info['ground_action']
        ground_action_parsed = parse_gym_to_pddl_ground_action(ground_action)
        trajectory_lines.append(f"(operator: {ground_action_parsed})")

    # Finally write the last :state after the last action
    final_state_literals = trajectory[-1]['next_state']['literals']
    final_state_literals_parsed = ' '.join(parse_gym_to_pddl_literal(lit) for lit in final_state_literals)
    trajectory_lines.append(f"(:state {final_state_literals_parsed})")

    trajectory_lines.append(")")  # the closer of the trajectory file

    # Save to file
    with open(output_path, "w") as f:
        f.write('\n'.join(trajectory_lines))

    print(f"Trajectory saved to {output_path}")
