"""PDDLGym format conversion and visual-facts / image-object translation."""

import re
from typing import Set

from pddlgym.core import PDDLEnv
from pddlgym.parser import Operator

from src.utils.containers import shrink_whitespaces


# ============================================================================
# PDDLGym environment helpers
# ============================================================================

def set_problem_by_name(pddl_env: PDDLEnv, problem_name: str):
    """Fix a problem for the environment by its name."""
    if not problem_name.endswith('.pddl'):
        problem_name += '.pddl'
    for idx, prob in enumerate(pddl_env.problems):
        if prob.problem_fname.endswith(problem_name):
            pddl_env.fix_problem_index(idx)
            return
    raise ValueError(f"Problem '{problem_name}' not found in the problem list of the environment you provided.")


def ground_action(operator: Operator, assignment: dict) -> str:
    action_name: str = operator.name
    grounded_params = [str(assignment[lifted_param]) for lifted_param in operator.params]
    return f"{action_name}({', '.join(grounded_params)})"


# ============================================================================
# Format conversion: PDDLGym ↔ PDDL strings
# ============================================================================

def parse_gym_to_pddl_literal(literal: str) -> str:
    """Convert 'on(a:block,b:block)' -> '(on a b)'"""
    name, args = literal.split('(')
    args = args.rstrip(')')
    if not args:
        return f"({name})"
    args_clean = [arg.split(':')[0] for arg in args.split(',')]
    parsed_literal = f"({name} {' '.join(args_clean)})"
    return shrink_whitespaces(parsed_literal)


def parse_gym_to_pddl_ground_action(ground_action_str: str) -> str:
    """Convert 'put-down(a:block, robot:robot)' -> '(putdown a robot)'"""
    if '(' in ground_action_str:
        name, args = ground_action_str.split('(')
        args = args.rstrip(')')
        args_clean = [arg.split(':')[0] for arg in args.split(',')]
        parsed_action = f"({name} {' '.join(args_clean)})"
    else:
        parsed_action = f"({ground_action_str})"
    return shrink_whitespaces(parsed_action)


def multi_replace_predicate(p: str, mapping: dict[str, str]) -> str:
    keys = sorted(mapping.keys(), key=len, reverse=True)
    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, keys)) + r')\b')
    return pattern.sub(lambda m: mapping[m.group(0)], p)


# ============================================================================
# Visual facts / image-object translation
# ============================================================================

def translate_pddlgym_state_to_image_predicates(
    pddlgym_state_literals: list[str],
    imaged_obj_to_gym_obj_name: dict[str, str],
) -> list[str]:
    """Translate PDDLGym literal object names to image object names."""
    gym_to_image = {gym: img for img, gym in imaged_obj_to_gym_obj_name.items()}
    translated = []

    for literal in pddlgym_state_literals:
        m = re.match(r'([a-zA-Z0-9_-]+)\((.*)\)', literal)
        if not m:
            continue

        pred, args_str = m.groups()
        if not args_str.strip():
            translated.append(literal)
            continue

        args = []
        for arg in map(str.strip, args_str.split(',')):
            if ':' not in arg:
                args.append(arg)
                continue
            name, typ = (p.strip() for p in arg.split(':', 1))
            name = gym_to_image.get(name, name)
            args.append(f"{name}:{typ}")

        translated.append(f"{pred}({','.join(args)})")

    return translated


def extract_objects_from_pddlgym_state(
    pddlgym_state_objects: list[str],
    imaged_obj_to_gym_obj_name: dict[str, str],
) -> Set[str]:
    """Extract objects from a PDDLGym state and translate to image object names."""
    gym2img = {gym: img for img, gym in imaged_obj_to_gym_obj_name.items()}

    def translate(obj_str: str) -> str:
        if ":" not in obj_str:
            return obj_str
        name, typ = (s.strip() for s in obj_str.split(":", 1))
        return f"{gym2img.get(name, name)}:{typ}"

    return {translate(o) for o in pddlgym_state_objects}
