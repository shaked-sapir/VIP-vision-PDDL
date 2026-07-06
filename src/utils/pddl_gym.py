"""PDDLGym format conversion and visual-facts / image-object translation."""

import re
from typing import Callable, List, Set, Tuple

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
    """Convert 'put-down(a:block, robot:robot)' -> '(put-down a robot)'"""
    if '(' in ground_action_str:
        name, args = ground_action_str.split('(')
        args = args.rstrip(')')
        args_clean = [arg.split(':')[0] for arg in args.split(',')]
        parsed_action = f"({name} {' '.join(args_clean)})"
    else:
        parsed_action = f"({ground_action_str})"
    return shrink_whitespaces(parsed_action)


def multi_replace_predicate(p: str, mapping: dict[str, str]) -> str:
    if not mapping:
        return p
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


# ============================================================================
# Problem-file schema translation (gym schema → eval schema)
# ============================================================================

# A literal transform maps (predicate_name, args) → (predicate_name, args).
LiteralTransform = Callable[[str, List[str]], Tuple[str, List[str]]]
# An object-type transform maps an object name → its eval-schema type.
ObjectTypeTransform = Callable[[str], str]


def _translate_objects_block(obj_block: str, object_type_fn: ObjectTypeTransform) -> str:
    """Re-type every object in a `(:objects ...)` block body, one per line."""
    names: List[str] = []
    for segment in re.split(r'\n', obj_block):
        segment = segment.strip()
        if not segment or segment.startswith(';'):
            continue
        # Drop any existing "- type" suffix; keep the bare object names.
        head = re.split(r'\s*-\s*', segment)[0]
        names.extend(n for n in head.split() if n)
    return "\n".join(f"\t{name} - {object_type_fn(name)}" for name in names)


def _translate_literal(literal_body: str, literal_fn: LiteralTransform) -> str:
    """Translate a single `pred a b` literal body via `literal_fn`."""
    parts = literal_body.split()
    if not parts:
        return literal_body
    pred_name, args = parts[0], parts[1:]
    new_pred, new_args = literal_fn(pred_name, args)
    return " ".join([new_pred, *new_args]).strip()


def _translate_literal_region(text: str, literal_fn: LiteralTransform) -> str:
    """Translate every `(pred ...)` literal in a region of PDDL text."""
    return re.sub(
        r'\(([a-zA-Z][\w-]*(?:\s+[\w:-]+)*)\)',
        lambda m: f"({_translate_literal(m.group(1), literal_fn)})",
        text,
    )


def translate_problem_pddl_text(
    pddl_text: str,
    object_type_fn: ObjectTypeTransform,
    literal_fn: LiteralTransform,
) -> str:
    """Rewrite a PDDL problem's objects/init/goal from one schema to another.

    Preserves the problem's own objects, init, and goal — only the object
    *types* and the predicate names/args are rewritten, using the two
    domain-supplied callables. Keeps schema knowledge in the caller (the domain
    handler) while sharing the text manipulation here.

    Args:
        pddl_text: Raw problem PDDL text (gym schema).
        object_type_fn: Maps an object name to its eval-schema type.
        literal_fn: Maps (predicate_name, args) to the rewritten pair.

    Returns:
        The rewritten PDDL text.
    """
    # 1. Re-type the (:objects ...) block.
    text = re.sub(
        r'\(:objects\s+(.*?)\)',
        lambda m: f"(:objects\n{_translate_objects_block(m.group(1), object_type_fn)}\n)",
        pddl_text, flags=re.DOTALL,
    )

    # 2. Rewrite literals inside the (:init ...) block (up to (:goal).
    text = re.sub(
        r'\(:init\s+(.*?)\)\s*(?=\(:goal)',
        lambda m: f"(:init {_translate_literal_region(m.group(1), literal_fn)})\n  ",
        text, flags=re.DOTALL,
    )

    # 3. Rewrite literals inside the (:goal ...) block (runs to problem end).
    text = re.sub(
        r'\(:goal\s+(.*)\)\s*\)\s*$',
        lambda m: f"(:goal {_translate_literal_region(m.group(1), literal_fn)}))\n",
        text, flags=re.DOTALL,
    )

    return text
