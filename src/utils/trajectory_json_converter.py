"""
Convert a PDDL .trajectory file + problem.pddl into the trajectory JSON format
expected by the LLM-based object detector and fluent classifier few-shot pipeline.

Usage:
    from src.utils.trajectory_json_converter import convert_trajectory_to_json

    convert_trajectory_to_json(
        trajectory_path=Path("problem0.trajectory"),
        problem_path=Path("problem0.pddl"),
        output_path=Path("problem0_trajectory.json"),   # optional, defaults to <trajectory_dir>/<problem_name>_trajectory.json
    )

The .trajectory format:
    (
    (:init (pred1 arg1 arg2) (pred2 arg3) ...)
    (operator: (action_name arg1 arg2 ...))
    (:state (pred1 arg1 arg2) (pred2 arg3) ...)
    ...
    )

The output trajectory.json format:
    [
      {
        "step": 1,
        "current_state": {"literals": ["pred1(arg1:type1,arg2:type2)", ...], "objects": ["arg1:type1", ...]},
        "ground_action": "action_name(arg1:type1,arg2:type2)",
        "next_state": {"literals": [...], "objects": [...]}
      },
      ...
    ]
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional


def _parse_problem_objects(problem_path: Path) -> Dict[str, str]:
    """Parse a PDDL problem file to extract the object→type mapping.

    Returns:
        Dict mapping object name to its type, e.g. {"t1": "truck", "d1": "depot"}.
    """
    text = problem_path.read_text()

    # Extract the (:objects ...) block
    objects_match = re.search(r'\(:objects\s+(.*?)\)', text, re.DOTALL)
    if not objects_match:
        raise ValueError(f"No (:objects ...) block found in {problem_path}")

    obj_block = objects_match.group(1).strip()
    obj_map: Dict[str, str] = {}

    # Each line/segment is: name1 name2 ... - type
    for segment in re.split(r'\n', obj_block):
        segment = segment.strip()
        if not segment or segment.startswith(';'):
            continue
        # Match "name1 name2 ... - type"
        match = re.match(r'([\w\s]+)\s*-\s*(\w+)', segment)
        if match:
            names = match.group(1).strip().split()
            obj_type = match.group(2).strip()
            for name in names:
                obj_map[name] = obj_type

    return obj_map


def _type_predicate(pred_str: str, obj_map: Dict[str, str]) -> str:
    """Convert an untyped predicate string to typed format.

    Example: "at-truck t1 d2" → "at-truck(t1:truck,d2:depot)"
    """
    parts = pred_str.strip().split()
    pred_name = parts[0]
    args = parts[1:]

    if not args:
        # Zero-arity predicate
        return f"{pred_name}()"

    typed_args = []
    for arg in args:
        obj_type = obj_map.get(arg)
        if obj_type:
            typed_args.append(f"{arg}:{obj_type}")
        else:
            typed_args.append(arg)

    return f"{pred_name}({','.join(typed_args)})"


def _type_action(action_str: str, obj_map: Dict[str, str]) -> str:
    """Convert an untyped action string to typed format.

    Example: "drive t1 d2 d1" → "drive(t1:truck,d2:depot,d1:depot)"
    """
    return _type_predicate(action_str, obj_map)


def _parse_trajectory_file(trajectory_path: Path) -> tuple[List[str], List[str], List[List[str]]]:
    """Parse a .trajectory file into init predicates, actions, and state predicate lists.

    Returns:
        (init_preds, actions, states) where:
        - init_preds: list of untyped predicate strings from :init
        - actions: list of untyped action strings from (operator: ...)
        - states: list of lists of untyped predicate strings from :state blocks
    """
    text = trajectory_path.read_text().strip()

    # Extract all predicate blocks: (:init ...) and (:state ...)
    # Each block contains nested parens like (:init (clear a) (on a b) ...),
    # so we match one-or-more (...) groups inside.
    state_pattern = re.compile(r'\(:(init|state)\s+((?:\([^)]+\)\s*)+)\)', re.DOTALL)
    # Extract all action blocks: (operator: (action_name args...))
    action_pattern = re.compile(r'\(operator:\s*\(([^)]+)\)\)')

    state_matches = state_pattern.findall(text)
    action_matches = action_pattern.findall(text)

    if not state_matches:
        raise ValueError(f"No :init or :state blocks found in {trajectory_path}")

    def extract_predicates(block: str) -> List[str]:
        """Extract individual predicates from a state block string."""
        # Match (pred_name arg1 arg2 ...) patterns
        return re.findall(r'\(([^()]+)\)', block)

    # First match is :init, rest are :state
    init_preds = extract_predicates(state_matches[0][1])
    states = [extract_predicates(m[1]) for m in state_matches[1:]]

    return init_preds, action_matches, states


def convert_trajectory_to_json(
    trajectory_path: Path,
    problem_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """Convert a .trajectory file + problem.pddl into trajectory JSON.

    Args:
        trajectory_path: Path to the .trajectory file.
        problem_path: Path to the problem.pddl file (for object typing).
        output_path: Where to write the JSON. Defaults to
            <trajectory_dir>/<problem_name>_trajectory.json.

    Returns:
        Path to the written JSON file.
    """
    obj_map = _parse_problem_objects(problem_path)
    init_preds, actions, states = _parse_trajectory_file(trajectory_path)

    # Build the objects list (typed)
    objects_list = [f"{name}:{typ}" for name, typ in sorted(obj_map.items())]

    # Type all predicates
    typed_init = [_type_predicate(p, obj_map) for p in init_preds]
    typed_states = [[_type_predicate(p, obj_map) for p in s] for s in states]
    typed_actions = [_type_action(a, obj_map) for a in actions]

    if len(typed_actions) != len(typed_states):
        raise ValueError(
            f"Mismatch: {len(typed_actions)} actions but {len(typed_states)} "
            f"subsequent states in {trajectory_path}"
        )

    # Build trajectory steps
    trajectory_json: List[dict] = []
    current_literals = typed_init

    for i, (action, next_state_lits) in enumerate(zip(typed_actions, typed_states)):
        step = {
            "step": i + 1,
            "current_state": {
                "literals": current_literals,
                "objects": objects_list,
            },
            "ground_action": action,
            "next_state": {
                "literals": next_state_lits,
                "objects": objects_list,
            },
        }
        trajectory_json.append(step)
        current_literals = next_state_lits

    # Write output
    if output_path is None:
        problem_name = trajectory_path.stem.split("_gtrate")[0]  # strip gtrate suffix if present
        output_path = trajectory_path.parent / f"{problem_name}_trajectory.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trajectory_json, indent=4))

    return output_path
