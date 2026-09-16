"""Trace files in the grammar the Lamanna learners parse.

One state per line, ``(:state (p a b) (not (q c)) ...)``; one action per line,
``(:action (move a b))``; states and actions alternate, starting and ending
with a state. An atom absent from a state is unknown, so a masked fluent is
simply not written, whichever sign it carries underneath.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from pddl_plus_parser.models import Observation, State

from src.utils.pddl_state import get_state_unmasked_predicates

HEADER = "(:trajectory"
FOOTER = ")"

_SPACE_BEFORE_PAREN = re.compile(r"\s+\)")
_RUNS_OF_SPACE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """``text`` with single spaces and none before a closing paren."""
    return _SPACE_BEFORE_PAREN.sub(")", _RUNS_OF_SPACE.sub(" ", text.strip()))


def state_literals(state: State) -> List[str]:
    """The literals a partial state exposes, sorted: unmasked atoms with their sign."""
    return sorted(
        _normalise(predicate.untyped_representation)
        for predicate in get_state_unmasked_predicates(state)
    )


def _states(observation: Observation) -> List[State]:
    return [observation.components[0].previous_state] + [
        component.next_state for component in observation.components
    ]


def format_trajectory(observation: Observation) -> str:
    """The trace text for ``observation``, actions between consecutive states."""
    states = _states(observation)
    lines = [HEADER]
    for index, state in enumerate(states):
        lines.append(f"(:state {' '.join(state_literals(state))})")
        if index < len(observation.components):
            call = _normalise(str(observation.components[index].grounded_action_call))
            lines.append(f"(:action {call})")
    lines.append(FOOTER)
    return "\n\n".join(lines) + "\n"


def write_partial_trace(observation: Observation, out_path: Path) -> Path:
    """Write ``observation`` as a trace file at ``out_path`` and return the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(format_trajectory(observation))
    return out_path
