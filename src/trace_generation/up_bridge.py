"""Translate ``unified_planning`` states and actions into the eval schema.

The schema's spelling lives in :mod:`src.trace_generation.eval_schema`; this
module only decides which ``unified_planning`` object supplies each part of it.

State conversion drops fluents that evaluate to false, per the closed-world
assumption the ``.trajectory`` format is built on.

``unified_planning`` is imported lazily by the callers in ``sources.py``; the
type annotations here are strings so this module can be imported without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Sequence, Tuple

from src.trace_generation.eval_schema import (
    render_action,
    render_literal,
    typed_universe,
)
from src.trace_generation.trace_step import TraceState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from unified_planning.model import FNode, Problem, UPState
    from unified_planning.plans import ActionInstance


def object_types(problem: "Problem") -> Dict[str, str]:
    """Map each problem object's name to its PDDL type name."""
    return {obj.name: obj.type.name for obj in problem.all_objects}


def typed_objects(problem: "Problem") -> Tuple[str, ...]:
    """The problem's object universe as sorted eval-schema ``name:type`` strings."""
    return typed_universe(object_types(problem))


def ground_fluents(problem: "Problem") -> Tuple["FNode", ...]:
    """Every ground fluent expression of the problem, in a stable order.

    A ``UPState`` stores only the values that differ from its ancestors, so a
    full state must be read by evaluating each fluent rather than by inspecting
    the state's own mapping.
    """
    return tuple(problem.initial_values.keys())


def fluent_to_literal(fluent: "FNode", types: Dict[str, str]) -> str:
    """Render one ground fluent expression as an eval-schema literal."""
    return render_literal(fluent.fluent().name,
                          [str(arg) for arg in fluent.args], types)


def state_to_trace_state(
    state: "UPState",
    fluents: Sequence["FNode"],
    types: Dict[str, str],
    objects: Tuple[str, ...],
) -> TraceState:
    """Convert a ``UPState`` into a :class:`TraceState` of its true fluents.

    Args:
        state: The state to read.
        fluents: Every ground fluent of the problem, from :func:`ground_fluents`.
        types: Object name to type name, from :func:`object_types`.
        objects: The problem's object universe, from :func:`typed_objects`.

    Returns:
        A :class:`TraceState` holding the sorted true fluents. False fluents are
        dropped and recovered downstream by the closed-world assumption.
    """
    literals = sorted(
        fluent_to_literal(fluent, types)
        for fluent in fluents
        if state.get_value(fluent).is_true()
    )
    return TraceState(literals=tuple(literals), objects=objects)


def action_to_eval_string(instance: "ActionInstance", types: Dict[str, str]) -> str:
    """Render an ``ActionInstance`` as an eval-schema ground action string."""
    return render_action(instance.action.name,
                         [str(param) for param in instance.actual_parameters],
                         types)
