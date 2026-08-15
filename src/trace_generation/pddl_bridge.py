"""Translate ``pddl_plus_parser`` states and actions into the eval schema.

The mirror of :mod:`src.trace_generation.up_bridge`: same output spelling, from
a parsed ``.trajectory`` instead of from a simulator.

A parsed state already holds only the predicates the file listed, so the
closed-world assumption is applied by keeping the positive ones and dropping any
explicit negation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

from src.trace_generation.eval_schema import (
    render_action,
    render_literal,
    typed_universe,
)
from src.trace_generation.trace_step import TraceState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pddl_plus_parser.models import (
        ActionCall,
        Domain,
        GroundedPredicate,
        Problem,
        State,
    )


def object_types(problem: "Problem", domain: "Domain") -> Dict[str, str]:
    """Map every object name to its PDDL type name.

    The domain's ``:constants`` are included; ``ProblemParser`` leaves them out
    of ``problem.objects`` even though a trajectory may mention them.
    """
    types = {name: obj.type.name for name, obj in domain.constants.items()}
    types.update({name: obj.type.name for name, obj in problem.objects.items()})
    return types


def typed_objects(problem: "Problem", domain: "Domain") -> Tuple[str, ...]:
    """The problem's object universe as sorted eval-schema ``name:type`` strings."""
    return typed_universe(object_types(problem, domain))


def predicate_to_literal(predicate: "GroundedPredicate",
                         types: Dict[str, str]) -> str:
    """Render one grounded predicate as an eval-schema literal.

    The argument order is read off ``untyped_representation``, which is the
    predicate's ordered PDDL form.
    """
    return render_literal(predicate.name,
                          predicate.untyped_representation.strip("()").split()[1:],
                          types)


def state_to_trace_state(state: "State", types: Dict[str, str],
                         objects: Tuple[str, ...]) -> TraceState:
    """Convert a parsed ``State`` into a :class:`TraceState` of its true literals.

    Args:
        state: The state to read.
        types: Object name to type name, from :func:`object_types`.
        objects: The problem's object universe, from :func:`typed_objects`.

    Returns:
        A :class:`TraceState` holding the sorted positive literals.
    """
    literals = sorted(
        predicate_to_literal(predicate, types)
        for predicates in state.state_predicates.values()
        for predicate in predicates
        if predicate.is_positive
    )
    return TraceState(literals=tuple(literals), objects=objects)


def action_call_to_eval_string(call: "ActionCall", types: Dict[str, str]) -> str:
    """Render an ``ActionCall`` as an eval-schema ground action string."""
    return render_action(call.name, list(call.parameters), types)
