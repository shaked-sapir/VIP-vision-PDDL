"""Unit tests for the unified_planning -> eval-schema bridge.

These run against a real parsed problem rather than fakes, because the thing
under test is agreement with ``unified_planning``'s own types.

    ./venv11/bin/python -m pytest src/trace_generation/test_up_bridge.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.trace_generation.sources import parse_problem
from src.trace_generation.up_bridge import (
    action_to_eval_string,
    fluent_to_literal,
    ground_fluents,
    object_types,
    state_to_trace_state,
    typed_objects,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLOCKS_DOMAIN = PROJECT_ROOT / "src" / "domains" / "blocks" / "blocks.pddl"
BLOCKS_PROBLEM = (PROJECT_ROOT / "src" / "domains" / "blocks" / "problems"
                  / "problem1" / "problem1.pddl")


@pytest.fixture(scope="module")
def problem():
    """The parsed blocksworld problem1: four blocks, all on the table."""
    return parse_problem(BLOCKS_DOMAIN, BLOCKS_PROBLEM)


# ── objects and types ────────────────────────────────────────────────────

def test_object_types_maps_every_object_to_its_pddl_type(problem):
    assert object_types(problem) == {"a": "block", "b": "block",
                                     "c": "block", "d": "block"}


def test_typed_objects_is_the_sorted_eval_schema_universe(problem):
    assert typed_objects(problem) == ("a:block", "b:block", "c:block", "d:block")


# ── literals ─────────────────────────────────────────────────────────────

def test_ground_fluents_covers_the_whole_fluent_space(problem):
    literals = {fluent_to_literal(f, object_types(problem))
                for f in ground_fluents(problem)}
    # on/2 over 4 blocks (incl. self), clear/1, ontable/1, holding/1, handempty/0.
    assert "on(a:block,b:block)" in literals
    assert "clear(a:block)" in literals
    assert "handempty()" in literals
    assert len(literals) == 16 + 4 + 4 + 4 + 1


def test_literal_arguments_are_comma_separated_without_a_space(problem):
    fluent = next(f for f in ground_fluents(problem)
                  if f.fluent().name == "on" and str(f.args[0]) == "a"
                  and str(f.args[1]) == "b")
    assert fluent_to_literal(fluent, object_types(problem)) == "on(a:block,b:block)"


def test_nullary_fluent_renders_with_empty_parens(problem):
    fluent = next(f for f in ground_fluents(problem)
                  if f.fluent().name == "handempty")
    assert fluent_to_literal(fluent, object_types(problem)) == "handempty()"


def test_unknown_object_fails_loudly_instead_of_writing_a_bad_literal(problem):
    fluent = next(f for f in ground_fluents(problem)
                  if f.fluent().name == "clear")
    with pytest.raises(KeyError, match="not declared"):
        fluent_to_literal(fluent, {})


# ── actions ──────────────────────────────────────────────────────────────

def test_action_arguments_are_comma_space_separated(problem):
    from unified_planning.plans import ActionInstance

    instance = ActionInstance(problem.action("stack"),
                              [problem.object("a"), problem.object("b")])
    assert action_to_eval_string(instance, object_types(problem)) == \
        "stack(a:block, b:block)"


def test_unary_action_renders_with_one_typed_argument(problem):
    from unified_planning.plans import ActionInstance

    instance = ActionInstance(problem.action("pick_up"), [problem.object("a")])
    assert action_to_eval_string(instance, object_types(problem)) == "pick_up(a:block)"


# ── states ───────────────────────────────────────────────────────────────

def test_initial_state_keeps_true_fluents_and_drops_false_ones(problem):
    from unified_planning.shortcuts import SequentialSimulator

    fluents = ground_fluents(problem)
    types, objects = object_types(problem), typed_objects(problem)
    with SequentialSimulator(problem=problem) as simulator:
        state = simulator.get_initial_state()
        trace_state = state_to_trace_state(state, fluents, types, objects)

    assert set(trace_state.literals) == {
        "clear(a:block)", "clear(b:block)", "clear(c:block)", "clear(d:block)",
        "ontable(a:block)", "ontable(b:block)", "ontable(c:block)",
        "ontable(d:block)", "handempty()"}
    assert trace_state.objects == objects


def test_literals_are_sorted(problem):
    from unified_planning.shortcuts import SequentialSimulator

    fluents = ground_fluents(problem)
    with SequentialSimulator(problem=problem) as simulator:
        trace_state = state_to_trace_state(
            simulator.get_initial_state(), fluents,
            object_types(problem), typed_objects(problem))

    assert list(trace_state.literals) == sorted(trace_state.literals)


def test_a_successor_state_is_read_in_full_even_when_it_stores_a_delta(problem):
    """A UPState that keeps only its diff from a parent must still read whole.

    ``MAX_ANCESTORS`` is forced back to its stock value so the successor is a
    genuine delta; reading it via the state's own mapping would return four
    entries instead of the six fluents that actually hold.
    """
    from unified_planning.model import UPState
    from unified_planning.plans import ActionInstance
    from unified_planning.shortcuts import SequentialSimulator

    fluents = ground_fluents(problem)
    types, objects = object_types(problem), typed_objects(problem)

    original = UPState.MAX_ANCESTORS
    UPState.MAX_ANCESTORS = 20
    try:
        with SequentialSimulator(problem=problem) as simulator:
            state = simulator.get_initial_state()
            successor = simulator.apply(
                state, ActionInstance(problem.action("pick_up"),
                                      [problem.object("a")]))
            assert len(successor._values) < len(fluents), \
                "expected a delta state; UP's representation changed"
            trace_state = state_to_trace_state(successor, fluents, types, objects)
    finally:
        UPState.MAX_ANCESTORS = original

    assert set(trace_state.literals) == {
        "holding(a:block)",
        "clear(b:block)", "clear(c:block)", "clear(d:block)",
        "ontable(b:block)", "ontable(c:block)", "ontable(d:block)"}
