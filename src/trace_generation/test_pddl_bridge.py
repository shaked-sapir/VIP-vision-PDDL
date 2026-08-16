"""Unit tests for the pddl_plus_parser -> eval-schema bridge.

These run against really parsed domains and problems rather than fakes, because
the thing under test is agreement with ``pddl_plus_parser``'s own types.

    ./venv11/bin/python -m pytest src/trace_generation/test_pddl_bridge.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import GroundedPredicate, State

from src.trace_generation.pddl_bridge import (
    action_call_to_eval_string,
    object_types,
    predicate_to_literal,
    state_to_trace_state,
    typed_objects,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLOCKS_DOMAIN = PROJECT_ROOT / "src" / "domains" / "blocks" / "blocks.pddl"
BLOCKS_PROBLEM = (PROJECT_ROOT / "src" / "domains" / "blocks" / "problems"
                  / "problem1" / "problem1.pddl")
GRIPPER_DOMAIN = PROJECT_ROOT / "src" / "domains" / "gripper" / "gripper.pddl"
GRIPPER_PROBLEM = (PROJECT_ROOT / "src" / "domains" / "gripper" / "problems"
                   / "problem0" / "problem0.pddl")

# A domain whose :constants the problem's :objects does not repeat.
CONSTANTS_DOMAIN = """(define (domain toy)
 (:requirements :strips :typing)
 (:types block table)
 (:constants tbl - table)
 (:predicates (on ?x - block ?y - table) (clear ?x - block))
 (:action put :parameters (?x - block ?y - table)
   :precondition (and (clear ?x)) :effect (and (on ?x ?y))))
"""
CONSTANTS_PROBLEM = """(define (problem p0) (:domain toy)
 (:objects a b - block)
 (:init (clear a) (clear b))
 (:goal (and (on a tbl))))
"""


def _parse(domain_file: Path, problem_file: Path):
    domain = DomainParser(domain_file, partial_parsing=False).parse_domain()
    return domain, ProblemParser(problem_file, domain).parse_problem()


@pytest.fixture(scope="module")
def blocks():
    """blocksworld problem1: four blocks, all on the table, and handempty."""
    return _parse(BLOCKS_DOMAIN, BLOCKS_PROBLEM)


@pytest.fixture(scope="module")
def gripper():
    """gripper problem0: six balls, two rooms, two grippers."""
    return _parse(GRIPPER_DOMAIN, GRIPPER_PROBLEM)


@pytest.fixture
def with_constants(tmp_path):
    """A domain declaring ``tbl`` as a constant the problem never re-declares."""
    (tmp_path / "toy.pddl").write_text(CONSTANTS_DOMAIN)
    (tmp_path / "p0.pddl").write_text(CONSTANTS_PROBLEM)
    return _parse(tmp_path / "toy.pddl", tmp_path / "p0.pddl")


# ── objects and types ────────────────────────────────────────────────────

def test_object_types_maps_every_object_to_its_pddl_type(blocks):
    domain, problem = blocks
    assert object_types(problem, domain) == {"a": "block", "b": "block",
                                             "c": "block", "d": "block"}


def test_typed_objects_is_the_sorted_eval_schema_universe(blocks):
    domain, problem = blocks
    assert typed_objects(problem, domain) == ("a:block", "b:block",
                                              "c:block", "d:block")


def test_domain_constants_join_the_object_universe(with_constants):
    """``ProblemParser`` leaves :constants out of ``problem.objects``."""
    domain, problem = with_constants
    assert "tbl" not in problem.objects
    assert object_types(problem, domain) == {"tbl": "table", "a": "block",
                                             "b": "block"}
    assert typed_objects(problem, domain) == ("a:block", "b:block", "tbl:table")


# ── literals ─────────────────────────────────────────────────────────────

def test_literal_arguments_keep_their_order_and_are_comma_separated(blocks):
    """Two same-typed arguments, so a swap cannot hide behind the type names."""
    domain, problem = blocks
    predicate = GroundedPredicate(name="on",
                                  signature=domain.predicates["on"].signature,
                                  object_mapping={"?x": "a", "?y": "b"})
    assert predicate_to_literal(predicate, object_types(problem, domain)) == \
        "on(a:block,b:block)"


def test_nullary_predicate_renders_with_empty_parens(blocks):
    domain, problem = blocks
    predicate = next(iter(problem.initial_state_predicates["(handempty )"]))
    assert predicate_to_literal(predicate, object_types(problem, domain)) == \
        "handempty()"


def test_arguments_of_different_types_are_typed_individually(gripper):
    domain, problem = gripper
    predicate = next(p for p in problem.initial_state_predicates["(at ?b ?r)"]
                     if p.object_mapping["?b"] == "ball1")
    room = predicate.object_mapping["?r"]
    assert predicate_to_literal(predicate, object_types(problem, domain)) == \
        f"at(ball1:ball,{room}:room)"


def test_unknown_object_fails_loudly_instead_of_writing_a_bad_literal(blocks):
    _, problem = blocks
    predicate = next(iter(problem.initial_state_predicates["(clear ?x)"]))
    with pytest.raises(KeyError, match="not declared"):
        predicate_to_literal(predicate, {})


# ── states ───────────────────────────────────────────────────────────────

def test_initial_state_keeps_its_positive_literals(blocks):
    domain, problem = blocks
    types, objects = object_types(problem, domain), typed_objects(problem, domain)
    state = State(predicates=problem.initial_state_predicates,
                  fluents=problem.initial_state_fluents, is_init=True)

    trace_state = state_to_trace_state(state, types, objects)

    assert set(trace_state.literals) == {
        "clear(a:block)", "clear(b:block)", "clear(c:block)", "clear(d:block)",
        "ontable(a:block)", "ontable(b:block)", "ontable(c:block)",
        "ontable(d:block)", "handempty()"}
    assert trace_state.objects == objects


def test_literals_are_sorted(blocks):
    domain, problem = blocks
    state = State(predicates=problem.initial_state_predicates,
                  fluents=problem.initial_state_fluents, is_init=True)
    trace_state = state_to_trace_state(state, object_types(problem, domain),
                                       typed_objects(problem, domain))
    assert list(trace_state.literals) == sorted(trace_state.literals)


def test_an_explicitly_negated_predicate_is_dropped(blocks):
    """The eval schema says false by absence; a stored negation must not read true."""
    domain, problem = blocks
    types, objects = object_types(problem, domain), typed_objects(problem, domain)
    signature = domain.predicates["clear"].signature
    negated = GroundedPredicate(name="clear", signature=signature,
                                object_mapping={"?x": "a"}, is_positive=False)
    state = State(predicates={"(clear ?x)": {negated}}, fluents={})

    assert state_to_trace_state(state, types, objects).literals == ()


# ── actions ──────────────────────────────────────────────────────────────

def test_action_arguments_are_comma_space_separated(blocks):
    from pddl_plus_parser.models import ActionCall

    domain, problem = blocks
    call = ActionCall(name="stack", grounded_parameters=["a", "b"])
    assert action_call_to_eval_string(call, object_types(problem, domain)) == \
        "stack(a:block, b:block)"


def test_unary_action_renders_with_one_typed_argument(blocks):
    from pddl_plus_parser.models import ActionCall

    domain, problem = blocks
    call = ActionCall(name="pick_up", grounded_parameters=["a"])
    assert action_call_to_eval_string(call, object_types(problem, domain)) == \
        "pick_up(a:block)"
