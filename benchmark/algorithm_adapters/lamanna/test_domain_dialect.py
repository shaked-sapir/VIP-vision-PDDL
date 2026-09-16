"""Untyped parameters become ``object`` for the Lamanna parsers, nothing else moves.

    python -m pytest benchmark/algorithm_adapters/lamanna/test_domain_dialect.py
"""

from __future__ import annotations

from pathlib import Path

from benchmark.algorithm_adapters.lamanna.domain_dialect import typed_domain_text

ROOT = Path(__file__).resolve().parents[3]
DEPOT = ROOT / "src" / "domains" / "depot" / "depot.pddl"
HANOI = ROOT / "src" / "domains" / "hanoi" / "hanoi.pddl"

DOMAIN = """(define (domain d)
  (:requirements :strips :typing)
  (:types a b)
  (:predicates
      (p ?x - a ?y - b)
      (q ?x)
      (r))
  (:action act
    :parameters (?x - a ?y)
    :precondition (and (p ?x ?y) (q ?y))
    :effect (and (not (q ?y)) (r)))
)
"""


def test_untyped_predicate_parameters_get_the_root_type():
    typed = typed_domain_text(DOMAIN)
    assert "(q ?x - object)" in typed
    assert "(p ?x - a ?y - b)" in typed
    assert "(r)" in typed


def test_untyped_operator_parameters_get_the_root_type():
    typed = typed_domain_text(DOMAIN)
    assert ":parameters (?x - a ?y - object)" in typed


def test_action_bodies_are_untouched():
    typed = typed_domain_text(DOMAIN)
    assert "(and (p ?x ?y) (q ?y))" in typed
    assert "(and (not (q ?y)) (r))" in typed


def test_a_flat_type_list_is_rooted_at_object():
    assert "(:types a b - object)" in typed_domain_text(DOMAIN)


def test_an_existing_hierarchy_is_kept():
    text = DOMAIN.replace("(:types a b)", "(:types a b - thing\n thing - object)")
    assert "(:types a b - thing\n thing - object)" in typed_domain_text(text)


def test_a_fully_typed_domain_is_returned_unchanged():
    for domain in ("hanoi/hanoi.pddl", "blocks/blocks.pddl", "gripper/gripper.pddl", "n_puzzle/n_puzzle.pddl"):
        original = (ROOT / "src" / "domains" / domain).read_text()
        assert typed_domain_text(original) == original, domain


def test_grouped_variables_sharing_one_type_are_not_untyped():
    text = DOMAIN.replace(":parameters (?x - a ?y)", ":parameters (?x ?z - a ?y)")
    typed = typed_domain_text(text)
    assert ":parameters (?x ?z - a ?y - object)" in typed
    text = DOMAIN.replace("(q ?x)", "(q ?x ?w - b)")
    assert "(q ?x ?w - b)" in typed_domain_text(text)


def test_npuzzle_move_keeps_its_parameter_types(tmp_path, monkeypatch):
    """``(?t - tile ?from ?to - position)`` must survive an untyped predicate elsewhere."""
    from nolam.algorithm.ActionModel import ActionModel

    original = (ROOT / "src" / "domains" / "n_puzzle" / "n_puzzle.pddl").read_text()
    text = original.replace("(empty ?position - position)", "(empty ?position)")
    assert text != original, "fixture expects npuzzle to declare (empty ?position - position)"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "d.pddl").write_text(typed_domain_text(text))
    model = ActionModel(input_file=str(tmp_path / "d.pddl"))
    (move,) = model.operators
    assert dict(move.parameters) == {"?param_1": "tile", "?param_2": "position", "?param_3": "position"}
    assert "empty(object)" in model.predicates


def test_depot_clear_becomes_unary_for_the_parsers(tmp_path, monkeypatch):
    from nolam.algorithm.ActionModel import ActionModel

    monkeypatch.chdir(tmp_path)
    (tmp_path / "depot.pddl").write_text(typed_domain_text(DEPOT.read_text()))
    model = ActionModel(input_file=str(tmp_path / "depot.pddl"))
    assert "clear(object)" in model.predicates
    assert "clear()" not in model.predicates
