"""Planning runs on a copy without no-effect operators; nothing else sees the copy.

    python -m pytest benchmark/experiment_running_helpers/test_planning_copy.py
"""

from __future__ import annotations

import pytest

from benchmark.experiment_running_helpers import planning_copy as pc

MODEL = """(define (domain gripper)
(:requirements :typing)
(:types room ball gripper - object)
(:predicates (at-robby ?r - room) (at ?b - ball ?r - room) (free ?g - gripper) (carry ?b - ball ?g - gripper))

(:action move
	:parameters (?from - room ?to - room)
	:precondition (and (at-robby ?from))
	:effect (and
		))

(:action pick
	:parameters (?b - ball ?r - room ?g - gripper)
	:precondition (and (free ?g))
	:effect (and (carry ?b ?g) (not (free ?g))))

(:action drop
	:parameters (?b - ball ?r - room ?g - gripper)
	:precondition (and (carry ?b ?g))
	:effect (and (not (carry ?b ?g))))
)
"""


class TestWhichOperatorsAreInert:
    def test_an_empty_and_is_inert(self):
        assert pc.effect_is_empty("(:action a :parameters () :precondition (and (p)) :effect (and ))")

    def test_a_multiline_empty_and_is_inert(self):
        assert pc.effect_is_empty("(:action a\n:effect\t(and\n\t\n)\n)")

    def test_a_missing_effect_field_is_inert(self):
        assert pc.effect_is_empty("(:action a :parameters () :precondition (and (p)))")

    def test_a_delete_only_effect_is_not_inert(self):
        assert not pc.effect_is_empty("(:action a :effect (and (not (p ?x))))")

    def test_a_predicate_named_like_the_keyword_is_kept(self):
        assert not pc.effect_is_empty("(:action a :effect (and (android ?x)))")


class TestThePlanningText:
    def test_only_the_inert_operator_is_removed(self):
        text, dropped = pc.without_inert_operators(MODEL)
        assert dropped == ["move"]
        assert "(:action move" not in text
        assert "(:action pick" in text and "(:action drop" in text
        assert text.count("(") == text.count(")")

    def test_a_model_without_inert_operators_is_unchanged(self):
        healthy = MODEL.replace(":effect (and\n\t\t))", ":effect (and (at-robby ?to)))")
        assert healthy != MODEL
        text, dropped = pc.without_inert_operators(healthy)
        assert dropped == [] and text == healthy

    def test_every_operator_inert_leaves_a_domain_with_none(self):
        text, dropped = pc.without_inert_operators(
            "(define (domain d)\n(:action a :effect (and ))\n(:action b :effect (and ))\n)"
        )
        assert dropped == ["a", "b"]
        assert pc.ACTION_OPEN not in text


class TestTheCopyOnDisk:
    def test_a_healthy_model_is_planned_as_is(self, tmp_path):
        model = tmp_path / "m.pddl"
        model.write_text("(define (domain d)\n(:action a :effect (and (p)))\n)")
        with pc.planning_copy(model) as (path, dropped):
            assert path == model and dropped == []
        assert list(tmp_path.iterdir()) == [model]

    def test_the_copy_exists_only_inside_the_block_and_the_model_is_untouched(self, tmp_path):
        model = tmp_path / "m.pddl"
        model.write_text(MODEL)
        with pc.planning_copy(model) as (path, dropped):
            assert path != model and path.exists()
            assert dropped == ["move"]
            assert "(:action move" not in path.read_text()
        assert not path.exists()
        assert model.read_text() == MODEL


class TestSolvingMetrics:
    def _patch(self, monkeypatch, result):
        seen = {}

        def fake(model_path, ref_path, problems, timeout=60):
            seen["text"] = open(model_path).read()
            return result

        monkeypatch.setattr(pc, "problem_solving", fake)
        return seen

    def test_the_planner_sees_the_copy_and_the_row_names_what_was_dropped(self, tmp_path, monkeypatch):
        model = tmp_path / "m.pddl"
        model.write_text(MODEL)
        seen = self._patch(monkeypatch, {
            "solving_ratio": 0.0, "false_plans_ratio": 1.0, "unsolvable_ratio": 0.0,
            "timed_out": 0.0, "syntax_errors": 0.0,
        })
        metrics = pc.solving_metrics(model, tmp_path / "ref.pddl", ["p1"], timeout=5)
        assert "(:action move" not in seen["text"]
        assert metrics["planning_dropped_operators"] == "move"
        assert metrics["false_plans_ratio"] == 1.0
        assert metrics["planning_error_ratio"] == 0.0
        assert set(metrics) == set(pc.SOLVING_FIELDS)

    def test_an_outcome_in_no_bucket_is_reported_as_a_planning_error(self, tmp_path, monkeypatch):
        model = tmp_path / "m.pddl"
        model.write_text("(define (domain d)\n(:action a :effect (and (p)))\n)")
        self._patch(monkeypatch, {
            "solving_ratio": 0.5, "false_plans_ratio": 0.0, "unsolvable_ratio": 0.0,
            "timed_out": 0.0, "syntax_errors": 0.0,
        })
        metrics = pc.solving_metrics(model, tmp_path / "ref.pddl", ["p1", "p2"])
        assert metrics["planning_error_ratio"] == pytest.approx(0.5)
        assert metrics["planning_dropped_operators"] == ""

    def test_syntax_errors_are_recorded_and_are_not_planning_errors(self, tmp_path, monkeypatch):
        model = tmp_path / "m.pddl"
        model.write_text("(define (domain d)\n(:action a :effect (and (p)))\n)")
        self._patch(monkeypatch, {
            "solving_ratio": 0.0, "false_plans_ratio": 0.0, "unsolvable_ratio": 0.0,
            "timed_out": 0.0, "syntax_errors": 1.0,
        })
        metrics = pc.solving_metrics(model, tmp_path / "ref.pddl", ["p1"])
        assert metrics["planning_syntax_error_ratio"] == 1.0
        assert metrics["planning_error_ratio"] == 0.0
