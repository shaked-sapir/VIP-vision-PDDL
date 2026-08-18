"""Tests for the ROSAME-I baseline's GT final-state resolution and input dialect."""

from __future__ import annotations

from pathlib import Path

import pytest
from pddl_plus_parser.lisp_parsers import DomainParser

from benchmark.baselines.rosame_i_runner import RosameIBaselineRunner


def _make_cell(root: Path, staging: str = "trajectories") -> Path:
    """A ``<data_dir>/training/<staging>/problem1/`` tree; returns the problem dir."""
    problem_dir = root / "training" / staging / "problem1"
    problem_dir.mkdir(parents=True)
    return problem_dir


def _write_gt(root: Path, problem_name: str = "problem1") -> Path:
    gt_path = root / "gt_trajectories" / problem_name / f"{problem_name}.trajectory"
    gt_path.parent.mkdir(parents=True)
    gt_path.write_text("(:trajectory)")
    return gt_path


class TestResolveFinalStatePath:
    def test_returns_the_gt_trajectory(self, tmp_path: Path) -> None:
        problem_dir = _make_cell(tmp_path)
        gt_path = _write_gt(tmp_path)

        resolved = RosameIBaselineRunner.resolve_final_state_path(problem_dir, "problem1")

        assert resolved == gt_path

    def test_normalized_staging_dir_resolves_too(self, tmp_path: Path) -> None:
        problem_dir = _make_cell(tmp_path, staging="trajectories_normalized")
        gt_path = _write_gt(tmp_path)

        resolved = RosameIBaselineRunner.resolve_final_state_path(problem_dir, "problem1")

        assert resolved == gt_path

    def test_raises_when_gt_trajectories_is_missing(self, tmp_path: Path) -> None:
        problem_dir = _make_cell(tmp_path)

        with pytest.raises(FileNotFoundError, match="no GT trajectory for 'problem1'"):
            RosameIBaselineRunner.resolve_final_state_path(problem_dir, "problem1")

    def test_never_substitutes_the_degraded_in_dir_trajectory(self, tmp_path: Path) -> None:
        problem_dir = _make_cell(tmp_path)
        (problem_dir / "problem1.trajectory").write_text("(:trajectory)")

        with pytest.raises(FileNotFoundError):
            RosameIBaselineRunner.resolve_final_state_path(problem_dir, "problem1")

    def test_raises_on_an_unexpected_directory_layout(self, tmp_path: Path) -> None:
        problem_dir = tmp_path / "somewhere" / "problem1"
        problem_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="cannot locate gt_trajectories/"):
            RosameIBaselineRunner.resolve_final_state_path(problem_dir, "problem1")


class _EmptyObservation:
    components: list = []


class TestResolveFinalState:
    def test_raises_when_the_gt_trajectory_parses_to_zero_transitions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = RosameIBaselineRunner()
        monkeypatch.setattr(
            RosameIBaselineRunner,
            "_parse_trajectory_normalized",
            lambda self, domain, problem, path: _EmptyObservation(),
        )

        with pytest.raises(ValueError, match="zero transitions"):
            runner._resolve_final_state(None, None, tmp_path / "gt.trajectory", "problem1")


# ── input dialect is conformed to the reference domain ───────────────────

_DOMAIN_TEMPLATE = """(define (domain toy)
  (:requirements :strips :typing)
  (:types disc)
  (:predicates ({clear} ?x - disc))
  (:action move
    :parameters (?x - disc)
    :precondition ({clear} ?x)
    :effect (not ({clear} ?x)))
)
"""

_HYPHENATED_PROBLEM = """(define (problem problem1) (:domain toy)
  (:objects d1 - disc)
  (:init (clear-disc d1))
  (:goal (and (clear-disc d1)))
)
"""


def _domain(tmp_path: Path, clear: str):
    """Parse the toy domain declaring its ``clear`` predicate as given."""
    path = tmp_path / f"domain_{clear.replace('-', '_')}.pddl"
    path.write_text(_DOMAIN_TEMPLATE.format(clear=clear))
    return DomainParser(path, partial_parsing=True).parse_domain()


class TestDomainUsesHyphens:
    def test_detects_a_hyphenated_domain(self, tmp_path: Path) -> None:
        assert RosameIBaselineRunner._domain_uses_hyphens(_domain(tmp_path, "clear-disc"))

    def test_detects_a_normalized_domain(self, tmp_path: Path) -> None:
        assert not RosameIBaselineRunner._domain_uses_hyphens(
            _domain(tmp_path, "clear_disc")
        )


class TestConformedTempfile:
    def test_normalizes_input_for_a_normalized_domain(self, tmp_path: Path) -> None:
        source = tmp_path / "problem1.pddl"
        source.write_text(_HYPHENATED_PROBLEM)

        tmp = RosameIBaselineRunner._conformed_tempfile(
            source, ".pddl", _domain(tmp_path, "clear_disc")
        )
        try:
            assert "clear_disc" in tmp.read_text()
        finally:
            tmp.unlink(missing_ok=True)

    def test_leaves_input_verbatim_for_a_hyphenated_domain(self, tmp_path: Path) -> None:
        """A domain that skipped normalization keeps hyphens; inputs must match it."""
        source = tmp_path / "problem1.pddl"
        source.write_text(_HYPHENATED_PROBLEM)

        tmp = RosameIBaselineRunner._conformed_tempfile(
            source, ".pddl", _domain(tmp_path, "clear-disc")
        )
        try:
            assert tmp.read_text() == _HYPHENATED_PROBLEM
        finally:
            tmp.unlink(missing_ok=True)


class TestParseProblemAgainstEitherDialect:
    """One on-disk hyphenated problem must parse against a domain of either dialect."""

    @pytest.mark.parametrize("clear", ["clear-disc", "clear_disc"])
    def test_parses(self, tmp_path: Path, clear: str) -> None:
        problem_path = tmp_path / "problem1.pddl"
        problem_path.write_text(_HYPHENATED_PROBLEM)

        problem = RosameIBaselineRunner._parse_problem_normalized(
            _domain(tmp_path, clear), problem_path
        )

        assert problem.objects.keys() == {"d1"}
