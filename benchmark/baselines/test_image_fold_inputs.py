"""Tests for the shared imaged-ROSAME fold walk.

The GT-path, dialect and anchor cases moved here verbatim from
``test_rosame_i_runner.py`` when the walk was extracted; the anchor tests are
new, and cover the GT init state the ICAPS-26 arms need and the 24 arms drop.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest
from pddl_plus_parser.lisp_parsers import DomainParser

from benchmark.baselines import image_fold_inputs as inputs
from benchmark.baselines.image_fold_inputs import (
    BENCH_KEYS,
    DOMAIN_ALIASES,
    conformed_tempfile,
    domain_uses_hyphens,
    gt_anchors_from_observation,
    infer_bench_key,
    parse_problem_normalized,
    resolve_fold_inputs,
    resolve_gt_trajectory_path,
    resolve_images,
)


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


class TestResolveGtTrajectoryPath:
    def test_returns_the_gt_trajectory(self, tmp_path: Path) -> None:
        problem_dir = _make_cell(tmp_path)
        gt_path = _write_gt(tmp_path)

        assert resolve_gt_trajectory_path(problem_dir, "problem1") == gt_path

    def test_normalized_staging_dir_resolves_too(self, tmp_path: Path) -> None:
        problem_dir = _make_cell(tmp_path, staging="trajectories_normalized")
        gt_path = _write_gt(tmp_path)

        assert resolve_gt_trajectory_path(problem_dir, "problem1") == gt_path

    def test_raises_when_gt_trajectories_is_missing(self, tmp_path: Path) -> None:
        problem_dir = _make_cell(tmp_path)

        with pytest.raises(FileNotFoundError, match="no GT trajectory for 'problem1'"):
            resolve_gt_trajectory_path(problem_dir, "problem1")

    def test_never_substitutes_the_degraded_in_dir_trajectory(self, tmp_path: Path) -> None:
        problem_dir = _make_cell(tmp_path)
        (problem_dir / "problem1.trajectory").write_text("(:trajectory)")

        with pytest.raises(FileNotFoundError):
            resolve_gt_trajectory_path(problem_dir, "problem1")

    def test_raises_on_an_unexpected_directory_layout(self, tmp_path: Path) -> None:
        problem_dir = tmp_path / "somewhere" / "problem1"
        problem_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="cannot locate gt_trajectories/"):
            resolve_gt_trajectory_path(problem_dir, "problem1")


# ── GT anchors ──────────────────────────────────────────────────────────


class _Step:
    def __init__(self, previous_state, next_state) -> None:
        self.previous_state = previous_state
        self.next_state = next_state


class _Observation:
    def __init__(self, components) -> None:
        self.components = components


class _State:
    def __init__(self, predicates: List[str]) -> None:
        self.state_predicates = {
            name: [type("GP", (), {"untyped_representation": f"({name})"})()]
            for name in predicates
        }


class TestGtAnchors:
    def test_raises_when_the_gt_trajectory_parses_to_zero_transitions(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="zero transitions"):
            gt_anchors_from_observation(
                _Observation([]), tmp_path / "gt.trajectory", "problem1"
            )

    def test_init_is_the_first_previous_state_and_final_the_last_next(
        self, tmp_path: Path
    ) -> None:
        observation = _Observation(
            [
                _Step(_State(["a"]), _State(["b"])),
                _Step(_State(["b"]), _State(["c"])),
            ]
        )

        init, final = gt_anchors_from_observation(
            observation, tmp_path / "gt.trajectory", "problem1"
        )

        assert init == ["a"]
        assert final == ["c"]


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
        assert domain_uses_hyphens(_domain(tmp_path, "clear-disc"))

    def test_detects_a_normalized_domain(self, tmp_path: Path) -> None:
        assert not domain_uses_hyphens(_domain(tmp_path, "clear_disc"))


class TestConformedTempfile:
    def test_normalizes_input_for_a_normalized_domain(self, tmp_path: Path) -> None:
        source = tmp_path / "problem1.pddl"
        source.write_text(_HYPHENATED_PROBLEM)

        tmp = conformed_tempfile(source, ".pddl", _domain(tmp_path, "clear_disc"))
        try:
            assert "clear_disc" in tmp.read_text()
        finally:
            tmp.unlink(missing_ok=True)

    def test_leaves_input_verbatim_for_a_hyphenated_domain(self, tmp_path: Path) -> None:
        """A domain that skipped normalization keeps hyphens; inputs must match it."""
        source = tmp_path / "problem1.pddl"
        source.write_text(_HYPHENATED_PROBLEM)

        tmp = conformed_tempfile(source, ".pddl", _domain(tmp_path, "clear-disc"))
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

        problem = parse_problem_normalized(_domain(tmp_path, clear), problem_path)

        assert problem.objects.keys() == {"d1"}


# ── bench-key inference ─────────────────────────────────────────────────


class _NamedDomain:
    def __init__(self, name: str) -> None:
        self.name = name


class TestInferBenchKey:
    @pytest.mark.parametrize(
        "domain_name,expected",
        [("blocks", "blocksworld"), ("n_puzzle_typed", "npuzzle"), ("depot", "depot")],
    )
    def test_the_domain_name_wins(self, domain_name: str, expected: str) -> None:
        assert infer_bench_key(Path("/tmp/x.pddl"), _NamedDomain(domain_name)) == expected

    def test_falls_back_to_the_path(self, tmp_path: Path) -> None:
        path = tmp_path / "gripper" / "domain.pddl"
        path.parent.mkdir(parents=True)
        assert infer_bench_key(path, _NamedDomain("unheard-of")) == "gripper"

    def test_an_unknown_domain_keeps_its_own_name(self, tmp_path: Path) -> None:
        assert infer_bench_key(tmp_path / "d.pddl", _NamedDomain("Hiking")) == "hiking"

    def test_every_alias_target_is_a_bench_key(self) -> None:
        """The alias table is the only source of bench keys; it must close on itself."""
        assert set(DOMAIN_ALIASES.values()) == set(BENCH_KEYS)
        assert BENCH_KEYS <= set(DOMAIN_ALIASES)


# ── image resolution ────────────────────────────────────────────────────


class TestResolveImages:
    def test_frames_are_sorted_numerically_not_lexically(self, tmp_path: Path) -> None:
        for i in (0, 2, 10):
            (tmp_path / f"state_{i}.png").write_bytes(b"")

        assert [p.name for p in resolve_images(tmp_path, "blocksworld", "problem1")] == [
            "state_0.png",
            "state_2.png",
            "state_10.png",
        ]

    def test_no_frames_anywhere_is_an_empty_list(self, tmp_path: Path) -> None:
        assert resolve_images(tmp_path, "blocksworld", "no_such_problem") == []


# ── the walk itself ─────────────────────────────────────────────────────


_REAL_CELL = Path(
    "benchmark/data/blocksworld/blocks_predefined_problems1-10_final-version"
)


def _real_fold():
    """``(domain, prepared_trajectories)`` for a checked-in blocksworld cell."""
    from src.utils.config import load_config

    staging = _REAL_CELL / "training" / "trajectories"
    domain = DomainParser(
        Path(load_config()["domains"]["blocksworld"]["domain_file"]),
        partial_parsing=True,
    ).parse_domain()
    prepared = [
        (d / f"{d.name}.trajectory", d / "masking", d / f"{d.name}.pddl")
        for d in sorted(staging.iterdir())
        if d.is_dir() and (d / f"{d.name}.pddl").exists()
    ]
    return domain, prepared


@pytest.mark.skipif(not _REAL_CELL.exists(), reason="blocksworld image cell absent")
class TestGtInitAgreesWithTheProblem:
    """Plan §4.3 says the init anchor is the problem's ``:init``; the walk reads it
    off the GT trajectory instead, so that the two anchors come from one parse
    against one grounding. The two sources must agree, or §4.3 is misdescribed."""

    def test_on_every_problem_of_a_real_cell(self) -> None:
        domain, prepared = _real_fold()
        resolved = resolve_fold_inputs(domain, prepared, "blocksworld")
        assert len(resolved) == len(prepared) > 0

        for trace in resolved:
            from_problem = sorted(
                gp.untyped_representation[1:-1]
                for grounded in trace.problem.initial_state_predicates.values()
                for gp in grounded
            )
            assert sorted(trace.gt_init_predicates) == from_problem, trace.problem_name

    def test_frames_outnumber_actions_by_one(self) -> None:
        """Plan §4.1's mismatch, on real data: N actions come with N+1 images."""
        domain, prepared = _real_fold()
        for trace in resolve_fold_inputs(domain, prepared, "blocksworld"):
            assert len(trace.image_paths) == len(trace.action_strings) + 1


class TestResolveFoldInputs:
    def test_a_trajectory_without_frames_is_dropped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No frames on disk is the simulation-mode cell the callers skip on."""
        monkeypatch.setattr(inputs, "resolve_images", lambda *_: [])
        problem_dir = _make_cell(tmp_path)
        problem_pddl = problem_dir / "problem1.pddl"
        problem_pddl.write_text(_HYPHENATED_PROBLEM)

        resolved = resolve_fold_inputs(
            _domain(tmp_path, "clear_disc"),
            [(problem_dir / "problem1.trajectory", problem_dir / "m", problem_pddl)],
            "toy",
        )

        assert resolved == []
