"""Tests for the ROSAME-I baseline's GT final-state resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

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
