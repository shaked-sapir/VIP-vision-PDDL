"""Tests for :mod:`benchmark.experiment_running_helpers.cleaned_trajectories`.

    python -m pytest benchmark/experiment_running_helpers/test_cleaned_trajectories.py
"""

from pathlib import Path
from unittest import mock

import pytest

from benchmark.experiment_running_helpers import cleaned_trajectories as ct

MODULE = "benchmark.experiment_running_helpers.cleaned_trajectories"


def _fold(root: Path, problems):
    """``(trajectory, masking_info, problem_pddl)`` per problem, in fold order."""
    return [
        (root / f"{p}.trajectory", root / f"{p}.masking_info", root / f"{p}.pddl")
        for p in problems
    ]


def _write_marker(observation, path: Path) -> None:
    path.write_text(str(observation))


def _saved(out_dir: Path):
    """``{problem name: file content}`` of every trajectory written."""
    return {
        f.stem.removeprefix("final_observation_"): f.read_text()
        for f in out_dir.glob("final_observation_*.trajectory")
    }


@pytest.fixture
def stubbed_io():
    with mock.patch(f"{MODULE}.observation_to_trajectory_file", side_effect=_write_marker), \
         mock.patch(f"{MODULE}.extract_masking_info_from_observation", return_value=[]), \
         mock.patch(f"{MODULE}.save_masking_info"):
        yield


class TestSavePatchedObservations:
    def test_loop_subset_is_named_by_pool_index(self, tmp_path, stubbed_io):
        fold = _fold(tmp_path, ["problem7", "problem5", "problem3"])
        out = tmp_path / "final_observations"
        ct.save_patched_observations(
            ["obs of problem7", "obs of problem3"], fold, out, tmp_path / "d.pddl",
            observation_indices=[0, 2],
        )
        assert _saved(out) == {
            "problem7": "obs of problem7",
            "problem3": "obs of problem3",
        }

    def test_whole_fold_stays_positional(self, tmp_path, stubbed_io):
        fold = _fold(tmp_path, ["problem7", "problem5", "problem3"])
        out = tmp_path / "final_observations"
        ct.save_patched_observations(["o7", "o5", "o3"], fold, out, tmp_path / "d.pddl")
        assert _saved(out) == {"problem7": "o7", "problem5": "o5", "problem3": "o3"}

    def test_index_count_must_match(self, tmp_path, stubbed_io):
        fold = _fold(tmp_path, ["problem7", "problem5", "problem3"])
        with pytest.raises(ValueError):
            ct.save_patched_observations(
                ["a", "b"], fold, tmp_path / "out", tmp_path / "d.pddl",
                observation_indices=[0],
            )


class TestSaveObservationsToDir:
    def test_subset_is_named_by_pool_index(self, tmp_path, stubbed_io):
        fold = _fold(tmp_path, ["problem7", "problem5", "problem3"])
        out = tmp_path / "t_prime"
        ct.save_observations_to_dir(["o7", "o3"], fold, out, observation_indices=[0, 2])
        assert _saved(out) == {"problem7": "o7", "problem3": "o3"}
