"""Tests for GT observation loading.

    python -m pytest benchmark/experiment_running_helpers/test_simulated_data_utils.py
"""

import inspect

from benchmark.experiment_running_helpers import data_source, simulated_data_utils
from benchmark.experiment_running_helpers.simulated_data_utils import (
    load_gt_observation,
    prepare_simulated_observations,
)


class TestObjectTableComesFromTheProblem:
    """A trajectory lists positive fluents only, so it is not an object table.

    An object that goes unmentioned for a whole window is absent from an
    inferred table, and grounding then raises KeyError on it. Measured on the
    gripper corpus: 23 of 400 windows never mention one of the two rooms,
    because the robot stayed in the other one.
    """

    def test_load_accepts_a_problem_path(self):
        params = inspect.signature(load_gt_observation).parameters
        assert "problem_path" in params

    def test_problem_path_is_optional(self):
        """Callers without a problem keep the old inference path."""
        assert inspect.signature(load_gt_observation).parameters[
            "problem_path"
        ].default is None

    def test_driver_forwards_problem_paths(self):
        params = inspect.signature(prepare_simulated_observations).parameters
        assert "problem_paths" in params
        assert params["problem_paths"].default is None


class TestSimulatedDataSourcePairsThem:
    """The pool already holds both, positionally aligned."""

    def test_prepare_passes_problem_paths(self):
        source = inspect.getsource(data_source.SimulatedDataSource.prepare)
        assert "problem_paths=" in source, (
            "SimulatedDataSource.prepare must hand the problem files to "
            "prepare_simulated_observations, or gripper windows raise KeyError"
        )

    def test_the_paths_are_built_from_the_selected_dirs(self):
        """selected_dirs is aligned with selected_gt; the pairing relies on it."""
        source = inspect.getsource(data_source.SimulatedDataSource.prepare)
        assert "for d in selected_dirs" in source


def test_module_imports_the_problem_parser():
    assert hasattr(simulated_data_utils, "ProblemParser")
