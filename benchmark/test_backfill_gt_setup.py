"""The ``gt_setup`` channel from an experiment or a run config to the baseline runners.

    python -m pytest benchmark/test_backfill_gt_setup.py

``backfill_baseline`` takes the setup from ``--gt-setup`` or the experiment's
``run_params.json``; ``benchmark_runner`` takes it from ``shared.gt_setup``. Both
hand it only to the runners that accept it.
"""

from __future__ import annotations

import argparse

import pytest

from benchmark.backfill_baseline import (
    LEGACY_GT_SETUP,
    experiment_gt_setup,
    explicit_endpoint_flags,
    with_gt_setup,
)
from benchmark.baselines import resolve_baselines
from benchmark.benchmark_runner import _RUNNER_KWARG_KEYS, _build_main_kwargs
from src.utils.gt_setup import GtSetup

_IMAGE = {"data_source_type": "ImageDataSource"}
_SIMULATED = {"data_source_type": "SimulatedDataSource", "p_mask": 0.1, "p_noise": 0.0}


class TestExperimentGtSetup:
    def test_an_image_experiment_that_records_none_is_init(self) -> None:
        assert experiment_gt_setup(_IMAGE) is GtSetup.INIT

    @pytest.mark.parametrize("setup", list(GtSetup))
    def test_a_recorded_setup_is_used(self, setup: GtSetup) -> None:
        assert experiment_gt_setup({**_IMAGE, "gt_setup": setup.value}) is setup
        assert experiment_gt_setup({**_SIMULATED, "gt_setup": setup.value}) is setup

    def test_a_simulated_experiment_that_records_none_has_no_setup(self) -> None:
        assert experiment_gt_setup(_SIMULATED) is None

    def test_an_experiment_without_run_params_has_no_setup(self) -> None:
        assert experiment_gt_setup(None) is None

    @pytest.mark.parametrize("run_params", [None, _IMAGE, _SIMULATED, {**_IMAGE, "gt_setup": "final"}])
    def test_the_flag_overrides_whatever_is_recorded(self, run_params) -> None:
        assert experiment_gt_setup(run_params, "init_final") is GtSetup.INIT_FINAL
        assert experiment_gt_setup(run_params, LEGACY_GT_SETUP) is None

    def test_an_unknown_recorded_value_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown gt_setup"):
            experiment_gt_setup({**_IMAGE, "gt_setup": "both"})


class TestForwarding:
    def test_no_setup_adds_no_runner_option(self) -> None:
        assert with_gt_setup({"epochs": 5}, None) == {"epochs": 5}

    def test_a_setup_is_added_without_touching_the_input(self) -> None:
        base = {"epochs": 5}
        assert with_gt_setup(base, GtSetup.FINAL) == {"epochs": 5, "gt_setup": GtSetup.FINAL}
        assert base == {"epochs": 5}

    def test_runners_that_take_no_setup_are_built_without_it(self) -> None:
        runners = resolve_baselines(
            ["rosame_24", "offlam", "rosame_milp_24"], **with_gt_setup({}, GtSetup.NONE)
        )
        assert [getattr(r, "gt_setup", "absent") for r in runners] == [
            "absent", "absent", GtSetup.NONE,
        ]

    def test_the_explicit_endpoint_flags_are_named(self) -> None:
        nothing = argparse.Namespace(goal_mode=None, milp_gt_anchoring=None)
        both = argparse.Namespace(goal_mode="none", milp_gt_anchoring="none")
        assert explicit_endpoint_flags(nothing) == []
        assert explicit_endpoint_flags(both) == ["--goal-mode", "--milp-gt-anchoring"]


class TestRunConfig:
    def test_the_key_is_a_runner_option(self) -> None:
        assert "gt_setup" in _RUNNER_KWARG_KEYS

    def test_shared_gt_setup_reaches_the_runners_that_accept_it(self) -> None:
        kwargs = _build_main_kwargs({
            "algorithms": ["rosame_24", "rosame_milp_24", "rosame_i_24"],
            "gt_setup": "none",
        })
        assert [getattr(r, "gt_setup", "absent") for r in kwargs["baselines"]] == [
            "absent", GtSetup.NONE, GtSetup.NONE,
        ]

    def test_a_config_without_the_key_builds_runners_without_a_setup(self) -> None:
        kwargs = _build_main_kwargs({"algorithms": ["rosame_milp_24", "rosame_i_24"]})
        assert [r.gt_setup for r in kwargs["baselines"]] == [None, None]

    def test_an_unknown_setup_fails_when_the_runners_are_built(self) -> None:
        with pytest.raises(ValueError, match="unknown gt_setup"):
            _build_main_kwargs({"algorithms": ["rosame_milp_24"], "gt_setup": "both"})
