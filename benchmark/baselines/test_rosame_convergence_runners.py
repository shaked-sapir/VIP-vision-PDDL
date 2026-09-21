"""The symbolic ROSAME runners under the training-loss convergence rule.

    python -m pytest benchmark/baselines/test_rosame_convergence_runners.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Tuple

import pytest

from benchmark.algorithm_adapters.test_po_rosame_runner import _write_world
from benchmark.baselines import resolve_baselines
from benchmark.baselines.rosame_milp_runner import RosameMilpRunner, capped_solve_limit
from benchmark.baselines.rosame_runner import LearningBudget, RosameBaselineRunner

SYMBOLIC = ["rosame_24", "rosame_milp_24", "rosame_milp_24_tag"]
RULE = {"window": 5, "min_improvement": 0.002, "patience": 2, "min_epochs": 10}
LARGE_SWEEP = {
    "train_per_trajectory": False, "epochs": 2000,
    "agreement_stop": None, "rosame_convergence": RULE,
}


def _fold(root: Path) -> Tuple[Path, List[Tuple[Path, None, Path]]]:
    """The micro world as prepared-trajectory tuples, one problem file per trace."""
    domain_path, traj_paths = _write_world(root)
    prepared = []
    for index, traj in enumerate(traj_paths):
        problem = root / f"problem{index}.pddl"
        shutil.copy(root / "problem.pddl", problem)
        prepared.append((traj, None, problem))
    return domain_path, prepared


class TestKnobs:
    def test_the_large_sweep_block_reaches_every_symbolic_arm(self):
        for runner in resolve_baselines(SYMBOLIC, **LARGE_SWEEP):
            assert runner.train_per_trajectory is False, runner.name
            assert runner.epochs == 2000, runner.name
            assert runner.convergence_rule.as_stats() == RULE, runner.name
            if runner.uses_milp:
                assert runner.agreement_stop is None, runner.name

    def test_defaults_are_the_legacy_behaviour(self):
        dl_only, milp, _tag = resolve_baselines(SYMBOLIC)
        assert dl_only.convergence_rule is None and dl_only.epochs == 100
        assert milp.convergence_rule is None and milp.agreement_stop == 1.0

    def test_run_params_gain_keys_only_when_the_rule_is_on(self):
        dl_only, milp, _tag = resolve_baselines(SYMBOLIC)
        assert set(dl_only.run_params()) == {"train_per_trajectory", "batch_size", "rosame_seed"}
        assert "agreement_stop" not in milp.run_params()
        assert "rosame_convergence" not in milp.run_params()

        dl_only, milp, _tag = resolve_baselines(SYMBOLIC, **LARGE_SWEEP)
        assert dl_only.run_params()["epochs"] == 2000
        assert dl_only.run_params()["rosame_convergence"] == RULE
        assert milp.run_params()["agreement_stop"] is None
        assert milp.run_params()["rosame_convergence"] == RULE

    def test_a_typo_in_the_rule_block_fails_at_construction(self):
        with pytest.raises(ValueError):
            RosameBaselineRunner(rosame_convergence={"windw": 10})

    def test_the_knobs_do_not_leak_into_other_arms(self):
        (runner,) = resolve_baselines(["nolam"], **LARGE_SWEEP)
        assert not hasattr(runner, "convergence_rule")


class TestBudget:
    def test_no_budget_never_expires(self):
        budget = LearningBudget(None)
        assert not budget.exhausted() and budget.seconds_left() == float("inf")

    def test_a_spent_budget_is_exhausted(self):
        assert LearningBudget(0).exhausted()

    def test_a_solve_is_capped_by_what_is_left(self):
        assert capped_solve_limit(60, 1000.0) == 60
        assert capped_solve_limit(60, 12.7) == 12
        assert capped_solve_limit(60, -5.0) == 1
        assert capped_solve_limit(60, float("inf")) == 60


class TestRosame24:
    def test_trains_to_the_rule_and_records_it(self, tmp_path):
        domain_path, prepared = _fold(tmp_path)
        runner = RosameBaselineRunner(train_per_trajectory=False, epochs=600,
                                      rosame_convergence=RULE, batch_size=0)
        model, extra = runner.learn(domain_path, prepared, tmp_path / "work", timeout_seconds=300)

        assert model and "(:action move" in model
        assert extra["stop_reason"] in ("converged", "epochs_exhausted")
        assert extra["epochs_run"] >= RULE["min_epochs"]
        assert extra["best_epoch"] is not None and extra["best_loss"] is not None
        series = json.loads((tmp_path / "work" / "rosame_training" / "ROSAME_24.json").read_text())
        assert len(series["losses"]) == extra["epochs_run"]
        assert series["best_loss"] == min(series["losses"])

    def test_the_fold_budget_is_enforced(self, tmp_path):
        domain_path, prepared = _fold(tmp_path)
        runner = RosameBaselineRunner(train_per_trajectory=False, epochs=600,
                                      rosame_convergence=RULE, batch_size=0)
        _model, extra = runner.learn(domain_path, prepared, tmp_path / "work", timeout_seconds=0)
        assert (extra["stop_reason"], extra["epochs_run"]) == ("timeout", 1)

    def test_the_per_trajectory_schedule_is_unchanged_by_the_rule(self, tmp_path):
        domain_path, prepared = _fold(tmp_path)
        runner = RosameBaselineRunner(train_per_trajectory=True, epochs=3, rosame_convergence=RULE)
        model, extra = runner.learn(domain_path, prepared, tmp_path / "work", timeout_seconds=300)
        assert model and extra["stop_reason"] == "epochs_exhausted"
        assert not (tmp_path / "work" / "rosame_training").exists()


class TestRosameMilp24:
    def test_trains_past_agreement_and_records_the_series(self, tmp_path):
        domain_path, prepared = _fold(tmp_path)
        runner = RosameMilpRunner(epochs=40, pre_mip_epochs=5, agreement_stop=None,
                                  rosame_convergence=RULE, batch_size=0, mip_time_limit=20)
        model, extra = runner.learn(domain_path, prepared, tmp_path / "work", timeout_seconds=300)

        assert model and "(:action move" in model
        assert extra["stop_reason"] in ("converged", "epochs_exhausted")
        assert extra["stop_reason"] != "agreement_reached"
        assert extra["agreement_stop"] is None
        assert extra["first_solve_epoch"] is not None
        assert extra["epochs_run"] >= extra["first_solve_epoch"] + 1 + RULE["min_epochs"]
        series = json.loads((tmp_path / "work" / "rosame_training" / "ROSAME_MILP_24.json").read_text())
        assert len(series["losses"]) == len(series["base_losses"]) == len(series["ce_losses"])
        assert len(series["losses"]) == extra["epochs_run"]

    def test_the_legacy_default_still_stops_on_agreement(self, tmp_path):
        domain_path, prepared = _fold(tmp_path)
        runner = RosameMilpRunner(epochs=40, pre_mip_epochs=5, batch_size=0, mip_time_limit=20)
        _model, extra = runner.learn(domain_path, prepared, tmp_path / "work", timeout_seconds=300)
        assert extra["stop_reason"] in ("agreement_reached", "epochs_exhausted")
        assert extra["best_epoch"] is None      # no checkpointing when the rule is off
