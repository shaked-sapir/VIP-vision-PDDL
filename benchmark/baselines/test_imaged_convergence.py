"""The imaged arms under the training-loss convergence rule.

    python -m pytest benchmark/baselines/test_imaged_convergence.py
"""

from __future__ import annotations

from types import SimpleNamespace

from benchmark.algorithm_adapters.best_checkpoint import BestModelTracker
from benchmark.baselines import resolve_baselines
from benchmark.baselines.rosame26_runner import Rosame26BaselineRunner, _has_successful_round
from src.milp.rosame26_budget import (
    CONVERGE_MIN_EPOCHS,
    CONVERGE_PATIENCE,
    CONVERGE_WINDOW,
)

IMAGED_24 = ["rosame_i_24", "rosame_i_milp_24"]
IMAGED_RULE = {"window": 40, "min_improvement": 0.002, "patience": 3, "min_epochs": 50}

#: Flat epochs the rule needs before it can fire: a reference block plus patience.
_TO_CONVERGE = max(CONVERGE_MIN_EPOCHS, CONVERGE_WINDOW * (CONVERGE_PATIENCE + 1))


class TestImaged24Knobs:
    def test_the_rule_is_off_by_default_and_run_params_stay_empty(self):
        for runner in resolve_baselines(IMAGED_24):
            assert runner.convergence_rule is None, runner.name
            assert runner.run_params() == {}, runner.name

    def test_the_block_reaches_both_arms_with_the_imaged_window(self):
        for runner in resolve_baselines(IMAGED_24, rosame_convergence=IMAGED_RULE,
                                        agreement_stop=None):
            assert runner.convergence_rule.window == 40, runner.name
            assert runner.run_params() == {"rosame_convergence": IMAGED_RULE}, runner.name
        _dl_only, milp = resolve_baselines(IMAGED_24, agreement_stop=None)
        assert milp.agreement_stop is None

    def test_checkpointing_follows_the_rule(self):
        (off,) = resolve_baselines(["rosame_i_24"])
        (on,) = resolve_baselines(["rosame_i_24"], rosame_convergence=IMAGED_RULE)
        assert off._tracker().active is False and off._stop_check() is None
        assert on._tracker().active is True and on._stop_check() is not None


class TestRosame26Series:
    def test_a_round_counts_only_when_it_produced_labels(self):
        assert not _has_successful_round(None)
        assert not _has_successful_round(SimpleNamespace(rounds=[{"status": "NO_SOLUTION"}]))
        assert not _has_successful_round(SimpleNamespace(rounds=[{"status": "NO_TRACES"}]))
        assert _has_successful_round(SimpleNamespace(rounds=[{"status": "OPTIMAL"}]))

    def _check(self, repairer):
        trainer = SimpleNamespace(
            domain_model=SimpleNamespace(action_schemas=[]), mip_repairer=repairer)
        tracker = BestModelTracker(True)
        check = Rosame26BaselineRunner(budget_mode="converge")._stop_check(
            tracker, {"trainer": trainer})
        return check, tracker

    def test_the_dl_only_arm_scores_the_whole_history(self):
        check, _tracker = self._check(None)
        history = [{"total_loss": 1.0}] * _TO_CONVERGE
        assert not check(history[:-1])
        assert check(history)

    def test_the_milp_arm_restarts_at_its_first_successful_solve(self):
        repairer = SimpleNamespace(rounds=[])
        check, tracker = self._check(repairer)
        warmup = [{"total_loss": 1.0}] * 60
        assert not check(warmup)

        repairer.rounds.append({"status": "OPTIMAL"})
        solved_at = warmup + [{"total_loss": 5.0}]
        assert not check(solved_at)                    # the solve epoch itself is not scored
        assert tracker.best_loss is None               # and the tracker was reset

        almost = solved_at + [{"total_loss": 5.0}] * (_TO_CONVERGE - 1)
        assert not check(almost)
        assert check(almost + [{"total_loss": 5.0}])
        assert tracker.best_epoch >= len(solved_at)    # only post-solve epochs were scored
