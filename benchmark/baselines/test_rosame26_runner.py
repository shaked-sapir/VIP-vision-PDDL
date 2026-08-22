"""Tests for the ROSAME-I (26) baseline runner's identity and settings.

    python -m pytest benchmark/baselines/test_rosame26_runner.py

The learning path itself is covered where its pieces live —
``test_rosame26_data.py`` for the fold adapter, ``src/milp/test_rosame26_*.py``
for the model, the harness and the emitter. What is only true *here* is the
identity the dashboard keys on and the suffix rule that stops two budgets or two
resolutions from being averaged under one row name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.baselines import BASELINE_REGISTRY, get_baselines
from benchmark.baselines.base_runner import BaselineRunner
from src.milp.rosame26_budget import BudgetMode
from benchmark.baselines.rosame26_runner import (
    _BestModelTracker,
    _EPOCH_DEFAULT,
    _PROBE_EPOCHS,
    _EPOCHS,
    _RESIZE,
    _RESIZE_DEFAULT,
    Rosame26BaselineRunner,
)
from benchmark.baselines.rosame_i_runner import _RESIZE as _RESIZE_24
from benchmark.baselines.rosame_i_runner import _RESIZE_DEFAULT as _RESIZE_DEFAULT_24

_DOMAIN = """(define (domain {name})
  (:requirements :strips)
  (:predicates (clear ?d))
  (:action noop :parameters () :precondition () :effect ())
)
"""


def _domain_file(tmp_path: Path, name: str) -> Path:
    path = tmp_path / f"{name}.pddl"
    path.write_text(_DOMAIN.format(name=name))
    return path


class TestIdentity:
    def test_it_is_a_baseline_runner(self) -> None:
        assert isinstance(Rosame26BaselineRunner(), BaselineRunner)

    def test_the_registry_key_resolves_to_it(self) -> None:
        assert BASELINE_REGISTRY["rosame_i_26"] == [Rosame26BaselineRunner]
        assert isinstance(get_baselines(["rosame_i_26"])[0], Rosame26BaselineRunner)

    def test_names_are_distinct_from_every_other_arm(self) -> None:
        runners = [get_baselines([key])[0] for key in BASELINE_REGISTRY]
        assert len({runner.name for runner in runners}) == len(runners)
        assert len({runner.display_name for runner in runners}) == len(runners)

    def test_the_row_name_says_26(self) -> None:
        assert Rosame26BaselineRunner().name == "ROSAME-I_26"

    def test_the_colour_is_not_the_24_arms(self) -> None:
        colours = {get_baselines([key])[0].color for key in BASELINE_REGISTRY}
        assert len(colours) == len(BASELINE_REGISTRY)


class TestResolution:
    """Held equal to the 24 arm's, because resolution is a movable factor.

    Eight things move between the 24 and 26 arms; the delta is only about the
    network if the ones that need not move do not.
    """

    def test_it_matches_the_24_arm_on_every_domain(self) -> None:
        assert _RESIZE == _RESIZE_24
        assert _RESIZE_DEFAULT == _RESIZE_DEFAULT_24

    def test_an_explicit_override_wins_over_the_table(self, tmp_path: Path) -> None:
        runner = Rosame26BaselineRunner(resize=224)
        assert runner._resolve_resize("blocksworld") == 224

    def test_an_off_default_resize_is_suffixed(self, tmp_path: Path) -> None:
        runner = Rosame26BaselineRunner(resize=224)
        assert runner.row_name(_domain_file(tmp_path, "blocks")) == "ROSAME-I_26__res=224"

    def test_a_resize_equal_to_the_default_gets_no_suffix(self, tmp_path: Path) -> None:
        runner = Rosame26BaselineRunner(resize=_RESIZE_DEFAULT)
        assert runner.row_name(_domain_file(tmp_path, "blocks")) == "ROSAME-I_26"


class TestEpochBudget:
    def test_the_table_covers_every_domain_the_24_arm_covers(self) -> None:
        assert set(_EPOCHS) == set(_RESIZE_24)

    def test_the_table_agrees_with_the_default(self) -> None:
        """No domain is silently off-budget without a row-name suffix."""
        assert set(_EPOCHS.values()) == {_EPOCH_DEFAULT}

    def test_an_explicit_override_wins_over_the_table(self) -> None:
        assert Rosame26BaselineRunner(epochs=5000)._resolve_epochs("blocksworld") == 5000

    def test_gate_sevens_budget_is_suffixed(self, tmp_path: Path) -> None:
        """5000 outside the timeout must not be averaged with the grid's budget."""
        runner = Rosame26BaselineRunner(epochs=5000)
        assert runner.row_name(_domain_file(tmp_path, "blocks")) == "ROSAME-I_26__ep=5000"

    def test_the_default_budget_gets_no_suffix(self, tmp_path: Path) -> None:
        runner = Rosame26BaselineRunner(epochs=_EPOCH_DEFAULT)
        assert runner.row_name(_domain_file(tmp_path, "blocks")) == "ROSAME-I_26"

    def test_both_suffixes_compose(self, tmp_path: Path) -> None:
        runner = Rosame26BaselineRunner(epochs=5000, resize=224)
        assert (
            runner.row_name(_domain_file(tmp_path, "blocks"))
            == "ROSAME-I_26__res=224__ep=5000"
        )


class TestTheBudgetPreflight:
    """§1.2: the configured epoch count is a ceiling the cell's timeout lowers.

    The pre-flight binds *before* epoch 0 rather than being discovered at epoch
    4000, and it multiplies by ``n_seeds`` — a check that passes while the grid
    overruns by ``n_seeds``x is a check that was never run.
    """

    def test_a_generous_timeout_leaves_the_configured_count_alone(self) -> None:
        runner = Rosame26BaselineRunner(n_seeds=1)
        epochs, report = runner._budgeted_epochs("blocksworld", 100000)

        assert epochs == _EPOCH_DEFAULT
        assert report["fits"]

    def test_a_tight_timeout_lowers_the_count(self, capsys) -> None:
        runner = Rosame26BaselineRunner(n_seeds=1)
        epochs, report = runner._budgeted_epochs("blocksworld", 60)

        assert 0 < epochs < _EPOCH_DEFAULT
        assert not report["fits"]
        assert "budget" in capsys.readouterr().out

    def test_seeds_lower_it_further(self) -> None:
        one, _ = Rosame26BaselineRunner(n_seeds=1)._budgeted_epochs("blocksworld", 600)
        three, _ = Rosame26BaselineRunner(n_seeds=3)._budgeted_epochs("blocksworld", 600)

        assert three < one

    def test_fixed_mode_keeps_the_configured_count(self) -> None:
        """Gate 7's control cell runs 5000 whatever the projection says."""
        runner = Rosame26BaselineRunner(epochs=5000, budget_mode="fixed")
        epochs, report = runner._budgeted_epochs("blocksworld", 600)

        assert epochs == 5000
        assert not report["fits"]
        assert report["mode"] == "fixed"

    def test_converge_mode_also_keeps_the_configured_ceiling(self) -> None:
        """The plateau ends the run, not the projection."""
        runner = Rosame26BaselineRunner(epochs=5000, budget_mode="converge")
        epochs, _ = runner._budgeted_epochs("blocksworld", 600)

        assert epochs == 5000

    def test_the_report_carries_what_was_asked_for(self) -> None:
        runner = Rosame26BaselineRunner(n_seeds=1)
        _, report = runner._budgeted_epochs("blocksworld", 600)

        assert report["requested_epochs"] == _EPOCH_DEFAULT
        assert report["budget_seconds"] < 600

    def test_a_timeout_no_epoch_fits_returns_a_null_row(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        runner = Rosame26BaselineRunner()
        model, report = runner.learn(
            _domain_file(tmp_path, "blocks"),
            [(tmp_path / "x.trajectory", None, tmp_path / "p.pddl")],
            tmp_path,
            timeout_seconds=1,
        )

        assert model is None


class TestTheTimingProbe:
    """The probe only ever raises the count, and only when the budget binds.

    Lowering it would shrink a budget the pre-flight has already agreed to on
    the strength of a 20-epoch sample, which is the wrong way round: the seeded
    constant is deliberately conservative, and a probe that happens to measure
    high is not evidence enough to overrule it.
    """

    def test_it_does_not_run_when_the_budget_did_not_bind(self) -> None:
        runner = Rosame26BaselineRunner(n_seeds=1)
        epochs, probe = runner._reprojected_epochs(
            "blocksworld", object(), _EPOCH_DEFAULT, 100000, Path("/nowhere")
        )

        assert epochs == _EPOCH_DEFAULT
        assert probe["ran"] is False

    def test_it_does_not_run_for_a_control_cell(self) -> None:
        """Gate 7 opts out of the budget; there is nothing to re-project."""
        runner = Rosame26BaselineRunner(
            n_seeds=1, epochs=5000, budget_mode="fixed"
        )
        epochs, probe = runner._reprojected_epochs(
            "blocksworld", object(), 5000, 600, Path("/nowhere")
        )

        assert epochs == 5000
        assert probe["ran"] is False

    def test_a_probe_that_raises_is_not_a_cell_failure(self, tmp_path: Path) -> None:
        """``fold`` is a bare object, so building the probe trainer will fail."""
        runner = Rosame26BaselineRunner(n_seeds=1)
        epochs, probe = runner._reprojected_epochs(
            "blocksworld", object(), 50, 600, tmp_path
        )

        assert epochs == 50
        assert probe["ran"] is False
        assert probe["reason"]

    def test_the_probe_is_short_against_any_budget_it_would_inform(self) -> None:
        assert _PROBE_EPOCHS < _EPOCH_DEFAULT // 10


class TestBudgetModes:
    """Three explicit modes, replacing the implicit epochs+respect_budget pair."""

    def test_the_default_is_preflight(self) -> None:
        assert Rosame26BaselineRunner().budget_mode is BudgetMode.PREFLIGHT

    @pytest.mark.parametrize("mode", ["preflight", "fixed", "converge"])
    def test_each_mode_is_accepted(self, mode) -> None:
        assert Rosame26BaselineRunner(budget_mode=mode).budget_mode.value == mode

    def test_an_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown budget_mode"):
            Rosame26BaselineRunner(budget_mode="whenever")

    def test_preflight_gets_no_mode_suffix(self, tmp_path: Path) -> None:
        """The grid's mode is the default, so its rows stay unlabelled."""
        runner = Rosame26BaselineRunner(budget_mode="preflight")
        assert runner.row_name(_domain_file(tmp_path, "blocks")) == "ROSAME-I_26"

    @pytest.mark.parametrize("mode", ["fixed", "converge"])
    def test_the_other_modes_are_suffixed(self, tmp_path: Path, mode) -> None:
        """Two modes measure different things and must never share a row."""
        runner = Rosame26BaselineRunner(budget_mode=mode)
        assert runner.row_name(_domain_file(tmp_path, "blocks")).endswith(
            f"__mode={mode}"
        )

    def test_gate_sevens_row_name_carries_both(self, tmp_path: Path) -> None:
        runner = Rosame26BaselineRunner(epochs=5000, budget_mode="fixed")
        assert (
            runner.row_name(_domain_file(tmp_path, "blocks"))
            == "ROSAME-I_26__ep=5000__mode=fixed"
        )

    def test_only_converge_installs_a_stop_check(self) -> None:
        """preflight and fixed run their full count and emit the final epoch."""
        for mode in ("preflight", "fixed"):
            runner = Rosame26BaselineRunner(budget_mode=mode)
            assert runner._stop_check(_BestModelTracker(False), {}) is None
        assert (
            Rosame26BaselineRunner(budget_mode="converge")._stop_check(
                _BestModelTracker(True), {}
            )
            is not None
        )


class TestBestModelTracker:
    """Active only in converge mode, per the decision to leave (a) and (b) alone."""

    def test_it_is_inert_when_inactive(self) -> None:
        tracker = _BestModelTracker(False)
        tracker.observe(1.0, 0)
        assert tracker.best_loss is None
        assert tracker.best_epoch is None

    def test_it_records_nothing_before_a_model_is_bound(self) -> None:
        tracker = _BestModelTracker(True)
        tracker.observe(1.0, 0)
        assert tracker.best_loss is None

    def test_restoring_without_a_snapshot_is_a_no_op(self) -> None:
        _BestModelTracker(True).restore(object())


class TestSeedSelection:
    def test_it_trains_several_seeds_by_default(self) -> None:
        """The 24 arm's rule: n independent models, keep the lowest final loss."""
        assert Rosame26BaselineRunner().n_seeds == 3

    def test_seeds_are_derived_from_the_base(self) -> None:
        runner = Rosame26BaselineRunner(n_seeds=3, base_seed=100)
        assert [runner.base_seed + i for i in range(runner.n_seeds)] == [100, 101, 102]


class TestSimulationModeIsSkippedNotFailed:
    def test_a_fold_with_no_images_returns_a_null_row(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        runner = Rosame26BaselineRunner()
        model, report = runner.learn(
            _domain_file(tmp_path, "blocks"), [], tmp_path, timeout_seconds=1
        )

        assert model is None
        assert report == {}
        assert "simulation-mode" in capsys.readouterr().out
