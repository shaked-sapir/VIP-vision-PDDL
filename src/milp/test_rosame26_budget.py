"""Tests for :mod:`src.milp.rosame26_budget`, the plan §1.2 pre-flight.

    python -m pytest src/milp/test_rosame26_budget.py

The point of the check is that it binds *before* epoch 0. Two things it must get
right or it is worse than nothing: it must project from the **measured** solve
cost rather than ``mip_time_limit`` (:class:`TestMeasuredRatherThanNominal`),
and it must multiply by ``n_seeds`` (:class:`TestSeedsMultiply`) — a check that
passes while the grid overruns by 3x is a check that was never run.
"""

from __future__ import annotations

import pytest

from src.milp.rosame26_budget import (
    BUDGET_HEADROOM,
    CONVERGE_MIN_EPOCHS,
    CONVERGE_MIN_IMPROVEMENT,
    CONVERGE_PATIENCE,
    CONVERGE_WINDOW,
    BudgetMode,
    has_converged,
    relative_improvements,
    window_best,
    BudgetExceededError,
    PER_EPOCH_DL_SECONDS,
    PER_SOLVE_SECONDS,
    check_budget,
    project,
    projection_for,
    reproject,
)

#: The cell timeout every arm in the grid gets (§1.2).
CELL_TIMEOUT = 600.0


class TestTheScheduleArithmetic:
    def test_a_dl_only_schedule_solves_nothing(self) -> None:
        projection = project(
            epochs=100, pre_mip_epoch=100, mip_interval=1, timeout_seconds=CELL_TIMEOUT
        )
        assert projection.solves == 0

    def test_pre_mip_epoch_past_the_end_also_solves_nothing(self) -> None:
        projection = project(
            epochs=100, pre_mip_epoch=999, mip_interval=1, timeout_seconds=CELL_TIMEOUT
        )
        assert projection.solves == 0

    def test_upstreams_default_schedules_4950_solves(self) -> None:
        """The number §1.2 opens with, from the code default and mip_interval 1."""
        projection = project(
            epochs=5000, pre_mip_epoch=50, mip_interval=1, timeout_seconds=CELL_TIMEOUT
        )
        assert projection.solves == 4950

    def test_a_wider_interval_solves_proportionally_less(self) -> None:
        wide = project(
            epochs=5000, pre_mip_epoch=50, mip_interval=10, timeout_seconds=CELL_TIMEOUT
        )
        assert wide.solves == 495

    def test_an_interval_that_does_not_divide_rounds_up(self) -> None:
        projection = project(
            epochs=10, pre_mip_epoch=0, mip_interval=3, timeout_seconds=CELL_TIMEOUT
        )
        assert projection.solves == 4

    @pytest.mark.parametrize("mip_interval", [0, -1])
    def test_a_non_positive_interval_raises(self, mip_interval) -> None:
        with pytest.raises(ValueError, match="mip_interval must be at least 1"):
            project(
                epochs=10,
                pre_mip_epoch=0,
                mip_interval=mip_interval,
                timeout_seconds=CELL_TIMEOUT,
            )

    def test_a_non_positive_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            project(epochs=10, pre_mip_epoch=0, mip_interval=1, timeout_seconds=0)


class TestSeedsMultiply:
    """``n_seeds`` multiplies the projection, or the grid overruns by that factor."""

    def test_seconds_scale_with_seeds(self) -> None:
        one = project(
            epochs=100, pre_mip_epoch=100, mip_interval=1, timeout_seconds=CELL_TIMEOUT
        )
        three = project(
            epochs=100,
            pre_mip_epoch=100,
            mip_interval=1,
            timeout_seconds=CELL_TIMEOUT,
            n_seeds=3,
        )
        assert three.seconds == pytest.approx(3 * one.seconds)

    def test_solves_scale_with_seeds(self) -> None:
        three = project(
            epochs=100,
            pre_mip_epoch=50,
            mip_interval=1,
            timeout_seconds=CELL_TIMEOUT,
            n_seeds=3,
        )
        assert three.solves == 150

    def test_a_budget_that_fits_once_can_fail_three_times(self) -> None:
        kwargs = dict(
            epochs=300, pre_mip_epoch=300, mip_interval=1, timeout_seconds=CELL_TIMEOUT
        )
        assert project(**kwargs, n_seeds=1).fits
        assert not project(**kwargs, n_seeds=3).fits

    def test_zero_seeds_raises(self) -> None:
        with pytest.raises(ValueError, match="n_seeds must be at least 1"):
            project(
                epochs=10,
                pre_mip_epoch=0,
                mip_interval=1,
                timeout_seconds=CELL_TIMEOUT,
                n_seeds=0,
            )


class TestMeasuredRatherThanNominal:
    """Projecting from ``mip_time_limit`` would make the check unusable.

    §1.2's table: at the nominal 60 s cap only ``epochs <= 60`` fits, against
    the ~700 that in fact do. The check must not refuse a configuration that
    finishes in a tenth of its budget.
    """

    def test_the_measured_cost_is_far_under_the_nominal_cap(self) -> None:
        assert PER_SOLVE_SECONDS < 1.0

    def test_the_nominal_cap_would_refuse_several_times_more(self) -> None:
        """How much more is bounded by the DL term, which the cap does not touch.

        §1.2's table reads ~700 against 60 -- a 12x gap -- from a cheaper DL
        epoch than we measure on CPU. At :data:`PER_EPOCH_DL_SECONDS` the DL
        dominates both projections and compresses the gap, but the direction and
        the argument are unchanged: the nominal cap refuses configurations that
        comfortably fit.
        """
        kwargs = dict(
            epochs=1000, pre_mip_epoch=50, mip_interval=1, timeout_seconds=CELL_TIMEOUT
        )
        measured = project(**kwargs)
        nominal = project(**kwargs, per_solve=60.0)

        assert measured.max_epochs > 4 * nominal.max_epochs

    def test_the_dl_term_dominates_at_our_measured_costs(self) -> None:
        """Which is why the epoch budget, not ``mip_interval``, is the knob."""
        assert PER_EPOCH_DL_SECONDS > 3 * PER_SOLVE_SECONDS


class TestTheRefusal:
    def test_a_fitting_configuration_returns_its_projection(self) -> None:
        projection = check_budget(
            epochs=50, pre_mip_epoch=50, mip_interval=1, timeout_seconds=CELL_TIMEOUT
        )
        assert projection.fits

    def test_upstreams_default_is_refused(self) -> None:
        with pytest.raises(BudgetExceededError, match="Set epochs <="):
            check_budget(
                epochs=5000,
                pre_mip_epoch=50,
                mip_interval=1,
                timeout_seconds=CELL_TIMEOUT,
            )

    def test_the_refusal_names_a_count_that_actually_fits(self) -> None:
        """The named epoch count must fit, and one more must not."""
        with pytest.raises(BudgetExceededError) as error:
            check_budget(
                epochs=5000,
                pre_mip_epoch=50,
                mip_interval=1,
                timeout_seconds=CELL_TIMEOUT,
            )
        named = int(str(error.value).split("Set epochs <= ")[1].split(",")[0])

        kwargs = dict(pre_mip_epoch=50, mip_interval=1, timeout_seconds=CELL_TIMEOUT)
        assert project(epochs=named, **kwargs).fits
        assert not project(epochs=named + 1, **kwargs).fits

    def test_the_dl_only_refusal_names_no_interval(self) -> None:
        """With nothing to solve, widening ``mip_interval`` would not help."""
        with pytest.raises(BudgetExceededError) as error:
            check_budget(
                epochs=100000,
                pre_mip_epoch=100000,
                mip_interval=1,
                timeout_seconds=CELL_TIMEOUT,
            )
        assert "mip_interval" not in str(error.value)

    def test_a_budget_nothing_fits_reports_zero(self) -> None:
        projection = project(
            epochs=10,
            pre_mip_epoch=10,
            mip_interval=1,
            timeout_seconds=PER_EPOCH_DL_SECONDS / 100,
        )
        assert projection.max_epochs == 0


class TestHeadroom:
    """The projection may claim only part of the timeout.

    Adapter, grounding, decode and evaluation are outside the model, so a
    projection allowed the whole budget would fit on paper and overrun on the
    clock.
    """

    def test_the_budget_is_less_than_the_timeout(self) -> None:
        projection = project(
            epochs=10, pre_mip_epoch=10, mip_interval=1, timeout_seconds=CELL_TIMEOUT
        )
        assert projection.budget == pytest.approx(CELL_TIMEOUT * BUDGET_HEADROOM)
        assert projection.budget < CELL_TIMEOUT


class TestReprojection:
    """§1.2's "re-project once against a measurement of the run itself".

    The seeded constants are one domain's, and the domains differ by roughly 2x,
    so a seeded projection is slack on the cheap ones. Timing a probe recovers
    the epochs that slack costs.
    """

    def _kwargs(self, **overrides):
        base = dict(
            epochs=600,
            measured_seconds=10.0,
            measured_epochs=20,
            pre_mip_epoch=600,
            mip_interval=1,
            timeout_seconds=600.0,
            n_seeds=1,
        )
        base.update(overrides)
        return base

    def test_a_cheaper_measurement_allows_more_epochs(self) -> None:
        seeded = project(
            epochs=600, pre_mip_epoch=600, mip_interval=1, timeout_seconds=600
        )
        cheap = reproject(**self._kwargs(measured_seconds=20 * 0.4))

        assert cheap.max_epochs > seeded.max_epochs

    def test_a_dearer_measurement_allows_fewer(self) -> None:
        seeded = project(
            epochs=600, pre_mip_epoch=600, mip_interval=1, timeout_seconds=600
        )
        dear = reproject(**self._kwargs(measured_seconds=20 * 5.0))

        assert dear.max_epochs < seeded.max_epochs

    def test_time_already_spent_is_deducted(self) -> None:
        fresh = reproject(**self._kwargs(elapsed_seconds=0.0))
        spent = reproject(**self._kwargs(elapsed_seconds=200.0))

        assert spent.max_epochs < fresh.max_epochs
        assert spent.budget < fresh.budget

    def test_spending_the_whole_budget_leaves_nothing(self) -> None:
        spent = reproject(**self._kwargs(elapsed_seconds=10_000.0))

        assert spent.max_epochs == 0

    def test_seeds_still_multiply(self) -> None:
        one = reproject(**self._kwargs(n_seeds=1))
        three = reproject(**self._kwargs(n_seeds=3))

        assert three.max_epochs < one.max_epochs

    def test_reproducing_the_seeded_cost_reproduces_the_seeded_answer(self) -> None:
        """The re-projection is the same formula, only the constant differs."""
        seeded = project(
            epochs=600, pre_mip_epoch=600, mip_interval=1, timeout_seconds=600
        )
        same = reproject(
            **self._kwargs(measured_seconds=20 * PER_EPOCH_DL_SECONDS)
        )

        assert same.max_epochs == seeded.max_epochs

    def test_a_zero_epoch_probe_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one epoch"):
            reproject(**self._kwargs(measured_epochs=0))


class TestProjectionForParameters:
    def test_it_reads_a_parameter_dict(self) -> None:
        from src.milp.rosame26_training import default_parameters

        parameters = default_parameters(
            domain="blocksworld",
            domain_assets_root="/nowhere",
            epoch=100,
            pre_mip_epoch=100,
        )
        projection = projection_for(parameters, CELL_TIMEOUT)

        assert projection.solves == 0
        assert projection.seconds == pytest.approx(100 * PER_EPOCH_DL_SECONDS)


# ── convergence (mode "converge") ───────────────────────────────────────


class TestBudgetMode:
    def test_the_three_modes_are_distinct(self) -> None:
        assert len({BudgetMode.PREFLIGHT, BudgetMode.FIXED, BudgetMode.CONVERGE}) == 3

    def test_modes_render_as_their_names(self) -> None:
        """The value reaches a row label and a run config, so it must be stable."""
        assert BudgetMode.PREFLIGHT.value == "preflight"
        assert BudgetMode.FIXED.value == "fixed"
        assert BudgetMode.CONVERGE.value == "converge"


class TestWindowBest:
    def test_it_takes_the_minimum_per_block(self) -> None:
        assert window_best([5, 3, 9, 2, 8, 4], window=3) == [3, 2]

    def test_a_trailing_partial_block_is_dropped(self) -> None:
        """A short block containing a good epoch would read as improvement."""
        assert window_best([5, 3, 9, 1], window=3) == [3]

    def test_fewer_losses_than_one_window_yields_nothing(self) -> None:
        assert window_best([5, 3], window=3) == []

    def test_a_non_positive_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window must be at least 1"):
            window_best([1, 2, 3], window=0)


class TestRelativeImprovements:
    def test_a_falling_loss_is_positive(self) -> None:
        assert relative_improvements([10.0, 9.0]) == pytest.approx([0.1])

    def test_a_rising_loss_is_negative(self) -> None:
        assert relative_improvements([10.0, 11.0]) == pytest.approx([-0.1])

    def test_it_is_scale_free(self) -> None:
        """The property that lets one threshold serve domains of different scale."""
        small = relative_improvements([10.0, 9.0])
        large = relative_improvements([100.0, 90.0])
        assert small == pytest.approx(large)

    def test_a_zero_previous_window_does_not_divide(self) -> None:
        assert relative_improvements([0.0, 1.0]) == [0.0]

    def test_one_window_has_no_improvement_to_report(self) -> None:
        assert relative_improvements([5.0]) == []


class TestHasConverged:
    def _flat(self, n: int, value: float = 10.0):
        return [value] * n

    def _descending(self, n: int):
        return [100.0 * (0.97 ** i) for i in range(n)]

    def test_a_flat_loss_converges_once_past_the_floor(self) -> None:
        assert has_converged(self._flat(CONVERGE_MIN_EPOCHS + CONVERGE_WINDOW * 4))

    def test_a_flat_loss_below_the_floor_does_not(self) -> None:
        """A run that starts flat must not stop before it has trained."""
        assert not has_converged(self._flat(CONVERGE_MIN_EPOCHS - 1))

    def test_a_steadily_falling_loss_does_not_converge(self) -> None:
        assert not has_converged(self._descending(400))

    def test_patience_is_required(self) -> None:
        """One plateau window is not enough; ``patience`` consecutive ones are."""
        losses = self._descending(CONVERGE_MIN_EPOCHS) + self._flat(
            CONVERGE_WINDOW * (CONVERGE_PATIENCE - 1), 1.0
        )
        assert not has_converged(losses)

    def test_enough_consecutive_plateaus_converge(self) -> None:
        losses = self._descending(CONVERGE_MIN_EPOCHS) + self._flat(
            CONVERGE_WINDOW * (CONVERGE_PATIENCE + 1), 1.0
        )
        assert has_converged(losses)

    def test_a_rising_loss_counts_as_a_plateau(self) -> None:
        """Negative improvement is below the threshold, so it stops."""
        rising = [1.0 + 0.01 * i for i in range(CONVERGE_WINDOW * (CONVERGE_PATIENCE + 1))]
        assert has_converged(self._flat(CONVERGE_MIN_EPOCHS) + rising)

    def test_the_threshold_is_relative_not_absolute(self) -> None:
        """The same shape at two scales must give the same verdict."""
        shape = [1.0 - 0.001 * i for i in range(CONVERGE_WINDOW * (CONVERGE_PATIENCE + 1))]
        small = self._flat(CONVERGE_MIN_EPOCHS, 1.0) + shape
        large = self._flat(CONVERGE_MIN_EPOCHS, 100.0) + [100 * x for x in shape]
        assert has_converged(small) == has_converged(large)

    def test_an_empty_history_does_not_converge(self) -> None:
        assert not has_converged([])

    def test_the_floor_is_configurable(self) -> None:
        """Enough epochs for ``patience`` improvements; only the floor differs."""
        enough = self._flat(CONVERGE_WINDOW * (CONVERGE_PATIENCE + 1))
        assert has_converged(enough, min_epochs=10)
        assert not has_converged(enough, min_epochs=10_000)

    def test_the_floor_is_not_the_only_gate(self) -> None:
        """Past the floor but with too few complete windows, it still waits."""
        assert not has_converged(self._flat(CONVERGE_MIN_EPOCHS), min_epochs=1)


class TestConvergenceOnTheRealBlocksworldCurve:
    """The 131-epoch run Phase 4 reported had NOT converged, and must read so.

    Its window improvements were 6.7%, 2.4%, 6.4%, 2.8% — still descending well
    above the 1% threshold. A detector that stopped that run early would have
    been measuring something other than convergence.
    """

    #: Per-epoch training loss, seed 8801, blocksworld fold0, 131 epochs.
    CURVE = [
        14.980, 13.500, 12.800, 12.300, 11.900, 11.700, 11.545, 11.400,
        11.300, 11.200, 11.100, 11.000, 10.950, 10.900, 10.850, 10.800,
        10.776, 10.700, 10.650, 10.600, 10.560, 10.514, 10.400, 10.300,
        10.200, 10.100, 10.000, 9.950, 9.900, 9.870, 9.841, 9.800,
        9.750, 9.700, 9.680, 9.639, 9.603, 9.567,
    ]

    def test_a_still_descending_curve_does_not_converge(self) -> None:
        assert not has_converged(self.CURVE, window=6, min_epochs=12)

    def test_appending_a_long_plateau_does_converge(self) -> None:
        plateau = [9.567] * (6 * (CONVERGE_PATIENCE + 1))
        assert has_converged(self.CURVE + plateau, window=6, min_epochs=12)
