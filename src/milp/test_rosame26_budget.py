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
