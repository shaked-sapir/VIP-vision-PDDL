"""Tests for ``backfill_baseline._runner_kwargs``, the per-arm option channel.

    python -m pytest benchmark/test_backfill_runner_kwargs.py

The rule under test: an option meant for one arm must not reach another, and an
option the operator did not pass must leave that runner's own default alone. Both
are how gate 7's control-cell settings stay confined to the ICAPS-26 arm.
"""

from __future__ import annotations

import argparse

import pytest

from benchmark.backfill_baseline import _runner_kwargs
from benchmark.baselines import resolve_baselines
from benchmark.baselines.rosame26_runner import _EPOCH_DEFAULT


def _args(**overrides) -> argparse.Namespace:
    """A namespace with the three flags at their argparse defaults."""
    return argparse.Namespace(
        **{
            "epochs": None,
            "n_seeds": None,
            "ignore_budget": False,
            "budget_mode": None,
            **overrides,
        }
    )


class TestOnlyWhatWasPassedIsForwarded:
    def test_defaults_forward_nothing(self) -> None:
        assert _runner_kwargs(_args()) == {}

    def test_an_epoch_override_is_forwarded(self) -> None:
        assert _runner_kwargs(_args(epochs=5000)) == {"epochs": 5000}

    def test_a_seed_override_is_forwarded(self) -> None:
        assert _runner_kwargs(_args(n_seeds=1)) == {"n_seeds": 1}

    def test_an_explicit_mode_is_forwarded(self) -> None:
        assert _runner_kwargs(_args(budget_mode="converge")) == {
            "budget_mode": "converge"
        }

    def test_the_retained_flag_is_an_alias_for_fixed(self) -> None:
        assert _runner_kwargs(_args(ignore_budget=True)) == {"budget_mode": "fixed"}

    def test_an_explicit_mode_wins_over_the_alias(self) -> None:
        assert _runner_kwargs(
            _args(ignore_budget=True, budget_mode="converge")
        ) == {"budget_mode": "converge"}

    def test_gate_sevens_full_setting(self) -> None:
        assert _runner_kwargs(_args(epochs=5000, n_seeds=1, ignore_budget=True)) == {
            "epochs": 5000,
            "n_seeds": 1,
            "budget_mode": "fixed",
        }


class TestTheOptionsReachOnlyTheArmsThatTakeThem:
    """``resolve_baselines`` filters by ``__init__`` signature.

    ``--epochs`` and ``--ignore-budget`` are ICAPS-26 settings; forwarding them
    to a 24 arm would be a ``TypeError`` if the filter were not doing its job,
    which is what makes this a real check rather than a restatement.
    """

    def test_the_26_arm_takes_all_three(self) -> None:
        runner = resolve_baselines(
            ["rosame_i_26"], **_runner_kwargs(_args(epochs=5000, n_seeds=1, ignore_budget=True))
        )[0]

        assert runner.epochs == 5000
        assert runner.n_seeds == 1
        assert runner.budget_mode.value == "fixed"

    @pytest.mark.parametrize(
        "key", ["rosame_24", "rosame_i_24", "rosame_i_milp_24", "rosame_milp_24"]
    )
    def test_the_other_arms_survive_them(self, key) -> None:
        assert resolve_baselines(
            [key], **_runner_kwargs(_args(epochs=5000, n_seeds=1, ignore_budget=True))
        )

    def test_n_seeds_does_reach_the_24_image_arm(self) -> None:
        """It takes ``n_seeds`` too, and the flag is meant to reach it."""
        runner = resolve_baselines(["rosame_i_24"], **_runner_kwargs(_args(n_seeds=1)))[0]

        assert runner.n_seeds == 1

    def test_no_override_leaves_the_26_arms_defaults(self) -> None:
        runner = resolve_baselines(["rosame_i_26"], **_runner_kwargs(_args()))[0]

        assert runner.epochs is None
        assert runner._resolve_epochs("blocksworld") == _EPOCH_DEFAULT
        assert runner.budget_mode.value == "preflight"
