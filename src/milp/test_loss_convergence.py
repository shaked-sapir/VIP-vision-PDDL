"""The training-loss plateau rule, on synthetic curves.

    python -m pytest src/milp/test_loss_convergence.py
"""

from __future__ import annotations

import random

import pytest

from src.milp.loss_convergence import LossConvergenceRule, has_converged

RULE = LossConvergenceRule(window=10, min_improvement=0.002, patience=3, min_epochs=50)


def _descent(epochs: int, start: float = 10.0, rate: float = 0.99) -> list:
    """A curve that keeps improving by 1% per epoch."""
    return [start * rate ** i for i in range(epochs)]


def _noisy_plateau(epochs: int, level: float, noise: float, seed: int = 0) -> list:
    rng = random.Random(seed)
    return [level * (1 + rng.uniform(-noise, noise)) for _ in range(epochs)]


class TestRule:
    def test_a_steady_descent_never_stops(self):
        curve = _descent(600)
        assert not any(RULE.converged(curve[:n]) for n in range(1, len(curve) + 1))

    def test_a_noisy_plateau_stops_at_the_expected_block(self):
        """Noise of +-0.05% sits below the 0.2% threshold, so every plateau block
        fails to beat the running best set by the descent's last block, and the
        rule fires when the third plateau block closes: 3 x 10 = 30 epochs in."""
        descent = _descent(60)
        plateau = _noisy_plateau(200, level=descent[-1] * 0.999, noise=0.0005)
        curve = descent + plateau
        stops = [n for n in range(1, len(curve) + 1) if RULE.converged(curve[:n])]
        assert stops, "the plateau was never detected"
        assert stops[0] == 60 + 3 * RULE.window

    def test_the_floor_holds_back_a_flat_start(self):
        flat = [1.0] * 49
        assert not RULE.converged(flat)
        assert RULE.converged([1.0] * 50)

    def test_fewer_blocks_than_patience_cannot_stop(self):
        rule = LossConvergenceRule(window=10, patience=3, min_epochs=0)
        assert not rule.converged([1.0] * 39)   # 3 blocks -> 2 comparisons
        assert rule.converged([1.0] * 40)       # 4 blocks -> 3 comparisons

    def test_relative_so_scale_does_not_matter(self):
        shape = _descent(60) + _noisy_plateau(60, level=5.4, noise=0.0005)
        for scale in (0.01, 1.0, 1000.0):
            scaled = [scale * v for v in shape]
            assert RULE.converged(scaled) == RULE.converged(shape)

    def test_a_single_good_epoch_inside_a_block_is_what_counts(self):
        """Blocks are summarised by their lowest loss, so upticks within a block
        do not read as regress and one real improvement resets the patience."""
        rule = LossConvergenceRule(window=5, min_improvement=0.01, patience=2, min_epochs=0)
        curve = [10, 11, 12, 11, 10,   9, 12, 12, 12, 12,   8, 12, 12, 12, 12]
        assert not rule.converged(curve)


class TestRestartedSeries:
    """The MILP arms score only the post-first-solve series."""

    def test_scoring_across_the_solve_jump_fires_without_training(self):
        warmup = _descent(50, start=2.30, rate=0.9995)           # settles near 2.24
        after = _descent(40, start=2.60, rate=0.999)             # CE joined: higher scale, still improving
        rule = LossConvergenceRule(window=10, min_improvement=0.002, patience=3, min_epochs=50)
        assert rule.converged(warmup + after), (
            "the whole series never beats the warmup's running best, which is the bug "
            "the restart exists to avoid"
        )

    def test_the_restarted_series_does_not(self):
        after = _descent(40, start=2.60, rate=0.999)
        rule = LossConvergenceRule(window=10, min_improvement=0.002, patience=3, min_epochs=30)
        assert not rule.converged(after)


class TestConfig:
    def test_a_missing_block_is_rule_off(self):
        assert LossConvergenceRule.from_config(None) is None

    def test_a_block_overrides_the_defaults_it_names(self):
        rule = LossConvergenceRule.from_config({"window": 40, "min_epochs": 60})
        assert (rule.window, rule.patience, rule.min_epochs) == (40, 3, 60)

    def test_a_typo_fails_loudly(self):
        with pytest.raises(ValueError, match="windw"):
            LossConvergenceRule.from_config({"windw": 10})

    def test_an_instance_passes_through(self):
        assert LossConvergenceRule.from_config(RULE) is RULE

    def test_impossible_values_are_rejected(self):
        with pytest.raises(ValueError):
            LossConvergenceRule(window=0)
        with pytest.raises(ValueError):
            LossConvergenceRule(patience=0)

    def test_as_stats_round_trips(self):
        assert LossConvergenceRule.from_config(RULE.as_stats()) == RULE


def test_the_function_and_the_dataclass_agree():
    curve = _descent(60) + [5.0] * 60
    assert RULE.converged(curve) == has_converged(
        curve, window=10, min_improvement=0.002, patience=3, min_epochs=50
    )
