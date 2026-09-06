"""The ROSAME+MILP base-loss divisor is a knob, not a checkout."""

from benchmark.algorithm_adapters.rosame_milp.milp_loop import base_loss_divisor


def test_normalised_divides_by_the_steps_transitions():
    assert base_loss_divisor(15, True) == 15


def test_normalised_never_divides_by_zero():
    assert base_loss_divisor(0, True) == 1


def test_raw_sums_leave_the_terms_untouched():
    assert base_loss_divisor(15, False) == 1
