"""The four ground-truth setups.

    python -m pytest src/utils/test_gt_setup.py
"""

import pytest

from src.utils.gt_setup import GtSetup


@pytest.mark.parametrize("setup, expected", [
    (GtSetup.NONE, set()),
    (GtSetup.INIT, {0}),
    (GtSetup.FINAL, {4}),
    (GtSetup.INIT_FINAL, {0, 4}),
])
def test_state_indices_of_a_five_state_trajectory(setup: GtSetup, expected: set) -> None:
    assert setup.state_indices(5) == expected


def test_a_one_state_trajectory_has_one_endpoint() -> None:
    assert GtSetup.INIT_FINAL.state_indices(1) == {0}
    assert GtSetup.NONE.state_indices(0) == set()


def test_init_is_the_default_and_has_no_suffix() -> None:
    assert GtSetup.parse(None) is GtSetup.INIT
    assert GtSetup.INIT.name_suffix == ""
    assert GtSetup.INIT_FINAL.name_suffix == "__setup=init_final"


def test_parse_accepts_values_and_members_and_rejects_others() -> None:
    assert GtSetup.parse("final") is GtSetup.FINAL
    assert GtSetup.parse(GtSetup.NONE) is GtSetup.NONE
    with pytest.raises(ValueError, match="unknown gt_setup"):
        GtSetup.parse("both")
