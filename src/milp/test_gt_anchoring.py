"""Which states each MILP anchoring mode hard-fixes.

    python -m pytest src/milp/test_gt_anchoring.py
"""

import pytest

from src.milp.converter import GtAnchoring


@pytest.mark.parametrize("mode, indices, expected", [
    (GtAnchoring.NONE, {0, 3}, False),
    (GtAnchoring.INIT_ONLY, None, True),
    (GtAnchoring.ALL_GT_STATES, {3}, True),
    (GtAnchoring.GT_STATES_ONLY, {3}, False),
    (GtAnchoring.GT_STATES_ONLY, {0, 3}, True),
    (GtAnchoring.GT_STATES_ONLY, None, False),
])
def test_anchors_init(mode: GtAnchoring, indices, expected: bool) -> None:
    assert mode.anchors_init(indices) is expected


def test_only_the_two_listing_modes_fix_the_listed_states() -> None:
    assert {m for m in GtAnchoring if m.anchors_listed_states} == {
        GtAnchoring.ALL_GT_STATES, GtAnchoring.GT_STATES_ONLY}
