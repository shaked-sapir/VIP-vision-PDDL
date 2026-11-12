"""The experiment runner's check that a MILP arm fixes the states its setup gives.

    python -m pytest benchmark/test_gt_setup_params.py
"""

from dataclasses import replace

import pytest

from benchmark.experiment_runner import MILP_ANCHORING_BY_SETUP, _gt_setup_params
from benchmark.experiment_running_helpers.data_source import ImageDataSource
from src.milp.converter import GtAnchoring
from src.plan_denoising.milp_denoiser.config import PisamMilpConfig
from src.utils.gt_setup import GtSetup


def _config(anchoring: GtAnchoring) -> PisamMilpConfig:
    return replace(PisamMilpConfig.from_dict({}), gt_anchoring=anchoring)


def test_every_setup_has_an_anchoring() -> None:
    assert set(MILP_ANCHORING_BY_SETUP) == set(GtSetup)


def test_init_adds_no_run_parameter() -> None:
    assert _gt_setup_params(ImageDataSource(), [_config(GtAnchoring.NONE)]) == {}


def test_a_matching_anchoring_records_the_setup() -> None:
    source = ImageDataSource(gt_setup=GtSetup.INIT_FINAL)
    assert _gt_setup_params(source, [_config(GtAnchoring.ALL_GT_STATES)]) == {"gt_setup": "init_final"}
    assert _gt_setup_params(source, None) == {"gt_setup": "init_final"}


@pytest.mark.parametrize("setup, anchoring", [
    (GtSetup.INIT_FINAL, GtAnchoring.INIT_ONLY),
    (GtSetup.FINAL, GtAnchoring.ALL_GT_STATES),
    (GtSetup.NONE, GtAnchoring.INIT_ONLY),
])
def test_a_mismatched_anchoring_raises(setup: GtSetup, anchoring: GtAnchoring) -> None:
    with pytest.raises(ValueError, match="needs pisam_milp.gt_anchoring"):
        _gt_setup_params(ImageDataSource(gt_setup=setup), [_config(anchoring)])
