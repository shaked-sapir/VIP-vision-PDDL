"""The NPI-SAM ``schema_nonempty`` knob: config, encoder dialect, row label.

    python -m pytest benchmark/test_schema_nonempty_knob.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from benchmark.algorithms import PISAM_MILP_LOOP, milp_config_for, milp_work_subdir, pisam_milp_algorithm_name
from benchmark.backfill_anchored_solve_cap import read_anchored_config
from benchmark.backfill_nogt_s0 import noisy_initial_label, read_unanchored_config
from src.milp.converter import GtAnchoring
from src.milp.encoding_config import SchemaNonemptyRule
from src.plan_denoising.milp_denoiser.config import PisamMilpConfig

CONFIGS = Path(__file__).resolve().parent / "milp_configs"


def test_off_by_default_and_absent_from_the_label():
    config = PisamMilpConfig.from_dict({})
    assert config.schema_nonempty is SchemaNonemptyRule.NONE
    assert config.encoding_config().schema_nonempty is SchemaNonemptyRule.NONE
    assert "schema" not in pisam_milp_algorithm_name(milp_config_for(PISAM_MILP_LOOP, config))


def test_pre_and_add_reaches_the_encoder_and_only_that_family():
    config = PisamMilpConfig.from_dict({"schema_nonempty": "pre_and_add"})
    encoding = config.encoding_config()
    assert encoding.schema_nonempty is SchemaNonemptyRule.PRE_AND_ADD
    assert encoding.forbid_redundant_adds is False
    assert encoding.delete_implies_precondition is False
    assert config.as_stats()["schema_nonempty"] == "pre_and_add"


def test_the_knob_changes_the_arm_identity_and_the_label():
    plain = milp_config_for(PISAM_MILP_LOOP, PisamMilpConfig.from_dict({"subset_size": 4}))
    on = milp_config_for(PISAM_MILP_LOOP, PisamMilpConfig.from_dict({"subset_size": 4, "schema_nonempty": "pre_and_add"}))
    assert plain.arm_identity() != on.arm_identity()
    assert pisam_milp_algorithm_name(on) == "PISAM_MILP_LOOP__schema=pre_and_add__m=4"
    assert milp_work_subdir(PISAM_MILP_LOOP, on) == "pisam_milp_loop__schema=pre_and_add__m=4"


def test_an_invalid_value_is_refused():
    with pytest.raises(ValueError):
        PisamMilpConfig.from_dict({"schema_nonempty": "sometimes"})


@pytest.mark.parametrize("name, size", [("npisam_schema_nonempty_large.yaml", "4"), ("npisam_schema_nonempty_small.yaml", "half")])
def test_the_shipped_configs_parse_for_both_drivers(name, size):
    path = CONFIGS / name
    assert yaml.safe_load(path.read_text())["pisam_milp"]["schema_nonempty"] == "pre_and_add"
    anchored = milp_config_for(PISAM_MILP_LOOP, read_anchored_config(path))
    unanchored = milp_config_for(PISAM_MILP_LOOP, read_unanchored_config(path))
    assert anchored.gt_anchoring is GtAnchoring.INIT_ONLY
    assert unanchored.gt_anchoring is GtAnchoring.NONE
    assert anchored.schema_nonempty is unanchored.schema_nonempty is SchemaNonemptyRule.PRE_AND_ADD
    assert anchored.subset_size.as_stat() == unanchored.subset_size.as_stat()
    m = "__m=4" if size == "4" else ""
    assert pisam_milp_algorithm_name(anchored) == f"PISAM_MILP_LOOP__schema=pre_and_add{m}"
    assert noisy_initial_label(pisam_milp_algorithm_name(unanchored)) == f"PISAM_MILP_LOOP__gt=none__s0=noisy__schema=pre_and_add{m}"
