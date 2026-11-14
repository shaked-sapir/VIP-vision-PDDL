"""Building an image dashboard config from the variant records.

    python -m pytest benchmark/test_image_dashboard.py
"""

from pathlib import Path

import pytest

from benchmark.image_dashboard import (
    build_config,
    classifier_label,
    find_experiments,
    parse_classifier,
    setup_display_names,
)
from benchmark.image_variant_config import RunVariant
from src.utils.atomic_json import write_json_atomic
from src.utils.corpus_variant import CorpusVariant, ExampleProtocol
from src.utils.gt_setup import GtSetup

OFF = CorpusVariant("gpt-5.2", "off", ExampleProtocol.SHARED)
MEDIUM = CorpusVariant("gpt-5.2", "medium", ExampleProtocol.SHARED)
FAMILY = "TO=600__image-small"
TEMPLATE = {"results_dashboard": {
    "domains": ["gripper"], "display_names": {"ROSAME-I_26": "ROSAME-I 26"},
    "selectors": {"data_size": {"label": "Data size", "options": {"small": "Small", "large": "Large"}}},
}}


def _experiment(root: Path, name: str, variant: RunVariant, labels) -> Path:
    experiment = root / "gripper" / name
    (experiment / "evaluation_results").mkdir(parents=True)
    write_json_atomic(experiment / "evaluation_results" / "experiment_variant.json", variant.as_dict())
    cell = experiment / "testing" / "fold0_numtrajs3_gtrate0"
    cell.mkdir(parents=True)
    write_json_atomic(cell / "fold_result.json", [{"algorithm": label} for label in labels])
    return experiment


@pytest.fixture
def results(tmp_path: Path) -> Path:
    root = tmp_path / "results"
    _experiment(root, f"{FAMILY}__fs=shared__setup=init_final", RunVariant(OFF, GtSetup.INIT_FINAL),
                ["PISAM", "PISAM_MILP_LOOP__gt=all_gt_states"])
    _experiment(root, f"{FAMILY}__fs=shared__reason=medium__setup=init_final",
                RunVariant(MEDIUM, GtSetup.INIT_FINAL), ["ROSAME_MILP_24", "ROSAME-I_26"])
    _experiment(root, f"{FAMILY}__fs=shared", RunVariant(OFF, GtSetup.INIT), ["PISAM"])
    return root


def test_parse_classifier() -> None:
    assert parse_classifier("vlm=gpt-5.2,reasoning=medium") == MEDIUM
    assert parse_classifier("vlm=gpt-5.2") == OFF
    with pytest.raises(ValueError, match="--classifier expects"):
        parse_classifier("reasoning=off")
    assert classifier_label(MEDIUM) == "gpt-5.2, reasoning medium"


def test_experiments_are_found_by_record(results: Path) -> None:
    found = find_experiments(results, FAMILY, RunVariant(MEDIUM, GtSetup.INIT_FINAL), ["gripper"])
    assert found["gripper"].endswith("__fs=shared__reason=medium__setup=init_final")
    assert find_experiments(results, FAMILY, RunVariant(MEDIUM, GtSetup.FINAL), ["gripper"]) == {}


def test_only_setup_arms_are_renamed() -> None:
    names = setup_display_names(
        ["PISAM", "PISAM_MILP_LOOP__gt=all_gt_states", "ROSAME_24", "ROSAME_MILP_24", "ROSAME-I_26"],
        GtSetup.INIT_FINAL)
    assert names == {
        "PISAM": "PI-SAM (init+final GT)", "PISAM_MILP_LOOP__gt=all_gt_states": "NPI-SAM (init+final GT)",
        "ROSAME_24": "ROSAME (init+final GT)", "ROSAME_MILP_24": "ROSAME+MILP (init+final GT)"}


def test_config_puts_each_classifier_in_a_slot(results: Path) -> None:
    config = build_config(TEMPLATE, GtSetup.INIT_FINAL, [OFF, MEDIUM], FAMILY, results, "page")
    dashboard = config["results_dashboard"]

    assert dashboard["selectors"]["data_size"] == {
        "label": "Classifier",
        "options": {"small": "gpt-5.2, reasoning off", "large": "gpt-5.2, reasoning medium"}}
    assert dashboard["image"]["small"]["experiment_dir"]["gripper"].endswith("__fs=shared__setup=init_final")
    assert dashboard["image"]["large"]["experiment_dir"]["gripper"].endswith("__reason=medium__setup=init_final")
    assert dashboard["display_names"]["ROSAME_MILP_24"] == "ROSAME+MILP (init+final GT)"
    assert dashboard["display_names"]["ROSAME-I_26"] == "ROSAME-I 26"
    assert TEMPLATE["results_dashboard"]["selectors"]["data_size"]["label"] == "Data size"


def test_a_classifier_without_an_experiment_raises(results: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no experiment"):
        build_config(TEMPLATE, GtSetup.FINAL, [OFF], FAMILY, results, "page")
