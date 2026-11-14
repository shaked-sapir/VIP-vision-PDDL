"""Listing the dashboard's image variants from the experiments' records.

    python -m pytest benchmark/test_image_dashboard.py
"""

from pathlib import Path

import pytest

from benchmark.image_dashboard import (
    display_name,
    image_only_config,
    image_variants,
    variant_key,
    variant_label,
)
from src.utils.atomic_json import write_json_atomic
from src.utils.gt_setup import GtSetup

FAMILY = "TO=600__image-small"
OFF_BOTH = {"vlm_model": "gpt-5.2", "reasoning": "off", "example_protocol": "shared", "gt_setup": "init_final"}
MEDIUM_INIT = {"vlm_model": "gpt-5.2", "reasoning": "medium", "example_protocol": "shared", "gt_setup": "init"}


def _experiment(root: Path, domain: str, name: str, axes: dict, rows) -> None:
    experiment = root / domain / name
    (experiment / "evaluation_results").mkdir(parents=True)
    if axes:
        write_json_atomic(experiment / "evaluation_results" / "experiment_variant.json", axes)
    cell = experiment / "testing" / "fold0_numtrajs3_gtrate0"
    cell.mkdir(parents=True)
    write_json_atomic(cell / "fold_result.json", rows)


@pytest.fixture
def results(tmp_path: Path) -> Path:
    root = tmp_path / "results"
    for domain in ("gripper", "depot"):
        _experiment(root, domain, f"{FAMILY}__fs=shared__setup=init_final", OFF_BOTH,
                    [{"algorithm": "PISAM"}, {"algorithm": "ROSAME-I_26__res=native__ep=5000"}])
    _experiment(root, "gripper", f"{FAMILY}__fs=shared__reason=medium", MEDIUM_INIT,
                [{"algorithm": "ROSAME_MILP_24"}, {"algorithm": "PISAM__reasoning=off"}])
    _experiment(root, "gripper", "TO=600__gripper_data_from_PV", {}, [{"algorithm": "CDPS"}])
    return root


def test_key_and_label_name_every_axis() -> None:
    assert variant_key(OFF_BOTH) == "vlm_model=gpt-5.2__reasoning=off__example_protocol=shared__gt_setup=init_final"
    assert variant_label(MEDIUM_INIT) == "gpt-5.2 · reasoning medium · init GT"


@pytest.mark.parametrize("label, setup, row_setup, expected", [
    ("PISAM", GtSetup.INIT_FINAL, None, "PI-SAM (init+final GT)"),
    ("PISAM_MILP_LOOP__gt=all_gt_states", GtSetup.INIT_FINAL, None, "NPI-SAM (init+final GT)"),
    ("ROSAME_24", GtSetup.INIT, None, "ROSAME (init GT)"),
    ("ROSAME_MILP_24", GtSetup.INIT, None, "ROSAME+MILP (init+final GT)"),
    ("ROSAME_MILP_24", GtSetup.INIT, "init", "ROSAME+MILP (init GT)"),
    ("ROSAME-I_MILP_26__res=native__ep=5000", GtSetup.INIT, None, "ROSAME-I 26+MILP (init+final GT, native res)"),
    ("ROSAME-I_24", GtSetup.INIT_FINAL, None, "ROSAME-I 24 (final GT)"),
    ("CDPS", GtSetup.INIT, None, "CDPS"),
])
def test_display_names(label: str, setup: GtSetup, row_setup, expected: str) -> None:
    assert display_name(label, setup, row_setup) == expected


def test_one_variant_per_recorded_axes_across_domains(results: Path) -> None:
    variants = image_variants(results, FAMILY, ["gripper", "depot"])

    assert [v["label"] for v in variants] == [
        "gpt-5.2 · reasoning medium · init GT", "gpt-5.2 · reasoning off · init+final GT"]
    both = variants[1]
    assert sorted(both["experiment_dir"]) == ["depot", "gripper"]
    assert both["include_algorithms"] == ["PISAM"]
    assert both["display_names"]["ROSAME-I_26__res=native__ep=5000"] == "ROSAME-I 26 (init+final GT, native res)"


def test_rows_relabelled_with_a_data_variant_are_hidden(results: Path) -> None:
    medium = image_variants(results, FAMILY, ["gripper"])[0]

    assert medium["exclude_algorithms"] == ["PISAM__reasoning=off"]
    assert "PISAM__reasoning=off" not in medium["display_names"]
    assert medium["display_names"]["ROSAME_MILP_24"] == "ROSAME+MILP (init+final GT)"


def test_image_only_config_drops_simulation_and_opens_on_the_image_arms() -> None:
    config = {"results_dashboard": {"output_html": "a.html", "domains": ["gripper"],
                                    "simulation": {"small": {"prefix": {"gripper": "sim"}}},
                                    "selectors": {"default": {"mode": "simulation", "data_size": "small"}}}}

    trimmed = image_only_config(config, Path("out/image.html"))["results_dashboard"]

    assert trimmed["output_html"] == "out/image.html"
    assert all(block["prefix"] == {"gripper": "__not-in-this-page__"} for block in trimmed["simulation"].values())
    assert set(trimmed["simulation"]) == {"small", "large"}
    assert trimmed["selectors"]["default"] == {"mode": "image", "data_size": "small", "view": "baselines"}
    assert config["results_dashboard"]["simulation"]["small"]["prefix"] == {"gripper": "sim"}
