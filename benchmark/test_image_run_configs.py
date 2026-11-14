"""The image run configs: accepted by the batch runner, and the arms they select.

    python -m pytest benchmark/test_image_run_configs.py -q
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from benchmark import benchmark_runner
from benchmark.algorithms import cdps_family_names

SMALL = Path("benchmark/run_config_small_image.yaml")
SMALL_N9 = Path("benchmark/run_config_small_image_n9.yaml")
LARGE = Path("benchmark/run_config_large_image.yaml")
LARGE_SIMULATED = Path("benchmark/run_config_large.yaml")
REFERENCE_DOMAIN = Path("src/domains/hanoi/hanoi.pddl")
BASELINE_LABELS = ["PISAM", "ROSAME_24", "ROSAME_MILP_24"]


def _labels(config_path: Path) -> List[str]:
    """Every row label a config's cells carry, CDPS family first."""
    shared = benchmark_runner._load_config(config_path)["shared"]
    kwargs = benchmark_runner._build_main_kwargs(shared)
    family = cdps_family_names(
        kwargs["run_cdps"], kwargs["run_cdps_anchored"], kwargs["run_pisam_milp"],
        kwargs["run_pisam_milp_loop"], kwargs["pisam_milp_configs"],
    )
    return family + [runner.row_name(REFERENCE_DOMAIN) for runner in kwargs["baselines"]]


@pytest.mark.parametrize("config_path, milp_label", [
    (SMALL, "PISAM_MILP_LOOP__schema=pre_and_add"),
    (SMALL_N9, "PISAM_MILP_LOOP__schema=pre_and_add"),
    (LARGE, "PISAM_MILP_LOOP__schema=pre_and_add__m=4"),
])
def test_a_config_selects_the_four_symbolic_arms(config_path: Path, milp_label: str) -> None:
    assert _labels(config_path) == [milp_label, *BASELINE_LABELS]


@pytest.mark.parametrize("config_path", [SMALL, SMALL_N9, LARGE])
def test_a_config_is_an_image_config_that_resumes_and_caps_one_solve(config_path: Path) -> None:
    config = benchmark_runner._load_config(config_path)
    shared = config["shared"]
    assert config["source"] == "image"
    assert shared["resume"] is True
    assert shared["normalize"] is False
    assert shared["gt_rates"] == [0]
    assert shared["pisam_milp"]["time_limit_seconds"] == 60
    assert shared["pisam_milp"]["gt_anchoring"] == "init_only"
    assert len(benchmark_runner._expand_cells(config)) == len(config["domains"])


def test_the_two_small_configs_differ_only_in_sizes_and_domains() -> None:
    small = benchmark_runner._load_config(SMALL)
    small_n9 = benchmark_runner._load_config(SMALL_N9)
    assert small["run_name"] == small_n9["run_name"]
    assert small["shared"]["num_trajectories"] == [3, 4, 5, 6, 7, 8]
    assert small_n9["shared"]["num_trajectories"] == [3, 4, 5, 6, 7]
    without_sizes = lambda shared: {k: v for k, v in shared.items() if k != "num_trajectories"}
    assert without_sizes(small["shared"]) == without_sizes(small_n9["shared"])
    assert {d["domain_key"] for d in small["domains"]} == {"blocksworld", "hanoi", "npuzzle"}
    assert {d["domain_key"] for d in small_n9["domains"]} == {"depot", "gripper"}


def test_the_large_image_config_keeps_the_large_sweep_settings() -> None:
    image = benchmark_runner._load_config(LARGE)["shared"]
    simulated = benchmark_runner._load_config(LARGE_SIMULATED)["shared"]
    changed = {"num_trajectories", "algorithms", "pisam_milp", "nolam_noise",
               "nolam_allow_neg_precs", "nolam_seed"}
    for key in set(simulated) - changed:
        assert image[key] == simulated[key], key
    assert image["num_trajectories"] == [100]
    for key, value in simulated["pisam_milp"].items():
        if key not in {"ablations", "time_limit_seconds"}:
            assert image["pisam_milp"][key] == value, key
    assert image["pisam_milp"]["schema_nonempty"] == "pre_and_add"
