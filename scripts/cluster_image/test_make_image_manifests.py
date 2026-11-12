"""Tests for :mod:`scripts.cluster_image.make_image_manifests` and the checked-in manifests."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import pytest
import yaml

from scripts.cluster_image.make_image_manifests import (
    config_rows,
    manifest_rows,
    write_symbolic_manifest,
)

SMALL_CONFIGS = [
    Path("benchmark/run_config_small_image.yaml"),
    Path("benchmark/run_config_small_image_n9.yaml"),
]
LARGE_CONFIGS = [Path("benchmark/run_config_large_image.yaml")]
MANIFEST_DIR = Path("scripts/cluster_image")


def _config(path: Path, run_name: str, domains: List[str], source: str = "image", n_folds: int = 2) -> Path:
    path.write_text(yaml.safe_dump({
        "run_name": run_name,
        "source": source,
        "shared": {"n_folds": n_folds},
        "domains": [{"domain_key": d, "data_dir": f"benchmark/data/{d}/c"} for d in domains],
    }))
    return path


def test_one_row_per_domain_and_fold_in_config_order(tmp_path: Path) -> None:
    config = _config(tmp_path / "a.yaml", "exp", ["hanoi", "depot"])
    assert config_rows(config) == [
        ("hanoi", "benchmark/data/hanoi/c", "exp", 0, config.as_posix()),
        ("hanoi", "benchmark/data/hanoi/c", "exp", 1, config.as_posix()),
        ("depot", "benchmark/data/depot/c", "exp", 0, config.as_posix()),
        ("depot", "benchmark/data/depot/c", "exp", 1, config.as_posix()),
    ]


def test_a_simulated_config_is_refused(tmp_path: Path) -> None:
    config = _config(tmp_path / "a.yaml", "exp", ["hanoi"], source="simulated")
    with pytest.raises(ValueError, match="source must be 'image'"):
        config_rows(config)


def test_a_domain_in_two_configs_of_one_experiment_is_refused(tmp_path: Path) -> None:
    first = _config(tmp_path / "a.yaml", "exp", ["hanoi"])
    second = _config(tmp_path / "b.yaml", "exp", ["hanoi"])
    with pytest.raises(ValueError, match="appears in two configs"):
        manifest_rows([first, second])


def test_the_symbolic_manifest_has_the_header_the_template_skips(tmp_path: Path) -> None:
    out = tmp_path / "m.csv"
    write_symbolic_manifest([("depot", "benchmark/data/depot/c", "exp", 0, "cfg.yaml")], out)
    assert out.read_text().splitlines() == [
        "domain,data_dir,experiment,fold,config",
        "depot,benchmark/data/depot/c,exp,0,cfg.yaml",
    ]


@pytest.mark.parametrize("name, configs, n_rows", [
    ("image_small", SMALL_CONFIGS, 25),
    ("image_large", LARGE_CONFIGS, 10),
])
def test_the_checked_in_manifests_match_their_configs(name: str, configs: List[Path], n_rows: int) -> None:
    rows = manifest_rows(configs)
    assert len(rows) == n_rows

    with (MANIFEST_DIR / f"{name}_symbolic_manifest.csv").open(newline="") as handle:
        symbolic = list(csv.reader(handle))[1:]
    with (MANIFEST_DIR / f"{name}_pixel_manifest.csv").open(newline="") as handle:
        pixel = list(csv.reader(handle))[1:]

    assert symbolic == [[str(value) for value in row] for row in rows]
    assert pixel == [row[:4] for row in symbolic]
