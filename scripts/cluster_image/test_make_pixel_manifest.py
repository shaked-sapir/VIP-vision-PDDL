"""Tests for :mod:`scripts.cluster_image.make_pixel_manifest`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.cluster_image.make_pixel_manifest import (
    experiment_folds,
    manifest_rows,
    repo_relative_data_dir,
    write_manifest,
)


def _experiment(root: Path, name: str, data_dir: str, cells: list[str]) -> Path:
    experiment = root / name
    (experiment / "evaluation_results").mkdir(parents=True)
    (experiment / "evaluation_results" / "run_params.json").write_text(
        json.dumps({"data_dir": data_dir})
    )
    for cell in cells:
        (experiment / "testing" / cell).mkdir(parents=True)
    return experiment


def test_a_recorded_path_from_another_checkout_is_made_repo_relative() -> None:
    recorded = "/Users/x/VIP-vision-PDDL-depot-rerun/benchmark/data/depot/corpus"
    assert repo_relative_data_dir(recorded) == Path("benchmark/data/depot/corpus")


def test_a_path_outside_benchmark_data_raises() -> None:
    with pytest.raises(ValueError, match="not under benchmark/data/"):
        repo_relative_data_dir("/tmp/elsewhere/corpus")


def test_folds_are_read_off_the_cell_names(tmp_path: Path) -> None:
    experiment = _experiment(
        tmp_path, "exp", "/a/benchmark/data/d/c",
        ["fold0_numtrajs3_gtrate0", "fold0_numtrajs4_gtrate0", "fold2_numtrajs3_gtrate0"],
    )
    (experiment / "testing" / "notes").mkdir()
    assert experiment_folds(experiment) == [0, 2]


def test_one_row_per_domain_and_fold(tmp_path: Path) -> None:
    experiments = {
        "hanoi": _experiment(tmp_path, "h", "/a/benchmark/data/hanoi/c", ["fold1_numtrajs3_gtrate0"]),
        "depot": _experiment(tmp_path, "d", "/b/benchmark/data/depot/c",
                             ["fold0_numtrajs3_gtrate0", "fold1_numtrajs3_gtrate0"]),
    }
    assert manifest_rows(experiments) == [
        ("depot", "benchmark/data/depot/c", "d", 0),
        ("depot", "benchmark/data/depot/c", "d", 1),
        ("hanoi", "benchmark/data/hanoi/c", "h", 1),
    ]
    assert manifest_rows(experiments, domains=["hanoi"]) == [
        ("hanoi", "benchmark/data/hanoi/c", "h", 1)
    ]


def test_the_manifest_has_the_header_the_template_skips(tmp_path: Path) -> None:
    out = tmp_path / "m.csv"
    write_manifest([("depot", "benchmark/data/depot/c", "d", 0)], out)
    assert out.read_text().splitlines() == [
        "domain,data_dir,experiment,fold",
        "depot,benchmark/data/depot/c,d,0",
    ]
