"""Merging mirrored result roots into the main tree.

    python -m pytest benchmark/test_merge_result_roots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.merge_result_roots import (
    PROVENANCE_FILENAME,
    Source,
    load_sources,
    main,
    merge_source,
    milp_work_dirname,
)

EXPERIMENT = "simulation-final-run__mask=0.1__noise=0.1"
CELL = "fold0_numtrajs5_gtrate0"
FOLD_INFO = {"trajectories": [{"problem": "problem1"}], "test_problems": ["problem2"]}


def _cell(root: Path, rows, fold_info=FOLD_INFO, experiment=EXPERIMENT) -> Path:
    cell = root / "blocksworld" / experiment / "testing" / CELL
    cell.mkdir(parents=True)
    (cell / "fold_info.json").write_text(json.dumps(fold_info))
    (cell / "fold_result.json").write_text(json.dumps(rows))
    return cell


def _row(label: str, value: float) -> dict:
    return {"algorithm": label, "precision_overall": value}


def _labels(cell: Path) -> dict:
    return {r["algorithm"]: r["precision_overall"] for r in json.loads((cell / "fold_result.json").read_text())}


@pytest.fixture
def target(tmp_path) -> Path:
    root = tmp_path / "running_results"
    _cell(root, [_row("ROSAME_24", 0.5), _row("PISAM_MILP_LOOP", 0.6)])
    return root


class TestRows:
    def test_selected_rows_are_added_and_same_label_rows_replaced(self, tmp_path, target):
        root = tmp_path / "running_results_x"
        _cell(root, [_row("PISAM", 0.9), _row("ROSAME_24", 0.7), _row("LEFT_OUT", 0.1)])

        plan = merge_source(Source(root, ("PISAM", "ROSAME_24")), target, apply=True)

        assert _labels(target / "blocksworld" / EXPERIMENT / "testing" / CELL) == {
            "ROSAME_24": 0.7, "PISAM_MILP_LOOP": 0.6, "PISAM": 0.9,
        }
        assert dict(plan.merged) == {"PISAM": 1, "ROSAME_24": 1}
        assert dict(plan.replaced) == {"ROSAME_24": 1}
        assert dict(plan.unselected) == {"LEFT_OUT": 1}

    def test_a_dry_run_changes_nothing(self, tmp_path, target):
        root = tmp_path / "running_results_x"
        _cell(root, [_row("PISAM", 0.9)])
        before = _labels(target / "blocksworld" / EXPERIMENT / "testing" / CELL)

        plan = merge_source(Source(root, ("PISAM",)), target, apply=False)

        assert plan.merged["PISAM"] == 1
        assert _labels(target / "blocksworld" / EXPERIMENT / "testing" / CELL) == before

    def test_the_experiment_filter_limits_the_rows(self, tmp_path, target):
        root = tmp_path / "running_results_x"
        _cell(root, [_row("PISAM", 0.9)])
        plan = merge_source(Source(root, ("PISAM",), experiments="large-corpora__*"), target, apply=True)
        assert plan.merged["PISAM"] == 0
        assert "PISAM" not in _labels(target / "blocksworld" / EXPERIMENT / "testing" / CELL)

    def test_a_cell_of_another_fold_is_skipped(self, tmp_path, target):
        root = tmp_path / "running_results_x"
        _cell(root, [_row("PISAM", 0.9)], fold_info={"trajectories": [{"problem": "problem9"}], "test_problems": ["problem2"]})
        plan = merge_source(Source(root, ("PISAM",)), target, apply=True)
        assert plan.merged["PISAM"] == 0
        assert plan.skipped_cells == [f"blocksworld/{EXPERIMENT}/{CELL}: fold_info differs"]

    def test_a_cell_missing_from_the_target_is_skipped(self, tmp_path, target):
        root = tmp_path / "running_results_x"
        _cell(root, [_row("PISAM", 0.9)], experiment="simulation-final-run__mask=0.4__noise=0.4")
        plan = merge_source(Source(root, ("PISAM",)), target, apply=True)
        assert plan.skipped_cells == [f"blocksworld/simulation-final-run__mask=0.4__noise=0.4/{CELL}: not in target"]


class TestArtifacts:
    def test_label_named_files_come_along_and_provenance_is_written(self, tmp_path, target):
        root = tmp_path / "running_results_x"
        label = "ROSAME_MILP_24__goal=none"
        src = _cell(root, [_row(label, 0.9), _row("PISAM_MILP_LOOP__solve=60", 0.8)])
        (src / f"learned_domain_{label}.pddl").write_text("(define (domain d))")
        (src / "baseline_models" / label).mkdir(parents=True)
        (src / "baseline_models" / label / "model.pddl").write_text("(define (domain d))")
        (src / "rosame_training").mkdir()
        (src / "rosame_training" / "ROSAME_MILP_24.json").write_text("[1, 2]")
        work = src / "pisam_milp_loop__solve=60"
        work.mkdir()
        (work / "milp_loop_rounds.json").write_text("{}")
        (work / "milp_loop_round_models").mkdir()
        (work / "milp_loop_round_models" / "round0.pddl").write_text("big")
        dst = target / "blocksworld" / EXPERIMENT / "testing" / CELL
        (dst / "rosame_training").mkdir()
        (dst / "rosame_training" / "ROSAME_MILP_24.json").write_text("[0]")

        merge_source(Source(root, (label, "PISAM_MILP_LOOP__solve=60")), target, apply=True)

        assert (dst / f"learned_domain_{label}.pddl").is_file()
        assert (dst / "baseline_models" / label / "model.pddl").is_file()
        assert (dst / "rosame_training" / f"{label}.json").read_text() == "[1, 2]"
        assert (dst / "rosame_training" / "ROSAME_MILP_24.json").read_text() == "[0]"
        assert (dst / "pisam_milp_loop__solve=60" / "milp_loop_rounds.json").is_file()
        assert not (dst / "pisam_milp_loop__solve=60" / "milp_loop_round_models").exists()
        provenance = json.loads((dst / PROVENANCE_FILENAME).read_text())
        assert provenance[label]["source_root"] == str(root)
        assert f"learned_domain_{label}.pddl" in provenance[label]["artifacts"]
        assert provenance["PISAM_MILP_LOOP__solve=60"]["artifacts"] == ["pisam_milp_loop__solve=60/milp_loop_rounds.json"]

    def test_work_directory_names_follow_the_label(self):
        assert milp_work_dirname("PISAM_MILP_LOOP__gt=none__s0=noisy__m=4__solve=60") == "pisam_milp_loop__gt=none__s0=noisy__m=4__solve=60"
        assert milp_work_dirname("PISAM_MILP_SR__gt=none__s0=noisy") == "pisam_milp_single_round__gt=none__s0=noisy"
        assert milp_work_dirname("PISAM_MILP_LOOP") == "pisam_milp_loop"
        assert milp_work_dirname("ROSAME_24__s0=noisy") is None


class TestConfig:
    def test_the_paper_config_parses_and_keeps_the_rechecks_last(self):
        config = Path(__file__).resolve().parent / "paper_result_sources.yaml"
        target, sources = load_sources(config)
        assert target == Path("benchmark/running_results")
        assert all(s.labels for s in sources)
        assert [s.root.name for s in sources[-2:]] == [
            "running_results_baselines_nogt_s0_recheck_rosame",
            "running_results_baselines_nogt_s0_recheck_lamanna",
        ]
        plain = next(s for s in sources if s.root.name == "running_results_baselines_nogt_s0")
        assert set(sources[-2].labels) | set(sources[-1].labels) == set(plain.labels)

    def test_main_applies_the_config_in_order(self, tmp_path, target):
        first = tmp_path / "running_results_first"
        _cell(first, [_row("PISAM", 0.1)])
        second = tmp_path / "running_results_second"
        _cell(second, [_row("PISAM", 0.2)])
        config = tmp_path / "sources.yaml"
        config.write_text(json.dumps({
            "target": str(target),
            "sources": [{"root": str(first), "labels": ["PISAM"]}, {"root": str(second), "labels": ["PISAM"]}],
        }))

        assert main(["--config", str(config)]) == 0

        assert _labels(target / "blocksworld" / EXPERIMENT / "testing" / CELL)["PISAM"] == 0.2
