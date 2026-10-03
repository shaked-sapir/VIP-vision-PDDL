"""Tests for the unanchored-arm backfill on a degraded initial state.

    python -m pytest benchmark/test_backfill_nogt_s0.py

Learning and planning are not exercised here: every test stops at staging.
"""

import json
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Tuple

import pytest

import benchmark.backfill_nogt_s0 as driver

from benchmark.algorithms import PISAM_MILP_LOOP, PISAM_MILP_SINGLE_ROUND
from benchmark.backfill_cdps import ExperimentSettings
from benchmark.backfill_nogt_s0 import (
    _run_sequential,
    OBSERVATIONS_DIRNAME,
    PROVENANCE_FILENAME,
    StagingOptions,
    backfill_cell,
    degradation_spec,
    noisy_initial_label,
    output_cell,
    read_unanchored_config,
    resolve_arm,
    row_recorded_error,
    stage_cell_observations,
    staged_trajectories,
)
from benchmark.simulated_version.test_leading_state_corruption import (
    BLOCKS_DOMAIN,
    PROBLEM,
    spec,
    write_blocks_fixture,
)
from src.milp.converter import GtAnchoring
from src.observation_degradation.masking import MaskingType
from src.observation_degradation.noising import NoisingType

REPO_ROOT = Path(__file__).resolve().parents[1]
LARGE_CONFIG = REPO_ROOT / "benchmark" / "run_config_large.yaml"
SMALL_CONFIG = REPO_ROOT / "benchmark" / "run_config.yaml"
CELL_NAME = "fold0_numtrajs1_gtrate0"


def _make_experiment(tmp_path: Path) -> Tuple[Path, ExperimentSettings]:
    """A one-cell simulated experiment over the blocks fixture."""
    paths = write_blocks_fixture(tmp_path, 0.2, 0.2)
    exp_dir = tmp_path / "running_results" / "blocksworld" / "exp__mask=0.2__noise=0.2"
    cell = exp_dir / "testing" / CELL_NAME
    cell.mkdir(parents=True)
    shutil.move(str(paths["frozen_dir"]), str(cell / "original_observations"))
    shutil.copy2(BLOCKS_DOMAIN, cell / "domain_reference.pddl")
    stem = f"original_observation_{PROBLEM}"
    (cell / "fold_info.json").write_text(json.dumps({
        "fold": 0,
        "num_trajectories": 1,
        "gt_rate_percentage": 0,
        "trajectories": [{
            "problem": PROBLEM,
            "trajectory_file": f"{stem}.trajectory",
            "masking_file": f"{stem}.masking_info",
            "problem_file": f"{PROBLEM}.pddl",
        }],
        "test_problems": [PROBLEM],
    }))
    settings = ExperimentSettings(
        data_dir=paths["data_dir"],
        problem_dir=paths["data_dir"],
        bench_name="blocksworld",
        cdps_search_params={},
        learn_timeout=5,
        planning_timeout=5,
        frame_axiom_mode="after_gt_only",
    )
    return cell, settings


def _snapshot(directory: Path) -> Dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*")) if path.is_file()
    }


def _options(tmp_path: Path, hide_masked: bool = True) -> StagingOptions:
    return StagingOptions(out_root=tmp_path / "out", hide_masked=hide_masked, prepare_only=True)


class TestLabels:
    def test_the_tag_follows_the_anchoring_part(self):
        assert noisy_initial_label("PISAM_MILP_LOOP__gt=none__m=4") == (
            "PISAM_MILP_LOOP__gt=none__s0=noisy__m=4"
        )

    def test_an_anchored_label_is_refused(self):
        with pytest.raises(ValueError, match="not an unanchored arm"):
            noisy_initial_label("PISAM_MILP_LOOP__m=4")

    def test_large_sweep_loop_arm(self):
        arm = resolve_arm(PISAM_MILP_LOOP, LARGE_CONFIG)
        assert arm.row_name == "PISAM_MILP_LOOP__gt=none__s0=noisy__m=4"
        assert arm.work_subdir == "pisam_milp_loop__gt=none__s0=noisy__m=4"

    def test_small_sweep_arms(self):
        loop = resolve_arm(PISAM_MILP_LOOP, SMALL_CONFIG)
        single = resolve_arm(PISAM_MILP_SINGLE_ROUND, SMALL_CONFIG)
        assert loop.row_name == "PISAM_MILP_LOOP__gt=none__s0=noisy"
        assert single.row_name == "PISAM_MILP_SR__gt=none__s0=noisy"
        assert loop.work_subdir != single.work_subdir


class TestSolveTimeLimit:
    def test_the_limit_reaches_the_config_and_the_label(self):
        arm = resolve_arm(PISAM_MILP_LOOP, SMALL_CONFIG, solve_time_limit=60)
        assert arm.milp_config.time_limit_seconds == 60
        assert arm.milp_config.resolve_time_limit(300) == 60
        assert arm.row_name == "PISAM_MILP_LOOP__gt=none__s0=noisy__solve=60"
        assert arm.work_subdir == "pisam_milp_loop__gt=none__s0=noisy__solve=60"

    def test_without_a_limit_a_solve_inherits_the_budget(self):
        arm = resolve_arm(PISAM_MILP_LOOP, SMALL_CONFIG)
        assert arm.milp_config.time_limit_seconds is None
        assert arm.milp_config.resolve_time_limit(300) == 300
        assert "solve=" not in arm.row_name

    def test_the_loop_never_lets_a_solve_exceed_the_limit_or_the_budget(self):
        from src.plan_denoising.milp_denoiser.loop import _remaining_budget
        assert _remaining_budget(300, 60, elapsed=10.0) == 60
        assert _remaining_budget(300, 60, elapsed=270.0) == 30


class TestConfig:
    def test_anchoring_is_forced_to_none(self):
        assert read_unanchored_config(LARGE_CONFIG).gt_anchoring is GtAnchoring.NONE
        assert read_unanchored_config(SMALL_CONFIG).gt_anchoring is GtAnchoring.NONE
        assert read_unanchored_config(None).gt_anchoring is GtAnchoring.NONE

    def test_the_ablations_block_is_ignored(self):
        """run_config_large.yaml ablates gt_anchoring; the block still parses."""
        assert "ablations" in LARGE_CONFIG.read_text()
        assert read_unanchored_config(LARGE_CONFIG).subset_size.value == 4

    def test_a_bare_block_is_accepted(self, tmp_path):
        block = tmp_path / "block.yaml"
        block.write_text("subset_size: 3\ngt_anchoring: init_only\n")
        config = read_unanchored_config(block)
        assert config.subset_size.value == 3
        assert config.gt_anchoring is GtAnchoring.NONE


class TestDegradationSpec:
    def test_reads_the_recorded_rates(self):
        run_params = {"p_mask": 0.1, "p_noise": 0.3,
                      "masking_strategy": "percentage", "noising_strategy": "percentage"}
        got = degradation_spec(run_params, seed=7)
        assert (got.masking_p, got.noising_p, got.seed) == (0.1, 0.3, 7)
        assert got.masking_strategy is MaskingType.PERCENTAGE
        assert got.noising_strategy is NoisingType.PERCENTAGE

    def test_a_non_simulated_experiment_is_refused(self):
        with pytest.raises(ValueError, match="not a simulated experiment"):
            degradation_spec({"learning_timeout_seconds": 300}, seed=42)


class TestOutputLayout:
    def test_the_out_tree_mirrors_the_source_tree(self, tmp_path):
        cell = tmp_path / "running_results" / "hanoi" / "exp__mask=0.1__noise=0.2" / "testing" / CELL_NAME
        assert output_cell(tmp_path / "out", cell) == (
            tmp_path / "out" / "hanoi" / "exp__mask=0.1__noise=0.2" / "testing" / CELL_NAME
        )


class TestStaging:
    def test_tuples_carry_no_ground_truth_index(self, tmp_path):
        cell, settings = _make_experiment(tmp_path)
        out_cell = output_cell(tmp_path / "out", cell)
        fold_info = json.loads((cell / "fold_info.json").read_text())
        prepared = stage_cell_observations(
            cell, out_cell, settings, spec(0.2, 0.2), 0, fold_info, hide_masked=True,
        )
        assert len(prepared) == 1
        trajectory, masking, problem_pddl, gt_indices = prepared[0]
        assert gt_indices == set()
        assert trajectory.parent == out_cell / OBSERVATIONS_DIRNAME
        assert trajectory.exists() and masking.exists()
        assert problem_pddl.is_absolute()

    def test_a_packed_cell_stages_the_same_files(self, tmp_path):
        cell, settings = _make_experiment(tmp_path)
        fold_info = json.loads((cell / "fold_info.json").read_text())
        loose = stage_cell_observations(
            cell, tmp_path / "loose", settings, spec(0.2, 0.2), 0, fold_info, hide_masked=False,
        )
        with tarfile.open(cell / "original_observations.tar.gz", "w:gz") as archive:
            archive.add(cell / "original_observations", arcname="original_observations")
        shutil.rmtree(cell / "original_observations")
        packed = stage_cell_observations(
            cell, tmp_path / "packed", settings, spec(0.2, 0.2), 0, fold_info, hide_masked=False,
        )
        assert loose[0][0].read_text() == packed[0][0].read_text()

    def test_a_cell_without_observations_raises(self, tmp_path):
        cell, settings = _make_experiment(tmp_path)
        shutil.rmtree(cell / "original_observations")
        fold_info = json.loads((cell / "fold_info.json").read_text())
        with pytest.raises(FileNotFoundError):
            stage_cell_observations(
                cell, tmp_path / "out", settings, spec(0.2, 0.2), 0, fold_info, hide_masked=True,
            )


class TestBackfillCellPrepareOnly:
    def test_the_source_cell_is_not_modified(self, tmp_path):
        cell, settings = _make_experiment(tmp_path)
        before = _snapshot(cell)
        status = backfill_cell(
            cell, settings, spec(0.2, 0.2), resolve_arm(PISAM_MILP_LOOP, LARGE_CONFIG),
            _options(tmp_path), force=False, dry_run=False,
        )
        assert status == "staged"
        assert _snapshot(cell) == before

    def test_the_out_cell_is_self_describing(self, tmp_path):
        cell, settings = _make_experiment(tmp_path)
        options = _options(tmp_path)
        backfill_cell(
            cell, settings, spec(0.2, 0.2), resolve_arm(PISAM_MILP_LOOP, LARGE_CONFIG),
            options, force=False, dry_run=False,
        )
        out_cell = output_cell(options.out_root, cell)
        assert (out_cell / "fold_info.json").read_text() == (cell / "fold_info.json").read_text()
        assert (out_cell / "domain_reference.pddl").exists()
        provenance = json.loads((out_cell / PROVENANCE_FILENAME).read_text())
        assert provenance["source_cell"] == str(cell.resolve())
        assert provenance["redrawn_state_indices"] == [0, 1]
        assert provenance["masked_values_hidden"] is True
        assert (provenance["masking_p"], provenance["noising_p"]) == (0.2, 0.2)
        assert not (out_cell / "fold_result.json").exists()

    def test_a_dry_run_writes_nothing(self, tmp_path):
        cell, settings = _make_experiment(tmp_path)
        options = _options(tmp_path)
        status = backfill_cell(
            cell, settings, spec(0.2, 0.2), resolve_arm(PISAM_MILP_LOOP, LARGE_CONFIG),
            options, force=False, dry_run=True,
        )
        assert status == "dry"
        assert not options.out_root.exists()


class TestStagingReuse:
    def test_matching_staging_is_reused(self, tmp_path):
        cell, settings = _make_experiment(tmp_path)
        options = _options(tmp_path)
        backfill_cell(
            cell, settings, spec(0.2, 0.2), resolve_arm(PISAM_MILP_LOOP, LARGE_CONFIG),
            options, force=False, dry_run=False,
        )
        out_cell = output_cell(options.out_root, cell)
        fold_info = json.loads((cell / "fold_info.json").read_text())
        reused = staged_trajectories(out_cell, settings, spec(0.2, 0.2), fold_info, True)
        assert reused is not None and len(reused) == 1
        assert reused[0][3] == set()

    @pytest.mark.parametrize("degradation, hide_masked", [
        (spec(0.2, 0.2), False),
        (spec(0.2, 0.3), True),
        (spec(0.2, 0.2, seed=43), True),
    ])
    def test_different_parameters_force_a_restage(self, tmp_path, degradation, hide_masked):
        cell, settings = _make_experiment(tmp_path)
        options = _options(tmp_path)
        backfill_cell(
            cell, settings, spec(0.2, 0.2), resolve_arm(PISAM_MILP_LOOP, LARGE_CONFIG),
            options, force=False, dry_run=False,
        )
        out_cell = output_cell(options.out_root, cell)
        fold_info = json.loads((cell / "fold_info.json").read_text())
        assert staged_trajectories(out_cell, settings, degradation, fold_info, hide_masked) is None

    def test_a_missing_file_forces_a_restage(self, tmp_path):
        cell, settings = _make_experiment(tmp_path)
        options = _options(tmp_path)
        backfill_cell(
            cell, settings, spec(0.2, 0.2), resolve_arm(PISAM_MILP_LOOP, LARGE_CONFIG),
            options, force=False, dry_run=False,
        )
        out_cell = output_cell(options.out_root, cell)
        fold_info = json.loads((cell / "fold_info.json").read_text())
        next((out_cell / OBSERVATIONS_DIRNAME).glob("*.masking_info")).unlink()
        assert staged_trajectories(out_cell, settings, spec(0.2, 0.2), fold_info, True) is None


class TestSequentialRun:
    def test_a_failing_cell_is_counted_and_does_not_stop_the_rest(self, tmp_path):
        cell, settings = _make_experiment(tmp_path)
        broken = cell.parent / "fold1_numtrajs1_gtrate0"
        shutil.copytree(cell, broken)
        shutil.rmtree(broken / "original_observations")
        options = _options(tmp_path)
        arm = resolve_arm(PISAM_MILP_LOOP, LARGE_CONFIG)
        tasks = [(broken, settings, spec(0.2, 0.2)), (cell, settings, spec(0.2, 0.2))]

        assert _run_sequential(tasks, arm, options, force=False) == 1
        assert (output_cell(options.out_root, cell) / PROVENANCE_FILENAME).exists()


class TestRetryErrors:
    @staticmethod
    def _write_row(out_cell: Path, row_name: str, error) -> Path:
        out_cell.mkdir(parents=True, exist_ok=True)
        path = out_cell / "fold_result.json"
        specific = {"error": error} if error is not None else {}
        path.write_text(json.dumps([{"algorithm": row_name, "algorithm_specific": specific}]))
        return path

    def test_an_error_row_is_recognised(self, tmp_path):
        path = self._write_row(tmp_path, "ARM", "Received illegal state component")
        assert row_recorded_error(path, "ARM") is True
        assert row_recorded_error(path, "OTHER") is False

    def test_a_clean_or_missing_row_is_not_an_error(self, tmp_path):
        assert row_recorded_error(self._write_row(tmp_path, "ARM", None), "ARM") is False
        assert row_recorded_error(tmp_path / "absent.json", "ARM") is False

    def _run(self, tmp_path, monkeypatch, error):
        cell, settings = _make_experiment(tmp_path)
        arm = resolve_arm(PISAM_MILP_LOOP, LARGE_CONFIG)
        options = StagingOptions(
            out_root=tmp_path / "out", hide_masked=True, prepare_only=False, retry_errors=True,
        )
        out_cell = output_cell(options.out_root, cell)
        result_path = self._write_row(out_cell, arm.row_name, error)
        calls = []

        def fake_phase(**kwargs):
            calls.append(kwargs["trajectories"])
            return {"algorithm": arm.row_name, "algorithm_specific": {}, "solving_ratio": 1.0}

        monkeypatch.setattr(driver, "run_cdps_phase", fake_phase)
        status = backfill_cell(
            cell, settings, spec(0.2, 0.2), arm, options, force=False, dry_run=False,
        )
        return status, calls, json.loads(result_path.read_text())

    def test_an_errored_cell_is_restaged_and_rerun(self, tmp_path, monkeypatch):
        status, calls, rows = self._run(tmp_path, monkeypatch, "boom")
        assert status == "done"
        assert len(calls) == 1 and calls[0][0][0].exists()
        assert rows[0]["algorithm_specific"] == {} and rows[0]["solving_ratio"] == 1.0

    def test_a_clean_cell_is_still_skipped(self, tmp_path, monkeypatch):
        status, calls, _rows = self._run(tmp_path, monkeypatch, None)
        assert status == "skip"
        assert calls == []
