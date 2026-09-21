"""Tests for the per-arm convergence readers.

    python -m pytest benchmark/evaluation/cfm/test_convergence.py
"""

import json
from pathlib import Path

import pytest

from benchmark.evaluation.cfm.convergence import (
    cell_convergence,
    milp_rounds,
    pisam_series,
    pre_mip_epoch,
    rosame_series,
)


def _instance(tmp_path: Path, name: str = "fold0_numtrajs100_gtrate0") -> Path:
    inst = tmp_path / "testing" / name
    inst.mkdir(parents=True)
    return inst


class TestPisamSeries:
    def test_reads_the_live_stream(self, tmp_path):
        inst = _instance(tmp_path)
        arm = inst / "pisam_milp_loop__m=4"
        arm.mkdir()
        rows = [{"round": 1, "v_raw": 90.0, "improved": True},
                {"round": 2, "v_raw": 80.0, "improved": True}]
        (arm / "milp_loop_rounds.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")
        s = pisam_series(inst)["pisam_milp_loop__m=4"]
        assert s["x"] == [1, 2] and s["v"] == [90.0, 80.0]

    def test_tolerates_a_torn_final_line(self, tmp_path):
        """The stream is appended live, so a reader can catch a half-written row."""
        inst = _instance(tmp_path)
        arm = inst / "pisam_milp_loop__m=4"
        arm.mkdir()
        (arm / "milp_loop_rounds.jsonl").write_text(
            json.dumps({"round": 1, "v_raw": 5.0, "improved": True}) + '\n{"round":2,"v_ra')
        assert pisam_series(inst)["pisam_milp_loop__m=4"]["x"] == [1]

    def test_falls_back_to_the_end_of_run_file(self, tmp_path):
        inst = _instance(tmp_path)
        arm = inst / "pisam_milp_loop__gt=none__m=4"
        arm.mkdir()
        (arm / "milp_loop_rounds.json").write_text(json.dumps(
            {"rounds": [{"round": 1, "v_raw": 7.0, "improved": False}]}))
        assert pisam_series(inst)["pisam_milp_loop__gt=none__m=4"]["v"] == [7.0]

    def test_rounds_without_a_score_are_dropped(self, tmp_path):
        inst = _instance(tmp_path)
        arm = inst / "pisam_milp_loop__m=4"
        arm.mkdir()
        (arm / "milp_loop_rounds.jsonl").write_text(
            json.dumps({"round": 1, "v_raw": None}) + "\n"
            + json.dumps({"round": 2, "v_raw": 3.0, "improved": True}) + "\n")
        assert pisam_series(inst)["pisam_milp_loop__m=4"]["x"] == [2]


class TestRosameSeries:
    def _write(self, inst: Path, arm: str, records):
        d = inst / "anytime_snapshots" / arm
        d.mkdir(parents=True)
        (d / "snapshots.json").write_text(json.dumps(records))

    def test_reads_loss_per_epoch(self, tmp_path):
        inst = _instance(tmp_path)
        self._write(inst, "ROSAME_24", [
            {"epoch": 0, "loss": 9.0}, {"epoch": 1, "loss": 7.0}])
        s = rosame_series(inst)["ROSAME_24"]
        assert s["x"] == [0, 1] and s["loss"] == [9.0, 7.0]

    def test_agreement_is_none_for_a_dl_only_arm(self, tmp_path):
        inst = _instance(tmp_path)
        self._write(inst, "ROSAME_24", [{"epoch": 0, "loss": 1.0}])
        assert rosame_series(inst)["ROSAME_24"]["agreement"] == [None]

    def _rounds(self, inst: Path, arm: str, rounds):
        (inst / "fold_result.json").write_text(json.dumps(
            [{"algorithm": arm, "algorithm_specific": {"milp_rounds": rounds}}]))

    def test_agreement_comes_from_the_rounds_not_the_snapshot(self, tmp_path):
        """A snapshot's own ``agreement`` lags a round; the round record does not."""
        inst = _instance(tmp_path)
        self._write(inst, "ROSAME_MILP_24", [
            {"epoch": 49, "loss": 2.0, "agreement": None},
            {"epoch": 50, "loss": 1.9, "agreement": 0.7391}])
        self._rounds(inst, "ROSAME_MILP_24", [
            {"epoch": 49, "agreement": 0.7391}, {"epoch": 50, "agreement": 1.0}])
        assert rosame_series(inst)["ROSAME_MILP_24"]["agreement"] == [0.7391, 1.0]

    def test_a_round_past_the_last_snapshot_extends_the_axis(self, tmp_path):
        """The loop returns on the stop check, so the final round is never captured."""
        inst = _instance(tmp_path)
        self._write(inst, "ROSAME_MILP_24", [{"epoch": 49, "loss": 2.0}])
        self._rounds(inst, "ROSAME_MILP_24", [
            {"epoch": 49, "agreement": 0.7}, {"epoch": 50, "agreement": 1.0}])
        s = rosame_series(inst)["ROSAME_MILP_24"]
        assert s["x"] == [49, 50]
        assert s["agreement"] == [0.7, 1.0]
        assert s["loss"] == [2.0, None]

    def test_agreement_is_none_without_a_fold_result(self, tmp_path):
        inst = _instance(tmp_path)
        self._write(inst, "ROSAME_MILP_24", [{"epoch": 50, "loss": 2.0}])
        assert rosame_series(inst)["ROSAME_MILP_24"]["agreement"] == [None]


class TestRosameTrainingSeries:
    """With snapshots off, the loss comes from ``rosame_training/<arm>.json``."""

    def _series(self, inst: Path, arm: str, payload):
        d = inst / "rosame_training"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{arm}.json").write_text(json.dumps(payload))

    def test_reads_the_loss_without_any_snapshot(self, tmp_path):
        inst = _instance(tmp_path)
        self._series(inst, "ROSAME_24", {
            "losses": [9.0, 7.0, 6.5], "stop_reason": "converged", "best_epoch": 2})
        s = rosame_series(inst)["ROSAME_24"]
        assert s["x"] == [0, 1, 2] and s["loss"] == [9.0, 7.0, 6.5]
        assert (s["stop_reason"], s["stop_epoch"], s["best_epoch"]) == ("converged", 2, 2)

    def test_the_series_file_wins_over_a_sparser_snapshot_index(self, tmp_path):
        inst = _instance(tmp_path)
        snapshots = inst / "anytime_snapshots" / "ROSAME_24"
        snapshots.mkdir(parents=True)
        (snapshots / "snapshots.json").write_text(json.dumps([{"epoch": 0, "loss": 9.0}]))
        self._series(inst, "ROSAME_24", {"losses": [9.0, 7.0]})
        assert rosame_series(inst)["ROSAME_24"]["x"] == [0, 1]

    def test_agreement_and_the_first_solve_join_on_epoch(self, tmp_path):
        inst = _instance(tmp_path)
        self._series(inst, "ROSAME_MILP_24", {
            "losses": [3.0, 2.0, 2.6, 2.5], "first_solve_epoch": 1,
            "stop_reason": "epochs_exhausted", "best_epoch": 3})
        (inst / "fold_result.json").write_text(json.dumps([{
            "algorithm": "ROSAME_MILP_24",
            "algorithm_specific": {"milp_rounds": [
                {"epoch": 1, "agreement": 0.6}, {"epoch": 2, "agreement": 0.9}]}}]))
        s = rosame_series(inst)["ROSAME_MILP_24"]
        assert s["agreement"] == [None, 0.6, 0.9, None]
        assert s["first_solve_epoch"] == 1

    def test_an_empty_series_file_is_ignored(self, tmp_path):
        inst = _instance(tmp_path)
        self._series(inst, "ROSAME_24", {"losses": []})
        assert rosame_series(inst) == {}


class TestMilpRounds:
    def test_maps_each_arm_to_its_epoch_agreement(self, tmp_path):
        inst = _instance(tmp_path)
        (inst / "fold_result.json").write_text(json.dumps([
            {"algorithm": "ROSAME_MILP_24", "algorithm_specific": {
                "milp_rounds": [{"epoch": 49, "agreement": 0.7391},
                                {"epoch": 50, "agreement": 1.0}]}},
            {"algorithm": "ROSAME_24", "algorithm_specific": {}},
        ]))
        assert milp_rounds(inst) == {"ROSAME_MILP_24": {49: 0.7391, 50: 1.0}}

    def test_missing_file_is_empty(self, tmp_path):
        assert milp_rounds(_instance(tmp_path)) == {}

    def test_absent_snapshots_yield_nothing(self, tmp_path):
        """v2 ran without snapshot_interval, so this is the common case there."""
        assert rosame_series(_instance(tmp_path)) == {}


class TestCellConvergence:
    def test_groups_folds_under_their_training_size(self, tmp_path):
        for fold in (0, 1):
            inst = _instance(tmp_path, f"fold{fold}_numtrajs100_gtrate0")
            arm = inst / "pisam_milp_loop__m=4"
            arm.mkdir()
            (arm / "milp_loop_rounds.jsonl").write_text(
                json.dumps({"round": 1, "v_raw": 1.0 + fold, "improved": True}) + "\n")
        c = cell_convergence(tmp_path / "testing")
        assert list(c["pisam"]) == [100]
        assert len(c["pisam"][100]["pisam_milp_loop__m=4"]) == 2

    def test_missing_testing_dir_is_empty_not_an_error(self, tmp_path):
        assert cell_convergence(tmp_path / "nope") == {
            "pisam": {}, "rosame": {}, "pre_mip": {}}


def test_pre_mip_epoch_comes_from_the_fold_result(tmp_path):
    inst = _instance(tmp_path)
    (inst / "fold_result.json").write_text(json.dumps([
        {"algorithm": "ROSAME_24", "algorithm_specific": {}},
        {"algorithm": "ROSAME_MILP_24", "algorithm_specific": {"pre_mip_epochs": 50}}]))
    assert pre_mip_epoch(inst) == 50
