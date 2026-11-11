"""The realised-corruption fields of fold_info.json: written, merged, read back.

    python -m pytest benchmark/experiment_running_helpers/test_fold_corruption.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.algorithm_adapters.lamanna.realised_rates import RealisedCorruption, RealisedRates
ROOT = Path(__file__).resolve().parents[2]
BLOCKS_DOMAIN = ROOT / "src" / "domains" / "blocks" / "blocks.pddl"

from benchmark.experiment_running_helpers.fold_corruption import (  # noqa: E402
    CORRUPTION_KEY,
    FOLD_INFO_FILENAME,
    TOP_LEVEL_FIELDS,
    corruption_fields,
    read_fold_corruption,
    record_fold_corruption,
    write_fold_corruption,
)

# Per-frame: frame 0 clean (10 atoms), two later frames with 1 hidden and 3 flipped in all.
FRAME0 = RealisedRates(n_states=1, n_atoms=10, n_masked=0, n_flipped=0)
LATER = RealisedRates(n_states=2, n_atoms=20, n_masked=1, n_flipped=3)
TOTAL = RealisedRates(n_states=3, n_atoms=30, n_masked=1, n_flipped=3)
CORRUPTION = RealisedCorruption(TOTAL, FRAME0, LATER, n_flipped_with_next=2, n_flip_persisted=1)


class TestFields:
    def test_the_named_fields_and_their_definitions(self):
        fields = corruption_fields(CORRUPTION, Path("/corpus/gt_trajectories"))
        assert tuple(k for k in fields if k != CORRUPTION_KEY) == TOP_LEVEL_FIELDS
        assert fields["realised_mask_rate"] == pytest.approx(1 / 30)
        assert fields["realised_flip_rate"] == pytest.approx(3 / 29)  # over visible atoms
        assert fields["flip_persistence"] == 0.5
        assert fields["n_atoms"] == 30

    def test_the_block_keeps_frame0_apart_from_the_later_frames(self):
        block = corruption_fields(CORRUPTION, Path("/corpus/gt_trajectories"))[CORRUPTION_KEY]
        assert block["frame0"]["flip_rate"] == 0.0
        assert block["later_frames"]["flip_rate"] == pytest.approx(3 / 19)
        assert block["later_frames"]["mask_rate"] == pytest.approx(1 / 20)
        assert (block["n_flipped_with_next"], block["n_flip_persisted"]) == (2, 1)
        assert block["gt_root"] == "/corpus/gt_trajectories"

    def test_nothing_flipped_records_a_null_persistence(self):
        clean = RealisedCorruption(FRAME0, FRAME0, RealisedRates(0, 0, 0, 0), 0, 0)
        fields = corruption_fields(clean)
        assert fields["flip_persistence"] is None
        assert fields[CORRUPTION_KEY]["gt_root"] is None


class TestWriter:
    def test_merges_into_an_existing_fold_info(self, tmp_path: Path):
        info = tmp_path / FOLD_INFO_FILENAME
        info.write_text(json.dumps({"fold": 0, "trajectories": [{"problem": "problem3"}]}))
        write_fold_corruption(tmp_path, CORRUPTION, Path("/gt"))
        saved = json.loads(info.read_text())
        assert saved["fold"] == 0 and saved["trajectories"] == [{"problem": "problem3"}]
        assert saved["n_atoms"] == 30
        assert saved[CORRUPTION_KEY]["gt_root"] == "/gt"

    def test_creates_the_file_when_the_fold_has_none(self, tmp_path: Path):
        path = write_fold_corruption(tmp_path, CORRUPTION)
        assert path == tmp_path / FOLD_INFO_FILENAME
        assert read_fold_corruption(tmp_path)["n_flipped"] == 3

    def test_a_rewrite_replaces_the_earlier_measurement(self, tmp_path: Path):
        write_fold_corruption(tmp_path, CORRUPTION)
        write_fold_corruption(tmp_path, RealisedCorruption(FRAME0, FRAME0, RealisedRates(0, 0, 0, 0), 0, 0))
        saved = json.loads((tmp_path / FOLD_INFO_FILENAME).read_text())
        assert saved["realised_flip_rate"] == 0.0 and saved["n_atoms"] == 10

    def test_an_unmeasured_fold_reads_as_none(self, tmp_path: Path):
        assert read_fold_corruption(tmp_path) is None
        (tmp_path / FOLD_INFO_FILENAME).write_text(json.dumps({"fold": 0}))
        assert read_fold_corruption(tmp_path) is None


class TestRecorder:
    def test_a_fold_whose_gt_is_missing_is_left_unrecorded(self, tmp_path: Path, capsys):
        prepared = [(tmp_path / "x.trajectory", None, tmp_path / "problem9.pddl", {0})]
        result = record_fold_corruption(tmp_path, prepared, BLOCKS_DOMAIN, {}, tag="fold 9")
        assert result is None
        assert not (tmp_path / FOLD_INFO_FILENAME).exists()
        assert "not recorded" in capsys.readouterr().out
