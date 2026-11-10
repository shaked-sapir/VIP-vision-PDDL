"""The image section's recorded corruption and the variant-label mode inheritance.

    python -m pytest benchmark/evaluation/cfm/test_build_dashboard_image.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.evaluation.cfm.build_dashboard import image_corruption_summary, inherit_modes
from benchmark.experiment_running_helpers.fold_corruption import CORRUPTION_KEY


def _block(n_atoms, n_masked, n_flipped, frame0, later, with_next, persisted):
    return {
        "n_states": 3, "n_atoms": n_atoms, "n_masked": n_masked, "n_flipped": n_flipped,
        "frame0": frame0, "later_frames": later,
        "n_flipped_with_next": with_next, "n_flip_persisted": persisted,
    }


def _fold(tmp_path: Path, name: str, block) -> Path:
    fold = tmp_path / name
    fold.mkdir()
    payload = {"fold": 0}
    if block is not None:
        payload[CORRUPTION_KEY] = block
    (fold / "fold_info.json").write_text(json.dumps(payload))
    return fold


class TestImageCorruptionSummary:
    def test_counts_are_pooled_across_folds_before_dividing(self, tmp_path: Path):
        a = _fold(tmp_path, "a", _block(30, 0, 3,
                                         {"n_atoms": 10, "n_masked": 0, "n_flipped": 0},
                                         {"n_atoms": 20, "n_masked": 0, "n_flipped": 3}, 2, 1))
        b = _fold(tmp_path, "b", _block(60, 6, 0,
                                         {"n_atoms": 20, "n_masked": 2, "n_flipped": 0},
                                         {"n_atoms": 40, "n_masked": 4, "n_flipped": 0}, 0, 0))
        out = image_corruption_summary([a, b])
        assert out["n_instances"] == 2
        assert out["mask_rate"] == pytest.approx(6 / 90)
        assert out["flip_rate"] == pytest.approx(3 / 84)          # over visible atoms
        assert out["frame0_flip_rate"] == 0.0
        assert out["later_flip_rate"] == pytest.approx(3 / 56)
        assert out["flip_persistence"] == 0.5
        assert out["n_atoms_per_state"] == 15.0

    def test_unmeasured_folds_are_skipped_and_none_measured_is_none(self, tmp_path: Path):
        bare = _fold(tmp_path, "bare", None)
        assert image_corruption_summary([bare]) is None
        measured = _fold(tmp_path, "m", _block(10, 0, 1,
                                                {"n_atoms": 5, "n_masked": 0, "n_flipped": 0},
                                                {"n_atoms": 5, "n_masked": 0, "n_flipped": 1}, 0, 0))
        out = image_corruption_summary([bare, measured])
        assert out["n_instances"] == 1
        assert out["flip_persistence"] is None  # nothing flipped had a next frame


class TestInheritModes:
    REGISTRY = {"ROSAME-I_26": ["image"], "ROSAME-I_MILP_26": ["image"], "ROSAME_24": ["simulation"]}

    def test_a_variant_takes_its_arms_modes(self):
        out = inherit_modes(self.REGISTRY, ["ROSAME-I_26__res=native__mode=converge"])
        assert out["ROSAME-I_26__res=native__mode=converge"] == ["image"]

    def test_the_longest_registered_prefix_wins(self):
        out = inherit_modes(self.REGISTRY, ["ROSAME-I_MILP_26__res=native"])
        assert out["ROSAME-I_MILP_26__res=native"] == ["image"]

    def test_registered_and_unknown_labels_are_left_alone(self):
        out = inherit_modes(self.REGISTRY, ["ROSAME_24", "PISAM__s0=noisy", "CDPS"])
        assert out["ROSAME_24"] == ["simulation"]
        assert "PISAM__s0=noisy" not in out and "CDPS" not in out
        assert out is not self.REGISTRY
