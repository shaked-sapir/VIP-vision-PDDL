"""The batch runner gates baselines per cell and records what it dropped.

    python -m pytest benchmark/test_benchmark_runner_gate.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.baselines.base_runner import BaselineRunner
from benchmark.baselines.regime import DegradationRegime
from benchmark.benchmark_runner import (
    _build_main_kwargs,
    _cell_regime,
    _gate_cell_baselines,
    _gate_is_strict,
)


class _MaskFreeOnly(BaselineRunner):
    @property
    def name(self) -> str:
        return "MASK_FREE_ONLY"

    @property
    def input_kind(self) -> str:
        return "symbolic"

    @property
    def paper(self) -> str:
        return "test"

    @property
    def uses_milp(self) -> bool:
        return False

    def supports(self, regime):
        return (True, "") if regime.mask_free else (False, "needs masking_p == 0")

    def learn(self, domain_path, prepared_trajectories, work_dir, timeout_seconds=60):
        return None, {}


class TestGateMode:
    def test_defaults_to_strict(self):
        assert _gate_is_strict({}) is True

    def test_off_is_accepted(self):
        assert _gate_is_strict({"baseline_regime_gate": "off"}) is False

    def test_a_typo_fails_loudly(self):
        with pytest.raises(ValueError):
            _gate_is_strict({"baseline_regime_gate": "lenient"})

    def test_the_key_is_not_forwarded_to_main(self):
        kwargs = _build_main_kwargs({"algorithms": ["cdps"], "baseline_regime_gate": "off"})
        assert "baseline_regime_gate" not in kwargs


class TestCellRegime:
    def test_image_cells_carry_no_rates(self):
        cell = {"masking_p": None, "noising_p": None}
        regime = _cell_regime("image", cell, Path("/corpus"))
        assert regime == DegradationRegime.image(Path("/corpus"))

    def test_simulated_cells_carry_their_grid_point(self):
        cell = {"masking_p": 0.1, "noising_p": 0.3}
        regime = _cell_regime("simulated", cell, Path("/corpus"))
        assert regime == DegradationRegime.simulated(0.1, 0.3, Path("/corpus"))


class TestPerCellKwargs:
    def test_drops_and_records_in_a_gated_cell(self):
        base = {"baselines": [_MaskFreeOnly()], "n_folds": 5}
        cell = _gate_cell_baselines(base, DegradationRegime.simulated(0.1, 0.0), strict=True)
        assert cell["baselines"] == []
        assert cell["skipped_baselines"] == {"MASK_FREE_ONLY": "needs masking_p == 0"}
        assert cell["n_folds"] == 5
        assert base["baselines"], "the run-wide kwargs must not be mutated"

    def test_keeps_in_the_arms_own_cell(self):
        base = {"baselines": [_MaskFreeOnly()]}
        cell = _gate_cell_baselines(base, DegradationRegime.simulated(0.0, 0.3), strict=True)
        assert [r.name for r in cell["baselines"]] == ["MASK_FREE_ONLY"]
        assert cell["skipped_baselines"] == {}

    def test_off_keeps_everything(self):
        base = {"baselines": [_MaskFreeOnly()]}
        cell = _gate_cell_baselines(base, DegradationRegime.image(), strict=False)
        assert len(cell["baselines"]) == 1
        assert cell["skipped_baselines"] == {}
