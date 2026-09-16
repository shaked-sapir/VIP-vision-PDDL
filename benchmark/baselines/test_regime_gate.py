"""The regime gate keeps an arm out of the cells it was not built for.

    python -m pytest benchmark/baselines/test_regime_gate.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.baselines import resolve_baselines
from benchmark.baselines.base_runner import BaselineRunner
from benchmark.baselines.regime import DegradationRegime, gate_baselines


class _Stub(BaselineRunner):
    def __init__(self, label: str, bound=None) -> None:
        self._label = label
        self.bound = bound

    @property
    def name(self) -> str:
        return self._label

    @property
    def input_kind(self) -> str:
        return "symbolic"

    @property
    def paper(self) -> str:
        return "test"

    @property
    def uses_milp(self) -> bool:
        return False

    def learn(self, domain_path, prepared_trajectories, work_dir, timeout_seconds=60):
        return None, {}


class _NoiseOnly(_Stub):
    def supports(self, regime):
        if regime.mask_free:
            return True, ""
        return False, "needs masking_p == 0"

    def for_regime(self, regime):
        return _NoiseOnly(self.name, bound=regime)


class TestRegime:
    def test_simulated_zero_axes_are_recognised(self):
        regime = DegradationRegime.simulated(0.0, 0.2)
        assert regime.mask_free and not regime.noise_free
        assert regime.describe() == "simulated(mask=0, noise=0.2)"

    def test_image_is_neither_mask_free_nor_noise_free(self):
        regime = DegradationRegime.image()
        assert not regime.mask_free and not regime.noise_free
        assert regime.describe() == "image"

    def test_a_simulated_regime_needs_both_rates(self):
        with pytest.raises(ValueError):
            DegradationRegime("simulated", 0.1, None)

    def test_from_run_params_reads_the_recorded_cell(self):
        params = {"data_source_type": "SimulatedDataSource", "p_mask": 0.1, "p_noise": 0.0}
        regime = DegradationRegime.from_run_params(params, Path("/corpus"))
        assert regime == DegradationRegime.simulated(0.1, 0.0, Path("/corpus"))
        assert DegradationRegime.from_run_params({"data_source_type": "ImageDataSource"}).source == "image"


class TestGate:
    def test_keeps_supported_and_names_the_dropped(self):
        runners = [_Stub("ALWAYS"), _NoiseOnly("NOISE_ONLY")]
        kept, dropped = gate_baselines(runners, DegradationRegime.simulated(0.1, 0.0))
        assert [r.name for r in kept] == ["ALWAYS"]
        assert dropped == {"NOISE_ONLY": "needs masking_p == 0"}

    def test_binds_the_kept_runner_to_the_cell(self):
        regime = DegradationRegime.simulated(0.0, 0.3, Path("/corpus"))
        kept, dropped = gate_baselines([_NoiseOnly("NOISE_ONLY")], regime)
        assert dropped == {}
        assert kept[0].bound == regime

    def test_a_runner_without_an_opinion_passes_through_unchanged(self):
        always = _Stub("ALWAYS")
        kept, _ = gate_baselines([always], DegradationRegime.image())
        assert kept[0] is always

    def test_strict_off_keeps_everything_but_still_binds(self):
        regime = DegradationRegime.simulated(0.1, 0.1)
        kept, dropped = gate_baselines([_NoiseOnly("NOISE_ONLY")], regime, strict=False)
        assert dropped == {}
        assert kept[0].bound == regime

    def test_the_registered_arms_run_everywhere_by_default(self):
        runners = resolve_baselines(["rosame_24"])
        for regime in (DegradationRegime.image(), DegradationRegime.simulated(0.1, 0.1)):
            kept, dropped = gate_baselines(runners, regime)
            assert len(kept) == 1 and dropped == {}
