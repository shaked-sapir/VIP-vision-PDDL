"""Backfill reads a cell's regime from run_params.json and gates by it.

    python -m pytest benchmark/test_backfill_gate.py
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.backfill_baseline import _experiment_regime
from benchmark.baselines.regime import DegradationRegime


def _experiment(tmp_path: Path, params: dict) -> Path:
    exp_dir = tmp_path / "exp"
    (exp_dir / "evaluation_results").mkdir(parents=True)
    (exp_dir / "evaluation_results" / "run_params.json").write_text(json.dumps(params))
    return exp_dir


def test_a_simulated_experiment_yields_its_grid_point(tmp_path):
    exp_dir = _experiment(tmp_path, {
        "data_source_type": "SimulatedDataSource", "p_mask": 0.0, "p_noise": 0.2,
    })
    assert _experiment_regime(exp_dir, Path("/corpus")) == DegradationRegime.simulated(
        0.0, 0.2, Path("/corpus")
    )


def test_an_image_experiment_yields_the_image_regime(tmp_path):
    exp_dir = _experiment(tmp_path, {"data_source_type": "ImageDataSource"})
    assert _experiment_regime(exp_dir, Path("/corpus")) == DegradationRegime.image(Path("/corpus"))


def test_a_missing_run_params_disables_gating(tmp_path, capsys):
    assert _experiment_regime(tmp_path / "nowhere", Path("/corpus")) is None
    assert "not regime-gated" in capsys.readouterr().out


def test_a_simulated_run_without_rates_disables_gating(tmp_path, capsys):
    exp_dir = _experiment(tmp_path, {"data_source_type": "SimulatedDataSource"})
    assert _experiment_regime(exp_dir, Path("/corpus")) is None
    assert "not regime-gated" in capsys.readouterr().out
