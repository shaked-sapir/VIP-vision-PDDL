"""The OffLAM arm: gated to noise-free cells, no knobs, recorded.

    python -m pytest benchmark/baselines/test_offlam_runner.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.baselines import resolve_baselines
from benchmark.baselines.offlam_runner import OffLAMRunner
from benchmark.baselines.regime import DegradationRegime, gate_baselines
from benchmark.baselines.test_nolam_runner import BLOCKS_DOMAIN, GT, PROBLEM


class TestGate:
    def test_only_noise_free_simulated_cells(self):
        runner = OffLAMRunner()
        assert runner.supports(DegradationRegime.simulated(0.3, 0.0)) == (True, "")
        ok, why = runner.supports(DegradationRegime.simulated(0.0, 0.1))
        assert not ok and "noising_p=0.1" in why
        ok, why = runner.supports(DegradationRegime.image())
        assert not ok and "image" in why

    def test_the_anchor_cell_admits_both_lamanna_arms(self):
        runners = resolve_baselines(["nolam", "offlam"])
        kept, dropped = gate_baselines(runners, DegradationRegime.simulated(0.0, 0.0))
        assert [r.name for r in kept] == ["NOLAM", "OffLAM"]
        assert dropped == {}

    def test_a_masked_noisy_cell_admits_neither(self):
        runners = resolve_baselines(["nolam", "offlam"])
        kept, dropped = gate_baselines(runners, DegradationRegime.simulated(0.1, 0.1))
        assert kept == []
        assert set(dropped) == {"NOLAM", "OffLAM"}


class TestIdentity:
    def test_registered_with_no_knobs(self):
        (runner,) = resolve_baselines(["offlam"], nolam_noise=0.2, rosame_seed=1)
        assert runner.run_params() == {}
        assert runner.row_name(BLOCKS_DOMAIN) == "OffLAM"
        assert runner.factors() == {"input_kind": "symbolic", "paper": "aij25", "uses_milp": False}


@pytest.fixture
def masked_fold(tmp_path):
    """A corpus with one GT trajectory frozen with one atom hidden in state 1."""
    data_dir = tmp_path / "corpus"
    problem_dir = data_dir / "training" / "trajectories" / "problem1"
    problem_dir.mkdir(parents=True)
    (problem_dir / "problem1.pddl").write_text(PROBLEM)
    gt_dir = data_dir / "gt_trajectories" / "problem1"
    gt_dir.mkdir(parents=True)
    (gt_dir / "problem1.trajectory").write_text(GT)
    cell = tmp_path / "cell"
    obs_dir = cell / "original_observations"
    obs_dir.mkdir(parents=True)
    # (holding a) hidden in state 1: absent from the text, recorded with its GT sign.
    degraded = GT.replace("(:state (clear b) (ontable b) (holding a))",
                          "(:state (clear b) (ontable b))", 1)
    (obs_dir / "original_observation_problem1.trajectory").write_text(degraded)
    (obs_dir / "original_observation_problem1.masking_info").write_text("\n(holding a - block)\n\n\n\n")
    prepared = [(
        obs_dir / "original_observation_problem1.trajectory",
        obs_dir / "original_observation_problem1.masking_info",
        problem_dir / "problem1.pddl",
        {0},
    )]
    return data_dir, cell, prepared


class TestLearn:
    def test_learns_a_model_from_a_masked_fold_and_records_the_row(self, masked_fold):
        data_dir, cell, prepared = masked_fold
        runner = OffLAMRunner().for_regime(DegradationRegime.simulated(0.05, 0.0, data_dir))

        model, report = runner.learn(BLOCKS_DOMAIN, prepared, cell, timeout_seconds=120)

        assert model is not None and "(:action pick_up" in model
        assert "(:requirements :typing)" in model
        assert report["terminated_by"] == "completed"
        assert report["omega_nominal"] == pytest.approx(0.95)
        assert report["nominal_mask_rate"] == 0.05
        assert report["realised_mask_rate"] == pytest.approx(1 / 55)
        assert report["realised_noise_rate"] == 0.0
        assert report["n_traces"] == 1
        assert (cell / "offlam_workspace" / "learner.log").exists()

    def test_the_hidden_atom_is_absent_from_the_trace_it_reads(self, masked_fold):
        data_dir, cell, prepared = masked_fold
        runner = OffLAMRunner().for_regime(DegradationRegime.simulated(0.05, 0.0, data_dir))
        runner.learn(BLOCKS_DOMAIN, prepared, cell, timeout_seconds=120)
        trace = (cell / "offlam_workspace" / "traces" / "0_problem1.trajectory").read_text()
        state_lines = [l for l in trace.splitlines() if l.startswith("(:state")]
        assert "(holding a)" not in state_lines[1]
        assert "(not (holding a))" not in state_lines[1]
        assert "(holding a)" in state_lines[3]
