"""The NOLAM arm: gated to mask-free cells, handed the realised noise, recorded.

    python -m pytest benchmark/baselines/test_nolam_runner.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.algorithm_adapters.lamanna.realised_rates import RealisedRates
from benchmark.baselines import resolve_baselines
from benchmark.baselines.nolam_runner import NOLAMRunner
from benchmark.baselines.regime import DegradationRegime, gate_baselines

ROOT = Path(__file__).resolve().parents[2]
BLOCKS_DOMAIN = ROOT / "src" / "domains" / "blocks" / "blocks.pddl"

PROBLEM = """(define (problem tiny) (:domain blocks)
  (:objects a - block b - block)
  (:init (clear a) (clear b) (ontable a) (ontable b) (handempty))
  (:goal (and (holding a))))
"""
GT = """(
(:init (clear a) (clear b) (ontable a) (ontable b) (handempty))
(operator: (pick_up a))
(:state (clear b) (ontable b) (holding a))
(operator: (stack a b))
(:state (clear a) (ontable b) (on a b) (handempty))
(operator: (unstack a b))
(:state (clear b) (ontable b) (holding a))
(operator: (put_down a))
(:state (clear a) (clear b) (ontable a) (ontable b) (handempty))
)
"""


class TestGate:
    def test_only_mask_free_simulated_cells(self):
        runner = NOLAMRunner()
        assert runner.supports(DegradationRegime.simulated(0.0, 0.3)) == (True, "")
        ok, why = runner.supports(DegradationRegime.simulated(0.1, 0.0))
        assert not ok and "masking_p=0.1" in why
        ok, why = runner.supports(DegradationRegime.image())
        assert not ok and "image" in why

    def test_binding_keeps_the_run_wide_runner_unbound(self):
        runner = NOLAMRunner()
        regime = DegradationRegime.simulated(0.0, 0.2, Path("/corpus"))
        (bound,), dropped = gate_baselines([runner], regime)
        assert dropped == {}
        assert bound.regime == regime
        assert runner.regime is None


class TestKnobs:
    def test_registered_and_defaults(self):
        (runner,) = resolve_baselines(["nolam"])
        assert runner.nolam_noise == "oracle"
        assert runner.nolam_allow_neg_precs is False
        assert runner.nolam_seed == 0
        assert runner.row_name(BLOCKS_DOMAIN) == "NOLAM"

    def test_knobs_reach_the_runner_and_the_record(self):
        (runner,) = resolve_baselines(
            ["nolam"], nolam_noise=0.2, nolam_allow_neg_precs=True, nolam_seed=5
        )
        assert runner.run_params() == {
            "nolam_noise": 0.2, "nolam_allow_neg_precs": True, "nolam_seed": 5,
        }
        assert runner.row_name(BLOCKS_DOMAIN) == "NOLAM__e=0.2__negprecs"

    def test_a_string_float_is_accepted_from_the_cli(self):
        assert NOLAMRunner(nolam_noise="0.3").nolam_noise == 0.3

    def test_an_impossible_noise_is_rejected(self):
        with pytest.raises(ValueError):
            NOLAMRunner(nolam_noise=1.0)

    def test_the_knobs_do_not_leak_into_other_arms(self):
        (runner,) = resolve_baselines(["rosame_24"], nolam_noise=0.2)
        assert not hasattr(runner, "nolam_noise")


class TestResolveE:
    def test_oracle_prefers_the_realised_rate(self):
        runner = NOLAMRunner().for_regime(DegradationRegime.simulated(0.0, 0.2))
        rates = RealisedRates(n_states=3, n_atoms=30, n_masked=0, n_flipped=6)
        assert runner._resolve_e(rates) == (0.2, "realised")

    def test_oracle_falls_back_to_the_nominal_rate(self):
        runner = NOLAMRunner().for_regime(DegradationRegime.simulated(0.0, 0.2))
        assert runner._resolve_e(None) == (0.2, "nominal")

    def test_oracle_without_a_regime_is_an_error(self):
        with pytest.raises(ValueError):
            NOLAMRunner()._resolve_e(None)

    def test_a_pinned_value_ignores_the_data(self):
        runner = NOLAMRunner(nolam_noise=0.1).for_regime(DegradationRegime.simulated(0.0, 0.4))
        rates = RealisedRates(n_states=3, n_atoms=30, n_masked=0, n_flipped=12)
        assert runner._resolve_e(rates) == (0.1, "pinned")


@pytest.fixture
def clean_fold(tmp_path):
    """A corpus with one clean GT trajectory frozen as the fold's observation."""
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
    (obs_dir / "original_observation_problem1.trajectory").write_text(GT)
    (obs_dir / "original_observation_problem1.masking_info").write_text("\n\n\n\n\n")
    prepared = [(
        obs_dir / "original_observation_problem1.trajectory",
        obs_dir / "original_observation_problem1.masking_info",
        problem_dir / "problem1.pddl",
        {0},
    )]
    return data_dir, cell, prepared


class TestLearn:
    def test_learns_a_model_from_a_clean_fold_and_records_the_row(self, clean_fold):
        data_dir, cell, prepared = clean_fold
        regime = DegradationRegime.simulated(0.0, 0.0, data_dir)
        runner = NOLAMRunner().for_regime(regime)

        model, report = runner.learn(BLOCKS_DOMAIN, prepared, cell, timeout_seconds=120)

        assert model is not None and "(:action pick_up" in model
        assert report["terminated_by"] == "completed"
        assert report["nolam_e_source"] == "realised"
        assert report["nolam_e_used"] == 0.0
        assert report["realised_noise_rate"] == 0.0
        assert report["realised_mask_rate"] == 0.0
        assert report["n_traces"] == 1
        assert (cell / "nolam_workspace" / "traces" / "0_problem1.trajectory").exists()
        assert (cell / "nolam_workspace" / "learner.log").exists()

    def test_unobserved_operators_are_left_out_and_recorded(self, clean_fold):
        """An operator NOLAM never saw would be emitted with an empty (and),
        which Fast Downward rejects with an internal error; it is dropped, as
        PI-SAM drops it, and the row says so."""
        data_dir, cell, prepared = clean_fold
        traj, masking, problem, gt = prepared[0]
        traj.write_text(
            "(\n(:init (clear a) (clear b) (ontable a) (ontable b) (handempty))\n"
            "(operator: (pick_up a))\n(:state (clear b) (ontable b) (holding a))\n"
            "(operator: (put_down a))\n(:state (clear a) (clear b) (ontable a) (ontable b) (handempty))\n)\n"
        )
        (data_dir / "gt_trajectories" / "problem1" / "problem1.trajectory").write_text(traj.read_text())
        masking.write_text("\n\n\n")
        runner = NOLAMRunner().for_regime(DegradationRegime.simulated(0.0, 0.0, data_dir))
        model, report = runner.learn(BLOCKS_DOMAIN, prepared, cell, timeout_seconds=120)
        assert "(:action pick_up" in model and "(:action put_down" in model
        assert "(:action stack" not in model and "(:action unstack" not in model
        assert report["unobserved_operators"] == ["stack", "unstack"]
        assert report["operators_dropped_empty"] == ["stack", "unstack"]
        assert report["n_operators_emitted"] == 2
        assert "(and\n\t\n)" not in model

    def test_the_clean_fold_recovers_the_pick_up_effects(self, clean_fold):
        data_dir, cell, prepared = clean_fold
        runner = NOLAMRunner().for_regime(DegradationRegime.simulated(0.0, 0.0, data_dir))
        model, _ = runner.learn(BLOCKS_DOMAIN, prepared, cell, timeout_seconds=120)
        pick_up = model[model.index("(:action pick_up"):model.index("(:action put_down")]
        assert "(holding ?param_1)" in pick_up
        assert "(not (ontable ?param_1))" in pick_up
