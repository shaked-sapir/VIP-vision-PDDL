"""Realised corruption rates are measured, not read off the config.

    python -m pytest benchmark/algorithm_adapters/lamanna/test_realised_rates.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.algorithm_adapters.lamanna.realised_rates import (
    gt_trajectory_lookup,
    realised_rates,
)

ROOT = Path(__file__).resolve().parents[3]
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
)
"""

# State 1: (holding a) flipped to false, (clear a) flipped to true.
# State 2: (clear b) hidden (its GT sign is negative there).
DEGRADED = """(
(:init (clear a) (clear b) (ontable a) (ontable b) (handempty))
(operator: (pick_up a))
(:state (clear b) (clear a) (ontable b))
(operator: (stack a b))
(:state (clear a) (ontable b) (on a b) (handempty))
)
"""
# The typed spelling ``save_masking_info`` writes.
MASKING_INFO = "\n\n(not (clear b - block))\n"

N_ATOMS = 11
N_STATES = 3


@pytest.fixture
def layout(tmp_path):
    data_dir = tmp_path / "corpus"
    problem_dir = data_dir / "training" / "trajectories" / "problem7"
    problem_dir.mkdir(parents=True)
    (problem_dir / "problem7.pddl").write_text(PROBLEM)
    gt_dir = data_dir / "gt_trajectories" / "problem7"
    gt_dir.mkdir(parents=True)
    (gt_dir / "problem7.trajectory").write_text(GT)
    cell = tmp_path / "cell" / "original_observations"
    cell.mkdir(parents=True)
    (cell / "original_observation_problem7.trajectory").write_text(DEGRADED)
    (cell / "original_observation_problem7.masking_info").write_text(MASKING_INFO)
    prepared = [(
        cell / "original_observation_problem7.trajectory",
        cell / "original_observation_problem7.masking_info",
        problem_dir / "problem7.pddl",
        {0},
    )]
    return data_dir, prepared


def test_lookup_maps_problem_names_to_gt_files(layout):
    data_dir, _ = layout
    lookup = gt_trajectory_lookup(data_dir)
    assert set(lookup) == {"problem7"}
    assert lookup["problem7"].name == "problem7.trajectory"


def test_counts_hidden_and_flipped_atoms_against_gt(layout):
    data_dir, prepared = layout
    rates = realised_rates(prepared, BLOCKS_DOMAIN, gt_trajectory_lookup(data_dir))
    assert rates.n_states == N_STATES
    assert rates.n_atoms == N_STATES * N_ATOMS
    assert rates.n_masked == 1
    assert rates.n_flipped == 2
    assert rates.mask_rate == pytest.approx(1 / 33)
    assert rates.noise_rate == pytest.approx(2 / 32)


def test_a_masked_atom_is_not_a_flip_even_when_its_sign_disagrees(layout):
    """The masked (clear b) is absent from the degraded state 2 text, so a
    positives-only reading would call it false; its recorded GT sign is what
    counts, and a hidden atom is hidden, not wrong."""
    data_dir, prepared = layout
    rates = realised_rates(prepared, BLOCKS_DOMAIN, gt_trajectory_lookup(data_dir))
    assert rates.n_flipped == 2  # the two in state 1 only


def test_a_clean_fold_measures_zero(layout):
    data_dir, prepared = layout
    traj, masking, problem, gt = prepared[0]
    traj.write_text(GT)
    masking.write_text("\n\n\n")
    rates = realised_rates([(traj, masking, problem, gt)], BLOCKS_DOMAIN, gt_trajectory_lookup(data_dir))
    assert (rates.n_masked, rates.n_flipped) == (0, 0)
    assert rates.mask_rate == 0.0
    assert rates.noise_rate == 0.0


def test_a_missing_gt_entry_is_an_error(layout):
    _, prepared = layout
    with pytest.raises(KeyError):
        realised_rates(prepared, BLOCKS_DOMAIN, {})


def test_as_dict_carries_counts_and_rates(layout):
    data_dir, prepared = layout
    payload = realised_rates(prepared, BLOCKS_DOMAIN, gt_trajectory_lookup(data_dir)).as_dict()
    assert set(payload) == {"n_states", "n_atoms", "n_masked", "n_flipped", "mask_rate", "noise_rate"}
