"""The realised-corruption backfill: cell discovery, frozen tuples, config reading.

    python -m pytest benchmark/test_backfill_realised_corruption.py
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from benchmark.backfill_realised_corruption import (
    backfill_experiment,
    dashboard_image_experiments,
    frozen_fold_trajectories,
    measurable_cells,
    project_root,
)
from benchmark.experiment_running_helpers.fold_corruption import read_fold_corruption

ROOT = Path(__file__).resolve().parents[1]
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
)
"""
DEGRADED = """(
(:init (clear a) (clear b) (ontable a) (ontable b) (handempty))
(operator: (pick_up a))
(:state (clear b) (clear a) (ontable b))
)
"""


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    (corpus / "training" / "image_trajectories" / "problem7").mkdir(parents=True)
    (corpus / "training" / "image_trajectories" / "problem7" / "problem7.pddl").write_text(PROBLEM)
    (corpus / "gt_trajectories" / "problem7").mkdir(parents=True)
    (corpus / "gt_trajectories" / "problem7" / "problem7.trajectory").write_text(GT)
    return corpus


def _experiment(tmp_path: Path, corpus: Path) -> Path:
    experiment = tmp_path / "running_results" / "blocksworld" / "TO=600__tiny"
    (experiment / "evaluation_results").mkdir(parents=True)
    (experiment / "evaluation_results" / "run_params.json").write_text(
        json.dumps({"data_dir": str(corpus), "normalized": False})
    )
    cell = experiment / "testing" / "fold0_numtrajs1_gtrate0"
    (cell / "original_observations").mkdir(parents=True)
    (cell / "original_observations" / "original_observation_problem7.trajectory").write_text(DEGRADED)
    (cell / "original_observations" / "original_observation_problem7.masking_info").write_text("\n\n")
    (cell / "domain_reference.pddl").write_text(BLOCKS_DOMAIN.read_text())
    (cell / "fold_info.json").write_text(json.dumps({"fold": 0, "num_trajectories": 1}))
    (experiment / "testing" / "fold0_gtrate0_shared").mkdir()  # no observations: not a cell
    return experiment


def test_only_fold_instances_with_frozen_observations_are_measurable(tmp_path: Path):
    experiment = _experiment(tmp_path, _corpus(tmp_path))
    assert [c.name for c in measurable_cells(experiment)] == ["fold0_numtrajs1_gtrate0"]
    assert measurable_cells(tmp_path / "nowhere") == []


def test_frozen_tuples_pair_each_observation_with_its_problem_pddl(tmp_path: Path):
    corpus = _corpus(tmp_path)
    cell = _experiment(tmp_path, corpus) / "testing" / "fold0_numtrajs1_gtrate0"
    [(trajectory, masking, pddl, gt)] = frozen_fold_trajectories(cell, corpus)
    assert trajectory.name == "original_observation_problem7.trajectory"
    assert masking.suffix == ".masking_info"
    assert pddl == corpus / "training" / "image_trajectories" / "problem7" / "problem7.pddl"
    assert gt == {0}


def test_backfill_writes_the_fields_and_keeps_the_fold_info(tmp_path: Path):
    corpus = _corpus(tmp_path)
    experiment = _experiment(tmp_path, corpus)
    assert backfill_experiment(experiment, None) == (1, 1)
    cell = experiment / "testing" / "fold0_numtrajs1_gtrate0"
    info = json.loads((cell / "fold_info.json").read_text())
    assert info["fold"] == 0 and info["num_trajectories"] == 1
    # State 1: (holding a) and (clear a) flipped out of 11 atoms; frame 0 clean.
    assert info["n_atoms"] == 22
    assert info["realised_flip_rate"] == 2 / 22
    block = read_fold_corruption(cell)
    assert block["frame0"]["n_flipped"] == 0 and block["later_frames"]["n_flipped"] == 2
    assert block["gt_root"] == str(corpus / "gt_trajectories")


def test_a_missing_corpus_skips_the_experiment_loudly(tmp_path: Path, capsys):
    experiment = _experiment(tmp_path, tmp_path / "gone")
    assert backfill_experiment(experiment, None) == (0, 0)
    assert "does not exist" in capsys.readouterr().out


def test_dashboard_config_names_the_image_cells(tmp_path: Path):
    config = tmp_path / "dashboard_config.yaml"
    config.write_text(yaml.safe_dump({"results_dashboard": {"image": {
        "small": {"experiment_dir": {"blocksworld": "benchmark/running_results/blocksworld/TO=600__x  "}},
        "large": {"experiment_dir": {}},
    }}}))
    assert dashboard_image_experiments(config, "small") == {
        "blocksworld": project_root / "benchmark/running_results/blocksworld/TO=600__x"
    }
    assert dashboard_image_experiments(config, "large") == {}
