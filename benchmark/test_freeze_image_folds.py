"""The image setup pass: which cells it freezes and what each one holds.

    python -m pytest benchmark/test_freeze_image_folds.py -q
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from benchmark.backfill_baseline import _build_prepared_trajectories
from benchmark.experiment_running_helpers.run_fold import cv_split, select_training_pool
from benchmark.freeze_image_folds import (
    FoldPlan,
    freeze_domain,
    is_frozen,
    load_image_config,
)
from src.utils.corpus_layout import GT_SUBPATH, IMAGE_POOL_SUBPATH, pool_problem_dirs
from src.utils.masking import save_masking_info
from src.utils.pddl_trajectory import build_trajectory_file

N_PROBLEMS = 5
EXPERIMENT = "exp"

_PROBLEM = """(define (problem {name}) (:domain blocks)
  (:objects a - block b - block)
  (:init (clear a) (clear b) (ontable a) (ontable b) (handempty))
  (:goal (and (on a b))))
"""
_STATES = [
    ["clear(a:block)", "clear(b:block)", "ontable(a:block)", "ontable(b:block)", "handempty()"],
    ["clear(b:block)", "ontable(b:block)", "holding(a:block)"],
    ["clear(a:block)", "on(a:block,b:block)", "ontable(b:block)", "handempty()"],
]
_ACTIONS = ["pick_up(a:block)", "stack(a:block,b:block)"]


def _steps() -> List[dict]:
    return [
        {"step": i + 1,
         "current_state": {"literals": list(_STATES[i])},
         "ground_action": action,
         "next_state": {"literals": list(_STATES[i + 1])}}
        for i, action in enumerate(_ACTIONS)
    ]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Five two-step blocks problems, every one with inferred files and GT."""
    root = tmp_path / "corpus"
    for index in range(N_PROBLEMS):
        name = f"problem{index}"
        folder = root / IMAGE_POOL_SUBPATH / name
        gt_folder = root / GT_SUBPATH / name
        folder.mkdir(parents=True)
        gt_folder.mkdir(parents=True)
        (folder / f"{name}.pddl").write_text(_PROBLEM.format(name=name))
        for target in (folder, gt_folder):
            (target / f"{name}_trajectory.json").write_text(json.dumps(_steps()))
            build_trajectory_file(_steps(), name, target)
        save_masking_info(folder, name, [set() for _ in range(len(_ACTIONS) + 1)])
    return root


def _plan(**overrides: object) -> FoldPlan:
    shared = {"n_folds": 2, "cv_scheme": "montecarlo", "num_trajectories": [2, 3],
              "gt_rates": [0], "normalize": False}
    shared.update(overrides)
    return FoldPlan.from_shared(shared)


def test_an_absent_key_takes_the_runner_default() -> None:
    plan = FoldPlan.from_shared({})
    assert (plan.n_folds, plan.cv_scheme, plan.normalize) == (5, "montecarlo", True)
    assert plan.num_trajectories == (3, 4, 5, 6, 7, 8)
    assert plan.frame_axiom_mode == "after_gt_only"


def test_a_simulated_config_is_refused(tmp_path: Path) -> None:
    config = tmp_path / "c.yaml"
    config.write_text("run_name: x\nsource: simulated\ndomains: [{domain_key: hanoi, data_dir: d}]\n")
    with pytest.raises(ValueError, match="source must be 'image'"):
        load_image_config(config)


def test_every_cell_of_the_plan_is_frozen_with_the_runner_split(corpus: Path, tmp_path: Path) -> None:
    out_root = tmp_path / "src_tree"
    plan = _plan()

    written, skipped = freeze_domain("blocksworld", corpus, EXPERIMENT, plan, out_root)

    assert (written, skipped) == (4, 0)
    testing = out_root / "blocksworld" / EXPERIMENT / "testing"
    problem_dirs = pool_problem_dirs(corpus / IMAGE_POOL_SUBPATH)
    for fold in range(2):
        train_idx, test_idx = cv_split(N_PROBLEMS, fold, n_folds=2, scheme="montecarlo")
        order = select_training_pool([problem_dirs[i] for i in train_idx], fold, 3)
        for size in (2, 3):
            cell = testing / f"fold{fold}_numtrajs{size}_gtrate0"
            info = json.loads((cell / "fold_info.json").read_text())
            assert [t["problem"] for t in info["trajectories"]] == [d.name for d in order[:size]]
            assert info["test_problems"] == [problem_dirs[i].name for i in test_idx]
            assert is_frozen(cell, size)


def test_a_frozen_cell_is_what_the_baseline_backfill_reads(corpus: Path, tmp_path: Path) -> None:
    out_root = tmp_path / "src_tree"
    freeze_domain("blocksworld", corpus, EXPERIMENT, _plan(), out_root)

    experiment = out_root / "blocksworld" / EXPERIMENT
    cell = experiment / "testing" / "fold0_numtrajs3_gtrate0"
    info = json.loads((cell / "fold_info.json").read_text())
    prepared = _build_prepared_trajectories(cell / "original_observations", corpus, info)

    assert len(prepared) == 3
    assert all(masking is not None for _, masking, _, _ in prepared)
    run_params = json.loads((experiment / "evaluation_results" / "run_params.json").read_text())
    assert run_params["data_dir"] == str(corpus)
    assert run_params["normalized"] is False
    assert run_params["data_source_type"] == "ImageDataSource"


def test_a_second_pass_rewrites_nothing(corpus: Path, tmp_path: Path) -> None:
    out_root = tmp_path / "src_tree"
    freeze_domain("blocksworld", corpus, EXPERIMENT, _plan(), out_root)

    assert freeze_domain("blocksworld", corpus, EXPERIMENT, _plan(), out_root) == (0, 4)
    assert freeze_domain("blocksworld", corpus, EXPERIMENT, _plan(), out_root, folds=[1], force=True) == (2, 0)


def test_a_training_size_beyond_the_fold_pool_raises(corpus: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exceeds fold 0's training pool of 4"):
        freeze_domain("blocksworld", corpus, EXPERIMENT, _plan(num_trajectories=[5]), tmp_path / "o")


def test_a_picked_problem_without_inferred_files_raises(corpus: Path, tmp_path: Path) -> None:
    for inferred in (corpus / IMAGE_POOL_SUBPATH).glob("*/problem*.masking_info"):
        inferred.unlink()
    for inferred in (corpus / IMAGE_POOL_SUBPATH).glob("*/problem*.trajectory"):
        inferred.unlink()
    with pytest.raises(FileNotFoundError):
        freeze_domain("blocksworld", corpus, EXPERIMENT, _plan(), tmp_path / "o")


def _literal_sets(trajectory: Path) -> List[List[str]]:
    """Each line of a trajectory file as its sorted parenthesised terms."""
    return [sorted(line.replace(")", ") ").split(") ")) for line in trajectory.read_text().splitlines()]


def test_frozen_observations_equal_the_ones_a_live_fold_saves(corpus: Path, tmp_path: Path) -> None:
    from benchmark.experiment_running_helpers.learning_helpers import _prepare_milp_driver_inputs
    from benchmark.experiment_running_helpers.trajectory_utils import prepare_fold_trajectories

    out_root = tmp_path / "src_tree"
    freeze_domain("blocksworld", corpus, EXPERIMENT, _plan(), out_root)
    cell = out_root / "blocksworld" / EXPERIMENT / "testing" / "fold0_numtrajs3_gtrate0"

    problem_dirs = pool_problem_dirs(corpus / IMAGE_POOL_SUBPATH)
    train_idx, _ = cv_split(N_PROBLEMS, 0, n_folds=2, scheme="montecarlo")
    order = select_training_pool([problem_dirs[i] for i in train_idx], 0, 3)
    live = tmp_path / "live"
    _prepare_milp_driver_inputs(
        cell / "domain_reference.pddl", prepare_fold_trajectories(order, 3, 0), live, None, None, "t",
    )

    frozen_files = sorted((cell / "original_observations").iterdir())
    live_files = sorted((live / "original_observations").iterdir())
    assert [f.name for f in frozen_files] == [f.name for f in live_files]
    assert len(frozen_files) == 6
    for frozen, saved in zip(frozen_files, live_files):
        assert _literal_sets(frozen) == _literal_sets(saved)
