"""Fold preparation: the initial state of a prepared trajectory, and missing inputs.

    python -m pytest benchmark/experiment_running_helpers/test_trajectory_utils.py
"""

import json
import re
from pathlib import Path
from typing import List

import pytest

from benchmark.experiment_running_helpers.trajectory_utils import (
    load_gt_indices,
    pregenerate_all_gt_frame_axiom_files,
    prepare_fold_trajectories,
)
from src.utils.corpus_layout import GT_SUBPATH, IMAGE_POOL_SUBPATH
from src.utils.gt_setup import GtSetup
from src.utils.masking import save_masking_info
from src.utils.pddl_trajectory import build_trajectory_file

DOMAIN = Path("src/domains/blocks/blocks.pddl")
NAME = "problem0"

PROBLEM = """(define (problem problem0) (:domain blocks)
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


def _gt_steps() -> List[dict]:
    return [
        {"step": i + 1,
         "current_state": {"literals": list(_STATES[i])},
         "ground_action": action,
         "next_state": {"literals": list(_STATES[i + 1])}}
        for i, action in enumerate(_ACTIONS)
    ]


def _positive_literals(trajectory_text: str, state_index: int) -> List[str]:
    """The literals of one ``(:init`` / ``(:state`` line of a trajectory file."""
    lines = [line for line in trajectory_text.splitlines()
             if line.startswith("(:init") or line.startswith("(:state")]
    return sorted(re.findall(r"\([a-z_]+(?: [a-z0-9_]+)*\)", lines[state_index]))


@pytest.fixture
def problem_dir(tmp_path: Path) -> Path:
    """A one-problem corpus whose inferred state 0 lacks one true fact."""
    corpus = tmp_path / "corpus"
    folder = corpus / IMAGE_POOL_SUBPATH / NAME
    gt_folder = corpus / GT_SUBPATH / NAME
    folder.mkdir(parents=True)
    gt_folder.mkdir(parents=True)
    (folder / f"{NAME}.pddl").write_text(PROBLEM)
    (gt_folder / f"{NAME}_trajectory.json").write_text(json.dumps(_gt_steps()))

    steps = _gt_steps()
    steps[0]["current_state"]["literals"] = steps[0]["current_state"]["literals"][1:]
    steps[-1]["next_state"]["literals"] = steps[-1]["next_state"]["literals"][1:]
    build_trajectory_file(steps, NAME, folder)
    save_masking_info(folder, NAME, [set() for _ in range(len(steps) + 1)])
    return folder


def test_gt_rate_zero_writes_the_true_initial_state(problem_dir: Path) -> None:
    inferred = (problem_dir / f"{NAME}.trajectory").read_text()

    pregenerate_all_gt_frame_axiom_files([problem_dir], DOMAIN, [0], "after_gt_only")

    prepared = (problem_dir / f"{NAME}_gtrate0_frame_axioms.trajectory").read_text()
    truth = (problem_dir / f"{NAME}.trajectory").with_name("truth.trajectory")
    build_trajectory_file(_gt_steps(), "truth", problem_dir)
    assert _positive_literals(prepared, 0) == _positive_literals(truth.read_text(), 0)
    assert _positive_literals(prepared, 0) != _positive_literals(inferred, 0)


def test_gt_rate_zero_keeps_the_inferred_file_and_leaves_no_intermediate(problem_dir: Path) -> None:
    inferred = (problem_dir / f"{NAME}.trajectory").read_text()

    pregenerate_all_gt_frame_axiom_files([problem_dir], DOMAIN, [0], "after_gt_only")

    assert (problem_dir / f"{NAME}.trajectory").read_text() == inferred
    assert not list(problem_dir.glob("*_gtends*"))
    assert load_gt_indices(problem_dir / f"{NAME}_gtrate0_frame_axioms.gt_indices") == {0}


def test_a_selected_problem_without_a_prepared_trajectory_raises(problem_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="1 of the 1 selected"):
        prepare_fold_trajectories([problem_dir], 1, 0)


def test_a_prepared_problem_is_returned(problem_dir: Path) -> None:
    pregenerate_all_gt_frame_axiom_files([problem_dir], DOMAIN, [0], "after_gt_only")

    prepared = prepare_fold_trajectories([problem_dir], 1, 0)

    assert len(prepared) == 1
    assert prepared[0][3] == {0}


def _prepared_states(problem_dir: Path) -> List[List[str]]:
    text = (problem_dir / f"{NAME}_gtrate0_frame_axioms.trajectory").read_text()
    return [_positive_literals(text, i) for i in range(len(_STATES))]


def _true_states(problem_dir: Path) -> List[List[str]]:
    build_trajectory_file(_gt_steps(), "truth", problem_dir)
    text = (problem_dir / "truth.trajectory").read_text()
    return [_positive_literals(text, i) for i in range(len(_STATES))]


@pytest.mark.parametrize("setup, true_states", [
    (GtSetup.NONE, []),
    (GtSetup.INIT, [0]),
    (GtSetup.FINAL, [2]),
    (GtSetup.INIT_FINAL, [0, 2]),
])
def test_a_setup_writes_exactly_its_endpoint_states(
    problem_dir: Path, setup: GtSetup, true_states: List[int]
) -> None:
    pregenerate_all_gt_frame_axiom_files([problem_dir], DOMAIN, [0], "after_gt_only", gt_setup=setup)

    prepared, truth = _prepared_states(problem_dir), _true_states(problem_dir)
    for index in (0, 2):
        assert (prepared[index] == truth[index]) is (index in true_states)
    assert load_gt_indices(problem_dir / f"{NAME}_gtrate0_frame_axioms.gt_indices") == set(true_states)


def test_a_pool_prepared_for_one_setup_refuses_another(problem_dir: Path) -> None:
    pregenerate_all_gt_frame_axiom_files([problem_dir], DOMAIN, [0], "after_gt_only", gt_setup=GtSetup.FINAL)

    with pytest.raises(ValueError, match="prepared for gt_setup 'final'"):
        pregenerate_all_gt_frame_axiom_files([problem_dir], DOMAIN, [0], "after_gt_only")


def test_a_setup_other_than_init_refuses_a_gt_rate(problem_dir: Path) -> None:
    with pytest.raises(ValueError, match="gt_rate 0 only"):
        pregenerate_all_gt_frame_axiom_files(
            [problem_dir], DOMAIN, [0, 25], "after_gt_only", gt_setup=GtSetup.INIT_FINAL)
