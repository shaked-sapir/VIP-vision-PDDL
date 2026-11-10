"""With ``example_dir`` the one-shot example is the held-out frame, not frame 0.

    python -m pytest src/fluent_classification/test_external_example.py
"""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.object_detection  # noqa: F401  (breaks a circular import in visualize)
from src.fluent_classification.llm_blocks_fluent_classifier import LLMBlocksFluentClassifier

STAGED = Path("benchmark/data/blocksworld/precheck_frames__seed7/training/trajectories")


def _problem(tmp_path: Path, name: str) -> Path:
    source = STAGED / name
    if not source.exists():
        pytest.skip(f"{source} not on disk")
    target = tmp_path / name
    target.mkdir()
    shutil.copy(source / f"{name}_trajectory.json", target / f"{name}_trajectory.json")
    (target / "state_000000.png").write_bytes(b"")
    return target


def _classifier(problem: Path, example_dir: Path | None) -> LLMBlocksFluentClassifier:
    return LLMBlocksFluentClassifier(
        llm_backend=SimpleNamespace(temperature=0.0),
        init_state_image_path=problem / "state_000000.png",
        type_to_objects={"block": ["red", "cyan", "blue", "green"]},
        example_dir=example_dir,
    )


def _state0_facts(problem: Path) -> set:
    return set(json.loads(next(problem.glob("*_trajectory.json")).read_text())[0]["current_state"]["literals"])


def test_without_example_dir_the_example_is_frame_0(tmp_path):
    problem = _problem(tmp_path, "problem0")
    classifier = _classifier(problem, None)
    image, _ = classifier.fewshot_examples[0]
    assert image == problem / "state_000000.png"
    assert classifier.example_dir is None


def test_with_example_dir_the_example_is_the_held_out_frame(tmp_path):
    problem = _problem(tmp_path, "problem0")
    example = _problem(tmp_path, "problem1")
    assert _state0_facts(problem) != _state0_facts(example), "the two frames must differ"

    classifier = _classifier(problem, example)
    image, _ = classifier.fewshot_examples[0]
    assert image == example / "state_000000.png"
    assert classifier.gt_json_trajectory_path == example / "problem1_trajectory.json"


def test_an_example_folder_needs_exactly_one_frame(tmp_path):
    problem = _problem(tmp_path, "problem0")
    example = _problem(tmp_path, "problem1")
    (example / "state_000001.png").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="exactly one"):
        _classifier(problem, example)
