"""Staging a fresh image corpus from an existing one.

    python -m pytest benchmark/test_stage_image_corpus.py
"""

from pathlib import Path

import pytest

from benchmark.stage_image_corpus import stage_corpus
from src.utils.corpus_layout import GT_SUBPATH, IMAGE_POOL_SUBPATH


@pytest.fixture
def source(tmp_path: Path) -> Path:
    corpus = tmp_path / "old"
    for name in ("problem0", "problem1"):
        folder = corpus / IMAGE_POOL_SUBPATH / name
        gt = corpus / GT_SUBPATH / name
        folder.mkdir(parents=True)
        gt.mkdir(parents=True)
        (folder / f"{name}.pddl").write_text("(define)")
        (folder / "state_0.png").write_bytes(b"png0")
        (folder / "state_1.png").write_bytes(b"png1")
        (folder / f"{name}.trajectory").write_text("inferred")
        (folder / f"{name}.masking_info").write_text("inferred")
        (folder / f"{name}_trajectory.json").write_text("noisy")
        (gt / f"{name}_trajectory.json").write_text("clean")
        (gt / f"{name}.trajectory").write_text("gt")
    return corpus


def test_frames_problem_and_clean_gt_are_copied(source: Path, tmp_path: Path) -> None:
    target = tmp_path / "new"

    assert stage_corpus(source, target) == ["problem0", "problem1"]

    folder = target / IMAGE_POOL_SUBPATH / "problem0"
    assert sorted(p.name for p in folder.iterdir()) == [
        "problem0.pddl", "problem0_trajectory.json", "state_0.png", "state_1.png"]
    assert (folder / "problem0_trajectory.json").read_text() == "clean"
    assert (target / GT_SUBPATH / "problem0" / "problem0.trajectory").read_text() == "gt"


def test_excluded_problems_are_left_out(source: Path, tmp_path: Path) -> None:
    target = tmp_path / "new"

    assert stage_corpus(source, target, exclude=["problem1"]) == ["problem0"]
    assert not (target / IMAGE_POOL_SUBPATH / "problem1").exists()
    assert not (target / GT_SUBPATH / "problem1").exists()


def test_an_existing_target_pool_is_refused(source: Path, tmp_path: Path) -> None:
    target = tmp_path / "new"
    stage_corpus(source, target)
    with pytest.raises(FileExistsError):
        stage_corpus(source, target)


def test_a_problem_without_gt_raises(source: Path, tmp_path: Path) -> None:
    for path in (source / GT_SUBPATH / "problem1").iterdir():
        path.unlink()
    with pytest.raises(FileNotFoundError, match="problem1"):
        stage_corpus(source, tmp_path / "new")
