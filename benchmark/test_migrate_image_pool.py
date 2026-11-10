"""The one-off move of a corpus's image side into ``training/image_trajectories/``."""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "migrate_image_pool.py"
_spec = importlib.util.spec_from_file_location("migrate_image_pool", _SCRIPT)
migrate = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = migrate
_spec.loader.exec_module(migrate)


def _old_corpus(root: Path, *, frames: int = 3, inferred: bool = True,
                gt: bool = True, normalized: bool = False) -> Path:
    """A corpus in the pre-2026-11-10 layout, image side inside the symbolic pool."""
    pool = root / "training" / "trajectories" / "problem1"
    pool.mkdir(parents=True)
    (pool / "problem1.pddl").write_text("(define (problem problem1))")
    (pool / "problem1_trajectory.json").write_text("[]")
    (pool / "plan.txt").write_text("(a)\n")
    for i in range(frames):
        (pool / f"state_{i:04d}.png").write_bytes(b"png")
    if inferred:
        (pool / "problem1.trajectory").write_text("((:init))")
        (pool / "problem1.masking_info").write_text("\n")
        (pool / "problem1_gtrate0_frame_axioms.gt_indices").write_text("[0]")
    if gt:
        gt_dir = root / "gt_trajectories" / "problem1"
        gt_dir.mkdir(parents=True)
        (gt_dir / "problem1.trajectory").write_text("((:init))")
    if normalized:
        norm = root / "training" / "trajectories_normalized" / "problem1"
        norm.mkdir(parents=True)
        (norm / "problem1.pddl").write_text("(define (problem problem1))")
        (norm / "state_0000.png").write_bytes(b"png")
        (root / "training" / "trajectories_normalized" / "_domain.pddl").write_text("(define (domain d))")
    return root


def test_the_image_side_moves_and_the_problem_files_are_copied(tmp_path):
    corpus = _old_corpus(tmp_path)
    report = migrate.migrate_corpus(corpus)

    image = corpus / "training" / "image_trajectories" / "problem1"
    pool = corpus / "training" / "trajectories" / "problem1"
    assert sorted(p.name for p in image.iterdir()) == [
        "problem1.masking_info", "problem1.pddl", "problem1.trajectory",
        "problem1_gtrate0_frame_axioms.gt_indices", "problem1_trajectory.json",
        "state_0000.png", "state_0001.png", "state_0002.png"]
    assert sorted(p.name for p in pool.iterdir()) == [
        "plan.txt", "problem1.pddl", "problem1_trajectory.json"]
    assert (report.problems_moved, report.files_moved) == (1, 6)


def test_gt_trajectories_are_never_touched(tmp_path):
    corpus = _old_corpus(tmp_path)
    before = (corpus / "gt_trajectories" / "problem1" / "problem1.trajectory").read_text()
    migrate.migrate_corpus(corpus)
    assert (corpus / "gt_trajectories" / "problem1" / "problem1.trajectory").read_text() == before


def test_a_trajectory_without_its_gt_twin_aborts(tmp_path):
    corpus = _old_corpus(tmp_path, gt=False)
    with pytest.raises(FileNotFoundError, match="only trajectory"):
        migrate.migrate_corpus(corpus)
    assert (corpus / "training" / "trajectories" / "problem1" / "problem1.trajectory").exists()


def test_the_normalized_tree_is_renamed_without_its_frames(tmp_path):
    corpus = _old_corpus(tmp_path, normalized=True)
    report = migrate.migrate_corpus(corpus)

    new = corpus / "training" / "image_trajectories_normalized"
    assert not (corpus / "training" / "trajectories_normalized").exists()
    assert (new / "_domain.pddl").exists()
    assert (new / "problem1" / "problem1.pddl").exists()
    assert not list(new.rglob("*.png"))
    assert (report.normalized_renamed, report.normalized_frames_dropped) == (True, 1)


def test_frames_from_copies_shipped_frames_in(tmp_path):
    corpus = _old_corpus(tmp_path / "corpus", frames=0, inferred=True)
    shipped = tmp_path / "shipped" / "problem1"
    shipped.mkdir(parents=True)
    (shipped / "state_0000.png").write_bytes(b"shipped")

    report = migrate.migrate_corpus(corpus, frames_from=tmp_path / "shipped")

    image = corpus / "training" / "image_trajectories" / "problem1"
    assert (image / "state_0000.png").read_bytes() == b"shipped"
    assert (shipped / "state_0000.png").exists()
    assert report.frames_copied_in == 1


def test_a_problem_without_an_image_side_is_left_alone(tmp_path):
    corpus = _old_corpus(tmp_path, frames=0, inferred=False)
    report = migrate.migrate_corpus(corpus)
    assert not (corpus / "training" / "image_trajectories").exists()
    assert report.skipped == ["problem1"]


def test_dry_run_moves_nothing(tmp_path):
    corpus = _old_corpus(tmp_path, normalized=True)
    report = migrate.migrate_corpus(corpus, dry_run=True)
    assert report.problems_moved == 1
    assert not (corpus / "training" / "image_trajectories").exists()
    assert (corpus / "training" / "trajectories_normalized").exists()
    assert (corpus / "training" / "trajectories" / "problem1" / "state_0000.png").exists()
