"""Normalization reads the image pool and writes its text files, not its frames."""

from pathlib import Path

from benchmark.experiment_running_helpers.normalize import normalize_experiment_data

GRIPPER_DOMAIN = Path(__file__).resolve().parents[1] / "src" / "domains" / "gripper" / "gripper.pddl"


def _image_corpus(root: Path) -> Path:
    image = root / "training" / "image_trajectories" / "problem0"
    image.mkdir(parents=True)
    (image / "problem0.pddl").write_text(
        "(define (problem problem0) (:domain gripper)\n"
        "  (:objects rooma - room)\n  (:init (at-robby rooma))\n  (:goal (at-robby rooma)))\n")
    (image / "problem0.trajectory").write_text("((:init (at-robby rooma)))")
    (image / "state_000000.png").write_bytes(b"png")
    return root


def test_the_text_files_are_normalized_into_the_image_pool_twin(tmp_path):
    data_dir = _image_corpus(tmp_path)
    normalized_dir, domain = normalize_experiment_data(data_dir, GRIPPER_DOMAIN)

    assert normalized_dir == data_dir / "training" / "image_trajectories_normalized"
    assert domain == normalized_dir / "_domain.pddl"
    assert "at_robby" in (normalized_dir / "problem0" / "problem0.pddl").read_text()
    assert "at_robby" in (normalized_dir / "problem0" / "problem0.trajectory").read_text()


def test_frames_stay_in_the_image_pool(tmp_path):
    data_dir = _image_corpus(tmp_path)
    normalized_dir, _ = normalize_experiment_data(data_dir, GRIPPER_DOMAIN)

    assert not list(normalized_dir.rglob("*.png"))
    assert (data_dir / "training" / "image_trajectories" / "problem0" / "state_000000.png").exists()
    assert "at-robby" in (data_dir / "training" / "image_trajectories" / "problem0" / "problem0.pddl").read_text()
