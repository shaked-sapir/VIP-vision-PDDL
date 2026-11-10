from pathlib import Path

from src.utils.corpus_layout import (
    IMAGE_POOL_DIR_NAMES,
    IMAGE_POOL_NORMALIZED_SUBPATH,
    IMAGE_POOL_SUBPATH,
    POOL_SUBPATH,
    frames_dir,
    problem_files,
)


def test_the_pools_are_siblings_under_training():
    assert POOL_SUBPATH.parent == IMAGE_POOL_SUBPATH.parent == IMAGE_POOL_NORMALIZED_SUBPATH.parent
    assert IMAGE_POOL_DIR_NAMES == {"image_trajectories", "image_trajectories_normalized"}


def test_an_image_folder_holds_its_own_frames():
    folder = Path("/c") / IMAGE_POOL_SUBPATH / "problem3"
    assert frames_dir(folder) == folder


def test_a_normalized_folder_reads_the_image_pool_twin():
    folder = Path("/c") / IMAGE_POOL_NORMALIZED_SUBPATH / "problem3"
    assert frames_dir(folder) == Path("/c") / IMAGE_POOL_SUBPATH / "problem3"


def test_problem_files_are_the_pddl_and_the_gt_json():
    assert problem_files("problem7") == ("problem7.pddl", "problem7_trajectory.json")
