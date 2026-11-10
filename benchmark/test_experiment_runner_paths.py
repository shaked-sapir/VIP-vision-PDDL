"""The runner reads the pool its data source is for."""

from pathlib import Path

from benchmark.experiment_runner import _resolve_experiment_paths
from src.utils.corpus_layout import IMAGE_POOL_SUBPATH, POOL_SUBPATH


def test_an_image_run_reads_the_image_pool():
    _, trajectories_dir, _, _ = _resolve_experiment_paths(
        Path("/d"), "exp", "hanoi", None, IMAGE_POOL_SUBPATH)
    assert trajectories_dir == Path("/d") / "training" / "image_trajectories"


def test_a_simulated_run_reads_the_symbolic_pool():
    _, trajectories_dir, _, _ = _resolve_experiment_paths(
        Path("/d"), "exp", "hanoi", None, POOL_SUBPATH)
    assert trajectories_dir == Path("/d") / "training" / "trajectories"
