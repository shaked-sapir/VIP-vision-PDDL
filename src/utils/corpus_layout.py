"""Where the three pools of a corpus live, relative to its root.

    <corpus>/training/trajectories/problemN/        symbolic pool: pddl, plan, GT json
    <corpus>/training/image_trajectories/problemN/  image pool: pddl, GT json, frames,
                                                    inferred .trajectory + .masking_info
    <corpus>/gt_trajectories/problemN/              the GT trajectory

A simulated run reads the symbolic pool, an image run reads the image pool,
and neither reads across. Normalizing an image pool writes its text files to
``training/image_trajectories_normalized/``; the frames stay in the image pool.
"""

from __future__ import annotations

from pathlib import Path

POOL_SUBPATH = Path("training") / "trajectories"
IMAGE_POOL_SUBPATH = Path("training") / "image_trajectories"
IMAGE_POOL_NORMALIZED_SUBPATH = Path("training") / "image_trajectories_normalized"
GT_SUBPATH = Path("gt_trajectories")

#: Directory names an image problem folder may sit under.
IMAGE_POOL_DIR_NAMES = frozenset({IMAGE_POOL_SUBPATH.name, IMAGE_POOL_NORMALIZED_SUBPATH.name})

FRAME_GLOB = "state_*.png"


def frames_dir(image_problem_dir: Path) -> Path:
    """The folder holding ``state_*.png`` for an image problem folder.

    A folder under the normalized pool maps to its twin in the image pool,
    which is the only place frames are written.
    """
    image_problem_dir = Path(image_problem_dir)
    if image_problem_dir.parent.name == IMAGE_POOL_NORMALIZED_SUBPATH.name:
        return image_problem_dir.parent.parent / IMAGE_POOL_SUBPATH.name / image_problem_dir.name
    return image_problem_dir


def problem_files(problem_name: str) -> tuple[str, str]:
    """The two files a problem folder is defined by: its pddl and its GT json."""
    return f"{problem_name}.pddl", f"{problem_name}_trajectory.json"


def corpus_root(problem_dir: Path) -> Path:
    """The corpus a problem folder belongs to: ``<corpus>/training/<pool>/problemN`` → ``<corpus>``.

    Any pool directly under ``training/`` qualifies, the normalized copies
    included. Raises ``ValueError`` for a folder that is not laid out that way,
    so a caller never silently measures against the wrong ``gt_trajectories/``.
    """
    problem_dir = Path(problem_dir)
    pool = problem_dir.parent
    if pool.parent.name != POOL_SUBPATH.parent.name:
        raise ValueError(f"{problem_dir} is not a problem folder under a corpus's training/ pool")
    return pool.parent.parent
