"""Stage a fresh image corpus from an existing one: frames, problems and GT, no inference.

    python -m benchmark.stage_image_corpus --source <old corpus> --target <new corpus> \\
        [--exclude problem9]

The target's image pool holds, per problem, the source's ``state_*.png``, its
``.pddl`` and the clean ``_trajectory.json`` of the source's
``gt_trajectories/``; the target's ``gt_trajectories/`` is a copy of the
source's. Inferred files are not copied, so the target can be classified under
another protocol (``benchmark/infer_image_subset.py --all``).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List

from src.utils.corpus_layout import (
    FRAME_GLOB,
    GT_SUBPATH,
    IMAGE_POOL_SUBPATH,
    pool_problem_dirs,
    problem_files,
)
from src.utils.trajectory_json_converter import convert_trajectory_to_json


def stage_problem(source_dir: Path, source_gt: Path, target_dir: Path, target_gt: Path) -> int:
    """Copy one problem's frames, pddl and GT into the target; return the frame count.

    A GT folder holding only the ``.trajectory`` gets its ``_trajectory.json``
    converted from it.

    Raises:
        FileNotFoundError: if the problem lacks frames, its pddl or its GT.
    """
    name = source_dir.name
    pddl_name, gt_json_name = problem_files(name)
    frames = sorted(source_dir.glob(FRAME_GLOB))
    has_gt = (source_gt / gt_json_name).exists() or (source_gt / f"{name}.trajectory").exists()
    if not frames or not (source_dir / pddl_name).exists() or not has_gt:
        raise FileNotFoundError(f"{name}: frames, {pddl_name} or its GT under {source_gt} are missing")

    target_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        shutil.copy2(frame, target_dir / frame.name)
    shutil.copy2(source_dir / pddl_name, target_dir / pddl_name)
    shutil.copytree(source_gt, target_gt, dirs_exist_ok=True)
    if not (target_gt / gt_json_name).exists():
        convert_trajectory_to_json(
            trajectory_path=target_gt / f"{name}.trajectory",
            problem_path=target_dir / pddl_name,
            output_path=target_gt / gt_json_name,
        )
    shutil.copy2(target_gt / gt_json_name, target_dir / gt_json_name)
    return len(frames)


def stage_corpus(source: Path, target: Path, exclude: Iterable[str] = ()) -> List[str]:
    """Stage every problem of ``source`` not named in ``exclude``; return their names.

    Raises:
        FileExistsError: if the target's image pool already exists.
    """
    target_pool = target / IMAGE_POOL_SUBPATH
    if target_pool.exists():
        raise FileExistsError(f"{target_pool} already exists")
    excluded = set(exclude)
    staged: List[str] = []
    for source_dir in pool_problem_dirs(source / IMAGE_POOL_SUBPATH):
        name = source_dir.name
        if name in excluded:
            continue
        stage_problem(source_dir, source / GT_SUBPATH / name,
                      target_pool / name, target / GT_SUBPATH / name)
        staged.append(name)
    return staged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--exclude", nargs="*", default=[], help="problem folder names to leave out")
    args = parser.parse_args()
    staged = stage_corpus(args.source, args.target, args.exclude)
    print(f"staged {len(staged)} problems into {args.target}")


if __name__ == "__main__":
    main()
