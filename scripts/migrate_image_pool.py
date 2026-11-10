"""Move a corpus's image side out of ``training/trajectories/`` into ``training/image_trajectories/``.

One-off migration to the layout in ``src/utils/corpus_layout.py``. For every
``training/trajectories/problemN/`` that holds image-side files (frames, an
inferred ``.trajectory`` / ``.masking_info``, the derived ``_gtrate*`` files)
the files move to ``training/image_trajectories/problemN/`` and the problem's
pddl and GT json are copied beside them. ``training/trajectories_normalized/``
is renamed to ``training/image_trajectories_normalized/`` and loses its frame
copies. ``gt_trajectories/`` and the symbolic pool's own files are not touched.

    python scripts/migrate_image_pool.py benchmark/data/hanoi/large_corpus_n2500
    python scripts/migrate_image_pool.py <corpus> --frames-from src/domains/depot/problems
    python scripts/migrate_image_pool.py <corpus>... --dry-run

``--frames-from`` copies ``<dir>/problemN/state_*.png`` in for corpora whose
frames were never stored in the data dir (the shipped depot and gripper ones).
A problem whose inferred ``.trajectory`` would move without a GT counterpart
under ``gt_trajectories/`` aborts the corpus, since that file would then be the
only trajectory it has.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.utils.corpus_layout import (  # noqa: E402
    FRAME_GLOB,
    GT_SUBPATH,
    IMAGE_POOL_NORMALIZED_SUBPATH,
    IMAGE_POOL_SUBPATH,
    POOL_SUBPATH,
    problem_files,
)

IMAGE_SIDE_GLOBS = (FRAME_GLOB, "*.trajectory", "*.masking_info", "*.gt_indices")
OLD_NORMALIZED_SUBPATH = Path("training") / "trajectories_normalized"


@dataclass
class Report:
    """What one corpus migration did."""

    corpus: Path
    problems_moved: int = 0
    files_moved: int = 0
    frames_copied_in: int = 0
    normalized_renamed: bool = False
    normalized_frames_dropped: int = 0
    skipped: List[str] = field(default_factory=list)


def image_side_files(problem_dir: Path) -> List[Path]:
    """The files of a symbolic-pool folder that belong to the image pool."""
    found: List[Path] = []
    for pattern in IMAGE_SIDE_GLOBS:
        found.extend(problem_dir.glob(pattern))
    return sorted(set(found))


def _check_gt_present(corpus: Path, problem_dir: Path, moving: Sequence[Path]) -> None:
    """Refuse to move a ``.trajectory`` whose GT twin does not exist."""
    name = problem_dir.name
    if any(f.suffix == ".trajectory" for f in moving):
        gt = corpus / GT_SUBPATH / name / f"{name}.trajectory"
        if not gt.is_file():
            raise FileNotFoundError(
                f"{problem_dir} holds a .trajectory but {gt} is missing; refusing to "
                f"move the corpus's only trajectory for {name}.")


def migrate_problem(corpus: Path, problem_dir: Path, image_dir: Path,
                    frames_from: Optional[Path], report: Report, dry_run: bool) -> None:
    """Move one problem's image-side files and copy its problem files."""
    name = problem_dir.name
    moving = image_side_files(problem_dir)
    incoming = sorted((frames_from / name).glob(FRAME_GLOB)) if frames_from else []
    if not moving and not incoming:
        report.skipped.append(name)
        return
    _check_gt_present(corpus, problem_dir, moving)

    if not dry_run:
        image_dir.mkdir(parents=True, exist_ok=True)
        for f in moving:
            shutil.move(str(f), str(image_dir / f.name))
        for f in incoming:
            shutil.copy(f, image_dir / f.name)
        for file_name in problem_files(name):
            src = problem_dir / file_name
            if src.is_file():
                shutil.copy(src, image_dir / file_name)
    report.problems_moved += 1
    report.files_moved += len(moving)
    report.frames_copied_in += len(incoming)


def migrate_normalized(corpus: Path, report: Report, dry_run: bool) -> None:
    """Rename the old normalized tree and drop the frame copies inside it."""
    old = corpus / OLD_NORMALIZED_SUBPATH
    new = corpus / IMAGE_POOL_NORMALIZED_SUBPATH
    if not old.is_dir():
        return
    if new.exists():
        raise FileExistsError(f"Both {old} and {new} exist; resolve by hand.")
    frames = list(old.rglob(FRAME_GLOB))
    if not dry_run:
        for f in frames:
            f.unlink()
        old.rename(new)
    report.normalized_renamed = True
    report.normalized_frames_dropped = len(frames)


def migrate_corpus(corpus: Path, *, frames_from: Optional[Path] = None,
                   dry_run: bool = False) -> Report:
    """Migrate one corpus; returns what was done."""
    corpus = Path(corpus)
    pool = corpus / POOL_SUBPATH
    if not pool.is_dir():
        raise FileNotFoundError(f"{corpus} has no {POOL_SUBPATH}.")
    report = Report(corpus)
    for problem_dir in sorted(d for d in pool.iterdir() if d.is_dir()):
        migrate_problem(corpus, problem_dir, corpus / IMAGE_POOL_SUBPATH / problem_dir.name,
                        frames_from, report, dry_run)
    migrate_normalized(corpus, report, dry_run)
    return report


def format_report(report: Report) -> str:
    """One line per corpus."""
    parts = [f"{report.corpus}: {report.problems_moved} problems moved "
             f"({report.files_moved} files)"]
    if report.frames_copied_in:
        parts.append(f"{report.frames_copied_in} frames copied in")
    if report.normalized_renamed:
        parts.append(f"normalized tree renamed, {report.normalized_frames_dropped} frame copies dropped")
    if report.skipped:
        parts.append(f"{len(report.skipped)} problems had no image side")
    return ", ".join(parts)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("corpora", nargs="+", type=Path, help="Corpus roots to migrate")
    parser.add_argument("--frames-from", type=Path, default=None,
                        help="Directory of problemN/ folders whose state_*.png to copy in")
    parser.add_argument("--dry-run", action="store_true", help="Report only")
    args = parser.parse_args(argv)

    for corpus in args.corpora:
        report = migrate_corpus(corpus, frames_from=args.frames_from, dry_run=args.dry_run)
        print(("[dry-run] " if args.dry_run else "") + format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
