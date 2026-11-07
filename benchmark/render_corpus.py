"""Render ``state_*.png`` frames into an existing symbolic corpus, problem by problem.

Each ``training/trajectories/problemN/`` is replayed from its GT trajectory
(``gt_trajectories/problemN/problemN.trajectory``) through the domain's
renderer and gets one frame per state. GT files are never touched. A folder
that already holds the right number of frames is skipped, so a run can be
resumed or split into chunks:

    python benchmark/render_corpus.py --domain hanoi \\
        --corpus benchmark/data/hanoi/large_corpus_n2500 --domain-file src/domains/hanoi/hanoi.pddl
    python benchmark/render_corpus.py ... --chunk 3 --num-chunks 10   # problems 3, 13, 23, ...
    python benchmark/render_corpus.py ... --limit 3                   # the first three only
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trace_generation.cutter import CutMode, cut
from src.trace_generation.emitter import GT_SUBPATH, POOL_SUBPATH, write_window_frames
from src.trace_generation.frame_renderer import FrameRenderer, RenderedSource
from src.trace_generation.renderer_registry import build_renderer
from src.trace_generation.sources import FRAME_GLOB, TrajectorySource


@dataclass(frozen=True)
class RenderOutcome:
    """What happened to one problem folder."""

    name: str
    frames: int
    skipped: bool


def _natural_key(path: Path) -> List:
    import re

    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", path.name)]


def problem_dirs(corpus: Path) -> List[Path]:
    """The corpus's problem folders in natural order."""
    pool = Path(corpus) / POOL_SUBPATH
    found = sorted((d for d in pool.iterdir() if d.is_dir()), key=_natural_key)
    if not found:
        raise FileNotFoundError(f"No problem folders under {pool}.")
    return found


def select(dirs: Sequence[Path], *, chunk: int, num_chunks: int,
           limit: Optional[int]) -> List[Path]:
    """Every ``num_chunks``-th folder starting at ``chunk``, capped at ``limit``."""
    chosen = list(dirs[chunk::num_chunks])
    return chosen[:limit] if limit is not None else chosen


def expected_frames(trajectory_file: Path) -> int:
    """States in a ``.trajectory``: one more than its operator lines."""
    text = trajectory_file.read_text()
    return text.count("(operator:") + 1


def frames_present(problem_dir: Path) -> int:
    """How many ``state_*.png`` the folder already holds."""
    return len(list(problem_dir.glob(FRAME_GLOB)))


def render_problem(problem_dir: Path, gt_trajectory: Path, domain_file: Path,
                   renderer: FrameRenderer) -> int:
    """Render one problem's frames into its folder; returns the frame count.

    Frames are rendered into a scratch directory and copied in one go, so a
    folder with a wrong frame count always means an interrupted run.
    """
    name = problem_dir.name
    if hasattr(renderer, "use_problem"):
        renderer.use_problem(problem_dir / f"{name}.pddl")
    scratch = Path(tempfile.mkdtemp(prefix=f"render_{name}_"))
    try:
        source = RenderedSource(
            TrajectorySource(gt_trajectory, problem_dir / f"{name}.pddl", domain_file),
            renderer, frames_dir=scratch)
        windows = cut(source.steps(), mode=CutMode.NONE)
        if not windows:
            raise ValueError(f"{gt_trajectory} yielded no window.")
        for stale in problem_dir.glob(FRAME_GLOB):
            stale.unlink()
        write_window_frames(windows[0], problem_dir)
        return windows[0].length + 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def render_corpus(corpus: Path, domain: str, domain_file: Path, *,
                  chunk: int = 0, num_chunks: int = 1, limit: Optional[int] = None,
                  visualizer_root: Optional[Path] = None,
                  force: bool = False) -> List[RenderOutcome]:
    """Render every selected problem of ``corpus`` that is not already complete."""
    corpus = Path(corpus)
    selected = select(problem_dirs(corpus), chunk=chunk, num_chunks=num_chunks, limit=limit)
    renderer = build_renderer(domain, visualizer_root=visualizer_root)
    outcomes: List[RenderOutcome] = []
    try:
        for index, problem_dir in enumerate(selected):
            name = problem_dir.name
            gt = corpus / GT_SUBPATH / name / f"{name}.trajectory"
            wanted = expected_frames(gt)
            if not force and frames_present(problem_dir) == wanted:
                outcomes.append(RenderOutcome(name, wanted, skipped=True))
                continue
            frames = render_problem(problem_dir, gt, domain_file, renderer)
            outcomes.append(RenderOutcome(name, frames, skipped=False))
            if (index + 1) % 50 == 0 or index + 1 == len(selected):
                print(f"  {index + 1}/{len(selected)} problems done "
                      f"({sum(not o.skipped for o in outcomes)} rendered)", flush=True)
    finally:
        if hasattr(renderer, "close"):
            renderer.close()
    return outcomes


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--domain", required=True, help="Renderer registry key")
    parser.add_argument("--corpus", type=Path, required=True,
                        help="Corpus root holding training/trajectories and gt_trajectories")
    parser.add_argument("--domain-file", type=Path, required=True,
                        help="The eval-schema domain the corpus is written against")
    parser.add_argument("--chunk", type=int, default=0, help="This task's chunk index")
    parser.add_argument("--num-chunks", type=int, default=1, help="How many chunks in all")
    parser.add_argument("--limit", type=int, default=None,
                        help="Render at most this many problems of the chunk")
    parser.add_argument("--visualizer-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Re-render folders that already have every frame")
    args = parser.parse_args(argv)

    outcomes = render_corpus(args.corpus, args.domain, args.domain_file,
                             chunk=args.chunk, num_chunks=args.num_chunks,
                             limit=args.limit, visualizer_root=args.visualizer_root,
                             force=args.force)
    rendered = [o for o in outcomes if not o.skipped]
    print(f"{args.domain} {args.corpus}: {len(rendered)} problems rendered, "
          f"{len(outcomes) - len(rendered)} already complete, "
          f"{sum(o.frames for o in rendered)} frames written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
