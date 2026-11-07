"""Feasibility gate for the renderers: re-render existing problem folders and diff.

A reference folder holds ``state_*.png`` frames beside the GT ``.trajectory``
and problem ``.pddl`` they were produced from. The gate replays the trajectory
through :class:`RenderedSource` with the domain's registered renderer and
counts, per frame, the pixels that differ from the stored frame.

    python benchmark/render_gate.py --domain depot \\
        --reference src/domains/depot/problems/problem1 ...

Reference layouts understood:

    <dir>/state_*.png + <dir>/<name>.trajectory + <dir>/<name>.pddl
    <corpus>/training/trajectories/<name>/ (frames, pddl)
        + <corpus>/gt_trajectories/<name>/<name>.trajectory
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trace_generation.frame_renderer import FrameRenderer, RenderedSource
from src.trace_generation.pddlgym_renderer import (
    BLOCKS_SCHEMA,
    PDDLGymRenderer,
    blocks_encounter_order,
    prime_block_colors,
)
from src.trace_generation.renderer_registry import build_renderer
from src.trace_generation.sources import FRAME_GLOB, TrajectorySource


@dataclass(frozen=True)
class FrameDiff:
    """How one re-rendered frame compares with its stored reference."""

    index: int
    differing_pixels: int
    size_mismatch: bool

    @property
    def identical(self) -> bool:
        """True when the two frames have the same size and every pixel matches."""
        return not self.size_mismatch and self.differing_pixels == 0


@dataclass(frozen=True)
class ProblemDiff:
    """The per-frame diffs of one reference problem."""

    name: str
    frames: List[FrameDiff]

    @property
    def identical(self) -> bool:
        """True when every frame reproduces exactly."""
        return all(frame.identical for frame in self.frames)


def reference_files(reference: Path) -> tuple[Path, Path, List[Path]]:
    """``(trajectory, problem_pddl, frames)`` for either reference layout."""
    reference = Path(reference)
    name = reference.name
    problem = reference / f"{name}.pddl"
    frames = sorted(reference.glob(FRAME_GLOB))
    trajectory = reference / f"{name}.trajectory"
    if not trajectory.is_file():
        corpus = reference.parent.parent.parent
        trajectory = corpus / "gt_trajectories" / name / f"{name}.trajectory"
    if not trajectory.is_file():
        raise FileNotFoundError(f"No GT .trajectory for reference {reference}.")
    if not frames:
        raise FileNotFoundError(f"No {FRAME_GLOB} frames in {reference}.")
    return trajectory, problem, frames


def diff_frames(rendered: Path, reference: Path) -> tuple[int, bool]:
    """``(differing_pixels, size_mismatch)`` between two PNGs, compared as RGB."""
    a = np.asarray(Image.open(rendered).convert("RGB"))
    b = np.asarray(Image.open(reference).convert("RGB"))
    if a.shape != b.shape:
        return 0, True
    return int((a != b).any(axis=2).sum()), False


def compare_reference(
    reference: Path,
    domain_file: Path,
    renderer: FrameRenderer,
    frames_dir: Optional[Path] = None,
) -> ProblemDiff:
    """Re-render one reference problem and diff every frame."""
    trajectory, problem, stored = reference_files(reference)
    if hasattr(renderer, "use_problem"):
        renderer.use_problem(problem)
    source = RenderedSource(TrajectorySource(trajectory, problem, domain_file),
                            renderer, frames_dir=frames_dir)
    steps = list(source.steps())
    rendered = [steps[0].frame_before] + [step.frame_after for step in steps]
    if len(rendered) != len(stored):
        raise ValueError(
            f"{reference} holds {len(stored)} frames but its trajectory has "
            f"{len(rendered)} states.")
    diffs = []
    for index, (ours, theirs) in enumerate(zip(rendered, stored)):
        pixels, mismatch = diff_frames(ours, theirs)
        diffs.append(FrameDiff(index, pixels, mismatch))
    return ProblemDiff(reference.name, diffs)


def blocks_renderer_primed_like(reference: Path, domain_file: Path) -> FrameRenderer:
    """A blocksworld renderer whose colours follow ``reference``'s first frame.

    PDDLGym colours blocks in order of first sight, so a walk's frames depend on
    the first state the process drew. The registry primes colours in sorted
    name order instead; this primes them as the old walk did, to show the two
    differ in nothing else.
    """
    from pddlgym.rendering.blocks import render

    trajectory, problem, _ = reference_files(reference)
    first = next(TrajectorySource(trajectory, problem, domain_file).steps()).prev_state
    order = blocks_encounter_order(first)
    return PDDLGymRenderer(render, BLOCKS_SCHEMA,
                           prime_colors=lambda _: prime_block_colors(order),
                           name="pddlgym.blocks[encounter-order colours]")


def run_gate(domain: str, domain_file: Path, references: Sequence[Path],
             visualizer_root: Optional[Path] = None,
             blocks_colors: str = "sorted") -> List[ProblemDiff]:
    """Compare every reference folder of one domain.

    Args:
        domain: A renderable domain key.
        domain_file: The eval-schema domain the references are written against.
        references: Problem folders to re-render.
        visualizer_root: The planning-visualizer checkout, for depot and gripper.
        blocks_colors: ``"sorted"`` uses the registry's renderer; ``"encounter"``
            primes blocksworld's colours from each reference's first frame.
    """
    if domain == "blocksworld" and blocks_colors == "encounter":
        return [compare_reference(Path(ref), domain_file,
                                  blocks_renderer_primed_like(Path(ref), domain_file))
                for ref in references]
    renderer = build_renderer(domain, visualizer_root=visualizer_root)
    try:
        return [compare_reference(Path(ref), domain_file, renderer) for ref in references]
    finally:
        if hasattr(renderer, "close"):
            renderer.close()


def format_report(domain: str, results: Sequence[ProblemDiff]) -> str:
    """One line per problem: its frame count and per-frame differing pixels."""
    lines = [f"{domain}:"]
    for result in results:
        pixels = [("size" if f.size_mismatch else str(f.differing_pixels))
                  for f in result.frames]
        verdict = "identical" if result.identical else "DIFFERS"
        lines.append(f"  {result.name}: {len(result.frames)} frames, {verdict}; "
                     f"differing px per frame: [{', '.join(pixels)}]")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--domain", required=True)
    parser.add_argument("--domain-file", type=Path, required=True)
    parser.add_argument("--reference", type=Path, nargs="+", required=True,
                        help="Problem folders holding state_*.png and their GT")
    parser.add_argument("--visualizer-root", type=Path, default=None)
    parser.add_argument("--blocks-colors", choices=["sorted", "encounter"],
                        default="sorted",
                        help="Prime blocksworld colours in sorted name order "
                             "(the renderer's convention) or as the reference's "
                             "own first frame did")
    args = parser.parse_args(argv)

    results = run_gate(args.domain, args.domain_file, args.reference,
                       visualizer_root=args.visualizer_root,
                       blocks_colors=args.blocks_colors)
    print(format_report(args.domain, results))
    return 0 if all(r.identical for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
