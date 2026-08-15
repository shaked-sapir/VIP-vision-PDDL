"""Compose a source, the cutter and the emitter into one corpus on disk.

This is the only place the three pieces meet, and it stays source-agnostic: it
sees a :class:`TraceSource`, a cut policy and a destination.

    <corpus>/training/trajectories/problemN/    the pool a fold trains on
    <corpus>/gt_trajectories/problemN/          the matching ground truth
    <corpus>/generation_info.json               source + cut parameters
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from src.trace_generation.cutter import CutMode, cut
from src.trace_generation.emitter import (
    POOL_SUBPATH,
    EmittedProblem,
    build_generation_info,
    emit_window,
    write_generation_info,
)
from src.trace_generation.sources import (
    DEFAULT_MAX_STEPS,
    ProblemWalkSource,
    TraceSource,
    TrajectorySource,
)

logger = logging.getLogger(__name__)

DEFAULT_PROBLEM_PREFIX = "problem"


class SourceKind(str, Enum):
    """Which :class:`TraceSource` a corpus is generated from."""

    PROBLEM = "problem"
    TRAJECTORY = "trajectory"


@dataclass(frozen=True)
class Corpus:
    """Where a generated corpus landed, and what went into it."""

    root: Path
    trajectories_dir: Path
    problems: Tuple[EmittedProblem, ...]
    info: Dict[str, Any]
    info_file: Path

    @property
    def num_problems(self) -> int:
        """How many problem folders were written."""
        return len(self.problems)


def build_source(
    kind: SourceKind,
    *,
    domain_file: Path,
    problem_file: Optional[Path] = None,
    trajectory_file: Optional[Path] = None,
    backend: str = "native",
    p_rnd: float = 1.0,
    seed: Optional[int] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    stop_at_goal: bool = False,
    attach_frames: bool = False,
    frames_dir: Optional[Path] = None,
) -> TraceSource:
    """Build the source ``kind`` names, reading only the arguments it needs.

    Args:
        kind: ``PROBLEM`` walks ``problem_file``; ``TRAJECTORY`` replays
            ``trajectory_file``.
        domain_file: The PDDL domain, required by both kinds.
        problem_file: The problem to walk, or the trajectory's problem.
        trajectory_file: The trajectory to replay. ``TRAJECTORY`` only.
        backend: Walk backend. ``PROBLEM`` only.
        p_rnd: Probability of a random action. ``PROBLEM`` only.
        seed: Walk RNG seed, independent of the cutter's. ``PROBLEM`` only.
        max_steps: Cap on walked transitions. ``PROBLEM`` only.
        stop_at_goal: Stop when the plan completes. ``PROBLEM`` only.
        attach_frames: Pair each step with its images. ``TRAJECTORY`` only.
        frames_dir: Where those images live. ``TRAJECTORY`` only.

    Returns:
        A configured :class:`TraceSource`.

    Raises:
        ValueError: A file needed by ``kind`` is missing, or one belonging to
            the other kind was supplied.
    """
    kind = SourceKind(kind)
    if problem_file is None:
        raise ValueError(f"source_kind={kind.value} requires a problem file.")

    if kind is SourceKind.PROBLEM:
        if trajectory_file is not None:
            raise ValueError(
                "source_kind=problem walks the problem file; it has no use for "
                f"trajectory_file={str(trajectory_file)!r}. Use "
                "source_kind=trajectory to replay it.")
        return ProblemWalkSource(
            problem_file, domain_file, backend=backend, p_rnd=p_rnd, seed=seed,
            max_steps=max_steps, stop_at_goal=stop_at_goal)

    if trajectory_file is None:
        raise ValueError("source_kind=trajectory requires a trajectory file.")
    return TrajectorySource(
        trajectory_file, problem_file, domain_file,
        attach_frames=attach_frames, frames_dir=frames_dir)


def generate_corpus(
    source: TraceSource,
    *,
    corpus_root: Path,
    cut_mode: CutMode,
    length_range: Optional[Tuple[int, int]] = None,
    buckets: Optional[Sequence[int]] = None,
    skip: int = 1,
    num_problems: Optional[int] = None,
    seed: Optional[int] = None,
    problem_prefix: str = DEFAULT_PROBLEM_PREFIX,
    render: bool = False,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Corpus:
    """Cut ``source`` into windows and write each one as a problem folder.

    Args:
        source: The trace to cut. Its steps are pulled lazily.
        corpus_root: Directory to write the corpus into; created if absent.
        cut_mode: How to split the trace. See :func:`~src.trace_generation.cutter.cut`.
        length_range: Inclusive ``(min, max)`` window length, for ``UNIFORM``.
        buckets: Window lengths to cycle, for ``BUCKETS``.
        skip: Steps discarded between consecutive windows.
        num_problems: Cap on the number of problem folders.
        seed: Seed for the cutter's window-length RNG, independent of the walk's.
        problem_prefix: Folder name prefix; folders are numbered from 0.
        render: Copy each window's frames as ``state_*.png``.
        extra_info: Extra keys merged into ``generation_info.json``.

    Returns:
        A :class:`Corpus` describing what was written.

    Raises:
        ValueError: The trace yielded no window, so there is no corpus to write.
    """
    corpus_root = Path(corpus_root)
    cut_mode = CutMode(cut_mode)

    windows = cut(source.steps(), mode=cut_mode, length_range=length_range,
                  buckets=buckets, skip=skip, num_problems=num_problems, seed=seed)
    if not windows:
        raise ValueError(
            f"{source.describe()['source_file']} yielded no window under "
            f"cut_mode={cut_mode.value}; there is no corpus to write. Check the "
            "trace length against length_range/buckets and skip.")
    if num_problems is not None and len(windows) < num_problems:
        logger.warning("Asked for %d problems but the trace only yielded %d.",
                       num_problems, len(windows))

    described = source.describe()
    problems = tuple(
        emit_window(window, problem_name=f"{problem_prefix}{index}", index=index,
                    domain_name=source.domain_name, corpus_root=corpus_root,
                    render=render, source=described["source_file"])
        for index, window in enumerate(windows)
    )

    info = build_generation_info(problems, _manifest_params(
        described, cut_mode, length_range, buckets, skip, num_problems, seed,
        render, extra_info))
    return Corpus(
        root=corpus_root,
        trajectories_dir=corpus_root / POOL_SUBPATH,
        problems=problems,
        info=info,
        info_file=write_generation_info(corpus_root, info),
    )


def _manifest_params(
    described: Dict[str, Any],
    cut_mode: CutMode,
    length_range: Optional[Tuple[int, int]],
    buckets: Optional[Sequence[int]],
    skip: int,
    num_problems: Optional[int],
    seed: Optional[int],
    render: bool,
    extra_info: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge the source's own description with the cut parameters."""
    params: Dict[str, Any] = dict(described)
    params.update({
        "cut_mode": cut_mode.value,
        "length_range": list(length_range) if length_range is not None else None,
        "buckets": list(buckets) if buckets is not None else None,
        "skip": skip,
        "num_problems": num_problems,
        "cut_seed": seed,
        "render": render,
    })
    if extra_info:
        params.update(extra_info)
    return params
