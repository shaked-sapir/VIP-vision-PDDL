"""Rendering as a strategy on a trace source.

A :class:`FrameRenderer` turns one symbolic :class:`TraceState` into an image.
:class:`RenderedSource` wraps any :class:`TraceSource` and yields its steps with
``frame_before`` / ``frame_after`` filled, so the cutter and the emitter see a
rendered trace exactly as they see one read from disk:

    TrajectorySource  + renderer  ->  frames for a pre-made trajectory
    ProblemWalkSource + renderer  ->  frames rendered on the fly

Each distinct frame is rendered once and written to ``frames_dir``; the emitter
copies it into every problem folder that needs it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, FrozenSet, Hashable, Iterator, Optional, Protocol, Tuple

from PIL import Image

from src.trace_generation.sources import TraceSource
from src.trace_generation.trace_step import TraceState, TraceStep

FRAME_NAME_FORMAT = "frame_{index:06d}.png"


class FrameRenderer(Protocol):
    """Render one symbolic state into an image.

    ``step`` is the state's 0-based position in the trace and ``action`` the
    ground action that led to it (``None`` for the initial state). A renderer
    that draws neither returns the same key for every occurrence of a state.
    """

    def render(self, state: TraceState, *, step: int,
               action: Optional[str]) -> Image.Image:
        """Render ``state``; ``step`` and ``action`` are context, not content."""

    def frame_key(self, state: TraceState, *, step: int,
                  action: Optional[str]) -> Hashable:
        """Cache key: two calls with equal keys must render the same image."""

    def describe(self) -> Dict[str, Any]:
        """The renderer's identity and settings, for ``generation_info.json``."""


def state_key(state: TraceState) -> Tuple[FrozenSet[str], Tuple[str, ...]]:
    """The cache key of a renderer whose image depends on the state alone."""
    return state.fluent_set(), state.objects


class RenderedSource(TraceSource):
    """A :class:`TraceSource` whose steps carry rendered frames."""

    def __init__(
        self,
        inner: TraceSource,
        renderer: FrameRenderer,
        *,
        frames_dir: Optional[Path] = None,
    ) -> None:
        """Wrap ``inner`` so every yielded step has both frames.

        Args:
            inner: The symbolic source to render.
            renderer: The rendering strategy.
            frames_dir: Where rendered frames are written. Defaults to a fresh
                temporary directory, which the caller may delete after emission.
        """
        self.inner = inner
        self.renderer = renderer
        self.frames_dir = Path(frames_dir) if frames_dir is not None \
            else Path(tempfile.mkdtemp(prefix="rendered_frames_"))
        self._cache: Dict[Hashable, Path] = {}

    @property
    def domain_name(self) -> str:
        """The inner source's domain name."""
        return self.inner.domain_name

    @property
    def objects(self) -> Tuple[str, ...]:
        """The inner source's object universe."""
        return self.inner.objects

    @property
    def frames_rendered(self) -> int:
        """How many distinct frames have been rendered so far."""
        return len(self._cache)

    def steps(self) -> Iterator[TraceStep]:
        """Yield the inner steps with ``frame_before`` / ``frame_after`` attached.

        Consecutive steps share a state, so the frame rendered as step ``k``'s
        ``frame_after`` is reused as step ``k + 1``'s ``frame_before``.
        """
        self._cache.clear()
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        before: Optional[Path] = None
        for index, step in enumerate(self.inner.steps()):
            if before is None:
                before = self._frame(step.prev_state, step=0, action=None)
            after = self._frame(step.next_state, step=index + 1, action=step.action)
            yield TraceStep(
                prev_state=step.prev_state,
                action=step.action,
                next_state=step.next_state,
                frame_before=before,
                frame_after=after,
            )
            before = after

    def describe(self) -> Dict[str, Any]:
        """The inner description plus the renderer's."""
        return {**self.inner.describe(), "renderer": self.renderer.describe()}

    def _frame(self, state: TraceState, *, step: int, action: Optional[str]) -> Path:
        """The frame for ``state``, rendered on first use of its key."""
        key = self.renderer.frame_key(state, step=step, action=action)
        if key not in self._cache:
            path = self.frames_dir / FRAME_NAME_FORMAT.format(index=len(self._cache))
            self.renderer.render(state, step=step, action=action).save(path)
            self._cache[key] = path
        return self._cache[key]
