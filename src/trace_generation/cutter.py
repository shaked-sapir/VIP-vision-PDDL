"""Cut a trace into non-overlapping windows. Pure: steps in, windows out.

No I/O and no environment, so the same cutter serves every source. Callers that
cut several traces into one corpus thread ``exclude_signatures`` through the
calls to keep deduplication global.

The step stream is pulled lazily and never further than the accepted windows
need, so a source whose steps are expensive to produce -- a rendering walk, a
planner walk -- pays only for the steps that are actually cut.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from itertools import islice
from typing import FrozenSet, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from src.trace_generation.trace_step import TraceStep

WindowSignature = Tuple[FrozenSet[str], FrozenSet[str]]


class CutMode(str, Enum):
    """How a trace is split into windows."""

    NONE = "none"
    UNIFORM = "uniform"
    BUCKETS = "buckets"


@dataclass(frozen=True)
class Window:
    """A contiguous slice of a trace that becomes one problem folder."""

    steps: Tuple[TraceStep, ...]
    signature: WindowSignature

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("A Window must contain at least one step.")

    @property
    def length(self) -> int:
        """Number of steps (== number of actions) in the window."""
        return len(self.steps)

    @property
    def objects(self) -> Tuple[str, ...]:
        """The window's object universe, taken from its initial state."""
        return self.steps[0].prev_state.objects


def window_signature(steps: Sequence[TraceStep]) -> WindowSignature:
    """The (initial fluents, final fluents) pair a window is deduplicated on."""
    return (
        steps[0].prev_state.fluent_set(),
        steps[-1].next_state.fluent_set(),
    )


def cut(
    steps: Iterable[TraceStep],
    *,
    mode: CutMode,
    length_range: Optional[Tuple[int, int]] = None,
    buckets: Optional[Sequence[int]] = None,
    skip: int = 1,
    num_problems: Optional[int] = None,
    seed: Optional[int] = None,
    exclude_signatures: Iterable[WindowSignature] = (),
) -> List[Window]:
    """Split ``steps`` into up to ``num_problems`` deduplicated windows.

    Args:
        steps: The trace, in order. Pulled left to right, lazily, and never
            reordered. ``NONE`` drains it; the other modes stop pulling once
            ``num_problems`` windows are accepted.
        mode: ``NONE`` takes the whole trace as one window. ``UNIFORM`` samples
            each window's length from ``length_range``. ``BUCKETS`` cycles
            ``buckets`` so the resulting pool is length-balanced.
        length_range: Inclusive ``(min, max)`` window length. Required for
            ``UNIFORM``.
        buckets: Window lengths to cycle. Required for ``BUCKETS``.
        skip: Steps discarded between consecutive windows.
        num_problems: Cap on the number of windows. ``None`` means take as many
            as the trace allows.
        seed: Seed for the window-length RNG. Independent of any walk RNG.
        exclude_signatures: Signatures already claimed elsewhere in the corpus.
            Windows matching one are dropped, their steps consumed.

    Returns:
        The accepted windows, in trace order. Shorter than ``num_problems`` when
        the trace runs out; the caller decides whether that is worth reporting.
    """
    _validate(mode, length_range, buckets, skip, num_problems)

    if num_problems is not None and num_problems <= 0:
        return []

    stream = iter(steps)
    seen: Set[WindowSignature] = set(exclude_signatures)

    if mode is CutMode.NONE:
        return _accept_whole_trace(stream, seen)

    windows: List[Window] = []
    for i, length in enumerate(_length_stream(mode, length_range, buckets, seed)):
        if num_problems is not None and len(windows) >= num_problems:
            break
        if i > 0:
            _discard(stream, skip)

        slice_ = tuple(islice(stream, length))
        if len(slice_) < length:
            break  # the trace ran out mid-window

        signature = window_signature(slice_)
        if signature in seen:
            continue
        seen.add(signature)
        windows.append(Window(steps=slice_, signature=signature))

    return windows


def _accept_whole_trace(stream: Iterator[TraceStep],
                        seen: Set[WindowSignature]) -> List[Window]:
    """Drain the trace into a single window, subject to the shared dedup set."""
    slice_ = tuple(stream)
    if not slice_:
        return []
    signature = window_signature(slice_)
    if signature in seen:
        return []
    return [Window(steps=slice_, signature=signature)]


def _discard(stream: Iterator[TraceStep], count: int) -> None:
    """Drop the next ``count`` steps, or fewer if the stream ends first."""
    for _ in islice(stream, count):
        pass


def _length_stream(mode: CutMode, length_range: Optional[Tuple[int, int]],
                   buckets: Optional[Sequence[int]], seed: Optional[int]) -> Iterator[int]:
    """Yield window lengths forever; the caller stops on the step budget."""
    if mode is CutMode.UNIFORM:
        rng = random.Random(seed)
        low, high = length_range  # type: ignore[misc]
        while True:
            yield rng.randint(low, high)
    else:
        while True:
            yield from buckets  # type: ignore[misc]


def _validate(mode: CutMode, length_range: Optional[Tuple[int, int]],
              buckets: Optional[Sequence[int]], skip: int,
              num_problems: Optional[int]) -> None:
    """Reject argument combinations that cannot produce a sane corpus."""
    if not isinstance(mode, CutMode):
        raise TypeError(f"mode must be a CutMode, got {mode!r}.")
    if skip < 0:
        raise ValueError(f"skip must be >= 0, got {skip}.")
    if num_problems is not None and num_problems < 0:
        raise ValueError(f"num_problems must be >= 0 or None, got {num_problems}.")

    if mode is CutMode.UNIFORM:
        if length_range is None:
            raise ValueError("mode=uniform requires length_range=(min, max).")
        low, high = length_range
        if low < 1 or high < low:
            raise ValueError(
                f"length_range must satisfy 1 <= min <= max, got {length_range}.")
    elif mode is CutMode.BUCKETS:
        if not buckets:
            raise ValueError("mode=buckets requires a non-empty buckets list.")
        if any(b < 1 for b in buckets):
            raise ValueError(f"every bucket must be >= 1, got {list(buckets)}.")
