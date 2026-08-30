"""Strategies for turning a window's frames into per-frame predicate dicts.

The expensive part of building an inferred trajectory is one VLM call per frame,
and those calls are independent: ``classify(frame_i)`` reads only frame *i*. The
assembly that follows looks sequential because frame *i+1*'s predicates serve as
both "next state of step *i*" and "current state of step *i+1*" -- but that is a
shared reference, not a computed dependency.

So the phase is separable, and this module is the seam. Both strategies return
one dict per frame **in frame order**; the caller indexes into that list rather
than consuming results as they arrive, so completion order can never reach the
trajectory.
"""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Protocol

DEFAULT_MAX_WORKERS = 12
#: Attempts per frame before the window is abandoned.
DEFAULT_MAX_ATTEMPTS = 3


def _announce(n_frames: int, classifier, concurrent: bool) -> None:
    """One line per window rather than one per frame.

    A window is 11 frames and a corpus is 2500 windows, so a per-frame line is
    30,000 lines of noise per domain -- and unordered ones once the calls
    overlap. The two warnings inside ``classify`` stay put: they are exceptional
    and they name their image.
    """
    mode = "concurrently" if concurrent else "one at a time"
    temperature = getattr(classifier, "temperature", "?")
    print(f"  Classifying {n_frames} frame(s) {mode} "
          f"(temperature={temperature})")


class FrameClassifier(Protocol):
    """Turns a window's frames into one predicate dict per frame, in order."""

    def classify_frames(self, classifier, image_paths: List[Path]) -> List[Dict]:
        ...


class SequentialFrameClassifier:
    """One frame at a time — the original behaviour, and the default."""

    def classify_frames(self, classifier, image_paths: List[Path]) -> List[Dict]:
        """Classify each frame in turn.

        Args:
            classifier: A :class:`FluentClassifier` with a ``classify`` method.
            image_paths: The window's frames, in frame order.

        Returns:
            One predicate dict per frame, positionally aligned with the input.
        """
        _announce(len(image_paths), classifier, concurrent=False)
        return [classifier.classify(path) for path in image_paths]


class _SharedBackoff:
    """Backoff shared by every worker in one window.

    Per-call backoff is wrong under concurrency: a rate-limit response reaches
    all workers at once, and they would retry in lockstep and reproduce it. This
    makes them wait on one clock, with jitter so they do not resume together.
    """

    def __init__(self, base_seconds: float = 2.0) -> None:
        self._base = base_seconds
        self._lock = threading.Lock()
        self._retry_after = 0.0

    def wait(self) -> None:
        """Block until the shared cooldown has elapsed."""
        with self._lock:
            remaining = self._retry_after - time.monotonic()
        if remaining > 0:
            time.sleep(remaining + random.uniform(0, 0.5))

    def penalise(self, attempt: int) -> None:
        """Extend the cooldown after a failure, exponentially in ``attempt``."""
        delay = self._base * (2 ** attempt)
        with self._lock:
            self._retry_after = max(self._retry_after, time.monotonic() + delay)


class ConcurrentFrameClassifier:
    """All frames of a window at once, all-or-nothing.

    A window needs every frame: a trajectory with a missing state cannot be
    repaired, so one exhausted frame fails the whole window rather than
    yielding a partial result the caller would have to detect.
    """

    def __init__(
        self,
        max_workers: int = DEFAULT_MAX_WORKERS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        self.max_workers = max_workers
        self.max_attempts = max_attempts

    def classify_frames(self, classifier, image_paths: List[Path]) -> List[Dict]:
        """Classify every frame concurrently, preserving frame order.

        Args:
            classifier: A :class:`FluentClassifier` with a ``classify`` method.
                Must be safe to call from several threads; the LLM classifiers
                assign only in ``__init__``.
            image_paths: The window's frames, in frame order.

        Returns:
            One predicate dict per frame, positionally aligned with the input.

        Raises:
            RuntimeError: If any frame failed every attempt. The window is
                abandoned rather than returned partially classified.
        """
        if not image_paths:
            return []

        _announce(len(image_paths), classifier, concurrent=True)
        backoff = _SharedBackoff()
        results: Dict[int, Dict] = {}
        failures: Dict[int, Exception] = {}

        def run(index: int) -> None:
            for attempt in range(self.max_attempts):
                backoff.wait()
                try:
                    results[index] = classifier.classify(image_paths[index])
                    return
                except Exception as error:  # noqa: BLE001 - recorded, then re-raised
                    failures[index] = error
                    backoff.penalise(attempt)
            # Left in `failures`; the caller below turns that into one error.

        workers = min(self.max_workers, len(image_paths))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, range(len(image_paths))))

        missing = [i for i in range(len(image_paths)) if i not in results]
        if missing:
            first = failures.get(missing[0])
            raise RuntimeError(
                f"{len(missing)} of {len(image_paths)} frames failed after "
                f"{self.max_attempts} attempts (first: {image_paths[missing[0]]}"
                f" -> {type(first).__name__}: {first}). Abandoning the window: a "
                f"trajectory missing a state cannot be repaired."
            )

        # Indexed, not appended: completion order must not reach the trajectory.
        return [results[i] for i in range(len(image_paths))]
