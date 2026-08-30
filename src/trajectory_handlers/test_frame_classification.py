"""Both frame-classification strategies must agree on order and on failure.

    python -m pytest src/trajectory_handlers/test_frame_classification.py

Every test runs against both, so the concurrent path cannot quietly drift from
the sequential one it replaces.
"""

import threading
import time
from pathlib import Path

import pytest

from src.trajectory_handlers.frame_classification import (
    ConcurrentFrameClassifier,
    SequentialFrameClassifier,
)

STRATEGIES = [
    pytest.param(SequentialFrameClassifier(), id="sequential"),
    pytest.param(ConcurrentFrameClassifier(max_workers=12), id="concurrent"),
]


class _Classifier:
    """Returns the frame's own name, optionally after a per-frame delay."""

    def __init__(self, delays=None, fail_on=frozenset()):
        self.delays = delays or {}
        self.fail_on = fail_on
        self.calls = []
        self._lock = threading.Lock()

    def classify(self, path):
        name = Path(path).stem
        time.sleep(self.delays.get(name, 0))
        with self._lock:
            self.calls.append(name)
        if name in self.fail_on:
            raise RuntimeError(f"boom on {name}")
        return {"frame": name}


def _paths(n):
    return [Path(f"state_{i:02d}.png") for i in range(n)]


@pytest.mark.parametrize("strategy", STRATEGIES)
class TestOrder:
    def test_results_follow_frame_order(self, strategy):
        paths = _paths(6)
        out = strategy.classify_frames(_Classifier(), paths)
        assert [r["frame"] for r in out] == [p.stem for p in paths]

    def test_order_holds_when_later_frames_finish_first(self, strategy):
        """The regression that would otherwise be invisible.

        Frame 0 is slow and frame 5 is fast, so completion order is the reverse
        of frame order. The trajectory pairs state i with state i+1, so a result
        list in completion order would silently mislabel every transition.
        """
        paths = _paths(6)
        delays = {"state_00": 0.25, "state_01": 0.20, "state_05": 0.0}
        out = strategy.classify_frames(_Classifier(delays), paths)
        assert [r["frame"] for r in out] == [p.stem for p in paths]

    def test_empty_window(self, strategy):
        assert strategy.classify_frames(_Classifier(), []) == []


class TestConcurrencyActuallyOverlaps:
    def test_concurrent_is_faster_than_the_sum_of_its_parts(self):
        paths = _paths(6)
        delays = {p.stem: 0.15 for p in paths}
        started = time.perf_counter()
        ConcurrentFrameClassifier(max_workers=6).classify_frames(
            _Classifier(delays), paths)
        elapsed = time.perf_counter() - started
        assert elapsed < 6 * 0.15 * 0.7, "frames did not overlap"


class TestFailure:
    """A window needs every frame, so a partial result is not a result."""

    def test_concurrent_abandons_the_window(self):
        strategy = ConcurrentFrameClassifier(max_workers=4, max_attempts=2)
        with pytest.raises(RuntimeError, match="cannot be repaired"):
            strategy.classify_frames(_Classifier(fail_on={"state_03"}), _paths(6))

    def test_it_retries_before_giving_up(self):
        classifier = _Classifier(fail_on={"state_01"})
        strategy = ConcurrentFrameClassifier(max_workers=4, max_attempts=3)
        with pytest.raises(RuntimeError):
            strategy.classify_frames(classifier, _paths(3))
        assert classifier.calls.count("state_01") == 3

    def test_a_transient_failure_does_not_fail_the_window(self):
        class Flaky(_Classifier):
            def classify(self, path):
                name = Path(path).stem
                with self._lock:
                    self.calls.append(name)
                if name == "state_01" and self.calls.count("state_01") == 1:
                    raise RuntimeError("transient")
                return {"frame": name}

        out = ConcurrentFrameClassifier(max_workers=4, max_attempts=3)\
            .classify_frames(Flaky(), _paths(3))
        assert [r["frame"] for r in out] == ["state_00", "state_01", "state_02"]


class TestConstruction:
    @pytest.mark.parametrize("kwargs", [{"max_workers": 0}, {"max_attempts": 0}])
    def test_rejects_nonsense(self, kwargs):
        with pytest.raises(ValueError):
            ConcurrentFrameClassifier(**kwargs)
