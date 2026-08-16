"""Unit tests for the pure window cutter.

    ./venv11/bin/python -m pytest src/trace_generation/test_cutter.py -q
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

import pytest

from src.trace_generation.cutter import (
    CutMode,
    Window,
    cut,
    window_signature,
)
from src.trace_generation.trace_step import TraceState, TraceStep

OBJECTS = ("a:block", "b:block")


def _state(marker: int) -> TraceState:
    """A state whose fluent set is unique per marker."""
    return TraceState(literals=(f"at(a:block,p{marker}:block)",), objects=OBJECTS)


def _trace(n: int) -> List[TraceStep]:
    """A trace of ``n`` steps whose every state is distinct."""
    return [
        TraceStep(prev_state=_state(i), action=f"move(a:block,p{i}:block)",
                  next_state=_state(i + 1))
        for i in range(n)
    ]


# ── mode=none ────────────────────────────────────────────────────────────

def test_none_takes_the_whole_trace_as_one_window():
    steps = _trace(7)
    windows = cut(steps, mode=CutMode.NONE)
    assert len(windows) == 1
    assert windows[0].length == 7
    assert list(windows[0].steps) == steps


def test_none_ignores_skip_and_num_problems():
    windows = cut(_trace(5), mode=CutMode.NONE, skip=3, num_problems=99)
    assert len(windows) == 1
    assert windows[0].length == 5


def test_none_respects_the_shared_dedup_set():
    steps = _trace(5)
    first = cut(steps, mode=CutMode.NONE)
    again = cut(steps, mode=CutMode.NONE,
                exclude_signatures=[w.signature for w in first])
    assert again == []


# ── mode=uniform ─────────────────────────────────────────────────────────

def test_uniform_windows_are_contiguous_and_skip_separated():
    steps = _trace(20)
    windows = cut(steps, mode=CutMode.UNIFORM, length_range=(3, 3),
                  skip=2, num_problems=3, seed=1)
    assert [w.length for w in windows] == [3, 3, 3]
    assert list(windows[0].steps) == steps[0:3]
    assert list(windows[1].steps) == steps[5:8]
    assert list(windows[2].steps) == steps[10:13]


def test_uniform_respects_num_problems():
    windows = cut(_trace(100), mode=CutMode.UNIFORM, length_range=(2, 4),
                  skip=1, num_problems=4, seed=7)
    assert len(windows) == 4


def test_uniform_stops_when_the_trace_runs_dry():
    windows = cut(_trace(10), mode=CutMode.UNIFORM, length_range=(4, 4),
                  skip=1, num_problems=10, seed=3)
    # windows at [0:4] and [5:9]; the third would need steps 10..13.
    assert [w.length for w in windows] == [4, 4]


def test_uniform_lengths_are_reproducible_under_a_seed():
    kwargs = dict(mode=CutMode.UNIFORM, length_range=(2, 9), skip=1,
                  num_problems=5, seed=42)
    first = cut(_trace(200), **kwargs)
    second = cut(_trace(200), **kwargs)
    assert [w.length for w in first] == [w.length for w in second]


def test_uniform_lengths_differ_across_seeds():
    kwargs = dict(mode=CutMode.UNIFORM, length_range=(2, 20), skip=1,
                  num_problems=8)
    a = [w.length for w in cut(_trace(500), seed=1, **kwargs)]
    b = [w.length for w in cut(_trace(500), seed=2, **kwargs)]
    assert a != b


# ── mode=buckets ─────────────────────────────────────────────────────────

def test_buckets_cycle_in_order():
    windows = cut(_trace(200), mode=CutMode.BUCKETS, buckets=[2, 5, 9],
                  skip=1, num_problems=7)
    assert [w.length for w in windows] == [2, 5, 9, 2, 5, 9, 2]


def test_buckets_emit_into_one_pool_not_per_length_groups():
    windows = cut(_trace(200), mode=CutMode.BUCKETS, buckets=[3, 8],
                  skip=0, num_problems=4)
    lengths = [w.length for w in windows]
    assert lengths == [3, 8, 3, 8], "windows must stay in trace order"


def test_buckets_stop_when_the_next_bucket_does_not_fit():
    # 3 fits at [0:3]; the next bucket (40) does not fit in a 10-step trace.
    windows = cut(_trace(10), mode=CutMode.BUCKETS, buckets=[3, 40],
                  skip=0, num_problems=5)
    assert [w.length for w in windows] == [3]


# ── deduplication ────────────────────────────────────────────────────────

def _repeating_trace(cycle: int, total: int) -> List[TraceStep]:
    """A trace that revisits the same states every ``cycle`` steps."""
    return [
        TraceStep(prev_state=_state(i % cycle), action=f"move(a:block,p{i}:block)",
                  next_state=_state((i + 1) % cycle))
        for i in range(total)
    ]


def test_duplicate_windows_are_dropped_but_their_steps_are_consumed():
    steps = _repeating_trace(cycle=4, total=40)
    windows = cut(steps, mode=CutMode.UNIFORM, length_range=(4, 4),
                  skip=0, num_problems=5, seed=1)
    # Every 4-step window on a 4-cycle has the same (init, final) signature.
    assert len(windows) == 1
    assert list(windows[0].steps) == steps[0:4]


def test_exclude_signatures_makes_dedup_global_across_traces():
    steps = _trace(6)  # exactly two 3-step windows fit
    first = cut(steps, mode=CutMode.UNIFORM, length_range=(3, 3), skip=0,
                num_problems=2, seed=1)
    assert len(first) == 2

    second = cut(steps, mode=CutMode.UNIFORM, length_range=(3, 3), skip=0,
                 num_problems=2, seed=1,
                 exclude_signatures=[w.signature for w in first])
    assert second == [], "identical windows from another trace must be dropped"


def test_excluded_window_is_skipped_and_the_next_fresh_one_is_taken():
    steps = _trace(12)
    first = cut(steps, mode=CutMode.UNIFORM, length_range=(3, 3), skip=0,
                num_problems=1, seed=1)

    second = cut(steps, mode=CutMode.UNIFORM, length_range=(3, 3), skip=0,
                 num_problems=1, seed=1,
                 exclude_signatures=[w.signature for w in first])
    assert len(second) == 1
    assert list(second[0].steps) == steps[3:6], "must advance past the duplicate"


def test_exclude_signatures_is_not_mutated():
    steps = _trace(12)
    excluded: list = []
    cut(steps, mode=CutMode.UNIFORM, length_range=(3, 3), skip=0,
        num_problems=2, seed=1, exclude_signatures=excluded)
    assert excluded == []


def test_signature_is_init_and_final_fluents_only():
    steps = _trace(5)
    sig = window_signature(steps)
    assert sig == (steps[0].prev_state.fluent_set(),
                   steps[-1].next_state.fluent_set())


def test_signature_ignores_literal_ordering():
    a = TraceState(literals=("p(x:t)", "q(y:t)"), objects=OBJECTS)
    b = TraceState(literals=("q(y:t)", "p(x:t)"), objects=OBJECTS)
    step_a = TraceStep(prev_state=a, action="m()", next_state=a)
    step_b = TraceStep(prev_state=b, action="m()", next_state=b)
    assert window_signature([step_a]) == window_signature([step_b])


# ── window accessors ─────────────────────────────────────────────────────

def test_window_exposes_length_and_object_universe():
    windows = cut(_trace(9), mode=CutMode.UNIFORM, length_range=(4, 4),
                  skip=0, num_problems=1, seed=1)
    assert windows[0].length == 4
    assert windows[0].objects == OBJECTS


def test_empty_window_is_rejected():
    with pytest.raises(ValueError, match="at least one step"):
        Window(steps=(), signature=(frozenset(), frozenset()))


# ── degenerate inputs ────────────────────────────────────────────────────

def test_empty_trace_yields_no_windows():
    for mode, kwargs in [
        (CutMode.NONE, {}),
        (CutMode.UNIFORM, {"length_range": (1, 2)}),
        (CutMode.BUCKETS, {"buckets": [2]}),
    ]:
        assert cut([], mode=mode, **kwargs) == []


def test_zero_num_problems_yields_no_windows():
    assert cut(_trace(10), mode=CutMode.NONE, num_problems=0) == []


def test_skip_zero_produces_back_to_back_windows():
    steps = _trace(9)
    windows = cut(steps, mode=CutMode.UNIFORM, length_range=(3, 3),
                  skip=0, num_problems=3, seed=1)
    assert list(windows[0].steps) == steps[0:3]
    assert list(windows[1].steps) == steps[3:6]
    assert list(windows[2].steps) == steps[6:9]


def test_num_problems_none_takes_everything_the_trace_allows():
    windows = cut(_trace(10), mode=CutMode.UNIFORM, length_range=(2, 2),
                  skip=0, num_problems=None, seed=1)
    assert len(windows) == 5


# ── laziness ─────────────────────────────────────────────────────────────

def _counting_stream(n: int) -> Tuple[Iterator, List[int]]:
    """A step generator plus a one-element list recording how many were pulled."""
    pulled = [0]

    def gen():
        for step in _trace(n):
            pulled[0] += 1
            yield step

    return gen(), pulled


def test_cut_accepts_a_generator_not_only_a_list():
    stream, _ = _counting_stream(9)
    windows = cut(stream, mode=CutMode.UNIFORM, length_range=(3, 3),
                  skip=0, num_problems=3, seed=1)
    assert [w.length for w in windows] == [3, 3, 3]


def test_cut_stops_pulling_once_num_problems_is_reached():
    stream, pulled = _counting_stream(1000)
    cut(stream, mode=CutMode.UNIFORM, length_range=(3, 3), skip=1,
        num_problems=2, seed=1)
    # two 3-step windows with one skipped step between them, and nothing beyond.
    assert pulled[0] == 7


def test_cut_pulls_no_skip_gap_after_the_last_window():
    stream, pulled = _counting_stream(1000)
    cut(stream, mode=CutMode.UNIFORM, length_range=(2, 2), skip=50,
        num_problems=1, seed=1)
    assert pulled[0] == 2


def test_dropped_duplicates_still_consume_their_skip_gap():
    stream, pulled = _counting_stream(1000)
    windows = cut(stream, mode=CutMode.UNIFORM, length_range=(3, 3), skip=2,
                  num_problems=1, seed=1,
                  exclude_signatures=[window_signature(_trace(1000)[0:3])])
    assert len(windows) == 1
    assert pulled[0] == 8, "3 dropped + 2 skipped + 3 accepted"


# ── validation ───────────────────────────────────────────────────────────

def test_uniform_without_length_range_raises():
    with pytest.raises(ValueError, match="requires length_range"):
        cut(_trace(5), mode=CutMode.UNIFORM)


def test_buckets_without_buckets_raises():
    with pytest.raises(ValueError, match="non-empty buckets"):
        cut(_trace(5), mode=CutMode.BUCKETS)


def test_inverted_length_range_raises():
    with pytest.raises(ValueError, match="1 <= min <= max"):
        cut(_trace(5), mode=CutMode.UNIFORM, length_range=(7, 3))


def test_zero_length_bucket_raises():
    with pytest.raises(ValueError, match="every bucket must be >= 1"):
        cut(_trace(5), mode=CutMode.BUCKETS, buckets=[3, 0])


def test_negative_skip_raises():
    with pytest.raises(ValueError, match="skip must be >= 0"):
        cut(_trace(5), mode=CutMode.NONE, skip=-1)


def test_string_mode_raises():
    with pytest.raises(TypeError, match="must be a CutMode"):
        cut(_trace(5), mode="uniform", length_range=(1, 2))
