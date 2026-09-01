"""Tests for transition batching in the ROSAME training loops.

    python -m pytest benchmark/algorithm_adapters/test_batched_steps.py
"""

import random

import torch

from benchmark.algorithm_adapters.po_rosame_runner import (
    DEFAULT_BATCH_SIZE,
    batched_steps,
)


def _trace(problem, n_transitions, width):
    return (problem,
            torch.rand(n_transitions, width),
            torch.rand(n_transitions, 3),
            torch.rand(n_transitions, width))


class TestBatching:
    def test_pools_transitions_across_traces(self):
        """10 traces of 15 transitions is 150 rows -> 2 batches at 128, not 10 steps."""
        cached = [_trace("p", 15, 20) for _ in range(10)]
        batches = list(batched_steps(cached, 128, random.Random(0)))
        assert len(batches) == 2
        assert [b[1].shape[0] for b in batches] == [128, 22]

    def test_every_transition_is_used_exactly_once(self):
        cached = [_trace("p", 15, 20) for _ in range(10)]
        total = sum(b[1].shape[0] for b in batched_steps(cached, 128, random.Random(0)))
        assert total == 150

    def test_default_is_the_upstream_value(self):
        assert DEFAULT_BATCH_SIZE == 128


class TestGroundingIsRespected:
    """build() emits a row per proposition of the grounded problem, so traces of
    different widths cannot share a step -- pooling them would trip the shape
    assert in _train_step."""

    def test_widths_are_never_mixed_in_one_batch(self):
        cached = ([_trace("small", 15, 20) for _ in range(4)]
                  + [_trace("big", 15, 50) for _ in range(4)])
        for _problem, s1, _a, s2 in batched_steps(cached, 128, random.Random(0)):
            assert s1.shape[1] == s2.shape[1]

    def test_each_group_is_batched_separately(self):
        cached = ([_trace("small", 15, 20) for _ in range(4)]
                  + [_trace("big", 15, 50) for _ in range(4)])
        widths = {b[1].shape[1] for b in batched_steps(cached, 128, random.Random(0))}
        assert widths == {20, 50}

    def test_the_problem_matches_its_batch(self):
        """The caller re-grounds to this problem, so it must own the batch."""
        cached = ([_trace("small", 15, 20) for _ in range(2)]
                  + [_trace("big", 15, 50) for _ in range(2)])
        for problem, s1, _a, _s2 in batched_steps(cached, 128, random.Random(0)):
            assert problem == ("small" if s1.shape[1] == 20 else "big")


class TestOptOut:
    def test_none_restores_one_step_per_trace(self):
        cached = [_trace("p", 15, 20) for _ in range(10)]
        assert len(list(batched_steps(cached, None, random.Random(0)))) == 10

    def test_zero_restores_one_step_per_trace(self):
        cached = [_trace("p", 15, 20) for _ in range(10)]
        assert len(list(batched_steps(cached, 0, random.Random(0)))) == 10

    def test_a_batch_larger_than_the_data_is_one_step(self):
        cached = [_trace("p", 15, 20) for _ in range(2)]
        batches = list(batched_steps(cached, 128, random.Random(0)))
        assert len(batches) == 1 and batches[0][1].shape[0] == 30
