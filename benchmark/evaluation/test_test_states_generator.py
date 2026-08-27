"""Tests for the S_test budget and the plan-bound abandon rule.

    python -m pytest benchmark/evaluation/test_test_states_generator.py
"""

import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmark.evaluation import test_states_generator as gen


class TestPerFoldBudget(unittest.TestCase):
    """The walk budget belongs to the fold, not to each problem."""

    def _run(self, n_problems: int, **kwargs) -> int:
        """Generate over ``n_problems`` fakes; return the per-problem quota used."""
        seen = {}

        def fake_collect(domain_str, problem_path, rng, num_trajectories, *a, **kw):
            seen["quota"] = num_trajectories
            return [["(p)"]]

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(gen, "_collect_problem_states", fake_collect):
            gen.generate_predictive_power_test_states(
                domain_ref_path=Path("d.pddl"),
                test_problem_paths=[f"p{i}.pddl" for i in range(n_problems)],
                output_dir=Path(tmp), **kwargs,
            )
        return seen["quota"]

    def test_budget_is_split_across_the_folds_problems(self):
        self.assertEqual(self._run(500), math.ceil(gen._DEFAULT_TRAJECTORIES_PER_FOLD / 500))
        self.assertEqual(self._run(10), math.ceil(gen._DEFAULT_TRAJECTORIES_PER_FOLD / 10))

    def test_every_problem_gets_at_least_one_walk(self):
        """A test split larger than the budget must still cover every problem."""
        self.assertEqual(self._run(5000), 1)

    def test_an_explicit_rate_overrides_the_budget(self):
        self.assertEqual(self._run(500, num_trajectories_per_problem=7), 7)


class TestAbandonPlanBound(unittest.TestCase):
    """A problem whose rejected walk repeats its length cannot be rescued."""

    def _collect(self, lengths):
        """Run _collect_problem_states against a walker yielding ``lengths``."""
        calls = {"n": 0}

        def fake_walk(*a, **kw):
            i = calls["n"]
            calls["n"] += 1
            length = lengths[min(i, len(lengths) - 1)]
            return [[f"(s{i}_{k})"] for k in range(length)]

        with mock.patch.object(gen, "_trajectory_states", fake_walk), \
                mock.patch.object(gen, "ground_actions_of", lambda *a: {}), \
                mock.patch.object(gen, "parse_problem", lambda *a: object()), \
                mock.patch.object(gen, "ground_fluents", lambda p: []), \
                mock.patch.object(gen, "_upstate_to_literals", lambda s, f: s):
            gen._collect_problem_states(
                "d.pddl", "p.pddl", random.Random(0),
                num_trajectories=34, p_rnd=0.2, traj_len_min=5,
                traj_len_max=30, max_planning_time=1,
            )
        return calls["n"]

    def test_a_repeating_short_walk_is_abandoned(self):
        """Three identical 2-state walks, not 34*3 of them."""
        attempts = self._collect([2])
        self.assertLessEqual(attempts, gen._ABANDON_AFTER_REPEATS + 1)

    def test_varying_short_walks_still_retry(self):
        """Different lengths mean the randomness is doing something; keep going."""
        self.assertGreater(self._collect([2, 3, 4, 2, 3, 4]), gen._ABANDON_AFTER_REPEATS + 1)

    def test_a_long_walk_is_accepted_without_retrying(self):
        self.assertEqual(self._collect([9]), 34)
