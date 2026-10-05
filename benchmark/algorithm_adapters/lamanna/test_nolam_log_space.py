"""The log-space NOLAM learner against the library's own arithmetic.

    python -m pytest benchmark/algorithm_adapters/lamanna/test_nolam_log_space.py
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
from nolam.algorithm import Configuration
from nolam.algorithm.Learner import Learner

from benchmark.algorithm_adapters.lamanna.domain_dialect import write_typed_domain
from benchmark.algorithm_adapters.lamanna.nolam_log_space import (
    HYPOTHESES,
    NO_EVIDENCE_POSTERIOR,
    LogSpaceLearner,
    TransitionCounts,
    posteriors,
    transition_probabilities,
)

DOMAIN = Path(__file__).resolve().parents[3] / "src" / "domains" / "depot" / "depot.pddl"
CATEGORIES = ("neg-neg", "neg-pos", "pos-neg", "pos-pos")


def _library_posteriors(counts: TransitionCounts, e: float, allow_prec_neg: bool):
    """The library's formula, written out: coefficient times powers, then normalised."""
    from benchmark.algorithm_adapters.lamanna.nolam_log_space import hypothesis_priors

    n = counts.as_tuple()
    coeff = math.factorial(sum(n)) / math.prod(math.factorial(k) for k in n)
    priors = hypothesis_priors(counts, allow_prec_neg)
    table = transition_probabilities(e)
    joint = [
        coeff * math.prod(p ** k for p, k in zip(table[h], n)) * priors[h] for h in HYPOTHESES
    ]
    total = sum(joint)
    return [value / total for value in joint]


def _counts(rng: random.Random, upper: int) -> TransitionCounts:
    return TransitionCounts(*(rng.randint(0, upper) for _ in range(4)))


class TestPosteriorMatchesTheLibraryFormula:
    @pytest.mark.parametrize("e", [0.01, 0.05, 0.1, 0.2, 0.3, 0.4])
    @pytest.mark.parametrize("allow_prec_neg", [False, True])
    def test_on_counts_the_library_can_handle(self, e, allow_prec_neg):
        rng = random.Random(7)
        for _ in range(300):
            counts = _counts(rng, 40)
            if counts.total == 0:
                continue
            ours = posteriors(counts, e, allow_prec_neg)
            theirs = _library_posteriors(counts, e, allow_prec_neg)
            assert ours == pytest.approx(theirs, rel=1e-9, abs=1e-300)

    def test_posteriors_sum_to_one(self):
        assert sum(posteriors(TransitionCounts(30, 5, 4, 60), 0.1, False)) == pytest.approx(1.0)


class TestCountsTheLibraryCannotHandle:
    def test_the_library_formula_overflows(self):
        with pytest.raises(OverflowError):
            _library_posteriors(TransitionCounts(900, 100, 100, 11), 0.1, False)

    def test_log_space_picks_the_hypothesis_that_generated_the_counts(self):
        e, total = 0.1, 200_000
        table = transition_probabilities(e)
        for hypothesis in (("pos", "neg"), ("none", "pos"), ("pos", "none")):
            n = [round(total * p) for p in table[hypothesis]]
            result = posteriors(TransitionCounts(*n), e, True)
            assert HYPOTHESES[result.index(max(result))] == hypothesis

    def test_same_proportions_give_the_same_answer_at_any_scale(self):
        small = TransitionCounts(8, 1, 80, 9)
        large = TransitionCounts(8_000, 1_000, 80_000, 9_000)
        pick = lambda c: max(range(9), key=posteriors(c, 0.1, False).__getitem__)  # noqa: E731
        assert HYPOTHESES[pick(small)] == HYPOTHESES[pick(large)] == ("pos", "neg")


class TestZeroProbabilities:
    def test_a_noise_free_effect_is_found(self):
        result = posteriors(TransitionCounts(0, 0, 50, 0), 0.0, False)
        assert HYPOTHESES[result.index(max(result))] == ("pos", "neg")

    def test_counts_no_hypothesis_allows_fall_back_as_the_library_does(self):
        # With e = 0 no hypothesis gives both an unchanged-true and a deleted atom
        # together with an added one positive probability.
        result = posteriors(TransitionCounts(0, 5, 5, 5), 0.0, False)
        assert result == [NO_EVIDENCE_POSTERIOR] * 9


class _StubCounts:
    """Replaces trace counting with fixed counts, for both learners alike."""

    def __init__(self, seed: int, upper: int) -> None:
        self.seed, self.upper = seed, upper

    def __call__(self, learner, trace_names, action_model) -> Dict:
        rng = random.Random(self.seed)
        for atoms in learner.op_stats.values():
            for atom in sorted(atoms):
                atoms[atom] = dict(zip(CATEGORIES, _counts(rng, self.upper).as_tuple()))
        return learner.op_stats


@pytest.fixture
def typed_domain(tmp_path) -> Path:
    return write_typed_domain(DOMAIN, tmp_path / "domain.pddl")


class TestWholeModelMatchesTheLibrary:
    @pytest.mark.parametrize("allow_prec_neg", [False, True])
    @pytest.mark.parametrize("e", [0.0, 0.05, 0.1, 0.3])
    @pytest.mark.parametrize("seed", range(5))
    def test_same_model_from_the_same_counts(
        self, typed_domain, monkeypatch, allow_prec_neg, e, seed
    ):
        monkeypatch.setattr(Configuration, "ALLOW_PREC_NEG", allow_prec_neg)
        monkeypatch.setattr(Configuration, "SAMPLING", False)
        stub = _StubCounts(seed, upper=60)
        monkeypatch.setattr(Learner, "count_traces", lambda self, t, m: stub(self, t, m))

        np.random.seed(0)
        theirs = str(Learner().learn(str(typed_domain), [], e))
        np.random.seed(0)
        ours = str(LogSpaceLearner().learn(str(typed_domain), [], e))

        assert ours == theirs

    def test_large_counts_crash_the_library_and_not_the_log_space_learner(
        self, typed_domain, monkeypatch
    ):
        monkeypatch.setattr(Configuration, "ALLOW_PREC_NEG", False)
        monkeypatch.setattr(Configuration, "SAMPLING", False)
        stub = _StubCounts(seed=1, upper=2_000)
        monkeypatch.setattr(Learner, "count_traces", lambda self, t, m: stub(self, t, m))

        with pytest.raises(OverflowError):
            Learner().learn(str(typed_domain), [], 0.1)
        model = LogSpaceLearner().learn(str(typed_domain), [], 0.1)
        assert model.operators

    def test_sampling_is_refused(self, typed_domain, monkeypatch):
        monkeypatch.setattr(Configuration, "SAMPLING", True)
        with pytest.raises(NotImplementedError):
            LogSpaceLearner().learn(str(typed_domain), [], 0.1)
