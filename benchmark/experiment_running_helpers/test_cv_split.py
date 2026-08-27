"""Tests for :func:`benchmark.experiment_running_helpers.run_fold.cv_split`.

    python -m pytest benchmark/experiment_running_helpers/test_cv_split.py
"""

import random

import pytest

from benchmark.experiment_running_helpers.run_fold import cv_split

N = 2500
K = 5


class TestKFold:
    """The disjointness properties `montecarlo` does not have."""

    def test_test_sets_partition_the_pool(self):
        tests = [set(cv_split(N, f, n_folds=K, scheme="kfold")[1]) for f in range(K)]
        assert set().union(*tests) == set(range(N))
        for i in range(K):
            for j in range(i + 1, K):
                assert not tests[i] & tests[j]

    def test_train_and_test_are_complementary(self):
        for fold in range(K):
            train, test = cv_split(N, fold, n_folds=K, scheme="kfold")
            assert not set(train) & set(test)
            assert len(train) + len(test) == N

    def test_rejects_out_of_range_fold(self):
        with pytest.raises(ValueError, match="out of range"):
            cv_split(N, K, n_folds=K, scheme="kfold")


class TestMonteCarlo:
    """The default must stay bit-identical: results on disk depend on it."""

    def test_matches_the_original_inline_logic(self):
        for fold in range(K):
            indices = list(range(N))
            random.seed(42 + fold)
            random.shuffle(indices)
            n_train = max(1, min(int(0.8 * N), N - 1))
            assert cv_split(N, fold, n_folds=K, scheme="montecarlo") == (
                indices[:n_train], indices[n_train:]
            )

    def test_test_sets_overlap(self):
        """The property that makes it Monte-Carlo rather than k-fold."""
        tests = [set(cv_split(N, f, n_folds=K, scheme="montecarlo")[1]) for f in range(K)]
        assert any(tests[i] & tests[j] for i in range(K) for j in range(i + 1, K))


class TestTrainingPoolGuard:
    """A training size larger than the pool must fail, not silently shrink."""

    def test_the_pool_size_follows_the_corpus(self):
        """What the guard compares against, for the sizes we run."""
        for corpus, expected_train in ((1000, 800), (2500, 2000), (743, 594)):
            train, _ = cv_split(corpus, 0, n_folds=5, scheme="kfold")
            assert len(train) == expected_train


def test_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="unknown cv_scheme"):
        cv_split(N, 0, n_folds=K, scheme="stratified")
