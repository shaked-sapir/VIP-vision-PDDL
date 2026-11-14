"""The subset picker, the resume marker and the usage summary of infer_image_subset.

    python -m pytest benchmark/test_infer_image_subset.py
"""

from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

from benchmark.experiment_running_helpers.run_fold import cv_split, select_training_pool
from benchmark.infer_image_subset import (
    append_usage,
    backend_usage,
    fold_training_subset,
    is_inferred,
    listed_problems,
    read_usage,
    summarise,
)

N_FOLDS = 5
N_TRAIN = 3


@pytest.fixture
def pool(tmp_path: Path) -> List[Path]:
    dirs = [tmp_path / f"problem{i}" for i in range(50)]
    for d in dirs:
        d.mkdir()
    return sorted(dirs)


def _fold_picks(pool: List[Path], fold: int) -> List[Path]:
    train_idx, _ = cv_split(len(pool), fold, n_folds=N_FOLDS, scheme="kfold")
    return select_training_pool([pool[i] for i in train_idx], fold, N_TRAIN)[:N_TRAIN]


def test_subset_holds_every_folds_first_n(pool: List[Path]) -> None:
    subset = fold_training_subset(pool, N_FOLDS, "kfold", N_TRAIN)

    for fold in range(N_FOLDS):
        assert set(_fold_picks(pool, fold)) <= set(subset)
    assert len(subset) == len(set(subset))
    assert set(subset) == {p for fold in range(N_FOLDS) for p in _fold_picks(pool, fold)}


def test_no_pick_is_in_its_own_folds_test_set(pool: List[Path]) -> None:
    for fold in range(N_FOLDS):
        _, test_idx = cv_split(len(pool), fold, n_folds=N_FOLDS, scheme="kfold")
        assert not set(_fold_picks(pool, fold)) & {pool[i] for i in test_idx}


def test_subset_rejects_a_size_larger_than_the_training_pool(pool: List[Path]) -> None:
    with pytest.raises(ValueError, match="exceeds fold 0"):
        fold_training_subset(pool, N_FOLDS, "kfold", 41)


def test_a_folder_is_inferred_only_with_both_files(tmp_path: Path) -> None:
    folder = tmp_path / "problem7"
    folder.mkdir()
    assert not is_inferred(folder)
    (folder / "problem7.trajectory").write_text("")
    assert not is_inferred(folder)
    (folder / "problem7.masking_info").write_text("")
    assert is_inferred(folder)


def test_listed_problems_keeps_order_and_rejects_unknown_names(pool: List[Path]) -> None:
    root = pool[0].parent
    assert listed_problems(["problem9", "", " problem2 "], root) == [root / "problem9", root / "problem2"]
    with pytest.raises(FileNotFoundError, match="problem999"):
        listed_problems(["problem999"], root)


def test_backend_usage_sums_the_backends_that_report() -> None:
    def backend(calls: int):
        return SimpleNamespace(usage=lambda: {
            "calls": calls, "prompt_tokens": 10 * calls, "cached_tokens": calls,
            "completion_tokens": 2 * calls, "request_seconds": 0.5 * calls})
    handler = SimpleNamespace(
        object_detector=SimpleNamespace(backend=SimpleNamespace()),
        fluent_classifier=SimpleNamespace(backend=backend(4)),
    )
    assert backend_usage(handler) == {
        "calls": 4, "prompt_tokens": 40, "cached_tokens": 4,
        "completion_tokens": 8, "request_seconds": 2.0}


ROWS = [
    {"problem": "a", "frames": 10, "seconds": 5.0, "calls": 10, "prompt_tokens": 15000,
     "cached_tokens": 5000, "completion_tokens": 3000, "request_seconds": 40.0},
    {"problem": "b", "frames": 0, "seconds": 1.0, "error": "RuntimeError: x"},
]


def test_summary_counts_successful_rows_only() -> None:
    summary = summarise(ROWS)

    assert summary["problems"] == 1 and summary["failed_rows"] == 1
    assert summary["frames"] == 10 and summary["calls"] == 10
    assert summary["request_seconds_per_call"] == 4.0
    assert summary["cache_hit_ratio"] == round(5000 / 15000, 3)
    assert "cost_usd" not in summary


def test_summary_prices_cached_and_uncached_input_apart() -> None:
    summary = summarise(ROWS, price_input=2.0, price_cached=0.2, price_output=10.0)

    assert summary["cost_usd"] == round((10000 * 2.0 + 5000 * 0.2 + 3000 * 10.0) / 1e6, 2)
    assert summary["cost_usd_per_frame"] == round((10000 * 2.0 + 5000 * 0.2 + 3000 * 10.0) / 1e6 / 10, 5)


def test_usage_log_round_trips(tmp_path: Path) -> None:
    log = tmp_path / "usage.jsonl"
    assert read_usage(log) == []
    for row in ROWS:
        append_usage(log, row)
    assert read_usage(log) == ROWS
