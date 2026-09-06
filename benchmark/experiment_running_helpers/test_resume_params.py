"""``run_params_conflicts`` treats ``baseline_params`` as optional on the saved side."""

from benchmark.experiment_running_helpers.resume import run_params_conflicts

BASE = {"n_folds": 5, "search_mode": "dfs", "timestamp": "x"}


def test_a_saved_run_without_baseline_params_can_be_resumed():
    current = {**BASE, "baseline_params": {"ROSAME_24": {"batch_size": 0}}}
    assert run_params_conflicts(BASE, current) == []


def test_differing_baseline_params_block_the_resume():
    existing = {**BASE, "baseline_params": {"ROSAME_24": {"batch_size": 128}}}
    current = {**BASE, "baseline_params": {"ROSAME_24": {"batch_size": 0}}}
    assert run_params_conflicts(existing, current) == ["baseline_params"]


def test_other_keys_stay_strict():
    assert run_params_conflicts(BASE, {**BASE, "search_mode": "ucs"}) == ["search_mode"]
