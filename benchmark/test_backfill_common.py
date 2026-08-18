"""Tests for backfill_common's problem-source resolution.

The rule under test: an experiment's cells speak whichever predicate dialect
``normalize_experiment_data`` left them in, and ``run_params.json``'s
``normalized`` field is the record of which that was.
"""

import json
from pathlib import Path

import pytest

from benchmark.backfill_common import (
    find_problem_pddl,
    resolve_data_dir,
    resolve_problem_dir,
)


def _make_experiment(tmp_path: Path, run_params: dict) -> Path:
    exp_dir = tmp_path / "running_results" / "gripper" / "TO=600__exp"
    (exp_dir / "evaluation_results").mkdir(parents=True)
    (exp_dir / "evaluation_results" / "run_params.json").write_text(
        json.dumps(run_params)
    )
    return exp_dir


def _make_data_dir(tmp_path: Path, *, with_normalized: bool) -> Path:
    data_dir = tmp_path / "data" / "gripper" / "multi_problem"
    raw = data_dir / "training" / "trajectories" / "problem0"
    raw.mkdir(parents=True)
    (raw / "problem0.pddl").write_text("(:init (at-robby rooma))")

    if with_normalized:
        norm = data_dir / "training" / "trajectories_normalized" / "problem0"
        norm.mkdir(parents=True)
        (norm / "problem0.pddl").write_text("(:init (at_robby rooma))")

    return data_dir


# ── resolve_problem_dir ──────────────────────────────────────────────────

def test_normalized_experiment_resolves_to_normalized_dir(tmp_path):
    exp_dir = _make_experiment(tmp_path, {"normalized": True})
    data_dir = _make_data_dir(tmp_path, with_normalized=True)

    assert resolve_problem_dir(exp_dir, data_dir) == (
        data_dir / "training" / "trajectories_normalized"
    )


def test_unnormalized_experiment_resolves_to_data_dir(tmp_path):
    """A normalized dir on disk must not override a normalized=false record.

    Simulated experiments never normalize, yet they routinely share a data_dir
    with an image experiment that did — so the directory's existence says
    nothing about this experiment's dialect.
    """
    exp_dir = _make_experiment(tmp_path, {"normalized": False})
    data_dir = _make_data_dir(tmp_path, with_normalized=True)

    assert resolve_problem_dir(exp_dir, data_dir) == data_dir


def test_missing_normalized_flag_resolves_to_data_dir(tmp_path):
    exp_dir = _make_experiment(tmp_path, {})
    data_dir = _make_data_dir(tmp_path, with_normalized=True)

    assert resolve_problem_dir(exp_dir, data_dir) == data_dir


def test_missing_run_params_resolves_to_data_dir(tmp_path):
    exp_dir = tmp_path / "running_results" / "gripper" / "TO=600__exp"
    exp_dir.mkdir(parents=True)
    data_dir = _make_data_dir(tmp_path, with_normalized=True)

    assert resolve_problem_dir(exp_dir, data_dir) == data_dir


def test_normalized_flag_but_dir_deleted_falls_back_and_warns(tmp_path, capsys):
    exp_dir = _make_experiment(tmp_path, {"normalized": True})
    data_dir = _make_data_dir(tmp_path, with_normalized=False)

    assert resolve_problem_dir(exp_dir, data_dir) == data_dir
    assert "normalized=true" in capsys.readouterr().out


# ── find_problem_pddl against each resolved root ─────────────────────────

@pytest.mark.parametrize(
    "normalized, expected_fluent",
    [(True, "at_robby"), (False, "at-robby")],
)
def test_resolved_root_yields_the_matching_dialect(
    tmp_path, normalized, expected_fluent
):
    exp_dir = _make_experiment(tmp_path, {"normalized": normalized})
    data_dir = _make_data_dir(tmp_path, with_normalized=True)

    problem_dir = resolve_problem_dir(exp_dir, data_dir)
    problem_pddl = find_problem_pddl(problem_dir, "problem0")

    assert problem_pddl is not None
    assert expected_fluent in problem_pddl.read_text()


def test_find_problem_pddl_handles_a_flat_trajectories_root(tmp_path):
    """The normalized dir has no training/trajectories/ level beneath it."""
    data_dir = _make_data_dir(tmp_path, with_normalized=True)
    norm = data_dir / "training" / "trajectories_normalized"

    assert find_problem_pddl(norm, "problem0") == norm / "problem0" / "problem0.pddl"


def test_find_problem_pddl_returns_none_for_unknown_problem(tmp_path):
    data_dir = _make_data_dir(tmp_path, with_normalized=False)

    assert find_problem_pddl(data_dir, "problem9") is None


# ── the override path still gets a dialect decision ──────────────────────

def test_data_dir_override_is_still_dialect_resolved(tmp_path):
    exp_dir = _make_experiment(tmp_path, {"normalized": True, "data_dir": "/nope"})
    override = _make_data_dir(tmp_path, with_normalized=True)

    data_dir, src = resolve_data_dir(exp_dir, override)
    assert (data_dir, src) == (override, "override")
    assert resolve_problem_dir(exp_dir, data_dir) == (
        override / "training" / "trajectories_normalized"
    )
