"""The child-process harness turns every learner failure into a recorded outcome.

    python -m pytest benchmark/algorithm_adapters/lamanna/test_subprocess_learn.py
"""

from __future__ import annotations

from benchmark.algorithm_adapters.lamanna import probes
from benchmark.algorithm_adapters.lamanna.subprocess_learn import (
    COMPLETED,
    ERROR,
    TIMEOUT,
    run_learner,
)


def test_a_completed_learner_returns_its_model_text(tmp_path):
    outcome = run_learner(probes.echo, {"text": "(define (domain d))"}, tmp_path / "ws", 30)
    assert outcome.terminated_by == COMPLETED
    assert outcome.completed
    assert outcome.model_text == "(define (domain d))"
    assert outcome.wall_seconds >= 0


def test_stdout_and_stderr_land_in_the_log(tmp_path):
    outcome = run_learner(probes.echo, {"text": "hello"}, tmp_path / "ws", 30)
    log = outcome.log_path.read_text()
    assert "stdout:hello" in log
    assert "stderr:hello" in log


def test_the_child_runs_inside_the_workspace(tmp_path):
    workspace = tmp_path / "ws"
    outcome = run_learner(probes.report_cwd, {}, workspace, 30)
    assert outcome.model_text == str(workspace.resolve())


def test_a_hung_learner_is_killed_and_reported_as_timeout(tmp_path):
    outcome = run_learner(probes.sleep_forever, {}, tmp_path / "ws", 2)
    assert outcome.terminated_by == TIMEOUT
    assert outcome.model_text is None
    assert "killed" in outcome.detail


def test_a_learner_calling_exit_does_not_take_the_caller_down(tmp_path):
    outcome = run_learner(probes.call_exit, {}, tmp_path / "ws", 30)
    assert outcome.terminated_by == ERROR
    assert outcome.model_text is None
    assert "exit(3)" in outcome.detail


def test_an_exception_is_reported_with_its_type(tmp_path):
    outcome = run_learner(probes.raise_error, {}, tmp_path / "ws", 30)
    assert outcome.terminated_by == ERROR
    assert "RuntimeError: probe failure" in outcome.detail
