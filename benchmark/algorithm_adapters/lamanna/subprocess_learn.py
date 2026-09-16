"""Run a learner function in a child process with its own cwd and a wall-clock cap.

The learners this serves take no time budget, call ``exit()`` on a parse error
and write files relative to the cwd. A child process is what turns those into
a ``timeout`` or ``error`` outcome the fold can record, instead of a dead fold.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping, Optional

COMPLETED = "completed"
TIMEOUT = "timeout"
ERROR = "error"

RESULT_FILENAME = "learner_result.json"
LOG_FILENAME = "learner.log"


@dataclass(frozen=True)
class LearnOutcome:
    """What the child process came back with."""

    model_text: Optional[str]
    terminated_by: str
    wall_seconds: float
    detail: str = ""
    log_path: Optional[Path] = None

    @property
    def completed(self) -> bool:
        return self.terminated_by == COMPLETED


def _redirect_output(log_path: str) -> None:
    """Send the child's fds 1 and 2 to ``log_path``, so library prints land there too."""
    sys.stdout.flush()
    sys.stderr.flush()
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)


def _child_main(fn: Callable[..., str], kwargs: dict, cwd: str, log_path: str) -> None:
    """Child entry point: chdir, redirect output, run ``fn``, persist the outcome."""
    result_path = os.path.join(cwd, RESULT_FILENAME)
    try:
        os.chdir(cwd)
        _redirect_output(log_path)
        model_text = fn(**kwargs)
        payload = {"status": COMPLETED, "model_text": model_text}
    except SystemExit as exc:
        payload = {"status": ERROR, "detail": f"learner called exit({exc.code})"}
    except BaseException as exc:  # noqa: BLE001 - every failure must become a recorded outcome
        payload = {
            "status": ERROR,
            "detail": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    with open(result_path, "w") as handle:
        json.dump(payload, handle)


def run_learner(
    fn: Callable[..., str],
    kwargs: Mapping[str, object],
    cwd: Path,
    timeout_seconds: Optional[float],
    log_path: Optional[Path] = None,
) -> LearnOutcome:
    """Call ``fn(**kwargs)`` in a child process rooted at ``cwd``.

    Args:
        fn: A module-level function returning the learned model's PDDL text.
            It is imported by name in the child, so it must be importable.
        kwargs: Its keyword arguments; picklable.
        cwd: The child's working directory, created if missing. Everything
            the learner writes relatively lands here.
        timeout_seconds: Wall-clock cap; ``None`` waits indefinitely.
        log_path: Where the child's stdout and stderr go; defaults to
            ``cwd / "learner.log"``.

    Returns:
        A :class:`LearnOutcome`; ``model_text`` is ``None`` unless
        ``terminated_by`` is :data:`COMPLETED`.
    """
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_path) if log_path is not None else cwd / LOG_FILENAME
    result_path = cwd / RESULT_FILENAME
    if result_path.exists():
        result_path.unlink()

    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_child_main, args=(fn, dict(kwargs), str(cwd), str(log_path))
    )
    start = perf_counter()
    process.start()
    process.join(timeout_seconds)
    wall = perf_counter() - start

    if process.is_alive():
        process.kill()
        process.join()
        return LearnOutcome(
            None, TIMEOUT, wall, f"killed after {timeout_seconds}s", log_path
        )

    if not result_path.exists():
        return LearnOutcome(
            None, ERROR, wall,
            f"child exited with code {process.exitcode} and left no result",
            log_path,
        )
    payload = json.loads(result_path.read_text())
    if payload.get("status") == COMPLETED:
        return LearnOutcome(payload.get("model_text"), COMPLETED, wall, "", log_path)
    return LearnOutcome(None, ERROR, wall, payload.get("detail", ""), log_path)
