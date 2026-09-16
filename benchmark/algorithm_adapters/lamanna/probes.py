"""Module-level functions for exercising :func:`run_learner` from tests.

The child process imports its target by name, so these cannot live inside a
test module.
"""

from __future__ import annotations

import os
import sys
import time


def echo(text: str) -> str:
    """Return ``text`` after printing a marker to stdout and stderr."""
    print(f"stdout:{text}")
    print(f"stderr:{text}", file=sys.stderr)
    return text


def report_cwd() -> str:
    """Return the child's working directory."""
    return os.getcwd()


def sleep_forever() -> str:  # pragma: no cover - killed by the caller
    while True:
        time.sleep(1)


def call_exit() -> str:
    exit(3)


def raise_error() -> str:
    raise RuntimeError("probe failure")
