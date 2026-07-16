"""Algorithm selection for experiments.

One flat namespace covers our own learner and the competitor baselines, so a run
can execute any subset of them (including a single one, standalone) over the same
data via ``--algorithms``.

- ``cdps`` — our Conflict-Directed Patch Search (the conflict-search denoiser).
  It is not a ``BaselineRunner``; it has its own execution path in
  ``run_single_fold`` (multi-CFM output + rich per-cell artifacts).
- everything else — competitor baselines from ``benchmark.baselines`` (e.g.
  ``rosame``), each a ``BaselineRunner``.

To rename our algorithm's label in every result/plot, change
``CDPS_ALGORITHM_NAME`` here (single source of truth).
"""

from __future__ import annotations

from typing import List, Tuple

from benchmark.baselines import BASELINE_REGISTRY, BaselineRunner, resolve_baselines

# Selector key (CLI/config) and the display/results label for our algorithm.
CDPS = "cdps"
CDPS_ALGORITHM_NAME = "CDPS"


def available_algorithms() -> List[str]:
    """All valid ``--algorithms`` names: our learner plus the baseline keys."""
    return [CDPS] + sorted(BASELINE_REGISTRY)


def resolve_algorithms(names: List[str]) -> Tuple[bool, List[BaselineRunner]]:
    """Split selected algorithm names into ``(run_cdps, baseline_runners)``.

    Args:
        names: Algorithm keys, e.g. ``["cdps", "rosame"]``, ``["rosame"]``.

    Returns:
        run_cdps: Whether to run our conflict-search learner.
        baseline_runners: Instantiated ``BaselineRunner`` objects for the rest.

    Raises:
        ValueError: If nothing is selected or a name is unknown.
    """
    lowered = [n.strip().lower() for n in names]
    if not lowered or lowered == ["none"]:
        raise ValueError(
            f"No algorithms selected. Available: {', '.join(available_algorithms())}"
        )

    run_cdps = CDPS in lowered
    baseline_names = [n for n in lowered if n != CDPS]
    baseline_runners = resolve_baselines(baseline_names) if baseline_names else []

    if not run_cdps and not baseline_runners:
        raise ValueError(
            f"No algorithms selected. Available: {', '.join(available_algorithms())}"
        )
    return run_cdps, baseline_runners
