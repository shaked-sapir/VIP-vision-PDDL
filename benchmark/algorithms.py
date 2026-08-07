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

# Selector keys (CLI/config) and the display/results labels for our algorithm.
# ``cdps`` anchors only the init state as GT; ``cdps_anchored`` also anchors the
# final state (init + final GT) — see ConflictDrivenPatchSearch.anchor_endpoints.
CDPS = "cdps"
CDPS_ALGORITHM_NAME = "CDPS"
CDPS_ANCHORED = "cdps_anchored"
CDPS_ANCHORED_ALGORITHM_NAME = "CDPS_ANCHORED"

# CDPS variant keys (our learner), distinct from BaselineRunner competitors.
_CDPS_KEYS = {CDPS, CDPS_ANCHORED}


def available_algorithms() -> List[str]:
    """All valid ``--algorithms`` names: our learner variants plus baseline keys."""
    return [CDPS, CDPS_ANCHORED] + sorted(BASELINE_REGISTRY)


def resolve_algorithms(
    names: List[str], **runner_kwargs
) -> Tuple[bool, bool, List[BaselineRunner]]:
    """Split selected algorithm names into ``(run_cdps, run_cdps_anchored, baselines)``.

    Args:
        names: Algorithm keys, e.g. ``["cdps", "rosame"]``, ``["cdps_anchored"]``.
        **runner_kwargs: Optional per-baseline options forwarded only to the
            runners whose ``__init__`` accepts them (e.g.
            ``train_per_trajectory`` for ROSAME-I).

    Returns:
        run_cdps: Whether to run our (init-anchored) conflict-search learner.
        run_cdps_anchored: Whether to run the init+final-anchored variant.
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
    run_cdps_anchored = CDPS_ANCHORED in lowered
    baseline_names = [n for n in lowered if n not in _CDPS_KEYS]
    baseline_runners = (
        resolve_baselines(baseline_names, **runner_kwargs) if baseline_names else []
    )

    if not run_cdps and not run_cdps_anchored and not baseline_runners:
        raise ValueError(
            f"No algorithms selected. Available: {', '.join(available_algorithms())}"
        )
    return run_cdps, run_cdps_anchored, baseline_runners
