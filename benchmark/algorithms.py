"""Algorithm selection for experiments.

One flat namespace covers our own learner and the competitor baselines, so a run
can execute any subset of them (including a single one, standalone) over the same
data via ``--algorithms``.

- ``cdps`` — our Conflict-Directed Patch Search (the conflict-search denoiser).
  It is not a ``BaselineRunner``; it has its own execution path in
  ``run_single_fold`` (multi-CFM output + rich per-cell artifacts).
- ``cdps_milp_single_round`` — same learner (PI-SAM), same artifacts, but the
  denoiser is one CP-SAT solve instead of a search. Shares CDPS's execution
  path, so it is a "CDPS-family" key too, not a baseline.
- everything else — competitor baselines from ``benchmark.baselines`` (e.g.
  ``rosame``), each a ``BaselineRunner``.

To rename our algorithm's label in every result/plot, change
``CDPS_ALGORITHM_NAME`` here (single source of truth).
"""

from __future__ import annotations

from typing import List, Tuple

from benchmark.baselines import BASELINE_REGISTRY, BaselineRunner, resolve_baselines
from src.pi_sam.plan_denoising.milp_version.config import CdpsMilpConfig
from src.pi_sam.plan_denoising.milp_version.converter import GtAnchoring

# Selector keys (CLI/config) and the display/results labels for our algorithm.
# ``cdps`` anchors only the init state as GT; ``cdps_anchored`` also anchors the
# final state (init + final GT) — see ConflictDrivenPatchSearch.anchor_endpoints.
CDPS = "cdps"
CDPS_ALGORITHM_NAME = "CDPS"
CDPS_ANCHORED = "cdps_anchored"
CDPS_ANCHORED_ALGORITHM_NAME = "CDPS_ANCHORED"

# The MILP denoiser (docs/cdps-milp-denoiser-design.md). Its results label is
# arm-suffixed — see cdps_milp_algorithm_name.
CDPS_MILP_SINGLE_ROUND = "cdps_milp_single_round"
CDPS_MILP_SINGLE_ROUND_ALGORITHM_NAME = "CDPS_MILP_SR"

# CDPS-family keys (our learner), distinct from BaselineRunner competitors.
_CDPS_KEYS = {CDPS, CDPS_ANCHORED, CDPS_MILP_SINGLE_ROUND}


def cdps_milp_algorithm_name(config: CdpsMilpConfig) -> str:
    """Results label for one ``cdps_milp`` arm.

    Arms that produce different models must not share a label, or a sweep
    silently averages two algorithms into one row. For ``single_round`` the
    behaviour-changing knobs are:

    - ``eq16`` — the precondition bias changes T' itself, not just the witness
      model (and voids the ``cost(MILP) <= cost(CDPS)`` lower-bound check);
    - ``gt_anchoring`` — decides which states are unrepairable.

    ``w_prior`` is inert here (a single round has no reference model) and
    ``solver``/``obs_weights`` have one implemented value each, so none of them
    enter the label.
    """
    parts = []
    if config.eq16:
        parts.append(f"eq16={config.lambda_pre:g}")
    if config.gt_anchoring is not GtAnchoring.INIT_ONLY:
        parts.append(f"gt={config.gt_anchoring.value}")
    suffix = ("__" + "__".join(parts)) if parts else ""
    return f"{CDPS_MILP_SINGLE_ROUND_ALGORITHM_NAME}{suffix}"


def available_algorithms() -> List[str]:
    """All valid ``--algorithms`` names: our learner variants plus baseline keys."""
    return [CDPS, CDPS_ANCHORED, CDPS_MILP_SINGLE_ROUND] + sorted(BASELINE_REGISTRY)


def resolve_algorithms(
    names: List[str], **runner_kwargs
) -> Tuple[bool, bool, bool, List[BaselineRunner]]:
    """Split selected algorithm names into the CDPS-family flags plus baselines.

    Args:
        names: Algorithm keys, e.g. ``["cdps", "rosame"]``, ``["cdps_anchored"]``.
        **runner_kwargs: Optional per-baseline options forwarded only to the
            runners whose ``__init__`` accepts them (e.g.
            ``train_per_trajectory`` for ROSAME-I).

    Returns:
        run_cdps: Whether to run our (init-anchored) conflict-search learner.
        run_cdps_anchored: Whether to run the init+final-anchored variant.
        run_cdps_milp: Whether to run the single-round MILP denoiser.
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
    run_cdps_milp = CDPS_MILP_SINGLE_ROUND in lowered
    baseline_names = [n for n in lowered if n not in _CDPS_KEYS]
    baseline_runners = (
        resolve_baselines(baseline_names, **runner_kwargs) if baseline_names else []
    )

    if not any((run_cdps, run_cdps_anchored, run_cdps_milp)) and not baseline_runners:
        raise ValueError(
            f"No algorithms selected. Available: {', '.join(available_algorithms())}"
        )
    return run_cdps, run_cdps_anchored, run_cdps_milp, baseline_runners
