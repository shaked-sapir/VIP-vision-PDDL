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
- ``cdps_milp_loop`` — the same MILP denoiser run repeatedly over sampled
  subsets, keeping the model that best reconstructs the original observations
  (``docs/cdps-milp-loop-plan.md``). Also CDPS-family.
- everything else — competitor baselines from ``benchmark.baselines`` (e.g.
  ``rosame``), each a ``BaselineRunner``.

To rename our algorithm's label in every result/plot, change
``CDPS_ALGORITHM_NAME`` here (single source of truth).
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Tuple

from benchmark.baselines import BASELINE_REGISTRY, BaselineRunner, resolve_baselines
from src.pi_sam.plan_denoising.milp_version.config import (
    CdpsMilpConfig,
    LearnerInput,
    MilpVariant,
    PoolPolicy,
    Sampler,
    SubsetSizeKind,
)
from src.pi_sam.plan_denoising.milp_version.encoding_config import PriorWeightMode
from src.pi_sam.plan_denoising.milp_version.converter import GtAnchoring

# Selector keys (CLI/config) and the display/results labels for our algorithm.
# ``cdps`` anchors only the init state as GT; ``cdps_anchored`` also anchors the
# final state (init + final GT) — see ConflictDrivenPatchSearch.anchor_endpoints.
CDPS = "cdps"
CDPS_ALGORITHM_NAME = "CDPS"
CDPS_ANCHORED = "cdps_anchored"
CDPS_ANCHORED_ALGORITHM_NAME = "CDPS_ANCHORED"

# The MILP denoiser (docs/cdps-milp-denoiser-design.md, docs/cdps-milp-loop-plan.md).
# Both labels are arm-suffixed — see cdps_milp_algorithm_name.
CDPS_MILP_SINGLE_ROUND = "cdps_milp_single_round"
CDPS_MILP_SINGLE_ROUND_ALGORITHM_NAME = "CDPS_MILP_SR"
CDPS_MILP_LOOP = "cdps_milp_loop"
CDPS_MILP_LOOP_ALGORITHM_NAME = "CDPS_MILP_LOOP"

# Which ``MilpVariant`` each key selects. The key wins over the ``variant:`` YAML
# key, so one ``cdps_milp`` block can serve both arms in a single run.
_MILP_VARIANT_BY_KEY = {
    CDPS_MILP_SINGLE_ROUND: MilpVariant.SINGLE_ROUND,
    CDPS_MILP_LOOP: MilpVariant.LOOP,
}

# CDPS-family keys (our learner), distinct from BaselineRunner competitors.
_CDPS_KEYS = {CDPS, CDPS_ANCHORED, CDPS_MILP_SINGLE_ROUND, CDPS_MILP_LOOP}


def _shared_milp_suffix_parts(config: CdpsMilpConfig) -> List[str]:
    """Label parts for the knobs both MILP variants share.

    - ``eq16`` — the precondition bias changes T' itself, not only the witness
      model (and voids the ``cost(MILP) <= cost(CDPS)`` lower-bound check);
    - ``gt_anchoring`` — decides which states are unrepairable.

    ``solver`` and ``obs_weights`` have one implemented value each, so neither
    can distinguish two arms and neither enters the label.
    """
    parts = []
    if config.eq16:
        parts.append(f"eq16={config.lambda_pre:g}")
    if config.gt_anchoring is not GtAnchoring.INIT_ONLY:
        parts.append(f"gt={config.gt_anchoring.value}")
    return parts


def _loop_suffix_parts(config: CdpsMilpConfig) -> List[str]:
    """Label parts for the loop-only knobs that change which model comes out.

    Every one of these changes the *sequence of candidates* the loop scores, so
    two arms differing in any of them are different algorithms. ``stop`` rules
    are deliberately absent: they cap the search, exactly as a timeout does, and
    are already recorded per row in ``algorithm_specific``.
    """
    parts = []
    if config.w_prior is not PriorWeightMode.TIEBREAK:
        parts.append(f"prior={config.w_prior.value}")
    if config.sampler is not Sampler.RANDOM:
        parts.append(f"samp={config.sampler.value}")
    if config.subset_size.kind is not SubsetSizeKind.HALF:
        parts.append(f"m={config.subset_size.as_stat()}")
    if config.learner_input is not LearnerInput.SUBSET_ONLY:
        parts.append(f"in={config.learner_input.value}")
    if config.pool_policy is not PoolPolicy.FROZEN:
        parts.append(f"pool={config.pool_policy.value}")
    if config.co_sample_conflicts:
        parts.append("cosample")
    return parts


def cdps_milp_algorithm_name(config: CdpsMilpConfig) -> str:
    """Results label for one ``cdps_milp`` arm, keyed on ``config.variant``.

    Arms that produce different models must not share a label, or a sweep
    silently averages two algorithms into one row. Only knobs that change the
    returned model enter the suffix; a default arm gets no suffix at all.

    ``w_prior`` appears for the loop only — a single round has no reference
    model, so there it is inert.
    """
    is_loop = config.variant is MilpVariant.LOOP
    base = (
        CDPS_MILP_LOOP_ALGORITHM_NAME if is_loop
        else CDPS_MILP_SINGLE_ROUND_ALGORITHM_NAME
    )
    parts = _shared_milp_suffix_parts(config)
    if is_loop:
        parts.extend(_loop_suffix_parts(config))
    suffix = ("__" + "__".join(parts)) if parts else ""
    return f"{base}{suffix}"


def milp_config_for(key: str, config: CdpsMilpConfig) -> CdpsMilpConfig:
    """``config`` with ``variant`` pinned to the one ``key`` names.

    The selected algorithm — not the YAML ``variant:`` key — decides which
    driver runs, so a run can execute both MILP arms over one shared block.
    """
    return replace(config, variant=_MILP_VARIANT_BY_KEY[key])


def cdps_family_names(
    run_cdps: bool,
    run_cdps_anchored: bool,
    run_cdps_milp: bool,
    run_cdps_milp_loop: bool,
    milp_config: Optional[CdpsMilpConfig] = None,
) -> List[str]:
    """Result labels for the selected CDPS-family arms, in execution order.

    Both the run banner and ``run_params["algorithms"]`` must name the arms the
    way the result rows do, or a sweep's manifest disagrees with its own data.
    Deriving both from here is what keeps them in step — in particular the MILP
    labels, which are arm-suffixed and so cannot be written as constants.

    Baselines are the caller's business: the two callers read different
    attributes off their runner objects.
    """
    config = milp_config if milp_config is not None else CdpsMilpConfig()
    names = []
    if run_cdps:
        names.append(CDPS_ALGORITHM_NAME)
    if run_cdps_anchored:
        names.append(CDPS_ANCHORED_ALGORITHM_NAME)
    for key, selected in (
        (CDPS_MILP_SINGLE_ROUND, run_cdps_milp),
        (CDPS_MILP_LOOP, run_cdps_milp_loop),
    ):
        if selected:
            names.append(cdps_milp_algorithm_name(milp_config_for(key, config)))
    return names


def available_algorithms() -> List[str]:
    """All valid ``--algorithms`` names: our learner variants plus baseline keys."""
    return [
        CDPS, CDPS_ANCHORED, CDPS_MILP_SINGLE_ROUND, CDPS_MILP_LOOP
    ] + sorted(BASELINE_REGISTRY)


def resolve_algorithms(
    names: List[str], **runner_kwargs
) -> Tuple[bool, bool, bool, bool, List[BaselineRunner]]:
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
        run_cdps_milp_loop: Whether to run the multi-round MILP loop.
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
    run_cdps_milp_loop = CDPS_MILP_LOOP in lowered
    baseline_names = [n for n in lowered if n not in _CDPS_KEYS]
    baseline_runners = (
        resolve_baselines(baseline_names, **runner_kwargs) if baseline_names else []
    )

    cdps_family = (run_cdps, run_cdps_anchored, run_cdps_milp, run_cdps_milp_loop)
    if not any(cdps_family) and not baseline_runners:
        raise ValueError(
            f"No algorithms selected. Available: {', '.join(available_algorithms())}"
        )
    return (*cdps_family, baseline_runners)
