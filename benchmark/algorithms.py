"""Algorithm selection for experiments.

One flat namespace covers our own learner and the competitor baselines, so a run
can execute any subset of them (including a single one, standalone) over the same
data via ``--algorithms``.

- ``cdps`` — our Conflict-Directed Patch Search (the conflict-search denoiser).
  It is not a ``BaselineRunner``; it has its own execution path in
  ``run_single_fold`` (multi-CFM output + rich per-cell artifacts).
- ``pisam_milp_single_round`` — same learner (PI-SAM), same artifacts, but the
  denoiser is one CP-SAT solve instead of a conflict search.
- ``pisam_milp_loop`` — the same MILP denoiser run repeatedly over sampled
  subsets, keeping the model that best reconstructs the original observations
  (``docs/pisam-milp-loop-plan.md``).
- everything else — competitor baselines from ``benchmark.baselines`` (e.g.
  ``rosame_24``), each a ``BaselineRunner``.

To rename our algorithm's label in every result/plot, change
``CDPS_ALGORITHM_NAME`` here (single source of truth).
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Tuple

from benchmark.baselines import BASELINE_REGISTRY, BaselineRunner, resolve_baselines
from src.plan_denoising.milp_denoiser.config import (
    PisamMilpConfig,
    LearnerInput,
    MilpVariant,
    PoolPolicy,
    Sampler,
    SubsetSizeKind,
)
from src.milp.encoding_config import PriorWeightMode
from src.milp.converter import GtAnchoring

# Selector keys (CLI/config) and the display/results labels for our algorithm.
# ``cdps`` anchors only the init state as GT; ``cdps_anchored`` also anchors the
# final state (init + final GT) — see learning_helpers.learn_cdps(anchor_endpoints=...).
CDPS = "cdps"
CDPS_ALGORITHM_NAME = "CDPS"
CDPS_ANCHORED = "cdps_anchored"
CDPS_ANCHORED_ALGORITHM_NAME = "CDPS_ANCHORED"

# The MILP denoiser (docs/pisam-milp-denoiser-design.md, docs/pisam-milp-loop-plan.md).
# Both labels are arm-suffixed — see pisam_milp_algorithm_name.
PISAM_MILP_SINGLE_ROUND = "pisam_milp_single_round"
PISAM_MILP_SINGLE_ROUND_ALGORITHM_NAME = "PISAM_MILP_SR"
PISAM_MILP_LOOP = "pisam_milp_loop"
PISAM_MILP_LOOP_ALGORITHM_NAME = "PISAM_MILP_LOOP"

# Which ``MilpVariant`` each key selects. The key wins over the ``variant:`` YAML
# key, so one ``pisam_milp`` block can serve both arms in a single run.
_MILP_VARIANT_BY_KEY = {
    PISAM_MILP_SINGLE_ROUND: MilpVariant.SINGLE_ROUND,
    PISAM_MILP_LOOP: MilpVariant.LOOP,
}

# Keys handled by our own learner's execution path (``run_cdps_phase``) rather
# than by a ``BaselineRunner``. Used negatively: anything not here falls through
# to ``resolve_baselines``, which raises on an unregistered name.
_OUR_LEARNER_KEYS = {CDPS, CDPS_ANCHORED, PISAM_MILP_SINGLE_ROUND, PISAM_MILP_LOOP}


def _shared_milp_suffix_parts(config: PisamMilpConfig) -> List[str]:
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


def _loop_suffix_parts(config: PisamMilpConfig) -> List[str]:
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


def pisam_milp_algorithm_name(config: PisamMilpConfig) -> str:
    """Results label for one ``pisam_milp`` arm, keyed on ``config.variant``.

    Arms that produce different models must not share a label, or a sweep
    silently averages two algorithms into one row. Only knobs that change the
    returned model enter the suffix; a default arm gets no suffix at all.

    ``w_prior`` appears for the loop only — a single round has no reference
    model, so there it is inert.
    """
    is_loop = config.variant is MilpVariant.LOOP
    base = (
        PISAM_MILP_LOOP_ALGORITHM_NAME if is_loop
        else PISAM_MILP_SINGLE_ROUND_ALGORITHM_NAME
    )
    parts = _shared_milp_suffix_parts(config)
    if is_loop:
        parts.extend(_loop_suffix_parts(config))
    suffix = ("__" + "__".join(parts)) if parts else ""
    return f"{base}{suffix}"


def milp_config_for(key: str, config: PisamMilpConfig) -> PisamMilpConfig:
    """``config`` with ``variant`` pinned to the one ``key`` names.

    The selected algorithm — not the YAML ``variant:`` key — decides which
    driver runs, so a run can execute both MILP arms over one shared block.
    """
    return replace(config, variant=_MILP_VARIANT_BY_KEY[key])


def milp_configs_for(key: str, configs: List[PisamMilpConfig]) -> List[PisamMilpConfig]:
    """The distinct configs arm ``key`` should actually run, in order.

    An ablation block is written once and applies to whichever MILP arms the run
    selects, so the two arms do not get the same list: the loop-only knobs are
    inert under ``single_round``, and crossing them there would run the identical
    solve several times and file the results under one label. Dropping those
    duplicates is what lets ``ablations: {pool_policy: [...]}`` coexist with
    ``algorithms: [pisam_milp_single_round, pisam_milp_loop]``.

    What survives dedup must still be pairwise distinguishable *in the results*,
    which is a stricter demand than being distinct configs: two arms that differ
    only in a knob outside :func:`pisam_milp_algorithm_name`'s suffix would be
    averaged into one row, and the row would name neither of them. That is
    caught here rather than discovered in a report.

    Args:
        key: A ``pisam_milp_*`` algorithm key.
        configs: The expansion of the ``pisam_milp:`` block.

    Returns:
        Variant-pinned configs, duplicates removed, first occurrence kept.

    Raises:
        ValueError: If two surviving configs share a results label.
    """
    pinned: List[PisamMilpConfig] = []
    seen = set()
    for config in configs:
        candidate = milp_config_for(key, config)
        identity = candidate.arm_identity()
        if identity in seen:
            continue
        seen.add(identity)
        pinned.append(candidate)

    by_label: dict = {}
    for config in pinned:
        by_label.setdefault(pisam_milp_algorithm_name(config), []).append(config)
    collisions = {label: arms for label, arms in by_label.items() if len(arms) > 1}
    if collisions:
        label = next(iter(collisions))
        raise ValueError(
            f"pisam_milp.ablations produced {len(collisions[label])} distinct "
            f"{key} arms that all report as {label!r}. They differ only in knobs "
            f"that do not enter the results label (e.g. `seed`), so their rows "
            f"would be merged. Ablate a knob the label carries, or run them as "
            f"separate experiments."
        )
    return pinned


def milp_work_subdir(key: str, config: PisamMilpConfig) -> str:
    """Fold subdirectory for one MILP arm's artifacts.

    Ablated arms of the same variant run in one fold, so the variant's name
    alone no longer identifies a directory. The label's suffix is what
    distinguishes them, and reusing it keeps the directory readable from the
    results row and vice versa. A non-ablated arm keeps the bare name it has
    always had, so existing folds stay where every reader already looks.
    """
    label = pisam_milp_algorithm_name(milp_config_for(key, config))
    _, _, suffix = label.partition("__")
    return f"{key}__{suffix}" if suffix else key


def cdps_family_names(
    run_cdps: bool,
    run_cdps_anchored: bool,
    run_pisam_milp: bool,
    run_pisam_milp_loop: bool,
    milp_configs: Optional[List[PisamMilpConfig]] = None,
) -> List[str]:
    """Result labels for the selected CDPS-family arms, in execution order.

    Both the run banner and ``run_params["algorithms"]`` must name the arms the
    way the result rows do, or a sweep's manifest disagrees with its own data.
    Deriving both from here is what keeps them in step — in particular the MILP
    labels, which are arm-suffixed and so cannot be written as constants, and
    which an ablation multiplies: one selected MILP key can contribute several
    rows, and the manifest has to say so before the run rather than after.

    Baselines are the caller's business: the two callers read different
    attributes off their runner objects.
    """
    configs = milp_configs if milp_configs else [PisamMilpConfig()]
    names = []
    if run_cdps:
        names.append(CDPS_ALGORITHM_NAME)
    if run_cdps_anchored:
        names.append(CDPS_ANCHORED_ALGORITHM_NAME)
    for key, selected in (
        (PISAM_MILP_SINGLE_ROUND, run_pisam_milp),
        (PISAM_MILP_LOOP, run_pisam_milp_loop),
    ):
        if selected:
            names.extend(
                pisam_milp_algorithm_name(c) for c in milp_configs_for(key, configs)
            )
    return names


def available_algorithms() -> List[str]:
    """All valid ``--algorithms`` names: our learner variants plus baseline keys."""
    return [
        CDPS, CDPS_ANCHORED, PISAM_MILP_SINGLE_ROUND, PISAM_MILP_LOOP
    ] + sorted(BASELINE_REGISTRY)


def resolve_algorithms(
    names: List[str], **runner_kwargs
) -> Tuple[bool, bool, bool, bool, List[BaselineRunner]]:
    """Split selected algorithm names into the CDPS-family flags plus baselines.

    Args:
        names: Algorithm keys, e.g. ``["cdps", "rosame_24"]``, ``["cdps_anchored"]``.
        **runner_kwargs: Optional per-baseline options forwarded only to the
            runners whose ``__init__`` accepts them (e.g.
            ``train_per_trajectory`` for the symbolic ROSAME runners).

    Returns:
        run_cdps: Whether to run our (init-anchored) conflict-search learner.
        run_cdps_anchored: Whether to run the init+final-anchored variant.
        run_pisam_milp: Whether to run the single-round MILP denoiser.
        run_pisam_milp_loop: Whether to run the multi-round MILP loop.
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
    run_pisam_milp = PISAM_MILP_SINGLE_ROUND in lowered
    run_pisam_milp_loop = PISAM_MILP_LOOP in lowered
    baseline_names = [n for n in lowered if n not in _OUR_LEARNER_KEYS]
    baseline_runners = (
        resolve_baselines(baseline_names, **runner_kwargs) if baseline_names else []
    )

    cdps_family = (run_cdps, run_cdps_anchored, run_pisam_milp, run_pisam_milp_loop)
    if not any(cdps_family) and not baseline_runners:
        raise ValueError(
            f"No algorithms selected. Available: {', '.join(available_algorithms())}"
        )
    return (*cdps_family, baseline_runners)
