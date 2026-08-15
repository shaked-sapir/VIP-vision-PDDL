"""Tests for MILP ablation expansion and arm selection.

    python -m benchmark.test_milp_ablations

The feature spans two layers on purpose: ``expand_pisam_milp_ablations`` lives in
``src`` because it only builds configs, while the *selection* rules that need a
results label live in ``benchmark.algorithms`` (``src`` must not import
``benchmark``). The tests sit on the benchmark side, which is the only side that
can see both.

What is actually at risk here is not the cross-product — it is the two ways an
ablation can quietly cost you a result:

1. running the same solve twice under two names (loop-only knobs crossed into
   ``single_round``), and
2. running two different algorithms under *one* name (a knob the results label
   does not carry), which merges them into a row that describes neither.

Both are cheap to write and expensive to discover in a finished report, so each
has a test that fails loudly rather than a comment saying it should not happen.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.algorithms import (
    PISAM_MILP_LOOP,
    PISAM_MILP_SINGLE_ROUND,
    cdps_family_names,
    pisam_milp_algorithm_name,
    milp_configs_for,
    milp_work_subdir,
)
from src.plan_denoising.milp_denoiser.config import (
    PisamMilpConfig,
    LearnerInput,
    MilpVariant,
    PoolPolicy,
    expand_pisam_milp_ablations,
)

# A block that names every knob the tests below override, so a test that changes
# one knob is visibly changing only that knob.
_BASE = {
    "eq16": "off",
    "lambda_pre": 0.4,
    "pool_policy": "frozen",
    "learner_input": "subset_only",
    "seed": 42,
}


def _labels(key: str, raw: dict) -> list:
    """Results labels for the arms ``key`` runs off config block ``raw``."""
    return [
        pisam_milp_algorithm_name(config)
        for config in milp_configs_for(key, expand_pisam_milp_ablations(raw))
    ]


def test_no_ablations_is_one_unchanged_config() -> None:
    """A block without ``ablations:`` behaves exactly as it did before."""
    configs = expand_pisam_milp_ablations(_BASE)
    assert len(configs) == 1, configs
    assert configs[0] == PisamMilpConfig.from_dict(_BASE)

    # And an absent block is still the default config, not an empty list — the
    # MILP arms have to be runnable with no `pisam_milp:` at all.
    assert expand_pisam_milp_ablations(None) == [PisamMilpConfig()]
    assert expand_pisam_milp_ablations({}) == [PisamMilpConfig()]


def test_ablation_is_a_cross_product() -> None:
    """Listed knobs multiply; unlisted ones keep the surrounding block's value."""
    raw = dict(_BASE, ablations={
        "pool_policy": ["frozen", "replace"],
        "learner_input": ["subset_only", "accumulated"],
    })
    configs = expand_pisam_milp_ablations(raw)
    assert len(configs) == 4, [c.pool_policy for c in configs]
    assert {(c.pool_policy, c.learner_input) for c in configs} == {
        (PoolPolicy.FROZEN, LearnerInput.SUBSET_ONLY),
        (PoolPolicy.FROZEN, LearnerInput.ACCUMULATED),
        (PoolPolicy.REPLACE, LearnerInput.SUBSET_ONLY),
        (PoolPolicy.REPLACE, LearnerInput.ACCUMULATED),
    }
    # An unlisted knob is not touched by the expansion.
    assert {c.seed for c in configs} == {42}


def test_ablation_overrides_rather_than_extends() -> None:
    """A listed knob's values are the complete set, base value included or not.

    The base says ``eq16: off``; the ablation says ``[on]``. Treating the list as
    an addition would run an unnamed second arm, which is the failure mode this
    whole feature exists to prevent.
    """
    configs = expand_pisam_milp_ablations(dict(_BASE, ablations={"eq16": [True]}))
    assert len(configs) == 1, configs
    assert configs[0].eq16 is True


def test_loop_only_knobs_collapse_under_single_round() -> None:
    """Crossing a loop-only knob costs the single-round arm nothing.

    ``pool_policy`` is read only by the loop, so the two combinations are the
    same solve for SR. Without dedup this would run it twice *and* file both
    under one label.
    """
    raw = dict(_BASE, ablations={"pool_policy": ["frozen", "replace"]})
    assert _labels(PISAM_MILP_LOOP, raw) == [
        "PISAM_MILP_LOOP", "PISAM_MILP_LOOP__pool=replace",
    ]
    assert _labels(PISAM_MILP_SINGLE_ROUND, raw) == ["PISAM_MILP_SR"]


def test_shared_knob_splits_both_arms() -> None:
    """A knob both variants read produces one arm per value on each."""
    raw = dict(_BASE, ablations={"eq16": [False, True]})
    assert _labels(PISAM_MILP_LOOP, raw) == [
        "PISAM_MILP_LOOP", "PISAM_MILP_LOOP__eq16=0.4",
    ]
    assert _labels(PISAM_MILP_SINGLE_ROUND, raw) == [
        "PISAM_MILP_SR", "PISAM_MILP_SR__eq16=0.4",
    ]


def test_variant_comes_from_the_key_not_the_config() -> None:
    """Both arms run off one block; each pins its own variant."""
    configs = expand_pisam_milp_ablations(_BASE)
    assert milp_configs_for(PISAM_MILP_LOOP, configs)[0].variant is MilpVariant.LOOP
    assert (milp_configs_for(PISAM_MILP_SINGLE_ROUND, configs)[0].variant
            is MilpVariant.SINGLE_ROUND)


def test_label_collision_is_rejected() -> None:
    """Distinct arms that share a results row fail before the run, not after.

    ``seed`` changes the sampler and so the model, but does not enter the label:
    the two arms would be averaged into one ``PISAM_MILP_LOOP`` row.
    """
    configs = expand_pisam_milp_ablations(dict(_BASE, ablations={"seed": [1, 2]}))
    assert len(configs) == 2, "the expansion itself is fine; selection is not"
    try:
        milp_configs_for(PISAM_MILP_LOOP, configs)
    except ValueError as err:
        assert "PISAM_MILP_LOOP" in str(err), err
    else:
        raise AssertionError("two same-label arms were accepted")


def test_unablatable_and_malformed_blocks_are_rejected() -> None:
    """Every way of writing a meaningless ``ablations:`` block is named."""
    cases = {
        "unknown key": {"no_such_knob": [1, 2]},
        "nested block": {"stop": [{"max_rounds": 1}]},
        "variant": {"variant": ["loop", "single_round"]},
        "scalar not list": {"pool_policy": "replace"},
        "empty list": {"pool_policy": []},
        "not a mapping": ["pool_policy"],
    }
    for name, ablations in cases.items():
        try:
            expand_pisam_milp_ablations(dict(_BASE, ablations=ablations))
        except (ValueError, TypeError):
            continue
        raise AssertionError(f"{name!r} ablation block was accepted")


def test_work_subdir_matches_the_label_and_keeps_the_old_default() -> None:
    """Ablated arms get suffixed dirs; a plain run keeps the path readers know.

    ``backfill_cdps`` and the anytime harness both look for the bare names, so
    the default must not move.
    """
    default = expand_pisam_milp_ablations(_BASE)[0]
    assert milp_work_subdir(PISAM_MILP_LOOP, default) == "pisam_milp_loop"
    assert (milp_work_subdir(PISAM_MILP_SINGLE_ROUND, default)
            == "pisam_milp_single_round")

    ablated = expand_pisam_milp_ablations(
        dict(_BASE, ablations={"pool_policy": ["replace"]}))[0]
    assert (milp_work_subdir(PISAM_MILP_LOOP, ablated)
            == "pisam_milp_loop__pool=replace")

    # Distinct arms must not share a directory, or the second overwrites the
    # first's artifacts inside one fold.
    raw = dict(_BASE, ablations={"pool_policy": ["frozen", "replace"]})
    dirs = [milp_work_subdir(PISAM_MILP_LOOP, c)
            for c in milp_configs_for(PISAM_MILP_LOOP, expand_pisam_milp_ablations(raw))]
    assert len(set(dirs)) == len(dirs), dirs


def test_family_names_announce_every_ablated_arm() -> None:
    """The run banner and ``run_params`` name each arm the results will show."""
    configs = expand_pisam_milp_ablations(
        dict(_BASE, ablations={"pool_policy": ["frozen", "replace"]}))
    names = cdps_family_names(
        run_cdps=True, run_cdps_anchored=False,
        run_pisam_milp=True, run_pisam_milp_loop=True, milp_configs=configs)
    assert names == [
        "CDPS", "PISAM_MILP_SR", "PISAM_MILP_LOOP", "PISAM_MILP_LOOP__pool=replace",
    ], names

    # Called without configs (the CLI path), it is what it always was.
    assert cdps_family_names(True, False, True, True) == [
        "CDPS", "PISAM_MILP_SR", "PISAM_MILP_LOOP",
    ]


if __name__ == "__main__":
    test_no_ablations_is_one_unchanged_config()
    test_ablation_is_a_cross_product()
    test_ablation_overrides_rather_than_extends()
    test_loop_only_knobs_collapse_under_single_round()
    test_shared_knob_splits_both_arms()
    test_variant_comes_from_the_key_not_the_config()
    test_label_collision_is_rejected()
    test_unablatable_and_malformed_blocks_are_rejected()
    test_work_subdir_matches_the_label_and_keeps_the_old_default()
    test_family_names_announce_every_ablated_arm()
    print("ALL TESTS PASSED")
