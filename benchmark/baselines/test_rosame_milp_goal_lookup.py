"""The ROSAME+MILP runner finds a trace's GT final state by its problem file stem.

    python -m pytest benchmark/baselines/test_rosame_milp_goal_lookup.py
"""

from __future__ import annotations

from pathlib import Path

from benchmark.backfill_baseline import _runner_kwargs
from benchmark.baselines import resolve_baselines
from src.milp.converter import GtAnchoring
from benchmark.baselines.rosame_milp_runner import (
    FREE_GOAL_MODE,
    RosameMilpBaseRunner,
    RosameMilpRunner,
    RosameMilpTagRunner,
    goal_fluents_for,
)


def test_stems_come_from_the_workspace_trajectory_files():
    traj_paths = ["/w/problem3/problem3.trajectory", "/w/problem11/problem11.trajectory"]
    assert RosameMilpBaseRunner._problem_stems(traj_paths) == ["problem3", "problem11"]


def test_every_stem_resolves_to_its_original_problem_file():
    prepared_trajectories = [
        (Path("obs/original_observation_problem3.trajectory"), None, Path("data/problem3/problem3.pddl"), set()),
        (Path("obs/original_observation_problem11.trajectory"), None, Path("data/problem11/problem11.pddl"), set()),
    ]
    lookup = RosameMilpBaseRunner._original_problem_paths(prepared_trajectories)
    stems = RosameMilpBaseRunner._problem_stems(
        ["/w/problem3/problem3.trajectory", "/w/problem11/problem11.trajectory"]
    )
    assert [lookup[stem] for stem in stems] == [
        Path("data/problem3/problem3.pddl"), Path("data/problem11/problem11.pddl"),
    ]


DOMAIN = Path("domain_reference.pddl")


def test_the_default_goal_mode_keeps_the_arm_label():
    assert RosameMilpRunner().row_name(DOMAIN) == "ROSAME_MILP_24"
    assert RosameMilpTagRunner().row_name(DOMAIN) == "ROSAME_MILP_24_TAG"
    assert "goal_mode" not in RosameMilpRunner().run_params()


def test_a_free_final_state_gets_its_own_label_on_both_arms():
    assert RosameMilpRunner(goal_mode=FREE_GOAL_MODE).row_name(DOMAIN) == "ROSAME_MILP_24__goal=none"
    assert RosameMilpTagRunner(goal_mode=FREE_GOAL_MODE).row_name(DOMAIN) == "ROSAME_MILP_24_TAG__goal=none"
    assert RosameMilpRunner(goal_mode=FREE_GOAL_MODE).run_params()["goal_mode"] == FREE_GOAL_MODE


def test_a_free_final_state_reads_no_ground_truth():
    assert goal_fluents_for(Path("data/problem3/problem3.pddl"), FREE_GOAL_MODE) is None


def test_the_command_line_options_reach_both_milp_arms_and_no_other():
    import argparse

    args = argparse.Namespace(
        epochs=None, n_seeds=None, ignore_budget=False, budget_mode=None,
        goal_mode=FREE_GOAL_MODE, mip_traces=4,
    )
    kwargs = _runner_kwargs(args)
    assert kwargs == {"goal_mode": FREE_GOAL_MODE, "mip_traces": 4}
    milp, tag, plain = resolve_baselines(
        ["rosame_milp_24", "rosame_milp_24_tag", "rosame_24"], **kwargs
    )
    assert (milp.goal_mode, milp.mip_traces) == (FREE_GOAL_MODE, 4)
    assert (tag.goal_mode, tag.mip_traces) == (FREE_GOAL_MODE, 4)
    assert tag.encoding_config.as_stats() != milp.encoding_config.as_stats()
    assert plain.row_name(DOMAIN) == "ROSAME_24"


def test_nothing_fixed_is_labelled_as_no_ground_truth():
    runner = RosameMilpRunner(goal_mode=FREE_GOAL_MODE, gt_anchoring=GtAnchoring.NONE)
    tag = RosameMilpTagRunner(goal_mode=FREE_GOAL_MODE, gt_anchoring=GtAnchoring.NONE)
    assert runner.row_name(DOMAIN) == "ROSAME_MILP_24__gt=none"
    assert tag.row_name(DOMAIN) == "ROSAME_MILP_24_TAG__gt=none"
    assert runner.run_params()["gt_anchoring"] == "none"


def test_a_free_initial_state_alone_has_its_own_label():
    assert RosameMilpRunner(gt_anchoring=GtAnchoring.NONE).row_name(DOMAIN) == "ROSAME_MILP_24__init=none"
    assert "gt_anchoring" not in RosameMilpRunner().run_params()


def test_the_anchoring_option_reaches_the_milp_arms_only():
    import argparse

    args = argparse.Namespace(
        epochs=None, n_seeds=None, ignore_budget=False, budget_mode=None,
        goal_mode=FREE_GOAL_MODE, milp_gt_anchoring="none",
    )
    kwargs = _runner_kwargs(args)
    assert kwargs == {"goal_mode": FREE_GOAL_MODE, "gt_anchoring": GtAnchoring.NONE}
    milp, tag, plain, nolam, offlam = resolve_baselines(
        ["rosame_milp_24", "rosame_milp_24_tag", "rosame_24", "nolam", "offlam"], **kwargs
    )
    assert milp.gt_anchoring is GtAnchoring.NONE and tag.gt_anchoring is GtAnchoring.NONE
    assert not hasattr(plain, "gt_anchoring")
    assert [r.row_name(DOMAIN) for r in (plain, nolam, offlam)] == ["ROSAME_24", "NOLAM", "OffLAM"]
