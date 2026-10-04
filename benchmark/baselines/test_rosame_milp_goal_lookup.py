"""The ROSAME+MILP runner finds a trace's GT final state by its problem file stem.

    python -m pytest benchmark/baselines/test_rosame_milp_goal_lookup.py
"""

from __future__ import annotations

from pathlib import Path

from benchmark.baselines.rosame_milp_runner import RosameMilpBaseRunner


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
