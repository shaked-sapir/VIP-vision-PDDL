"""Unit tests for the source -> cutter -> emitter composition.

The pieces are tested on their own elsewhere; these cover what only appears once
they are joined -- folder numbering, the two trees, the manifest, and the
failure modes a caller can actually hit.

    ./venv11/bin/python -m pytest src/trace_generation/test_corpus.py -q
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.trace_generation.corpus import (
    Corpus,
    SourceKind,
    build_source,
    generate_corpus,
)
from src.trace_generation.cutter import CutMode
from src.trace_generation.sources import ProblemWalkSource, TrajectorySource

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLOCKS_DOMAIN = PROJECT_ROOT / "src" / "domains" / "blocks" / "blocks.pddl"
BLOCKS_PROBLEM = (PROJECT_ROOT / "src" / "domains" / "blocks" / "problems"
                  / "problem1" / "problem1.pddl")

# gripper problem0 ships a 17-step trajectory beside its 18 state_*.png frames.
GRIPPER_DOMAIN = PROJECT_ROOT / "src" / "domains" / "gripper" / "gripper.pddl"
GRIPPER_DIR = (PROJECT_ROOT / "src" / "domains" / "gripper" / "problems"
               / "problem0")


def _walk_source(**kwargs) -> ProblemWalkSource:
    """A random-only blocksworld walk, so no test here invokes the planner."""
    defaults = dict(domain_file=BLOCKS_DOMAIN, problem_file=BLOCKS_PROBLEM,
                    p_rnd=1.0, seed=7, max_steps=60)
    defaults.update(kwargs)
    return build_source(SourceKind.PROBLEM, **defaults)


def _gripper_source(**kwargs) -> TrajectorySource:
    return build_source(SourceKind.TRAJECTORY,
                        domain_file=GRIPPER_DOMAIN,
                        problem_file=GRIPPER_DIR / "problem0.pddl",
                        trajectory_file=GRIPPER_DIR / "problem0.trajectory",
                        **kwargs)


def _corpus(tmp_path, source=None, **kwargs) -> Corpus:
    defaults = dict(cut_mode=CutMode.UNIFORM, length_range=(4, 7), skip=1,
                    num_problems=5, seed=3)
    defaults.update(kwargs)
    return generate_corpus(source or _walk_source(), corpus_root=tmp_path,
                           **defaults)


# ── build_source ─────────────────────────────────────────────────────────

def test_the_problem_kind_builds_a_walk():
    assert isinstance(_walk_source(), ProblemWalkSource)


def test_the_trajectory_kind_builds_a_replay():
    assert isinstance(_gripper_source(), TrajectorySource)


def test_the_kind_may_be_given_as_its_string_value():
    assert isinstance(build_source("problem", domain_file=BLOCKS_DOMAIN,
                                   problem_file=BLOCKS_PROBLEM),
                      ProblemWalkSource)


def test_walk_knobs_reach_the_source():
    source = _walk_source(p_rnd=0.25, seed=11, max_steps=5, stop_at_goal=True)
    assert (source.p_rnd, source.seed, source.max_steps, source.stop_at_goal) \
        == (0.25, 11, 5, True)


def test_frame_settings_reach_the_replay(tmp_path):
    source = _gripper_source(attach_frames=True, frames_dir=tmp_path)
    assert source.attach_frames is True
    assert source.frames_dir == tmp_path


@pytest.mark.parametrize("kind", ["problem", "trajectory"])
def test_a_missing_problem_file_is_refused(kind):
    with pytest.raises(ValueError, match="requires a problem file"):
        build_source(kind, domain_file=BLOCKS_DOMAIN)


def test_a_missing_trajectory_file_is_refused():
    with pytest.raises(ValueError, match="requires a trajectory file"):
        build_source(SourceKind.TRAJECTORY, domain_file=GRIPPER_DOMAIN,
                     problem_file=GRIPPER_DIR / "problem0.pddl")


def test_a_trajectory_file_handed_to_the_walk_is_refused_not_ignored():
    with pytest.raises(ValueError, match="source_kind=trajectory to replay"):
        build_source(SourceKind.PROBLEM, domain_file=GRIPPER_DOMAIN,
                     problem_file=GRIPPER_DIR / "problem0.pddl",
                     trajectory_file=GRIPPER_DIR / "problem0.trajectory")


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="rollout"):
        build_source("rollout", domain_file=BLOCKS_DOMAIN,
                     problem_file=BLOCKS_PROBLEM)


# ── what lands on disk ───────────────────────────────────────────────────

def test_a_corpus_writes_both_trees_and_the_manifest(tmp_path):
    corpus = _corpus(tmp_path)
    assert corpus.num_problems == 5
    assert corpus.trajectories_dir == tmp_path / "training" / "trajectories"
    assert corpus.info_file == tmp_path / "generation_info.json"
    assert corpus.info_file.exists()


def test_every_problem_folder_holds_exactly_the_pool_files(tmp_path):
    for problem in _corpus(tmp_path).problems:
        names = {p.name for p in problem.problem_dir.iterdir()}
        assert names == {f"{problem.name}.pddl", "plan.txt",
                         f"{problem.name}_trajectory.json"}


def test_every_problem_has_a_matching_gt_folder(tmp_path):
    for problem in _corpus(tmp_path).problems:
        assert problem.gt_dir.name == problem.name
        assert (problem.gt_dir / f"{problem.name}.trajectory").exists()


def test_folders_are_numbered_from_zero_with_the_prefix(tmp_path):
    corpus = _corpus(tmp_path, problem_prefix="task")
    assert [p.name for p in corpus.problems] == \
        ["task0", "task1", "task2", "task3", "task4"]
    assert all(p.problem_dir.name == p.name for p in corpus.problems)


def test_the_emitted_index_matches_the_folder_number(tmp_path):
    assert [p.index for p in _corpus(tmp_path).problems] == [0, 1, 2, 3, 4]


def test_window_lengths_stay_inside_the_requested_range(tmp_path):
    assert all(4 <= p.length <= 7 for p in _corpus(tmp_path).problems)


def test_buckets_mode_cycles_the_requested_lengths(tmp_path):
    corpus = _corpus(tmp_path, cut_mode=CutMode.BUCKETS, length_range=None,
                     buckets=[3, 6], num_problems=4)
    assert [p.length for p in corpus.problems] == [3, 6, 3, 6]


def test_the_whole_trace_becomes_one_problem_under_cut_mode_none(tmp_path):
    corpus = generate_corpus(_walk_source(max_steps=9), corpus_root=tmp_path,
                             cut_mode=CutMode.NONE)
    assert [p.length for p in corpus.problems] == [9]


# ── the manifest ─────────────────────────────────────────────────────────

def test_the_manifest_carries_the_source_description(tmp_path):
    info = json.loads(_corpus(tmp_path).info_file.read_text())
    assert info["source_kind"] == "problem"
    assert Path(info["source_file"]).name == "problem1.pddl"
    assert info["p_rnd"] == 1.0
    assert info["seed"] == 7


def test_the_manifest_carries_the_cut_parameters(tmp_path):
    info = json.loads(_corpus(tmp_path).info_file.read_text())
    assert info["cut_mode"] == "uniform"
    assert info["length_range"] == [4, 7]
    assert info["skip"] == 1
    assert info["cut_seed"] == 3
    assert info["render"] is False


def test_the_walk_seed_and_the_cut_seed_are_recorded_separately(tmp_path):
    info = _corpus(tmp_path, source=_walk_source(seed=11), seed=3).info
    assert (info["seed"], info["cut_seed"]) == (11, 3)


def test_the_manifest_records_a_shortfall_against_what_was_asked(tmp_path):
    """A short trace yields fewer windows; both numbers must survive."""
    info = _corpus(tmp_path, source=_walk_source(max_steps=12),
                   num_problems=5).info
    assert info["num_problems"] == 5
    assert info["num_windows"] < 5
    assert len(info["windows"]) == info["num_windows"]


def test_a_shortfall_is_warned_about(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="src.trace_generation.corpus"):
        _corpus(tmp_path, source=_walk_source(max_steps=12), num_problems=5)
    assert "only yielded" in caplog.text


def test_a_corpus_from_one_problem_has_a_uniform_object_universe(tmp_path):
    info = _corpus(tmp_path).info
    assert info["uniform_object_universe"] is True
    assert info["object_signature"] == ["a:block", "b:block", "c:block", "d:block"]


def test_extra_info_is_merged_into_the_manifest(tmp_path):
    info = _corpus(tmp_path, extra_info={"domain": "blocksworld"}).info
    assert info["domain"] == "blocksworld"
    assert info["cut_mode"] == "uniform"


# ── determinism and failure ──────────────────────────────────────────────

def test_the_same_seeds_reproduce_the_same_corpus(tmp_path):
    first = _corpus(tmp_path / "a")
    second = _corpus(tmp_path / "b")
    assert [p.length for p in first.problems] == [p.length for p in second.problems]
    for a, b in zip(first.problems, second.problems):
        assert (a.problem_dir / f"{a.name}.pddl").read_text() == \
               (b.problem_dir / f"{b.name}.pddl").read_text()


def test_a_different_cut_seed_gives_different_window_lengths(tmp_path):
    assert [p.length for p in _corpus(tmp_path / "a", seed=3).problems] != \
           [p.length for p in _corpus(tmp_path / "b", seed=99).problems]


def test_a_trace_that_yields_no_window_is_refused(tmp_path):
    with pytest.raises(ValueError, match="yielded no window"):
        generate_corpus(_walk_source(max_steps=0), corpus_root=tmp_path,
                        cut_mode=CutMode.NONE)


def test_a_trace_shorter_than_one_window_is_refused(tmp_path):
    with pytest.raises(ValueError, match="yielded no window"):
        _corpus(tmp_path, source=_walk_source(max_steps=3), length_range=(9, 9))


def test_nothing_is_written_when_the_corpus_is_refused(tmp_path):
    with pytest.raises(ValueError):
        generate_corpus(_walk_source(max_steps=0), corpus_root=tmp_path,
                        cut_mode=CutMode.NONE)
    assert not (tmp_path / "generation_info.json").exists()


# ── frames survive the composition ───────────────────────────────────────

def test_a_walk_corpus_has_no_images_even_when_rendering_is_asked_for(tmp_path):
    corpus = _corpus(tmp_path, render=True)
    assert all(not p.rendered for p in corpus.problems)
    assert not list(tmp_path.rglob("*.png"))


def test_replayed_frames_are_copied_out_as_state_images(tmp_path):
    corpus = generate_corpus(_gripper_source(attach_frames=True),
                             corpus_root=tmp_path, cut_mode=CutMode.NONE,
                             render=True)
    problem = corpus.problems[0]
    assert problem.rendered is True
    assert problem.length == 17
    assert len(list(problem.problem_dir.glob("state_*.png"))) == 18


def test_frames_are_not_copied_unless_rendering_is_asked_for(tmp_path):
    corpus = generate_corpus(_gripper_source(attach_frames=True),
                             corpus_root=tmp_path, cut_mode=CutMode.NONE)
    assert corpus.problems[0].rendered is False
    assert not list(tmp_path.rglob("*.png"))
