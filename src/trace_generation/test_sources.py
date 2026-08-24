"""Unit tests for the trace sources.

Tests split by whether they invoke Fast Downward. The planner-free ones cover
the walk's contract and are cheap; the planner-backed ones cover ``p_rnd``'s
effect and the goal-continuation behaviour, and cost a couple of seconds each.

    ./venv11/bin/python -m pytest src/trace_generation/test_sources.py -q
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pytest

from src.trace_generation import sources
from src.trace_generation.cutter import CutMode, cut
from src.trace_generation.emitter import emit_window
from src.trace_generation.sources import (
    ProblemWalkSource,
    TrajectorySource,
    WalkConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLOCKS_DOMAIN = PROJECT_ROOT / "src" / "domains" / "blocks" / "blocks.pddl"
PROBLEMS = PROJECT_ROOT / "src" / "domains" / "blocks" / "problems"

# problem1: four blocks on the table, goal (holding a) — one step away.
PROBLEM1 = PROBLEMS / "problem1" / "problem1.pddl"
# problem10: five blocks, a five-conjunct goal needing a four-step plan.
PROBLEM10 = PROBLEMS / "problem10" / "problem10.pddl"

PROBLEM10_GOAL = {"on(a:block,c:block)", "on(c:block,b:block)",
                  "on(d:block,e:block)", "clear(d:block)", "clear(a:block)"}

# gripper problem0 ships a 17-step trajectory beside its 18 state_*.png frames.
GRIPPER_DOMAIN = PROJECT_ROOT / "src" / "domains" / "gripper" / "gripper.pddl"
GRIPPER_DIR = (PROJECT_ROOT / "src" / "domains" / "gripper" / "problems"
               / "problem0")


def _walk(problem=PROBLEM1, **kwargs) -> ProblemWalkSource:
    """A source over ``problem``, random-only and short unless overridden."""
    defaults = dict(p_rnd=1.0, seed=7, max_steps=8)
    defaults.update(kwargs)
    return ProblemWalkSource(problem, BLOCKS_DOMAIN, walk=WalkConfig(**defaults))


def _actions(source: ProblemWalkSource):
    return [step.action for step in source.steps()]


# ── construction and validation ──────────────────────────────────────────

def test_pddlgym_backend_is_refused_rather_than_silently_ignored():
    with pytest.raises(NotImplementedError, match="pddlgym"):
        WalkConfig(backend="pddlgym")


@pytest.mark.parametrize("p_rnd", [-0.1, 1.1])
def test_p_rnd_outside_the_unit_interval_is_rejected(p_rnd):
    with pytest.raises(ValueError, match="p_rnd"):
        WalkConfig(p_rnd=p_rnd)


def test_a_source_without_a_walk_config_defaults_to_a_random_walk():
    assert ProblemWalkSource(PROBLEM1, BLOCKS_DOMAIN).walk == WalkConfig()


def test_walk_problem_rejects_a_negative_step_cap():
    with pytest.raises(ValueError, match="max_steps"):
        list(sources.walk_problem(None, {}, rng=random.Random(0), p_rnd=1.0,
                                  max_steps=-1))


def test_nothing_is_parsed_until_steps_is_called():
    source = _walk()
    assert source._problem is None
    source.objects
    assert source._problem is not None


def test_describe_records_the_walk_parameters():
    described = _walk(p_rnd=0.25, seed=3, max_steps=5).describe()
    assert described["source_kind"] == "problem"
    assert described["p_rnd"] == 0.25
    assert described["seed"] == 3
    assert described["max_steps"] == 5
    assert described["backend"] == "native"
    assert Path(described["source_file"]).name == "problem1.pddl"


def test_describe_records_every_knob_that_can_change_the_walk():
    described = _walk(preserve_solvability=True, stop_at_goal=True,
                      max_planning_time=17, max_replanning_time=13,
                      max_random_trials=5).describe()
    assert described["preserve_solvability"] is True
    assert described["stop_at_goal"] is True
    assert described["max_planning_time"] == 17
    assert described["max_replanning_time"] == 13
    assert described["max_random_trials"] == 5


def test_describe_records_absolute_paths():
    described = _walk(problem=Path("src/domains/blocks/problems/problem1/problem1.pddl")
                      ).describe()
    assert Path(described["source_file"]).is_absolute()
    assert Path(described["domain_file"]).is_absolute()


def test_domain_name_and_objects_come_from_the_files():
    source = _walk()
    assert source.domain_name == "blocks"
    assert source.objects == ("a:block", "b:block", "c:block", "d:block")


# ── the random walk (no planner) ─────────────────────────────────────────

def test_p_rnd_one_never_invokes_the_planner(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("the planner must not be consulted when p_rnd == 1")

    monkeypatch.setattr(sources, "_plan_from", explode)
    assert len(_actions(_walk(p_rnd=1.0))) == 8


def test_a_random_walk_ignores_the_problem_goal(monkeypatch):
    """problem1's goal is one step away; a pure random walk keeps going."""
    monkeypatch.setattr(sources, "_plan_from", lambda *a, **k: None)
    assert len(_actions(_walk(p_rnd=1.0, max_steps=12))) == 12


def test_max_steps_caps_the_walk():
    assert len(_actions(_walk(max_steps=3))) == 3


def test_max_steps_zero_yields_nothing():
    assert _actions(_walk(max_steps=0)) == []


def test_the_same_seed_reproduces_the_same_walk():
    assert _actions(_walk(seed=11)) == _actions(_walk(seed=11))


def test_a_different_seed_gives_a_different_walk():
    assert _actions(_walk(seed=11)) != _actions(_walk(seed=12))


# ── what the steps look like ─────────────────────────────────────────────

def test_steps_chain_end_to_end():
    steps = list(_walk().steps())
    for earlier, later in zip(steps, steps[1:]):
        assert earlier.next_state == later.prev_state


def test_no_step_is_a_no_op():
    for step in _walk().steps():
        assert step.prev_state.fluent_set() != step.next_state.fluent_set(), \
            f"{step.action} left the state unchanged"


def test_states_and_actions_are_in_the_eval_schema():
    step = next(iter(_walk().steps()))
    assert step.action.split("(")[0] in {"pick_up", "put_down", "stack", "unstack"}
    assert all(":" in literal for literal in step.prev_state.literals
               if not literal.startswith("handempty"))
    assert step.prev_state.objects == ("a:block", "b:block", "c:block", "d:block")


def test_every_state_carries_the_full_object_universe():
    objects = ("a:block", "b:block", "c:block", "d:block")
    for step in _walk().steps():
        assert step.prev_state.objects == objects
        assert step.next_state.objects == objects


def test_a_walk_has_no_frames():
    assert all(not step.has_frames for step in _walk().steps())


def test_a_walked_trace_survives_the_cutter_and_the_real_trajectory_parser(tmp_path):
    """The end the pipeline cares about: a walk must round-trip as a corpus."""
    from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser

    source = _walk(max_steps=6)
    window = cut(source.steps(), mode=CutMode.NONE)[0]
    emitted = emit_window(window, problem_name="problem0", index=0,
                          domain_name=source.domain_name, corpus_root=tmp_path)

    domain = DomainParser(BLOCKS_DOMAIN, partial_parsing=False).parse_domain()
    problem = ProblemParser(emitted.problem_dir / "problem0.pddl", domain).parse_problem()
    observation = TrajectoryParser(domain, problem).parse_trajectory(
        emitted.gt_dir / "problem0.trajectory")

    assert len(observation.components) == 6


# ── the guided walk (planner-backed) ─────────────────────────────────────

def test_a_pure_planner_walk_reaches_the_goal_and_stops():
    steps = list(_walk(PROBLEM10, p_rnd=0.0, max_steps=12, stop_at_goal=True).steps())
    assert len(steps) == 4
    assert PROBLEM10_GOAL <= set(steps[-1].next_state.literals)


def test_a_planner_walk_continues_past_the_goal_when_not_asked_to_stop():
    """The goal is one action away; the walk must still fill max_steps."""
    assert len(_actions(_walk(PROBLEM1, p_rnd=0.0, max_steps=9))) == 9


def test_randomness_diverges_from_the_plan_and_still_reaches_the_goal():
    planned = _actions(_walk(PROBLEM10, p_rnd=0.0, max_steps=16, stop_at_goal=True))
    mixed = list(_walk(PROBLEM10, p_rnd=0.5, max_steps=16, stop_at_goal=True).steps())

    assert [s.action for s in mixed] != planned
    assert len(mixed) > len(planned), "a detour should cost extra steps"
    assert PROBLEM10_GOAL <= set(mixed[-1].next_state.literals), \
        "replanning after a substitution should still land on the goal"


# ── preserve_solvability ─────────────────────────────────────────────────
#
# Blocks is fully reversible, so no random action there can ever be rejected
# and the flag would be untestable on it. `trap` is the smallest domain that
# separates the two settings: `smash` is irreversible and makes the goal
# unreachable, and `wait` is applicable but changes nothing.

TRAP_DOMAIN = """
(define (domain trap)
  (:requirements :strips :typing)
  (:types loc)
  (:predicates (at ?l - loc) (adj ?a - loc ?b - loc) (intact))
  (:action move
    :parameters (?from - loc ?to - loc)
    :precondition (and (at ?from) (adj ?from ?to))
    :effect (and (not (at ?from)) (at ?to)))
  (:action wait
    :parameters (?l - loc)
    :precondition (at ?l)
    :effect (at ?l))
  (:action smash
    :parameters (?l - loc)
    :precondition (at ?l)
    :effect (not (intact)))
)
"""

TRAP_PROBLEM = """
(define (problem trap0)
  (:domain trap)
  (:objects l0 l1 l2 l3 - loc)
  (:init (at l0) (intact)
         (adj l0 l1) (adj l1 l0)
         (adj l1 l2) (adj l2 l1)
         (adj l2 l3) (adj l3 l2)
         (adj l3 l0) (adj l0 l3))
  (:goal (and (at l2) (intact)))
)
"""

# The same problem with the adjacency dropped: smash is then the only action
# that changes anything, and once taken nothing applicable changes the state.
TRAP_DEAD_END = """
(define (problem trap_dead_end)
  (:domain trap)
  (:objects l0 - loc)
  (:init (at l0) (intact))
  (:goal (and (at l0) (intact)))
)
"""

TRAP_SEEDS = range(1, 7)


@pytest.fixture(scope="module")
def trap(tmp_path_factory):
    """The (domain, problem, dead-end problem) files of the trap domain."""
    directory = tmp_path_factory.mktemp("trap")
    domain = directory / "trap.pddl"
    problem = directory / "trap0.pddl"
    dead_end = directory / "trap_dead_end.pddl"
    domain.write_text(TRAP_DOMAIN)
    problem.write_text(TRAP_PROBLEM)
    dead_end.write_text(TRAP_DEAD_END)
    return domain, problem, dead_end


def _trap_walk(trap, *, seed: int, **kwargs):
    """A mostly-random guided walk over the trap domain, stopping at the goal."""
    domain, problem, _ = trap
    defaults = dict(p_rnd=0.99, seed=seed, max_steps=12, stop_at_goal=True)
    defaults.update(kwargs)
    return list(ProblemWalkSource(problem, domain,
                                  walk=WalkConfig(**defaults)).steps())


def _took_the_irreversible_action(steps) -> bool:
    return any(step.action.startswith("smash") for step in steps)


@pytest.fixture(scope="module")
def preserved_walks(trap):
    """One solvability-preserving walk per seed. Each costs several replans."""
    return {seed: _trap_walk(trap, seed=seed, preserve_solvability=True)
            for seed in TRAP_SEEDS}


def test_an_unguarded_walk_does_take_the_irreversible_action(trap):
    """The control: without the flag, smash is reachable and gets taken."""
    assert any(_took_the_irreversible_action(
        _trap_walk(trap, seed=seed, preserve_solvability=False))
        for seed in TRAP_SEEDS)


def test_preserve_solvability_never_takes_the_irreversible_action(preserved_walks):
    for seed, steps in preserved_walks.items():
        assert not _took_the_irreversible_action(steps), \
            f"seed {seed} smashed the goal: {[s.action for s in steps]}"


def test_preserve_solvability_still_reaches_the_goal(preserved_walks):
    for steps in preserved_walks.values():
        assert "at(l2:loc)" in steps[-1].next_state.literals
        assert "intact()" in steps[-1].next_state.literals


def test_a_preserved_substitution_is_never_a_no_op(preserved_walks):
    """`wait` is applicable and solvability-preserving, but changes nothing."""
    for steps in preserved_walks.values():
        for step in steps:
            assert step.prev_state.fluent_set() != step.next_state.fluent_set(), \
                f"{step.action} left the state unchanged"


def test_preserved_steps_chain_end_to_end(preserved_walks):
    for steps in preserved_walks.values():
        for earlier, later in zip(steps, steps[1:]):
            assert earlier.next_state == later.prev_state


def test_zero_random_trials_falls_back_to_the_plan(trap):
    """With no trial budget every substitution fails, leaving the plan intact."""
    planned = _trap_walk(trap, seed=1, p_rnd=0.0)
    starved = _trap_walk(trap, seed=1, preserve_solvability=True,
                         max_random_trials=0)
    assert [s.action for s in starved] == [s.action for s in planned]


# ── truncation is reported, not swallowed ────────────────────────────────

def test_a_dead_end_truncates_the_random_walk_with_a_warning(trap, caplog):
    domain, _, dead_end = trap
    source = ProblemWalkSource(dead_end, domain,
                               walk=WalkConfig(p_rnd=1.0, seed=1, max_steps=9))
    with caplog.at_level(logging.WARNING, logger="src.trace_generation.sources"):
        steps = list(source.steps())

    assert [s.action for s in steps] == ["smash(l0:loc)"]
    assert "Walk truncated after 1 of 9 steps" in caplog.text
    assert "dead end" in caplog.text


def test_an_unsolvable_state_truncates_the_guided_walk_with_a_warning(trap, caplog):
    """Seed 1 smashes on the first substitution, so no plan exists afterwards."""
    with caplog.at_level(logging.WARNING, logger="src.trace_generation.sources"):
        steps = _trap_walk(trap, seed=1, preserve_solvability=False)

    assert _took_the_irreversible_action(steps)
    assert "Walk truncated after 1 of 12 steps" in caplog.text
    assert "found no plan" in caplog.text


def test_a_walk_that_runs_to_the_cap_says_nothing(trap, caplog):
    with caplog.at_level(logging.WARNING, logger="src.trace_generation.sources"):
        assert len(_actions(_walk(PROBLEM1, p_rnd=1.0, max_steps=6))) == 6
    assert caplog.text == ""


# ── TrajectorySource ─────────────────────────────────────────────────────

def _gripper(**kwargs) -> TrajectorySource:
    """A source over gripper problem0's shipped trajectory."""
    return TrajectorySource(GRIPPER_DIR / "problem0.trajectory",
                            GRIPPER_DIR / "problem0.pddl",
                            GRIPPER_DOMAIN, **kwargs)


def test_a_trajectory_source_parses_nothing_until_it_is_asked():
    source = _gripper()
    assert source._problem is None
    source.objects
    assert source._problem is not None


def test_domain_name_and_objects_come_from_the_parsed_files():
    source = _gripper()
    assert source.domain_name == "gripper"
    assert source.objects == (
        "ball1:ball", "ball2:ball", "ball3:ball", "ball4:ball", "ball5:ball",
        "ball6:ball", "left:gripper", "right:gripper", "rooma:room", "roomb:room")


def test_describe_records_the_source_files():
    described = _gripper().describe()
    assert described["source_kind"] == "trajectory"
    assert Path(described["source_file"]).name == "problem0.trajectory"
    assert Path(described["problem_file"]).name == "problem0.pddl"
    assert described["attach_frames"] is False


def test_replayed_steps_are_in_the_eval_schema():
    step = next(iter(_gripper().steps()))
    assert step.action == "pick(ball1:ball, roomb:room, left:gripper)"
    assert "at(ball1:ball,roomb:room)" in step.prev_state.literals


def test_replayed_steps_chain_end_to_end():
    steps = list(_gripper().steps())
    assert len(steps) == 17
    for earlier, later in zip(steps, steps[1:]):
        assert earlier.next_state == later.prev_state


def test_a_replay_has_no_frames_unless_asked():
    assert all(not step.has_frames for step in _gripper().steps())


def test_attaching_frames_pairs_each_step_with_its_two_images():
    steps = list(_gripper(attach_frames=True).steps())
    assert all(step.has_frames for step in steps)
    assert steps[0].frame_before.name == "state_00.png"
    assert steps[0].frame_after.name == "state_01.png"
    assert steps[-1].frame_after.name == "state_17.png"
    for earlier, later in zip(steps, steps[1:]):
        assert earlier.frame_after == later.frame_before


def test_a_frame_count_that_does_not_match_the_steps_is_refused(tmp_path):
    source = _gripper(attach_frames=True, frames_dir=tmp_path)
    (tmp_path / "state_00.png").touch()
    with pytest.raises(ValueError, match="holds 1 .* 17 steps, which needs 18"):
        list(source.steps())


def test_a_walk_survives_being_emitted_and_read_back_as_a_trajectory(tmp_path):
    """The two sources must agree: replaying a walk reproduces it exactly."""
    walk = _walk(max_steps=6)
    walked = list(walk.steps())
    emitted = emit_window(cut(iter(walked), mode=CutMode.NONE)[0],
                          problem_name="problem0", index=0,
                          domain_name=walk.domain_name, corpus_root=tmp_path)

    replayed = list(TrajectorySource(emitted.gt_dir / "problem0.trajectory",
                                     emitted.problem_dir / "problem0.pddl",
                                     BLOCKS_DOMAIN).steps())

    assert replayed == walked
