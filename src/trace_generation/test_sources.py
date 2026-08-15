"""Unit tests for the trace sources.

Tests split by whether they invoke Fast Downward. The planner-free ones cover
the walk's contract and are cheap; the planner-backed ones cover ``p_rnd``'s
effect and the goal-continuation behaviour, and cost a couple of seconds each.

    ./venv11/bin/python -m pytest src/trace_generation/test_sources.py -q
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from src.trace_generation import sources
from src.trace_generation.cutter import CutMode, cut
from src.trace_generation.sources import DEFAULT_MAX_STEPS, ProblemWalkSource

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLOCKS_DOMAIN = PROJECT_ROOT / "src" / "domains" / "blocks" / "blocks.pddl"
PROBLEMS = PROJECT_ROOT / "src" / "domains" / "blocks" / "problems"

# problem1: four blocks on the table, goal (holding a) — one step away.
PROBLEM1 = PROBLEMS / "problem1" / "problem1.pddl"
# problem10: five blocks, a five-conjunct goal needing a four-step plan.
PROBLEM10 = PROBLEMS / "problem10" / "problem10.pddl"

PROBLEM10_GOAL = {"on(a:block,c:block)", "on(c:block,b:block)",
                  "on(d:block,e:block)", "clear(d:block)", "clear(a:block)"}


def _walk(problem=PROBLEM1, **kwargs) -> ProblemWalkSource:
    """A source over ``problem``, random-only and short unless overridden."""
    defaults = dict(p_rnd=1.0, seed=7, max_steps=8)
    defaults.update(kwargs)
    return ProblemWalkSource(problem, BLOCKS_DOMAIN, **defaults)


def _actions(source: ProblemWalkSource):
    return [step.action for step in source.steps()]


# ── construction and validation ──────────────────────────────────────────

def test_pddlgym_backend_is_refused_rather_than_silently_ignored():
    with pytest.raises(NotImplementedError, match="pddlgym"):
        ProblemWalkSource(PROBLEM1, BLOCKS_DOMAIN, backend="pddlgym")


@pytest.mark.parametrize("p_rnd", [-0.1, 1.1])
def test_p_rnd_outside_the_unit_interval_is_rejected(p_rnd):
    with pytest.raises(ValueError, match="p_rnd"):
        ProblemWalkSource(PROBLEM1, BLOCKS_DOMAIN, p_rnd=p_rnd)


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

    from src.trace_generation.emitter import emit_window

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
