"""Unit tests for the window emitter.

The strongest tests here parse the emitted artifacts back with the same parsers
the experiment pipeline uses, so a corpus that these tests accept is a corpus a
run can consume.

    ./venv11/bin/python -m pytest src/trace_generation/test_emitter.py -q
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser

from src.trace_generation.cutter import CutMode, cut
from src.trace_generation.emitter import (
    EmittedProblem,
    build_generation_info,
    emit_window,
    write_generation_info,
)
from src.trace_generation.trace_step import TraceState, TraceStep

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLOCKS_DOMAIN = PROJECT_ROOT / "src" / "domains" / "blocks" / "blocks.pddl"

OBJECTS = ("a:block", "b:block", "c:block")


def _blocks_trace() -> List[TraceStep]:
    """A short, semantically valid blocksworld trace: unstack a, put it down."""
    s0 = TraceState(
        literals=("on(a:block,b:block)", "ontable(b:block)", "ontable(c:block)",
                  "clear(a:block)", "clear(c:block)", "handempty()"),
        objects=OBJECTS,
    )
    s1 = TraceState(
        literals=("holding(a:block)", "ontable(b:block)", "ontable(c:block)",
                  "clear(b:block)", "clear(c:block)"),
        objects=OBJECTS,
    )
    s2 = TraceState(
        literals=("ontable(a:block)", "ontable(b:block)", "ontable(c:block)",
                  "clear(a:block)", "clear(b:block)", "clear(c:block)",
                  "handempty()"),
        objects=OBJECTS,
    )
    return [
        TraceStep(prev_state=s0, action="unstack(a:block,b:block)", next_state=s1),
        TraceStep(prev_state=s1, action="put_down(a:block)", next_state=s2),
    ]


def _window(steps: List[TraceStep] = None):
    """One window spanning the whole (default blocksworld) trace."""
    return cut(steps if steps is not None else _blocks_trace(), mode=CutMode.NONE)[0]


def _emit(tmp_path: Path, **kwargs) -> EmittedProblem:
    """Emit the default window into ``tmp_path`` as ``problem0``."""
    defaults = dict(problem_name="problem0", index=0, domain_name="blocks",
                    corpus_root=tmp_path)
    defaults.update(kwargs)
    window = defaults.pop("window", None) or _window()
    return emit_window(window, **defaults)


# ── layout ───────────────────────────────────────────────────────────────

def test_emits_the_pool_and_gt_trees_a_simulated_run_consumes(tmp_path):
    emitted = _emit(tmp_path)

    pool = tmp_path / "training" / "trajectories" / "problem0"
    gt = tmp_path / "gt_trajectories" / "problem0"

    assert (pool / "problem0.pddl").exists()
    assert (pool / "plan.txt").exists()
    assert (pool / "problem0_trajectory.json").exists()
    assert (gt / "problem0.trajectory").exists()
    assert (gt / "problem0_trajectory.json").exists()

    assert emitted.problem_dir == pool
    assert emitted.gt_dir == gt
    assert emitted.length == 2
    assert emitted.objects == OBJECTS


def test_export_gt_false_writes_the_pool_tree_only(tmp_path):
    emitted = _emit(tmp_path, export_gt=False)
    assert (tmp_path / "training" / "trajectories" / "problem0" / "problem0.pddl").exists()
    assert not (tmp_path / "gt_trajectories").exists()
    assert emitted.gt_dir is None


# ── problem PDDL ─────────────────────────────────────────────────────────

def test_emitted_problem_parses_against_the_real_domain(tmp_path):
    _emit(tmp_path)
    domain = DomainParser(BLOCKS_DOMAIN, partial_parsing=False).parse_domain()
    problem = ProblemParser(
        tmp_path / "training" / "trajectories" / "problem0" / "problem0.pddl",
        domain,
    ).parse_problem()

    assert set(problem.objects) == {"a", "b", "c"}
    assert all(o.type.name == "block" for o in problem.objects.values())


def test_init_is_the_windows_first_state(tmp_path):
    _emit(tmp_path)
    domain = DomainParser(BLOCKS_DOMAIN, partial_parsing=False).parse_domain()
    problem = ProblemParser(
        tmp_path / "training" / "trajectories" / "problem0" / "problem0.pddl",
        domain,
    ).parse_problem()

    init = {p.untyped_representation
            for grounded in problem.initial_state_predicates.values()
            for p in grounded}
    assert init == {"(on a b)", "(ontable b)", "(ontable c)",
                    "(clear a)", "(clear c)", "(handempty )"}


def test_goal_is_the_windows_full_final_state(tmp_path):
    _emit(tmp_path)
    text = (tmp_path / "training" / "trajectories" / "problem0" / "problem0.pddl").read_text()
    goal = text.split("(:goal")[1]
    for literal in ["(ontable a)", "(ontable b)", "(ontable c)",
                    "(clear a)", "(clear b)", "(clear c)", "(handempty)"]:
        assert literal in goal, f"{literal} missing from the goal"


def test_objects_block_carries_pddl_types(tmp_path):
    _emit(tmp_path)
    text = (tmp_path / "training" / "trajectories" / "problem0" / "problem0.pddl").read_text()
    objects_block = text.split("(:objects")[1].split(")")[0]
    assert "a - block" in objects_block
    assert ":" not in objects_block, "eval-schema colons must not reach the .pddl"


# ── plan.txt ─────────────────────────────────────────────────────────────

def test_plan_lists_the_windows_ground_actions_in_order(tmp_path):
    _emit(tmp_path)
    plan = (tmp_path / "training" / "trajectories" / "problem0" / "plan.txt").read_text()
    assert plan.splitlines() == ["(unstack a b)", "(put_down a)"]


# ── trajectory artifacts ─────────────────────────────────────────────────

def test_trajectory_json_is_the_eval_schema(tmp_path):
    _emit(tmp_path)
    path = tmp_path / "training" / "trajectories" / "problem0" / "problem0_trajectory.json"
    steps = json.loads(path.read_text())

    assert [s["step"] for s in steps] == [1, 2]
    assert steps[0]["ground_action"] == "unstack(a:block,b:block)"
    assert steps[0]["next_state"]["literals"] == steps[1]["current_state"]["literals"]
    assert steps[0]["current_state"]["objects"] == list(OBJECTS)


def test_gt_trajectory_parses_back_into_the_window_it_came_from(tmp_path):
    _emit(tmp_path)
    domain = DomainParser(BLOCKS_DOMAIN, partial_parsing=False).parse_domain()
    problem = ProblemParser(
        tmp_path / "training" / "trajectories" / "problem0" / "problem0.pddl",
        domain,
    ).parse_problem()
    observation = TrajectoryParser(domain, problem).parse_trajectory(
        tmp_path / "gt_trajectories" / "problem0" / "problem0.trajectory")

    assert len(observation.components) == 2
    assert str(observation.components[0].grounded_action_call).strip() == "(unstack a b)"
    assert str(observation.components[1].grounded_action_call).strip() == "(put_down a)"


def test_gt_and_pool_trajectory_json_agree(tmp_path):
    _emit(tmp_path)
    pool = json.loads((tmp_path / "training" / "trajectories" / "problem0"
                       / "problem0_trajectory.json").read_text())
    gt = json.loads((tmp_path / "gt_trajectories" / "problem0"
                     / "problem0_trajectory.json").read_text())
    assert pool == gt


# ── images ───────────────────────────────────────────────────────────────

def _framed_trace(tmp_path: Path) -> List[TraceStep]:
    """The blocksworld trace with three placeholder frames attached."""
    frames = []
    for i in range(3):
        frame = tmp_path / f"frame_{i}.png"
        frame.write_bytes(b"png-" + str(i).encode())
        frames.append(frame)

    steps = _blocks_trace()
    return [
        TraceStep(prev_state=s.prev_state, action=s.action, next_state=s.next_state,
                  frame_before=frames[i], frame_after=frames[i + 1])
        for i, s in enumerate(steps)
    ]


def test_render_copies_one_frame_per_state(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    window = _window(_framed_trace(source))
    emitted = _emit(tmp_path / "corpus", window=window, render=True)

    pool = emitted.problem_dir
    assert sorted(p.name for p in pool.glob("state_*.png")) == [
        "state_000000.png", "state_000001.png", "state_000002.png"]
    assert (pool / "state_000000.png").read_bytes() == b"png-0"
    assert (pool / "state_000002.png").read_bytes() == b"png-2"
    assert emitted.rendered is True


def test_render_on_a_symbolic_window_writes_no_images(tmp_path):
    emitted = _emit(tmp_path, render=True)
    assert list(emitted.problem_dir.glob("state_*.png")) == []
    assert emitted.rendered is False


def test_frames_are_not_copied_unless_render_is_asked_for(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    window = _window(_framed_trace(source))
    emitted = _emit(tmp_path / "corpus", window=window, render=False)
    assert list(emitted.problem_dir.glob("state_*.png")) == []
    assert emitted.rendered is False


# ── generation_info ──────────────────────────────────────────────────────

def _problem(name: str, index: int, length: int, objects) -> EmittedProblem:
    return EmittedProblem(name=name, index=index, length=length,
                          problem_dir=Path(name), objects=tuple(objects))


def test_generation_info_records_every_window_with_its_length(tmp_path):
    info = build_generation_info(
        [_problem("problem0", 0, 4, OBJECTS), _problem("problem1", 1, 9, OBJECTS)],
        {"cut_mode": "buckets", "buckets": [4, 9]},
    )
    assert info["cut_mode"] == "buckets"
    assert info["num_windows"] == 2
    assert [w["length"] for w in info["windows"]] == [4, 9]
    assert [w["name"] for w in info["windows"]] == ["problem0", "problem1"]


def test_generation_info_flags_a_uniform_object_universe():
    info = build_generation_info(
        [_problem("problem0", 0, 4, OBJECTS), _problem("problem1", 1, 4, OBJECTS)], {})
    assert info["uniform_object_universe"] is True
    assert info["object_signature"] == sorted(OBJECTS)
    assert "objects" not in info["windows"][0]


def test_generation_info_flags_a_mixed_object_universe():
    info = build_generation_info(
        [_problem("problem0", 0, 4, OBJECTS),
         _problem("problem1", 1, 4, ("a:block", "b:block"))], {})
    assert info["uniform_object_universe"] is False
    assert info["windows"][1]["objects"] == ["a:block", "b:block"]


def test_generation_info_lands_at_the_corpus_root(tmp_path):
    path = write_generation_info(tmp_path, {"seed": 7, "windows": []})
    assert path == tmp_path / "generation_info.json"
    assert json.loads(path.read_text())["seed"] == 7
