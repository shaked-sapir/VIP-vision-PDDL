"""The rendering feasibility gate as a test: re-render stored frames and diff.

Every case skips when its reference data is absent, so the suite stays green
on a checkout without the corpora. The references:

    depot, gripper      src/domains/<domain>/problems/problem{1,2,3} (shipped)
    blocksworld, hanoi, npuzzle
                        benchmark/data/<domain>/render_gate_reference_n3/,
                        three windows of the retired PDDLGym walk

``VIP_RENDER_GATE_DATA`` points the PDDLGym references at another checkout's
``benchmark/data``; ``VIP_PLANNING_VISUALIZER`` at the visualizer checkout.

    ./venv11/bin/python -m pytest benchmark/test_render_gate.py -q
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List

import pytest

from benchmark.render_gate import format_report, run_gate
from src.trace_generation.renderer_registry import visualizer_root_from_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("VIP_RENDER_GATE_DATA", PROJECT_ROOT / "benchmark" / "data"))

GYM_DOMAINS = {
    "blocksworld": PROJECT_ROOT / "src/domains/blocks/blocks.pddl",
    "hanoi": PROJECT_ROOT / "src/domains/hanoi/hanoi.pddl",
    "npuzzle": PROJECT_ROOT / "src/domains/n_puzzle/n_puzzle.pddl",
}
VISUALIZER_DOMAINS = {
    "depot": PROJECT_ROOT / "src/domains/depot/depot.pddl",
    "gripper": PROJECT_ROOT / "src/domains/gripper/gripper.pddl",
}


def _existing(paths: List[Path]) -> List[Path]:
    found = [p for p in paths if p.is_dir()]
    if len(found) != len(paths):
        pytest.skip(f"reference data absent: {[str(p) for p in paths if not p.is_dir()]}")
    return found


def _assert_identical(domain: str, results) -> None:
    assert all(r.identical for r in results), format_report(domain, results)


def _gym_references(domain: str) -> List[Path]:
    return _existing([
        DATA / domain / "render_gate_reference_n3" / "training" / "image_trajectories" / f"problem{i}"
        for i in range(3)])


@pytest.mark.parametrize("domain", ["hanoi", "npuzzle"])
def test_pddlgym_windows_reproduce_the_old_walk_frames(domain):
    _assert_identical(domain, run_gate(domain, GYM_DOMAINS[domain], _gym_references(domain)))


def test_blocksworld_reproduces_the_old_walk_up_to_colour_assignment_order():
    """Primed as the old walk's first frame primed them, the frames are identical."""
    references = _gym_references("blocksworld")
    _assert_identical("blocksworld", run_gate(
        "blocksworld", GYM_DOMAINS["blocksworld"], references, blocks_colors="encounter"))


def test_blocksworld_sorted_priming_matches_the_classifier_colour_map():
    """a red, b cyan, c blue, d green, e yellow: LLMBlocksFluentClassifier's map."""
    from pddlgym.rendering import blocks

    from src.trace_generation.renderer_registry import build_renderer
    from src.trace_generation.trace_step import TraceState

    blocks._block_name_to_color.clear()
    blocks._rng.seed(0)
    state = TraceState(
        tuple(f"ontable({b}:block)" for b in "edcba") + tuple(f"clear({b}:block)" for b in "abcde")
        + ("handempty()",),
        tuple(f"{b}:block" for b in "abcde"))
    build_renderer("blocksworld").render(state, step=0, action=None)
    expected = {"a": (0.9, 0.1, 0.1), "b": (0.44, 0.89, 0.96), "c": (0.16, 0.11, 0.66),
                "d": (0.15, 0.87, 0.16), "e": (0.95, 0.73, 0.25)}
    got = {k.name: tuple(round(float(x), 2) for x in v)
           for k, v in blocks._block_name_to_color.items()}
    assert got == expected


@pytest.mark.parametrize("domain", sorted(VISUALIZER_DOMAINS))
def test_visualizer_problems_reproduce_the_shipped_frames(domain):
    if shutil.which("npx") is None:
        pytest.skip("npx is not installed; the canvas worker needs Node")
    root = visualizer_root_from_config()
    if not (root / "backend" / "api" / "render_states.ts").is_file():
        pytest.skip(f"no planning-visualizer canvas worker under {root}")
    references = _existing([
        PROJECT_ROOT / "src" / "domains" / domain / "problems" / f"problem{i}"
        for i in (1, 2, 3)])
    _assert_identical(domain, run_gate(domain, VISUALIZER_DOMAINS[domain], references,
                                       visualizer_root=root))
