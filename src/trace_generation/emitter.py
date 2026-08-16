"""Write a cut window out as one problem folder, and the corpus manifest.

Layout produced, which is what a simulated experiment consumes:

    <corpus>/training/trajectories/problemN/problemN.pddl   pool + test problems
    <corpus>/training/trajectories/problemN/plan.txt
    <corpus>/training/trajectories/problemN/problemN_trajectory.json
    <corpus>/training/trajectories/problemN/state_*.png     only when rendering
    <corpus>/gt_trajectories/problemN/problemN.trajectory   discovered by the runner
    <corpus>/gt_trajectories/problemN/problemN_trajectory.json
    <corpus>/generation_info.json

``export_gt=False`` suppresses the ``gt_trajectories/`` export, for sources whose
states are not yet in the eval schema and must be translated downstream.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.trace_generation.cutter import Window
from src.utils.pddl_gym import parse_gym_to_pddl_ground_action, parse_gym_to_pddl_literal
from src.utils.pddl_trajectory import export_gt_trajectory

POOL_SUBPATH = Path("training") / "trajectories"
GT_SUBPATH = Path("gt_trajectories")
GENERATION_INFO_NAME = "generation_info.json"

# pddlgym.parser.FAST_DOWNWARD_STR, reproduced so a symbolic corpus can be
# emitted for a domain pddlgym has never heard of.
_PROBLEM_TEMPLATE = (
    "\n(define (problem {problem}) (:domain {domain})\n"
    "  (:objects\n        {objects}\n  )\n"
    "  (:init \n\t{init_state}\n  )\n"
    "  (:goal {goal})\n)\n"
)


@dataclass(frozen=True)
class EmittedProblem:
    """Where one window landed on disk, and what it was."""

    name: str
    index: int
    length: int
    problem_dir: Path
    objects: Tuple[str, ...]
    gt_dir: Optional[Path] = None
    source: Optional[str] = None
    rendered: bool = False


def emit_window(
    window: Window,
    *,
    problem_name: str,
    index: int,
    domain_name: str,
    corpus_root: Path,
    export_gt: bool = True,
    render: bool = False,
    frame_index_format: str = "06d",
    source: Optional[str] = None,
) -> EmittedProblem:
    """Write one window as a problem folder and return where it landed.

    Args:
        window: The cut window. Its steps must carry eval-schema strings.
        problem_name: Folder and problem name, e.g. ``"problem3"``.
        index: The window's 0-based position in the corpus.
        domain_name: The ``(:domain ...)`` name to reference.
        corpus_root: The corpus directory; both trees are created under it.
        export_gt: Write ``gt_trajectories/`` too. Turn off for sources whose
            states need schema translation before they can be called GT.
        render: Copy the window's frames as ``state_*.png``.
        frame_index_format: Format spec for the ``state_*.png`` index.
        source: The input file this window came from, recorded in the manifest.

    Returns:
        An :class:`EmittedProblem` describing the folder that was written.
    """
    corpus_root = Path(corpus_root)
    problem_dir = corpus_root / POOL_SUBPATH / problem_name
    problem_dir.mkdir(parents=True, exist_ok=True)

    steps_json = window_to_trajectory_json(window)

    _write_problem_pddl(window, problem_name, domain_name, problem_dir)
    _write_plan(window, problem_dir)
    _write_trajectory_json(steps_json, problem_name, problem_dir)

    gt_dir: Optional[Path] = None
    if export_gt:
        gt_dir = export_gt_trajectory(
            steps_json, problem_name, corpus_root / GT_SUBPATH / problem_name)

    rendered = False
    if render:
        rendered = _write_images(window, problem_dir, frame_index_format)

    return EmittedProblem(
        name=problem_name,
        index=index,
        length=window.length,
        problem_dir=problem_dir,
        objects=window.objects,
        gt_dir=gt_dir,
        source=source,
        rendered=rendered,
    )


# ── Trajectory JSON ──────────────────────────────────────────────────────

def window_to_trajectory_json(window: Window) -> List[Dict[str, Any]]:
    """Convert a window into the eval-schema step dicts every consumer reads."""
    return [
        {
            "step": i,
            "current_state": {
                "literals": list(step.prev_state.literals),
                "objects": list(step.prev_state.objects),
            },
            "ground_action": step.action,
            "next_state": {
                "literals": list(step.next_state.literals),
                "objects": list(step.next_state.objects),
            },
        }
        for i, step in enumerate(window.steps, start=1)
    ]


def _write_trajectory_json(steps_json: List[Dict[str, Any]], problem_name: str,
                           problem_dir: Path) -> Path:
    """Write the window's trajectory JSON into its problem folder."""
    path = problem_dir / f"{problem_name}_trajectory.json"
    with open(path, "w") as f:
        json.dump(steps_json, f, indent=4)
    return path


# ── Problem PDDL ─────────────────────────────────────────────────────────

def _write_problem_pddl(window: Window, problem_name: str, domain_name: str,
                        problem_dir: Path) -> Path:
    """Write init = the window's first state, goal = its full final state."""
    init_literals = window.steps[0].prev_state.literals
    goal_literals = window.steps[-1].next_state.literals

    objects_block = "\n\t".join(
        sorted(obj.replace(":", " - ") for obj in window.objects))
    init_block = "\n\t".join(
        parse_gym_to_pddl_literal(lit) for lit in sorted(init_literals))
    goal_block = "(and\n\t{})".format("\n\t".join(
        parse_gym_to_pddl_literal(lit) for lit in sorted(goal_literals)))

    path = problem_dir / f"{problem_name}.pddl"
    path.write_text(_PROBLEM_TEMPLATE.format(
        problem=problem_name,
        domain=domain_name,
        objects=objects_block,
        init_state=init_block,
        goal=goal_block,
    ))
    return path


# ── plan.txt ─────────────────────────────────────────────────────────────

def _write_plan(window: Window, problem_dir: Path) -> Path:
    """Write the window's ground actions, one per line, in PDDL form."""
    lines = [parse_gym_to_pddl_ground_action(step.action) for step in window.steps]
    path = problem_dir / "plan.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


# ── Images ───────────────────────────────────────────────────────────────

def _write_images(window: Window, problem_dir: Path,
                  frame_index_format: str) -> bool:
    """Copy the window's frames as state_0..state_k.png. False when it has none."""
    if not all(step.has_frames for step in window.steps):
        return False

    frames = [window.steps[0].frame_before] + [s.frame_after for s in window.steps]
    for idx, frame in enumerate(frames):
        shutil.copy(frame, problem_dir / f"state_{idx:{frame_index_format}}.png")
    return True


# ── Corpus manifest ──────────────────────────────────────────────────────

def build_generation_info(
    emitted: Sequence[EmittedProblem],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the corpus manifest from the emitted problems and run params.

    Records whether the corpus has a single object universe, which ROSAME-I
    requires, so the constraint can be checked instead of assumed.
    """
    signatures = {tuple(sorted(p.objects)) for p in emitted}
    uniform = len(signatures) <= 1

    windows: List[Dict[str, Any]] = []
    for problem in emitted:
        entry: Dict[str, Any] = {
            "name": problem.name,
            "index": problem.index,
            "length": problem.length,
        }
        if problem.source is not None:
            entry["source"] = problem.source
        if not uniform:
            entry["objects"] = sorted(problem.objects)
        windows.append(entry)

    info: Dict[str, Any] = dict(params)
    info["uniform_object_universe"] = uniform
    if uniform and signatures:
        info["object_signature"] = sorted(next(iter(signatures)))
    info["num_windows"] = len(windows)
    info["windows"] = windows
    return info


def write_generation_info(corpus_root: Path, info: Dict[str, Any]) -> Path:
    """Write ``generation_info.json`` at the corpus root."""
    path = Path(corpus_root) / GENERATION_INFO_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return path
