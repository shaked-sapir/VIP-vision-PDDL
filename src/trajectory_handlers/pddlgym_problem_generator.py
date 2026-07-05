"""Generate many distinct, solvable PDDLGym problems from a bundled problem.

This module produces problem folders that are shaped exactly like the external
(depot/gripper) domains — each folder contains a `.pddl` problem file, a GT
`.trajectory`, a `_trajectory.json`, a `plan.txt`, and a sequence of
`state_*.png` images — so they can be fed straight into the existing
external-style LLM inference pipeline.

Flow (ROSAME-style):
    1. Select one of PDDLGym's own bundled problems via `fix_problem_index(N)`.
       The chosen problem fixes the object count; its goal is ignored.
    2. Run ONE long random walk of state-changing actions (no no-ops), rendering
       every visited state.
    3. Cut the walk into non-overlapping windows whose lengths are sampled from a
       configurable range. Each window becomes one problem: init = first state,
       goal = full final state, plan = the window's ground actions.

The class composes a PDDLGymImageTrajectoryHandler so it can reuse rendering,
state-changing action sampling, and trajectory-step construction rather than
duplicating them.
"""

import random
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from pddlgym.parser import PDDLProblemParser
from pddlgym.structs import LiteralConjunction, State

from src.trajectory_handlers.pddlgym_trajectory_handler import PDDLGymImageTrajectoryHandler
from src.typings import TrajectoryStep
from src.utils.pddl_gym import parse_gym_to_pddl_ground_action


class _GymWalkHandler(PDDLGymImageTrajectoryHandler):
    """Concrete handler used only for its gym rendering/stepping helpers.

    The generator never runs inference, so the abstract inference hooks are
    stubbed out. This exists so we can compose the gym helpers without pulling
    in an LLM detector/classifier.
    """

    def init_visual_components(self, *args, **kwargs) -> None:  # noqa: D401
        raise NotImplementedError(
            "_GymWalkHandler is generation-only and has no visual components.")

    def run_pipeline(self, *args, **kwargs):  # noqa: D401
        raise NotImplementedError(
            "_GymWalkHandler is generation-only; use PDDLGymProblemGenerator.generate.")


@dataclass
class WalkStep:
    """One recorded step of the long random walk."""
    prev_obs: State
    action: object            # pddlgym Literal (ground action)
    next_obs: State
    frame_before: Path        # rendered image of prev_obs
    frame_after: Path         # rendered image of next_obs


def _natural_sort_key(fname: str):
    """Sort key that orders problem1, problem2, ..., problem10 numerically."""
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', fname)]


class PDDLGymProblemGenerator:
    """Generate distinct solvable problems from a bundled PDDLGym problem.

    Args:
        gym_domain_name: PDDLGym env id (e.g. "PDDLEnvBlocks_operator_actions-v0").
        problem_index: 0-based position into the env's bundled problems sorted in
            natural (numeric) order — position 0 is problem1, position 1 is
            problem2, etc. This mirrors legacy mode's problem ordering so the same
            index selects the same problem in both modes (PDDLGym's raw
            env.problems order is lexicographic, e.g. problem1, problem10, ...,
            which is why we re-sort here). The chosen problem fixes the object
            count; its goal is ignored.
        trajectory_size_limit: Upper bound on total walk length (safety cap).
    """

    def __init__(self, gym_domain_name: str, problem_index: int = 0,
                 trajectory_size_limit: int = 100000) -> None:
        self.gym_domain_name = gym_domain_name
        self.problem_index = problem_index

        # Handler wraps the PDDLGym env — reuses rendering + stepping helpers.
        self.handler = _GymWalkHandler(
            domain_name=gym_domain_name,
            trajectory_size_limit=trajectory_size_limit,
        )
        self.handler.pddl_env.fix_problem_index(
            self._resolve_env_index(problem_index))

    def _resolve_env_index(self, position: int) -> int:
        """Map a 0-based natural-order position to the env's raw problem index.

        PDDLGym exposes env.problems in lexicographic order (problem1, problem10,
        problem2, ...). We re-sort by natural (numeric) order so that position N
        selects problemN+1 — consistent with legacy mode.
        """
        problems = self.handler.pddl_env.problems
        order = sorted(
            range(len(problems)),
            key=lambda i: _natural_sort_key(Path(problems[i].problem_fname).name),
        )
        if not 0 <= position < len(order):
            raise IndexError(
                f"problem_index {position} out of range for "
                f"{self.gym_domain_name} ({len(order)} problems available).")
        return order[position]

    # ── Public API ───────────────────────────────────────────────────────

    def generate(self, output_dir: Path, num_problems: int,
                 length_range: Tuple[int, int], skip: int = 1,
                 seed: Optional[int] = None,
                 problem_prefix: str = "problem",
                 cursor_steps_limit: int = 1000) -> List[Path]:
        """Generate `num_problems` problem folders under `output_dir`.

        Runs a single continuous random walk, cutting it into unique windows on
        the fly: each window is written as a problem folder, and a walk that
        yields a duplicate (init, goal) signature simply keeps going. Stops once
        `num_problems` unique folders exist or the walk reaches
        `cursor_steps_limit` total steps.

        Args:
            output_dir: Directory to write problem<N>/ folders into.
            num_problems: Number of problems to generate.
            length_range: (min_len, max_len) inclusive window length in steps.
            skip: States discarded between consecutive windows.
            seed: RNG seed for reproducibility.
            problem_prefix: Folder/problem name prefix (default "problem").
            cursor_steps_limit: Upper bound on total walked steps (all steps,
                including duplicate windows and skips). Guarantees termination.

        Returns:
            List of generated problem folder paths.
        """
        rng = random.Random(seed)
        if seed is not None:
            # Seed pddlgym's own RNG so the walk (action sampling) is reproducible;
            # rng above only controls window lengths, not the trajectory itself.
            self.handler.pddl_env.action_space.seed(seed)
            self.handler.pddl_env.seed(seed)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scratch_dir = Path(tempfile.mkdtemp(prefix="pddlgym_walk_"))
        try:
            return self._walk_and_write(
                num_problems, length_range, skip, rng, cursor_steps_limit,
                output_dir, problem_prefix, scratch_dir)
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    # ── Streaming walk + write ───────────────────────────────────────────

    def _walk_and_write(self, num_problems: int, length_range: Tuple[int, int],
                        skip: int, rng: random.Random, cursor_steps_limit: int,
                        output_dir: Path, problem_prefix: str,
                        scratch_dir: Path) -> List[Path]:
        """Stream one continuous walk, writing a folder per unique window.

        Walks window-by-window off a single (never-reset) trajectory. Each window
        of `length` steps whose (init, goal) signature is new becomes a problem
        folder; duplicates are discarded but the walked steps are kept, so the
        next window takes their place. Stops when `num_problems` folders exist or
        total walked steps reach `cursor_steps_limit`.
        """
        env = self.handler.pddl_env
        obs, _ = env.reset()

        # Render the initial state as frame 0; `walk` grows as we step.
        self.handler.create_image(scratch_dir, 0)
        walk: List[WalkStep] = []

        generated: List[Path] = []
        seen_signatures = set()
        cursor = 0

        while len(generated) < num_problems and len(walk) < cursor_steps_limit:
            length = rng.randint(*length_range)

            # Extend the walk so the next window [cursor:cursor+length] exists,
            # capping at cursor_steps_limit total steps.
            obs = self._extend_walk(
                walk, obs, cursor + length, cursor_steps_limit, scratch_dir)
            if cursor + length > len(walk):
                break  # hit the step cap mid-window

            window = walk[cursor:cursor + length]
            cursor += length + skip

            signature = self._window_signature(window)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            problem_name = f"{problem_prefix}{len(generated)}"
            problem_dir = output_dir / problem_name
            problem_dir.mkdir(parents=True, exist_ok=True)
            self._write_problem(problem_name, window, problem_dir)
            generated.append(problem_dir)

        if len(generated) < num_problems:
            print(f"  ⚠️  Only generated {len(generated)}/{num_problems} unique "
                  f"problems — hit the {cursor_steps_limit}-step walk cap "
                  f"(state space may be small).")
        return generated

    def _extend_walk(self, walk: List[WalkStep], obs: State, target_len: int,
                     cursor_steps_limit: int, scratch_dir: Path) -> State:
        """Walk state-changing steps until `walk` reaches `target_len` steps.

        Renders each new state and appends a WalkStep. Stops early at
        `cursor_steps_limit`. Returns the current observation.
        """
        while len(walk) < target_len and len(walk) < cursor_steps_limit:
            i = len(walk) + 1  # frame index of the state we're about to reach
            prev_obs = obs
            obs, _done, action = self.handler._sample_state_changing_action(obs)
            self.handler.create_image(scratch_dir, i)
            walk.append(WalkStep(
                prev_obs=prev_obs,
                action=action,
                next_obs=obs,
                frame_before=self._frame_path(scratch_dir, i - 1),
                frame_after=self._frame_path(scratch_dir, i),
            ))
        return obs

    def _frame_path(self, scratch_dir: Path, idx: int) -> Path:
        return Path(scratch_dir) / f"state_{idx:{self.handler.seq_idx_format}}.png"

    def _window_signature(self, window: List[WalkStep]) -> Tuple:
        """A hashable (init, goal) signature for deduplication."""
        init_lits = frozenset(str(l) for l in window[0].prev_obs.literals)
        goal_lits = frozenset(str(l) for l in window[-1].next_obs.literals)
        return (init_lits, goal_lits)

    def _write_problem(self, problem_name: str, window: List[WalkStep],
                       problem_dir: Path) -> None:
        """Write pddl + trajectory json + plan + images for one window."""
        self._write_problem_pddl(problem_name, window, problem_dir)
        self._write_plan(window, problem_dir)
        self._write_images(window, problem_dir)
        self._write_gt_trajectory(problem_name, window, problem_dir)

    # ── Problem PDDL (objects + init + full-state goal) ──────────────────

    def _fluent_literals(self, obs: State) -> List:
        """State literals that are real fluents (drop action-name literals)."""
        action_names = set(self.handler.pddl_env.domain.actions)
        return [lit for lit in obs.literals
                if lit.predicate.name not in action_names]

    def _write_problem_pddl(self, problem_name: str, window: List[WalkStep],
                            problem_dir: Path) -> None:
        """Write the problem file: init = window start, goal = full final state."""
        domain = self.handler.pddl_env.domain
        init_obs = window[0].prev_obs
        goal_obs = window[-1].next_obs

        objects = sorted(init_obs.objects)
        initial_state = self._fluent_literals(init_obs)
        goal = LiteralConjunction(sorted(self._fluent_literals(goal_obs)))

        PDDLProblemParser.create_pddl_file(
            problem_dir / f"{problem_name}.pddl",
            objects=objects,
            initial_state=initial_state,
            problem_name=problem_name,
            domain_name=domain.domain_name,
            goal=goal,
            fast_downward_order=True,
        )

    # ── plan.txt ─────────────────────────────────────────────────────────

    def _write_plan(self, window: List[WalkStep], problem_dir: Path) -> None:
        """Write the window's ground actions, one per line, in PDDL format."""
        lines = []
        for step in window:
            operator, assignment = _select_operator_for_step(self.handler, step)
            ground_str = _ground_action_str(operator, assignment)
            lines.append(parse_gym_to_pddl_ground_action(ground_str))
        (problem_dir / "plan.txt").write_text("\n".join(lines) + "\n")

    # ── Images ───────────────────────────────────────────────────────────

    def _write_images(self, window: List[WalkStep], problem_dir: Path) -> None:
        """Copy the window's rendered frames as state_0..state_k.png."""
        # frame_before of step 0, then frame_after of every step.
        frames = [window[0].frame_before] + [s.frame_after for s in window]
        for idx, frame in enumerate(frames):
            dest = problem_dir / f"state_{idx:{self.handler.seq_idx_format}}.png"
            shutil.copy(frame, dest)

    # ── GT trajectory json ───────────────────────────────────────────────

    def _write_gt_trajectory(self, problem_name: str, window: List[WalkStep],
                             problem_dir: Path) -> None:
        """Build GT TrajectoryStep list for the window and write its JSON."""
        gt_trajectory: List[TrajectoryStep] = []
        for i, step in enumerate(window, start=1):
            gt_trajectory.append(self.handler._create_trajectory_step(
                step.prev_obs, step.action, i, step.next_obs))
        self.handler._write_gt_trajectory_json(
            problem_name, gt_trajectory, problem_dir)


# ── Module helpers (kept out of the class — pure functions) ──────────────

def _select_operator_for_step(handler: PDDLGymImageTrajectoryHandler,
                              step: WalkStep):
    from pddlgym.core import _select_operator
    return _select_operator(step.prev_obs, step.action, handler.pddl_env.domain)


def _ground_action_str(operator, assignment) -> str:
    from src.utils.pddl_gym import ground_action
    return ground_action(operator, assignment)
