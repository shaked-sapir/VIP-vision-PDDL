"""PDDLGym-based trajectory handler — generates images + GT trajectories from gym environments."""

import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import pddlgym
from PIL import Image
from matplotlib import pyplot as plt
from pddlgym.core import _select_operator
from pddlgym.structs import State, Literal

from src.trajectory_handlers.image_trajectory_handler import ImageTrajectoryHandler
from src.typings import TrajectoryState, TrajectoryStep
from src.utils.containers import serialize
from src.utils.pddl_gym import set_problem_by_name, ground_action

# Lazy planner imports — each planner is independently optional.
_PLANNERS = {}

try:
    from pddlgym_planners.ff import FF
    _PLANNERS["ff"] = FF
except ImportError:
    pass

try:
    from pddlgym_planners.fd import FD
    _PLANNERS["fd"] = FD
except ImportError:
    pass


def _resolve_planner(name: str):
    """Resolve a planner class by name, raising a clear error if unavailable."""
    key = name.lower()
    if key not in _PLANNERS:
        available = ", ".join(sorted(_PLANNERS)) or "none"
        raise RuntimeError(
            f"Planner '{name}' is not installed. "
            f"Available: [{available}]. "
            f"Install with: pip install pddlgym_planners")
    return _PLANNERS[key]


class PDDLGymImageTrajectoryHandler(ImageTrajectoryHandler):
    """Trajectory handler backed by a PDDLGym environment.

    Adds gym-specific capabilities: rendering images, stepping the environment,
    and generating ground-truth trajectories via random actions or a planner.

    Pipeline flow (run_pipeline):
        1. Generate images + GT trajectory via PDDLGym (create_trajectory_from_gym).
        2. Run visual inference on those images (construct_trajectory_from_images).
        3. Write inferred .trajectory + .masking_info files.

    To add a new PDDLGym-based domain:
        - For deterministic (non-LLM) detection: subclass this directly and
          implement init_visual_components (see BlocksImageTrajectoryHandler).
        - For LLM-based detection: subclass LLMImageTrajectoryHandler (which
          combines LLMVisualComponentsMixin with this class) and set
          detector_class + classifier_class.
        - Override _rename_ground_action if gym action names differ from your
          domain's PDDL format.
        - Override _manipulate_trajectory_json if the GT trajectory JSON from
          PDDLGym needs domain-specific transforms (e.g. type renaming).
    """

    def __init__(self, domain_name: str, trajectory_size_limit: int = 1000, **kwargs):
        super().__init__(domain_name=domain_name)
        self.pddl_env = pddlgym.make(domain_name)
        self.trajectory_size_limit = trajectory_size_limit
        digit_count = math.ceil(math.log10(trajectory_size_limit)) + 1
        self.seq_idx_format = f'0{digit_count}d'

    # ── Image rendering ─────────────────────────────────────────────────

    def create_image(self, image_output_dir: Path, image_sequential_idx: int) -> None:
        """Render and save the current gym state as a PNG image."""
        img = (self.pddl_env.render() if self.domain_name == "PDDLEnvMaze-v0"
               else self.pddl_env.render(mode='rgb_array'))
        plt.close('all')
        if img.dtype != "uint8":
            img = (img * 255).clip(0, 255).astype("uint8")
        Image.fromarray(img).save(
            os.path.join(image_output_dir, f"state_{image_sequential_idx:{self.seq_idx_format}}.png"))

    # ── Gym trajectory helpers ──────────────────────────────────────────

    @staticmethod
    def _create_trajectory_state(obs: State) -> TrajectoryState:
        return TrajectoryState(
            literals=[str(lit) for lit in obs.literals],
            objects=[str(obj) for obj in obs.objects],
            goal=([str(lit) for lit in obs.goal.literals]
                  if hasattr(obs.goal, "literals") else [str(obs.goal)]),
        )

    def _create_trajectory_step(self, curr_obs: State, action: Literal,
                                 action_index: int, next_obs: State) -> TrajectoryStep:
        selected_operator, assignment = _select_operator(curr_obs, action, self.pddl_env.domain)
        return TrajectoryStep(
            step=action_index,
            current_state=self._create_trajectory_state(curr_obs),
            ground_action=ground_action(selected_operator, assignment),
            operator_object_assignment=assignment,
            lifted_preconds=str(
                self.pddl_env.domain.operators[selected_operator.name].preconds.literals),
            next_state=self._create_trajectory_state(next_obs),
        )

    def _advance_to_start_index(self, start_index: int, obs: State) -> State:
        """Advance the environment to a specific state index using random actions."""
        if start_index <= 0:
            return obs
        print(f"  Running PDDLGym to state {start_index} (without rendering)...")
        for i in range(start_index):
            obs, done, _ = self._sample_state_changing_action(obs)
            if done:
                raise ValueError(f"Goal reached at step {i}, cannot start from step {start_index}")
        print(f"  Reached state {start_index}, starting trajectory generation")
        return obs

    def _sample_state_changing_action(self, obs: State,
                                       max_retries: int = 1000) -> tuple[State, bool, any]:
        """Sample a random action that changes the state.

        Raises:
            RuntimeError: If no state-changing action found within max_retries.
        """
        new_obs, done, action = obs, False, None
        for attempt in range(1, max_retries + 1):
            try:
                action = self.pddl_env.action_space.sample(obs)
                env_temp = deepcopy(self.pddl_env)
                new_obs, _, _, _, _ = env_temp.step(action)
                _ = self.pddl_env.action_space.sample(new_obs)  # validate state
                new_obs, _, done, _, _ = self.pddl_env.step(action)
                if new_obs != obs:
                    return new_obs, done, action
            except Exception as e:
                if attempt % 100 == 0:
                    print(f"  Warning: {attempt} failed sampling attempts ({e})")
                new_obs = obs
        raise RuntimeError(
            f"No state-changing action found after {max_retries} attempts. "
            f"The environment may be in a dead-end state.")

    def _generate_plan(self, obs: State, num_steps: int,
                        planner_name: str = "ff") -> List[Literal]:
        """Generate a plan using the specified planner.

        Args:
            obs: Current PDDLGym state.
            num_steps: Maximum number of actions to return.
            planner_name: Which planner to use ("ff" or "fd").

        Raises:
            RuntimeError: If the planner is not installed or fails.
        """
        planner_cls = _resolve_planner(planner_name)
        print(f"  Using {planner_name.upper()} planner to generate solution...")
        planner = planner_cls()
        try:
            plan = planner(self.pddl_env.domain, obs, timeout=300)
            print(f"  Planner found solution with {len(plan)} actions")
            return plan[:num_steps] if len(plan) > num_steps else plan
        except Exception as e:
            raise RuntimeError(f"{planner_name.upper()} planner failed: {e}")

    def _execute_trajectory(self, actions: List[Literal], images_output_path: Path,
                             initial_obs: State) -> tuple[list[TrajectoryStep], list[str]]:
        """Execute a sequence of actions, rendering each state."""
        GT_trajectory, ground_actions, obs = [], [], initial_obs
        self.create_image(images_output_path, 0)
        for i, action in enumerate(actions, start=1):
            prev_obs = obs
            obs, _, done, _, _ = self.pddl_env.step(action)
            self.create_image(images_output_path, i)
            step = self._create_trajectory_step(prev_obs, action, i, obs)
            GT_trajectory.append(step)
            ground_actions.append(step.ground_action)
            if done:
                print(f"  Goal reached at step {i}")
                break
        return GT_trajectory, ground_actions

    def _execute_random_trajectory(self, num_steps: int, images_output_path: Path,
                                    initial_obs: State) -> tuple[list[TrajectoryStep], list[str]]:
        """Generate and execute a trajectory using random state-changing actions."""
        GT_trajectory, ground_actions, obs = [], [], initial_obs
        self.create_image(images_output_path, 0)
        for i in range(1, num_steps + 1):
            prev_obs = obs
            obs, done, action = self._sample_state_changing_action(obs)
            self.create_image(images_output_path, i)
            step = self._create_trajectory_step(prev_obs, action, i, obs)
            GT_trajectory.append(step)
            ground_actions.append(step.ground_action)
            if done:
                print(f"  Goal reached at step {i}")
                break
        return GT_trajectory, ground_actions

    # ── Hook for domain-specific GT trajectory manipulation ─────────────

    def _manipulate_trajectory_json(self, gt_trajectory_json: list) -> list:
        """Override to apply domain-specific transforms to GT trajectory JSON."""
        return gt_trajectory_json

    def _write_gt_trajectory_json(self, problem_name: str,
                                  gt_trajectory: List[TrajectoryStep],
                                  output_path: Path) -> Path:
        """Serialize a GT trajectory (list of TrajectoryStep) to `{problem}_trajectory.json`."""
        gt_json = self._manipulate_trajectory_json(serialize(gt_trajectory))
        trajectory_log_path = Path(output_path) / f"{problem_name}_trajectory.json"
        with open(trajectory_log_path, "w") as f:
            json.dump(gt_json, f, indent=4)
        return trajectory_log_path

    # ── Main gym pipeline ───────────────────────────────────────────────

    def create_trajectory_from_gym(self, problem_name: str, images_output_path: Path,
                                    num_steps: int = 100, start_index: int = 0,
                                    planner: Optional[str] = None) -> List[str]:
        """Generate images + GT trajectory from a PDDLGym environment.

        Args:
            problem_name: PDDL problem identifier.
            images_output_path: Directory to write rendered images into.
            num_steps: Maximum trajectory length (default 100).
            start_index: Advance the env to this state before generating (default 0).
            planner: Which planner to use — "ff", "fd", or None for random actions
                     (default None).

        Returns:
            List of ground action strings (to be passed to create_trajectory_and_masks).
        """
        if num_steps > self.trajectory_size_limit:
            raise ValueError(f"Cannot exceed {self.trajectory_size_limit} steps")

        set_problem_by_name(self.pddl_env, problem_name)
        obs, _ = self.pddl_env.reset()
        os.makedirs(images_output_path, exist_ok=True)
        obs = self._advance_to_start_index(start_index, obs)

        if planner:
            plan = self._generate_plan(obs, num_steps, planner_name=planner)
            GT_trajectory, ground_actions = self._execute_trajectory(
                plan, images_output_path, obs)
        else:
            GT_trajectory, ground_actions = self._execute_random_trajectory(
                num_steps, images_output_path, obs)

        # Save GT trajectory JSON
        trajectory_log_path = self._write_gt_trajectory_json(
            problem_name, GT_trajectory, images_output_path)

        print(f"Images saved to '{images_output_path}'")
        print(f"Trajectory log saved to '{trajectory_log_path}'")
        return ground_actions

    # ── Unified entry point ─────────────────────────────────────────────

    def run_pipeline(self, problem_name: str, images_path: Path, **kwargs) -> List[dict]:
        """Run the full PDDLGym pipeline: generate images + GT, then infer trajectory.

        Chains create_trajectory_from_gym → create_trajectory_and_masks.

        Args:
            problem_name: PDDL problem identifier (e.g. "problem0").
            images_path: Directory where rendered images and output files are written.
            **kwargs: Forwarded to create_trajectory_from_gym. Supported options:
                num_steps (int): Max trajectory length (default 100).
                start_index (int): Skip to this state before generating (default 0).
                planner (str | None): Planner to use — "ff", "fd", or None for
                    random actions (default None).

        Returns:
            The inferred trajectory as a list of step dicts.
        """
        ground_actions = self.create_trajectory_from_gym(
            problem_name, images_path, **kwargs)
        return self.create_trajectory_and_masks(problem_name, ground_actions, images_path)
