import json
import os
from pathlib import Path
from PIL import Image

from pddlgym.core import PDDLEnv, _select_operator

from pddlgym.structs import State, Literal

from src.utils.pddl import set_problem_by_name


class ImageTrajectoryHandler:
    # TODO: add an upper limit for the number of images we can handle in a trajectory and format the ":04d" properly
    def __init__(self, pddl_env: PDDLEnv):
        super().__init__()
        self.pddl_env = pddl_env

    def _create_image(self, trajectory_output_dir: Path, image_sequential_idx: int):
        """
        renders a single image from the current state of the environment and saves it.
        :param trajectory_output_dir: directory to save the image to
        :param image_sequential_idx: the sequential index of the image in the trajectory, 0-indexed. e.g., 0 for the
                                     first image, 1 for the second image, etc.
        :return: (None)
        """
        img = self.pddl_env.render(mode='rgb_array')
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(trajectory_output_dir, f"state_{image_sequential_idx:04d}.png"))
        return

    @staticmethod
    def _create_trajectory_state(obs: State):
        return {
            "literals": [str(literal) for literal in obs.literals],
            "objects": [str(obj) for obj in obs.objects],
            "goal": [str(literal) for literal in obs.goal.literals]
        }

    def _create_trajectory_triplet(self, curr_obs: State, action: Literal, action_index: int, next_obs: State):
        selected_operator, assignment = _select_operator(curr_obs, action, self.pddl_env.domain)

        return {
            "step": action_index,
            "current_state": self._create_trajectory_state(curr_obs),
            "ground_action": str(action),

            # TODO later: the _select_operator seems to make it a "safe" action, but the blocksworld is not a domain prone to unsafety - discuss with Roni
            "operator_object_assignment": assignment,
            "lifted_preconds": str(self.pddl_env.domain.operators[selected_operator.name].preconds.literals),
            "next_state": self._create_trajectory_state(next_obs)
        }

    def create_image_trajectory(self, problem_name: str, trajectory_output_dir: Path, num_steps: int = 100):
        """
        This method creates a trajectory of randomly-taken actions within a pddlgym environment, using a specific
        problem, and saves the trajectory and its images into a directory.

        :param problem_name: The problem for which to create the trajectory
        :param trajectory_output_dir: the output directory for the trajectory images
        :param num_steps: the number of steps (actions) to take when generating the trajectory.
        :return: (None)
        """
        set_problem_by_name(self.pddl_env, problem_name)
        obs, info = self.pddl_env.reset()
        new_obs = obs

        # Create a directory to save images if it does not exist
        os.makedirs(trajectory_output_dir, exist_ok=True)

        # Create a file to save states and actions
        log_file_path = os.path.join(trajectory_output_dir, "trajectory.json")
        trajectory = []

        self._create_image(trajectory_output_dir, 0)

        # Run 1000 random moves and save the images
        for i in range(1, num_steps + 1):
            obs = new_obs

            # Sample a random valid action (action affecting the state) from the set of valid actions
            while new_obs == obs:
                action = self.pddl_env.action_space.sample(obs)
                new_obs, _, done, _, _ = self.pddl_env.step(action)

            print(f"chosed_action: {action}")
            self._create_image(trajectory_output_dir, i)
            state_action_entry = self._create_trajectory_triplet(curr_obs=obs, action=action, action_index=i,
                                                                 next_obs=new_obs)
            trajectory.append(state_action_entry)

            if done:
                break

        # Save the states and actions to the log file
        with open(log_file_path, 'w') as log_file:
            json.dump(trajectory, log_file, indent=4)

        print(f"Images saved to the directory '{trajectory_output_dir}'")
        print(f"States and actions log saved to '{log_file_path}'")
