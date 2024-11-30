import json
import os

import cv2
from PIL import Image

import pddlgym
from pddlgym.core import _select_operator
from pddlgym.rendering.blocks import _block_name_to_color


def alg(problem, num_steps=1000):
    """
    This the main workflow of predicates classification within an image in the blocks world.
    :param problem: the problem instance we desire to make trajectory from
    :param num_steps: the number of steps we want to have for the trajectory
    :return:
    """

    #TODO: check how to apply for a specific problem file
    env = pddlgym.make("PDDLEnvBlocks-v0")

    obs, info = env.reset()
    new_obs = obs

    # Create a directory to save images if it does not exist
    output_dir = "blocks_images"
    os.makedirs(output_dir, exist_ok=True)

    # Create a file to save states and actions
    log_file_path = os.path.join(output_dir, "states_and_actions.json")
    states_and_actions = []

    # Save the initial state
    # Save the initial state
    initial_entry = {
        "step": 0,
        "action": "initial_state",
        "state": str(obs)
    }
    states_and_actions.append(initial_entry)
    # Render the environment and save the image
    img = env.render(mode='rgb_array')
    img_pil = Image.fromarray(img)
    img_pil.save(os.path.join(output_dir, f"state_{0:04d}.png"))

    # extract colors:  the mapping is generated at problem initialization
    block_name_to_color = {str(obj): color for obj, color in _block_name_to_color.items()}

    # Run 1000 random moves and save the images
    for i in range(1, num_steps):
        obs = new_obs
        # Sample a random valid action from the set of valid actions

        while new_obs == obs:
            action = env.action_space.sample(obs)
            new_obs = env.step(action)[0]  # new_obs holds the "next_state", i.e. the state resulting in executing the action

        # Record the state and action
        state_action_entry = {
            "step": i,
            "current_state": str(obs),
            "ground_action": str(action),

            #TODO later: the _select_operator seems to make it a "safe" action, but the blocksworld is not a domain prone to unsafety - discuss with Roni
            "operator_object_assignment": _select_operator(obs, action, env.domain)[1],
            "lifted_preconds": str(env.domain.operators['pick-up'].preconds.literals),
            "ground_resulted_state": str(new_obs)
        }

        states_and_actions.append(state_action_entry)

        # Render the environment and save the image
        img = env.render()
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(output_dir, f"state_{i:04d}.png"))

        # TODO: Reset the environment if a terminal state is reached
        # if done:
        #     env.reset()

    # Save the states and actions to the log file
    with open(log_file_path, 'w') as log_file:
        json.dump(states_and_actions, log_file, indent=4)


    print(f"Images saved to the directory '{output_dir}'")
    print(f"States and actions log saved to '{log_file_path}'")


"""
extract predicates for each picture in trajectory, automatically managing the coloring stuff.
notice that only blocks get colors in the mapping, the colors of table and robot are unique and are the same
in all problems, so we can map them ahead.
"""