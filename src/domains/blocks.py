import os
import pddlgym
from pddlgym.core import _select_operator
import random
from PIL import Image
import json
import cv2

def sort_images_numerically(image_list):
    return sorted(image_list, key=lambda x: int(x.split('_')[1].split('.')[0]))

# Initialize the PDDLGym environment
env = pddlgym.make("PDDLEnvBlocks-v0")
obs, info = env.reset()

# Create a directory to save images if it does not exist
output_dir = "blocks_images"
os.makedirs(output_dir, exist_ok=True)

# Create a file to save states and actions
log_file_path = os.path.join(output_dir, "states_and_actions.json")
states_and_actions = []

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
# Run 1000 random moves and save the images
for i in range(1, 10):
    # Sample a random valid action from the set of valid actions
    new_obs = obs

    while new_obs == obs:
    # obs, _, done, _ = env.step(action) # obs holds the "next_state", i.e. the state resulting in executing the action
        action = env.action_space.sample(obs)
        new_obs = env.step(action)[0] # obs holds the "next_state", i.e. the state resulting in executing the action

    # obs = new_obs
    print(type(env.domain.operators['pick-up']))
    print(type(action))
    print(env.domain.operators["pick-up"])
    print(env.domain.operators["pick-up"].preconds)
    print(f"operator preconds: {type(env.domain.operators['pick-up'].preconds)}")
    print(f"operator preconds1: {env.domain.operators['pick-up'].preconds.pddl_variables()}")
    print(f"operator preconds2: {env.domain.operators['pick-up'].preconds.pddl_variables_typed()}")
    print(f"operator preconds literals: {env.domain.operators['pick-up'].preconds.literals}") # I need this one, but in its grounded version matching the concrete objects in the domain and taken action
    print(f"operator params: {env.domain.operators['pick-up'].params}")
    print(f"selected operator: {_select_operator(obs, action, env.domain)[1]}")
    print(f"selected operator: {json.dumps(_select_operator(obs, action, env.domain)[1], indent=4)}")

    print(type(obs))
    # Record the state and action
    state_action_entry = {
        "step": i,
        "action": str(action),
        # "action": action.predicates,
        "state": str(obs)
    }


    states_and_actions.append(state_action_entry)

    # Render the environment and save the image
    img = env.render()
    img_pil = Image.fromarray(img)
    img_pil.save(os.path.join(output_dir, f"state_{i:04d}.png"))

    # # Reset the environment if a terminal state is reached
    # if done:
    #     env.reset()

# Save the states and actions to the log file
with open(log_file_path, 'w') as log_file:
    json.dump(states_and_actions, log_file, indent=4)

# Create a video from the saved images
video_path = os.path.join(output_dir, "blocks_video.mp4")
image_files = [img for img in os.listdir(output_dir) if img.endswith(".png")]
image_files = sort_images_numerically(image_files)
video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 10, cv2.imread(os.path.join(output_dir, image_files[0])).shape[1::-1])

for image_file in image_files:
    video.write(cv2.imread(os.path.join(output_dir, image_file)))

video.release()

print(f"Images saved to the directory '{output_dir}'")
print(f"States and actions log saved to '{log_file_path}'")
print(f"Video saved to '{video_path}'")
