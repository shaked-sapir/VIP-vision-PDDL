"""
Generate a test sequence for the maze PDDLGym environment.

This script:
1. Initializes the maze environment
2. Generates a 10-step trajectory with distinct observations
3. Ensures no two consecutive images are identical
4. Saves images for each state
5. Saves a JSON file with the trajectory information including actions
"""

import json
from pathlib import Path
import pddlgym
import matplotlib.pyplot as plt
import numpy as np


def images_are_identical(img1, img2):
    """
    Check if two images are identical.

    Args:
        img1: First image as numpy array
        img2: Second image as numpy array

    Returns:
        bool: True if images are identical, False otherwise
    """
    return np.array_equal(img1, img2)


def generate_maze_test_sequence(num_steps=10, problem_index=0, max_attempts_per_step=20):
    """
    Generate a test sequence for the maze domain.
    Ensures no two consecutive observations are identical.

    Args:
        num_steps: Number of steps to generate (default: 10)
        problem_index: Which problem to use (default: 0)
        max_attempts_per_step: Maximum attempts to find a state-changing action (default: 20)

    Returns:
        tuple: (images_dir, trajectory_data)
    """
    print("="*80)
    print("GENERATING MAZE TEST SEQUENCE")
    print("="*80)
    print()

    # Initialize environment
    env_name = "PDDLEnvMaze-v0"
    print(f"Initializing environment: {env_name}")
    env = pddlgym.make(env_name)

    # Get problem
    problem_file = env.problems[problem_index]
    print(f"Problem file: {problem_file}")

    # Create output directory
    output_dir = Path(__file__).parent / "maze_test_sequence"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    # Reset environment
    print("Resetting environment...")
    state, debug_info = env.reset()

    # Initialize trajectory data
    trajectory_data = {
        "domain": "maze",
        "problem": str(problem_file),
        "num_steps": num_steps,
        "states": [],
        "actions": []
    }

    # Save initial state
    print(f"Saving initial state (state_0000.png)...")
    prev_img = env.render()
    plt.imsave(output_dir / "state_0000.png", prev_img)

    # Convert initial state to serializable format
    initial_state_literals = [str(lit) for lit in state.literals]
    trajectory_data["states"].append({
        "step": 0,
        "literals": initial_state_literals,
        "is_goal": env._is_goal_reached(state, debug_info) if hasattr(env, '_is_goal_reached') else False
    })

    # Generate trajectory with distinct observations
    print(f"\nGenerating {num_steps}-step trajectory with distinct observations...")
    step = 1

    while step <= num_steps:
        # Get valid actions
        valid_actions = list(env.action_space.all_ground_literals(state))

        if not valid_actions:
            print(f"  Step {step}: No valid actions available. Stopping.")
            break

        # Try to find an action that changes the observation
        action_found = False
        attempts = 0

        for action in valid_actions:
            if attempts >= max_attempts_per_step:
                print(f"  Step {step}: Could not find state-changing action after {max_attempts_per_step} attempts. Stopping.")
                break

            attempts += 1
            action_str = str(action)

            # Save current state to potentially restore
            saved_state = state

            # Execute action
            next_state, reward, done, truncated, next_debug_info = env.step(action)

            # Render new state
            current_img = env.render()

            # Check if observation changed
            if not images_are_identical(prev_img, current_img):
                # Found a state-changing action!
                state = next_state
                debug_info = next_debug_info
                action_found = True

                print(f"  Step {step}: Action = {action_str} (attempt {attempts})")

                # Save action
                trajectory_data["actions"].append({
                    "step": step,
                    "action": action_str,
                    "reward": float(reward),
                    "done": bool(done),
                    "truncated": bool(truncated)
                })

                # Save state image
                img_filename = f"state_{step:04d}.png"
                plt.imsave(output_dir / img_filename, current_img)
                print(f"  Step {step}: Saved {img_filename}")

                # Convert state to serializable format
                state_literals = [str(lit) for lit in state.literals]
                trajectory_data["states"].append({
                    "step": step,
                    "literals": state_literals,
                    "is_goal": env._is_goal_reached(state, debug_info) if hasattr(env, '_is_goal_reached') else False
                })

                # Update previous image for next comparison
                prev_img = current_img

                # Check if goal reached
                if done:
                    print(f"  Step {step}: Goal reached!")
                    break

                step += 1
                break
            else:
                # Action didn't change observation, try next action
                # Reset to saved state for next attempt
                state = saved_state
                continue

        if not action_found and attempts >= max_attempts_per_step:
            # Could not find any state-changing action
            break

    # Update actual number of steps
    trajectory_data["num_steps"] = len(trajectory_data["actions"])
    trajectory_data["goal_reached"] = trajectory_data["states"][-1]["is_goal"] if trajectory_data["states"] else False

    # Save trajectory JSON
    trajectory_file = output_dir / "trajectory.json"
    with open(trajectory_file, 'w') as f:
        json.dump(trajectory_data, f, indent=2)

    print()
    print("="*80)
    print("SEQUENCE GENERATION COMPLETE")
    print("="*80)
    print(f"Images saved: {len(trajectory_data['states'])} states")
    print(f"Actions executed: {len(trajectory_data['actions'])}")
    print(f"Goal reached: {trajectory_data['goal_reached']}")
    print(f"Output directory: {output_dir}")
    print(f"Trajectory JSON: {trajectory_file}")
    print()

    return output_dir, trajectory_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate maze test sequence with distinct observations"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of steps to generate (default: 10)"
    )
    parser.add_argument(
        "--problem",
        type=int,
        default=0,
        help="Problem index to use (default: 0)"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=20,
        help="Maximum attempts per step to find state-changing action (default: 20)"
    )

    args = parser.parse_args()

    generate_maze_test_sequence(
        num_steps=args.num_steps,
        problem_index=args.problem,
        max_attempts_per_step=args.max_attempts
    )
