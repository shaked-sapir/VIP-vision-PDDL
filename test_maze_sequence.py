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


def test_generate_maze_test_sequence(num_steps=10, problem_index=0, max_attempts_per_step=20):
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

    # Initialize trajectory data - using ground truth structure (list of steps)
    trajectory_data = []

    # Save initial state
    print(f"Saving initial state (state_0000.png)...")
    prev_img = env.render()
    plt.imsave(output_dir / "state_0000.png", prev_img)

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
                action_found = True

                print(f"  Step {step}: Action = {action_str} (attempt {attempts})")

                # Extract objects from state
                current_state_objects = sorted({str(obj) for obj in state.objects})
                next_state_objects = sorted({str(obj) for obj in next_state.objects})

                # Extract goal literals
                # state.goal is typically a frozenset of Literal objects
                goal_literals = []
                if hasattr(state, 'goal') and state.goal:
                    if isinstance(state.goal, (list, set, frozenset)):
                        goal_literals = [str(lit) for lit in state.goal]
                    else:
                        goal_literals = [str(state.goal)]

                # Parse operator object assignment from action
                # Action format: predicate(obj1:type1,obj2:type2,...)
                operator_assignment = {}
                if '(' in action_str:
                    pred_name = action_str.split('(')[0]
                    args_str = action_str.split('(')[1].rstrip(')')
                    if args_str:
                        args = [arg.strip() for arg in args_str.split(',')]
                        for i, arg in enumerate(args):
                            # Extract just the object name (before the colon)
                            obj_name = arg.split(':')[0] if ':' in arg else arg
                            operator_assignment[f"?arg{i}"] = obj_name

                # Create step entry in ground truth format
                step_entry = {
                    "step": step,
                    "current_state": {
                        "literals": [str(lit) for lit in state.literals],
                        "objects": current_state_objects,
                        "goal": goal_literals
                    },
                    "ground_action": action_str,
                    "operator_object_assignment": operator_assignment,
                    "lifted_preconds": f"[{action.predicate.name}]",  # Simplified
                    "next_state": {
                        "literals": [str(lit) for lit in next_state.literals],
                        "objects": next_state_objects
                    }
                }

                trajectory_data.append(step_entry)

                # Save state image
                img_filename = f"state_{step:04d}.png"
                plt.imsave(output_dir / img_filename, current_img)
                print(f"  Step {step}: Saved {img_filename}")

                # Update state and previous image for next iteration
                state = next_state
                debug_info = next_debug_info
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

    # Save trajectory JSON (in ground truth format)
    trajectory_file = output_dir / "maze_trajectory.json"
    with open(trajectory_file, 'w') as f:
        json.dump(trajectory_data, f, indent=4)

    print()
    print("="*80)
    print("SEQUENCE GENERATION COMPLETE")
    print("="*80)
    print(f"Images saved: {len(trajectory_data) + 1} states (including initial state)")
    print(f"Actions executed: {len(trajectory_data)}")
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

    test_generate_maze_test_sequence(
        num_steps=args.num_steps,
        problem_index=args.problem,
        max_attempts_per_step=args.max_attempts
    )
