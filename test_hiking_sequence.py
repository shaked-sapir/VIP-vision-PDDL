"""
Generate a test sequence for the hiking PDDLGym environment.

This script:
1. Initializes the hiking environment
2. Generates a 10-step trajectory
3. Saves images for each state
4. Saves a JSON file with the trajectory information
"""

import json
from pathlib import Path
import pddlgym
import matplotlib.pyplot as plt


def generate_hiking_test_sequence(num_steps=10, problem_index=0):
    """
    Generate a test sequence for the hiking domain.

    Args:
        num_steps: Number of steps to generate (default: 10)
        problem_index: Which problem to use (default: 0)

    Returns:
        tuple: (images_dir, trajectory_data)
    """
    print("="*80)
    print("GENERATING HIKING TEST SEQUENCE")
    print("="*80)
    print()

    # Initialize environment
    env_name = "PDDLEnvHiking-v0"
    print(f"Initializing environment: {env_name}")
    env = pddlgym.make(env_name)

    # Get problem
    problem_file = env.problems[problem_index]
    print(f"Problem file: {problem_file}")

    # Create output directory
    output_dir = Path(__file__).parent / "hiking_test_sequence"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    # Reset environment
    print("Resetting environment...")
    state, debug_info = env.reset()

    # Initialize trajectory data
    trajectory_data = {
        "domain": "hiking",
        "problem": str(problem_file),
        "num_steps": num_steps,
        "states": [],
        "actions": []
    }

    # Save initial state
    print(f"Saving initial state (state_0000.png)...")
    img = env.render()
    plt.imsave(output_dir / "state_0000.png", img)

    # Convert initial state to serializable format
    initial_state_literals = [str(lit) for lit in state.literals]
    trajectory_data["states"].append({
        "step": 0,
        "literals": initial_state_literals,
        "is_goal": env._is_goal_reached(state, debug_info) if hasattr(env, '_is_goal_reached') else False
    })

    # Generate trajectory
    print(f"\nGenerating {num_steps}-step trajectory...")
    for step in range(1, num_steps + 1):
        # Get valid actions
        valid_actions = list(env.action_space.all_ground_literals(state))

        if not valid_actions:
            print(f"  Step {step}: No valid actions available. Stopping.")
            break

        # Choose first valid action (or you could use random)
        action = valid_actions[0]
        action_str = str(action)
        print(f"  Step {step}: Action = {action_str}")

        # Execute action
        state, reward, done, truncated, debug_info = env.step(action)

        # Save action
        trajectory_data["actions"].append({
            "step": step,
            "action": action_str,
            "reward": float(reward),
            "done": bool(done),
            "truncated": bool(truncated)
        })

        # Render and save state
        img = env.render()
        img_filename = f"state_{step:04d}.png"
        plt.imsave(output_dir / img_filename, img)
        print(f"  Step {step}: Saved {img_filename}")

        # Convert state to serializable format
        state_literals = [str(lit) for lit in state.literals]
        trajectory_data["states"].append({
            "step": step,
            "literals": state_literals,
            "is_goal": env._is_goal_reached(state, debug_info) if hasattr(env, '_is_goal_reached') else False
        })

        # Check if goal reached
        if done:
            print(f"  Step {step}: Goal reached!")
            break

    # Update actual number of steps
    trajectory_data["num_steps"] = len(trajectory_data["actions"])
    trajectory_data["goal_reached"] = trajectory_data["states"][-1]["is_goal"]

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
        description="Generate hiking test sequence"
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

    args = parser.parse_args()

    generate_hiking_test_sequence(
        num_steps=args.num_steps,
        problem_index=args.problem
    )
