#!/usr/bin/env python3
import json
import os
import re

def fix_trajectory_json(json_path):
    """Fix the ground actions in a trajectory JSON file."""
    print(f"\nProcessing {json_path}...")

    with open(json_path, 'r') as f:
        data = json.load(f)

    corrections = []

    for step in data:
        # Parse the ground action
        ground_action = step['ground_action']

        # Extract action format: move(tile, from_pos, to_pos)
        match = re.match(r'move\((.+?),\s*(.+?),\s*(.+?)\)', ground_action)
        if not match:
            print(f"  WARNING: Could not parse ground action at step {step['step']}: {ground_action}")
            continue

        tile_in_action = match.group(1)
        from_pos = match.group(2)
        to_pos = match.group(3)

        # Find the actual tile at from_pos in current_state
        actual_tile = None
        for lit in step['current_state']['literals']:
            if lit.startswith('at('):
                # Parse at(tile, position)
                at_match = re.match(r'at\((.+?),\s*(.+?)\)', lit)
                if at_match:
                    lit_tile = at_match.group(1)
                    lit_pos = at_match.group(2)
                    if lit_pos == from_pos:
                        actual_tile = lit_tile
                        break

        if actual_tile is None:
            print(f"  WARNING: Could not find tile at {from_pos} in step {step['step']}")
            continue

        # Check if correction is needed
        if tile_in_action != actual_tile:
            old_action = ground_action
            new_action = f"move({actual_tile}, {from_pos}, {to_pos})"
            step['ground_action'] = new_action
            corrections.append({
                'step': step['step'],
                'old': old_action,
                'new': new_action
            })

    # Save corrected JSON
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"  Fixed {len(corrections)} ground actions in JSON file")
    return corrections, data

def fix_trajectory_file(trajectory_path, corrected_json_data):
    """Fix the operators in a .trajectory file based on corrected JSON data."""
    print(f"\nProcessing {trajectory_path}...")

    with open(trajectory_path, 'r') as f:
        content = f.read()

    # Build a mapping from step number to corrected operator
    step_to_operator = {}
    for step in corrected_json_data:
        # Parse ground action from JSON: move(tile, from, to)
        ground_action = step['ground_action']
        match = re.match(r'move\((.+?),\s*(.+?),\s*(.+?)\)', ground_action)
        if match:
            tile = match.group(1).split(':')[0]  # Remove type annotation if present
            from_pos = match.group(2).split(':')[0]
            to_pos = match.group(3).split(':')[0]
            step_to_operator[step['step']] = f"(move {tile} {from_pos} {to_pos})"

    # Split trajectory file into lines
    lines = content.split('\n')

    # Track which step we're on
    current_step = 0
    corrections = 0

    new_lines = []
    for line in lines:
        # Check if this is an operator line
        if line.strip().startswith('(operator:'):
            current_step += 1
            if current_step in step_to_operator:
                # Extract the current operator
                old_operator = line.strip()
                new_operator = f"(operator: {step_to_operator[current_step]})"

                if old_operator != new_operator:
                    corrections += 1
                    new_lines.append(new_operator)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    # Save corrected trajectory file
    with open(trajectory_path, 'w') as f:
        f.write('\n'.join(new_lines))

    print(f"  Fixed {corrections} operators in .trajectory file")
    return corrections

def process_problem_directory(problem_dir):
    """Process a single problem directory."""
    problem_name = os.path.basename(problem_dir)
    print(f"\n{'='*60}")
    print(f"Processing {problem_name}")
    print(f"{'='*60}")

    # Find the trajectory JSON file
    json_file = os.path.join(problem_dir, f"{problem_name}_trajectory.json")
    trajectory_file = os.path.join(problem_dir, f"{problem_name}.trajectory")

    if not os.path.exists(json_file):
        print(f"  WARNING: JSON file not found: {json_file}")
        return

    if not os.path.exists(trajectory_file):
        print(f"  WARNING: Trajectory file not found: {trajectory_file}")
        return

    # Step 1: Fix the JSON file
    corrections, corrected_data = fix_trajectory_json(json_file)

    # Step 2: Fix the .trajectory file
    if corrected_data:
        fix_trajectory_file(trajectory_file, corrected_data)

    print(f"\n✓ Completed {problem_name}")

def main():
    base_dir = "/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/data/n_puzzle_typed/multi_problem_02-01-2026T16:55:49__model=gpt-5.1__steps=300__planner/training/trajectories"

    # Get all problem directories
    problem_dirs = sorted([
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if d.startswith('problem') and os.path.isdir(os.path.join(base_dir, d))
    ])

    print(f"Found {len(problem_dirs)} problem directories")

    for problem_dir in problem_dirs:
        process_problem_directory(problem_dir)

    print(f"\n{'='*60}")
    print("ALL DONE!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
