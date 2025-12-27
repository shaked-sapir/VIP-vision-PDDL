#!/usr/bin/env python3
"""
Fix maze object names by replacing hyphens with underscores.
Changes: player-1 → player_1, loc-5-4 → loc_5_4
Preserves: move-dir-up, oriented-right, etc. (predicate names)
"""

import re
from pathlib import Path

def fix_object_names(content: str) -> str:
    """
    Replace hyphens with underscores in object names only.
    - player-1 → player_1
    - loc-5-4 → loc_5_4
    Preserves predicate names like move-dir-up, oriented-right, etc.
    """
    # Replace player-N with player_N (where N is a digit)
    content = re.sub(r'\bplayer-(\d+)\b', r'player_\1', content)

    # Replace loc-X-Y with loc_X_Y (where X and Y are digits)
    content = re.sub(r'\bloc-(\d+)-(\d+)\b', r'loc_\1_\2', content)

    return content

def process_file(file_path: Path) -> int:
    """Process a single file and return number of replacements made."""
    # Read file
    with open(file_path, 'r') as f:
        original_content = f.read()

    # Fix object names
    fixed_content = fix_object_names(original_content)

    # Count changes
    changes = sum(1 for a, b in zip(original_content, fixed_content) if a != b)

    # Write back if changed
    if original_content != fixed_content:
        with open(file_path, 'w') as f:
            f.write(fixed_content)

    return changes

# Base directory
base_dir = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/data/maze/experiment_07-12-2025T16:16:54__model=gpt-5.1__steps=100__planner/training/trajectories")

# Find all files to process
pddl_files = list(base_dir.glob("**/*.pddl"))
trajectory_files = list(base_dir.glob("**/*.trajectory"))
json_files = list(base_dir.glob("**/*_trajectory.json"))

all_files = pddl_files + trajectory_files + json_files

print(f"Found {len(all_files)} files to process:")
print(f"  - {len(pddl_files)} .pddl files")
print(f"  - {len(trajectory_files)} .trajectory files")
print(f"  - {len(json_files)} _trajectory.json files")
print()

# Process all files
total_changes = 0
files_modified = 0

for file_path in sorted(all_files):
    changes = process_file(file_path)
    if changes > 0:
        files_modified += 1
        total_changes += changes
        print(f"✓ {file_path.relative_to(base_dir)}: {changes} characters changed")

print()
print(f"✅ Complete!")
print(f"   Files modified: {files_modified}/{len(all_files)}")
print(f"   Total characters changed: {total_changes}")

# Show sample of changes
if pddl_files:
    print(f"\n📋 Sample from {pddl_files[0].name}:")
    with open(pddl_files[0], 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:45], 1):
            if 'player_' in line or 'loc_' in line:
                print(f"   Line {i}: {line.rstrip()}")
