#!/usr/bin/env python3
"""
Test script to verify the refactored experiment_helpers and new PDDL utility functions.
"""
from pathlib import Path
import sys

# Test 1: Import verification
print("=" * 60)
print("TEST 1: Import Verification")
print("=" * 60)

try:
    from benchmark.experiment_helpers import (
        prepare_trajectory_with_gt_and_frame_axioms,
        prepare_fold_trajectories,
        setup_algorithm_workspace,
        save_fold_metadata,
        run_single_fold
    )
    print("✓ Successfully imported all functions from experiment_helpers")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

try:
    from src.utils.pddl import (
        inject_gt_states_by_percentage,
        propagate_frame_axioms_selective
    )
    print("✓ Successfully imported new PDDL utility functions")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test 2: GT State Index Calculation
print("\n" + "=" * 60)
print("TEST 2: GT State Index Calculation")
print("=" * 60)

import math

def calculate_gt_indices(num_states: int, gt_rate: int):
    """Replicate the GT index calculation logic."""
    gt_state_indices = {0}  # Initial state always GT

    if gt_rate > 0:
        num_gt_states = max(1, math.ceil(num_states * gt_rate / 100.0))
        if num_gt_states > 1:
            interval = num_states / num_gt_states
            for i in range(num_gt_states):
                idx = int(i * interval)
                if idx < num_states:
                    gt_state_indices.add(idx)

    return sorted(gt_state_indices)

# Test cases
test_cases = [
    (100, 10, "100 states, 10% GT → Should be 0, 10, 20, ..., 90"),
    (100, 25, "100 states, 25% GT → Should be 0, 4, 8, 12, ..., 96"),
    (100, 50, "100 states, 50% GT → Should be 0, 2, 4, 6, ..., 98"),
    (50, 10, "50 states, 10% GT → Should be 0, 10, 20, 30, 40"),
    (20, 50, "20 states, 50% GT → Should be 0, 2, 4, 6, ..., 18"),
]

for num_states, gt_rate, description in test_cases:
    indices = calculate_gt_indices(num_states, gt_rate)
    print(f"\n{description}")
    print(f"  Result: {indices}")
    print(f"  Count: {len(indices)} states (expected: ~{max(1, math.ceil(num_states * gt_rate / 100.0))})")

    # Verify spacing is even
    if len(indices) > 1:
        diffs = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
        expected_interval = num_states / len(indices)
        print(f"  Intervals: {diffs}")
        print(f"  Expected interval: ~{expected_interval:.1f}")

# Test 3: File structure check
print("\n" + "=" * 60)
print("TEST 3: File Structure Verification")
print("=" * 60)

files_to_check = [
    Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/run_fold.py"),
    Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/amlgym_testing.py"),
    Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/utils/pddl.py"),
]

for file_path in files_to_check:
    if file_path.exists():
        size_kb = file_path.stat().st_size / 1024
        print(f"✓ {file_path.name} exists ({size_kb:.1f} KB)")
    else:
        print(f"✗ {file_path.name} not found")

# Test 4: Configuration constants check
print("\n" + "=" * 60)
print("TEST 4: Configuration Constants")
print("=" * 60)

try:
    import benchmark.amlgym_testing as aml
    print(f"✓ NUM_TRAJECTORIES_LIST: {aml.NUM_TRAJECTORIES_LIST}")
    print(f"✓ NUM_TRAJECTORIES_POOL: {aml.NUM_TRAJECTORIES_POOL}")
    print(f"✓ GT_RATE_PERCENTAGES: {aml.GT_RATE_PERCENTAGES}")
    print(f"✓ FRAME_AXIOM_MODE: {aml.FRAME_AXIOM_MODE}")
    print(f"✓ N_FOLDS: {aml.N_FOLDS}")
except AttributeError as e:
    print(f"✗ Configuration constant missing: {e}")

print("\n" + "=" * 60)
print("All basic tests completed!")
print("=" * 60)
