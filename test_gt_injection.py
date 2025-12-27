#!/usr/bin/env python3
"""
Test GT injection with actual trajectory files.
"""
from pathlib import Path
from src.utils.pddl import inject_gt_states_by_percentage

# Use one of the existing maze trajectories (use the base trajectory, not truncated)
trajectory_path = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/data/maze/experiment_07-12-2025T16:16:54__model=gpt-5.1__steps=100__planner/training/trajectories/problem1/problem1_truncated_1.trajectory")
masking_path = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/data/maze/experiment_07-12-2025T16:16:54__model=gpt-5.1__steps=100__planner/training/trajectories/problem1/problem1_truncated_1.masking_info")
json_trajectory_path = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/data/maze/experiment_07-12-2025T16:16:54__model=gpt-5.1__steps=100__planner/training/trajectories/problem1/problem1_trajectory.json")
domain_path = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/data/maze/experiment_07-12-2025T16:16:54__model=gpt-5.1__steps=100__planner/training/domain.pddl")

print("=" * 60)
print("Testing GT Injection with Real Trajectory")
print("=" * 60)

# Check files exist
for file_path in [trajectory_path, masking_path, json_trajectory_path, domain_path]:
    if file_path.exists():
        print(f"✓ Found: {file_path.name}")
    else:
        print(f"✗ Missing: {file_path}")
        exit(1)

# Count states in original trajectory
with open(trajectory_path, 'r') as f:
    content = f.read()
    num_states = content.count("(:state")
    num_operators = content.count("(operator:")
    print(f"\nOriginal trajectory:")
    print(f"  States: {num_states}")
    print(f"  Operators: {num_operators}")

# Test different GT rates
print("\n" + "=" * 60)
print("Testing GT Injection Rates")
print("=" * 60)

for gt_rate in [0, 10, 25, 50]:
    print(f"\nGT Rate: {gt_rate}%")
    try:
        gt_traj_path, gt_masking_path, gt_indices = inject_gt_states_by_percentage(
            trajectory_path=trajectory_path,
            masking_info_path=masking_path,
            json_trajectory_path=json_trajectory_path,
            domain_path=domain_path,
            gt_rate=gt_rate
        )

        print(f"  ✓ GT injection successful")
        print(f"  GT state indices: {sorted(gt_indices)}")
        print(f"  Number of GT states: {len(gt_indices)}")

        # Verify output files exist
        if gt_traj_path.exists():
            print(f"  ✓ Output trajectory created: {gt_traj_path.name}")
        if gt_masking_path.exists():
            print(f"  ✓ Output masking created: {gt_masking_path.name}")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("GT Injection Test Complete")
print("=" * 60)
